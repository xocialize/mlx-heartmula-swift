#!/usr/bin/env python3
"""V1 converter for the HeartMuLa-oss-3B MLX-Swift port (AB-T-0376, E27): upstream checkpoints → MLX-layout bundles.

  convert.py --ckpt WIP/heartmula-eval/_weights/ckpt --heartlib WIP/heartmula-eval/ref/heartlib --out WIP/heartmula-port/weights

Bundles (each a directory with model*.safetensors [+ index], config.json, LICENSE, NOTICE):
  HeartMuLa-oss-3B-bf16   the LM (backbone + local decoder + embeddings + heads), every tensor `.to(bfloat16)`, keys unchanged;
                          plus tokenizer.json and gen_config.json (HeartMuLaGen).
  HeartCodec-oss-fp32     the decoding half of HeartCodec at fp32 (README: bf16 degrades the codec — never bf16).
  HeartCodec-oss-fp16     the same tensors `.to(float16)` after the fp32 fold.

Codec transforms, and only these (MEASUREMENTS.md, V1 gate):
  • dropped — never loaded by `detokenize`: `scalar_model.encoder.*`, the RVQ's EMA buffers (`cluster_size`, `embed_avg`,
    `initted`) and `vq_embed.project_in.*`;
  • weight_norm folded in fp32 with torch's own kernel: `weight = torch._weight_norm(original1 = v, original0 = g, dim=0)`;
  • Conv1d (O, I, K) → (O, K, I); ConvTranspose1d (I, O, K) → (O, K, I) — MLX's layouts;
  • RVQ codebooks `vq_embed.layers.N._codebook.embed` (1, 8192, 32) → `vq_embed.codebooks.N.weight` (8192, 32);
  • the SQ-Codec decoder's positional list renamed to named modules (RENAMES below); every other key unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM_REPO = ("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "41f6fc6")
CODEC_REPO = ("HeartMuLa/HeartCodec-oss-20260123", "f889dab")
GEN_REPO = ("HeartMuLa/HeartMuLaGen", "2e18e01")
HEARTLIB = ("HeartMuLa/heartlib", "5858ca8")
SHARD_BYTES = 4_900_000_000

# SQ-Codec decoder (`scalar_model.decoder.N`, ScalarModel.__init__ order) → named modules.
#   0  weight_norm(Conv1d(128, 2048, k5))            look-ahead conv, NOT causal (symmetric padding 2)
#   1…5 ResDecoderBlock(up_conv = weight_norm(ConvTranspose1d, causal), convs = 5 × ResidualUnit)
#   6  PostProcessor(conv = Conv1d(64, 64, k7, causal), activation = PReLU)   — no weight_norm
#   7  weight_norm(Conv1d(64, 1, k7, causal))
RENAMES = [
    (r"^scalar_model\.decoder\.0\.", "scalar_model.pre_conv."),
    (r"^scalar_model\.decoder\.([1-5])\.up_conv\.layer\.", lambda m: f"scalar_model.blocks.{int(m.group(1)) - 1}.up_conv."),
    (r"^scalar_model\.decoder\.([1-5])\.convs\.", lambda m: f"scalar_model.blocks.{int(m.group(1)) - 1}.convs."),
    (r"^scalar_model\.decoder\.6\.", "scalar_model.post."),
    (r"^scalar_model\.decoder\.7\.", "scalar_model.out_conv."),
]
CONV_TRANSPOSE = re.compile(r"^scalar_model\.decoder\.[1-5]\.up_conv\.layer\.")


def rename(key: str) -> str:
    for pat, rep in RENAMES:
        if re.match(pat, key):
            return re.sub(pat, rep, key)
    return key


def read_all(d: Path) -> dict[str, torch.Tensor]:
    idx = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    out = {}
    for f in files:
        with safe_open(str(d / f), "pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def is_conv_weight(key: str, t: torch.Tensor) -> bool:
    return t.ndim == 3 and key.endswith("weight") and not key.endswith("embed")


def convert_codec(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    folded: set[str] = set()
    for k in sorted(src):
        if k.startswith("scalar_model.encoder.") or "vq_embed.project_in." in k:
            continue
        if re.search(r"_codebook\.(cluster_size|embed_avg|initted)$", k):
            continue
        m = re.match(r"^(.*)\.parametrizations\.weight\.original([01])$", k)
        if m:
            base = m.group(1)
            if base in folded:
                continue
            folded.add(base)
            g = src[f"{base}.parametrizations.weight.original0"].to(torch.float32)
            v = src[f"{base}.parametrizations.weight.original1"].to(torch.float32)
            w = torch._weight_norm(v, g, 0)
            name = base + ".weight"
            if CONV_TRANSPOSE.match(base + "."):
                w = w.permute(1, 2, 0)                  # (I, O, K) → (O, K, I)
            else:
                w = w.permute(0, 2, 1)                  # (O, I, K) → (O, K, I)
            out[rename(name)] = w.contiguous()
            continue
        mcb = re.match(r"^flow_matching\.vq_embed\.layers\.(\d+)\._codebook\.embed$", k)
        if mcb:
            t = src[k]
            assert t.shape[0] == 1, t.shape
            out[f"flow_matching.vq_embed.codebooks.{mcb.group(1)}.weight"] = t[0].contiguous()
            continue
        t = src[k]
        if is_conv_weight(k, t):                        # plain (non-weight-norm) Conv1d: ProjectLayer.ffn_1, PostProcessor.conv
            t = t.permute(0, 2, 1).contiguous()
        out[rename(k)] = t
    return out


def write_bundle(tensors: dict[str, torch.Tensor], out: Path, shard_bytes=SHARD_BYTES):
    out.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        t = tensors[k]
        b = t.numel() * t.element_size()
        if cur and size + b > shard_bytes:
            shards.append(cur); cur, size = {}, 0
        cur[k] = t; size += b
    if cur:
        shards.append(cur)
    if len(shards) == 1:
        save_file(shards[0], str(out / "model.safetensors"), metadata={"format": "mlx"})
        return ["model.safetensors"]
    names, wmap, total = [], {}, 0
    for i, s in enumerate(shards):
        n = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, str(out / n), metadata={"format": "mlx"})
        names.append(n)
        for k, t in s.items():
            wmap[k] = n; total += t.numel() * t.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": wmap}, indent=1))
    return names


NOTICE = """HeartMuLa-oss-3B / HeartCodec-oss — MLX conversion (mlx-heartmula-swift, AB-T-0376)

Licence: Apache License 2.0 (text in LICENSE, from {heartlib} @ {heartlib_rev}).

Grant chain (read 2026-10-28, AB-L-0307):
  • {heartlib} LICENSE is the verbatim Apache-2.0 text (added 2026-01-19); its README News: "20 Jan. 2026 — License
    update: We update the license of this repo and all related model weights to Apache 2.0."
  • {lm} @ {lm_rev}, {codec} @ {codec_rev} and {gen} @ {gen_rev} carry `license: apache-2.0` on their model cards and
    NO LICENSE file in the repository; the grant is the card statement plus the heartlib README above.
  • tokenizer.json is the Llama-3.2 BPE vocabulary (128 256 ids). HeartMuLa's weights are MuLa Labs' own (the backbone uses
    the Llama-3.2 architecture through torchtune builders; no Meta weights are loaded). The worst reading of the vocabulary
    file is the Llama 3.2 Community License, which permits commercial use.
  • Not on this path: MuQ-MuLan (CC-BY-NC-4.0; the pipeline passes muq_mulan=None and feeds a zero vector) and MuLaCover.

Conversion: {transforms}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--heartlib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="lm,codec32,codec16")
    a = ap.parse_args()
    only = set(a.only.split(","))
    lic = (a.heartlib / "LICENSE").read_text()
    common = dict(heartlib=HEARTLIB[0], heartlib_rev=HEARTLIB[1], lm=LM_REPO[0], lm_rev=LM_REPO[1], codec=CODEC_REPO[0],
                  codec_rev=CODEC_REPO[1], gen=GEN_REPO[0], gen_rev=GEN_REPO[1])
    receipt = {}

    if "lm" in only:
        t = time.perf_counter()
        src = read_all(a.ckpt / "HeartMuLa-oss-3B")
        tens = {k: v.to(torch.bfloat16).contiguous() for k, v in src.items()}
        d = a.out / "HeartMuLa-oss-3B-bf16"
        files = write_bundle(tens, d)
        cfg = json.loads((a.ckpt / "HeartMuLa-oss-3B" / "config.json").read_text())
        cfg["mlx_heartmula"] = dict(component="lm", tier="bf16", source=dict(repo=LM_REPO[0], revision=LM_REPO[1]),
                                    tokenizer_source=dict(repo=GEN_REPO[0], revision=GEN_REPO[1]), oracle=f"{HEARTLIB[0]}@{HEARTLIB[1]}",
                                    transforms="every tensor .to(bfloat16); keys unchanged", tensors=len(tens))
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        for f in ("tokenizer.json", "gen_config.json"):
            shutil.copy2(a.ckpt / f, d / f)
        (d / "LICENSE").write_text(lic)
        (d / "NOTICE").write_text(NOTICE.format(transforms="LM every tensor cast to bfloat16 (round-to-nearest-even), keys unchanged.", **common))
        receipt["lm"] = dict(tensors=len(tens), files=files, seconds=time.perf_counter() - t,
                             bytes=sum(v.numel() * v.element_size() for v in tens.values()))
        print("lm", receipt["lm"], flush=True)
        del src, tens

    if only & {"codec32", "codec16"}:
        t = time.perf_counter()
        src = read_all(a.ckpt / "HeartCodec-oss")
        tens32 = convert_codec(src)
        dropped = sorted(k for k in src if k.startswith("scalar_model.encoder.") or "vq_embed.project_in." in k
                         or re.search(r"_codebook\.(cluster_size|embed_avg|initted)$", k))
        cfg = json.loads((a.ckpt / "HeartCodec-oss" / "config.json").read_text())
        for tier, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
            if f"codec{tier[2:]}" not in only:
                continue
            tens = {k: v.to(dt).contiguous() if v.is_floating_point() else v for k, v in tens32.items()}
            d = a.out / f"HeartCodec-oss-{tier}"
            files = write_bundle(tens, d)
            c = dict(cfg)
            c["mlx_heartmula"] = dict(component="codec", tier=tier, source=dict(repo=CODEC_REPO[0], revision=CODEC_REPO[1]),
                                      oracle=f"{HEARTLIB[0]}@{HEARTLIB[1]}", renames=[[p, r if isinstance(r, str) else "blocks.{N-1}…"] for p, r in RENAMES],
                                      transforms=("decode path only (SQ-Codec encoder, RVQ EMA buffers, vq_embed.project_in dropped); weight_norm "
                                                  "folded in fp32 with torch._weight_norm(v, g, 0); Conv1d (O,I,K)->(O,K,I); ConvTranspose1d "
                                                  "(I,O,K)->(O,K,I); RVQ codebooks (1,8192,32)->(8192,32); SQ decoder list renamed"
                                                  + ("; then .to(float16)" if tier == "fp16" else "")),
                                      tensors=len(tens))
            (d / "config.json").write_text(json.dumps(c, indent=1))
            (d / "LICENSE").write_text(lic)
            (d / "NOTICE").write_text(NOTICE.format(transforms=c["mlx_heartmula"]["transforms"], **common))
            receipt[f"codec_{tier}"] = dict(tensors=len(tens), files=files, bytes=sum(v.numel() * v.element_size() for v in tens.values()),
                                            seconds=time.perf_counter() - t)
            print(f"codec_{tier}", receipt[f"codec_{tier}"], flush=True)
        receipt["codec_dropped_count"] = len(dropped)
        receipt["codec_source_tensors"] = len(src)
    (a.out / "convert_receipt.json").write_text(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
