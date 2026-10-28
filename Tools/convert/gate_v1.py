#!/usr/bin/env python3
"""V1 converter gate, clauses (a) and (d) (MEASUREMENTS.md): every bundle tensor equals the torch module's own tensor after the
stated transform — LM `param.to(bf16)`; codec the `.weight` torch evaluates through `parametrize` (weight_norm), transposed to
MLX's layout, at fp32 and `.to(fp16)` — 0 missing, 0 extra; the files a bundle must carry are present. Also writes the
torch-side references for clause (b) (`heartmula-gates convcheck`): one SQ-Codec look-ahead conv, one causal up-conv and one
dilated residual conv on a fixed CPU input.

  gate_v1.py --ckpt WIP/heartmula-eval/_weights/ckpt --weights WIP/heartmula-port/weights --refs WIP/heartmula-port/oracle/convcheck
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import RENAMES  # noqa: E402


def bundle(d: Path) -> dict[str, torch.Tensor]:
    idx = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    out = {}
    for f in files:
        with safe_open(str(d / f), "pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def unrename(key: str) -> str:
    """Bundle key → upstream module path (the inverse of convert.RENAMES)."""
    m = re.match(r"^scalar_model\.blocks\.(\d)\.up_conv\.(.*)$", key)
    if m:
        return f"scalar_model.decoder.{int(m.group(1)) + 1}.up_conv.layer.{m.group(2)}"
    m = re.match(r"^scalar_model\.blocks\.(\d)\.convs\.(.*)$", key)
    if m:
        return f"scalar_model.decoder.{int(m.group(1)) + 1}.convs.{m.group(2)}"
    for new, old in (("scalar_model.pre_conv.", "scalar_model.decoder.0."), ("scalar_model.post.", "scalar_model.decoder.6."),
                     ("scalar_model.out_conv.", "scalar_model.decoder.7.")):
        if key.startswith(new):
            return old + key[len(new):]
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    a = ap.parse_args()
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    report, ok = {}, True

    # ---- LM ----
    lm = HeartMuLa.from_pretrained(str(a.ckpt / "HeartMuLa-oss-3B"), dtype=torch.float32).eval()
    sd = lm.state_dict()
    b = bundle(a.weights / "HeartMuLa-oss-3B-bf16")
    missing = sorted(set(sd) - set(b)); extra = sorted(set(b) - set(sd))
    bad = [k for k in sorted(set(sd) & set(b)) if b[k].dtype != torch.bfloat16 or b[k].shape != sd[k].shape
           or not torch.equal(b[k], sd[k].to(torch.bfloat16))]
    report["lm_bf16"] = dict(tensors=len(b), module_tensors=len(sd), missing=missing, extra=extra, mismatched=bad)
    ok &= not (missing or extra or bad)
    print("lm_bf16", len(b), "missing", len(missing), "extra", len(extra), "mismatched", len(bad), flush=True)
    del lm, sd, b

    # ---- codec ----
    codec = HeartCodec.from_pretrained(str(a.ckpt / "HeartCodec-oss"), dtype=torch.float32).eval()
    mods = dict(codec.decoder.named_modules())
    params = dict(codec.decoder.named_parameters())
    buffers = dict(codec.decoder.named_buffers())

    def expected(key: str) -> torch.Tensor:
        m = re.match(r"^flow_matching\.vq_embed\.codebooks\.(\d+)\.weight$", key)
        if m:
            return buffers[f"flow_matching.vq_embed.layers.{m.group(1)}._codebook.embed"][0]
        up = unrename(key)
        mod_path, leaf = up.rsplit(".", 1)
        mod = mods[mod_path]
        if leaf == "weight" and hasattr(mod, "parametrizations"):
            w = mod.weight.detach()                                    # the parametrized weight torch's forward uses
            return w.permute(1, 2, 0) if isinstance(mod, torch.nn.ConvTranspose1d) else w.permute(0, 2, 1)
        t = params.get(up, buffers.get(up))
        assert t is not None, up
        t = t.detach()
        if isinstance(mod, torch.nn.Conv1d) and leaf == "weight":
            t = t.permute(0, 2, 1)
        return t

    # The reference key set is the checkpoint's (the module also holds vector-quantize-pytorch runtime buffers —
    # `beam_score_weights`, `layers.N.zero` — that are not in the checkpoint and that `get_output_from_indices` never reads).
    ckpt_keys = set(json.loads((a.ckpt / "HeartCodec-oss" / "model.safetensors.index.json").read_text())["weight_map"])
    decode_keys = set()
    for k in sorted(ckpt_keys):
        if k.startswith("scalar_model.encoder.") or "vq_embed.project_in." in k or re.search(r"_codebook\.(cluster_size|embed_avg|initted)$", k):
            continue
        if ".parametrizations.weight.original" in k:
            decode_keys.add(k.split(".parametrizations.")[0] + ".weight")
        else:
            decode_keys.add(k)
    for tier, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        b = bundle(a.weights / f"HeartCodec-oss-{tier}")
        mapped = {unrename(k) if not k.startswith("flow_matching.vq_embed.codebooks.") else
                  f"flow_matching.vq_embed.layers.{k.split('.')[3]}._codebook.embed": k for k in b}
        missing = sorted(decode_keys - set(mapped)); extra = sorted(set(mapped) - decode_keys)
        bad = []
        for k, t in b.items():
            e = expected(k).to(dt).contiguous()
            if t.dtype != dt or t.shape != e.shape or not torch.equal(t, e):
                bad.append(k)
        report[f"codec_{tier}"] = dict(tensors=len(b), decode_path_tensors=len(decode_keys), missing=missing, extra=extra, mismatched=bad)
        ok &= not (missing or extra or bad)
        print(f"codec_{tier}", len(b), "decode-path", len(decode_keys), "missing", len(missing), "extra", len(extra), "mismatched", len(bad), flush=True)

    # ---- (b) references for the MLX conv-layout check (CPU fp32) ----
    a.refs.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(27)
    sqm = codec.decoder.scalar_model
    with torch.no_grad():
        x0 = torch.randn(1, 128, 40, generator=g)
        y0 = sqm.decoder[0](x0)                                  # look-ahead conv, symmetric padding
        x1 = torch.randn(1, 2048, 20, generator=g)
        y1 = sqm.decoder[1].up_conv(x1)                          # causal ConvTranspose1d k10 s5 (no activation)
        x2 = torch.randn(1, 1024, 60, generator=g)
        y2 = sqm.decoder[1].convs[3].conv1(x2)                   # causal Conv1d k7 dilation 7
    for n, t in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        np.save(a.refs / f"{n}.npy", np.ascontiguousarray(t.numpy()))

    # ---- (d) files ----
    need = {"HeartMuLa-oss-3B-bf16": ["config.json", "tokenizer.json", "gen_config.json", "LICENSE", "NOTICE"],
            "HeartCodec-oss-fp32": ["config.json", "LICENSE", "NOTICE"], "HeartCodec-oss-fp16": ["config.json", "LICENSE", "NOTICE"]}
    files_missing = {d: [f for f in fs if not (a.weights / d / f).exists()] for d, fs in need.items()}
    report["files_missing"] = files_missing
    ok &= not any(files_missing.values())
    report["PASS_a_d"] = bool(ok)
    (a.refs.parent / "gate_v1_report.json").write_text(json.dumps(report, indent=1))
    print("V1 (a)+(d):", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
