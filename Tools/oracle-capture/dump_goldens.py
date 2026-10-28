#!/usr/bin/env python3
"""V1 oracle for the HeartMuLa-oss-3B MLX-Swift port (AB-T-0376, E27): goldens from upstream heartlib @ 5858ca8 on torch-MPS.

Per case: (1) the pipeline's own `preprocess` + `_forward` at bf16 under autocast (the V0 path) samples a reference rollout capped
at 400 frames — the codes are REFERENCE INPUTS, never a gate (AB-L-0090); (2) with those codes fixed, every lane's tensors are
dumped from upstream's own modules: the LM lanes twice (`bf16/` as shipped; `fp32/` = the bf16 parameters cast back to fp32, no
autocast — the gate lane), the codec lanes once (the codec is fp32 in both). The sampler's exponential noise and every
`torch.randn` of the codec are recorded so the Swift side can be fed the same draws (torch's generator is not reproducible in MLX).

What is not upstream's own call, and why: the teacher-forced lanes (E, P, L) assemble the inputs exactly as `generate_frame`
does (`_embed_tokens`, the token mask, `muq_linear`, `unconditional_text_embedding`, `_embed_audio`, `projection`, `audio_head`)
and run the backbone / decoder over many positions at once with the KV caches deleted and an explicit causal mask (torchtune
treats `mask=None` as NON-causal while a cache object exists). `meta.json` records the self-check that the teacher-forced
last-prompt c0 logits equal the ones `generate_frame` computed internally on the same prompt.

  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
  WIP/heartmula-eval/.venv/bin/python dump_goldens.py --ckpt WIP/heartmula-eval/_weights/ckpt --out WIP/heartmula-port/oracle/goldens
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

CASES = {"folk": ("p04_folk", 1004), "ballad": ("p12_zh_ballad", 1012)}
MAX_MS = 32_000          # 400 frames → two codec windows (372 at hop 320)
P_FRAMES = 64            # lane P: prompt + the first 64 reference frames
L_FRAMES = (0, 1, 31, 63)
TAIL = 16                # lane P keeps the post-norm hidden of the last TAIL positions
N_T = 0.6                # lane N: x = (1 − t)·noise + t·latents at t = 0.6 (a grid point of linspace(0, 1, 11))
CFG, TOPK, TEMP = 1.5, 50, 1.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(d: Path, name: str, t):
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(t, torch.Tensor):
        t = t.detach().to("cpu")
        t = t.to(torch.float32) if t.is_floating_point() else t.to(torch.int32)
        t = t.numpy()
    np.save(d / f"{name}.npy", np.ascontiguousarray(t))   # AB-L-0311: the Swift NPY reader is C-order only


def sync():
    torch.mps.synchronize()


def tags_for(style):
    return ",".join(t.strip().lower() for t in style.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()

    from heartlib import HeartMuLaGenPipeline
    from heartlib.heartmula import modeling_heartmula as mh
    from heartlib.heartcodec.models import transformer as hc_tf
    from torchtune.modules.common_utils import delete_kv_caches as _delete_kv_caches

    def delete_kv_caches(m):
        if m.caches_are_setup():
            _delete_kv_caches(m)

    # The two disclosed V0 deviations (MEASUREMENTS.md): torch.compile off; the MPS cast shim (numerics unchanged).
    HeartMuLaGenPipeline._apply_compile = lambda self, model: None

    def _timestep_embedding(self, timesteps, max_period=10000, scale=1000):
        half = self.flow_t_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=timesteps.device) / half).to(timesteps.dtype)
        args_ = timesteps[:, None] * freqs[None] * scale
        embedding = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        if self.flow_t_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    hc_tf.PixArtAlphaCombinedFlowEmbeddings.timestep_embedding = _timestep_embedding

    rows = {r["id"]: r for r in json.loads(a.prompts.read_text(encoding="utf-8"))["prompts"]}
    MPS = torch.device("mps")
    t0 = time.perf_counter()
    pipe = HeartMuLaGenPipeline.from_pretrained(str(a.ckpt), device={"mula": MPS, "codec": MPS},
                                                dtype={"mula": torch.bfloat16, "codec": torch.float32}, version="3B", lazy_load=False)
    sync()
    log(f"loaded in {time.perf_counter()-t0:.1f}s; mps current {torch.mps.current_allocated_memory()/1e9:.1f} GB")
    mula, codec = pipe.mula, pipe.codec

    # ---- T: tokenizer rows (all 12) ----------------------------------------------------------------------------------
    tok_dir = a.out / "tokenizer"
    for rid, r in rows.items():
        inp = pipe.preprocess({"tags": tags_for(r["style"]), "lyrics": r["lyrics"]}, cfg_scale=CFG)
        d = tok_dir / rid
        tags = tags_for(r["style"]).lower()
        tags = (tags if tags.startswith("<tag>") else f"<tag>{tags}")
        tags = (tags if tags.endswith("</tag>") else f"{tags}</tag>")
        save(d, "tokens", inp["tokens"][0]); save(d, "tokens_mask", inp["tokens_mask"][0].to(torch.int32))
        (d / "meta.json").write_text(json.dumps(dict(id=rid, tags=tags_for(r["style"]), lyrics=r["lyrics"], tags_string=tags,
                                                     muq_idx=int(inp["muq_idx"][0]), prompt_len=int(inp["tokens"].shape[1])),
                                                ensure_ascii=False, indent=1))
    log(f"T: {len(rows)} tokenizer rows")

    # ---- recorders ----------------------------------------------------------------------------------------------------
    rec = {"sample": [], "logits": []}
    _orig_mult, _orig_topk = mh._multinomial_sample_one_no_sync, mh.sample_topk

    def rec_mult(probs):
        q = torch.empty_like(probs).exponential_(1)
        out = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
        rec["sample"].append(dict(probs=probs.detach().clone(), q=q.detach().clone(), out=out.detach().clone()))
        return out

    def rec_topk(logits, topk, temperature):
        rec["logits"].append(logits.detach().clone())
        return _orig_topk(logits, topk, temperature)

    def causal(n, b):
        return torch.tril(torch.ones(n, n, dtype=torch.bool, device=MPS)).unsqueeze(0).expand(b, n, n)

    def prompt_embed(inp):
        """`generate_frame`'s input assembly for the prompt call (B = 2, CFG row second)."""
        tokens = inp["tokens"].to(MPS); mask = inp["tokens_mask"].to(MPS)
        b = tokens.shape[0]
        uncond = torch.cat([torch.zeros(b // 2, dtype=torch.bool, device=MPS), torch.ones(b // 2, dtype=torch.bool, device=MPS)])
        embeds = mula._embed_tokens(tokens, uncond_mask=uncond)
        h = (embeds * mask.unsqueeze(-1)).sum(dim=2, dtype=embeds.dtype)
        cs = mula.muq_linear(inp["muq_embed"].to(MPS, dtype=next(mula.parameters()).dtype))
        ue = mula.unconditional_text_embedding(torch.zeros(1, device=MPS, dtype=torch.long))
        cs = torch.where(uncond.view(b, 1).expand_as(cs), ue, cs)
        h[torch.arange(b, device=MPS), inp["muq_idx"]] = cs
        return h

    def frame_embed(codes_f):
        """Summed audio embeddings of frames (8, F) → (F, D) — the text column is masked out in every audio frame."""
        F = codes_f.shape[1]
        toks = codes_f.t().to(MPS) + mula.config.audio_vocab_size * torch.arange(8, device=MPS)
        return mula.audio_embeddings(toks.reshape(-1)).reshape(F, 8, -1).sum(dim=1)

    def lm_lanes(inp, codes, d: Path, autocast: bool):
        ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16) if autocast else torch.autocast(device_type="mps", enabled=False)
        meta = {}
        with torch.no_grad(), ctx:
            delete_kv_caches(mula.backbone); delete_kv_caches(mula.decoder)
            h = prompt_embed(inp)                                         # E
            save(d, "E_h", h)
            L = h.shape[1]
            fe = frame_embed(codes[:, :P_FRAMES]).to(h.dtype)
            x = torch.cat([h, fe.unsqueeze(0).expand(2, -1, -1)], dim=1)  # (2, L + 64, D)
            n = x.shape[1]
            pos = torch.arange(n, device=MPS).unsqueeze(0).expand(2, n)
            hid = mula.backbone(x, input_pos=pos, mask=causal(n, 2))     # P
            logits = mula.codebook0_head(hid[:, L - 1:, :])               # (2, 65, 8197)
            save(d, "P_hidden_tail", hid[:, -TAIL:, :]); save(d, "P_logits", logits)
            save(d, "P_input", x)
            for j in L_FRAMES:                                            # L
                last_h = hid[:, L - 1 + j, :]
                c = codes[:, j].to(MPS)
                emb = torch.stack([mula._embed_audio(i, c[i:i + 1].expand(2)) for i in range(7)], dim=1)  # (2, 7, D)
                cur = torch.cat([last_h.unsqueeze(1), emb.to(last_h.dtype)], dim=1)                     # (2, 8, D)
                pos8 = torch.arange(8, device=MPS).unsqueeze(0).expand(2, 8)
                dh = mula.decoder(mula.projection(cur), input_pos=pos8, mask=causal(8, 2))
                li = torch.stack([torch.mm(dh[:, i, :], mula.audio_head[i - 1]) for i in range(1, 8)], dim=1)  # (2, 7, 8197)
                save(d, f"L_f{j}_input", cur); save(d, f"L_f{j}_logits", li)
            # F: generate_frame itself on the prompt, caches set up as _forward does, sampler noise recorded.
            mula.setup_caches(2)
            rec["sample"].clear(); rec["logits"].clear()
            c0_internal = {}
            hk = mula.codebook0_head.register_forward_hook(lambda m, i, o: c0_internal.setdefault("c0", o.detach().clone()))
            mh._multinomial_sample_one_no_sync, mh.sample_topk = rec_mult, rec_topk
            try:
                frame = mula.generate_frame(tokens=inp["tokens"].to(MPS), tokens_mask=inp["tokens_mask"].to(MPS), input_pos=inp["pos"].to(MPS),
                                            temperature=TEMP, topk=TOPK, cfg_scale=CFG,
                                            continuous_segments=inp["muq_embed"].to(MPS, dtype=next(mula.parameters()).dtype),
                                            starts=inp["muq_idx"])
            finally:
                mh._multinomial_sample_one_no_sync, mh.sample_topk = _orig_mult, _orig_topk
                hk.remove()
            assert len(rec["logits"]) == 8 and len(rec["sample"]) == 8
            save(d, "F_guided_logits", torch.cat(rec["logits"], 0))                     # (8, 8197)
            save(d, "F_q", torch.cat([s["q"] for s in rec["sample"]], 0))               # (8, 8197)
            save(d, "F_probs", torch.cat([s["probs"] for s in rec["sample"]], 0))
            save(d, "F_codes", frame[0])                                                 # (8,)
            save(d, "F_c0_logits", c0_internal["c0"])                                    # (2, 8197) cond / uncond
            dd = (c0_internal["c0"].float() - logits[:, 0, :].float())
            meta.update(self_check_c0_relL2=float(dd.norm() / logits[:, 0, :].float().norm()), F_codes=frame[0].tolist(),
                        prompt_len=L, n_positions=n, F_logits_dtype=str(rec["logits"][0].dtype), F_q_dtype=str(rec["sample"][0]["q"].dtype))
            delete_kv_caches(mula.backbone); delete_kv_caches(mula.decoder)
        sync()
        return meta

    def codec_lanes(codes, d: Path):
        fm = codec.decoder.flow_matching
        sq = codec.decoder.scalar_model
        rnd, wins = [], []
        orig_randn, orig_inf, orig_solve = torch.randn, fm.inference_codes, fm.solve_euler

        def randn_rec(*args, **kw):
            t = orig_randn(*args, **kw); rnd.append(t.detach().clone()); return t

        def inf_rec(codes_l, true_latents, latent_length, incontext_length, **kw):
            wins.append(dict(codes=codes_l[0].detach().clone(), true_latents=true_latents.detach().clone(),
                             latent_length=int(latent_length), incontext_length=int(incontext_length), kw=dict(kw)))
            out = orig_inf(codes_l, true_latents, latent_length, incontext_length, **kw)
            wins[-1]["out"] = out.detach().clone()
            return out

        def solve_rec(x, incontext_x, incontext_length, t_span, mu, guidance_scale):
            wins[-1].update(noise=x.detach().clone(), incontext_x=incontext_x.detach().clone(), mu=mu.detach().clone(),
                            ictx_len_solver=int(incontext_length), t_span=t_span.detach().clone(), guidance=float(guidance_scale))
            return orig_solve(x, incontext_x, incontext_length, t_span, mu, guidance_scale)

        torch.randn, fm.inference_codes, fm.solve_euler = randn_rec, inf_rec, solve_rec
        try:
            t = time.perf_counter()
            wav = codec.detokenize(codes.to(MPS))
            sync(); det_s = time.perf_counter() - t
        finally:
            torch.randn, fm.inference_codes, fm.solve_euler = orig_randn, orig_inf, orig_solve
        save(d, "D_codes", codes); save(d, "D_wave", wav)
        for i, w in enumerate(wins):
            p = f"W_w{i}_"
            save(d, p + "codes", w["codes"][0]); save(d, p + "noise", w["noise"]); save(d, p + "incontext_x", w["incontext_x"])
            save(d, p + "mu", w["mu"]); save(d, p + "latents", w["out"])
        save(d, "S_t_span", wins[0]["t_span"])
        with torch.no_grad():
            for i, w in enumerate(wins):                                                 # N
                x = (1 - N_T) * w["noise"] + N_T * w["out"]
                inp = torch.cat([torch.cat([x, x], 0), torch.cat([w["incontext_x"], w["incontext_x"]], 0),
                                 torch.cat([torch.zeros_like(w["mu"]), w["mu"]], 0)], 2)
                tt = torch.tensor([N_T, N_T], device=MPS, dtype=x.dtype)
                v = fm.estimator(inp, timestep=tt)
                save(d, f"N_w{i}_x", x); save(d, f"N_w{i}_velocity", v)
            save(d, "S_timestep_emb", fm.estimator.adaln_single.emb.timestep_embedding(torch.tensor([N_T, N_T], device=MPS)))
            lat = wins[0]["out"]                                                          # Q
            lat = lat.reshape(lat.shape[0], lat.shape[1], 2, lat.shape[2] // 2).permute(0, 2, 1, 3).reshape(2, lat.shape[1], -1)
            q_in = lat.transpose(1, 2).contiguous()                                       # (2, 128, 744)
            q_out = sq.decode(q_in).squeeze(1)                                            # (2, 1 428 480)
            save(d, "Q_input", q_in); save(d, "Q_wave", q_out)
        sync()
        return dict(windows=len(wins), randn_calls=[list(r.shape) for r in rnd], detokenize_s=det_s,
                    window_meta=[dict(latent_length=w["latent_length"], incontext_length=w["incontext_length"],
                                      ictx_len_solver=w["ictx_len_solver"], guidance=w["guidance"], codes_shape=list(w["codes"].shape))
                                 for w in wins],
                    wave_shape=list(wav.shape), wave_peak=float(wav.abs().max()))

    cases = [c for c in a.cases.split(",") if c]
    metas = {}
    for c in cases:                                                       # bf16 phase: rollout + shipped-tier LM lanes + codec
        rid, seed = CASES[c]
        r = rows[rid]
        d = a.out / c
        inp = pipe.preprocess({"tags": tags_for(r["style"]), "lyrics": r["lyrics"]}, cfg_scale=CFG)
        torch.manual_seed(seed)
        t = time.perf_counter()
        with torch.no_grad():
            out = pipe._forward(inp, max_audio_length_ms=MAX_MS, temperature=TEMP, topk=TOPK, cfg_scale=CFG)
        sync()
        codes = out["frames"].to("cpu")                                   # (8, F)
        roll_s = time.perf_counter() - t
        save(d, "codes", codes)
        log(f"{c}: rollout {codes.shape[1]} frames in {roll_s:.1f}s ({codes.shape[1]/roll_s:.1f} fr/s)")
        m = dict(case=c, id=rid, seed=seed, tags=tags_for(r["style"]), frames=int(codes.shape[1]), max_frames=MAX_MS // 80 + 1,
                 ended_on_end_token=int(codes.shape[1]) < MAX_MS // 80 + 1, rollout_s=roll_s, cfg=CFG, topk=TOPK, temperature=TEMP,
                 p_frames=P_FRAMES, l_frames=list(L_FRAMES), n_t=N_T)
        torch.manual_seed(seed)
        m["bf16"] = lm_lanes(inp, codes, d / "bf16", autocast=True)
        log(f"{c}: bf16 LM lanes; self-check {m['bf16']['self_check_c0_relL2']:.2e}")
        m["codec"] = codec_lanes(codes, d / "codec")
        log(f"{c}: codec lanes {m['codec']['windows']} windows, wave {m['codec']['wave_shape']} in {m['codec']['detokenize_s']:.1f}s")
        metas[c] = (m, inp, codes)
        save(d, "S_rope_theta_backbone", mula.backbone.layers[0].attn.pos_embeddings.theta)
        save(d, "S_rope_theta_decoder", mula.decoder.layers[0].attn.pos_embeddings.theta)

    rope_dtypes_bf16 = {k: (str(r.theta.dtype), str(r.cache.dtype)) for k, r in
                        (('backbone', mula.backbone.layers[0].attn.pos_embeddings), ('decoder', mula.decoder.layers[0].attn.pos_embeddings))}
    mula.to(torch.float32)                                                # fp32 phase: bf16 values, fp32 arithmetic, no autocast
    for tdec in (mula.backbone, mula.decoder):                            # the RoPE tables are rebuilt in fp32, as a fresh fp32 model has them
        seen = set()
        for layer in tdec.layers:
            rope = layer.attn.pos_embeddings
            if id(rope) not in seen:
                seen.add(id(rope)); rope.rope_init(); rope.to(MPS)
    sync()
    for c in cases:
        save(a.out / c, "S_rope_theta_backbone_fp32", mula.backbone.layers[0].attn.pos_embeddings.theta)
        save(a.out / c, "S_rope_theta_decoder_fp32", mula.decoder.layers[0].attn.pos_embeddings.theta)
    log(f"LM cast bf16 → fp32; mps current {torch.mps.current_allocated_memory()/1e9:.1f} GB")
    for c in cases:
        m, inp, codes = metas[c]
        torch.manual_seed(CASES[c][1])
        m["fp32"] = lm_lanes(inp, codes, a.out / c / "fp32", autocast=False)
        log(f"{c}: fp32 LM lanes; self-check {m['fp32']['self_check_c0_relL2']:.2e}; F codes {m['fp32']['F_codes']}")
        m.update(rope_dtypes_bf16_phase=rope_dtypes_bf16, torch=torch.__version__, heartlib="5858ca8", hwm=os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
                 t=time.strftime("%Y-%m-%d %H:%M:%S"))
        (a.out / c / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
