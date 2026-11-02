# E27 HeartMuLa-oss-3B → MLX-Swift port (AB-T-0376) — V1 / V2 measurements

Hardware: Mac Studio Mac17,14, Apple M5 Max, 137 GB; macOS 27.2. Upstream `HeartMuLa/heartlib` @ `5858ca8`
(`WIP/heartmula-eval/ref/heartlib`), weights `HeartMuLa-oss-3B-happy-new-year@41f6fc6` + `HeartCodec-oss-20260123@f889dab` +
`HeartMuLaGen@2e18e01` (`WIP/heartmula-eval/_weights/ckpt/`), venv `WIP/heartmula-eval/.venv` (torch 2.10.0, torchtune 0.4.0,
vector-quantize-pytorch 1.27.15, tokenizers 0.22.1). Package: `PROD/HeartMuLa/mlx-heartmula-swift` (`HeartMuLaCore`,
`heartmula-gates`, `Tools/`; its own repo). Scratch: this folder (`oracle/goldens`, `weights`, `receipts` — gitignored).
Full port from upstream; `Acelogic/heartlib-mlx` has **no LICENSE file** and was not read for code (structure at most; in
practice not opened). V0: AB-R-0628; listening pass GO: AB-D-0187.

**The oracle is upstream heartlib on torch-MPS** — the reference implementation itself, not a port (unlike E26). The two
disclosed V0 deviations carry over unchanged: `torch.compile` off, and the MPS shim in
`PixArtAlphaCombinedFlowEmbeddings.timestep_embedding` (`.type(t.type())` → `.to(t.dtype)`, the same cast). Every dump runs
with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4` / `LOW=0.3` and goes through `np.ascontiguousarray` (AB-L-0311).

## Gates (stated 2026-10-28 11:05 PDT, before any golden, converter output or Swift result existed — AB-L-0090)

**Oracle cases.** Two V0 rows (`WIP/acestep-eval/cues/prompts.json`), tags by V0's mechanical transform
(`",".join(t.strip().lower() for t in style.split(","))`), lyrics verbatim, CFG 1.5, top-k 50, temperature 1.0:
`folk` = p04 (en, seed 1004 — reached the cap in V0, so it will not stop early) and `ballad` = p12 (zh, seed 1012 — exercises
the CJK byte-level BPE). For each case the **reference rollout** is the pipeline's own `_forward` at **bf16 with autocast**
(the V0 path) capped at `max_audio_length_ms = 32 000` (400 frames, enough for two codec windows); its codes are a
**reference input only, never a gate** (AB-L-0090). If a case ends on its end token before 400 frames, the golden says so and
the codec lanes use what was emitted.

**Two dtype lanes, dumped from the same inputs:**

- `fp32/` — **the gate lane.** The LM's parameters are the converter's bf16 values cast back to fp32 (`model.to(bf16).to(fp32)`),
  run in fp32 with no autocast; the codec is the checkpoint's fp32. The Swift side loads the **shipped bf16 LM bundle cast to
  fp32** and the fp32 codec bundle and runs fp32 on the GPU with `MLX_ENABLE_TF32=0`. Both sides therefore hold identical
  weights; what is compared is the arithmetic. (Disclosed choice: an fp32-checkpoint LM lane would need a 15.7 GB scratch
  bundle and would gate weights nobody ships.)
- `bf16/` — the shipped tier as V0 ran it (bf16 LM under `torch.autocast(bf16)`, fp32 codec). Reported at V2 beside the fp32
  lane, **no pass rule at V2**; the V2b tier rules are pre-registered at V2b, before any tier result, on paired seeds with ≥ 2
  draws or pooled (AB-L-0312).

**Goldens per case** (`oracle/goldens/<case>/{fp32,bf16}/`, plus `meta.json` and the tokenizer rows):

1. **T** — `tags_ids`, `lyrics_ids`, the `(L, 9)` token matrix and mask that `preprocess` builds, for both cases **and for all
   12 V0 rows** (tokenizer only, no model).
2. **E** — the summed prompt embedding `h` (2, L, 3072) as `generate_frame` builds it: masked text embedding (the
   unconditional embedding on the CFG row), `muq_linear(zeros)` written at `muq_idx` (the unconditional embedding on the CFG
   row).
3. **P backbone, teacher-forced** — prompt + the first 64 reference frames (each frame's 8 codes embedded and summed, text
   column masked) through the backbone without a KV cache (torchtune's causal default): post-norm hidden of the last 16
   positions and `codebook0_head` logits at every position from the last prompt token on (65 positions × 2 CFG rows × 8197).
4. **L local decoder, teacher-forced** — for frames 0, 1, 31, 63: `[last_h ; embed_0(c0) … embed_6(c6)]` → `projection` →
   the 3-layer decoder over 8 positions (causal, no cache) → `audio_head[i]` logits for codebooks 1…7, both CFG rows.
5. **F one CFG frame** — `generate_frame` itself on the prompt (KV caches set up, batch 2, CFG 1.5, top-k 50, temperature 1.0)
   with `_multinomial_sample_one_no_sync` recorded: for codebook 0 and each of the 7 local steps the guided logits, the
   exponential noise `q`, and the sampled code. This is the "fixed RNG state": torch's generator is not reproducible in MLX,
   so the Swift sampler is fed the recorded `q` and must take the same `argmax(probs / q)`.
6. **N DiT velocity** — the estimator on window 0 and window 1 at t = 0.6 (a grid point of the 10-step schedule) at the state
   `x = 0.4·noise + 0.6·window_latents`, with the window's own in-context latents and condition, CFG pair (unconditional =
   zero condition) — both halves of the batch-2 output.
7. **W window solves** — `HeartCodec.detokenize` on the reference codes (the first 400 frames → 2 windows of 372 frames at hop
   320) with every `torch.randn` recorded in call order: per window the codes slice, the condition (`vq_embed` →
   `cond_feature_emb` → nearest ×2), the in-context latents and length, the noise, and the 10-step result.
8. **Q SQ-Codec** — `scalar_model.decode` on the fixed window-0 latents (744 × 256 → 2 × 128 → 2 × 1 428 480 samples).
9. **D end to end** — the stereo waveform `detokenize` returns for those codes (2 × 1 536 000 samples for 400 frames).

**V1 — converter gate.** `Tools/convert/convert.py` writes three bundles: `HeartMuLa-oss-3B-bf16` (LM to bf16),
`HeartCodec-oss-fp32` and `HeartCodec-oss-fp16` (never bf16 — upstream README: bf16 degrades the codec). Transforms, and only
these: drop what decoding never loads (the SQ-Codec encoder, the RVQ's EMA buffers `cluster_size` / `embed_avg` / `initted`,
`vq_embed.project_in`); fold `weight_norm` (`parametrizations.weight.original0/1`) to `weight = g · v / ‖v‖` over all dims but
0, in fp32; Conv1d (O, I, K) → (O, K, I); ConvTranspose1d (I, O, K) → (O, K, I); RVQ codebooks (1, 8192, 32) → plain
(8192, 32) embeddings; the SQ-Codec decoder's `decoder.N` list renamed to named modules (documented table in the converter);
every other key unchanged (`audio_head` stays one (7, 3072, 8197) parameter). Bundles carry `config.json` (the source config +
an `mlx_heartmula` section: source repo / revision, tier, transforms, oracle pin), `gen_config.json`, `tokenizer.json` (LM
bundle), the Apache-2.0 `LICENSE` text (heartlib's) and a `NOTICE` (grant chain: card tags + heartlib README 2026-01-20; no
LICENSE file in the generator repos; Llama-3.2 tokenizer vocabulary). **PASS =** (a) every bundle tensor equals the torch
module's own tensor after the stated transform — LM: `param.to(bf16)` bit-equal; codec: the parametrized `.weight` torch
evaluates in its forward (`module.weight` under `parametrize`) transposed, bit-equal at fp32 and `.to(fp16)` bit-equal at fp16
— 0 missing, 0 extra; (b) MLX `conv1d` / `conv_transposed1d` on the converted layouts reproduce torch's forward of the
SQ-Codec's first conv, one up-conv and one dilated residual conv to relmax ≤ 1e-5 (CPU fp32); (c) the Swift weight-free
modules' flattened parameter paths equal the bundle keys (S0 — 0 missing, 0 extra, every tier); (d) tokenizer, configs,
LICENSE, NOTICE present.

**V2 — Swift core gates** (`heartmula-gates` lanes; relL2 = ‖Δ‖₂/‖ref‖₂; SNR dB = −20·log₁₀ relL2). **Both cases must pass
every lane at fp32** (`MLX_ENABLE_TF32=0`):

| lane | what is compared | rule |
|---|---|---|
| T tokenizer | the Swift Llama-3 BPE (no normaliser, `\p{N}{1,3}` split, `ignore_merges`, BOS template) + `preprocess` on all 12 rows: tags ids, lyrics ids, token matrix, mask, `muq_idx` | id-exact |
| S scalars | the scaled-RoPE θ table (torchtune `apply_scaling`) vs the golden; the codec windowing (372 / 320 / 52 / 104 latents, 1 428 480 / 1 228 800 / 199 680 samples), code-repeat rule, `linspace(0, 1, 11)`, the DiT timestep embedding (2, 512) at t = 0.6 | θ ≤ 1e-7 rel, timestep emb relL2 ≤ 1e-6, integers exact |
| E prompt embedding | `h` (2, L, 3072) | relL2 ≤ 1e-6 |
| P backbone | teacher-forced hidden (last 16) and c0 logits (65 positions × 2 rows): argmax agreement; logits relL2 | 100 % argmax*; relL2 ≤ 1e-3 each |
| L local decoder | logits for codebooks 1…7 at the 4 frames, both rows | relL2 ≤ 1e-3; 100 % argmax* |
| F CFG frame | guided c0 logits and the 7 guided local logits; the 8 codes sampled with the recorded `q` | relL2 ≤ 1e-3; codes exact |
| N DiT velocity | both windows, both CFG halves | relL2 ≤ 1e-3 each |
| W window solve | the condition per window; the 10-step latents of window 0 and of window 1 (window 1 run on the Swift window 0's carried 104 latents) | condition relL2 ≤ 1e-5; latents relL2 ≤ 1e-3 per window |
| Q SQ-Codec | decode of the fixed golden window-0 latents, both channels | SNR ≥ 80 dB; sample count exact |
| D end to end | the Swift `detokenize` of the reference codes with the recorded noise, against the golden waveform | SNR ≥ 80 dB**; sample count exact |

\* **Argmax near-tie clause (stated now):** a position whose golden top-2 logit margin is < 1e-3 (absolute) is reported and
excluded from the 100 %; every other position must agree. \** **Rounding near-tie clause (stated now):** SQ-Codec `decode`
starts with `round(9·x)/9`, a discontinuity, so a 1e-6 latent difference can move a value across a rounding boundary. If D
is below 80 dB, the lane is read by counting the elements where the Swift and golden grids `round(9·x)` differ: it **passes
only if every such element is a near-tie in the golden (|frac(9·x_golden) − 0.5| < 1e-3) and the Swift waveform re-decoded
with the golden grid substituted at exactly those elements reaches ≥ 80 dB.** Any non-tie grid disagreement fails D.

**V2 PASSES when every lane meets its rule on both cases at fp32, and** a real render through the Swift pipeline at the shipped
tier (bf16 LM + fp32 codec; `render` lane: the folk request at a 120 s cap, seed 1004, MLX's RNG — not torch's, so a different
rollout from V0's, reported as such) terminates on the end token or the cap, reports the effective duration, writes finite
48 kHz stereo **through the output limiter**, and records `phys_footprint` after load and at peak (AB-L-0113) with MLX
active / peak beside it. Footprints and stage wall-clock are **reported, not gated** (AB-D-0162; V0 estimate 16–20 GB at bf16
LM + fp32 codec; the declared engine footprint comes at V3). The bf16 lanes are reported against `bf16/` with no rule.

**Output limiter (specified now, outside every parity lane — P…D compare pre-limiter values).** V0 outputs peaked at 1.03–1.37
on 8 of 12 songs. After `detokenize`, the whole song is scaled by `min(1, 0.891 / peak)` — a −1 dBFS peak ceiling applied as
one gain (no compressor, no per-sample clipping; songs already under −1 dBFS are untouched; the gain is reported in the
render result). ACE-Step's port normalises to −1 dBFS likewise.

**Not in V1/V2.** V2b tiers (8-bit / 4-bit LM, fp16 codec, CFG 3.0 as the style lever, the gender re-check), the engine
package (V3), publishing (V4), speed work (AB-D-0162: declared, not gated), reference audio (upstream raises
`NotImplementedError`).

## V1 — oracle capture

Run 2026-10-28 11:10–11:12 PDT (`receipts/dump_goldens.log`, `oracle/goldens/<case>/meta.json`; 142 MB). Load 8.1 s (page
cache), MPS current 14.5 GB at bf16 LM + fp32 codec, 23.7 GB after the LM cast to fp32. A first attempt died after the bf16
phase because `rope_init()` rebuilt the RoPE tables on the CPU; fixed (`rope.to(mps)`) and the whole dump re-run from scratch —
nothing from the first attempt is used.

| | folk (p04, seed 1004) | ballad (p12, seed 1012) |
|---|---|---|
| reference rollout (bf16 + autocast; reference input only) | 401 frames in 30.7 s (13.1 frames/s), **no end token** inside 32 s | 401 frames in 65.8 s (6.1 frames/s), no end token |
| prompt | 184 positions | 185 positions |
| **self-check**: teacher-forced last-prompt c0 logits vs the ones `generate_frame` computed internally | fp32 **1.04e-6**; bf16 4.4e-3 | fp32 **1.0e-6**; bf16 6.7e-3 |
| F (fp32): the 8 codes sampled with the recorded q | [8110, 1924, 6290, 5477, 5908, 1737, 7923, 3788] | [8110, 3398, 6949, 7394, 2979, 6064, 7885, 2875] |
| codec: windows / randn calls / samples | 2 (in-context 0 then 104) / 4 (744, 744, 640, 744 × 256) / 2 × 1 539 840 in 10.9 s | same / same / 2 × 1 539 840 in 10.7 s |
| raw waveform peak | 0.833 | 0.979 |

Facts the dump settled: the 401-frame codes are padded by `cat(codes, codes)` to 692 → windows at 0 and 320 (window 1
re-uses frames 0–290 after the end, as upstream does); `torch.randn` is called 4 times but only the two per-window
`inference_codes` draws reach the output (the first-latent and pad draws are masked to zero); the bf16 model's RoPE `theta` /
`cache` buffers are **fp32** (not cast by `from_pretrained`); torch-MPS `linspace(0, 1, 11)` differs from `i/10` by one ulp at
t = 0.9 (6e-8). The T rows: 12 / 12 prompts dumped (the tokenizer runs without a model).

## V1 — converter

Run 2026-10-28 11:08–11:21 PDT (`receipts/{convert,gate_v1,contract,convcheck}.log`, `oracle/gate_v1_report.json`).

| bundle | tensors | dtypes | size | (a) bit-equal to torch's own | (c) S0 |
|---|---|---|---|---|---|
| `HeartMuLa-oss-3B-bf16` | 289 (2 shards) | BF16 | 7.88 GB | **289 / 289** vs `param.to(bf16)`, 0 missing / 0 extra | 289 = 289 ✅ |
| `HeartCodec-oss-fp32` | 506 (2 shards) | F32 | 6.29 GB | **506 / 506** vs the parametrized `.weight` transposed (weight_norm folded), 0 / 0 | 506 = 506 ✅ |
| `HeartCodec-oss-fp16` | 506 (1 shard) | F16 | 3.15 GB | **506 / 506** vs the same `.to(fp16)`, 0 / 0 | 506 = 506 ✅ |

Codec: 818 source tensors − 255 dropped (the SQ-Codec encoder, the RVQ EMA buffers, `vq_embed.project_in`) − 57 weight_norm (g, v) pairs folded to one weight each = 506. **(b)** MLX on
the converted layouts vs torch on the CPU (fp32): look-ahead conv k5 relmax **1.1e-6**, causal up-conv k10 s5 **3.0e-6**,
causal dilated conv k7 d7 **2.5e-6** (≤ 1e-5) ✅. **(d)** tokenizer, gen config, configs, LICENSE, NOTICE present ✅.

*Disclosed (gate script, not converter):* the first `gate_v1.py` run reported 9 "missing" codec tensors — vector-quantize-
pytorch runtime buffers (`beam_score_weights`, `layers.N.zero`) that exist on the torch module but not in the checkpoint and
that `get_output_from_indices` never reads. The script's reference set was the module's buffers instead of the checkpoint's
keys (the stated rule: "the torch module's own tensor"); corrected to the checkpoint key set and re-run: 506 / 506. The
bundles were not touched.

**V1 PASS** — every clause (a)–(d) on all three bundles.

## V2 — Swift core

Run 2026-10-28 11:21–11:29 PDT, `PROD/HeartMuLa/mlx-heartmula-swift` @ V2 (`HeartMuLaCore`, `heartmula-gates`), release build
(`--build-system swiftbuild`), mlx-swift 0.32.3, `MLX_ENABLE_TF32=0` (`receipts/core-{fp32,shipped}-{folk,ballad}.log`,
`receipts/tokenizer.log`, `receipts/render-p04-120-shipped.log`, audio `receipts/e2e/render-p04-120-shipped.wav`).

### The gate lane — fp32 (`MLX_ENABLE_TF32=0`), against the `fp32/` goldens

| lane | folk (en) | ballad (zh) | rule | |
|---|---|---|---|---|
| T tokenizer + preprocess | **12 / 12 rows id-exact** (tokens, mask, muq_idx, length) | (same run) | id-exact | ✅ |
| S scalars | θ backbone rel 0, decoder **8.4e-8**; timestep emb 0; windowing 372/320/52/104 · 1 428 480/1 228 800/199 680, padded 692, starts [0, 320], 1 539 840 samples; t_span 6e-8 (reported) | same | θ ≤ 1e-7; ≤ 1e-6; exact | ✅ |
| E prompt embedding | relL2 **0** | 0 | ≤ 1e-6 | ✅ |
| P backbone (65 × 2 positions) | argmax **130 / 130** (no near-ties); logits 1.1e-6; hidden tail 1.9e-6 | 130 / 130; 1.3e-6; 2.1e-6 | 100 %; ≤ 1e-3 | ✅ |
| L local decoder (4 frames × 7 × 2) | 1.3–1.6e-6, argmax 14/14 each (chained from the Swift backbone: 1.6–2.2e-6) | 1.3–1.6e-6, 14/14 (chained 1.7–2.2e-6) | ≤ 1e-3; 100 % | ✅ |
| F CFG frame (recorded q) | **codes exact** [8110, 1924, 6290, 5477, 5908, 1737, 7923, 3788]; guided logits ≤ 2.3e-6; c0 pair 9.9e-7 | **exact** [8110, 3398, …, 2875]; ≤ 2.8e-6; 1.1e-6 | ≤ 1e-3; exact | ✅ |
| N DiT velocity t = 0.6 | w0 2.1e-6 / 2.2e-6; w1 2.3e-6 / 2.3e-6 (uncond / cond) | 2.2e-6 / 2.2e-6; 2.2e-6 / 2.3e-6 | ≤ 1e-3 | ✅ |
| W window solve | condition 6.2e-7 / 6.2e-7; latents w0 1.7e-6, w1 1.6e-6 (on the Swift window 0's carry) | 6.1e-7; 1.6e-6, 1.6e-6 | ≤ 1e-5; ≤ 1e-3 | ✅ |
| Q SQ-Codec | **117.7 dB**, 2 × 1 428 480 | **114.8 dB** | ≥ 80 dB | ✅ |
| D end to end | **117.8 dB**, 2 × 1 539 840; **0 grid disagreements** (the near-tie clause never engaged) | **115.0 dB**; 0 | ≥ 80 dB | ✅ |

**9 / 9 lanes PASS on both cases** (+ T on all 12 rows). The two frameworks agree to fp32 rounding noise (1e-7 … 3e-6) across
a 28-layer backbone, the local decoder, the sampler (identical codes from the same noise), a 30-layer DiT over 10 steps, and
the SQ-Codec — and because the latents agree to 1.6e-6 no element crosses a `round(9x)` boundary.

*Disclosed (implementation fixed, rule unchanged):* the first fp32 run read **S FAIL** — θ rel 3.0e-7 (backbone) / 5.4e-7
(decoder) against ≤ 1e-7, every other lane PASS. Cause: torchtune's `apply_scaling` computes `2π / freq` and
`old_context / wavelen` with a Python scalar over a 0-d tensor, which torch evaluates as `reciprocal() * scalar`; the Swift
code divided directly. Mirrored (`LlamaFlavor.ropeTheta`); now backbone bit-equal, decoder 8.4e-8 (one element of 192 is
1 ulp off in torch's own fp32 `pow`). Every lane was re-run on the fixed code; the numbers above are from that run.

### The shipped tier — bf16 LM + fp32 codec, against the `bf16/` goldens (reported, no V2 rule)

| lane | folk | ballad |
|---|---|---|
| P | argmax 119 / 122 (8 near-ties excluded, 2 of them differ); logits relL2 **6.7e-3**; hidden tail 1.0e-2; input **0** | 120 / 122 (8; 3); **7.4e-3**; 1.1e-2; 0 |
| L | 4.6–5.9e-3 (argmax 13–14 of 14 or ties) | 5.1–5.7e-3 |
| F | **codes exact** (same 8 as the golden); guided logits ≤ 1.1e-2 | **exact**; ≤ 1.6e-2 |
| N, W, Q, D | identical to the fp32 lane (the codec is fp32 in both): D **117.8 dB** | D **115.0 dB** |

Reading. The bf16 LM differs from torch's bf16 by ~7e-3 relL2 on logits (both are bf16 approximations; E26 saw 1e-2 between
two MLX bf16 paths). The frame-embedding sum was the one place the Swift side rounded differently (3.4e-3 at the input):
torch sums the 8 bf16 embeddings with an fp32 accumulator, so `frameEmbedding` now does too and the input is bit-equal; the
rest is inside the bf16 kernels. The sampled codes still match the golden's on both cases. V2b owns the bf16 tier rule.

**TF32 finding (disclosed — a V3 requirement).** The first shipped-tier run was launched without `MLX_ENABLE_TF32=0`
(`receipts/core-shipped-tf32on-*.log`, from the code before the embedding-sum change). With MLX's default TF32 matmuls the
fp32 codec drifts from torch-MPS fp32: condition 1.7e-3, velocity 1.0–1.3e-3, window latents 5–7e-3, SQ-Codec **50 dB**, and
end to end **30.2 / 25.5 dB** with **4 491 / 4 937 grid disagreements** (≈ 90 % of them not near-ties): the `round(9x)/9`
quantiser at the head of SQ-Codec amplifies a 1e-3 latent drift into 1/9 steps. TF32 is ≈ 2.5× faster on the codec (1.4 vs
3.5 s per window). The oracle's fp32 codec has no TF32, so the port runs the codec with TF32 off; the engine package (V3)
must set `MLX_ENABLE_TF32=0` before its first MLX op (it is process-wide; the LM is bf16 and unaffected). Whether TF32's
drift is audible is a V2b question, not assumed either way.

### A real render — the shipped tier, folk (p04) at a 120 s cap, seed 1004 (MLX's RNG), TF32 off

| | Swift (this run) | torch-MPS oracle, V0 (AB-R-0628) |
|---|---|---|
| frames / stop | 1217; **ended on the end token at 97.36 s** | p04: 1501, cap (V0 mean 1363; 8 of 12 on the end token) |
| LM (3B backbone + 7 local steps per frame, batch 2 CFG) | 33.8 s = **36.0 frames/s = 2.9× realtime** (prefill 0.40 s for 184 positions) | 10.1 frames/s mean (0.81×) |
| codec: 4 windows × 10 steps × CFG | 3.5–3.6 s per window (14.1 s) | ≈ 4.9 s per window |
| SQ-Codec + crossfade | 3.1 s | 2.4 s |
| end to end (load excluded) | **51.1 s for 97.36 s → RTF 1.91** | RTF 0.68 mean (p04: 0.60) |
| audio | 48 kHz stereo, finite; raw peak 0.979 → **limiter gain 0.910**, out peak 0.891; RMS 0.132 | peaks 1.03–1.37 on 8 of 12, unlimited |
| `phys_footprint` after load / peak | **14.26 GB / 22.43 GB** | 16.4 / 39.1 GB (torch allocator cache) |
| MLX active / peak | 14.21 / 21.27 GB | MPS current-allocated peak 22.8 GB |

Sanity listen-by-proxy (not a gate; one take, a different RNG stream from V0's): the V0 scorer on this render reads **PER
0.022** (V0's p04 take: 0.031), WER 0.074, AES PQ 8.38 / CE 7.68 (V0 p04: 8.54 / 8.08), DNSMOS OVRL 2.20 — the transcript is
the written lyrics in order (`receipts/score-render-p04.log`). So the MLX-RNG sampling path, which no parity lane covers
(lane F replays torch's noise), produces a sung song.

**V2 PASS** by the pre-registered rule: every fp32 lane on both cases, and a real shipped-tier render that stops on the end
token, reports its effective duration (97.36 s of a 120 s cap), writes finite 48 kHz stereo through the −1 dBFS limiter, and
records `phys_footprint` (14.26 / 22.43 GB — inside V0's 16–20 GB working-set estimate after load; a 32 GB Mac).

### Not measured / left for later rungs

- **V2b tiers**: the bf16 lyric check on the 12 V0 songs (paired seeds, ≥ 2 draws or pooled — AB-L-0312), 8-bit / 4-bit LM,
  the fp16 codec (bundle converted and gated at V1; not run), TF32-on codec audibility, CFG 3.0 as the style lever, the
  voice-gender re-check (AB-D-0187).
- **V3**: `MLXHeartMuLa` on `Capability.music` with `MLX_ENABLE_TF32=0` set before the first MLX op, the declared character
  (style adherence, voice-gender gap, `zh` PER), `phys_footprint` per tier. **V4** publish.
- Not exercised: CFG ≤ 1 (single-row) path, cancellation mid-window, songs shorter than one codec window (the repeat rule is
  unit-tested only).

## V2b — tiers: gate (stated 2026-10-28 12:35 PDT, before any tier bundle, tier code or take existed — AB-L-0090)

Operator GO 2026-10-28 (AB-T-0376 "V2b — operator GO"). **What a tier is.** The Swift core (`HeartMuLaCore`, the package at the
V2b commit) renders the 12 V0 songs in one process per tier and draw (`heartmula-gates batch`, the lane added for this rung),
release build, `MLX_ENABLE_TF32=0` for the whole process (AB-L-0315), MLX cache limit 2 GiB. The AR stage is SAMPLED from MLX's
generator, so tiers with different weights take different rollouts from the same seed: every number below is a distribution
over songs, not a parity measure (AB-L-0090).

| tier | LM | codec | how |
|---|---|---|---|
| `bf16` | the V1 bundle `HeartMuLa-oss-3B-bf16` | fp32 (`HeartCodec-oss-fp32`) | the tier V2 gated and rendered |
| `lm8` | **pre-quantised** bundle `HeartMuLa-oss-3B-8bit`: every Linear inside `backbone.layers.*` and `decoder.layers.*` (q/k/v/output_proj, w1/w2/w3 — 28 + 3 layers × 7) affine, group 64, 8-bit, produced from the bf16 bundle with `MLX.quantized` (the op `QuantizedLinear` calls); `text_embeddings`, `audio_embeddings`, `unconditional_text_embedding`, `muq_linear`, `projection`, `codebook0_head`, `audio_head` and the norms stay bf16. `config.json` carries `"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}`; the loader builds `QuantizedLinear`s from it **before** loading — never quantise-at-load | fp32 | |
| `lm4` | the same at 4-bit (`HeartMuLa-oss-3B-4bit`) | fp32 | |
| `c16` | bf16 | **fp16** (`HeartCodec-oss-fp16`, V1-gated) | the `bf16` tier's saved codes re-decoded (`heartmula-gates decode`) with the **same codec noise**, so the codec is the only difference |

Implementation change declared now: the codec noise is drawn from MLX's generator re-seeded with a value derived from the request
seed immediately before `detokenize` (it was the LM's stream continued), so a decode can be repeated on saved codes. The V2 render
is not re-run for it (its gate was termination, duration, finiteness and footprint, not bit-exactness).

**V1-style checks on the new bundles (must pass before any take):** S0 — the weight-free Swift model quantised from the config has
flattened paths == the bundle keys (`weight` + `scales` + `biases` on exactly the 217 layer Linears, nothing else); and the
pre-quantised tensors equal `MLX.quantized` of the bf16 bundle's weights (bit-equal, recomputed by the gate, not by the converter).

**Songs, draws, scorer.** `WIP/acestep-eval/cues/prompts.json` (draw **A**, seeds 1001–1012) and `prompts-b.json` (draw **B**, seeds
2001–2012 — same rows, E25's second seed set); every tier renders both draws on the same seeds (paired across tiers). Request per
row as V0: tags = the `style` string by V0's mechanical transform, lyrics verbatim, 120 s cap, CFG 1.5, top-k 50, temperature 1.0,
codec 10 steps / guidance 1.25. Output = the pipeline's product: 48 kHz stereo float WAV **after the −1 dBFS limiter** (raw peak
and gain in the manifest; V0 scored unlimited audio — the limiter is one gain ≤ 1). Written to
`WIP/heartmula-eval/_out/heartmula-swift-<tier>-<a|b>/` with the V0 harness's manifest row shape and scored by
`WIP/heartmula-eval/harness/score.py` **byte-unchanged** (sha256 `f19b30f2…aedf`; Demucs htdemucs → Whisper-large-v3-turbo MLX →
phoneme error; AudioBox-Aesthetics PQ / CE on the mix; DNSMOS on the vocals), venv `WIP/acestep-eval/.venv-score`, after the
render processes have exited. Catastrophic = PER > 0.60. Reference row: V0 upstream torch-MPS bf16 (AB-R-0628): PER 0.165
(en 0.107 / zh 0.457), 1 catastrophic, PQ 8.30, CE 7.76.

**Rules** (pooled = the mean over the 24 takes A ∪ B; read on the pre-registered seeds, never re-rolled, no best-of-N):

| rule | compares | passes when |
|---|---|---|
| **B** | `bf16` vs V0 torch | pooled PER ≤ 0.165 + 0.05 = **0.215** and ≤ 0.35; pooled PQ ≥ 8.00 and CE ≥ 7.46 (V0 − 0.3); catastrophic ≤ 2 in **each** draw |
| **Q8** | `lm8` vs `bf16` | pooled PER ≤ bf16's + 0.05 and ≤ 0.35; pooled PQ and CE each ≥ bf16's − 0.3; catastrophic (of 24) ≤ bf16's + 2 |
| **Q4** | `lm4` vs `bf16` | the same as Q8 |
| **C16** | `c16` vs `bf16` (identical codes and noise) | pooled PER ≤ bf16's + 0.02; pooled PQ and CE each ≥ bf16's − 0.1; per-song waveform SNR vs the fp32 decode reported |
| **M** (reported) | each tier | `phys_footprint` after load and peak over the process (100 ms sampler, AB-L-0113), MLX peak. Bars to note: the engine admits at 0.7 × RAM — 16.8 GB on a 24 GB Mac, 22.4 GB on 32 GB, 33.6 GB on 48 GB |
| **S** (reported, AB-D-0162) | each tier | wall, RTF, LM frames/s, codec s per window; never a gate |

**Noise, declared now.** AB-L-0312: a 12-song single-take PER mean of a sampled AR moves ≈ 0.08 between draws, so the 0.05 margins
are read on the 24-take pools (≈ 0.06 expected spread), and each tier's |PER_A − PER_B| is reported beside its pool as the yardstick.
A tier missing Q8 / Q4 by ≤ 0.03 is reported as **marginal** — the FAIL stands, the miss is declared, nothing is re-rolled. Because
`c16` decodes the same codes, its Δ measures the codec alone (Whisper's own run-to-run noise is ≈ 0 on identical audio).

**Offer and default (decided by these rules, not by taste — AB-D-0187; speed never decides — AB-D-0162).** V2b passes when **B**
passes. V3 offers every tier whose rule passes. **Default LM tier** = the most compact passing tier whose pooled PER is also within
**0.03** of `bf16`'s (the stricter default bar); otherwise `bf16`. **Default codec** = fp16 if C16 passes, else fp32.

**Reported, not gates (the AB-D-0187 carry):** a `cfg3` run — the `bf16` tier at CFG 3.0 (the example CLI's default) on draw A —
with PER / AES and the V0 CLAP style retrieval (`laion/larger_clap_music_and_speech`, 12-way rank) beside `bf16`-A's, as the
cheapest style lever. Voice gender has no valid automatic instrument (V0 deviation 5): the `bf16`-A and `cfg3` takes of the six
male-voice rows are staged for the operator's ear, not scored. **Rendering discipline:** solo — `pgrep` for other model processes
before each process and a 10 s overlap log during it; a song that overlapped another GPU job keeps its audio (the sampled path does
not depend on timing) and is re-timed solo for S only. Results are appended below this line only after every tier is scored.

## V2b — RESULT (2026-10-28 14:25, receipt AB-R-0636; read after every tier was scored; `WIP/heartmula-eval/_out/summary-v2b.md` is the generated table, `WIP/heartmula-eval/_logs/{batch,decode,score}-<tier>-<draw>.log` and `v2b_overlap.log` the receipts)

**B ✅ PASS, Q8 ✅ PASS, Q4 ❌ FAIL, C16 ✅ PASS → V2b PASSES. By the pre-registered rule V3 offers `bf16` and `lm8` (not
`lm4`) with the fp32 and fp16 codecs; the default is `lm8` + fp16 codec.** Renders 12:38–13:51 PDT, one process at a time,
`MLX_ENABLE_TF32=0`, cache 2 GiB, **0 overlap events** in the 10 s log; scored 13:51–14:10 with the byte-unchanged scorer
(sha256 `f19b30f2…` checked at the start). Pre-quantised bundles before any take: `HeartMuLa-oss-3B-8bit` 723 tensors / 4.94 GB
and `-4bit` 3.38 GB — S0 723 = 723 with exactly 217 scaled Linears, all inside the layers, and every tensor bit-equal to
`MLX.quantized` of the bf16 weights recomputed by `qcheck` (`receipts/{quantize,qcheck}-{8,4}bit.log`). A 10 s unscored smoke on
the 4-bit path rendered twice gave byte-identical WAVs (renders are seed-deterministic).

| tier | draw | PER (en / zh) | cat. | AES PQ / CE | DNS | end token | audio s | RTF | LM fr/s | codec s / window | phys after load / peak GB | MLX peak | CLAP rank (A) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 torch-MPS bf16 (AB-R-0628) | A | 0.165 (0.107 / 0.457) | 1 | 8.30 / 7.76 | 2.14 | 8/12 | 109.1 | 0.68 | 10.1 | 4.9 | 16.4 / 39.1 | — | 5.42 |
| **bf16** (bf16 LM + fp32 codec) | A / B | 0.180 / 0.071 | 1 / 0 | 8.32 / 7.73 · 8.29 / 7.70 | 2.19 / 2.25 | 9 / 10 | 102.9 / 104.1 | 1.85 / 1.83 | 34.5 / 33.8 | 3.55 / 3.52 | **14.26 / 24.03** | 21.3 | 4.50 |
| bf16 | **pooled 24** | **0.125** (0.139 / 0.059) | 1 | **8.30 / 7.71** | 2.22 | 19/24 | 103.5 | 1.84 | 34.2 | 3.53 | | | |
| **lm8** (8-bit LM + fp32 codec) | A / B | 0.169 / 0.094 | 1 / 0 | 8.35 / 7.80 · 8.30 / 7.80 | 2.16 / 2.24 | 6 / 8 | 107.3 / 109.6 | 2.18 / 2.19 | 45.0 / 45.2 | 3.53 / 3.52 | **11.33 / 21.03** | 18.4 | 4.50 |
| lm8 | **pooled 24** | **0.132** (0.139 / 0.092) | 1 | **8.32 / 7.80** | 2.20 | 14/24 | 108.4 | 2.19 | 45.1 | 3.53 | | | |
| **lm4** (4-bit LM + fp32 codec) | A / B | 0.196 / 0.243 | 0 / 1 | 8.24 / 7.74 · 8.23 / 7.73 | 2.22 / 2.11 | 5 / 6 | 108.0 / 101.0 | 2.47 / 2.49 | 56.5 / 56.6 | 3.52 / 3.52 | **9.76 / 19.35** | 16.8 | 4.42 |
| lm4 | **pooled 24** | **0.220** (0.246 / 0.088) | 1 | **8.23 / 7.74** | 2.16 | 11/24 | 104.5 | 2.48 | 56.6 | 3.52 | | | |
| **c16** (bf16 codes → fp16 codec, same noise) | A / B | 0.118 / 0.070 | 0 / 0 | 8.32 / 7.73 · 8.29 / 7.70 | 2.20 / 2.24 | (= bf16) | = bf16 | decode 6.4 s / song | — | ≈ 1.0 | decode-only process 11.11 / 16.79 | 14.7 | 4.58 |
| c16 | **pooled 24** | **0.094** (0.100 / 0.062) | 0 | **8.31 / 7.72** | 2.22 | | | | | | | | |
| cfg3 (bf16, CFG 3.0; reported) | A | 0.115 (0.121 / 0.088) | 0 | 8.37 / 7.80 | 2.42 | 8/12 | 101.9 | 1.78 | 33.2 | 3.53 | 14.26 / 23.82 | 21.3 | 4.42 |
| lm8 + fp16 codec (the default pairing; post-hoc M / S measurement, not scored) | A | — | — | — | — | 6/12 | = lm8 A (codes byte-identical) | **3.08** | 45 | **1.02** | **8.18 / 13.77** | 11.7 | — |

| rule | reading | |
|---|---|---|
| **B** | pooled PER **0.125** (≤ 0.215, ≤ 0.35); PQ 8.30 (≥ 8.00), CE 7.71 (≥ 7.46); catastrophic A 1 / B 0 (≤ 2 each) | ✅ |
| **Q8** | pooled PER **0.132** vs bf16 0.125: **Δ +0.006** (≤ +0.05); PQ 8.32 / CE 7.80 (≥ 8.00 / 7.41); catastrophic 1 (≤ 3); inside the default bar (Δ ≤ 0.03) | ✅ |
| **Q4** | pooled PER **0.220**: **Δ +0.094** (≤ +0.05; a miss of 0.044 — beyond the 0.03 marginal band); PQ 8.23 / CE 7.74 pass; catastrophic 1 | ❌ |
| **C16** | pooled PER **0.094** vs 0.125 (Δ −0.032, ≤ +0.02); PQ 8.31 / CE 7.72 (≥ 8.20 / 7.61); waveform SNR vs the fp32 decode **23.6 dB** mean (15.5–28.4) | ✅ |

Noise yardstick |PER_A − PER_B|: bf16 **0.109**, lm8 0.075, lm4 0.047, c16 0.048 — the AB-L-0312 step again (bf16's draw A carries
two outliers, below). Q8's +0.006 sits far inside it; Q4's +0.094 is about one step, read on 24-take pools with both draws worse
than either bf16 pool's mean (0.196 / 0.243 vs 0.125) — a real 4-bit lyric cost, as E26 / E25 found for their LMs.

**Disclosed readings.**

1. **The PER instrument is fragile on near-identical audio.** `c16` decodes the *same codes with the same noise* as `bf16`, so the
   sung content is the same; yet p09-A reads PER 0.807 through the fp32 codec and 0.098 through fp16. The fp32 transcript is the
   lyrics in order followed by a Whisper repetition loop on the outro ("it is not me" × 40; 224 words for 109 written), the fp16
   one a short loop (128 words). C16's −0.032 is that one take, not a better codec. The rule stands as scored. bf16-A's other
   outlier, p02 (0.570), sings the lyrics and then repeats the chorus to the cap (transcript 1.56× the lyrics).
2. **fp16 codec drift is the TF32 mechanism again** (AB-L-0315): fp16 latents cross `round(9x)/9` boundaries, so the fp16 decode
   is 23.6 dB from the fp32 decode of the same codes. PER / AES / DNS do not move (C16 passes); audibility is the listening
   pass's question, not assumed. The fp16 codec is 3.5× faster per window (1.0 vs 3.5 s) and 3.1 GB smaller.
3. **End token vs the 120 s cap — an observation, not in the pre-registered rule** (the parent session asked for it mid-run):
   capped takes bf16 **5/24**, lm8 **10/24**, lm4 **13/24**, cfg3 4/12 (V0 torch 4/12). On paired seeds lm8 runs to the cap on
   6 takes where bf16 ended and its songs are 62 frames (≈ 5 s) longer on average. What the extra capped lm8 takes sing: the
   lyrics complete and in order, then the chorus or outro repeated (English capped takes' transcripts 1.21× the lyrics vs bf16's
   capped 1.24×; ended takes 1.04×); one real loop — lm8 p06-A, "keep it movin" × 26 to the cap, PER 0.805, the tier's only
   catastrophic take. lm4's capped takes ramble more (1.34×, 6 of 12 over 1.3×). So 8-bit delays or loses the end token about
   twice as often as bf16 without losing lyrics on average; the cost is runtime to the cap and repeated sections. It goes into
   the V3 offer's notes (`duration` is a cap; `effective.duration_s` is reported) — the rule is not changed after the fact.
4. **CFG 3.0 is not a style lever here.** CLAP style rank 4.42 vs bf16-A's 4.50 (V0 torch 5.42; the other systems ≈ 2); PER
   0.115 vs 0.180 and DNS 2.42 vs 2.19 are better on this draw, inside the draw noise. Reported, not adopted; the style gap stays
   declared. Voice gender: the six male-voice rows' bf16-A and cfg3 takes are staged for the operator (below); no automatic reading.
5. **Swift bf16 vs V0 torch:** pooled PER 0.125 vs 0.165 (en 0.139 vs 0.107; zh 0.059 vs 0.457 — p11's lost verse did not recur: every
   Swift take of p11 reads PER ≤ 0.26), PQ / CE 8.30 / 7.71 vs 8.30 / 7.76. Different RNG streams, so a distribution comparison only.
6. The `lm8 + fp16` row was rendered after the scoring (14:10–14:20, solo) only to measure the default pairing's footprint and
   speed; its codes are byte-identical to `lm8`-A's (the codec noise no longer shares the LM's stream), so its lyrics are `lm8`-A's.

**Memory (M) against the engine's 0.7 × RAM admission:** `bf16` + fp32 codec 14.26 / 24.03 GB → a **48 GB** Mac (over 32 GB's
22.4); `bf16` + fp16 codec ≈ 11.1 GB after load (decode-only process; render peak not measured) → 32 GB; `lm8` + fp32 codec
11.33 / 21.03 → 32 GB; **`lm8` + fp16 codec 8.18 / 13.77 → a 24 GB Mac** (bar 16.8); `lm4` 9.76 / 19.35 (not offered).
**Speed (S, never a gate):** RTF bf16 1.84, lm8 2.19, lm4 2.48, lm8 + fp16 codec **3.08** (LM 45 frames/s, codec 1.0 s / window).

**Offer and default (the pre-registered decision rule).** Offer `bf16` and `lm8`, each with the fp32 or fp16 codec; not `lm4`
(Q4 FAIL, +0.094 PER and the most cap-runs). **Default: `lm8` LM + fp16 codec** — the most compact passing LM tier inside the
0.03 default bar, and C16 passed — 8.2 / 13.8 GB, a 24 GB Mac, RTF 3.1. For V3's notes: lm8 runs to the cap about twice as often
as bf16 (observation 3); the fp16 codec is 23.6 dB from fp32 (observation 2). If the operator's ear rejects either after the
staged listening, the fallback is the next tier up (bf16 LM and / or fp32 codec), which this table already measured.

**Listening (evidence, not a gate).** Every tier's draw-A takes are staged as
`WIP/acestep-eval/_out/listen/<id>__heartmula-swift-{bf16,lm8,lm4,c16,cfg3}.wav` (60 symlinks; `harness/stage_listen_tiers.py`).

## V3 — the engine package `MLXHeartMuLa`: gate (stated 2026-10-28 16:55 PDT, before the package was written, built or run — AB-L-0090)

Operator GO 2026-10-28 ("Start V3 now"), scope from V2b's offer. `HeartMuLaPackage` (`@InferenceActor`, `ModelPackage`, module
`MLXHeartMuLa` in `PROD/HeartMuLa/mlx-heartmula-swift`) wraps `HeartMuLaPipeline` on `Capability.music`, built against
mlx-engine-swift **0.72.0** (contract 1.57.0 ≥ 1.56.0, the music capability). Pattern: YuE2 / MiniMax V3.

**Declared, because gated.** The core only: `caption` → the tags (comma-split, trimmed, lower-cased, re-joined — exactly the V0 /
V2b harness's `tags_for`), `lyrics` (section tags inline; empty refused — no instrumental mode was gated), `seed` (nil → drawn),
`durationSeconds` as a **cap** (nil → 120 s, heartlib's pipeline default and the V2b length; `maxDurationSeconds` **240**,
upstream's documented CLI default; the prompt + frames must fit the backbone's 8 192 positions, refused legibly otherwise). The
effective duration is the audio's own length (`effective.duration_s` at the MES layer) and `truncated` is true exactly when the
cap, not the end token, ended the song (C3). **Undeclared, because the model has no such input or nothing gated it:**
`.referenceAudio` / `.sourceAudio` / edits (the MuQ reference slot is a zero vector upstream), `.planning` / `.score` /
`.anchors` (no planner), `.vocalLanguage` (no language slot; it sings the lyrics' script), `.timestamps`, `.guidanceScale`
(CFG 1.5 / codec CFG 1.25 is the recipe; V2b's CFG 3.0 row is an observation, not a gated control). No metaData levers. Progress
phases: `generate` (per LM frame of the cap) → `denoise` (per Euler step, stage = codec window) → `decode` (per SQ-Codec window)
→ `postprocess`; cancellation checked at each of those seams, `CancellationError` rethrown unchanged.

**Tiers (V2b's offer).** `HeartMuLaConfiguration.quant` (the LM) ∈ {`.int8` default, `.bf16`} × `.codec` ∈ {`.fp16` default,
`.fp32`}; the 8-bit LM loads the **pre-quantised** bundle (no quantise-at-load); any other `Quant` is refused by `load()` with
the reason, before any weight is touched. Footprints: `QuantFootprint(.bf16)` and `(.int8)` carry the fp16-codec pairing; the
fp32-codec pairings declare their own resident / activation through `FootprintConfigured` hints. Default-ness is the host's
(`MLXServeEngine.setDefault` keeps ACE-Step for `music`; this package is offered by `PackageID("heartmula-oss-3b")`).

**Licence (C7 / C8).** `weightLicense: .apache2`, `portCodeLicense: .mit`; informational notices: the grant chain (the generator
HF repos carry no LICENSE file — the grant is the model cards + heartlib's README of 2026-01-20; `tokenizer.json` is the
Llama-3.2 tokenizer's vocabulary), and the measured character (weak style-tag adherence: CLAP rank ≈ 4.5 vs ≈ 2 for the other
music models, CFG 3.0 doesn't help; the male-voice gap; the 8-bit LM's cap rate 10/24 vs bf16 5/24 with one loop; weak Mandarin,
V0 zh PER 0.457). Provenance `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` @ `41f6fc68490e11dc43fdabaa6b5767946408c903`, tier 3.
Weight sources (four roles' repos, unpublished V4 candidates): `mlx-community/HeartMuLa-oss-3B-{bf16,8bit}`,
`mlx-community/HeartCodec-oss-{fp32,fp16}`; today the package loads explicit bundle directories.

**TF32 (AB-L-0315).** MLX latches `MLX_ENABLE_TF32` on the first matmul of the process (`static` in `env::enable_tf32()`), so a
package cannot switch it off once a host has run another model. The package (a) exports `HeartMuLaPackage.configureProcess()`
(sets `MLX_ENABLE_TF32=0`) for the host to call at engine start, (b) calls it at the top of `load()`, and (c) probes the
effective setting in `load()` (a 256³ fp32 matmul on the GPU vs the CPU stream; TF32 reads ~1e-4 relL2, fp32 ~1e-7; threshold
1e-5). Two pre-registered checks decide the policy:
- **T1 — the probe classifies:** fresh processes with `MLX_ENABLE_TF32=0` and `=1` read "off" and "on" respectively (relL2 on
  each side of 1e-5 by ≥ 10×).
- **T2 — is the default tier TF32-immune?** `lm8` + fp16 codec, p04 folk, seed 1004, 120 s cap, rendered with TF32 on and off in
  two solo processes. **Immune** = codes byte-identical AND waveform SNR ≥ 60 dB between the two. If immune: fp16-codec tiers
  load whatever the latch says (the reading is logged); fp32-codec tiers refuse under TF32-on with a legible error naming the
  fix (`configureProcess()` before the first MLX op). If not immune: every tier refuses under TF32-on.

**Offline gate** — `swift test --filter MLXHeartMuLaTests` (no kernel evals). PASS = every test green: C1 (`music` only), C7 /
C8 (Apache-2.0 + MIT, `.permissiveOnly` admits, an `.advisory` engine records 0 advisories, `missingRequiredDisplayNotices`
empty), C11 (parameters exactly {caption, lyrics, durationSeconds, seed}, features empty, maxDuration 240,
`declarationProblems` empty), C0 (contract ≥ 1.56.0), provenance (40-char sha), C-memory (footprints `[.bf16, .int8]` ≥ the
measurements below, the fp32-codec hints, nil hints on the fp16 pairings), MAT-1..5 per offered pairing (4), CAN-1..3
(pre-cancelled `run()` → `CancellationError`; cadence generate/token, denoise/step, decode/chunk), the unloaded refusal, the
unoffered-quant refusal, and the request plane (every undeclared feature, bound and empty field refused with
`unsupportedRequestFeature`; the caption → tags mapping; nil → 120 s).

**Live gate — per offered pairing (4), one process at a time, solo** — `heartmula-gates validate` through `MLXServeEngine`
in-process (register under the default policy → prepare → one song: p04 folk, seed 1004, 120 s cap → refusals → evict),
`phys_footprint` sampled at 100 ms, MLX cache 1 GiB, `MLX_ENABLE_TF32=0`. PASS per pairing = registered with **0 licence
advisories**; `needsDownload` false on the explicit bundles; the song finite 48 kHz stereo, length in (10 s, cap], sample peak ≤
−1 dBFS + one 16-bit LSB (the limiter), RMS in (−40, −3) dBFS; phases reported (`generate` > 100, `denoise` ≥ 10 per window,
`decode` ≥ 1, `postprocess` 1); `truncated` consistent with the length (truncated ⇒ length ≥ cap − 1 s; not truncated ⇒ length <
cap); every refusal refused with `PackageError`; phys after evict < 1.5 GB.

**Envelope — per offered pairing** — `heartmula-gates envelope`: loaded as `load()` loads, a 240 s run with the end token ignored
(3 000 frames, the longest KV cache) on the longest V0 lyrics, then the codec over all of it; phys_footprint peak.

**Footprint rule (C-memory).** resident = phys after `prepare` in `validate`, rounded up to the next 0.5 GB; activation = max(the
validate song's peak, the envelope's peak, V2b's 12-song process peak where one exists for that pairing) − that resident,
rounded up to the next 0.5 GB. Expectations written down (V2b): lm8 + fp16 ≈ 8.2 / 13.8 GB → ≈ 8.5 + 5.5 (a 24 GB Mac by the
engine's 0.7 × RAM rule); lm8 + fp32 11.3 / 21.0; bf16 + fp16 ≈ 11.1 / (unmeasured); bf16 + fp32 14.3 / 24.0 (48 GB).

**Test app.** As for E25–E26: the AppKit test app's project on `/Volumes/DEV_VOL1` is closed to tooling, so "through
MLXServeEngine" is met on the CLI consumer path (the same register / prepare / run / evict calls, in-process, with the engine's
admission and licence gate); the app link is recorded as open. **C0–C13** reviewed item by item after the lanes ran. Results are
appended below this line only after the lanes ran. Nothing is pushed or published at V3.

## V3 — RESULT (2026-10-28 17:25, receipt AB-R-0641; read after every lane ran; receipts `WIP/heartmula-port/receipts/{build-v3,test-v3}.log`, `v3-t1-tf32probe.log`, `v3-t2/`, `validate-<pairing>{,-retime,-final}.log`, `envelope-<pairing>.log`, `v3-tf32policy.log`, `v3_overlap{,2}.log`, `validate/p04__engine_<pairing>.wav`)

Package `PROD/HeartMuLa/mlx-heartmula-swift`, module `MLXHeartMuLa` (`HeartMuLaPackage`, surface `heartmula-oss-3b`), built against
mlx-engine-swift **0.73.0** (SwiftPM resolved `from: 0.72.0` to the newer tag; contract 1.58.0 ≥ 1.56.0).

**Offline gate ✅ — 18 / 18 tests green** (`swift test --filter MLXHeartMuLaTests`, no kernel evals): licence Apache-2.0 + MIT,
`.permissiveOnly` admits, an `.advisory` and a `.blocking` engine both register it with 0 advisories, no required notice, three
informational notices (grant chain incl. "no LICENSE file", 2026-01-20, Llama-3.2; the measured character; the cap note);
parameters exactly {caption, lyrics, durationSeconds, seed}, features empty, maxDuration 240, `declarationProblems` empty;
contract ≥ 1.56; provenance 40-char; footprints `[.bf16, .int8]` + fp32-codec hints ≥ every reading below; MAT-1..5 for all four
pairings; CAN-1..3 (generate/token, denoise/step, decode/chunk); the unloaded refusal; `.int4` / `.fp32` LM refused by `load()`
before any weight; the request plane (14 refusals, the tags mapping, the cap → frames mapping). The first run had one red test,
a float-noise bound in the test itself (a stride-made cap of 190.79999999999998 s × 12.5 rounds to exactly 2 385 frames =
190.8 s); the bound got a 1e-9 s tolerance — disclosed, no package change.

**T1 ✅** the probe reads 2.71e-7 with `MLX_ENABLE_TF32=0` and 7.72e-4 with `=1` or unset (37× and 77× either side of 1e-5),
repeatable to the digit. **T2 ✅ immune** — `lm8` + fp16, p04 seed 1004: codes AND the WAV byte-identical with TF32 on and off
(∞ dB), and byte-identical to V2b's `lm8c16`-A codes (the V3 Core edits — progress / cancel probes, `maxMs` rounding — change no
number). So, by the pre-registered rule, the fp16-codec tiers load whatever latched and the fp32-codec tiers refuse under
TF32-on. Verified live (`tf32policy`, reported, not a bar): TF32 latched on by a matmul before the engine → the fp32 codec is
refused with the fix named ("call HeartMuLaPackage.configureProcess() at engine start … or choose the fp16 codec"), the fp16
codec loads; latched off → both load.

**Live gate ✅ PASS on all four pairings** — `heartmula-gates validate` in-process through `MLXServeEngine` (register under the
default policy → prepare → p04 folk, seed 1004, 120 s cap → ten refusals → evict), `MLX_ENABLE_TF32=0`, cache 1 GiB:

| pairing (`Quant` + codec) | phys after prepare | engine charges | peak over the song | song | solo wall (RTF) | phases | after evict |
|---|---|---|---|---|---|---|---|
| **int8 + fp16 (default)** | 8.32 GB | 8.50 GB (final build) | 12.69 / 12.82 / 12.22 GB (3 runs) | 70.24 s, end token (`truncated` false), peak −1.00 dBFS, rms −16.3 dBFS | 23.1 s (**3.04**) | generate 878, denoise 30 (3 windows × 10), decode 3, postprocess 1 | 0.22 GB |
| int8 + fp32 | 11.33 / 11.47 GB | (hint) | 19.47 / 19.60 GB | same codes → 70.24 s, rms −16.1 dBFS | 30.9 s (2.27) | same | 0.20 GB |
| bf16 + fp16 | 11.26 GB | — | 15.75 / 15.78 GB | 97.36 s, end token, rms −17.4 dBFS | 39.4 s (2.47) | generate 1 217, denoise 40, decode 4, postprocess 1 | 0.22 GB |
| bf16 + fp32 | 14.40 GB | (hint) | 21.22 GB | 97.36 s, end token, rms −17.4 dBFS | 52.6 s (1.85) | same | 0.20 GB |

Every pairing: 0 licence advisories, `needsDownload` false on the explicit bundles, the song finite 48 kHz stereo in (10 s, cap]
with the limiter's −1 dBFS ceiling held, `truncated` consistent, all ten refusals `PackageError.unsupportedRequestFeature` with
the control named (planning, score, edit/cover, referenceAudio, vocalLanguage, anchors, durationSeconds 400 > 240 and 0.5 < 1,
empty lyrics, empty caption). Re-runs reproduce each pairing's WAV byte for byte. (The 97.36 s bf16 song is V2's p04 render
length again.)

**Envelope** (240 s, end token ignored, the longest V0 lyrics — p06 hip-hop, 238 prompt positions + 3 000 frames → **240.00 s**
of audio, 10 codec windows): phys peak int8 + fp16 **12.80 GB**, int8 + fp32 18.40, bf16 + fp16 15.68, bf16 + fp32 21.61 —
under each pairing's V2b 12-song peak: the KV cache of 3 238 positions is small next to the codec's activation.

**Declared (the pre-registered rule; resident = the larger phys after prepare, activation = max(song, envelope, V2b process
peak) − resident, each rounded up to 0.5 GB):**

| pairing | declared | binding peak | engine's 0.7 × RAM class |
|---|---|---|---|
| **int8 + fp16 (default)** — `QuantFootprint(.int8)` | **8.5 + 5.5 = 14.0 GB** | 13.77 (V2b 12 songs) | **24 GB Mac** (≤ 16.8) |
| bf16 + fp16 — `QuantFootprint(.bf16)` | 11.5 + 6.0 = 17.5 GB | 16.79 (V2b c16 decode process) | 32 GB |
| int8 + fp32 — hint | 11.5 + 10.0 = 21.5 GB | 21.03 (V2b) | 32 GB (≤ 22.4) |
| bf16 + fp32 — hint | 14.5 + 10.0 = 24.5 GB | 24.03 (V2b) | 48 GB |

**Disclosed against the pre-registration:**
1. **`guidanceScale` is ignored, not refused.** The pre-registration said every undeclared feature is refused; the contract's
   `MusicFeature.guidanceScale` says an undeclared scale is NOT refused (an ignored scale still yields the song asked for), and
   the engine's `MusicControls.refusal` never checks it. The package follows the contract (as MiniMax does); a test pins it.
2. **The cap bounds the audio.** heartlib's rollout counts its first frame on top of `max_audio_length_ms / 80`, so its 120 s cap
   yields 1 501 frames = 120.08 s (every V2b capped take). Before any validate run the package was made to pass
   `floor(cap × 12.5)` frames in all (`SongRequest.lmSeconds(cappingAudioAt:)`), so a capped engine song is ≤ the cap (the
   envelope: exactly 240.00 s). Caps under 1 s are refused (`minDurationSeconds`).
3. **Overlaps — and a blind watcher.** Another session's headless Gaea server spawned image workers (qwen-image, z-image,
   flux2-klein) on and off throughout. The first run's watcher matched `ps comm`, which macOS truncates to `/tmp/g371r/main/`
   when `args` follows, so it saw nothing and the wait-for-free passed; a corrected watcher (full args) then logged a qwen
   worker 17:07:33–17:08:09. The validate lanes int8 + fp16, int8 + fp32 and bf16 + fp16 were therefore marked and **re-timed
   solo** (the fixed runner waited 155 s for the GPU); the timings above are the solo runs (the first int8 + fp32 run read RTF
   0.57). A z-image worker appeared at the first watcher sample after the int8 + fp16 re-time ended — at most its last ≤ 5 s;
   its RTF (3.04) matches V2b's 3.08. bf16 + fp32 and the four envelopes ran clean (lesson AB-L-0317). phys_footprint is per process, so no memory
   reading depends on this; songs are byte-identical across runs. **V2b's watcher had the same `comm` blind spot**: its
   "0 overlaps" covered gates and python jobs only and could not see a Gaea worker — V2b's verdicts rest on scores and
   per-process memory, which an overlap cannot move, but its RTFs carry that caveat.
4. **Contract version.** Pre-registered against 0.72.0 (contract 1.57.0); SwiftPM resolved 0.73.0 (1.58.0). Both ≥ 1.56.0.

**C0–C13 (reviewable):** **C0** `contractVersion` = 1.58.0 (≥ 1.56.0, test). **C1** `Capability.music` only, one surface
`heartmula-oss-3b`. **C2** `MusicRequest` → `MusicResponse` with canonical `.wav` audio (48 kHz, 2 ch), canonical field names.
**C3** `Audio` serialized (16-bit PCM WAV bytes); `truncated` exact; the effective duration is the audio's length (MES
`effective.duration_s`). **C4** no modes; tiers are configuration, not surfaces. **C5** no metaData keys. **C6** `specialties: []`
(a general song generator; non-default by the host's `setDefault`, not by a specialty). **C7** `weightLicense: .apache2`
(allowlisted), grant chain as an informational notice and in each bundle's NOTICE. **C8** `portCodeLicense: .mit`
(THIRD_PARTY_NOTICES credits heartlib Apache-2.0). **C9** `HeartMuLaConfiguration` (`quant`, `codec`, revisions,
`cacheLimitBytes`) Codable with environment URLs excluded; `QuantConfigured`, `FootprintConfigured`, `WeightSourcing`,
`WeightPrewarming`, `ModelStorable`. **C10** `RequirementsManifest`: two `QuantFootprint`s + two hints measured as
phys_footprint through the engine, `requiredBackends [.metalGPU]`, macOS 26. **C11** the surface declares exactly the core;
the engine refuses the rest before admission (validated live). **C12** the package switches over `Quant` with `default`, no
closed-enum assumption over `Capability`. **C13** `registration = .of(HeartMuLaPackage.self)`; no compute in `init`; weights
in `load()`, released in `unload()` (phys 0.20–0.22 GB after evict); `run` checks cancellation at entry and at every LM frame,
Euler step and SQ-Codec window, rethrowing `CancellationError` unchanged; no private queue. **One process-wide side effect,
declared:** `load()` sets `MLX_ENABLE_TF32=0` (`configureProcess()`), which is effective only before the process's first matmul
— the contract has no field for a process-environment requirement (asked of mlx-engine: AB-A-0215).

**Test app.** As pre-registered: met on the CLI consumer path (the same `MLXServeEngine` register / prepare / run / evict calls,
in-process, with the engine's admission and licence gate); the AppKit test app link stays open.

## Verdict (2026-10-28)

**V3 PASS.** `MLXHeartMuLa` is admitted on `Capability.music` through `MLXServeEngine` on all four offered pairings, with the
footprints measured and declared: **default int8 LM + fp16 codec, 8.5 + 5.5 GB — a 24 GB Mac, RTF ≈ 3**; bf16 + fp16 17.5 GB and
int8 + fp32 21.5 GB (32 GB Macs); bf16 + fp32 24.5 GB (48 GB). Apache-2.0, non-default beside ACE-Step, the measured character in
the notices. Hosts call `HeartMuLaPackage.configureProcess()` at engine start for the fp32 codec; the default tier is
TF32-immune. Nothing pushed or published; V4 is the operator's call.

## V4 — published (2026-11-01, the operator's decision AB-D-0203)

The operator: "Publish V4" — after the listening pass over the V2b takes found no defects (relayed by the music research
session; recorded as AB-D-0203, modelled on YuE2's AB-D-0186, with Apache-2.0 weights instead of non-commercial ones).

**Code:** [github.com/xocialize/mlx-heartmula-swift](https://github.com/xocialize/mlx-heartmula-swift), public, MIT. Tags
`v0.1.0` (`79b0c47`) and **`v0.1.1`** (`f3d1f83`, use `from: "0.1.1"`); products `MLXHeartMuLa`, `HeartMuLaCore`; depends on
mlx-swift ≥ 0.32.3 and mlx-engine-swift ≥ 0.72.0. **v0.1.0 has a defect, fixed in v0.1.1:** its declared download globs
(`config.json`, `*.safetensors`, `tokenizer.json`, `gen_config.json`) did not carry `model.safetensors.index.json`, so an engine
download of the two SHARDED repos (bf16 LM, fp32 codec) could not load (`missing file … model.safetensors`); the default pairing
(both single-file) was unaffected. Found by the fresh-download check below (log `receipts/v4-published-artifact-check-v0.1.0.log`);
v0.1.1 declares `*.json` + `*.safetensors` and its layout test now asserts that every file the loader opens is declared. The
v0.1.0 tag stays (it was public); the README and cards point at 0.1.1.

**Weights** (Apache-2.0; the names `HeartMuLaConfiguration` declares, pinned at the weight commits):

| repo @ pinned commit | contents | size |
|---|---|---|
| [`mlx-community/HeartMuLa-oss-3B-8bit`](https://huggingface.co/mlx-community/HeartMuLa-oss-3B-8bit) @ `ff4fbf6` | the LM, pre-quantised 8-bit (723 tensors: 217 quantised Linears), tokenizer, gen config — the default | 4.94 GB |
| [`mlx-community/HeartMuLa-oss-3B-bf16`](https://huggingface.co/mlx-community/HeartMuLa-oss-3B-bf16) @ `625d8d0` | the LM, bf16 (289 tensors, 2 shards + index), tokenizer, gen config | 7.88 GB |
| [`mlx-community/HeartCodec-oss-fp16`](https://huggingface.co/mlx-community/HeartCodec-oss-fp16) @ `5865bd0` | the codec decoder, fp16 (506 tensors) — the default | 3.15 GB |
| [`mlx-community/HeartCodec-oss-fp32`](https://huggingface.co/mlx-community/HeartCodec-oss-fp32) @ `a03fdf4` | the codec decoder, fp32 (506 tensors, 2 shards + index) | 6.29 GB |

Each repo ships heartlib's Apache-2.0 text as `LICENSE`, a `NOTICE` (the gated bundle's grant chain — no LICENSE file in the
generator repos, the card + heartlib's README of 2026-01-20, the Llama-3.2 tokenizer vocabulary — plus the Apache §4 statement of
exactly what changed and "not endorsed by the HeartMuLa team or MuLa Labs"), and a card (`license: apache-2.0`, `base_model` the
HeartMuLa repo, `library_name: mlx`) with its own Use section written for that repo (the default 8-bit LM; the bf16 LM choice; the
default fp16 codec, TF32-immune; the fp32 codec with the TF32 requirement), the declared footprints, the measured character (weak
style tags, the male-voice gap, the 8-bit cap rate 10/24 vs 5/24 with its one loop, weak and uneven Mandarin) and a link to the
Swift package. Not published: the 4-bit LM (V2b Q4 FAIL). A second commit on each repo's `main` changed only `README.md`
(`from: "0.1.1"`); the pins are the weight commits. Cards and NOTICEs as uploaded: `WIP/heartmula-port/publish/*/`.

**Published-artifact check** (`receipts/v4-published-artifact-check.log`, `v4-fresh-validate-{int8-fp16,bf16-fp32}.log`): a
fresh `git clone --branch v0.1.1` built from scratch; `heartmula-gates validate --store <empty dir>` hands the engine that models
root (`useModelStore`), the package declares its pinned repos, and **the engine's own materializer downloads them** (no `hf`
pre-download), then the V3 checks run:

| pairing | download (engine) | phys after prepare / engine charges | peak | song | refusals | result |
|---|---|---|---|---|---|---|
| int8 + fp16 (default) | 8-bit LM + fp16 codec, 153.5 s | 8.34 / 8.50 GB | 12.21 GB | 70.24 s, **byte-identical to V3's** | 10 / 10 | PASS |
| bf16 + fp32 | bf16 LM + fp32 codec, 287.0 s | 14.42 / 14.50 GB | 21.54 GB | 97.36 s, **byte-identical to V3's** | 10 / 10 | PASS |

`needsDownload` true on the empty store, false after; 0 licence advisories. **sha256 of every downloaded weight / config file ==
the gated bundle: 16 / 16, 0 mismatches** (`receipts/v4-gated-sha256.txt` vs `v4-fresh-sha256.txt`). The two runs together
downloaded all four repos, so every published byte the package loads is the gated byte, end to end. Both runs started with no
other model process on the GPU (full-args watcher, AB-L-0317).
