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
