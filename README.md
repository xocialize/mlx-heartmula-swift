# mlx-heartmula-swift

[HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year) — tags + lyrics → a complete song with
vocals, 48 kHz stereo — ported to Swift-MLX for Apple Silicon. A full port from upstream
[`HeartMuLa/heartlib`](https://github.com/HeartMuLa/heartlib) @ 5858ca8, which is the parity oracle (run on torch-MPS).
Task AB-T-0376 (E27); V0 gate AB-R-0628; operator listening pass GO, AB-D-0187.

**Licence.** This code is MIT. The weights are **Apache-2.0** by their model cards and heartlib's README (the 2026-01-20
switch covers "all related model weights"); the generator repos carry no LICENSE file, and `tokenizer.json` is the Llama-3.2
vocabulary. Every bundle the converter writes carries heartlib's Apache-2.0 text and a NOTICE with the grant chain
(`THIRD_PARTY_NOTICES.md`).

**Character, declared (V0 + listening pass).** Lyrics are followed closely (English PER 0.107; Mandarin 0.457 with one lost
verse); the aesthetics proxy scores it highest of the slot on content enjoyment (CE 7.76). Tags steer it least of the slot
(CLAP style retrieval rank 5.42 vs ≈ 2), and it has a voice-gender gap — 11 of 12 V0 takes sound female, 1 of 6 male-voice
rows got a male voice. Raw codec output peaks above full scale (to 1.37), so the pipeline applies a −1 dBFS limiter.

## What it does

- **Prompt** (`PromptBuilder`): `<tag>…</tag>` and the lyrics, lower-cased, Llama-3 BPE with BOS / EOS, one MuQ slot between
  them (a zero vector upstream: only `muq_linear`'s bias); 9 columns per position, a CFG row with the unconditional embedding.
- **Generate** (`Rollout`, `FrameGenerator`): the 3B backbone (torchtune Llama-3.2: interleaved RoPE θ 500 000 with Llama-3
  scaling) emits codebook 0 per 12.5 Hz frame at batch 2 (CFG 1.5, top-k 50); the 3-layer local decoder emits codebooks 1–7
  through per-codebook `audio_head` matrices. Stops on the audio end token (any code ≥ 8193) or the cap — the cap is a cap,
  and the result reports the effective duration.
- **Decode** (`Detokenizer`): RVQ embeddings → condition at 25 Hz; a 30-layer LLaMA-DiT with adaLN-single solves 10 Euler
  steps per 29.76 s window (CFG 1.25), carrying 104 latents of in-context between windows; the SQ-Codec decoder turns each
  window's two 128-d channel halves into stereo; windows crossfade linearly over 199 680 samples.

```swift
import HeartMuLaCore
setenv("MLX_ENABLE_TF32", "0", 1)            // before the first MLX op: the fp32 codec is TF32-sensitive (see Tools/README.md)
let pipe = try HeartMuLaPipeline(lmDir: lmBundle, codecDir: codecBundle)          // bf16 LM (7.9 GB) + fp32 codec (6.3 GB)
let song = try pipe.generate(MusicRequest(tags: "folk,acoustic guitar,female vocal", lyrics: "[verse]\n…", maxSeconds: 120, seed: 7))
// song.audio (2, samples) @ 48 kHz, song.effectiveSeconds, song.endedOnEndToken, song.limiterGain
```

## Status

V1 (converter + goldens) and V2 (`HeartMuLaCore` at the pre-registered parity floors) — see `MEASUREMENTS.md`. Not yet:
V2b tiers (8-/4-bit LM, fp16 codec), V3 engine package on `Capability.music`, V4 publish.

## Build

`swift build -c release --build-system swiftbuild` (the Xcode build system ships the MLX metallib). macOS 26+, Apple Silicon.
