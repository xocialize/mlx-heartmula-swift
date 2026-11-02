# mlx-heartmula-swift

[HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year) — tags + lyrics → a complete song with
vocals, 48 kHz stereo — ported to Swift-MLX for Apple Silicon. A full port from upstream
[`HeartMuLa/heartlib`](https://github.com/HeartMuLa/heartlib) @ 5858ca8, which is the parity oracle (run on torch-MPS).
Task AB-T-0376 (E27); V0 gate AB-R-0628; operator listening pass GO, AB-D-0187.

**Licence.** This code is MIT. The weights are **Apache-2.0** by their model cards and heartlib's README (the 2026-01-20
switch covers "all related model weights"); the generator repos carry no LICENSE file, and `tokenizer.json` is the Llama-3.2
vocabulary. Every bundle the converter writes carries heartlib's Apache-2.0 text and a NOTICE with the grant chain
(`THIRD_PARTY_NOTICES.md`).

**Character, declared (V0, the listening passes, V2b).** Lyrics are followed closely in English: PER 0.125 with the bf16 LM
and 0.132 with the 8-bit LM on the fleet's 12 songs, against the PyTorch reference's 0.165. The aesthetics proxy scores it
highest of the slot on content enjoyment (CE 7.76). Tags steer it least of the slot: CLAP style retrieval rank about 4.5
(PyTorch reference 5.42) against ≈ 2 for the other music models, and CFG 3.0 doesn't help. It has a voice-gender gap: 11 of
12 V0 takes sound female, and 1 of 6 male-voice rows got a male voice. Mandarin is weak and uneven (reference PER 0.457 with
one lost verse). The 8-bit LM runs to the length cap twice as often as bf16 (10 of 24 songs against 5 of 24): the lyrics are
sung, then choruses repeat, and one take looped a line. Raw codec output peaks above full scale (to 1.37), so the pipeline
applies a −1 dBFS limiter.

## Install

```swift
.package(url: "https://github.com/xocialize/mlx-heartmula-swift", from: "0.1.0")
// products: "MLXHeartMuLa" (the MLXEngine package), "HeartMuLaCore" (the model alone)
```

Weights (Apache-2.0, each hash-equal to the bundle these gates passed, pinned by `HeartMuLaConfiguration`):

| repo | what | size |
|---|---|---|
| [mlx-community/HeartMuLa-oss-3B-8bit](https://huggingface.co/mlx-community/HeartMuLa-oss-3B-8bit) | the LM, pre-quantised 8-bit (default) | 4.94 GB |
| [mlx-community/HeartMuLa-oss-3B-bf16](https://huggingface.co/mlx-community/HeartMuLa-oss-3B-bf16) | the LM, bf16 | 7.88 GB |
| [mlx-community/HeartCodec-oss-fp16](https://huggingface.co/mlx-community/HeartCodec-oss-fp16) | the codec, fp16 (default) | 3.15 GB |
| [mlx-community/HeartCodec-oss-fp32](https://huggingface.co/mlx-community/HeartCodec-oss-fp32) | the codec, fp32 (needs TF32 off) | 6.29 GB |

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
let pipe = try HeartMuLaPipeline(lmDir: lmBundle, codecDir: codecBundle)          // e.g. 8-bit LM (4.9 GB) + fp16 codec (3.2 GB)
let song = try pipe.generate(SongRequest(tags: "folk,acoustic guitar,female vocal", lyrics: "[verse]\n…", maxSeconds: 120, seed: 7))
// song.audio (2, samples) @ 48 kHz, song.effectiveSeconds, song.endedOnEndToken, song.limiterGain
```

### Through MLXEngine (`MLXHeartMuLa`)

`HeartMuLaPackage` is a `ModelPackage` on `Capability.music` (mlx-engine-swift contract ≥ 1.56.0), a permissive, **non-default**
offer: the host keeps ACE-Step as the capability's default and offers this one by name.

```swift
import MLXHeartMuLa
import MLXServeCore
import MLXToolKit
HeartMuLaPackage.configureProcess()          // at engine start, before ANY model runs: MLX latches MLX_ENABLE_TF32 on the first matmul
let engine = MLXServeEngine()
await engine.useModelStore(ModelStore(root: modelsRoot))                       // where the engine keeps downloaded weights
let id = try await engine.register(HeartMuLaPackage.registration,
                                   configuration: HeartMuLaConfiguration(quant: .int8, codec: .fp16),  // the default tier
                                   id: PackageID("heartmula-oss-3b"))
try await engine.prepare(.music, package: id)                                  // downloads the two pinned repos on first use
let r = try await engine.run(MusicRequest(caption: "folk, acoustic guitar, female vocal", lyrics: "[verse]\n…",
                                          durationSeconds: 120, seed: 7), package: id) as! MusicResponse
// r.audio: .wav, 48 kHz stereo, peak ≤ −1 dBFS; its length is the effective duration; r.truncated: the cap ended it
```

The surface declares the core only — the caption (read as comma-separated tags), lyrics, `durationSeconds` as a **cap** (nil →
120 s, at most 240 s; the audio never runs past it) and seed — and the engine refuses reference audio, edits, planning, a score,
anchors and a vocal language before admission; an undeclared `guidanceScale` is ignored (the contract's rule). For local
bundles, pass `lmDirectory:` / `codecDirectory:` instead of a models root. Tiers: LM `.int8` (the pre-quantised bundle) or `.bf16`, codec `.fp16` or
`.fp32`; the fp32 codec refuses to load if TF32 latched on in the process (the fp16 tiers are byte-identical either way).

## Status

V0–V4 done (task AB-T-0376): V1 converter + goldens, V2 `HeartMuLaCore` at the pre-registered parity floors, V2b tiers, V3 the
`MLXHeartMuLa` engine package (admitted through `MLXServeEngine` on all four pairings), V4 published (AB-D-0203) — see
`MEASUREMENTS.md`. Tiers: bf16 or pre-quantised 8-bit LM (layer Linears affine g64), fp32 or fp16 codec. Declared footprints
(phys_footprint, resident + activation):

| LM + codec | declared | fits (engine 0.7 × RAM) | speed, M5 Max |
|---|---|---|---|
| **8-bit + fp16 (default)** | 8.5 + 5.5 GB | 24 GB Mac | 3.0× real time |
| bf16 + fp16 | 11.5 + 6.0 GB | 32 GB | 2.5× |
| 8-bit + fp32 | 11.5 + 10.0 GB | 32 GB | 2.3× |
| bf16 + fp32 | 14.5 + 10.0 GB | 48 GB | 1.85× |

4-bit failed its lyric rule and is not offered or published. Open: the AppKit test-app link (the engine path is proven on the
CLI consumer path).

## Build

`swift build -c release --build-system swiftbuild` (the Xcode build system ships the MLX metallib). macOS 26+, Apple Silicon.
