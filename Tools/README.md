# Tools

Nothing here assumes a directory layout: every location is an argument. Run with the V0 oracle's venv
(`mlxengine-audio/WIP/heartmula-eval/.venv`: Homebrew Python 3.12, torch 2.10.0, torchtune 0.4.0, torchao 0.9.0,
transformers 4.57.0, tokenizers 0.22.1, vector-quantize-pytorch 1.27.15, heartlib @ 5858ca8 installed `-e --no-deps`).

| path | what |
|---|---|
| `oracle-capture/dump_goldens.py` | **V1 oracle.** Upstream heartlib on torch-MPS (`torch.compile` off, the MPS `timestep_embedding` cast shim; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4`). Per case (p04 folk, p12 zh ballad): a bf16 reference rollout of 400 frames (reference inputs only), then the LM lanes E/P/L/F at `bf16/` (as shipped, autocast) and `fp32/` (bf16 values cast to fp32), the codec lanes N/W/Q/D once (fp32), every sampler / `torch.randn` draw recorded; plus the 12 V0 rows' tokenizer goldens. `--ckpt <ckpt> --out <goldens> --prompts <prompts.json> [--cases folk,ballad]` (~2 min) |
| `convert/convert.py` | **V1 converter.** → `HeartMuLa-oss-3B-bf16`, `HeartCodec-oss-fp32`, `HeartCodec-oss-fp16` (decode path only, weight_norm folded with `torch._weight_norm`, MLX conv layouts, RVQ codebooks as embeddings, SQ-Codec decoder renamed), each with `config.json` (+ `mlx_heartmula`), `LICENSE`, `NOTICE`; the LM bundle also `tokenizer.json` + `gen_config.json`. `--ckpt <ckpt> --heartlib <heartlib checkout> --out <dir> [--only lm,codec32,codec16]` |
| `convert/gate_v1.py` | **V1 gate (a)+(d).** Every bundle tensor bit-equal to the torch module's own (the LM `.to(bf16)`; the codec's parametrized `.weight` transposed, fp32 and `.to(fp16)`); files present; writes the conv references for clause (b). `--ckpt --weights --refs` |

| `harness/run_tiers.sh` | **V2b renders.** bf16 / lm8 / lm4 × draws A, B (12 songs each), c16 (the bf16 codes re-decoded with the fp16 codec, same noise), cfg3 (bf16 at CFG 3.0, draw A), one process at a time with a 10 s overlap log; `MLX_ENABLE_TF32=0`. Writes `WIP/heartmula-eval/_out/heartmula-swift-<tier>-<a\|b>/` in the V0 harness's manifest shape. |
| `harness/summarize_tiers.py`, `harness/style_fit_tiers.py` | V2b summary (pooled tables, the pre-registered rules) and the V0 CLAP style instrument over any `_out/<system>`; versioned here, RUN from `WIP/heartmula-eval/harness/` (they import its `common.py`). Scoring is the V0 `score.py`, byte-unchanged. |

Swift-side lanes are `heartmula-gates` subcommands (header of `Sources/heartmula-gates/main.swift`): `contract` (S0),
`convcheck` (V1 b), `tokenizer` (T), `core` (S E P L F N W Q D), `render`, and for V2b `quantize` (the pre-quantised 8- / 4-bit
LM bundles: the 217 layer Linears affine g64 via `MLX.quantized`), `qcheck` (S0 on the quantised model + bit-equality with
`MLX.quantized` of the bf16 weights), `batch` (every row of a prompts file → WAV + `.codes.bin` + manifest) and `decode` (a
batch's saved codes through another codec bundle with the same codec noise). Build with the Xcode build system so the MLX
metallib ships: `swift build -c release --build-system swiftbuild`. **Run every lane and every render with
`MLX_ENABLE_TF32=0`**: with MLX's default TF32 matmuls the fp32 codec drifts 1e-3 from torch and the SQ-Codec's
`round(9x)/9` grid amplifies it to 25–30 dB end to end (V2, `receipts/core-shipped-tf32on-*.log`).
