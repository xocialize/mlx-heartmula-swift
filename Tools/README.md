# Tools

Nothing here assumes a directory layout: every location is an argument. Run with the V0 oracle's venv
(`mlxengine-audio/WIP/heartmula-eval/.venv`: Homebrew Python 3.12, torch 2.10.0, torchtune 0.4.0, torchao 0.9.0,
transformers 4.57.0, tokenizers 0.22.1, vector-quantize-pytorch 1.27.15, heartlib @ 5858ca8 installed `-e --no-deps`).

| path | what |
|---|---|
| `oracle-capture/dump_goldens.py` | **V1 oracle.** Upstream heartlib on torch-MPS (`torch.compile` off, the MPS `timestep_embedding` cast shim; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4`). Per case (p04 folk, p12 zh ballad): a bf16 reference rollout of 400 frames (reference inputs only), then the LM lanes E/P/L/F at `bf16/` (as shipped, autocast) and `fp32/` (bf16 values cast to fp32), the codec lanes N/W/Q/D once (fp32), every sampler / `torch.randn` draw recorded; plus the 12 V0 rows' tokenizer goldens. `--ckpt <ckpt> --out <goldens> --prompts <prompts.json> [--cases folk,ballad]` (~2 min) |
| `convert/convert.py` | **V1 converter.** → `HeartMuLa-oss-3B-bf16`, `HeartCodec-oss-fp32`, `HeartCodec-oss-fp16` (decode path only, weight_norm folded with `torch._weight_norm`, MLX conv layouts, RVQ codebooks as embeddings, SQ-Codec decoder renamed), each with `config.json` (+ `mlx_heartmula`), `LICENSE`, `NOTICE`; the LM bundle also `tokenizer.json` + `gen_config.json`. `--ckpt <ckpt> --heartlib <heartlib checkout> --out <dir> [--only lm,codec32,codec16]` |
| `convert/gate_v1.py` | **V1 gate (a)+(d).** Every bundle tensor bit-equal to the torch module's own (the LM `.to(bf16)`; the codec's parametrized `.weight` transposed, fp32 and `.to(fp16)`); files present; writes the conv references for clause (b). `--ckpt --weights --refs` |

Swift-side lanes are `heartmula-gates` subcommands (header of `Sources/heartmula-gates/main.swift`): `contract` (S0),
`convcheck` (V1 b), `tokenizer` (T), `core` (S E P L F N W Q D), `render`. Build with the Xcode build system so the MLX
metallib ships: `swift build -c release --build-system swiftbuild`. **Run every lane and every render with
`MLX_ENABLE_TF32=0`**: with MLX's default TF32 matmuls the fp32 codec drifts 1e-3 from torch and the SQ-Codec's
`round(9x)/9` grid amplifies it to 25–30 dB end to end (V2, `receipts/core-shipped-tf32on-*.log`).
