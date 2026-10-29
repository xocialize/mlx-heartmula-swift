# Third-party notices

| component | origin | licence | how it is used |
|---|---|---|---|
| HeartMuLa-oss-3B weights | `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` @ 41f6fc6 (HF; card read 2026-10-28) | **Apache-2.0** — card front-matter + "License: Apache 2.0"; the repo has **no LICENSE file**; heartlib's README (2026-01-20): "We update the license of this repo and all related model weights to Apache 2.0." | converted to bf16 by `Tools/convert/convert.py`; the bundle carries heartlib's Apache-2.0 text and a NOTICE with the grant chain; not redistributed by this package |
| HeartCodec weights | `HeartMuLa/HeartCodec-oss-20260123` @ f889dab | Apache-2.0 (card; no LICENSE file; same grant chain) | decode half converted (weight_norm folded, MLX conv layouts) at fp32 and fp16 |
| tokenizer + gen config | `HeartMuLa/HeartMuLaGen` @ 2e18e01 | Apache-2.0 (card; no LICENSE file). `tokenizer.json` is the Llama-3.2 BPE vocabulary — worst reading the Llama 3.2 Community License, which permits commercial use | copied into the LM bundle; `BPETokenizer.swift` reads it |
| heartlib | `HeartMuLa/heartlib` @ 5858ca8 | Apache-2.0 (LICENSE file, added 2026-01-19) | the parity ORACLE on torch-MPS (`Tools/oracle-capture`); the Swift core is a translation of its modules (`heartmula/modeling_heartmula.py`, `heartcodec/models/{flow_matching,transformer,sq_codec,decoder}.py`, `pipelines/music_generation.py`) |
| torchtune | 0.4.0 (`llama3_2` builders, `Llama3ScaledRoPE`, `MultiHeadAttention`) | BSD-3-Clause | the backbone / local-decoder semantics (interleaved RoPE, Llama-3 scaling, fp32 RMSNorm) are reproduced; no code copied |
| vector-quantize-pytorch | 1.27.15 (`ResidualVQ.get_output_from_indices`) | MIT | the RVQ lookup semantics are reproduced |
| mlx-swift | ml-explore/mlx-swift 0.32.3 | MIT | runtime |
| mlx-engine-swift | xocialize/mlx-engine-swift 0.73.0 (`MLXToolKit`, `MLXServeCore`, `MLXServeConformance`) | MIT | the engine contract `MLXHeartMuLa` implements; the coordinator the V3 lanes drive |
| BPE tokenizer, NPY / WAV / WeightIO / Footprint helpers | mlx-minimax-music3-swift / mlx-ace-step-swift / mlx-yue2-swift (fleet, MIT) | MIT | lifted in shape |
| `Acelogic/heartlib-mlx` | GitHub | **no LICENSE file** (unlicensed) | NOT used — no code read or copied |
| MuQ-MuLan | `OpenMuQ/MuQ-MuLan-large` | CC-BY-NC-4.0 | NOT on this path — the pipeline feeds a zero 512-d vector (`muq_linear` bias only); reference audio is unsupported upstream |
