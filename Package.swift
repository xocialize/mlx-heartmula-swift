// swift-tools-version: 6.2
// mlx-heartmula-swift — HeartMuLa-oss-3B (MuLa Labs; tags + lyrics → a 48 kHz stereo song through a 3B Llama-3.2-shaped global
// LM and a 3-layer local decoder over 8 × 8197 RVQ codes at 12.5 Hz, then HeartCodec: a flow-matching LLaMA-DiT to 25 Hz latents
// and the SQ-Codec decoder to audio) ported to Swift-MLX. Gates and numbers: MEASUREMENTS.md (mirrors
// mlxengine-audio/WIP/heartmula-port/MEASUREMENTS.md); task AB-T-0376 (E27), V0 AB-R-0628, listening pass GO AB-D-0187.
//
// Licence: this package's code is MIT. The weights (HeartMuLa/HeartMuLa-oss-3B-happy-new-year, HeartMuLa/HeartCodec-oss-20260123,
// HeartMuLa/HeartMuLaGen) are Apache-2.0 by their model cards and heartlib's README (2026-01-20 switch for "all related model
// weights"); the generator repos carry no LICENSE file; tokenizer.json is the Llama-3.2 vocabulary. Tools/convert/convert.py
// writes the full grant chain into each bundle's NOTICE.
//
// Pipeline (HeartMuLaCore, 1:1 with HeartMuLa/heartlib @ 5858ca8 — the oracle, run on torch-MPS):
//   • Prompt     — `<tag>…</tag>` + lyrics, lower-cased, Llama-3 byte-level BPE with BOS / EOS, one zero MuQ slot between them;
//                  9 columns per position (8 audio codebooks + text), CFG row with the unconditional text embedding.
//   • Backbone   — torchtune llama3_2 3B (28 × 3072, 24 q / 8 kv heads × 128, SwiGLU 8192, interleaved RoPE θ 500 000 with
//                  Llama-3 scaling ×32) over summed embeddings; `codebook0_head` → codebook 0 (CFG 1.5, top-k 50).
//   • Local      — the 300M decoder (3 × 3072, 8 q / 4 kv heads × 384) over [last hidden ; codebook embeddings], per-codebook
//                  `audio_head` matrices → codebooks 1…7 (CFG 1.5, top-k 50); stops when any code ≥ 8193.
//   • HeartCodec — RVQ embeddings (8 × 8192 × 32 → 512) → `cond_feature_emb` → nearest ×2 to 25 Hz; LLaMA-DiT (24 × 1536 +
//                  6 × 3072, adaLN-single) flow matching, 10 Euler steps, CFG 1.25, 29.76 s windows at hop 320 frames with the
//                  104-latent in-context carry; SQ-Codec decoder (round(9x)/9, up 5·4·4·4·3 ×2 = 1920) per stereo channel;
//                  linear crossfade of 199 680 samples; −1 dBFS output limiter.
//   • MLXHeartMuLa — the engine-facing package on `Capability.music` (mlx-engine-swift, contract ≥ 1.56.0): the core surface only
//                  (caption as tags, lyrics, duration as a cap, seed), Apache-2.0 weights with the grant chain and the measured
//                  character as notices, tiers bf16 / pre-quantised 8-bit LM × fp32 / fp16 codec with `phys_footprint`
//                  footprints, non-default beside ACE-Step.
//   • heartmula-gates — the V1 contract / conv-layout checks and the V2 parity lanes against the torch-MPS goldens
//                  (Tools/oracle-capture), a render lane, the V2b batch / decode lanes and the V3 validate / envelope / tf32
//                  lanes (MLXServeEngine register → prepare → run → refusals → evict).
import PackageDescription

let package = Package(
    name: "mlx-heartmula-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "HeartMuLaCore", targets: ["HeartMuLaCore"]),
        .library(name: "MLXHeartMuLa", targets: ["MLXHeartMuLa"]),
        .executable(name: "heartmula-gates", targets: ["heartmula-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.72.0"),
    ],
    targets: [
        .target(
            name: "HeartMuLaCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/HeartMuLaCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXHeartMuLa",
            dependencies: [
                "HeartMuLaCore",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXHeartMuLa",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "heartmula-gates",
            dependencies: [
                "HeartMuLaCore", "MLXHeartMuLa",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/heartmula-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "HeartMuLaCoreTests",
            dependencies: ["HeartMuLaCore"],
            path: "Tests/HeartMuLaCoreTests"
        ),
        .testTarget(
            name: "MLXHeartMuLaTests",
            dependencies: [
                "HeartMuLaCore", "MLXHeartMuLa",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXHeartMuLaTests"
        ),
    ]
)
