// TF32.swift — the process-wide TF32 switch the fp32 codec depends on (AB-L-0315). MLX reads `MLX_ENABLE_TF32` once, on the
// first matmul of the process (`static` in `env::enable_tf32()`), and on Apple's neural accelerators (NAX) TF32 routes fp32
// matmuls through a 10-bit-mantissa path: ~1e-3 drift, which SQ-Codec's `round(9x)/9` turns into 25–30 dB end to end. So a
// host disables it before its first MLX op (`disableForProcess()`), and a loader checks what actually latched (`probe()`).
import Darwin
import Foundation
import MLX
import MLXRandom

public enum TF32 {
    /// Sets `MLX_ENABLE_TF32=0`. Effective only if no matmul has run in this process yet.
    public static func disableForProcess() { setenv("MLX_ENABLE_TF32", "0", 1) }

    /// The environment as set (not necessarily what latched).
    public static var environmentDisables: Bool { ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" }

    /// relL2 of a 256³ fp32 matmul on the GPU against the CPU stream (which never uses TF32): ~1e-7 with TF32 off, ~1e-4 with
    /// it latched on. Fixed key, so the reading is repeatable and the global RNG is untouched.
    public static func probeRelL2() -> Double {
        let k = MLXRandom.split(key: MLXRandom.key(0x7F32), into: 2)
        let a = MLXRandom.normal([256, 256], dtype: .float32, key: k[0])
        let b = MLXRandom.normal([256, 256], dtype: .float32, key: k[1])
        let gpu = matmul(a, b, stream: .gpu), cpu = matmul(a, b, stream: .cpu)
        let d = gpu - cpu
        let num = sqrt(sum(d * d)).item(Float.self), den = sqrt(sum(cpu * cpu)).item(Float.self)
        return Double(num) / Double(max(den, 1e-30))
    }

    /// The pre-registered threshold (V3 T1): TF32 is off when the probe reads below 1e-5.
    public static let offThreshold = 1e-5

    /// What latched: true when fp32 matmuls are exact (TF32 off).
    public static func isOff() -> Bool { probeRelL2() < offThreshold }
}
