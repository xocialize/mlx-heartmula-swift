// Llama.swift — torchtune 0.4.0's `llama3_2` TransformerDecoder as HeartMuLa uses it (tok_embeddings and output replaced by
// Identity: the body over input embeddings, returning the post-norm hidden). Facts kept from torchtune:
//   • RMSNorm normalises in fp32 and casts back BEFORE multiplying by `scale` (keys `sa_norm.scale`, `mlp_norm.scale`, `norm.scale`);
//   • RoPE is the INTERLEAVED form (pairs (x[2i], x[2i+1]); MLX `traditional: true`), applied on [b, s, h, d] in fp32 and cast back,
//     with Llama-3 frequency scaling (`Llama3ScaledRoPE.apply_scaling`: factor 32, low 1, high 4, old context 8192) — the θ table
//     is computed here exactly as torchtune does (fp32 `1 / base^(2i/d)`, then the per-frequency rule) and the angles as
//     `pos · θ` in fp32, so no reciprocal round trip (MLXFast's `freqs` path) touches them;
//   • GQA with kv-major head grouping (`expand(b, s, n_kv, q_per_kv, d)`), SDPA scale 1/√d, causal for a multi-token call,
//     no mask for a single cached step;
//   • SwiGLU `w2(silu(w1 x) · w3 x)`; residuals `h = x + attn(sa_norm x)`, `out = h + mlp(mlp_norm h)`.
import Foundation
import MLX
import MLXFast
import MLXNN

public struct LlamaFlavor: Sendable {
    public let layers: Int, heads: Int, kvHeads: Int, dim: Int, intermediate: Int, maxSeqLen: Int
    public let ropeBase: Float, scaleFactor: Float, normEps: Float
    public var headDim: Int { dim / heads }

    public static let llama3B = LlamaFlavor(layers: 28, heads: 24, kvHeads: 8, dim: 3072, intermediate: 8192, maxSeqLen: 8192,
                                            ropeBase: 500_000, scaleFactor: 32, normEps: 1e-5)
    public static let llama300M = LlamaFlavor(layers: 3, heads: 8, kvHeads: 4, dim: 3072, intermediate: 8192, maxSeqLen: 2048,
                                              ropeBase: 500_000, scaleFactor: 32, normEps: 1e-5)

    public static func named(_ s: String) throws -> LlamaFlavor {
        switch s {
        case "llama-3B": return .llama3B
        case "llama-300M": return .llama300M
        default: throw HeartMuLaError.badConfig("unsupported flavour \(s)")
        }
    }

    /// torchtune `Llama3ScaledRoPE.rope_init` θ (dim/2,), fp32, low 1 / high 4 / old context 8192 — torch's arithmetic step for
    /// step: `base ** (arange/d)` rounded to fp32 then `1.0 / ·`; inside `apply_scaling` a Python scalar divided by a 0-d tensor is
    /// `Tensor.__rtruediv__` = `reciprocal() * scalar`, so `2π / freq` and `old_context / wavelen` are reciprocal-then-multiply in fp32.
    public func ropeTheta() -> [Float] {
        let d = headDim
        let lowFreqWavelen: Float = 8192 / 1, highFreqWavelen: Float = 8192 / 4
        let twoPi = Float(2 * Double.pi)
        var out = [Float]()
        for i in stride(from: 0, to: d, by: 2) {
            let freq: Float = 1 / Float(pow(Double(ropeBase), Double(Float(i) / Float(d))))
            let wavelen = (1 / freq) * twoPi
            if wavelen < highFreqWavelen { out.append(freq) }
            else if wavelen > lowFreqWavelen { out.append(freq / scaleFactor) }
            else {
                let smooth = ((1 / wavelen) * 8192 - 1) / (4 - 1)
                out.append((1 - smooth) * freq / scaleFactor + smooth * freq)
            }
        }
        return out
    }
}

/// A non-parameter constant held outside the Module tree (an MLXArray property of a Module would join its parameters and
/// break the key contract).
public final class Constant: @unchecked Sendable {
    public let value: MLXArray
    public init(_ value: MLXArray) { self.value = value }
}

/// Append-only K/V store for one layer, grown in 512-slot steps (the mlx-swift-lm idiom).
public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var offset = 0
    let step: Int

    public init(step: Int = 512) { self.step = step }

    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let t = k.dim(2)
        let prev = offset
        if keys == nil || prev + t > keys!.dim(2) {
            let needed = ((prev + t + step - 1) / step) * step
            let shape = [k.dim(0), k.dim(1), needed, k.dim(3)]
            let nk = MLXArray.zeros(shape, dtype: k.dtype), nv = MLXArray.zeros(shape, dtype: v.dtype)
            if let ok = keys, let ov = values, prev > 0 {
                nk[0..., 0..., ..<prev, 0...] = ok[0..., 0..., ..<prev, 0...]
                nv[0..., 0..., ..<prev, 0...] = ov[0..., 0..., ..<prev, 0...]
            }
            keys = nk; values = nv
        }
        keys![0..., 0..., prev ..< (prev + t), 0...] = k
        values![0..., 0..., prev ..< (prev + t), 0...] = v
        offset = prev + t
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...])
    }
}

/// torchtune RMSNorm: `(x.float() * rsqrt(mean(x²) + eps)).type_as(x) * scale`.
public final class TTRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(dim: Int, eps: Float) {
        self.eps = eps
        _scale.wrappedValue = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let n = (xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)).asType(x.dtype)
        return n * scale
    }
}

/// Interleaved RoPE on (B, L, H, D) at integer positions `start ..< start + L`, fp32 math, cast back to x's dtype.
public func interleavedRoPE(_ x: MLXArray, theta: MLXArray, start: Int) -> MLXArray {
    let (b, l, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let pos = MLXArray(Int32(start) ..< Int32(start + l)).asType(.float32)
    let ang = pos.expandedDimensions(axis: 1) * theta.expandedDimensions(axis: 0)          // (L, D/2)
    let c = cos(ang).reshaped(1, l, 1, d / 2), s = sin(ang).reshaped(1, l, 1, d / 2)
    let xf = x.asType(.float32).reshaped(b, l, h, d / 2, 2)
    let x0 = xf[.ellipsis, 0], x1 = xf[.ellipsis, 1]
    return stacked([x0 * c - x1 * s, x1 * c + x0 * s], axis: -1).reshaped(b, l, h, d).asType(x.dtype)
}

public final class LlamaAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    let theta: Constant
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    init(_ f: LlamaFlavor, theta: Constant) {
        heads = f.heads; kvHeads = f.kvHeads; headDim = f.headDim; scale = 1 / Float(f.headDim).squareRoot()
        self.theta = theta
        _qProj.wrappedValue = Linear(f.dim, f.heads * f.headDim, bias: false)
        _kProj.wrappedValue = Linear(f.dim, f.kvHeads * f.headDim, bias: false)
        _vProj.wrappedValue = Linear(f.dim, f.kvHeads * f.headDim, bias: false)
        _outputProj.wrappedValue = Linear(f.dim, f.dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let start = cache?.offset ?? 0
        let q = interleavedRoPE(qProj(x).reshaped(b, l, heads, headDim), theta: theta.value, start: start).transposed(0, 2, 1, 3)
        var k = interleavedRoPE(kProj(x).reshaped(b, l, kvHeads, headDim), theta: theta.value, start: start).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        if let cache { (k, v) = cache.update(k, v) }
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: l > 1 ? .causal : .none)
        return outputProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

public final class LlamaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hidden: Int) {
        _w1.wrappedValue = Linear(dim, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hidden, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

public final class LlamaLayer: Module {
    @ModuleInfo(key: "sa_norm") var saNorm: TTRMSNorm
    @ModuleInfo(key: "attn") var attn: LlamaAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: TTRMSNorm
    @ModuleInfo(key: "mlp") var mlp: LlamaMLP

    init(_ f: LlamaFlavor, theta: Constant) {
        _saNorm.wrappedValue = TTRMSNorm(dim: f.dim, eps: f.normEps)
        _attn.wrappedValue = LlamaAttention(f, theta: theta)
        _mlpNorm.wrappedValue = TTRMSNorm(dim: f.dim, eps: f.normEps)
        _mlp.wrappedValue = LlamaMLP(dim: f.dim, hidden: f.intermediate)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let h = x + attn(saNorm(x), cache: cache)
        return h + mlp(mlpNorm(h))
    }
}

/// The body: layers + final norm over input embeddings (B, L, D) → post-norm hidden (B, L, D).
public final class LlamaTransformer: Module {
    public let flavor: LlamaFlavor
    /// The θ table (dim/2,) fp32 — held outside the parameter tree.
    public let theta: Constant
    @ModuleInfo(key: "layers") var layers: [LlamaLayer]
    @ModuleInfo(key: "norm") var norm: TTRMSNorm

    init(_ f: LlamaFlavor) {
        flavor = f
        let t = Constant(MLXArray(f.ropeTheta()))
        theta = t
        _layers.wrappedValue = (0 ..< f.layers).map { _ in LlamaLayer(f, theta: t) }
        _norm.wrappedValue = TTRMSNorm(dim: f.dim, eps: f.normEps)
    }

    public func newCaches(step: Int = 512) -> [KVCache] { layers.map { _ in KVCache(step: step) } }

    public func callAsFunction(_ h: MLXArray, caches: [KVCache]?) -> MLXArray {
        var x = h
        for (i, layer) in layers.enumerated() { x = layer(x, cache: caches?[i]) }
        return norm(x)
    }
}
