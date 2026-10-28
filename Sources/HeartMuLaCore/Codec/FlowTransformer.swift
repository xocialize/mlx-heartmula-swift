// FlowTransformer.swift — heartcodec/models/transformer.py `LlamaTransformer` (the flow-matching estimator, adaLN-single):
//   proj_in (Conv1d 1024 → 1536, k3 p1, × 3^−½, Linear) → 24 blocks at 1536 (24 heads × 64) → norm_out (LayerNorm, no affine,
//   eps 1e-6) · (1 + scale) + shift from `scale_shift_table + embedded_timestep` → concat [input ; s] → connection_proj
//   (Conv1d 2560 → 3072, k3) → 6 blocks at 3072 (24 heads × 128) → norm_out_2 · (1 + scale_2) + shift_2 → proj_out (Conv1d
//   3072 → 256, k3, Linear).
// A block: RMSNorm (eps 1e-6, `weight · x · rsqrt(mean(x²) + eps)`), modulated by its `scale_shift_table` (6, D) + the
// timestep's 6·D vector (shift/scale/gate for attention and MLP); full (non-causal) self-attention with INTERLEAVED RoPE
// (base 10 000 over the whole head, positions 0…T−1, angles `t · inv_freq` in the activations' dtype); SwiGLU MLP (hidden
// 4096 / 8192: Llama's `2·4d/3` rounded up to 256).
// Timestep: `t · 1000 · exp(−ln(10⁴)·i/256)` → [cos ; sin] (512) → linear_1 → SiLU → linear_2 = embedded_timestep;
// adaLN modulation = linear(SiLU(embedded_timestep)) (6·D). Keys are the checkpoint's (`flow_matching.estimator.…`).
import Foundation
import MLX
import MLXFast
import MLXNN

/// heartcodec RMSNorm: `weight * (x * rsqrt(mean(x²) + eps))`.
public final class CodecRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(dim: Int, eps: Float = 1e-6) { self.eps = eps; _weight.wrappedValue = MLXArray.ones([dim]) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight * (x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps))
    }
}

/// `RotaryEmbedding` + `apply_rope_vec`: interleaved pairs, inv_freq = 1 / base^(2i/d) and angles in x's dtype.
func codecRoPE(_ x: MLXArray, invFreq: MLXArray) -> MLXArray {
    // x (B, H, T, D)
    let (b, h, t, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let pos = MLXArray(Int32(0) ..< Int32(t)).asType(x.dtype)
    let ang = pos.expandedDimensions(axis: 1) * invFreq.asType(x.dtype).expandedDimensions(axis: 0)   // (T, D/2)
    let s = sin(ang).reshaped(1, 1, t, d / 2), c = cos(ang).reshaped(1, 1, t, d / 2)
    let xp = x.reshaped(b, h, t, d / 2, 2)
    let x1 = xp[.ellipsis, 0], x2 = xp[.ellipsis, 1]
    return stacked([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1).reshaped(b, h, t, d)
}

public final class CodecAttention: Module {
    let heads: Int, headDim: Int
    let invFreq: Constant
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads; self.headDim = headDim
        // torch: 1.0 / (base ** (arange(0, d, 2, dtype) / d)) in fp32
        invFreq = Constant(MLXArray(stride(from: 0, to: headDim, by: 2).map { i -> Float in
            1 / Float(pow(10_000.0, Double(Float(i) / Float(headDim))))
        }))
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = codecRoPE(qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3), invFreq: invFreq.value)
        let k = codecRoPE(kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3), invFreq: invFreq.value)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim))
    }
}

public final class CodecMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "up") var up: Linear
    @ModuleInfo(key: "down") var down: Linear
    init(dim: Int) {
        var hidden = Int(2 * (4 * dim) / 3)
        hidden = 256 * ((hidden + 255) / 256)
        _gate.wrappedValue = Linear(dim, hidden, bias: false)
        _up.wrappedValue = Linear(dim, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

public final class CodecBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: CodecRMSNorm
    @ModuleInfo(key: "attn") var attn: CodecAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: CodecRMSNorm
    @ModuleInfo(key: "mlp") var mlp: CodecMLP
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(dim: Int, heads: Int, headDim: Int) {
        _attnNorm.wrappedValue = CodecRMSNorm(dim: dim)
        _attn.wrappedValue = CodecAttention(dim: dim, heads: heads, headDim: headDim)
        _mlpNorm.wrappedValue = CodecRMSNorm(dim: dim)
        _mlp.wrappedValue = CodecMLP(dim: dim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([6, dim])
    }

    /// x (B, T, D); timestep (B, 6·D).
    func callAsFunction(_ x: MLXArray, timestep: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let mod = scaleShiftTable.expandedDimensions(axis: 0) + timestep.reshaped(b, 6, -1)     // (B, 6, D)
        let parts = split(mod, parts: 6, axis: 1)                                                 // each (B, 1, D)
        let (shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP) = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
        var h = attnNorm(x) * (1 + scaleMSA) + shiftMSA
        var y = x + gateMSA * attn(h)
        h = mlpNorm(y) * (1 + scaleMLP) + shiftMLP
        y = y + gateMLP * mlp(h)
        return y
    }
}

/// `ProjectLayer`: Conv1d (k3, padding 1) on (B, T, C) × k^−½, then Linear.
public final class ProjectLayer: Module {
    let kernel: Int
    @ModuleInfo(key: "ffn_1") var ffn1: Conv1d
    @ModuleInfo(key: "ffn_2") var ffn2: Linear
    init(_ inC: Int, _ outC: Int, kernel: Int = 3) {
        self.kernel = kernel
        _ffn1.wrappedValue = Conv1d(inputChannels: inC, outputChannels: outC, kernelSize: kernel, padding: kernel / 2)
        _ffn2.wrappedValue = Linear(outC, outC)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ffn2(ffn1(x) * Float(pow(Double(kernel), -0.5)))
    }
}

public final class TimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ inC: Int, _ dim: Int) {
        _linear1.wrappedValue = Linear(inC, dim)
        _linear2.wrappedValue = Linear(dim, dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class FlowEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedder
    init(_ dim: Int) { _timestepEmbedder.wrappedValue = TimestepEmbedder(512, dim) }

    /// `timestep_embedding`: t (B,) → (B, 512) = [cos ; sin] of t · 1000 · exp(−ln(10⁴) · arange(256) / 256).
    public static func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = 256
        let freqs = exp(-Float(log(10_000.0)) * MLXArray(Int32(0) ..< Int32(half)).asType(.float32) / Float(half)).asType(t.dtype)
        let args = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0) * Float(1000)
        return concatenated([cos(args), sin(args)], axis: -1)
    }
}

public final class AdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") var emb: FlowEmbeddings
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ dim: Int) {
        _emb.wrappedValue = FlowEmbeddings(dim)
        _linear.wrappedValue = Linear(dim, 6 * dim)
    }
    /// → (modulation (B, 6·D), embedded_timestep (B, D))
    func callAsFunction(_ t: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let e = emb.timestepEmbedder(FlowEmbeddings.sinusoid(t).asType(dtype))
        return (linear(silu(e)), e)
    }
}

public struct CodecConfig: Codable, Sendable {
    public var dim: Int = 512
    public var codebook_size: Int = 8192
    public var codebook_dim: Int = 32
    public var num_quantizers: Int = 8
    public var attention_head_dim: Int = 64
    public var in_channels: Int = 1024
    public var num_attention_heads: Int = 24
    public var num_layers: Int = 24
    public var num_layers_2: Int = 6
    public var out_channels: Int = 256
    public var sample_rate: Int = 48_000
    public var num_samples: Int = 2
    public var upsample_factors: [Int] = [5, 4, 4, 4, 3]
    public var upsample_kernel_sizes: [Int] = [10, 8, 8, 8, 6]
    public var latent_hidden_dim: Int = 128
    public var default_kernel_size: Int = 7
    public var delay_kernel_size: Int = 5
    public var init_channel: Int = 64
    public var res_kernel_size: Int = 7
}

public final class FlowTransformer: Module {
    @ModuleInfo(key: "proj_in") var projIn: ProjectLayer
    @ModuleInfo(key: "transformer_blocks") var blocks: [CodecBlock]
    @ModuleInfo(key: "transformer_blocks_2") var blocks2: [CodecBlock]
    @ModuleInfo(key: "connection_proj") var connectionProj: ProjectLayer
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ParameterInfo(key: "scale_shift_table_2") var scaleShiftTable2: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: ProjectLayer
    @ModuleInfo(key: "adaln_single") var adaln: AdaLayerNormSingle
    @ModuleInfo(key: "adaln_single_2") var adaln2: AdaLayerNormSingle

    init(_ c: CodecConfig) {
        let inner = c.num_attention_heads * c.attention_head_dim, inner2 = inner * 2
        _projIn.wrappedValue = ProjectLayer(c.in_channels, inner)
        _blocks.wrappedValue = (0 ..< c.num_layers).map { _ in CodecBlock(dim: inner, heads: c.num_attention_heads, headDim: c.attention_head_dim) }
        _blocks2.wrappedValue = (0 ..< c.num_layers_2).map { _ in CodecBlock(dim: inner2, heads: c.num_attention_heads, headDim: c.attention_head_dim * 2) }
        _connectionProj.wrappedValue = ProjectLayer(c.in_channels + inner, inner2)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([2, inner])
        _scaleShiftTable2.wrappedValue = MLXArray.zeros([2, inner2])
        _projOut.wrappedValue = ProjectLayer(inner2, c.out_channels)
        _adaln.wrappedValue = AdaLayerNormSingle(inner)
        _adaln2.wrappedValue = AdaLayerNormSingle(inner2)
    }

    static func layerNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
    }

    /// hidden (B, T, 1024) = [x ; incontext_x ; mu], t (B,) → velocity (B, T, 256).
    public func callAsFunction(_ hidden: MLXArray, timestep t: MLXArray) -> MLXArray {
        var s = projIn(hidden)
        let (mod1, emb1) = adaln(t, dtype: s.dtype)
        for blk in blocks { s = blk(s, timestep: mod1) }
        let ss = split(scaleShiftTable.expandedDimensions(axis: 0) + emb1.expandedDimensions(axis: 1), parts: 2, axis: 1)
        s = Self.layerNorm(s) * (1 + ss[1]) + ss[0]
        var x = connectionProj(concatenated([hidden, s], axis: -1))
        let (mod2, emb2) = adaln2(t, dtype: x.dtype)
        for blk in blocks2 { x = blk(x, timestep: mod2) }
        let ss2 = split(scaleShiftTable2.expandedDimensions(axis: 0) + emb2.expandedDimensions(axis: 1), parts: 2, axis: 1)
        x = Self.layerNorm(x) * (1 + ss2[1]) + ss2[0]
        return projOut(x)
    }
}
