// SQCodec.swift — heartcodec/models/sq_codec.py `ScalarModel.decode` (the decoding half only; the converter drops the encoder):
//   round(9x)/9 (torch.round: half to even) → pre_conv (weight-normed Conv1d 128 → 2048, k5, symmetric padding 2 — the one
//   NON-causal conv) → 5 × ResDecoderBlock [up_conv: weight-normed ConvTranspose1d, k = 2·stride, causal (the last `stride`
//   samples trimmed), no activation; 5 × ResidualUnit (PReLU(conv1: k7, dilation 1/3/5/7/9, causal left pad 6·d) → PReLU(conv2:
//   k1) + x)] at 2048→1024 (×5), 1024→512 (×4), 512→256 (×4), 256→128 (×4), 128→64 (×3) → post (each frame repeated ×2,
//   causal Conv1d 64 → 64 k7, PReLU) → out_conv (weight-normed causal Conv1d 64 → 1, k7). 25 Hz latents → 48 kHz (×1920).
// Activations are (B, T, C) (MLX's conv layout); weights arrive folded and transposed by the converter (O, K, I).
// Module names are the converter's renames of `scalar_model.decoder.N` (Tools/convert/convert.py RENAMES).
import Foundation
import MLX
import MLXNN

/// torch `nn.PReLU()` (one shared slope).
public final class PReLU1: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    override init() { _weight.wrappedValue = MLXArray([Float(0.25)]) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { which(x .>= 0, x, weight * x) }
}

/// Conv1d with explicit causal (left) or symmetric padding; weight (O, K, I).
public final class PaddedConv1d: Module, UnaryLayer {
    let leftPad: Int, rightPad: Int, dilation: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ inC: Int, _ outC: Int, kernel: Int, dilation: Int = 1, causal: Bool) {
        self.dilation = dilation
        let total = dilation * (kernel - 1)
        if causal { leftPad = total; rightPad = 0 } else { leftPad = total / 2; rightPad = total / 2 }
        _weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        _bias.wrappedValue = MLXArray.zeros([outC])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xp = (leftPad + rightPad) > 0 ? padded(x, widths: [.init((0, 0)), .init((leftPad, rightPad)), .init((0, 0))]) : x
        return conv1d(xp, weight, stride: 1, padding: 0, dilation: dilation) + bias
    }
}

/// Causal ConvTranspose1d (kernel = 2·stride, padding 0, the trailing `stride` samples dropped); weight (O, K, I).
public final class CausalConvTranspose1d: Module, UnaryLayer {
    let stride: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int) {
        self.stride = stride
        _weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        _bias.wrappedValue = MLXArray.zeros([outC])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = convTransposed1d(x, weight, stride: stride, padding: 0) + bias
        return y[0..., ..<(y.dim(1) - stride), 0...]
    }
}

public final class ResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") public var conv1: PaddedConv1d
    @ModuleInfo(key: "conv2") var conv2: PaddedConv1d
    @ModuleInfo(key: "activation1") var act1: PReLU1
    @ModuleInfo(key: "activation2") var act2: PReLU1

    init(_ c: Int, dilation: Int, kernel: Int) {
        _conv1.wrappedValue = PaddedConv1d(c, c, kernel: kernel, dilation: dilation, causal: true)
        _conv2.wrappedValue = PaddedConv1d(c, c, kernel: 1, causal: true)
        _act1.wrappedValue = PReLU1()
        _act2.wrappedValue = PReLU1()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { act2(conv2(act1(conv1(x)))) + x }
}

public final class ResDecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "up_conv") public var upConv: CausalConvTranspose1d
    @ModuleInfo(key: "convs") public var convs: [ResidualUnit]

    init(_ inC: Int, _ outC: Int, stride: Int, kernel: Int, resKernel: Int) {
        _upConv.wrappedValue = CausalConvTranspose1d(inC, outC, kernel: kernel, stride: stride)
        _convs.wrappedValue = [1, 3, 5, 7, 9].map { ResidualUnit(outC, dilation: $0, kernel: resKernel) }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = upConv(x)
        for c in convs { h = c(h) }
        return h
    }
}

public final class PostProcessor: Module, UnaryLayer {
    let repeats: Int
    @ModuleInfo(key: "conv") var conv: PaddedConv1d
    @ModuleInfo(key: "activation") var activation: PReLU1

    init(_ c: Int, repeats: Int, kernel: Int) {
        self.repeats = repeats
        _conv.wrappedValue = PaddedConv1d(c, c, kernel: kernel, causal: true)
        _activation.wrappedValue = PReLU1()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let r = broadcast(x.expandedDimensions(axis: 2), to: [b, t, repeats, c]).reshaped(b, t * repeats, c)
        return activation(conv(r))
    }
}

public final class SQDecoder: Module {
    @ModuleInfo(key: "pre_conv") public var preConv: PaddedConv1d
    @ModuleInfo(key: "blocks") public var blocks: [ResDecoderBlock]
    @ModuleInfo(key: "post") var post: PostProcessor
    @ModuleInfo(key: "out_conv") var outConv: PaddedConv1d

    init(_ c: CodecConfig) {
        let n = c.upsample_factors.count
        _preConv.wrappedValue = PaddedConv1d(c.latent_hidden_dim, c.init_channel << n, kernel: c.delay_kernel_size, causal: false)
        _blocks.wrappedValue = (0 ..< n).map { i in
            ResDecoderBlock(c.init_channel << (n - i), c.init_channel << (n - i - 1), stride: c.upsample_factors[i],
                            kernel: c.upsample_kernel_sizes[i], resKernel: c.res_kernel_size)
        }
        _post.wrappedValue = PostProcessor(c.init_channel, repeats: c.num_samples, kernel: c.default_kernel_size)
        _outConv.wrappedValue = PaddedConv1d(c.init_channel, 1, kernel: c.default_kernel_size, causal: true)
    }

    /// latents (B, T, 128) → audio (B, T·1920), quantised to the 1/9 grid first.
    public func decode(_ latents: MLXArray) -> MLXArray {
        var x = round(9 * latents) / 9
        x = preConv(x)
        for blk in blocks { x = blk(x) }
        x = outConv(post(x))
        return x.squeezed(axis: -1)
    }

    /// The 1/9 grid index `round(9x)` (the D lane's near-tie diagnostic).
    public static func grid(_ latents: MLXArray) -> MLXArray { round(9 * latents) }
}

public final class HeartCodecModel: Module {
    public let config: CodecConfig
    @ModuleInfo(key: "flow_matching") public var flowMatching: FlowMatching
    @ModuleInfo(key: "scalar_model") public var scalarModel: SQDecoder

    public init(_ c: CodecConfig) {
        config = c
        _flowMatching.wrappedValue = FlowMatching(c)
        _scalarModel.wrappedValue = SQDecoder(c)
    }
}
