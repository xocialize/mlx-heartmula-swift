// HeartMuLaModel.swift — heartlib heartmula/modeling_heartmula.py `HeartMuLa`: the 3B backbone and the 300M local decoder
// (torchtune bodies, Llama.swift), `text_embeddings` (128 256), `audio_embeddings` (8 codebooks × 8197, offset k·8197),
// `unconditional_text_embedding` (1 row — the CFG row's text everywhere, and its MuQ slot), `muq_linear` (512 → 3072; the
// pipeline feeds zeros, so only its bias reaches the prompt), `projection` (3072 → 3072, backbone → decoder), `codebook0_head`
// (3072 → 8197) and `audio_head` (7, 3072, 8197) — one parameter, `decoder_h · audio_head[i − 1]` for codebook i.
// Keys are the checkpoint's own (the converter only casts to bf16).
import Foundation
import MLX
import MLXNN

public struct HeartMuLaConfig: Codable, Sendable {
    public var backbone_flavor: String = "llama-3B"
    public var decoder_flavor: String = "llama-300M"
    public var text_vocab_size: Int = 128_256
    public var audio_vocab_size: Int = 8197
    public var audio_num_codebooks: Int = 8
    public var muq_dim: Int = 512
}

public final class HeartMuLaModel: Module {
    public let config: HeartMuLaConfig
    @ModuleInfo(key: "backbone") public var backbone: LlamaTransformer
    @ModuleInfo(key: "decoder") public var decoder: LlamaTransformer
    @ModuleInfo(key: "text_embeddings") var textEmbeddings: Embedding
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "unconditional_text_embedding") var unconditionalTextEmbedding: Embedding
    @ModuleInfo(key: "projection") var projection: Linear
    @ModuleInfo(key: "codebook0_head") var codebook0Head: Linear
    @ParameterInfo(key: "audio_head") var audioHead: MLXArray
    @ModuleInfo(key: "muq_linear") var muqLinear: Linear

    public init(_ c: HeartMuLaConfig) throws {
        config = c
        let bf = try LlamaFlavor.named(c.backbone_flavor), df = try LlamaFlavor.named(c.decoder_flavor)
        _backbone.wrappedValue = LlamaTransformer(bf)
        _decoder.wrappedValue = LlamaTransformer(df)
        _textEmbeddings.wrappedValue = Embedding(embeddingCount: c.text_vocab_size, dimensions: bf.dim)
        _audioEmbeddings.wrappedValue = Embedding(embeddingCount: c.audio_vocab_size * c.audio_num_codebooks, dimensions: bf.dim)
        _unconditionalTextEmbedding.wrappedValue = Embedding(embeddingCount: 1, dimensions: bf.dim)
        _projection.wrappedValue = Linear(bf.dim, df.dim, bias: false)
        _codebook0Head.wrappedValue = Linear(bf.dim, c.audio_vocab_size, bias: false)
        _audioHead.wrappedValue = MLXArray.zeros([c.audio_num_codebooks - 1, df.dim, c.audio_vocab_size])
        _muqLinear.wrappedValue = Linear(c.muq_dim, bf.dim, bias: true)
    }

    public var codebooks: Int { config.audio_num_codebooks }
    public var dtype: DType { codebook0Head.weight.dtype }

    /// `generate_frame`'s prompt-call input (B = 2: conditional row, then the CFG row): the masked sum over the 9 columns is the
    /// text column alone (the audio columns are masked off), the CFG row's text is the unconditional embedding at every
    /// position, and position `muqIndex` is overwritten by `muq_linear(zeros)` / the unconditional embedding. → (2, L, D)
    public func promptEmbedding(_ p: HeartMuLaPrompt) -> MLXArray {
        let n = p.length
        let text = MLXArray((0 ..< n).map { p.tokens[$0 * 9 + 8] })
        let cond = textEmbeddings(text)                                           // (L, D)
        let muq = muqLinear(MLXArray.zeros([1, config.muq_dim], dtype: dtype))    // (1, D)
        let unc = unconditionalTextEmbedding(MLXArray([Int32(0)]))                 // (1, D)
        let isMuq = (MLXArray(Int32(0) ..< Int32(n)) .== Int32(p.muqIndex)).expandedDimensions(axis: 1)
        let row0 = which(isMuq, muq, cond)
        let row1 = broadcast(unc, to: [n, unc.dim(1)])
        return stacked([row0, row1], axis: 0)
    }

    /// The embedding of one or more audio frames: Σ_k audio_embeddings(code_k + k·8197); codes (..., 8) int → (..., D). Summed
    /// with an fp32 accumulator and rounded once, as torch's `sum(dim=2, dtype=bf16)` does on the bf16 tier.
    public func frameEmbedding(_ codes: MLXArray) -> MLXArray {
        let offsets = MLXArray((0 ..< codebooks).map { Int32($0 * config.audio_vocab_size) })
        let e = audioEmbeddings(codes.asType(.int32) + offsets)
        return e.asType(.float32).sum(axis: -2).asType(e.dtype)
    }

    /// `_embed_audio(codebook, tokens)`.
    public func embedAudio(_ codebook: Int, _ tokens: MLXArray) -> MLXArray {
        audioEmbeddings(tokens.asType(.int32) + Int32(codebook * config.audio_vocab_size))
    }

    public func c0Logits(_ lastHidden: MLXArray) -> MLXArray { codebook0Head(lastHidden) }

    /// The local decoder over `[last_h ; embeddings …]` (B, T, D) → post-norm hidden (B, T, D_dec).
    public func localDecoder(_ x: MLXArray, caches: [KVCache]?) -> MLXArray { decoder(projection(x), caches: caches) }

    /// `decoder_h[:, -1] · audio_head[i − 1]` for codebook i ∈ 1…7: (B, D_dec) → (B, 8197).
    public func audioLogits(_ h: MLXArray, codebook i: Int) -> MLXArray { matmul(h, audioHead[i - 1]) }
}
