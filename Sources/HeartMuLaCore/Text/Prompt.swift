// Prompt.swift — `HeartMuLaGenPipeline.preprocess` (heartlib pipelines/music_generation.py): tags lower-cased and wrapped in
// `<tag>…</tag>` (unless already), lyrics lower-cased; each encoded with BOS (the template) and EOS appended when missing
// (gen_config: bos 128000, eos 128001); the prompt is [tags ids ; one MuQ slot ; lyrics ids], 9 columns per position (8 audio
// codebooks, then text), only the text column live in the mask. The MuQ slot holds token 0 in the matrix; its embedding is
// replaced by `muq_linear(zeros(512))` (the pipeline never loads MuQ-MuLan) — see HeartMuLaModel.promptEmbedding.
import Foundation

public struct GenConfig: Codable, Sendable {
    public var text_bos_id: Int = 128_000
    public var text_eos_id: Int = 128_001
    public var audio_eos_id: Int = 8193
    public var empty_id: Int = 0
    public init() {}
}

public struct HeartMuLaPrompt: Sendable {
    public let tagsIDs: [Int]
    public let lyricsIDs: [Int]
    /// (L, 9) row-major: columns 0…7 audio (all `empty_id`), column 8 text.
    public let tokens: [Int32]
    /// (L, 9) row-major; true only in the text column.
    public let mask: [Bool]
    public let muqIndex: Int
    public var length: Int { tokens.count / 9 }
    public static let columns = 9
}

public enum PromptBuilder {
    public static func build(tags: String, lyrics: String, tokenizer: BPETokenizer, config: GenConfig = GenConfig()) -> HeartMuLaPrompt {
        var t = tags.lowercased()
        if !t.hasPrefix("<tag>") { t = "<tag>" + t }
        if !t.hasSuffix("</tag>") { t += "</tag>" }
        var tagsIDs = tokenizer.encode(t)
        if tagsIDs.first != config.text_bos_id { tagsIDs.insert(config.text_bos_id, at: 0) }
        if tagsIDs.last != config.text_eos_id { tagsIDs.append(config.text_eos_id) }
        var lyricsIDs = tokenizer.encode(lyrics.lowercased())
        if lyricsIDs.first != config.text_bos_id { lyricsIDs.insert(config.text_bos_id, at: 0) }
        if lyricsIDs.last != config.text_eos_id { lyricsIDs.append(config.text_eos_id) }
        let n = tagsIDs.count + 1 + lyricsIDs.count
        var tokens = [Int32](repeating: 0, count: n * 9)
        var mask = [Bool](repeating: false, count: n * 9)
        for (i, id) in tagsIDs.enumerated() { tokens[i * 9 + 8] = Int32(id) }
        for (i, id) in lyricsIDs.enumerated() { tokens[(tagsIDs.count + 1 + i) * 9 + 8] = Int32(id) }
        for i in 0 ..< n { mask[i * 9 + 8] = true }
        return HeartMuLaPrompt(tagsIDs: tagsIDs, lyricsIDs: lyricsIDs, tokens: tokens, mask: mask, muqIndex: tagsIDs.count)
    }
}
