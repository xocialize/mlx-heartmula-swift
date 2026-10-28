// BPETokenizer.swift — the Llama-3.2 byte-level BPE as Hugging Face `tokenizers` runs HeartMuLaGen's tokenizer.json: 128 000
// ranks + 256 added `<|…|>` tokens; no normaliser (no NFC); pre-tokeniser `Split` (isolated) with the regex read from the file
// (Llama-3's `\p{N}{1,3}` variant) then ByteLevel without its own regex; model BPE with `ignore_merges` (a pre-token that is a
// whole vocabulary entry is emitted as is); post-processor TemplateProcessing `<|begin_of_text|> $A`. Shaped after the fleet's
// Qwen2 tokenizer (mlx-minimax-music3-swift / mlx-ace-step-swift BPETokenizer.swift, MIT) with those three differences; gated
// id-exact on the 12 V0 prompts (lane T).
import Foundation

public final class BPETokenizer: @unchecked Sendable {
    public let vocabulary: [String: Int]
    private let byRank: [Int: String]
    private let mergeRanks: [String: Int]          // "a b" → rank
    public let addedTokens: [String: Int]
    private let addedByID: [Int: String]
    private let addedMaxLength: Int
    public let prependsBOS: Int?                   // TemplateProcessing's leading special token
    private let ignoreMerges: Bool
    private let pattern: NSRegularExpression
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()
    private static let byteToUnicode: [UInt8: Character] = makeByteToUnicode()
    private static let unicodeToByte: [Character: UInt8] = Dictionary(uniqueKeysWithValues: byteToUnicode.map { ($1, $0) })

    public init(file: URL) throws {
        let data = try ConfigIO.json(file)
        guard let model = data["model"] as? [String: Any], (model["type"] as? String) == "BPE",
              let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any] else {
            throw HeartMuLaError.badConfig("tokenizer.json: not a BPE model")
        }
        guard data["normalizer"] == nil || data["normalizer"] is NSNull else {
            throw HeartMuLaError.badConfig("tokenizer.json: a normaliser is configured; this tokenizer implements none")
        }
        vocabulary = vocab
        byRank = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        var ranks = [String: Int](); ranks.reserveCapacity(merges.count)
        for (i, m) in merges.enumerated() {
            if let pair = m as? [String], pair.count == 2 { ranks[pair[0] + " " + pair[1]] = i }
            else if let s = m as? String { ranks[s] = i }
        }
        mergeRanks = ranks
        ignoreMerges = (model["ignore_merges"] as? Bool) ?? false
        var added = [String: Int](), addedByID = [Int: String](), maxLen = 0
        for t in (data["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = t["content"] as? String, let id = t["id"] as? Int else { continue }
            added[content] = id; addedByID[id] = content
            maxLen = max(maxLen, content.count)
        }
        addedTokens = added; self.addedByID = addedByID; addedMaxLength = maxLen
        var bos: Int?
        if let post = data["post_processor"] as? [String: Any] {
            let processors = (post["processors"] as? [[String: Any]]) ?? [post]
            for p in processors where (p["type"] as? String) == "TemplateProcessing" {
                if let first = (p["single"] as? [[String: Any]])?.first, let special = first["SpecialToken"] as? [String: Any],
                   let id = special["id"] as? String { bos = added[id] }
            }
        }
        prependsBOS = bos
        var regex: String?
        if let pre = data["pre_tokenizer"] as? [String: Any] {
            let pres = (pre["pretokenizers"] as? [[String: Any]]) ?? [pre]
            for p in pres where (p["type"] as? String) == "Split" {
                regex = (p["pattern"] as? [String: Any])?["Regex"] as? String
            }
        }
        guard let regex else { throw HeartMuLaError.badConfig("tokenizer.json: no Split pre-tokeniser regex") }
        pattern = try NSRegularExpression(pattern: regex)
    }

    // MARK: encode

    /// `tokenizer.encode(text).ids`: added tokens split off, regex, byte-level BPE, BOS prepended by the template.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        var ids: [Int] = []
        if addSpecialTokens, let bos = prependsBOS { ids.append(bos) }
        for segment in splitAddedTokens(text) {
            switch segment {
            case .added(let id): ids.append(id)
            case .text(let s): ids += encodeOrdinary(s)
            }
        }
        return ids
    }

    private enum Segment { case added(Int), text(String) }

    /// Every added token is `<|…|>`; longest match at each `<|` position.
    private func splitAddedTokens(_ text: String) -> [Segment] {
        let chars = Array(text)
        var out: [Segment] = []
        var pending = ""
        var i = 0
        while i < chars.count {
            if chars[i] == "<", i + 1 < chars.count, chars[i + 1] == "|" {
                var matched: (Int, Int)? = nil
                var j = i + 2
                while j < chars.count, j - i < addedMaxLength {
                    if chars[j] == ">", let id = addedTokens[String(chars[i ... j])] { matched = (j - i + 1, id); break }
                    j += 1
                }
                if let (length, id) = matched {
                    if !pending.isEmpty { out.append(.text(pending)); pending = "" }
                    out.append(.added(id)); i += length; continue
                }
            }
            pending.append(chars[i]); i += 1
        }
        if !pending.isEmpty { out.append(.text(pending)) }
        return out
    }

    /// regex pieces → byte-level BPE (no normaliser).
    public func encodeOrdinary(_ text: String) -> [Int] {
        let ns = text as NSString
        var ids: [Int] = []
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            ids += bpe(ns.substring(with: match.range))
        }
        return ids
    }

    private func bpe(_ piece: String) -> [Int] {
        lock.lock(); if let hit = cache[piece] { lock.unlock(); return hit }; lock.unlock()
        let mapped = String(piece.utf8.map { Self.byteToUnicode[$0]! })
        if ignoreMerges, let id = vocabulary[mapped] {
            lock.lock(); cache[piece] = [id]; lock.unlock()
            return [id]
        }
        var word: [String] = mapped.map { String($0) }
        while word.count > 1 {
            var best = Int.max, at = -1
            for i in 0 ..< word.count - 1 {
                if let r = mergeRanks[word[i] + " " + word[i + 1]], r < best { best = r; at = i }
            }
            if at < 0 { break }
            let a = word[at], b = word[at + 1], merged = a + b
            var next: [String] = []; next.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == a, word[i + 1] == b { next.append(merged); i += 2 } else { next.append(word[i]); i += 1 }
            }
            word = next
        }
        let ids = word.map { vocabulary[$0] ?? -1 }
        lock.lock(); cache[piece] = ids; lock.unlock()
        return ids
    }

    // MARK: decode

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var out = ""
        var bytes: [UInt8] = []
        func flush() { if !bytes.isEmpty { out += String(decoding: bytes, as: UTF8.self); bytes.removeAll() } }
        for id in ids {
            if let s = addedByID[id] { flush(); if !skipSpecialTokens { out += s } }
            else if let tok = byRank[id] { for ch in tok { if let b = Self.unicodeToByte[ch] { bytes.append(b) } } }
        }
        flush()
        return out
    }

    private static func makeByteToUnicode() -> [UInt8: Character] {
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) { bs.append(b); cs.append(256 + n); n += 1 }
        var map = [UInt8: Character]()
        for (b, c) in zip(bs, cs) { map[UInt8(b)] = Character(UnicodeScalar(c)!) }
        return map
    }
}
