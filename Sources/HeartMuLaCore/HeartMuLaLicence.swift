// HeartMuLaLicence.swift — the offer's licence fields as the V0 gate verified them (AB-R-0628, WIP/heartmula-eval/licence/) and
// as the engine's allowlist classifies them (Apache-2.0: permissive). The engine package (V3) and any MES adapter read these;
// nothing else in the fleet restates them.
//
// Facts behind the strings: HeartMuLa/HeartMuLa-oss-3B-happy-new-year @ 41f6fc6, HeartMuLa/HeartCodec-oss-20260123 @ f889dab and
// HeartMuLa/HeartMuLaGen @ 2e18e01 carry `license: apache-2.0` on their model cards but no LICENSE file; heartlib's README
// (HeartMuLa/heartlib @ 5858ca8, 2026-01-20 entry) switches "all related model weights" to Apache-2.0 and ships the licence
// text; `tokenizer.json` is the Llama-3.2 tokenizer's vocabulary (128 256 entries) as HeartMuLaGen ships it.
public enum HeartMuLaLicence {
    public static let spdx = "Apache-2.0"
    public static let termsURL = "https://github.com/HeartMuLa/heartlib/blob/main/LICENSE"
    public static let commercialUse = true
    public static let acceptedRequired = false
    public static let namedOnly = false
    public static let attribution =
        "HeartMuLa-oss-3B and HeartCodec-oss by the HeartMuLa team (huggingface.co/HeartMuLa), Apache-2.0"
    /// The grant chain, informational.
    public static let notes =
        "Apache-2.0 weights. The generator's Hugging Face repos carry no LICENSE file: the grant is their model cards "
        + "(license: apache-2.0) and heartlib's README of 2026-01-20, which moves all related model weights to Apache-2.0; the "
        + "licence text and a NOTICE with this chain ship with every bundle. The text tokenizer is the Llama-3.2 tokenizer's "
        + "vocabulary as HeartMuLaGen ships it."
    /// The measured character (V0 AB-R-0628, the operator's listening pass AB-D-0187, V2b AB-R-0636), informational.
    public static let character =
        "Measured character: it follows the lyrics closely in English but steers weakly by style tags (CLAP style retrieval "
        + "rank about 4.5 against about 2 for the other music models; a higher CFG does not help) and most voices come out "
        + "female whatever the tags ask (1 of 6 male-voice prompts got a male voice). Mandarin is weak (lyric error rate 0.457 "
        + "against 0.107 for English). The 8-bit tier runs to the length cap twice as often as bf16 (10 of 24 songs against "
        + "5 of 24): the lyrics are sung, then choruses repeat, and one song in 24 looped a single line."
}
