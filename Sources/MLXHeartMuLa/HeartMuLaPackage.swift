import Foundation
import HeartMuLaCore
import MLX
import MLXToolKit

/// HeartMuLa-oss-3B (the HeartMuLa team, Apache-2.0 weights) on the canonical `music` surface (contract ≥ 1.56.0): style tags +
/// lyrics with section tags (+ a length cap, a seed) → one 48 kHz stereo `.wav` song and whether the cap cut it short. A
/// permissive, NON-DEFAULT music offer beside ACE-Step 1.5 (E27, AB-T-0376; listening pass GO with its character declared,
/// AB-D-0187): the host keeps ACE-Step as the capability's default (`MLXServeEngine.setDefault`) and offers this package by name.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `HeartMuLaConfiguration`, materializes the two declared bundles into
/// its store (or the host names them), pages them in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// What the surface declares (`MusicControls`): the core only — caption (read as comma-separated tags, the model's only style
/// input), lyrics (`[verse]` / `[chorus]` / … inline), seed, and `durationSeconds` as a CAP (nil → 120 s, at most 240 s): the song
/// ends on its own end token when the lyrics run out; the audio's length is the effective duration and `truncated` says when the
/// cap, not the song, ended it. NOTHING else is honoured, so nothing else is declared and the engine refuses it before admission
/// (C11): no `.referenceAudio` / `.sourceAudio` (upstream's MuQ reference slot is a zero vector), no `.planning` / `.score` /
/// `.anchors` (no planner), no `.vocalLanguage` (no language slot — it sings the lyrics' script), no `.timestamps`. An undeclared
/// `guidanceScale` is ignored, as the contract says (CFG 1.5 on the LM and 1.25 on the codec are the recipe).
///
/// TF32: the fp32 codec needs `MLX_ENABLE_TF32=0` for the whole process (AB-L-0315), and MLX latches that on the first matmul.
/// A host calls `HeartMuLaPackage.configureProcess()` at engine start; `load()` calls it too and then probes what latched.
///
/// Sampling: the LM SAMPLES its codes (top-k 50 after CFG) from MLX's generator seeded with `seed`, and the codec's noise from
/// an independent seed derived from it, so a seed reproduces a song within Swift-MLX — never across bindings (AB-L-0090).
@InferenceActor
public final class HeartMuLaPackage: ModelPackage {
    public typealias Configuration = HeartMuLaConfiguration

    /// Footprints MEASURED as phys_footprint (M5 Max, 2026-10-28, MLX cache 1 GiB, pool-inclusive as the fleet declares —
    /// AB-L-0113 / AB-L-0155): resident = phys after `prepare` through MLXServeEngine (the `validate` lane, the larger of two
    /// runs), activation = the largest of the validate song's peak, the 240 s `envelope` lane's peak and V2b's process peak for
    /// that pairing, minus the resident (WIP/heartmula-port/MEASUREMENTS.md §V3). Declared rounded up to 0.5 GB:
    ///   int8 + fp16 (default) 8.32 GB resident, peak 12.82 (song) / 12.80 (envelope) / 13.77 (V2b 12 songs) → 8.5 + 5.5;
    ///   bf16 + fp16 11.26, peak 15.78 / 15.68 / 16.79 (V2b c16 decode process) → 11.5 + 6.0;
    ///   int8 + fp32 11.47, peak 19.60 / 18.40 / 21.03 → 11.5 + 10.0 (hint);
    ///   bf16 + fp32 14.40, peak 21.22 / 21.61 / 24.03 → 14.5 + 10.0 (hint).
    /// By the engine's 0.7 × RAM admission: the default is a 24 GB-Mac tier (14.0 ≤ 16.8), bf16 + fp16 and int8 + fp32 32 GB
    /// (17.5 / 21.5 ≤ 22.4), bf16 + fp32 48 GB (24.5).
    nonisolated static let int8ResidentBytes: UInt64 = 8_500_000_000
    nonisolated static let int8PeakActivationBytes: UInt64 = 5_500_000_000
    nonisolated static let bf16ResidentBytes: UInt64 = 11_500_000_000
    nonisolated static let bf16PeakActivationBytes: UInt64 = 6_000_000_000
    nonisolated static let int8FP32ResidentBytes: UInt64 = 11_500_000_000
    nonisolated static let int8FP32PeakActivationBytes: UInt64 = 10_000_000_000
    nonisolated static let bf16FP32ResidentBytes: UInt64 = 14_500_000_000
    nonisolated static let bf16FP32PeakActivationBytes: UInt64 = 10_000_000_000

    /// heartlib's documented `--max_audio_length_ms` default (240 000); the backbone's 8 192 positions hold it with the prompt.
    public nonisolated static let maxDurationSeconds: Double = 240
    /// The cap when a request names none (heartlib's pipeline default, the V0 / V2b length; most songs end before it).
    public nonisolated static let defaultDurationSeconds: Double = 120
    /// The shortest cap taken (a few frames would be noise, not a song).
    public nonisolated static let minDurationSeconds: Double = 1
    public nonisolated static let sampleRate = 48_000
    public nonisolated static let channels = 2

    public nonisolated static var controls: MusicControls {
        MusicControls(features: [], maxDurationSeconds: maxDurationSeconds)
    }

    /// The offer fields a media-engine adapter reads — not expressible on `PackageManifest`.
    public nonisolated static let namedOnly = HeartMuLaLicence.namedOnly
    public nonisolated static let acceptedRequired = HeartMuLaLicence.acceptedRequired
    public nonisolated static let commercialUse = HeartMuLaLicence.commercialUse

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: Apache-2.0 weights by the model cards + heartlib's README (no LICENSE file in the generator repos — the grant
            // chain is the informational notice and each bundle's NOTICE). C8: port code MIT (heartlib is Apache-2.0 —
            // THIRD_PARTY_NOTICES.md).
            license: LicenseDeclaration(
                weightLicense: .apache2, portCodeLicense: .mit,
                notices: [
                    LicenseNotice(text: HeartMuLaLicence.notes, display: .informational, license: .apache2,
                                  termsURL: HeartMuLaLicence.termsURL),
                    LicenseNotice(text: HeartMuLaLicence.character, display: .informational),
                    LicenseNotice(text: "The length you ask for is a cap: the song ends on its own when the lyrics run out (most do, at "
                                  + "1½–2 minutes). With the default tier a song renders about 3× faster than real time on an M5 Max (a 70 s song in 23 s).",
                                  display: .informational),
                ]),
            provenance: Provenance(sourceRepo: HeartMuLaConfiguration.upstreamModelRepo,
                                   revision: HeartMuLaConfiguration.upstreamModelRevision, tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    // the fp16-codec pairings; `.int8` = the pre-quantised 8-bit LM (the default tier)
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes, peakActivationBytes: bf16PeakActivationBytes),
                    QuantFootprint(quant: .int8, residentBytes: int8ResidentBytes, peakActivationBytes: int8PeakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                MusicContract.descriptor(
                    name: "heartmula-oss-3b",
                    summary: "HeartMuLa-oss-3B (Apache-2.0) song generator (.wav, 48 kHz stereo, up to 240 s): comma-separated style tags as "
                        + "the caption (genre, mood, instruments, voice) + lyrics with [verse] / [chorus] / [bridge] tags → a full song that "
                        + "sings the written lyrics in order (English lyric error rate 0.11–0.14 on the fleet's 12-song set; Mandarin weak, "
                        + "0.46). The tags steer it weakly (genre and voice gender often drift; most voices come out female). durationSeconds "
                        + "is a cap — the song ends on its own end token when the lyrics run out, and `truncated` says when the cap cut it. "
                        + "No language, anchor, reference or edit controls.",
                    controls: Self.controls
                )
            ]
        )
    }

    /// The LM tiers this package offers — V2b's reading (WIP/heartmula-port/MEASUREMENTS.md §V2b): `.int8` (rule Q8 PASS, pooled
    /// PER 0.132, the default with the fp16 codec) and `.bf16` (rule B PASS, 0.125). The 4-bit LM failed Q4 (0.220) and has no
    /// bundle here; any other quant is refused by `load()` with the reason.
    public nonisolated static let offeredQuants: [Quant] = [.int8, .bf16]
    /// Both codec precisions are offered (C16 PASS).
    public nonisolated static let offeredCodecs: [HeartMuLaCodecPrecision] = [.fp16, .fp32]
    /// The codecs that run unchanged when TF32 latched on in the host process — V3 T2: the default tier's song (p04, seed 1004)
    /// is byte-identical, codes and audio, with TF32 on and off. The fp32 codec is not among them (AB-L-0315): `load()` refuses it
    /// when the probe finds TF32 on.
    public nonisolated static let tf32ImmuneCodecs: Set<HeartMuLaCodecPrecision> = [.fp16]

    /// Disables TF32 for this process (`MLX_ENABLE_TF32=0`). A host calls it at engine start, before ANY model runs: MLX latches
    /// the setting on the process's first matmul, and the fp32 codec needs it off.
    public nonisolated static func configureProcess() { TF32.disableForProcess() }

    private let configuration: Configuration
    private var pipeline: HeartMuLaPipeline?
    /// What the load-time probe read (relL2 of a GPU vs CPU fp32 matmul); nil until loaded.
    public private(set) var tf32ProbeRelL2: Double?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard pipeline == nil else { return }
        guard Self.offeredQuants.contains(configuration.quant) else {
            throw HeartMuLaPackageError.unsupportedTier(
                "LM quant \(configuration.quant) — this package offers \(Self.offeredQuants.map { "\($0)" }.joined(separator: " / ")) "
                + "(the 4-bit LM failed its lyric rule at V2b)")
        }
        // before this package's first MLX op: effective only if the host has not run a matmul yet (configureProcess)
        Self.configureProcess()
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw HeartMuLaPackageError.missingWeights(
                "source not materialized: \(missing.map(\.repo).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let bundles = configuration.resolvedBundles(storeRoot: storeRoot) else {
            throw HeartMuLaPackageError.missingWeights("unresolved bundle directories (no store root)")
        }
        let probe = TF32.probeRelL2()
        tf32ProbeRelL2 = probe
        if probe >= TF32.offThreshold, !Self.tf32ImmuneCodecs.contains(configuration.codec) {
            throw HeartMuLaPackageError.tf32Enabled(String(
                format: "the %@ codec needs TF32 off for the whole process, and it latched on (probe relL2 %.1e ≥ %.0e) — call "
                    + "HeartMuLaPackage.configureProcess() at engine start, before any model runs, or choose the fp16 codec",
                configuration.codec.rawValue, probe, TF32.offThreshold))
        }
        MLX.Memory.cacheLimit = Int(configuration.cacheLimitBytes)
        // Weight loads ride the CPU stream inside the Core (WeightIO.load) — the Metal-watchdog rule; the 8-bit LM is a
        // pre-quantised bundle, so no bf16 copy is ever resident.
        do {
            pipeline = try HeartMuLaPipeline(lmDir: bundles.lm, codecDir: bundles.codec, dtype: .shipped)
        } catch let error as HeartMuLaError {
            throw HeartMuLaPackageError.loadFailed(error.description)
        }
        MLX.Memory.clearCache()
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every LM frame (generate), every Euler step
        // (denoise), every SQ-Codec window (decode) — the Core's `isCancelled` probe; the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .music, let req = request as? MusicRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req)
        try Task.checkCancellation()
        let song: SongResult
        do {
            song = try pipeline.generate(plan.request, progress: { p in
                switch p {
                case .generate(let frame, let maxFrames):
                    RunProgress.report(.generate, step: frame, totalSteps: maxFrames)
                case .denoise(let window, let windows, let step, let steps):
                    RunProgress.report(.denoise, step: step, totalSteps: steps, stage: window, totalStages: windows)
                case .decode(let window, let windows):
                    RunProgress.report(.decode, step: window, totalSteps: windows)
                }
            }, isCancelled: { Task.isCancelled })
        } catch HeartMuLaError.cancelled {
            throw CancellationError()
        } catch HeartMuLaError.invalidInput(let why) {
            // e.g. lyrics too long for the backbone's context at this cap — the request's fault, said legibly
            throw PackageError.unsupportedRequestFeature(why)
        } catch let error as HeartMuLaError {
            throw HeartMuLaPackageError.generationFailed(error.description)
        }
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        let interleaved = song.audio.transposed(1, 0).asArray(Float.self)
        let wav = HeartMuLaAudioIO.encodeWAV16(interleaved: interleaved, channels: Self.channels, sampleRate: song.sampleRate)
        // `truncated` is honest (C3): true when the cap, not the model's end token, ended the song.
        return MusicResponse(audio: Audio(format: .wav, data: wav, sampleRate: song.sampleRate, channels: Self.channels),
                             truncated: !song.endedOnEndToken)
    }

    /// What the request asks for, judged against the declaration (the engine admits only declared features; a direct caller
    /// meets the same rules here) and against the Core's own bounds, mapped onto the Core's request.
    public struct Plan {
        public let request: SongRequest
        /// The cap the audio honours, in seconds.
        public let capSeconds: Double
    }

    /// The caption as HeartMuLa's tags — the V0 / V2b harness's `tags_for`: split on commas, trimmed, lower-cased, re-joined.
    public nonisolated static func tags(from caption: String) -> String {
        caption.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }.joined(separator: ",")
    }

    public nonisolated static func plan(_ req: MusicRequest) throws -> Plan {
        if let refusal = controls.refusal(for: req) { throw PackageError.unsupportedRequestFeature(refusal) }
        let tags = tags(from: req.caption)
        guard !tags.isEmpty else {
            throw PackageError.unsupportedRequestFeature("empty caption — HeartMuLa's only style input is its tags (comma-separated)")
        }
        let lyrics = req.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lyrics.isEmpty else {
            throw PackageError.unsupportedRequestFeature("empty lyrics — this surface sings lyrics; no instrumental mode is gated")
        }
        let cap = req.durationSeconds ?? defaultDurationSeconds
        guard cap.isFinite, cap >= minDurationSeconds, cap <= maxDurationSeconds else {
            throw PackageError.unsupportedRequestFeature(String(
                format: "durationSeconds(%g) — the cap must be %g…%g s", cap, minDurationSeconds, maxDurationSeconds))
        }
        // the cap bounds the AUDIO: floor(cap × 12.5) frames in all (heartlib's own count runs one frame past its cap)
        let r = SongRequest(tags: tags, lyrics: lyrics, maxSeconds: SongRequest.lmSeconds(cappingAudioAt: cap),
                            seed: req.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max)))
        return Plan(request: r, capSeconds: cap)
    }
}

extension HeartMuLaPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(HeartMuLaPackage.self) }
}

/// Wrapper-level errors (weight resolution, tier, TF32, a Core failure). Runtime request errors use `PackageError`.
public enum HeartMuLaPackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    case unsupportedTier(String)
    case tf32Enabled(String)
    case loadFailed(String)
    case generationFailed(String)
    public var description: String {
        switch self {
        case .missingWeights(let s): return "HeartMuLa: \(s)"
        case .unsupportedTier(let s): return "HeartMuLa: unsupported tier \(s)"
        case .tf32Enabled(let s): return "HeartMuLa: \(s)"
        case .loadFailed(let s): return "HeartMuLa: load failed — \(s)"
        case .generationFailed(let s): return "HeartMuLa: \(s)"
        }
    }
}
