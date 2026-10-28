// WAV.swift — PCM WAV out for songs: 16-bit interleaved (any channel count) and 32-bit float. Lifted in shape from
// mlx-yue2-swift (mlx-auk-swift lineage). `samples` is (frames, channels) row-major, i.e. interleaved.
import Foundation

public enum WAV {
    public static func write16(_ samples: [Float], channels: Int, sampleRate: Int, to url: URL) throws {
        try encode(samples, channels: channels, sampleRate: sampleRate, float: false).write(to: url)
    }

    public static func writeFloat32(_ samples: [Float], channels: Int, sampleRate: Int, to url: URL) throws {
        try encode(samples, channels: channels, sampleRate: sampleRate, float: true).write(to: url)
    }

    static func encode(_ samples: [Float], channels: Int, sampleRate: Int, float: Bool) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { Swift.withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { Swift.withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytesPerSample = float ? 4 : 2
        let blockAlign = channels * bytesPerSample
        let payloadBytes = samples.count * bytesPerSample
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + payloadBytes)); data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(float ? 3 : 1); u16(UInt16(channels)); u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * blockAlign)); u16(UInt16(blockAlign)); u16(UInt16(bytesPerSample * 8))
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(payloadBytes))
        if float {
            samples.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        } else {
            let pcm = samples.map { Int16(Swift.max(-32768, Swift.min(32767, ($0 * 32767).rounded()))) }
            pcm.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        return data
    }
}
