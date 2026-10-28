// Footprint.swift — `phys_footprint`, the fleet's declared-memory figure (AB-L-0113): task_info(TASK_VM_INFO).phys_footprint,
// the same number proc_pid_rusage's ri_phys_footprint and the Python harness's FootprintSampler report. Plus a 100 ms sampler
// that keeps the peak, mirroring WIP/heartmula-eval/harness/common.py.
import Darwin
import Foundation

public enum Footprint {
    /// Bytes of physical footprint of this process (0 when the call fails).
    public static func physBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    /// Samples `physBytes` on a background thread; `peak` is the largest value seen, `mark()` samples now.
    public final class Sampler: @unchecked Sendable {
        public private(set) var peak: UInt64 = 0
        /// (seconds since start, bytes) at every sample — the harness's `phys_series.json`.
        public private(set) var series: [(Double, UInt64)] = []
        private let t0 = Date()
        private let period: TimeInterval
        private var running = true
        private let lock = NSLock()
        private var thread: Thread?

        public init(period: TimeInterval = 0.1) {
            self.period = period
            let t = Thread { [weak self] in
                while let self, self.isRunning { _ = self.mark(); Thread.sleep(forTimeInterval: period) }
            }
            t.qualityOfService = .utility
            thread = t
            t.start()
        }

        private var isRunning: Bool { lock.lock(); defer { lock.unlock() }; return running }

        @discardableResult
        public func mark() -> UInt64 {
            let b = Footprint.physBytes()
            lock.lock(); peak = max(peak, b); series.append((Date().timeIntervalSince(t0), b)); lock.unlock()
            return b
        }

        public func resetPeak() { lock.lock(); peak = Footprint.physBytes(); lock.unlock() }

        public func stop() -> UInt64 { lock.lock(); running = false; lock.unlock(); return peak }
    }
}
