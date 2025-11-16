//
//  AudioCaptureEngine.swift
//  smart-turn-ios
//
//  Real-time audio capture at 16kHz for turn detection.
//
//  Uses AVAudioEngine to:
//  - Capture microphone input
//  - Convert to 16kHz mono PCM
//  - Buffer up to 8 seconds of audio
//  - Provide audio chunks for turn detection
//
//  CRITICAL: AVAudioEngine Bug Workaround
//  =====================================
//  This class recreates AVAudioEngine on EVERY start to work around a critical Apple bug
//  where reusing the same engine instance causes tap callbacks to stop firing.
//
//  Bug Symptoms (if engine is reused):
//  - First start: Works perfectly ✅
//  - Second start: BROKEN ❌ - tap callback never fires, no audio captured
//  - Third start: Works ✅
//  - Fourth start: BROKEN ❌
//  - Pattern: Alternating success/failure on every other restart
//
//  Root Cause:
//  AVAudioEngine becomes corrupted during stop/start cycles. Even though:
//  - audioEngine.start() succeeds (no error)
//  - audioEngine.isRunning returns true
//  - installTap() succeeds (no error)
//  The tap callback is never invoked, resulting in zero audio data.
//  The engine then silently stops itself mid-session.
//
//  Solution:
//  1. Declare audioEngine as `var` (not `let`) and implicitly unwrapped optional
//  2. Create fresh instance on EVERY startCapture(): `audioEngine = AVAudioEngine()`
//  3. Deallocate on EVERY stopCapture(): `audioEngine = nil`
//  4. NEVER call audioEngine.reset() - it leaves engine in corrupted state
//
//  DO NOT REFACTOR to reuse the engine instance - this will break audio on every other start!
//

import AVFoundation
import Accelerate
import Combine

/// Protocol for objects that want to receive raw audio buffers
protocol AudioBufferConsumer: AnyObject {
    /// Called when a new audio buffer is available (on background thread)
    func didReceiveAudioBuffer(_ buffer: AVAudioPCMBuffer)
}

/// Errors that can occur during audio capture
enum AudioCaptureError: Error {
    case microphoneNotAuthorized
    case audioEngineStartFailed
    case audioSessionSetupFailed
    case invalidAudioFormat

    var localizedDescription: String {
        switch self {
        case .microphoneNotAuthorized:
            return "Microphone access not authorized. Please grant permission in Settings."
        case .audioEngineStartFailed:
            return "Failed to start audio engine. Please check microphone availability."
        case .audioSessionSetupFailed:
            return "Failed to configure audio session."
        case .invalidAudioFormat:
            return "Audio format conversion failed."
        }
    }
}

/// Real-time audio capture engine with 16kHz downsampling
@MainActor
final class AudioCaptureEngine: ObservableObject {

    // MARK: - Constants

    /// Target sample rate (16kHz for Whisper)
    static let targetSampleRate: Double = 16_000

    /// Maximum buffer length in seconds
    static let maxBufferSeconds: Double = 8.0

    /// Maximum buffer size in samples
    static let maxBufferSamples: Int = Int(targetSampleRate * maxBufferSeconds)

    /// Buffer size for audio processing (in samples)
    static let processingBufferSize: AVAudioFrameCount = 4096

    // MARK: - Properties

    @Published private(set) var isRecording = false
    @Published private(set) var audioLevel: Float = 0.0
    @Published private(set) var bufferDuration: Double = 0.0

    // CRITICAL: AVAudioEngine instance (recreated on every start)
    // ============================================================
    // MUST be `var` and implicitly unwrapped optional (not `let` or regular optional)
    // because we recreate the entire engine on every startCapture() call.
    //
    // Why recreate instead of reuse?
    // - AVAudioEngine has a bug where tap callbacks stop firing after stop/start
    // - Symptoms: alternating success/failure (works, broken, works, broken...)
    // - Only fix: Create fresh instance every time (see file header for details)
    //
    // DO NOT change to `let` or reuse - this will break audio on every other start!
    private var audioEngine: AVAudioEngine!
    private var audioBuffer: [Float] = []
    private let bufferLock = NSLock()

    // For sample rate conversion
    private var converter: AVAudioConverter?

    // Audio buffer consumers (for speech recognition, etc.)
    private var consumers: [WeakAudioBufferConsumer] = []
    private let consumersLock = NSLock()

    // Flag to track if we've logged empty buffer warnings (suppress repeated logs)
    private var hasLoggedEmptyBuffers = false

    // MARK: - Initialization

    init() {
        // No initialization needed:
        // - audioEngine is created on-demand in startCapture() (workaround for Apple bug)
        // - All other properties have default values
    }

    // MARK: - Public Methods

    /// Request microphone permission
    func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { continuation in
            #if os(iOS)
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
            #else
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
            #endif
        }
    }

    /// Check current microphone permission status
    var microphonePermissionStatus: Bool {
        #if os(iOS)
        if #available(iOS 17.0, *) {
            return AVAudioApplication.shared.recordPermission == .granted
        } else {
            return AVAudioSession.sharedInstance().recordPermission == .granted
        }
        #else
        return AVAudioSession.sharedInstance().recordPermission == .granted
        #endif
    }

    /// Start audio capture
    func startCapture() throws {
        // Check if already recording
        if isRecording {
            print("⚠️ Already recording, ignoring startCapture()")
            return
        }

        print("🎙️ Starting audio capture...")

        // CRITICAL FIX: Recreate AVAudioEngine on EVERY start
        // ====================================================
        // This recreates the entire AVAudioEngine instance from scratch to work around
        // an Apple bug where reusing the same instance after stop/start causes tap
        // callbacks to never fire (alternating success/failure pattern).
        //
        // Evidence of the bug (if engine is reused):
        // - audioEngine.start() succeeds (no error thrown)
        // - audioEngine.isRunning returns true (engine claims to be running)
        // - installTap() succeeds (no error thrown)
        // - BUT: The tap callback is NEVER invoked (zero audio data captured)
        // - Engine then silently stops itself (isRunning becomes false mid-session)
        // - Pattern: 1st start works, 2nd fails, 3rd works, 4th fails, etc.
        //
        // This bug was discovered through extensive testing and the ONLY reliable fix
        // is to create a completely fresh AVAudioEngine instance on every start.
        //
        // DO NOT optimize this by caching/reusing the engine - it WILL break audio!
        audioEngine = AVAudioEngine()
        print("🔧 Created fresh AVAudioEngine instance")

        // Reset empty buffer logging flag for new session
        hasLoggedEmptyBuffers = false

        // Check permission
        guard microphonePermissionStatus else {
            print("❌ Microphone permission not granted")
            throw AudioCaptureError.microphoneNotAuthorized
        }

        // Warn if running on simulator
        #if targetEnvironment(simulator)
        print("⚠️ Running on iOS Simulator")
        print("   Audio capture may not work properly on simulator")
        print("   For best results, test on a real iOS device")
        #endif

        // Configure audio session
        print("🔧 Configuring audio session...")
        try configureAudioSession()

        // Setup audio tap
        print("🔧 Setting up audio tap...")
        try setupAudioTap()

        // Start engine
        print("🔧 Starting audio engine...")
        try startEngine()

        isRecording = true
        print("✅ Audio capture started")
    }

    /// Stop audio capture
    func stopCapture() {
        guard isRecording else {
            print("⚠️ Already stopped, ignoring stopCapture()")
            return
        }

        print("⏹️ Stopping audio capture...")

        // Stop engine first
        if audioEngine.isRunning {
            audioEngine.stop()
            print("✅ Audio engine stopped")
        }

        // ALWAYS remove tap (even if we think there isn't one)
        // This prevents format mismatch crashes and ensures clean shutdown
        audioEngine.inputNode.removeTap(onBus: 0)
        print("✅ Audio tap removed")

        // IMPORTANT: We do NOT call audioEngine.reset() here because:
        // - reset() leaves the engine in a corrupted/invalid state
        // - This causes tap callbacks to fail on subsequent starts
        // - Instead, we deallocate the entire engine (see below) and create fresh on next start
        // - This is the ONLY reliable way to avoid AVAudioEngine tap callback bugs

        isRecording = false

        // Clear buffer
        bufferLock.lock()
        audioBuffer.removeAll()
        bufferDuration = 0.0
        bufferLock.unlock()

        // Clear consumers (they should have unregistered, but clean up stale refs)
        consumersLock.lock()
        consumers.removeAll { $0.value == nil }
        consumersLock.unlock()

        // Deactivate audio session
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            print("✅ Audio session deactivated")
        } catch {
            print("⚠️ Failed to deactivate audio session: \(error)")
        }

        // CRITICAL: Deallocate the entire AVAudioEngine instance
        // =======================================================
        // This is part of the workaround for the AVAudioEngine tap callback bug.
        // By setting to nil, we force complete deallocation and release all resources.
        // On next startCapture(), we create a completely fresh instance from scratch.
        //
        // DO NOT skip this step - it's essential for the bug workaround to work!
        audioEngine = nil
        print("✅ Audio engine deallocated")

        print("✅ Audio capture stopped")
    }

    /// Get current audio buffer (last 8 seconds)
    func getCurrentBuffer() -> [Float] {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        return Array(audioBuffer)
    }

    /// Clear audio buffer
    func clearBuffer() {
        bufferLock.lock()
        audioBuffer.removeAll()
        bufferDuration = 0.0
        bufferLock.unlock()
    }

    /// Register an audio buffer consumer (e.g., speech recognition)
    func addConsumer(_ consumer: AudioBufferConsumer) {
        consumersLock.lock()
        defer { consumersLock.unlock() }

        // Remove any deallocated consumers
        consumers.removeAll { $0.value == nil }

        // Add new consumer
        consumers.append(WeakAudioBufferConsumer(value: consumer))
        print("📢 Audio consumer registered (total: \(consumers.count))")
    }

    /// Unregister an audio buffer consumer
    func removeConsumer(_ consumer: AudioBufferConsumer) {
        consumersLock.lock()
        defer { consumersLock.unlock() }

        consumers.removeAll { $0.value === consumer || $0.value == nil }
        print("📢 Audio consumer unregistered (remaining: \(consumers.count))")
    }

    // MARK: - Private Methods

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()

        do {
            // Configure for recording
            try session.setCategory(.record, mode: .measurement)
            try session.setActive(true)

            // Set preferred sample rate (hardware will do conversion if needed)
            try session.setPreferredSampleRate(Self.targetSampleRate)
        } catch {
            throw AudioCaptureError.audioSessionSetupFailed
        }
    }

    private func setupAudioTap() throws {
        // Remove any existing tap first to prevent format mismatch crashes
        // This is safe even if no tap exists (no-op if no tap installed)
        audioEngine.inputNode.removeTap(onBus: 0)
        print("🔧 Removed any existing audio tap")

        // CRITICAL: Query audio session's ACTUAL hardware sample rate
        // ===========================================================
        // DO NOT use inputNode.outputFormat(forBus:) - it returns stale/cached format!
        // After engine recreation, the cached format doesn't match actual hardware.
        // Always query AVAudioSession directly for current hardware configuration.
        let actualSampleRate = AVAudioSession.sharedInstance().sampleRate
        let rawChannels = AVAudioSession.sharedInstance().inputNumberOfChannels

        // Ensure at least 1 channel (some devices may report 0 if no mic connected)
        let actualChannels = max(1, rawChannels)

        print("🔧 Audio session actual: \(actualSampleRate) Hz, \(rawChannels) channels (using: \(actualChannels))")

        // Check if format is valid (simulator often returns 0 Hz)
        guard actualSampleRate > 0 else {
            print("❌ Audio session sample rate invalid (0 Hz)")
            throw AudioCaptureError.invalidAudioFormat
        }

        // Create input format based on actual audio session settings
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: actualSampleRate,
            channels: AVAudioChannelCount(actualChannels),
            interleaved: false
        ) else {
            print("❌ Failed to create input format")
            throw AudioCaptureError.invalidAudioFormat
        }
        print("🔧 Input format created: \(inputFormat.sampleRate) Hz, \(inputFormat.channelCount) channels")

        // Create target format (16kHz mono)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioCaptureError.invalidAudioFormat
        }
        print("🔧 Target format: \(targetFormat.sampleRate) Hz, \(targetFormat.channelCount) channels")

        // Create converter
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            print("❌ Failed to create audio converter")
            throw AudioCaptureError.invalidAudioFormat
        }
        self.converter = converter
        print("✅ Audio converter created")

        // Install tap with the ACTUAL hardware input format
        // ==================================================
        // CRITICAL: Must use the inputFormat we created from AVAudioSession.sampleRate
        // DO NOT use a hardcoded format or inputNode.outputFormat(forBus:) - format mismatch!
        // The format parameter MUST match the actual hardware configuration.
        audioEngine.inputNode.installTap(
            onBus: 0,
            bufferSize: Self.processingBufferSize,
            format: inputFormat  // MUST be actual hardware format from audio session
        ) { [weak self] buffer, _ in
            self?.processCapturedAudio(buffer: buffer)
        }
        print("✅ Audio tap installed")
    }

    private func startEngine() throws {
        do {
            try audioEngine.start()

            // CRITICAL: Verify engine actually started
            // Sometimes start() succeeds but engine is not running (AVAudioEngine bug)
            if !audioEngine.isRunning {
                print("❌ CRITICAL: audioEngine.start() succeeded but isRunning = false!")
                print("   This is an AVAudioEngine bug - engine in invalid state")
                throw AudioCaptureError.audioEngineStartFailed
            }
            print("✅ Audio engine verified running (isRunning = true)")
        } catch {
            print("❌ Audio engine start failed: \(error)")
            throw AudioCaptureError.audioEngineStartFailed
        }
    }

    /// Process captured audio buffer
    private func processCapturedAudio(buffer: AVAudioPCMBuffer) {
        // Skip empty buffers (happens during microphone initialization after sample rate change)
        // Empty buffers have frameLength == 0 or all samples are zero
        guard buffer.frameLength > 0 else {
            if !hasLoggedEmptyBuffers {
                print("⚠️ Skipping empty buffers (frameLength: 0) - mic initializing...")
                hasLoggedEmptyBuffers = true
            }
            return
        }

        // Quick check: if buffer has data, verify it's not all zeros
        if let channelData = buffer.floatChannelData?[0] {
            var rms: Float = 0
            vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(buffer.frameLength))

            // Skip if RMS is effectively zero (< 0.0001 = -80 dB)
            // This filters out empty/silent initialization buffers
            if rms < 0.0001 {
                if !hasLoggedEmptyBuffers {
                    print("⚠️ Skipping silent buffers (RMS < 0.0001) - mic initializing...")
                    hasLoggedEmptyBuffers = true
                }
                return
            }
        }

        // If we reach here, we have valid audio data - reset the flag for next session
        if hasLoggedEmptyBuffers {
            print("✅ Microphone initialized - receiving audio data")
            hasLoggedEmptyBuffers = false
        }

        // Distribute raw buffer to consumers FIRST (before conversion)
        // This allows speech recognition to get the original high-quality buffer
        consumersLock.lock()
        let currentConsumers = consumers.compactMap { $0.value }
        consumersLock.unlock()

        for consumer in currentConsumers {
            consumer.didReceiveAudioBuffer(buffer)
        }

        // Now proceed with conversion for turn detection
        guard let converter = converter else { return }

        // Calculate output buffer capacity
        let inputSampleRate = buffer.format.sampleRate
        let outputSampleRate = Self.targetSampleRate
        let ratio = outputSampleRate / inputSampleRate
        let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio)

        // Create output buffer
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: converter.outputFormat,
            frameCapacity: outputCapacity
        ) else {
            return
        }

        // Convert sample rate
        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)

        guard error == nil else {
            return
        }

        // Extract Float samples
        guard let channelData = outputBuffer.floatChannelData?[0] else { return }
        let frameCount = Int(outputBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))

        // Calculate audio level (RMS)
        var rms: Float = 0
        vDSP_rmsqv(samples, 1, &rms, vDSP_Length(frameCount))

        // Update buffer
        bufferLock.lock()
        audioBuffer.append(contentsOf: samples)

        // Keep only last 8 seconds
        // Use replaceSubrange instead of removeFirst for O(n) → O(1) under lock
        if audioBuffer.count > Self.maxBufferSamples {
            let overflow = audioBuffer.count - Self.maxBufferSamples
            audioBuffer.replaceSubrange(0..<overflow, with: EmptyCollection())
        }

        let currentBufferDuration = Double(audioBuffer.count) / Self.targetSampleRate
        bufferLock.unlock()

        // Update UI on main thread
        Task { @MainActor in
            self.audioLevel = rms
            self.bufferDuration = currentBufferDuration
        }
    }
}

// MARK: - Weak Consumer Wrapper

/// Weak wrapper to avoid retain cycles with audio buffer consumers
private struct WeakAudioBufferConsumer {
    weak var value: AudioBufferConsumer?
}

// MARK: - Audio Level Helpers

extension AudioCaptureEngine {
    /// Convert RMS to decibels
    func rmsToDecibels(_ rms: Float) -> Float {
        guard rms > 0 else { return -160 }
        return 20 * log10(rms)
    }

    /// Get normalized audio level (0.0 to 1.0)
    var normalizedAudioLevel: Float {
        // Map RMS to 0-1 range (typical speech RMS is 0.01 to 0.3)
        min(max(audioLevel / 0.3, 0), 1)
    }
}
