//
//  AudioCaptureEngine.swift
//  meh
//
//  Real-time audio capture at 16kHz for turn detection.
//
//  Uses AVAudioEngine to:
//  - Capture microphone input
//  - Convert to 16kHz mono PCM
//  - Buffer up to 8 seconds of audio
//  - Provide audio chunks for turn detection
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

    private let audioEngine = AVAudioEngine()
    private let inputNode: AVAudioInputNode
    private var audioBuffer: [Float] = []
    private let bufferLock = NSLock()

    // For sample rate conversion
    private var converter: AVAudioConverter?

    // Audio buffer consumers (for speech recognition, etc.)
    private var consumers: [WeakAudioBufferConsumer] = []
    private let consumersLock = NSLock()

    // MARK: - Initialization

    init() {
        self.inputNode = audioEngine.inputNode
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
        // This prevents format mismatch crashes on restart
        inputNode.removeTap(onBus: 0)
        print("✅ Audio tap removed")

        // Reset audio engine to clear internal state
        audioEngine.reset()
        print("✅ Audio engine reset")

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
        // This is safe even if no tap exists
        inputNode.removeTap(onBus: 0)
        print("🔧 Removed any existing audio tap")

        // Get input format (usually 48kHz on modern devices)
        let inputFormat = inputNode.outputFormat(forBus: 0)
        print("🔧 Input format: \(inputFormat.sampleRate) Hz, \(inputFormat.channelCount) channels")

        // Check if format is valid (simulator often returns 0 Hz)
        guard inputFormat.sampleRate > 0 else {
            print("❌ Input format invalid (0 Hz sample rate)")
            throw AudioCaptureError.invalidAudioFormat
        }

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

        // Install tap with the ACTUAL input format (not a hardcoded one)
        inputNode.installTap(
            onBus: 0,
            bufferSize: Self.processingBufferSize,
            format: inputFormat  // Use the actual hardware format
        ) { [weak self] buffer, _ in
            self?.processCapturedAudio(buffer: buffer)
        }
        print("✅ Audio tap installed")
    }

    private func startEngine() throws {
        do {
            try audioEngine.start()
        } catch {
            throw AudioCaptureError.audioEngineStartFailed
        }
    }

    /// Process captured audio buffer
    private func processCapturedAudio(buffer: AVAudioPCMBuffer) {
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
