//
//  SpeechRecognitionManager.swift
//  smart-turn-ios
//
//  Manages real-time speech-to-text using Apple's Speech framework.
//  Works alongside AudioCaptureEngine by processing the same audio buffer.
//

import Foundation
import Speech
import AVFoundation
import Combine

/// Manages Apple Speech framework for real-time transcription
/// Uses shared AudioCaptureEngine to avoid dual audio engine conflict
@MainActor
final class SpeechRecognitionManager: ObservableObject, AudioBufferConsumer {

    // MARK: - Published Properties

    @Published var isEnabled = false {
        didSet {
            if isEnabled && !isAuthorized {
                Task { await requestAuthorization() }
            } else if !isEnabled && isRecognizing {
                stopRecognition()
            }
        }
    }

    @Published private(set) var isAuthorized = false
    @Published private(set) var isRecognizing = false
    @Published private(set) var transcribedText: String = ""
    @Published private(set) var statusMessage: String = "Speech recognition disabled"

    // MARK: - Private Properties

    private let speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private weak var audioEngine: AudioCaptureEngine?

    // MARK: - Initialization

    init(audioEngine: AudioCaptureEngine) {
        self.audioEngine = audioEngine

        // Initialize speech recognizer (uses device locale)
        speechRecognizer = SFSpeechRecognizer()

        // Check if speech recognition is available
        guard speechRecognizer != nil else {
            print("❌ Speech recognition not available for device locale")
            statusMessage = "Speech recognition not available"
            return
        }

        print("✅ SpeechRecognitionManager initialized with shared audio engine")
    }

    // MARK: - AudioBufferConsumer

    /// Receives audio buffers from shared AudioCaptureEngine (called on background thread)
    nonisolated func didReceiveAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        // Forward buffer to speech recognition request
        // Note: SFSpeechAudioBufferRecognitionRequest.append() is thread-safe
        Task { @MainActor [weak self] in
            self?.recognitionRequest?.append(buffer)
        }
    }

    // MARK: - Authorization

    /// Request speech recognition authorization
    func requestAuthorization() async {
        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        await MainActor.run {
            switch status {
            case .authorized:
                isAuthorized = true
                statusMessage = "Speech recognition ready"
                print("✅ Speech recognition authorized")
            case .denied:
                isAuthorized = false
                statusMessage = "Speech recognition denied"
                print("❌ Speech recognition denied by user")
            case .restricted:
                isAuthorized = false
                statusMessage = "Speech recognition restricted"
                print("❌ Speech recognition restricted")
            case .notDetermined:
                isAuthorized = false
                statusMessage = "Speech recognition not determined"
                print("⚠️ Speech recognition authorization not determined")
            @unknown default:
                isAuthorized = false
                statusMessage = "Unknown authorization status"
                print("❌ Unknown speech recognition authorization status")
            }
        }
    }

    // MARK: - Recognition

    /// Start real-time speech recognition using shared audio engine
    func startRecognition() throws {
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            throw SpeechRecognitionError.recognizerNotAvailable
        }

        guard isAuthorized else {
            throw SpeechRecognitionError.notAuthorized
        }

        guard let audioEngine = audioEngine else {
            throw SpeechRecognitionError.audioEngineError
        }

        // Cancel any ongoing recognition
        if isRecognizing {
            stopRecognition()
        }

        print("🎙️ Starting Apple Speech recognition...")

        // Create recognition request
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true  // Real-time partial results
        request.requiresOnDeviceRecognition = false  // Use cloud for better accuracy

        recognitionRequest = request

        // Register to receive audio buffers from shared engine
        audioEngine.addConsumer(self)

        // Start recognition task
        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor [weak self] in
                guard let self = self else { return }

                if let error = error {
                    print("❌ Speech recognition error: \(error.localizedDescription)")
                    self.statusMessage = "Recognition error: \(error.localizedDescription)"
                    self.stopRecognition()
                    return
                }

                if let result = result {
                    // Update transcribed text with the best transcription
                    let transcription = result.bestTranscription.formattedString
                    self.transcribedText = transcription

                    if result.isFinal {
                        print("✅ Final transcription: \"\(transcription)\"")
                    } else {
                        print("📝 Partial transcription: \"\(transcription)\"")
                    }
                }
            }
        }

        isRecognizing = true
        statusMessage = "🎙️ Listening..."
        print("✅ Speech recognition started (using shared audio engine)")
    }

    /// Stop speech recognition
    func stopRecognition() {
        guard isRecognizing else { return }

        print("⏹️ Stopping speech recognition...")

        // Unregister from shared audio engine
        if let audioEngine = audioEngine {
            audioEngine.removeConsumer(self)
        }

        // Cancel recognition
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()

        recognitionRequest = nil
        recognitionTask = nil

        isRecognizing = false
        statusMessage = isAuthorized ? "Speech recognition ready" : "Speech recognition disabled"

        // Note: Audio session is managed by AudioCaptureEngine, not here
        // Deactivating would kill the shared audio engine

        print("✅ Speech recognition stopped")
    }

    /// Clear transcription text
    func clearTranscription() {
        transcribedText = ""
        if isAuthorized {
            statusMessage = "Speech recognition ready"
        }
    }
}

// MARK: - Errors

enum SpeechRecognitionError: Error, LocalizedError {
    case recognizerNotAvailable
    case notAuthorized
    case audioEngineError

    var errorDescription: String? {
        switch self {
        case .recognizerNotAvailable:
            return "Speech recognizer is not available"
        case .notAuthorized:
            return "Speech recognition is not authorized. Please enable it in Settings."
        case .audioEngineError:
            return "Failed to start audio engine"
        }
    }
}
