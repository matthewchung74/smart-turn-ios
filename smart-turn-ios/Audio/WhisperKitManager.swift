//
//  WhisperKitManager.swift
//  meh
//
//  Manages WhisperKit model loading/unloading and speech-to-text transcription.
//  Models are bundled with the app - no downloads required.
//

import Foundation
import Combine
import WhisperKit

/// Available WhisperKit model sizes (bundled with app)
enum WhisperModel: String, CaseIterable {
    case base = "openai_whisper-base"
    case small = "openai_whisper-small"

    var displayName: String {
        switch self {
        case .base: return "Base"
        case .small: return "Small"
        }
    }
}

/// Manages WhisperKit lifecycle and transcription
@MainActor
final class WhisperKitManager: ObservableObject {

    // MARK: - Published Properties

    @Published var isEnabled = false {
        didSet {
            print("🔍 DEBUG: isEnabled changed from \(oldValue) to \(isEnabled)")
            print("🔍 DEBUG: isLoaded = \(isLoaded), isLoading = \(isLoading)")

            if isEnabled && !isLoaded {
                print("🔍 DEBUG: Starting model load Task...")
                Task {
                    print("🔍 DEBUG: Inside model load Task, calling loadModel()...")
                    await loadModel()
                    print("🔍 DEBUG: loadModel() returned")
                }
            } else if !isEnabled && isLoaded {
                print("🔍 DEBUG: Starting model unload Task...")
                Task { await unloadModel() }
            } else {
                print("🔍 DEBUG: No action taken (isLoaded=\(isLoaded))")
            }
        }
    }

    @Published var selectedModel: WhisperModel = .base {
        didSet {
            // Reload if already enabled
            if isEnabled && isLoaded {
                Task {
                    await unloadModel()
                    await loadModel()
                }
            }
        }
    }

    @Published private(set) var isLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var statusMessage: String = "Transcription disabled"
    @Published private(set) var transcribedText: String = ""
    @Published private(set) var isTranscribing = false
    @Published private(set) var isStreaming = false

    // MARK: - Private Properties

    private var whisperKit: WhisperKit?
    private var streamTranscriber: AudioStreamTranscriber?

    // MARK: - Private Methods

    /// Load the selected model from bundle
    private func loadModel() async {
        print("🔍 DEBUG: loadModel() called, isLoading=\(isLoading)")
        guard !isLoading else {
            print("🔍 DEBUG: Already loading, returning early")
            return
        }

        print("🔍 DEBUG: Setting isLoading = true")
        isLoading = true
        statusMessage = "Loading \(selectedModel.displayName) model..."
        print("📥 Loading WhisperKit model from bundle: \(selectedModel.rawValue)")

        do {
            print("🔍 DEBUG: Getting bundle path...")
            // Load models from app bundle
            // Models should be in: Models/WhisperKit/whisperkit-coreml/openai_whisper-base/
            guard let bundlePath = Bundle.main.resourcePath else {
                print("❌ DEBUG: Bundle.main.resourcePath is nil!")
                throw WhisperKitError.modelNotLoaded
            }

            print("🔍 DEBUG: Bundle path: \(bundlePath)")

            // Point directly to the model folder (which is copied to bundle root)
            let modelPath = "\(bundlePath)/\(selectedModel.rawValue)"
            print("📂 Looking for models at: \(modelPath)")

            print("🔍 DEBUG: Creating WhisperKitConfig...")
            let config = WhisperKitConfig(
                modelFolder: modelPath,
                verbose: true,
                logLevel: .debug
            )

            print("🔍 DEBUG: Calling WhisperKit(config)... THIS MAY TAKE 20+ SECONDS")
            print("🔍 DEBUG: If you don't see any more logs, WhisperKit init is hanging or crashing")

            whisperKit = try await WhisperKit(config)

            print("🔍 DEBUG: WhisperKit(config) returned successfully!")

            isLoaded = true
            isLoading = false
            statusMessage = "\(selectedModel.displayName) model ready"

            print("✅ WhisperKit loaded from bundle: \(selectedModel.rawValue)")

        } catch {
            print("🔍 DEBUG: Caught error during model loading")
            isLoading = false
            isLoaded = false
            isEnabled = false
            statusMessage = "Failed to load model: \(error.localizedDescription)"

            print("❌ WhisperKit loading failed: \(error)")
        }
    }

    /// Unload the current model
    private func unloadModel() async {
        guard whisperKit != nil else { return }

        print("🗑️ Unloading WhisperKit model")

        whisperKit = nil
        isLoaded = false
        transcribedText = ""

        statusMessage = "Transcription disabled"
        print("✅ WhisperKit unloaded")
    }

    /// Transcribe audio buffer with progressive callback updates
    func transcribe(audioBuffer: [Float]) async throws -> String {
        guard let whisperKit = whisperKit else {
            throw WhisperKitError.modelNotLoaded
        }

        guard !isTranscribing else {
            print("⚠️ Already transcribing, skipping...")
            return transcribedText
        }

        isTranscribing = true
        statusMessage = "🎤 Transcribing..."

        do {
            print("🎯 Starting transcription of \(audioBuffer.count) samples...")

            // WhisperKit transcribe method with callback for progressive updates
            let results = try await whisperKit.transcribe(
                audioArray: audioBuffer,
                callback: { progress in
                    // Log the progress to see what data we get
                    print("📊 CALLBACK - Progress update received:")
                    print("   Type: \(type(of: progress))")
                    print("   Value: \(progress)")

                    // Use Mirror to inspect all properties
                    let mirror = Mirror(reflecting: progress)
                    for child in mirror.children {
                        if let label = child.label {
                            print("   - \(label): \(child.value)")
                        }
                    }

                    return nil  // Continue processing (return false to cancel)
                }
            )

            print("📊 Transcription results: \(results.count) segments")

            // Extract text from all segments
            let text = results.map { $0.text }.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            transcribedText = text

            isTranscribing = false

            if !text.isEmpty {
                statusMessage = "✅ Transcribed (\(text.count) chars)"
                print("✅ Transcription complete: \"\(text)\"")
            } else {
                statusMessage = "⚠️ No speech detected"
                print("⚠️ Transcription returned empty text")
            }

            return text

        } catch {
            isTranscribing = false
            statusMessage = "❌ Transcription failed: \(error.localizedDescription)"
            print("❌ Transcription error: \(error)")
            throw error
        }
    }

    /// Clear transcription text
    func clearTranscription() {
        transcribedText = ""
        if isLoaded {
            statusMessage = "\(selectedModel.displayName) model ready"
        }
    }

    // MARK: - Streaming Transcription

    /// Start real-time streaming transcription
    func startStreamingTranscription() async throws {
        guard let whisperKit = whisperKit else {
            throw WhisperKitError.modelNotLoaded
        }

        guard !isStreaming else {
            print("⚠️ Already streaming, skipping...")
            return
        }

        print("🎙️ Starting streaming transcription...")

        // Ensure tokenizer is available
        guard let tokenizer = whisperKit.tokenizer else {
            throw WhisperKitError.modelNotLoaded
        }

        // Create decoding options
        let decodingOptions = DecodingOptions(
            verbose: false,
            task: .transcribe,
            temperature: 0.0,
            temperatureFallbackCount: 5,
            sampleLength: 224,
            topK: 5,
            usePrefillPrompt: true,
            usePrefillCache: true,
            skipSpecialTokens: true,
            withoutTimestamps: false,
            clipTimestamps: []
        )

        // Create AudioStreamTranscriber with state callback
        let transcriber = AudioStreamTranscriber(
            audioEncoder: whisperKit.audioEncoder,
            featureExtractor: whisperKit.featureExtractor,
            segmentSeeker: whisperKit.segmentSeeker,
            textDecoder: whisperKit.textDecoder,
            tokenizer: tokenizer,
            audioProcessor: whisperKit.audioProcessor,
            decodingOptions: decodingOptions,
            requiredSegmentsForConfirmation: 2,
            silenceThreshold: 0.3,
            compressionCheckWindow: 60,
            useVAD: true,
            stateChangeCallback: { [weak self] oldState, newState in
                print("🔔 stateChangeCallback invoked!")
                print("   - Confirmed segments: \(newState.confirmedSegments.count)")
                print("   - Unconfirmed segments: \(newState.unconfirmedSegments.count)")

                Task { @MainActor [weak self] in
                    guard let self = self else {
                        print("⚠️ self is nil in stateChangeCallback")
                        return
                    }

                    // Update transcribed text with confirmed segments
                    let confirmed = newState.confirmedSegments.map { $0.text }.joined(separator: " ")
                    let unconfirmed = newState.unconfirmedSegments.map { $0.text }.joined(separator: " ")

                    print("   - Confirmed text: \"\(confirmed)\"")
                    print("   - Unconfirmed text: \"\(unconfirmed)\"")

                    // Combine confirmed + unconfirmed for real-time display
                    let fullText = [confirmed, unconfirmed]
                        .filter { !$0.isEmpty }
                        .joined(separator: " ")
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                    if !fullText.isEmpty {
                        self.transcribedText = fullText
                        print("📝 Streaming update: \"\(fullText)\"")
                    } else {
                        print("⚠️ fullText is empty after combining segments")
                    }
                }
            }
        )

        streamTranscriber = transcriber
        isStreaming = true
        statusMessage = "🎙️ Streaming..."

        print("🎙️ About to call startStreamTranscription()...")

        // Start the stream (this runs until stopped)
        Task {
            do {
                print("🎙️ Calling startStreamTranscription() now...")
                try await transcriber.startStreamTranscription()
                print("✅ startStreamTranscription() returned successfully")
            } catch {
                await MainActor.run {
                    self.isStreaming = false
                    self.statusMessage = "❌ Stream error: \(error.localizedDescription)"
                    print("❌ Streaming error: \(error)")
                }
            }
        }

        print("🎙️ startStreamingTranscription() method completed (Task started in background)")
    }

    /// Stop streaming transcription
    func stopStreamingTranscription() async {
        guard let transcriber = streamTranscriber, isStreaming else {
            return
        }

        print("⏹️ Stopping streaming transcription...")

        await transcriber.stopStreamTranscription()

        isStreaming = false
        streamTranscriber = nil
        statusMessage = isLoaded ? "\(selectedModel.displayName) model ready" : "Transcription disabled"

        print("✅ Streaming stopped")
    }
}

// MARK: - Errors

enum WhisperKitError: Error, LocalizedError {
    case modelNotLoaded
    case transcriptionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "WhisperKit model is not loaded. Please load a model first."
        case .transcriptionFailed(let message):
            return "Transcription failed: \(message)"
        }
    }
}
