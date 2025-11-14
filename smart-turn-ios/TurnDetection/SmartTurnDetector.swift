//
//  SmartTurnDetector.swift
//  meh
//
//  Main turn detection orchestrator that:
//  - Manages audio capture
//  - Extracts Whisper features
//  - Runs ONNX Runtime inference
//  - Detects turn completion
//
//  IMPLEMENTATION NOTE:
//  This implementation uses ONNX Runtime instead of Core ML due to the model's
//  INT8 quantization which is not supported by Core ML conversion tools.
//  See CRITICAL_MODEL_BLOCKER.md for full technical details.
//

import Foundation
import Combine
import OnnxRuntimeBindings

/// Turn detection result
struct TurnDetectionResult {
    /// Raw logit output from model
    let logit: Float

    /// Probability of turn completion (sigmoid of logit)
    let probability: Float

    /// Binary prediction (true = turn complete, false = incomplete)
    let isTurnComplete: Bool

    /// Timestamp of prediction
    let timestamp: Date

    /// Audio buffer duration used for prediction
    let audioBufferDuration: Double

    /// Inference time in milliseconds
    let inferenceTimeMs: Double

    /// Computed probability as percentage string
    var probabilityPercentage: String {
        String(format: "%.1f%%", probability * 100)
    }
}

/// Errors specific to turn detection
enum TurnDetectionError: Error {
    case modelNotLoaded
    case modelInitializationFailed(String)
    case featureExtractionFailed
    case inferenceFailed(String)
    case insufficientAudio
    case tensorCreationFailed(String)

    var localizedDescription: String {
        switch self {
        case .modelNotLoaded:
            return "Turn detection model is not available. Please try again later."
        case .modelInitializationFailed(let details):
            return "Failed to initialize ONNX model: \(details)"
        case .featureExtractionFailed:
            return "Unable to process audio. Please check your microphone."
        case .inferenceFailed(let details):
            return "Turn detection inference failed: \(details)"
        case .insufficientAudio:
            return "Please speak for at least half a second before detection."
        case .tensorCreationFailed(let details):
            return "Failed to create input tensor: \(details)"
        }
    }
}

/// Main turn detection coordinator using ONNX Runtime
@MainActor
final class SmartTurnDetector: ObservableObject {

    // MARK: - Constants

    /// Probability threshold for turn completion (0.5 = 50%)
    static let turnCompleteThreshold: Float = 0.5

    /// Minimum audio duration required for detection (seconds)
    static let minimumAudioDuration: Double = 0.5

    /// ONNX model input/output tensor names
    private static let inputTensorName = "input_features"
    private static let outputTensorName = "logits"

    /// Expected input shape: [batch, mel_bins, frames]
    private static let inputShape: [NSNumber] = [1, 80, 800]

    // MARK: - Published Properties

    @Published private(set) var lastResult: TurnDetectionResult?
    @Published private(set) var isProcessing = false
    @Published var errorMessage: String?

    // MARK: - Private Properties

    private let ortEnv: ORTEnv
    private let ortSession: ORTSession
    private let featureExtractor: WhisperFeatureExtractor
    private let audioEngine: AudioCaptureEngine

    // Performance tracking
    private var inferenceHistory: [Double] = []
    private let maxHistorySize = 10

    // MARK: - Initialization

    init?(audioEngine: AudioCaptureEngine) {
        self.audioEngine = audioEngine

        // Initialize feature extractor
        do {
            self.featureExtractor = try WhisperFeatureExtractor()
        } catch {
            print("❌ Failed to initialize WhisperFeatureExtractor: \(error)")
            return nil
        }

        // Initialize ONNX Runtime environment
        do {
            self.ortEnv = try ORTEnv(loggingLevel: .warning)
        } catch {
            print("❌ Failed to initialize ONNX Runtime environment: \(error)")
            return nil
        }

        // Load ONNX model
        guard let modelPath = Bundle.main.path(forResource: "smart-turn-v3.0", ofType: "onnx") else {
            print("❌ ONNX model file not found in bundle. Expected: smart-turn-v3.0.onnx")
            print("   Please ensure the model is added to the Xcode project as a resource.")
            return nil
        }

        do {
            // Create session options for optimization
            let sessionOptions = try ORTSessionOptions()

            // Set optimization level (all optimizations enabled)
            try sessionOptions.setLogSeverityLevel(.warning)
            try sessionOptions.setGraphOptimizationLevel(.all)

            // Set number of threads for inference (2 threads for optimal mobile performance)
            try sessionOptions.setIntraOpNumThreads(2)

            // Create session
            self.ortSession = try ORTSession(
                env: ortEnv,
                modelPath: modelPath,
                sessionOptions: sessionOptions
            )

            print("✅ ONNX Runtime session initialized successfully")
            print("   Model: smart-turn-v3.0.onnx")
            print("   Input: \(Self.inputTensorName) [\(Self.inputShape.map { "\($0)" }.joined(separator: ", "))]")
            print("   Output: \(Self.outputTensorName)")

        } catch {
            print("❌ Failed to create ONNX Runtime session: \(error)")
            return nil
        }
    }

    // MARK: - Public Methods

    /// Run turn detection on current audio buffer
    func detectTurn() async throws -> TurnDetectionResult {
        // Get current audio buffer
        let audioBuffer = audioEngine.getCurrentBuffer()

        // Validate minimum audio length
        let audioDuration = Double(audioBuffer.count) / AudioCaptureEngine.targetSampleRate
        guard audioDuration >= Self.minimumAudioDuration else {
            throw TurnDetectionError.insufficientAudio
        }

        isProcessing = true
        defer { isProcessing = false }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Step 1: Extract Whisper mel-spectrogram features
        let features: [[Float]]
        do {
            features = try featureExtractor.extractFeatures(from: audioBuffer)
        } catch {
            print("❌ Feature extraction failed: \(error)")
            throw TurnDetectionError.featureExtractionFailed
        }

        // Validate feature dimensions
        guard features.count == 80, features.allSatisfy({ $0.count == 800 }) else {
            print("❌ Invalid feature dimensions: \(features.count) x \(features.first?.count ?? 0)")
            throw TurnDetectionError.featureExtractionFailed
        }

        // Step 2: Create ONNX Runtime tensor from features
        let inputTensor = try createORTTensor(from: features)

        // Step 3: Run inference (model already applies sigmoid internally)
        let probability = try runInference(with: inputTensor)

        let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000  // ms

        // Step 4: Determine if turn is complete
        let isTurnComplete = probability >= Self.turnCompleteThreshold

        // Create result (store probability as logit for compatibility)
        let result = TurnDetectionResult(
            logit: probability,  // Model outputs probability, not logit
            probability: probability,
            isTurnComplete: isTurnComplete,
            timestamp: Date(),
            audioBufferDuration: audioDuration,
            inferenceTimeMs: inferenceTime
        )

        // Update state
        lastResult = result
        updateInferenceHistory(inferenceTime)

        return result
    }

    /// Get average inference time from recent predictions
    var averageInferenceTime: Double {
        guard !inferenceHistory.isEmpty else { return 0 }
        return inferenceHistory.reduce(0, +) / Double(inferenceHistory.count)
    }

    // MARK: - Private Methods - ONNX Runtime

    /// Create ONNX Runtime tensor from 2D feature array
    /// - Parameter features: [[Float]] of shape [80, 800] (mel_bins, frames)
    /// - Returns: ORTValue tensor of shape [1, 80, 800] (batch, mel_bins, frames)
    private func createORTTensor(from features: [[Float]]) throws -> ORTValue {
        // Flatten features into 1D array: [batch=1, mels=80, frames=800]
        var flatArray: [Float] = []
        flatArray.reserveCapacity(1 * 80 * 800)

        for melRow in features {
            flatArray.append(contentsOf: melRow)
        }

        // Create NSMutableData from flat array
        let dataSize = flatArray.count * MemoryLayout<Float>.size
        let tensorData = NSMutableData(bytes: &flatArray, length: dataSize)

        // Create ORT tensor
        do {
            let ortValue = try ORTValue(
                tensorData: tensorData,
                elementType: .float,
                shape: Self.inputShape
            )
            return ortValue
        } catch {
            throw TurnDetectionError.tensorCreationFailed(error.localizedDescription)
        }
    }

    /// Run ONNX Runtime inference
    /// - Parameter inputTensor: Input tensor [1, 80, 800]
    /// - Returns: Probability value (model already applies sigmoid internally)
    private func runInference(with inputTensor: ORTValue) throws -> Float {
        do {
            // Run inference
            let outputs = try ortSession.run(
                withInputs: [Self.inputTensorName: inputTensor],
                outputNames: [Self.outputTensorName],
                runOptions: nil
            )

            // Extract output tensor
            guard let outputTensor = outputs[Self.outputTensorName] else {
                throw TurnDetectionError.inferenceFailed("Output tensor '\(Self.outputTensorName)' not found")
            }

            // Get tensor data
            let tensorData = try outputTensor.tensorData()

            // Extract probability (output shape is [1, 1], already sigmoid-ed)
            let probabilityPointer = tensorData.bytes.assumingMemoryBound(to: Float.self)
            let probability = probabilityPointer.pointee

            return probability

        } catch let error as TurnDetectionError {
            throw error
        } catch {
            throw TurnDetectionError.inferenceFailed(error.localizedDescription)
        }
    }

    /// Track inference performance
    private func updateInferenceHistory(_ time: Double) {
        inferenceHistory.append(time)
        if inferenceHistory.count > maxHistorySize {
            inferenceHistory.removeFirst()
        }
    }
}

// MARK: - Convenience Extensions

extension SmartTurnDetector {
    /// Run detection and update published state
    func detectTurnAndUpdate(completion: ((TurnDetectionResult?) -> Void)? = nil) {
        Task {
            do {
                let result = try await detectTurn()
                print("""
                    🎯 Turn Detection Result:
                       - Probability: \(result.probabilityPercentage)
                       - Turn Complete: \(result.isTurnComplete)
                       - Inference: \(String(format: "%.1f", result.inferenceTimeMs))ms
                       - Audio Duration: \(String(format: "%.2f", result.audioBufferDuration))s
                       - Raw Output: \(String(format: "%.3f", result.logit))
                    """)
                errorMessage = nil
                completion?(result)
            } catch {
                errorMessage = error.localizedDescription
                print("❌ Turn detection error: \(error)")
                completion?(nil)
            }
        }
    }

    /// Clear last detection result
    func clearResult() {
        lastResult = nil
    }
}

// MARK: - Performance Monitoring

extension SmartTurnDetector {
    /// Check if inference meets the 12ms target
    var meetsPerformanceTarget: Bool {
        averageInferenceTime <= 12.0
    }

    /// Performance status message
    var performanceStatus: String {
        let avg = averageInferenceTime
        if avg == 0 {
            return "No data"
        } else if avg <= 12 {
            return "Excellent (\(String(format: "%.1f", avg))ms)"
        } else if avg <= 20 {
            return "Good (\(String(format: "%.1f", avg))ms)"
        } else {
            return "Slow (\(String(format: "%.1f", avg))ms)"
        }
    }
}
