//
//  WhisperFeatureExtractor.swift
//  meh
//
//  Whisper-compatible mel-spectrogram feature extraction for audio preprocessing.
//
//  This implementation matches the Transformers WhisperFeatureExtractor:
//  - 16kHz sample rate
//  - 80 mel filterbanks
//  - 400-sample window (25ms)
//  - 160-sample hop (10ms)
//  - Hann window
//  - Normalization to [-1, 1] range
//

import Foundation
import Accelerate

/// Errors that can occur during feature extraction
enum FeatureExtractionError: Error {
    case invalidAudioLength
    case fftSetupFailed
    case invalidParameters

    var localizedDescription: String {
        switch self {
        case .invalidAudioLength:
            return "Audio length must be > 0 and <= \(WhisperFeatureExtractor.maxAudioSamples) samples (~\(String(format: "%.2f", WhisperFeatureExtractor.maxAudioSeconds))s at 16kHz)"
        case .fftSetupFailed:
            return "Failed to create FFT setup"
        case .invalidParameters:
            return "Invalid feature extraction parameters"
        }
    }
}

/// Whisper-compatible feature extractor for converting audio to mel-spectrograms
final class WhisperFeatureExtractor {

    // MARK: - Constants (matching Whisper Tiny configuration)

    /// Sample rate for input audio (16kHz)
    static let sampleRate: Int = 16_000

    /// Number of mel filterbanks
    static let nMels: Int = 80

    /// FFT size (next power of 2 >= 400)
    static let nFFT: Int = 512

    /// Window size in samples (25ms at 16kHz)
    static let windowSize: Int = 400

    /// Hop size in samples (10ms at 16kHz)
    static let hopSize: Int = 160

    /// Target number of frames for model input (800 frames)
    static let targetFrames: Int = 800

    /// Maximum audio length in samples (calculated to produce exactly 800 frames)
    /// Formula: (targetFrames - 1) * hopSize + windowSize
    /// = (800 - 1) * 160 + 400 = 128,240 samples
    static let maxAudioSamples: Int = (targetFrames - 1) * hopSize + windowSize

    /// Maximum audio length in seconds
    static let maxAudioSeconds: Double = Double(maxAudioSamples) / Double(sampleRate)

    // MARK: - Properties

    private let fftSetup: vDSP_DFT_Setup
    private let melFilterbank: [[Float]]
    private let hannWindow: [Float]

    // Pre-allocated buffers for feature extraction (reused across calls)
    private var paddedAudioBuffer: [Float]
    private var fftRealBuffer: [Float]
    private var fftImagBuffer: [Float]

    // MARK: - Initialization

    init() throws {
        // Create FFT setup for forward transform
        guard let setup = vDSP_DFT_zrop_CreateSetup(
            nil,
            vDSP_Length(Self.nFFT),
            .FORWARD
        ) else {
            throw FeatureExtractionError.fftSetupFailed
        }
        self.fftSetup = setup

        // Create Hann window
        self.hannWindow = Self.createHannWindow(size: Self.windowSize)

        // Create mel filterbank
        self.melFilterbank = Self.createMelFilterbank(
            sampleRate: Self.sampleRate,
            nFFT: Self.nFFT,
            nMels: Self.nMels
        )

        // Pre-allocate buffers for reuse
        // This significantly reduces memory allocations during real-time processing
        self.paddedAudioBuffer = [Float](repeating: 0, count: Self.maxAudioSamples)
        self.fftRealBuffer = [Float](repeating: 0, count: Self.nFFT)
        self.fftImagBuffer = [Float](repeating: 0, count: Self.nFFT)
    }

    deinit {
        vDSP_DFT_DestroySetup(fftSetup)
    }

    // MARK: - Public API

    /// Extract mel-spectrogram features from audio samples
    ///
    /// - Parameter audioSamples: Float array of audio samples at 16kHz
    /// - Returns: 2D array of shape [nMels=80, nFrames=800] containing mel-spectrogram features
    /// - Throws: FeatureExtractionError if audio is invalid
    func extractFeatures(from audioSamples: [Float]) throws -> [[Float]] {
        // Validate input length
        guard !audioSamples.isEmpty && audioSamples.count <= Self.maxAudioSamples else {
            throw FeatureExtractionError.invalidAudioLength
        }

        // Validate audio samples for NaN/Inf
        // This prevents corrupt audio from causing undefined behavior in FFT
        for sample in audioSamples {
            guard sample.isFinite else {
                print("❌ Invalid audio sample detected (NaN/Inf) - rejecting buffer")
                throw FeatureExtractionError.invalidParameters
            }
        }

        // Step 1: Pad or truncate audio to exactly 8 seconds (128,000 samples)
        let paddedAudio = padOrTruncate(audioSamples, targetLength: Self.maxAudioSamples)

        // Step 2: Compute STFT (Short-Time Fourier Transform)
        let stft = computeSTFT(paddedAudio)

        // Step 3: Compute power spectrogram (magnitude squared)
        let powerSpec = computePowerSpectrogram(stft)

        // Step 4: Apply mel filterbank
        let melSpec = applyMelFilterbank(powerSpec)

        // Step 5: Convert to log scale (with small epsilon to avoid log(0))
        let logMelSpec = applyLogScale(melSpec)

        // Step 6: Normalize to approximately [-1, 1] range (matching Whisper preprocessing)
        let normalizedSpec = normalizeFeatures(logMelSpec)

        return normalizedSpec
    }

    // MARK: - Private Methods

    /// Pad audio to target length with zeros at the beginning, or truncate from the end
    /// Uses pre-allocated buffer to avoid memory allocation
    private func padOrTruncate(_ audio: [Float], targetLength: Int) -> [Float] {
        // Reuse pre-allocated buffer
        paddedAudioBuffer.withUnsafeMutableBufferPointer { buffer in
            if audio.count >= targetLength {
                // Truncate: copy last targetLength samples
                let startIdx = audio.count - targetLength
                audio.withUnsafeBufferPointer { audioBuffer in
                    buffer.baseAddress?.update(
                        from: audioBuffer.baseAddress! + startIdx,
                        count: targetLength
                    )
                }
            } else {
                // Pad: zeros at beginning, then audio
                let padSize = targetLength - audio.count
                // Zero out padding region
                buffer.baseAddress?.update(repeating: 0, count: padSize)
                // Copy audio after padding
                audio.withUnsafeBufferPointer { audioBuffer in
                    (buffer.baseAddress! + padSize).update(
                        from: audioBuffer.baseAddress!,
                        count: audio.count
                    )
                }
            }
        }

        return Array(paddedAudioBuffer.prefix(targetLength))
    }

    /// Compute Short-Time Fourier Transform
    /// Returns array of complex spectra, one per frame
    private func computeSTFT(_ audio: [Float]) -> [[DSPComplex]] {
        let numFrames = (audio.count - Self.windowSize) / Self.hopSize + 1
        var frames: [[DSPComplex]] = []

        for frameIndex in 0..<numFrames {
            let startIdx = frameIndex * Self.hopSize
            let endIdx = min(startIdx + Self.windowSize, audio.count)

            // Extract frame and apply Hann window
            var frame = Array(audio[startIdx..<endIdx])
            if frame.count < Self.windowSize {
                frame.append(contentsOf: [Float](repeating: 0, count: Self.windowSize - frame.count))
            }

            vDSP_vmul(frame, 1, hannWindow, 1, &frame, 1, vDSP_Length(Self.windowSize))

            // Pad to FFT size
            if frame.count < Self.nFFT {
                frame.append(contentsOf: [Float](repeating: 0, count: Self.nFFT - frame.count))
            }

            // Compute FFT
            let spectrum = computeFFT(frame)
            frames.append(spectrum)
        }

        return frames
    }

    /// Compute FFT for a single frame
    /// Uses pre-allocated buffers to avoid repeated allocations
    private func computeFFT(_ frame: [Float]) -> [DSPComplex] {
        // Copy frame to real buffer (reuse pre-allocated buffer)
        fftRealBuffer.withUnsafeMutableBufferPointer { realBuffer in
            frame.withUnsafeBufferPointer { frameBuffer in
                realBuffer.baseAddress?.update(
                    from: frameBuffer.baseAddress!,
                    count: min(frame.count, Self.nFFT)
                )
            }
        }

        // Zero out imaginary buffer
        fftImagBuffer.withUnsafeMutableBufferPointer { imagBuffer in
            imagBuffer.baseAddress?.update(repeating: 0, count: Self.nFFT)
        }

        // Perform FFT in-place
        var realOutput = [Float](repeating: 0, count: Self.nFFT)
        var imaginaryOutput = [Float](repeating: 0, count: Self.nFFT)

        vDSP_DFT_Execute(
            fftSetup,
            &fftRealBuffer,
            &fftImagBuffer,
            &realOutput,
            &imaginaryOutput
        )

        // Convert to DSPComplex array (only first half due to symmetry)
        let numBins = Self.nFFT / 2 + 1
        return (0..<numBins).map { i in
            DSPComplex(real: realOutput[i], imag: imaginaryOutput[i])
        }
    }

    /// Compute power spectrogram (magnitude squared)
    private func computePowerSpectrogram(_ stft: [[DSPComplex]]) -> [[Float]] {
        return stft.map { frame in
            frame.map { complex in
                // Power = real^2 + imag^2
                complex.real * complex.real + complex.imag * complex.imag
            }
        }
    }

    /// Apply mel filterbank to power spectrogram
    private func applyMelFilterbank(_ powerSpec: [[Float]]) -> [[Float]] {
        // powerSpec shape: [numFrames, nFFT/2 + 1]
        // melFilterbank shape: [nMels, nFFT/2 + 1]
        // output shape: [nMels, numFrames]

        var melSpec = [[Float]](repeating: [Float](repeating: 0, count: powerSpec.count), count: Self.nMels)

        for (melIdx, melFilter) in melFilterbank.enumerated() {
            for (frameIdx, frame) in powerSpec.enumerated() {
                // Dot product: sum(melFilter * frame)
                var result: Float = 0
                vDSP_dotpr(melFilter, 1, frame, 1, &result, vDSP_Length(min(melFilter.count, frame.count)))
                melSpec[melIdx][frameIdx] = result
            }
        }

        return melSpec
    }

    /// Apply log scale with small epsilon
    private func applyLogScale(_ melSpec: [[Float]]) -> [[Float]] {
        let epsilon: Float = 1e-10
        return melSpec.map { row in
            row.map { value in
                log10(max(value, epsilon))
            }
        }
    }

    /// Normalize features to approximately [-1, 1] range
    /// Matches Whisper's normalization strategy
    private func normalizeFeatures(_ logMelSpec: [[Float]]) -> [[Float]] {
        // Compute global mean and std
        let allValues = logMelSpec.flatMap { $0 }
        var mean: Float = 0
        var stdDev: Float = 0

        vDSP_normalize(allValues, 1, nil, 1, &mean, &stdDev, vDSP_Length(allValues.count))

        // Normalize: (x - mean) / (std + epsilon)
        let epsilon: Float = 1e-8
        return logMelSpec.map { row in
            row.map { value in
                (value - mean) / (stdDev + epsilon)
            }
        }
    }

    // MARK: - Static Helper Methods

    /// Create Hann window for smoothing frames
    private static func createHannWindow(size: Int) -> [Float] {
        var window = [Float](repeating: 0, count: size)
        vDSP_hann_window(&window, vDSP_Length(size), Int32(vDSP_HANN_NORM))
        return window
    }

    /// Create mel filterbank matrix
    /// Returns array of shape [nMels, nFFT/2 + 1]
    private static func createMelFilterbank(sampleRate: Int, nFFT: Int, nMels: Int) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1

        // Mel scale conversion functions
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        // Create mel points spaced evenly in mel scale
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sampleRate) / 2.0)
        let melPoints = (0...nMels + 1).map { i in
            melToHz(minMel + Float(i) * (maxMel - minMel) / Float(nMels + 1))
        }

        // Convert mel points to FFT bin indices
        let fftFreqs = (0..<nFreqs).map { Float($0) * Float(sampleRate) / Float(nFFT) }

        // Build filterbank
        var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nMels)

        for m in 0..<nMels {
            let leftMel = melPoints[m]
            let centerMel = melPoints[m + 1]
            let rightMel = melPoints[m + 2]

            for (f, freq) in fftFreqs.enumerated() {
                if freq >= leftMel && freq <= centerMel {
                    // Rising edge
                    filterbank[m][f] = (freq - leftMel) / (centerMel - leftMel)
                } else if freq > centerMel && freq <= rightMel {
                    // Falling edge
                    filterbank[m][f] = (rightMel - freq) / (rightMel - centerMel)
                }
            }
        }

        return filterbank
    }
}

// MARK: - DSPComplex Extension

/// Simple complex number type for FFT operations
struct DSPComplex {
    var real: Float
    var imag: Float
}
