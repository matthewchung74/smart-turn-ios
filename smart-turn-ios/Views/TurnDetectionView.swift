//
//  TurnDetectionView.swift
//  meh
//
//  Main UI for real-time turn detection with audio visualization.
//

import SwiftUI
import Accelerate

/// Log entry for state history
struct StateLogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let message: String
    let level: LogLevel

    enum LogLevel {
        case info      // Normal state changes
        case success   // Detection completed
        case warning   // Cooldown, insufficient buffer
        case error     // Errors

        var color: Color {
            switch self {
            case .info: return .primary
            case .success: return .green
            case .warning: return .orange
            case .error: return .red
            }
        }
    }

    // Cached DateFormatter for performance (DateFormatter initialization is expensive)
    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "mm:ss"
        return formatter
    }()

    var timeString: String {
        Self.timeFormatter.string(from: timestamp)
    }
}

struct TurnDetectionView: View {
    @StateObject private var audioEngine = AudioCaptureEngine()
    @StateObject private var detector: SmartTurnDetector

    @State private var showPermissionAlert = false
    @State private var silenceMonitorTimer: Timer?
    @State private var isStarting = false

    // State history log
    @State private var stateLog: [StateLogEntry] = []

    // Accumulated transcription with line breaks
    @State private var accumulatedTranscription: String = ""
    @State private var lastSegmentText: String = ""

    // Silence detection state
    @State private var silenceStartTime: Date?
    @State private var hasDetectedThisSilence = false  // Prevents re-triggering during same silence period
    @State private var resultDisplayTimer: Timer?

    // Silence detection thresholds
    private let silenceThreshold: Float = 0.005  // RMS threshold for silence
    private let silenceDuration: TimeInterval = 1.5  // 1.5 seconds of silence triggers detection
    private let minimumBufferForDetection: Double = 0.5  // 0.5s minimum buffer

    init() {
        let engine = AudioCaptureEngine()
        _audioEngine = StateObject(wrappedValue: engine)

        // Initialize detector with failable initializer
        guard let detector = SmartTurnDetector(audioEngine: engine) else {
            fatalError("❌ CRITICAL: SmartTurnDetector initialization failed. Check that smart-turn-v3.0.onnx is included in the app bundle.")
        }
        _detector = StateObject(wrappedValue: detector)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Turn Detection Indicator (MAIN FOCUS)
                    turnIndicatorSection

                    // Buffer Status
                    bufferStatusSection

                    // Audio Level Visualization
                    audioLevelSection

                    // Controls
                    controlsSection
                }
                .padding()
            }
            .onAppear {
                print("✅ App launched - Speech recognition will be enabled when you press Start")
            }
            .alert("Microphone Permission Required", isPresented: $showPermissionAlert) {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This app needs microphone access to detect turn completion. Please grant permission in Settings.")
            }
        }
    }

    // MARK: - View Components

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 60))
                .foregroundStyle(recordingStatusColor)
                .symbolEffect(.pulse, isActive: audioEngine.isRecording)

            if isStarting {
                ProgressView()
                    .padding(.top, 8)
                Text("Starting audio...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                Text(audioEngine.isRecording ? "Listening..." : "Ready")
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.top)
    }

    private var audioLevelSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Audio Level")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Shows mic input volume")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Audio level meter
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Capsule()
                        .fill(Color.gray.opacity(0.2))

                    // Level indicator
                    Capsule()
                        .fill(audioLevelGradient)
                        .frame(width: geometry.size.width * CGFloat(audioEngine.normalizedAudioLevel))
                        .animation(.easeOut(duration: 0.1), value: audioEngine.normalizedAudioLevel)
                }
            }
            .frame(height: 20)

            // dB value
            Text(String(format: "%.1f dB", audioEngine.rmsToDecibels(audioEngine.audioLevel)))
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var bufferStatusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Audio Buffer")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%.2fs / 8.0s", audioEngine.bufferDuration))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Buffer progress
            ProgressView(value: audioEngine.bufferDuration, total: 8.0)
                .tint(.blue)

            Text("Rolling window of recent speech (keeps last 8s, needs ≥0.5s to detect)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    private var turnIndicatorSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Transcription Box
            VStack(alignment: .leading, spacing: 8) {
                Text("Transcription")
                    .font(.headline)
                    .foregroundColor(.secondary)

                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            // Show accumulated transcription (completed segments)
                            if !accumulatedTranscription.isEmpty {
                                Text(accumulatedTranscription)
                                    .font(.body)
                                    .foregroundColor(.primary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }

                            // Show current live transcription (in progress)
                            if !detector.speechRecognitionManager.transcribedText.isEmpty {
                                Text(detector.speechRecognitionManager.transcribedText)
                                    .font(.body)
                                    .foregroundColor(.blue)  // Blue for live/in-progress
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }

                            // Placeholder when empty
                            if accumulatedTranscription.isEmpty && detector.speechRecognitionManager.transcribedText.isEmpty {
                                Text("Transcription will appear here...")
                                    .font(.body)
                                    .foregroundColor(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }

                            // Invisible anchor for auto-scroll
                            Color.clear
                                .frame(height: 1)
                                .id("bottom")
                        }
                        .padding(12)
                    }
                    .frame(height: 100)
                    .background(Color.black.opacity(0.05))
                    .cornerRadius(8)
                    .onChange(of: detector.speechRecognitionManager.transcribedText) {
                        // Auto-scroll to bottom when transcription updates
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                    .onChange(of: accumulatedTranscription) {
                        // Auto-scroll when new segment is added
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
            }

            // State History
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("State History")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Spacer()
                    if !stateLog.isEmpty {
                        Button("Clear") {
                            stateLog.removeAll()
                        }
                        .font(.caption)
                    }
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            if stateLog.isEmpty {
                                Text("Press Start to begin...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding()
                            } else {
                                ForEach(stateLog) { entry in
                                    HStack(alignment: .top, spacing: 8) {
                                        Text("[\(entry.timeString)]")
                                            .font(.system(.caption, design: .monospaced))
                                            .foregroundColor(.secondary)

                                        Text(entry.message)
                                            .font(.caption)
                                            .foregroundColor(entry.level.color)
                                    }
                                    .id(entry.id)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                    }
                    .frame(height: 200)
                    .background(Color.black.opacity(0.05))
                    .cornerRadius(8)
                    .onChange(of: stateLog.count) {
                        if let lastEntry = stateLog.last {
                            withAnimation {
                                proxy.scrollTo(lastEntry.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.05))
        .cornerRadius(16)
    }

    private var controlsSection: some View {
        VStack(spacing: 16) {
            // Speech Recognition Status
            HStack {
                Image(systemName: "apple.logo")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Real-time transcription")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                if detector.speechRecognitionManager.isAuthorized {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.caption)
                } else {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                        .font(.caption)
                }
            }
            .padding(.vertical, 4)

            // Start/Stop Recording Button
            Button {
                handleRecordingToggle()
            } label: {
                if isStarting {
                    HStack {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.white)
                        Text("Starting...")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                } else {
                    Label(
                        audioEngine.isRecording ? "Stop" : "Start",
                        systemImage: audioEngine.isRecording ? "stop.circle.fill" : "mic.circle.fill"
                    )
                    .font(.title3)
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(audioEngine.isRecording ? .red : .green)
            .disabled(isStarting)
            .controlSize(.large)
        }
    }

    // MARK: - Computed Properties

    private var recordingStatusColor: Color {
        if detector.isProcessing {
            return .blue
        } else if audioEngine.isRecording {
            return .green
        } else {
            return .gray
        }
    }

    private var audioLevelGradient: LinearGradient {
        LinearGradient(
            colors: [.green, .yellow, .orange, .red],
            startPoint: .leading,
            endPoint: .trailing
        )
    }

    // MARK: - Logging

    private func addLog(_ message: String, level: StateLogEntry.LogLevel = .info) {
        let entry = StateLogEntry(timestamp: Date(), message: message, level: level)
        stateLog.append(entry)

        // Keep only last 100 entries to prevent memory issues
        if stateLog.count > 100 {
            stateLog.removeFirst(stateLog.count - 100)
        }
    }

    // MARK: - Actions

    private func handleRecordingToggle() {
        if audioEngine.isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        isStarting = true

        Task {
            // First, enable speech recognition and request permission if needed
            await MainActor.run {
                if !detector.speechRecognitionManager.isEnabled {
                    detector.speechRecognitionManager.isEnabled = true
                }
            }

            // Wait a moment for authorization to complete if needed
            if !detector.speechRecognitionManager.isAuthorized {
                await detector.speechRecognitionManager.requestAuthorization()
            }

            // Check microphone permission (this may take a few seconds on first launch)
            let hasPermission = await audioEngine.requestMicrophonePermission()

            await MainActor.run {
                if hasPermission {
                    do {
                        try audioEngine.startCapture()
                        isStarting = false
                        addLog("🎙️ Recording started", level: .success)

                        // Start Apple Speech recognition if authorized
                        if detector.speechRecognitionManager.isAuthorized {
                            do {
                                try detector.speechRecognitionManager.startRecognition()
                                addLog("🎙️ Real-time transcription started", level: .success)
                            } catch {
                                addLog("❌ Transcription failed: \(error.localizedDescription)", level: .error)
                                print("❌ Speech recognition error: \(error)")
                            }
                        } else {
                            addLog("⚠️ Speech recognition not authorized", level: .warning)
                        }

                        // Start monitoring for silence to trigger turn detection
                        startSilenceMonitoring()
                    } catch {
                        print("❌ Failed to start capture: \(error)")
                        detector.errorMessage = error.localizedDescription
                        isStarting = false
                    }
                } else {
                    isStarting = false
                    showPermissionAlert = true
                }
            }
        }
    }

    private func stopRecording() {
        audioEngine.stopCapture()
        stopSilenceMonitoring()

        // Stop Apple Speech recognition
        if detector.speechRecognitionManager.isRecognizing {
            detector.speechRecognitionManager.stopRecognition()
            addLog("⏹️ Transcription stopped", level: .info)
        }

        // Clear accumulated transcription
        accumulatedTranscription = ""
        lastSegmentText = ""

        addLog("⏹️ Recording stopped", level: .info)
    }

    private func startSilenceMonitoring() {
        // Monitor audio levels every 0.1 seconds to detect silence
        silenceMonitorTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            monitorForSilence()
        }
    }

    private func stopSilenceMonitoring() {
        silenceMonitorTimer?.invalidate()
        silenceMonitorTimer = nil
        silenceStartTime = nil
        hasDetectedThisSilence = false
        detector.clearResult()
    }

    private func monitorForSilence() {
        // Calculate RMS from the most recent 0.1s of audio (1600 samples @ 16kHz)
        let samples = audioEngine.getCurrentBuffer()
        guard !samples.isEmpty else { return }

        let recentSamples = samples.suffix(1600)  // Last 0.1s (ArraySlice - no copy)
        var rms: Float = 0
        recentSamples.withUnsafeBufferPointer { buffer in
            vDSP_rmsqv(buffer.baseAddress!, 1, &rms, vDSP_Length(buffer.count))
        }

        let currentLevel = rms
        let currentDB = audioEngine.rmsToDecibels(currentLevel)
        let isSilent = currentLevel < silenceThreshold

        // Debug: Log audio level and state
        print("🔊 Audio: \(String(format: "%.1f", currentDB)) dB (RMS: \(String(format: "%.4f", currentLevel))) | Silent: \(isSilent) | Buffer: \(String(format: "%.1f", audioEngine.bufferDuration))s")

        if isSilent {
            // Start silence tracking if not already started
            if silenceStartTime == nil && audioEngine.bufferDuration >= minimumBufferForDetection {
                silenceStartTime = Date()
                hasDetectedThisSilence = false  // Reset flag for new silence period
                print("🟡 Silence started (buffer ready)")
            }

            // Check if we've been silent long enough to trigger detection
            if let silenceStart = silenceStartTime {
                let silenceDuration = Date().timeIntervalSince(silenceStart)
                print("⏱️  Silence duration: \(String(format: "%.1f", silenceDuration))s")

                // Trigger detection once per silence period (talking → 1s silence transition)
                if silenceDuration >= self.silenceDuration && !hasDetectedThisSilence {
                    print("✅ TRIGGERING TURN DETECTION")
                    hasDetectedThisSilence = true  // Prevent re-triggering during same silence

                    // Run turn detection only (streaming handles transcription)
                    detector.detectTurnAndUpdate { result in
                        guard let result = result else { return }

                        // Log result without utterance (transcription shown separately)
                        let resultText = result.isTurnComplete
                            ? "✅ Turn change detected (\(result.probabilityPercentage))"
                            : "⏳ Turn change not detected (\(result.probabilityPercentage))"

                        self.addLog(resultText, level: result.isTurnComplete ? .success : .warning)

                        // Add current transcription segment to accumulated text with separator
                        let utterance = self.detector.speechRecognitionManager.transcribedText
                        if !utterance.isEmpty && utterance != self.lastSegmentText {
                            if !self.accumulatedTranscription.isEmpty {
                                self.accumulatedTranscription += "\n----------\n"
                            }
                            self.accumulatedTranscription += utterance
                            self.lastSegmentText = utterance

                            // Restart speech recognition to get fresh segment (with delay for audio engine)
                            if self.detector.speechRecognitionManager.isRecognizing {
                                self.detector.speechRecognitionManager.stopRecognition()

                                // Wait for audio engine to fully stop before restarting
                                Task { @MainActor in
                                    try? await Task.sleep(nanoseconds: 500_000_000) // 500ms delay
                                    do {
                                        try self.detector.speechRecognitionManager.startRecognition()
                                        print("✅ Speech recognition restarted successfully")
                                    } catch {
                                        print("❌ Failed to restart recognition: \(error)")
                                    }
                                }
                            }
                        }

                        // Auto-clear result after 3 seconds
                        self.resultDisplayTimer?.invalidate()
                        self.resultDisplayTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { _ in
                            Task { @MainActor in
                                self.detector.clearResult()
                                print("🔄 Auto-cleared result")
                                self.addLog("🔄 Result cleared", level: .info)
                            }
                        }
                    }
                }
            }
        } else {
            // Speaking detected - reset silence tracking
            if silenceStartTime != nil {
                print("🟢 Speaking detected")
                addLog("🗣️ Speaking detected", level: .info)
            }

            silenceStartTime = nil
            hasDetectedThisSilence = false

            // Cancel auto-clear timer and clear result
            resultDisplayTimer?.invalidate()
            Task { @MainActor in
                detector.clearResult()
            }
        }
    }
}

#Preview {
    TurnDetectionView()
}
