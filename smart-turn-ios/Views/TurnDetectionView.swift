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

    var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "mm:ss"
        return formatter.string(from: timestamp)
    }
}

struct TurnDetectionView: View {
    @StateObject private var audioEngine = AudioCaptureEngine()
    @StateObject private var detector: SmartTurnDetector

    @State private var showPermissionAlert = false
    @State private var silenceMonitorTimer: Timer?
    @State private var isStarting = false
    @State private var showInstructions = true

    // State history log
    @State private var stateLog: [StateLogEntry] = []

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
                    // Instructions (dismissible)
                    if showInstructions {
                        instructionsSection
                    }

                    // Header
                    headerSection

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

    private var instructionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("How to Use", systemImage: "info.circle.fill")
                    .font(.headline)
                    .foregroundColor(.blue)
                Spacer()
                Button("Dismiss") {
                    withAnimation {
                        showInstructions = false
                    }
                }
                .font(.caption)
            }

            VStack(alignment: .leading, spacing: 8) {
                instructionRow(number: "1", text: "Tap **Start** - app records last 8 seconds of audio")
                instructionRow(number: "2", text: "**Speak naturally** - fill buffer to at least 0.5s")
                instructionRow(number: "3", text: "**Pause for 1.5 seconds** - triggers detection")
                instructionRow(number: "4", text: "**Green** = Turn change detected, **Orange** = Not detected")
            }
            .font(.subheadline)

            VStack(alignment: .leading, spacing: 4) {
                Text("📊 **Audio Buffer**: Rolling 8-second window of your speech")
                    .font(.caption2)
                Text("📜 **State History**: Timestamped log of all detection events")
                    .font(.caption2)
            }
            .foregroundColor(.secondary)
            .padding(.top, 4)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }

    private func instructionRow(number: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(number)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .frame(width: 20, height: 20)
                .background(Circle().fill(Color.blue))
            Text(try! AttributedString(markdown: text))
                .foregroundColor(.primary)
        }
    }

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
        VStack(alignment: .leading, spacing: 12) {
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
                .frame(height: 250)
                .background(Color.black.opacity(0.05))
                .cornerRadius(8)
                .onChange(of: stateLog.count) { _ in
                    if let lastEntry = stateLog.last {
                        withAnimation {
                            proxy.scrollTo(lastEntry.id, anchor: .bottom)
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
            // Start/Stop Recording (only button needed)
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
            // Check permission (this may take a few seconds on first launch)
            let hasPermission = await audioEngine.requestMicrophonePermission()

            await MainActor.run {
                if hasPermission {
                    do {
                        try audioEngine.startCapture()
                        isStarting = false
                        addLog("🎙️ Recording started", level: .success)

                        // Start monitoring for silence to trigger detection
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
                    print("✅ TRIGGERING DETECTION")
                    hasDetectedThisSilence = true  // Prevent re-triggering during same silence

                    // Run detection and log result after completion
                    detector.detectTurnAndUpdate { result in
                        guard let result = result else { return }

                        let resultText = result.isTurnComplete
                            ? "✅ Turn change detected (\(result.probabilityPercentage))"
                            : "⏳ Turn change not detected (\(result.probabilityPercentage))"

                        self.addLog(resultText, level: result.isTurnComplete ? .success : .warning)

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
