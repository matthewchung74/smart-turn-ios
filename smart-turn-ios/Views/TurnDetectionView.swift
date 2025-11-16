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

/// State machine for recording lifecycle
enum RecordingState: Equatable {
    case idle                           // Not recording, ready to start
    case starting                       // Requesting permissions and initializing
    case recording                      // Active: audio capture + speech recognition + silence monitoring
    case detectingTurn                  // Running ONNX turn detection inference
    case cooldown                       // Post-detection cooldown (prevents re-triggering during same silence)
    case stopping                       // Cleaning up resources
    case error(String)                  // Error state with message

    var description: String {
        switch self {
        case .idle: return "Idle"
        case .starting: return "Starting..."
        case .recording: return "Recording"
        case .detectingTurn: return "Detecting Turn"
        case .cooldown: return "Cooldown"
        case .stopping: return "Stopping..."
        case .error(let message): return "Error: \(message)"
        }
    }

    /// Valid state transitions
    func canTransition(to newState: RecordingState) -> Bool {
        switch (self, newState) {
        // From idle
        case (.idle, .starting): return true

        // From starting
        case (.starting, .recording): return true
        case (.starting, .error): return true

        // From recording
        case (.recording, .detectingTurn): return true
        case (.recording, .stopping): return true
        case (.recording, .error): return true

        // From detectingTurn
        case (.detectingTurn, .cooldown): return true
        case (.detectingTurn, .recording): return true  // If no cooldown needed
        case (.detectingTurn, .stopping): return true
        case (.detectingTurn, .error): return true

        // From cooldown
        case (.cooldown, .recording): return true  // Speaking detected, exit cooldown
        case (.cooldown, .stopping): return true
        case (.cooldown, .error): return true

        // From stopping
        case (.stopping, .idle): return true
        case (.stopping, .error): return true

        // From error
        case (.error, .idle): return true
        case (.error, .starting): return true

        // Self-transition allowed
        case _ where self == newState: return true

        default: return false
        }
    }
}

struct TurnDetectionView: View {
    @StateObject private var audioEngine = AudioCaptureEngine()
    @StateObject private var detector: SmartTurnDetector

    // State machine with version tracking for race condition prevention
    @State private var recordingState: RecordingState = .idle
    @State private var stateVersion: Int = 0  // Incremented on each state change

    // Timing measurement for loading indicator
    @State private var startingTimestamp: Date?

    // State transition method (validates before mutation)
    private func transitionTo(_ newState: RecordingState) {
        guard recordingState.canTransition(to: newState) else {
            print("⚠️ Invalid state transition: \(recordingState.description) → \(newState.description)")
            return
        }

        let oldState = recordingState
        recordingState = newState
        stateVersion += 1  // Increment version to invalidate old async callbacks
        print("🔄 State transition: \(oldState.description) → \(newState.description) (v\(stateVersion))")
        addLog("State: \(newState.description)", level: .info)

        // Track loading indicator timing
        if newState == .starting {
            startingTimestamp = Date()
            print("⏱️ [TIMING] Loading indicator START")
        } else if newState == .recording, let startTime = startingTimestamp {
            let duration = Date().timeIntervalSince(startTime) * 1000  // Convert to ms
            print("⏱️ [TIMING] Loading indicator END - Duration: \(String(format: "%.0f", duration))ms")
            startingTimestamp = nil
        }
    }

    @State private var showPermissionAlert = false
    @State private var silenceMonitorTimer: Timer?

    // State history log
    @State private var stateLog: [StateLogEntry] = []

    // Accumulated transcription with line breaks
    @State private var accumulatedTranscription: String = ""
    @State private var lastSegmentText: String = ""

    // Silence detection state
    @State private var silenceStartTime: Date?
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
                .symbolEffect(.pulse, isActive: recordingState == .recording || recordingState == .cooldown)

            if recordingState == .starting {
                ProgressView()
                    .padding(.top, 8)
                Text("Starting audio...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                // Show "Listening..." for both Recording and Cooldown states
                let isActivelyRecording = recordingState == .recording || recordingState == .cooldown
                Text(isActivelyRecording ? "Listening..." : recordingState.description)
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

                            // Placeholder when empty - show loading state during startup
                            if accumulatedTranscription.isEmpty && detector.speechRecognitionManager.transcribedText.isEmpty {
                                if recordingState == .starting {
                                    HStack(spacing: 12) {
                                        ProgressView()
                                            .progressViewStyle(.circular)
                                        Text("Initializing speech recognition...")
                                            .font(.body)
                                            .foregroundColor(.secondary)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                } else if recordingState == .recording || recordingState == .cooldown {
                                    Text("Speak to see transcription...")
                                        .font(.body)
                                        .foregroundColor(.secondary)
                                        .italic()
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                } else {
                                    Text("Transcription will appear here...")
                                        .font(.body)
                                        .foregroundColor(.secondary)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
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
                let timestamp = Date().timeIntervalSince1970
                print("👆 [TAP] Button tapped at \(timestamp), state: \(recordingState.description), disabled: \(recordingState == .starting || recordingState == .stopping)")
                handleRecordingToggle()
                print("👆 [TAP] handleRecordingToggle() returned")
            } label: {
                if recordingState == .starting || recordingState == .stopping {
                    HStack {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.white)
                        Text(recordingState.description)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                } else {
                    // Show "Stop" for both Recording and Cooldown (we're still actively recording)
                    let isActivelyRecording = recordingState == .recording || recordingState == .cooldown
                    Label(
                        isActivelyRecording ? "Stop" : "Start",
                        systemImage: isActivelyRecording ? "stop.circle.fill" : "mic.circle.fill"
                    )
                    .font(.title3)
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(recordingState == .recording || recordingState == .cooldown ? .red : .green)
            .disabled(recordingState == .starting || recordingState == .stopping)
            .controlSize(.large)
        }
    }

    // MARK: - Computed Properties

    private var recordingStatusColor: Color {
        switch recordingState {
        case .idle:
            return .gray
        case .starting, .stopping:
            return .orange
        case .recording:
            return .green
        case .detectingTurn:
            return .blue
        case .cooldown:
            return .yellow
        case .error:
            return .red
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
        print("🔘 [TOGGLE] Button pressed, current state: \(recordingState.description)")

        switch recordingState {
        case .idle, .error:
            print("🔘 [TOGGLE] Calling startRecording()")
            startRecording()
        case .recording, .detectingTurn, .cooldown:
            print("🔘 [TOGGLE] Calling stopRecording()")
            stopRecording()
        case .starting, .stopping:
            // Already transitioning, ignore
            print("🔘 [TOGGLE] Ignoring - already in transitioning state")
            break
        }
    }

    private func startRecording() {
        // Guard against multiple simultaneous starts (race condition before button disables)
        switch recordingState {
        case .idle, .error:
            break  // OK to start
        default:
            print("⚠️ [START] Ignoring duplicate start request, current state: \(recordingState.description)")
            addLog("⚠️ Already starting/recording, ignoring", level: .warning)
            return
        }

        print("🎬 [START] Beginning start sequence from state: \(recordingState.description)")

        // Transition to starting state
        transitionTo(.starting)

        Task {
            print("🔄 [START] Task started, requesting speech recognition...")

            // First, enable speech recognition and request permission if needed
            await MainActor.run {
                if !detector.speechRecognitionManager.isEnabled {
                    print("🔄 [START] Enabling speech recognition...")
                    detector.speechRecognitionManager.isEnabled = true
                }
            }

            // Wait a moment for authorization to complete if needed
            if !detector.speechRecognitionManager.isAuthorized {
                print("🔄 [START] Requesting speech recognition authorization...")
                await detector.speechRecognitionManager.requestAuthorization()
                print("✅ [START] Speech authorization complete")
            }

            // Check microphone permission (this may take a few seconds on first launch)
            print("🔄 [START] Requesting microphone permission...")
            let hasPermission = await audioEngine.requestMicrophonePermission()
            print("✅ [START] Microphone permission: \(hasPermission)")

            await MainActor.run {
                guard hasPermission else {
                    print("❌ [START] Microphone permission denied, transitioning to error")
                    transitionTo(.error("Microphone permission denied"))
                    showPermissionAlert = true
                    return
                }

                print("🔄 [START] Starting audio capture...")

                do {
                    // Start audio capture
                    try audioEngine.startCapture()
                    print("✅ [START] Audio capture started")
                    addLog("🎙️ Recording started", level: .success)

                    // Start Apple Speech recognition if authorized
                    if detector.speechRecognitionManager.isAuthorized {
                        print("🔄 [START] Starting speech recognition...")
                        do {
                            try detector.speechRecognitionManager.startRecognition()
                            print("✅ [START] Speech recognition started")
                            addLog("🎙️ Real-time transcription started", level: .success)
                        } catch {
                            print("❌ [START] Speech recognition failed: \(error)")
                            addLog("❌ Transcription failed: \(error.localizedDescription)", level: .error)
                        }
                    } else {
                        print("⚠️ [START] Speech recognition not authorized")
                        addLog("⚠️ Speech recognition not authorized", level: .warning)
                    }

                    // Start monitoring for silence to trigger turn detection
                    print("🔄 [START] Starting silence monitoring...")
                    startSilenceMonitoring()

                    // Transition to recording state
                    print("✅ [START] Transitioning to recording state")
                    transitionTo(.recording)

                } catch {
                    print("❌ [START] Failed to start capture: \(error)")
                    transitionTo(.error(error.localizedDescription))
                }
            }
        }
    }

    private func stopRecording() {
        print("⏹️ [STOP] Stop requested, current state: \(recordingState.description)")

        // Transition to stopping state
        transitionTo(.stopping)

        // Stop audio capture
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

        // Transition to idle state
        transitionTo(.idle)
    }

    private func startSilenceMonitoring() {
        // Monitor audio levels every 0.1 seconds to detect silence
        silenceMonitorTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            Task { @MainActor in
                self.monitorForSilence()
            }
        }
        // Ensure timer runs on main RunLoop
        if let timer = silenceMonitorTimer {
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    private func stopSilenceMonitoring() {
        silenceMonitorTimer?.invalidate()
        silenceMonitorTimer = nil
        resultDisplayTimer?.invalidate()
        resultDisplayTimer = nil
        silenceStartTime = nil
        detector.clearResult()
    }

    // MARK: - Silence Monitoring (Refactored)

    /// Calculate current audio level from recent samples
    private func calculateCurrentAudioLevel() -> (rms: Float, db: Float, isSilent: Bool) {
        let samples = audioEngine.getCurrentBuffer()
        guard !samples.isEmpty else { return (0, -160, true) }

        let recentSamples = samples.suffix(1600)  // Last 0.1s (ArraySlice - no copy)
        var rms: Float = 0
        recentSamples.withUnsafeBufferPointer { buffer in
            vDSP_rmsqv(buffer.baseAddress!, 1, &rms, vDSP_Length(buffer.count))
        }

        let db = audioEngine.rmsToDecibels(rms)
        let isSilent = rms < silenceThreshold
        return (rms, db, isSilent)
    }

    /// Handle silence detected - track duration and trigger turn detection
    private func handleSilenceDetected() {
        // Start silence tracking if not already started (and not in cooldown)
        if silenceStartTime == nil && audioEngine.bufferDuration >= minimumBufferForDetection && recordingState != .cooldown {
            silenceStartTime = Date()
            print("🟡 Silence started (buffer ready)")
        }

        // Check if we've been silent long enough to trigger detection (only if not in cooldown)
        guard let silenceStart = silenceStartTime, recordingState == .recording else { return }

        let silenceDuration = Date().timeIntervalSince(silenceStart)
        print("⏱️  Silence duration: \(String(format: "%.1f", silenceDuration))s")

        // Trigger detection once per silence period
        if silenceDuration >= self.silenceDuration {
            print("✅ TRIGGERING TURN DETECTION")
            transitionTo(.detectingTurn)

            // Capture current state version to detect stale callbacks
            let capturedVersion = stateVersion

            detector.detectTurnAndUpdate { result in
                // Check if state changed while detection was running
                guard self.stateVersion == capturedVersion else {
                    print("⚠️ Stale turn detection callback (v\(capturedVersion) != v\(self.stateVersion)), ignoring result")
                    return
                }
                self.handleTurnDetectionResult(result)
            }
        }
    }

    /// Handle turn detection result - log, accumulate transcription, transition state
    private func handleTurnDetectionResult(_ result: TurnDetectionResult?) {
        guard let result = result else {
            // Detection failed, return to recording
            transitionTo(.recording)
            return
        }

        // Log result
        let resultText = result.isTurnComplete
            ? "✅ Turn change detected (\(result.probabilityPercentage))"
            : "⏳ Turn change not detected (\(result.probabilityPercentage))"
        addLog(resultText, level: result.isTurnComplete ? .success : .warning)

        // Accumulate transcription and restart recognition
        accumulateTranscriptionSegment()

        // Auto-clear result after 3 seconds
        setupAutoClearTimer()

        // Transition to cooldown state to prevent re-triggering
        transitionTo(.cooldown)
    }

    /// Accumulate current transcription segment and restart recognition
    private func accumulateTranscriptionSegment() {
        let utterance = detector.speechRecognitionManager.transcribedText
        guard !utterance.isEmpty && utterance != lastSegmentText else { return }

        // Add separator if needed
        if !accumulatedTranscription.isEmpty {
            accumulatedTranscription += "\n----------\n"
        }
        accumulatedTranscription += utterance
        lastSegmentText = utterance

        // Restart speech recognition to get fresh segment (with delay for audio engine)
        guard detector.speechRecognitionManager.isRecognizing else { return }

        detector.speechRecognitionManager.stopRecognition()

        // Wait for audio engine to fully stop before restarting
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 500_000_000) // 500ms delay

            // Check if we're still in a valid state for speech recognition
            // (Recording or Cooldown are both valid - Cooldown still needs transcription)
            guard self.recordingState == .recording || self.recordingState == .cooldown else {
                print("⚠️ Invalid state for recognition restart: \(self.recordingState.description), skipping")
                return
            }

            do {
                try self.detector.speechRecognitionManager.startRecognition()
                print("✅ Speech recognition restarted successfully")
            } catch {
                print("❌ Failed to restart recognition: \(error)")
            }
        }
    }

    /// Setup auto-clear timer to clear result after 3 seconds
    private func setupAutoClearTimer() {
        resultDisplayTimer?.invalidate()

        // Capture state version to detect if user stops recording before timer fires
        let capturedVersion = stateVersion

        resultDisplayTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { _ in
            Task { @MainActor in
                // Check if state changed (e.g., user stopped recording)
                guard self.stateVersion == capturedVersion else {
                    print("⚠️ State changed before auto-clear timer fired, skipping clear")
                    return
                }

                self.detector.clearResult()
                print("🔄 Auto-cleared result")
                self.addLog("🔄 Result cleared", level: .info)
            }
        }
    }

    /// Handle speaking detected - exit cooldown, reset state, clear timers
    private func handleSpeakingDetected() {
        // Log if we were tracking silence
        if silenceStartTime != nil {
            print("🟢 Speaking detected")
            addLog("🗣️ Speaking detected", level: .info)
        }

        silenceStartTime = nil

        // Exit cooldown state when speaking is detected
        if recordingState == .cooldown {
            transitionTo(.recording)
            addLog("🔄 Exited cooldown (speaking detected)", level: .info)
        }

        // Cancel auto-clear timer and clear result
        resultDisplayTimer?.invalidate()
        Task { @MainActor in
            detector.clearResult()
        }
    }

    /// Main silence monitoring method (now simplified - calls focused helper methods)
    private func monitorForSilence() {
        // Calculate current audio level
        let (rms, db, isSilent) = calculateCurrentAudioLevel()

        // Debug: Log audio level and state
        print("🔊 Audio: \(String(format: "%.1f", db)) dB (RMS: \(String(format: "%.4f", rms))) | Silent: \(isSilent) | Buffer: \(String(format: "%.1f", audioEngine.bufferDuration))s")

        if isSilent {
            handleSilenceDetected()
        } else {
            handleSpeakingDetected()
        }
    }
}

#Preview {
    TurnDetectionView()
}
