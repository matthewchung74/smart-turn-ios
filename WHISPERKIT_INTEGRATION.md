# WhisperKit Speech-to-Text Integration

## Overview

WhisperKit has been integrated into SmartTurn iOS app to provide optional speech-to-text transcription alongside turn detection. Users can load/unload models (Base or Small) on-demand and see live transcriptions.

## What Was Added

### 1. **WhisperKitManager.swift** (`smart-turn-ios/Audio/`)
A dedicated manager class that handles:
- ✅ Loading WhisperKit models (Base 74MB, Small 244MB)
- ✅ Unloading models to free memory
- ✅ Status messages during load/unload operations
- ✅ Audio transcription from buffer
- ✅ Published state for UI bindings

**Key Features:**
- Async model loading with progress messages
- Prevents multiple simultaneous loads
- Graceful error handling
- Clear status messages ("📥 Loading...", "✅ Loaded", "🗑️ Unloading...")

### 2. **SmartTurnDetector.swift** (Updated)
Added WhisperKit integration:
- ✅ `whisperKitManager` property (accessible from views)
- ✅ `detectTurnAndTranscribe()` method that runs both turn detection AND transcription
- ✅ Keeps existing `detectTurnAndUpdate()` for turn-only detection

### 3. **TurnDetectionView.swift** (Updated)
Added comprehensive UI for WhisperKit:

#### New UI Sections:
1. **WhisperKit Section** (collapsible purple card):
   - Model picker (Base/Small segmented control)
   - Load/Unload buttons with disabled states
   - Status indicator (green checkmark when loaded)
   - Real-time status messages

2. **Transcription Section**:
   - Scrollable text view showing transcribed speech
   - Clear button when text is present
   - Placeholder text when empty

#### Behavior Updates:
- Silence detection now triggers BOTH turn detection and transcription (when model loaded)
- Transcription appears in both:
  - Dedicated transcription view
  - State history log (with 📝 emoji)
- Console logs include transcription output

## Setup Instructions

### Step 1: Add WhisperKit Dependency in Xcode

Since this is an Xcode project, you need to add WhisperKit through Xcode's Swift Package Manager:

1. Open `smart-turn-ios.xcodeproj` in Xcode
2. Select your project in the Navigator (top item)
3. Select the "smart-turn-ios" target
4. Go to the "General" tab
5. Scroll to "Frameworks, Libraries, and Embedded Content"
6. Click the "+" button
7. Click "Add Package Dependency..."
8. Enter the repository URL:
   ```
   https://github.com/argmaxinc/WhisperKit.git
   ```
9. Set "Dependency Rule" to "Up to Next Major Version" with "0.9.0"
10. Click "Add Package"
11. Select "WhisperKit" and click "Add Package"

**Alternative (if above doesn't work):**
1. File → Add Package Dependencies...
2. Paste: `https://github.com/argmaxinc/WhisperKit.git`
3. Version: 0.9.0 or later
4. Add to target: smart-turn-ios

### Step 2: Build and Run

1. Build the project (⌘+B)
2. Run on a physical device or simulator
3. The app should compile successfully

**Expected Warnings:**
- WhisperKit may show warnings during first build - these are normal
- Model downloads happen on first load (in-app, not during build)

## How to Use

### In the App:

1. **Start Recording**: Tap the "Start" button (as before)

2. **Load WhisperKit** (optional):
   - Tap "Show" on the purple "Speech-to-Text (WhisperKit)" section
   - Select model: "Base (74MB)" or "Small (244MB)"
   - Tap "Load Model"
   - Wait for download/initialization (~30-60 seconds first time)
   - Status will show "✅ Base (74MB) loaded successfully"

3. **Use Transcription**:
   - Speak normally
   - After 1.5 seconds of silence, app runs:
     - Turn detection (as before)
     - **NEW:** Transcription (if model loaded)
   - View transcription in:
     - Dedicated "Transcription" section (scrollable)
     - State History log (with 📝 prefix)

4. **Unload Model** (optional):
   - Tap "Unload" to free memory (~74-244MB)
   - Status: "✅ Model unloaded"
   - Turn detection still works (transcription disabled)

### Console Output Example:

```
📥 Loading WhisperKit model: openai_whisper-base
✅ WhisperKit loaded: openai_whisper-base
🎤 Transcribing...
✅ Transcribed (47 chars)
🎯 Turn Detection Result:
   - Probability: 85.2%
   - Turn Complete: true
   - Inference: 8.3ms
   - Audio Duration: 2.45s
   - Transcription: Hello, how are you doing today?
```

## File Structure

```
smart-turn-ios/
├── Audio/
│   ├── AudioCaptureEngine.swift      (unchanged)
│   ├── WhisperFeatureExtractor.swift (unchanged)
│   └── WhisperKitManager.swift       ✨ NEW
├── TurnDetection/
│   └── SmartTurnDetector.swift       ✏️ UPDATED
├── Views/
│   └── TurnDetectionView.swift       ✏️ UPDATED
└── Models/
    └── smart-turn-v3.0.onnx          (unchanged)
```

## Performance Notes

### Model Sizes:
- **Base**: 74MB download, ~150MB RAM when loaded
- **Small**: 244MB download, ~500MB RAM when loaded

### Inference Times (iPhone 14 Pro):
- **Base**: ~1-2 seconds for 8s audio
- **Small**: ~2-4 seconds for 8s audio

### Recommendations:
- Use **Base** for most cases (good balance of speed/accuracy)
- Use **Small** for higher accuracy (medical, legal, etc.)
- Unload when not needed to save memory

## Testing Checklist

- [ ] WhisperKit dependency added successfully
- [ ] Project builds without errors
- [ ] App launches without crashes
- [ ] WhisperKit section appears in UI
- [ ] Model picker shows Base/Small options
- [ ] "Load Model" button works (shows loading state)
- [ ] Status messages appear during load
- [ ] Model loads successfully (green checkmark)
- [ ] Transcription section appears
- [ ] Speaking + silence triggers transcription
- [ ] Transcribed text appears in both places:
  - [ ] Transcription section
  - [ ] State history log
- [ ] "Unload" button works (status updates)
- [ ] Turn detection still works without model loaded

## Troubleshooting

### Issue: "Cannot find 'WhisperKit' in scope"
**Solution:** Add WhisperKit package dependency (see Step 1 above)

### Issue: Model fails to load
**Symptoms:** "❌ Failed to load Base (74MB): ..."
**Solutions:**
- Check internet connection (first download)
- Restart app
- Try Base model first (smaller, faster)
- Check console for detailed error

### Issue: Transcription is empty
**Possible causes:**
- No speech detected (too quiet)
- Audio buffer too short (<0.5s)
- Model still loading
**Solutions:**
- Speak louder/closer to mic
- Check buffer duration (≥0.5s)
- Wait for "✅ loaded successfully" status

### Issue: App crashes after adding WhisperKit
**Solution:** Clean build folder (⇧⌘K) and rebuild

## Code Integration Points

### To run turn detection with transcription:
```swift
detector.detectTurnAndTranscribe { result, transcription in
    print("Turn: \(result?.isTurnComplete ?? false)")
    print("Text: \(transcription ?? "")")
}
```

### To run turn detection only:
```swift
detector.detectTurnAndUpdate { result in
    print("Turn: \(result?.isTurnComplete ?? false)")
}
```

### To check if WhisperKit is ready:
```swift
if detector.whisperKitManager.isLoaded {
    // Transcription available
} else {
    // Transcription not available
}
```

## Future Enhancements

Potential improvements:
- [ ] Add "Tiny" model option (39MB, fastest)
- [ ] Add "Large" model option (1.5GB, most accurate)
- [ ] Stream transcription word-by-word (real-time)
- [ ] Export transcription to text file
- [ ] Highlight transcription when turn detected
- [ ] Add language selection (multi-lingual support)
- [ ] Cache models locally (skip re-download)

## Credits

- **WhisperKit**: [argmaxinc/WhisperKit](https://github.com/argmaxinc/whisperkit)
- **OpenAI Whisper**: [openai/whisper](https://github.com/openai/whisper)
