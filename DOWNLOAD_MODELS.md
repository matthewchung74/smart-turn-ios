# Download WhisperKit Models for Bundling

## Quick Setup (5 minutes)

Run these commands to download Base and Small models:

```bash
cd ~/Desktop/smart-turn-ios

# Download Base model (~74MB)
curl -L "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-base/AudioEncoder.mlmodelc.zip" -o base_encoder.zip
curl -L "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-base/TextDecoder.mlmodelc.zip" -o base_decoder.zip

# Download Small model (~244MB)
curl -L "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-small/AudioEncoder.mlmodelc.zip" -o small_encoder.zip
curl -L "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-small/TextDecoder.mlmodelc.zip" -o small_decoder.zip

# Create directory structure
mkdir -p smart-turn-ios/Models/WhisperKit/openai_whisper-base
mkdir -p smart-turn-ios/Models/WhisperKit/openai_whisper-small

# Unzip models
unzip base_encoder.zip -d smart-turn-ios/Models/WhisperKit/openai_whisper-base/
unzip base_decoder.zip -d smart-turn-ios/Models/WhisperKit/openai_whisper-base/
unzip small_encoder.zip -d smart-turn-ios/Models/WhisperKit/openai_whisper-small/
unzip small_decoder.zip -d smart-turn-ios/Models/WhisperKit/openai_whisper-small/

# Cleanup zips
rm base_encoder.zip base_decoder.zip small_encoder.zip small_decoder.zip

echo "✅ Models downloaded!"
echo "Next: Drag 'smart-turn-ios/Models/WhisperKit' folder into Xcode project"
```

## Alternative: Use WhisperKit CLI

If you have Homebrew:

```bash
# Install CLI
brew install whisperkit-cli

# Download models
whisperkit-cli download --model openai_whisper-base --output ~/Desktop/smart-turn-ios/smart-turn-ios/Models/WhisperKit
whisperkit-cli download --model openai_whisper-small --output ~/Desktop/smart-turn-ios/smart-turn-ios/Models/WhisperKit
```

## Add to Xcode Project

1. Open `smart-turn-ios.xcodeproj` in Xcode
2. Drag `smart-turn-ios/Models/WhisperKit/` folder into the Xcode navigator
3. **Important**: Check "Copy items if needed" and "Create folder references"
4. Verify models appear in project navigator under Models/WhisperKit/

## Verify

Models should be in:
```
smart-turn-ios/Models/WhisperKit/
├── openai_whisper-base/
│   ├── AudioEncoder.mlmodelc/
│   └── TextDecoder.mlmodelc/
└── openai_whisper-small/
    ├── AudioEncoder.mlmodelc/
    └── TextDecoder.mlmodelc/
```

Total size: ~320MB (will add to app bundle)
