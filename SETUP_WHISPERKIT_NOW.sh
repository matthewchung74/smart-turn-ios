#!/bin/bash
set -e

echo "🚀 WhisperKit Base Model Setup"
echo "==============================="
echo ""
echo "This script will:"
echo "  1. Create Models/WhisperKit directory structure"
echo "  2. Clone the whisperkit-coreml repo"
echo "  3. Download ONLY Base model (74MB)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

cd ~/Desktop/smart-turn-ios

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p smart-turn-ios/Models/WhisperKit

# Clone repo
cd smart-turn-ios/Models/WhisperKit
if [ -d "whisperkit-coreml" ]; then
    echo "⚠️  whisperkit-coreml directory already exists"
    echo "Entering directory to download models..."
    cd whisperkit-coreml
else
    echo "📥 Cloning whisperkit-coreml repo..."
    git clone https://huggingface.co/argmaxinc/whisperkit-coreml
    cd whisperkit-coreml
fi

# Download ONLY Base model (not Small)
echo "📥 Downloading Base model ONLY (74MB, ~30-60 seconds)..."
git lfs pull --include="openai_whisper-base/*"

echo ""
echo "✅ Base model downloaded successfully!"
echo ""
echo "📍 Model location: ~/Desktop/smart-turn-ios/smart-turn-ios/Models/WhisperKit/whisperkit-coreml/openai_whisper-base/"
echo ""
echo "🔧 Next Steps in Xcode:"
echo "   1. If whisperkit-coreml is still in project, remove it first:"
echo "      - Right-click 'whisperkit-coreml' → Delete → Remove Reference"
echo ""
echo "   2. Add ONLY the base model folder:"
echo "      - Right-click 'Models/WhisperKit' in Xcode"
echo "      - Select 'Add Files to smart-turn-ios...'"
echo "      - Navigate to: Models/WhisperKit/whisperkit-coreml/"
echo "      - Select ONLY the 'openai_whisper-base' folder (NOT the parent)"
echo "      - ⚠️  IMPORTANT: Check these options:"
echo "         ❌ 'Copy items if needed' (UNCHECKED)"
echo "         ✅ 'Create folder references' (blue folders)"
echo "         ✅ Target: 'smart-turn-ios'"
echo "      - Click 'Add'"
echo ""
echo "   3. Clean Build Folder: Product → Clean Build Folder (Shift+⌘+K)"
echo "   4. Build: ⌘+B"
echo ""
echo "Expected structure in Xcode:"
echo "  Models/"
echo "    └── WhisperKit/"
echo "        └── openai_whisper-base/ (blue folder) ← ONLY THIS"
echo ""
