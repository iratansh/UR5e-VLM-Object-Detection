#!/bin/bash
# Setup script for UR5e Vision System Testing Environment
# This script creates a conda environment and installs all necessary packages

set -e  # Exit on any error

echo "🚀 Setting up UR5e Vision System Test Environment"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - will use compatible packages"
    MACOS=true
else
    echo "🐧 Detected Linux - full package support available"
    MACOS=false
fi

# Create conda environment
echo "📦 Creating conda environment 'ur5e_vision_test'..."
if conda env list | grep -q "ur5e_vision_test"; then
    echo "⚠️  Environment 'ur5e_vision_test' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ur5e_vision_test -y
    else
        echo "✅ Using existing environment"
        echo "🔄 Activating environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ur5e_vision_test
        echo "Environment activated. You can now run: python test_2d_3d_transformation.py"
        exit 0
    fi
fi

# Create environment from file
echo "🔧 Creating environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ur5e_vision_test

# Install additional packages that might not be available via conda
echo "📥 Installing additional packages..."

# Install PyAudio for speech recognition (macOS may need special handling)
echo "🎤 Installing PyAudio for speech recognition..."
if [ "$MACOS" = true ]; then
    # On macOS, try to install PyAudio, but don't fail if it doesn't work
    echo "🍎 Installing PyAudio for macOS..."
    pip install pyaudio || {
        echo "⚠️  PyAudio installation failed. Trying alternative approach..."
        echo "💡 If you want speech recognition, try: brew install portaudio"
        echo "💡 Then run: pip install pyaudio"
    }
else
    pip install pyaudio
fi

# Try to install PyRealSense2 (optional, will fail gracefully)
echo "📷 Installing RealSense packages (optional)..."
pip install pyrealsense2 || {
    echo "⚠️  PyRealSense2 installation failed (this is OK for testing without RealSense)"
    echo "💡 The test uses mock depth data, so RealSense is not required"
}

# Install additional development tools
echo "🔧 Installing development tools..."
pip install pytest-cov black flake8 || echo "⚠️  Some development tools failed to install (optional)"

# Test the installation
echo "🧪 Testing installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core packages
try:
    import cv2
    print('✅ OpenCV imported successfully')
except ImportError as e:
    print(f'❌ OpenCV import failed: {e}')

try:
    import numpy as np
    print('✅ NumPy imported successfully')
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import torch
    print('✅ PyTorch imported successfully')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import transformers
    print('✅ Transformers imported successfully')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')

try:
    import PIL
    print('✅ PIL imported successfully')
except ImportError as e:
    print(f'❌ PIL import failed: {e}')

# Test camera access
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('✅ Camera access working')
        cap.release()
    else:
        print('⚠️  Camera not accessible (check permissions)')
except Exception as e:
    print(f'⚠️  Camera test failed: {e}')

# Test speech recognition (optional)
try:
    import speech_recognition as sr
    print('✅ Speech recognition available')
except ImportError:
    print('⚠️  Speech recognition not available (install PyAudio if needed)')

# Test RealSense (optional)
try:
    import pyrealsense2 as rs
    print('✅ RealSense library available')
except ImportError:
    print('⚠️  RealSense library not available (using mock depth)')

print('🎉 Core environment setup complete!')
"

echo ""
echo "🎯 Environment Setup Complete!"
echo "================================"
echo ""
echo "To use the environment:"
echo "  conda activate ur5e_vision_test"
echo ""
echo "To run the 2D-3D transformation test:"
echo "  python test_2d_3d_transformation.py"
echo ""
echo "To deactivate the environment:"
echo "  conda deactivate"
echo ""
echo "📋 What's included:"
echo "  ✅ Python 3.9"
echo "  ✅ OpenCV for computer vision"
echo "  ✅ PyTorch for ML models"
echo "  ✅ Transformers for OWL-ViT"
echo "  ✅ NumPy, SciPy for numerical computing"
echo "  ✅ Speech recognition (if PyAudio works)"
echo "  ✅ Matplotlib, Seaborn for visualization"
echo "  ✅ Development tools (pytest, jupyter)"
echo ""
echo "🔧 Optional components:"
echo "  - RealSense library (uses mock depth if not available)"
echo "  - Speech recognition (can use manual input instead)"
echo "  - Development tools (for advanced usage)"
echo ""
echo "🎮 Ready to test!"
echo "Run: python test_2d_3d_transformation.py"
echo ""
echo "🆘 If you encounter issues:"
echo "  - Camera not working: Check macOS privacy settings"
echo "  - Speech not working: Install PyAudio with 'brew install portaudio'"
echo "  - Import errors: Make sure environment is activated"
echo ""
echo "Ready to test your 2D-3D coordinate transformations! 🚀" 