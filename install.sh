#!/usr/bin/env bash

set -e

echo "=== Checking Python version ==="

PYTHON=$(which python3 || which python)
PY_VER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "Detected Python version: $PY_VER"

# Compare version strings
PY_OK=$($PYTHON -c 'import sys; print(sys.version_info < (3, 13))')

if [[ "$PY_OK" != "True" ]]; then
    echo "❌ Python 3.13 or newer detected. PyTorch may not yet support this."
    echo "Please downgrade to Python 3.12 or earlier."
    exit 1
fi

echo "✅ Python version is suitable."

echo "=== Checking for pip ==="
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON get-pip.py
fi

echo "=== Installing dependencies ==="

REQUIRED_PKGS=(
    numpy
    pillow
    opencv-python
    ffmpeg-python
)

for pkg in "${REQUIRED_PKGS[@]}"; do
    echo "Installing $pkg..."
    pip install "$pkg"
done

echo "=== Detecting CUDA version (if any) ==="

CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
elif [[ -f "/usr/local/cuda/version.txt" ]]; then
    CUDA_VERSION=$(grep -oP 'CUDA Version \K[0-9.]+' /usr/local/cuda/version.txt)
elif [[ "$OS" == "Windows_NT" ]]; then
    CUDA_VERSION=$(where nvcc.exe 2> /dev/null | head -n 1 | xargs -I{} cmd /c "{} --version" | findstr /R "release [0-9.]*" | findstr /R /O "[0-9.]*")
fi

if [[ -n "$CUDA_VERSION" ]]; then
    echo "Found CUDA version: $CUDA_VERSION"
else
    echo "⚠️  No CUDA detected. Will install CPU-only PyTorch."
fi

echo "=== Installing PyTorch ==="

# Use official selector to install torch for current OS and Python
if [[ -n "$CUDA_VERSION" ]]; then
    # For Linux, Mac, Windows (wheel auto-resolves CUDA)
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch
fi

echo "✅ All done!"
