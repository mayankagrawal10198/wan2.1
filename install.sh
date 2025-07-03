#!/bin/bash

# Wan2.1 I2V Model Installation Script (Improved)
# Sets up the environment, verifies packages, and logs issues clearly

set -e  # Exit on any command error

echo "🚀 Wan2.1 I2V Model Installation Script"
echo "========================================"

# --- Step 1: Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Need Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# --- Step 2: Check NVIDIA GPU ---
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ NVIDIA GPU not detected. The model will run on CPU (very slow)."
fi

# --- Step 3: Setup virtual environment ---
echo "📦 Creating virtual environment (wan21-env)..."
python3 -m venv wan21-env

echo "🔧 Activating virtual environment..."
source wan21-env/bin/activate

# --- Step 4: Upgrade pip ---
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# --- Step 5: Install PyTorch (CUDA 12.4) ---
echo "🔧 Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# --- Step 6: Install Python dependencies with retry and log ---
echo "📦 Installing Python dependencies from requirements.txt..."
pip install --no-cache-dir --force-reinstall -r requirements.txt 2>&1 | tee install.log || {
    echo "❌ Some packages failed to install. Check install.log for details."
    exit 1
}

# --- Step 7: Verify installation with pip check ---
echo "🔍 Verifying installation..."
pip check || {
    echo "❌ Package conflicts or missing dependencies found. Fix issues before continuing."
    exit 1
}

# --- Step 8: Show installed packages ---
echo "📦 Installed packages summary:"
pip list

# --- Step 9: Create working directories ---
echo "📁 Creating working directories..."
mkdir -p input output temp

# --- Step 10: Test PyTorch + CUDA ---
echo "🧪 Testing PyTorch installation..."
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name()}')
"

# --- Completion Message ---
echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment:"
echo "   source wan21-env/bin/activate"
echo ""
echo "2. Place your input images in the 'input' directory"
echo ""
echo "3. Run the model:"
echo "   python run_wan21.py --image input/your_image.jpg --prompt 'Your description'"
echo ""
echo "4. Check system requirements (optional):"
echo "   python run_wan21.py --check_system"
echo ""
echo "🛠 Log saved to: install.log"
