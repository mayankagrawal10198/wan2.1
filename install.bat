@echo off
REM Wan2.1 I2V Model Installation Script for Windows
REM This script sets up the environment for running Wan2.1 I2V model locally

echo 🚀 Wan2.1 I2V Model Installation Script
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if CUDA is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ NVIDIA GPU not detected. The model will run on CPU (very slow).
) else (
    echo ✅ NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv wan21-env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call wan21-env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo 🔧 Installing PyTorch with CUDA support...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

REM Install other dependencies
echo 📦 Installing other dependencies...
pip install -r requirements.txt

REM Create directories
echo 📁 Creating directories...
if not exist input mkdir input
if not exist output mkdir output
if not exist temp mkdir temp

REM Test installation
echo 🧪 Testing installation...
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else '❌ CUDA not available'); print(f'✅ GPU: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else '❌ No GPU detected')"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Activate the virtual environment:
echo    wan21-env\Scripts\activate
echo.
echo 2. Place your input images in the 'input' directory
echo.
echo 3. Run the model:
echo    python run_wan21.py --image input\your_image.jpg --prompt "Your description"
echo.
echo 4. Or use RTX A6000 optimized settings:
echo    python run_wan21.py --image input\your_image.jpg --prompt "Your description" --rtx_a6000
echo.
echo 5. Check system requirements:
echo    python run_wan21.py --check_system
echo.
echo 📚 For more examples, see the 'examples' directory
echo 📖 For detailed documentation, see README.md
echo.
pause 