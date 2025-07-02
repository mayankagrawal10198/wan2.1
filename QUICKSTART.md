# Quick Start Guide - Wan2.1 I2V Model

## 🚀 Get Started in 5 Minutes

### Prerequisites
- **GPU**: RTX A6000 (48GB VRAM) ✅ **Your setup is perfect!**
- **CUDA**: 12.7 ✅ **Compatible!**
- **Python**: 3.8+ 
- **Storage**: 50-100GB free space

### Step 1: Install Dependencies

**Option A: Automatic Installation (Recommended)**
```bash
# On macOS/Linux
chmod +x install.sh
./install.sh

# On Windows
install.bat
```

**Option B: Manual Installation**
```bash
# Create virtual environment
python -m venv wan21-env

# Activate environment
source wan21-env/bin/activate  # macOS/Linux
# or
wan21-env\Scripts\activate     # Windows

# Install PyTorch with CUDA 12.7 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Test Your Setup
```bash
# Check system requirements
python run_wan21.py --check_system
```

### Step 3: Generate Your First Video

1. **Place an image in the `input` directory**
   ```bash
   # Copy your image to input directory
   cp your_image.jpg input/
   ```

2. **Generate video with RTX A6000 optimizations**
   ```bash
   python run_wan21.py --image input/your_image.jpg --prompt "A beautiful scene with gentle movement" --rtx_a6000
   ```

3. **Find your video in the `output` directory**

### 🎯 Quick Examples

**Basic Usage:**
```bash
python run_wan21.py --image input/photo.jpg --prompt "Cinematic scene with smooth motion"
```

**High Quality (RTX A6000 Optimized):**
```bash
python run_wan21.py --image input/photo.jpg --prompt "Professional cinematic quality" --rtx_a6000
```

**Custom Settings:**
```bash
python run_wan21.py --image input/photo.jpg \
  --prompt "Dynamic scene with camera movement" \
  --num_frames 81 \
  --guidance_scale 6.0 \
  --num_inference_steps 60
```

**720P Resolution:**
```bash
python run_wan21.py --image input/photo.jpg \
  --prompt "High resolution cinematic scene" \
  --height 720 --width 1280
```

### 📊 Expected Performance (RTX A6000)

| Resolution | Frames | Time | Quality |
|------------|--------|------|---------|
| 480P | 81 | ~2-3 min | High |
| 720P | 81 | ~4-5 min | Very High |
| Custom | 81 | ~3-4 min | High |

### 🔧 Troubleshooting

**CUDA Out of Memory:**
```bash
# Use lower frame count
python run_wan21.py --image input/photo.jpg --num_frames 49

# Use lower precision
python run_wan21.py --image input/photo.jpg --torch_dtype float16
```

**Model Download Issues:**
```bash
# Clear cache
huggingface-cli delete-cache

# Check internet connection
python -c "from huggingface_hub import HfApi; print('Connection OK')"
```

**Performance Issues:**
```bash
# Check GPU status
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 📁 File Structure
```
wan2.1/
├── input/          # Place your images here
├── output/         # Generated videos saved here
├── examples/       # Example scripts
├── run_wan21.py    # Main script
└── README.md       # Full documentation
```

### 🎬 Advanced Usage

**Batch Processing:**
```bash
python examples/batch_processing.py
```

**RTX A6000 Optimized Example:**
```bash
python examples/rtx_a6000_optimized.py
```

**Basic Example:**
```bash
python examples/basic_example.py
```

### 💡 Tips for Best Results

1. **Use high-quality input images** (1920x1080 or higher)
2. **Write descriptive prompts** with movement details
3. **Use the `--rtx_a6000` flag** for optimal performance
4. **Experiment with different prompts** for varied results
5. **Use 81 frames** for smooth 5-second videos

### 🆘 Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Review the [examples](examples/) directory for code examples
3. Check system requirements: `python run_wan21.py --check_system`
4. Verify GPU setup: `nvidia-smi`

### 🎉 You're Ready!

Your RTX A6000 with 48GB VRAM is perfect for this model. You should expect:
- **Fast generation times** (2-5 minutes per video)
- **High-quality output** (480P to 720P)
- **Smooth performance** with full optimizations enabled

Start creating amazing videos from your images! 🚀 