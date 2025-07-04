# Wan2.1 I2V Model - Local Setup

This repository provides a complete setup for running the Wan2.1 Image-to-Video (I2V) 14B 480P model locally on your machine.

## System Requirements

### Hardware Requirements
- **GPU**: RTX A6000 (48GB VRAM) ✅ **Your setup is optimal!**
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50-100GB free space for model downloads

### Software Requirements
- **Python**: 3.10.x (recommended)
- **CUDA**: 12.7 ✅ **Your setup is compatible!**
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv wan21-env

# Activate on macOS/Linux
source wan21-env/bin/activate

# Activate on Windows
wan21-env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.7 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run the Model

```bash
# Basic usage with default settings
python run_wan21.py --image path/to/your/image.jpg --prompt "Your video description"

# With custom settings
python run_wan21.py --image input.jpg --prompt "Cinematic scene with gentle movement" --num_frames 81 --guidance_scale 5.0

# With custom resolution
python run_wan21.py --image input.jpg --prompt "Your description" --width 720 --height 576
```

## Default Settings

The model uses high-quality default settings that work well on most modern GPUs:

### Default Configuration
- **Frames**: 81 (5 seconds at 16 FPS)
- **Inference Steps**: 50
- **Guidance Scale**: 5.0
- **Resolution**: 480x832 (auto-calculated if not specified)
- **Data Type**: bfloat16 (optimal for modern GPUs)

### Customization
All settings can be overridden via command line arguments:
```bash
# Custom frames and quality
python run_wan21.py --image input.jpg --prompt "Your description" --num_frames 65 --num_inference_steps 40

# Custom resolution
python run_wan21.py --image input.jpg --prompt "Your description" --width 720 --height 576

# Custom quality settings
python run_wan21.py --image input.jpg --prompt "Your description" --guidance_scale 6.0 --num_inference_steps 60
```

## Web Interface

A beautiful web interface is available for easy video generation:

### Features
- **Resolution Selection**: 480p (320×400) or 720p (576×720)
- **Model Selection**: Automatically uses appropriate model for each resolution
  - 480p: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
  - 720p: `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`
- **Drag & Drop Upload**: Easy image upload with format validation
- **Real-time Feedback**: Loading states and progress indicators

### Start Web Interface
```bash
python start_web.py
```
Then open: http://localhost:8080

For detailed web interface documentation, see [WEB_README.md](WEB_README.md).

## Video Guidance Feature

The application now supports **video-guided generation** using the Wan2.1 VACE (Video-Aware Conditional Editing) pipeline. This allows you to provide a reference video to guide the movement and motion of your generated video.

### How It Works

1. **Standard I2V**: Upload only an image → Uses Wan2.1 I2V pipeline
2. **Video-Guided**: Upload image + video → Uses Wan2.1 VACE pipeline

### Video Guidance Benefits

- **Motion Control**: The generated video follows the movement patterns from your reference video
- **Temporal Consistency**: Better frame-to-frame coherence
- **Creative Control**: More precise control over the final video's motion
- **Robust Processing**: Reliable frame extraction that handles various video formats and metadata issues

### Supported Video Formats

- **Input Videos**: MP4, AVI, MOV, MKV (max 50MB)
- **Output**: MP4 format with 16 FPS

### Usage Examples

#### Web Interface
1. Upload your image
2. **Optionally** upload a reference video for movement guidance
3. Enter your prompts
4. Select resolution (480p or 720p)
5. Generate video

#### Programmatic Usage
```python
from wan21_pipeline import WanVACEPipelineWrapper

# Initialize VACE pipeline
pipeline = WanVACEPipelineWrapper()

# Generate video with video guidance
pipeline.generate_video_with_guidance(
    image_path="input.jpg",
    video_path="reference_video.mp4",
    prompt="A beautiful scene with gentle movement",
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    conditioning_scale=1.0
)
```

### Model Downloads

The VACE model is automatically downloaded when you run:
```bash
python download_models.py
```

This downloads:
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` → `models/wan21-480p`
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` → `models/wan21-720p`
- `Wan-AI/Wan2.1-VACE-14B-diffusers` → `models/wan21-vace`

## Usage Examples

### Basic Video Generation
```python
from wan21_pipeline import Wan21Pipeline

# Initialize pipeline
pipeline = Wan21Pipeline()

# Generate video
pipeline.generate_video(
    image_path="input.jpg",
    prompt="A beautiful scene with gentle movement",
    output_path="output.mp4"
)
```

### Advanced Configuration
```python
# Custom settings for optimal performance on RTX A6000
pipeline = Wan21Pipeline(
    model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    torch_dtype="bfloat16",
    enable_optimizations=True
)

# Generate with custom parameters
pipeline.generate_video(
    image_path="input.jpg",
    prompt="Cinematic quality scene with smooth motion",
    negative_prompt="static, blurred, low quality",
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50,
    output_path="high_quality_output.mp4"
)
```

## Performance Expectations

With your RTX A6000 (48GB VRAM):
- **480P video (81 frames)**: ~2-3 minutes
- **720P video (81 frames)**: ~4-5 minutes
- **Custom resolution**: Performance scales with frame count

## File Structure

```
wan2.1/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_wan21.py             # Main script for running the model
├── wan21_pipeline.py        # Pipeline class for easy usage
├── utils.py                 # Utility functions
├── config.py                # Configuration settings
├── app.py                   # Flask web application
├── start_web.py            # Web interface startup script
├── templates/
│   └── index.html          # Web interface HTML
├── examples/                # Example scripts
│   ├── basic_example.py
│   ├── batch_processing.py
│   └── simple_example.py
├── input/                   # Place your input images here
├── output/                  # Generated videos will be saved here
├── uploads/                 # Web interface uploaded images
└── WEB_README.md           # Web interface documentation
```

## Configuration

Edit `config.py` to customize default settings:

```python
# Default model settings
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_NUM_FRAMES = 81
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 50
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_frames` to 49 or 33
   - Enable optimizations in pipeline
   - Use `torch.float16` instead of `bfloat16`

2. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face access permissions
   - Clear cache: `huggingface-cli delete-cache`

3. **Performance Issues**
   - Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU memory: `nvidia-smi`
   - Enable optimizations in pipeline

### Memory Optimization

For optimal performance on your RTX A6000:

```python
# Enable all optimizations
pipeline = Wan21Pipeline(
    enable_optimizations=True,
    enable_attention_slicing=True,
    enable_vae_slicing=True
)
```

## Advanced Usage

### Batch Processing
```python
from examples.batch_processing import BatchProcessor

processor = BatchProcessor()
processor.process_directory(
    input_dir="input/",
    output_dir="output/",
    prompt="Cinematic scene"
)
```

### Custom Resolutions
```python
# Generate custom resolution video
pipeline.generate_video(
    image_path="input.jpg",
    prompt="Custom resolution scene",
    height=720,
    width=1280,
    output_path="custom_resolution.mp4"
)
```

## License

This project is for educational and research purposes. Please respect the original model licenses and terms of use.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Check system requirements
4. Verify CUDA installation

## Performance Tips for RTX A6000

Your RTX A6000 with 48GB VRAM is excellent for this model:

1. **Use bfloat16**: Optimal balance of speed and quality
2. **Enable optimizations**: Better memory management
3. **Full frame count**: You can use 81 frames without issues
4. **Higher resolution**: Can handle 720P comfortably
5. **Batch processing**: Can process multiple videos efficiently 