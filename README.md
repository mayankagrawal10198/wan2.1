# Wan2.1 Video Generation API

A FastAPI web application for generating videos from images using Wan2.1 I2V and VACE models with LoRA support.

## Features

- **I2V Pipeline**: Image-to-video generation
- **VACE Pipeline**: Video-assisted creative enhancement
- **LoRA Support**: CausVid LoRA integration for enhanced quality
- **Web Interface**: FastAPI-based REST API with web UI
- **Memory Optimized**: Efficient GPU memory management for RTX A6000

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
```bash
python download_models.py
```

### 3. Run the Web Application
```bash
python app.py

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app.py
```

Access the web interface at: http://localhost:8080

## Configuration

Edit `config.py` to customize:
- Model settings and optimizations
- LoRA configuration
- Memory management parameters

## API Endpoints

- `POST /generate` - Generate videos from images
- `GET /video/{filename}` - Stream generated videos
- `GET /download/{filename}` - Download videos
- `GET /health` - Health check

## Supported Formats

**Images**: JPG, PNG, BMP, TIFF, WebP
**Videos**: MP4, AVI, MOV, MKV

## System Requirements

- NVIDIA GPU with 48GB+ VRAM (tested on RTX A6000)
- Python 3.8+
- CUDA 11.8+

## License

MIT License 