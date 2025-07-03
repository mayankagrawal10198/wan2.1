"""
Configuration settings for Wan2.1 I2V Model
Optimized for RTX A6000 with 48GB VRAM
"""

import torch

# Model Configuration
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_TORCH_DTYPE = "bfloat16"  # Optimal for RTX A6000
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video Generation Parameters
DEFAULT_NUM_FRAMES = 121  # Full frame count for RTX A6000
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_FPS = 24

# Resolution Settings
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 720
DEFAULT_MAX_AREA = DEFAULT_HEIGHT * DEFAULT_WIDTH

# Memory Optimization Settings
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
ENABLE_MODEL_CPU_OFFLOAD = True
ENABLE_SEQUENTIAL_CPU_OFFLOAD = False

# File Paths
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TEMP_DIR = "temp"

# Supported Image Formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# Default Prompts
DEFAULT_PROMPT = "A beautiful scene with gentle movement and cinematic quality"
DEFAULT_NEGATIVE_PROMPT = "static, blurred, low quality, distorted, ugly, bad anatomy"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model Download Settings
CACHE_DIR = None  # Use default Hugging Face cache
LOCAL_FILES_ONLY = False
REVISION = "main" 