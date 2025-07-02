"""
Utility functions for Wan2.1 I2V Model
"""

import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from typing import Tuple, Optional, Union
from config import SUPPORTED_IMAGE_FORMATS, INPUT_DIR, OUTPUT_DIR, TEMP_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_image_path(image_path: str) -> bool:
    """Validate if the image path exists and is a supported format."""
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist: {image_path}")
        return False
    
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        logger.error(f"Unsupported image format: {file_ext}")
        return False
    
    return True


def load_and_preprocess_image(image_path: str, target_height: int = 480, target_width: int = 832) -> Optional[Image.Image]:
    """
    Load and preprocess image for the model.
    
    Args:
        image_path: Path to the input image
        target_height: Target height for the image
        target_width: Target width for the image
    
    Returns:
        Preprocessed PIL Image or None if error
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Calculate optimal dimensions
        max_area = target_height * target_width
        aspect_ratio = image.height / image.width
        
        # Calculate new dimensions maintaining aspect ratio
        if aspect_ratio > 1:  # Portrait
            new_height = min(target_height, int(np.sqrt(max_area * aspect_ratio)))
            new_width = int(new_height / aspect_ratio)
        else:  # Landscape
            new_width = min(target_width, int(np.sqrt(max_area / aspect_ratio)))
            new_height = int(new_width * aspect_ratio)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.info(f"Preprocessed image: {image_path} -> {new_width}x{new_height}")
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None


def calculate_optimal_dimensions(image: Image.Image, max_area: int = 480 * 832, 
                                vae_scale_factor: int = 8, patch_size: int = 16) -> Tuple[int, int]:
    """
    Calculate optimal dimensions for the model.
    
    Args:
        image: Input PIL Image
        max_area: Maximum area constraint
        vae_scale_factor: VAE scale factor
        patch_size: Transformer patch size
    
    Returns:
        Tuple of (height, width)
    """
    aspect_ratio = image.height / image.width
    mod_value = vae_scale_factor * patch_size
    
    # Calculate dimensions
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    
    return height, width


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def check_gpu_memory():
    """Check available GPU memory and log information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Total VRAM: {total_memory:.1f} GB")
        logger.info(f"Allocated: {allocated_memory:.1f} GB")
        logger.info(f"Cached: {cached_memory:.1f} GB")
        logger.info(f"Free: {total_memory - allocated_memory:.1f} GB")
        
        return {
            "gpu_name": gpu_name,
            "total_memory": total_memory,
            "allocated_memory": allocated_memory,
            "cached_memory": cached_memory,
            "free_memory": total_memory - allocated_memory
        }
    else:
        logger.warning("CUDA not available")
        return None


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_output_filename(input_path: str, suffix: str = "") -> str:
    """Generate output filename based on input path."""
    input_name = Path(input_path).stem
    timestamp = int(torch.cuda.Event().elapsed_time(torch.cuda.Event()) * 1000) if torch.cuda.is_available() else 0
    return f"{input_name}_{timestamp}{suffix}.mp4"


def validate_model_parameters(num_frames: int, guidance_scale: float, 
                            num_inference_steps: int) -> bool:
    """Validate model parameters."""
    if num_frames <= 0 or num_frames > 200:
        logger.error(f"Invalid num_frames: {num_frames}. Must be between 1 and 200.")
        return False
    
    if guidance_scale <= 0 or guidance_scale > 20:
        logger.error(f"Invalid guidance_scale: {guidance_scale}. Must be between 0 and 20.")
        return False
    
    if num_inference_steps <= 0 or num_inference_steps > 100:
        logger.error(f"Invalid num_inference_steps: {num_inference_steps}. Must be between 1 and 100.")
        return False
    
    return True


def log_system_info():
    """Log system information for debugging."""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    
    # Check GPU memory
    gpu_info = check_gpu_memory()
    if gpu_info:
        logger.info("=== GPU Information ===")
        for key, value in gpu_info.items():
            logger.info(f"{key}: {value}")
    
    logger.info("=== Environment ===")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {os.sys.version}")


def create_progress_callback(total_steps: int):
    """Create a progress callback function."""
    from tqdm import tqdm
    
    pbar = tqdm(total=total_steps, desc="Generating video")
    
    def callback(step: int, timestep: int, latents: torch.FloatTensor):
        pbar.update(1)
        if step == total_steps - 1:
            pbar.close()
    
    return callback 