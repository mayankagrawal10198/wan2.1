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


def calculate_pipeline_dimensions(image: Image.Image, pipeline, max_area: int = 480 * 832) -> Tuple[int, int]:
    """
    Calculate optimal dimensions for a specific pipeline using its actual VAE scale factor and patch size.
    
    Args:
        image: Input PIL Image
        pipeline: Loaded pipeline object
        max_area: Maximum area constraint
    
    Returns:
        Tuple of (height, width)
    """
    try:
        # Get VAE scale factor from pipeline
        vae_scale_factor = pipeline.vae_scale_factor_spatial if hasattr(pipeline, 'vae_scale_factor_spatial') else 8
        
        # Get patch size from transformer config
        patch_size = pipeline.transformer.config.patch_size[1] if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'config') else 16
        
        logger.info(f"Pipeline VAE scale factor: {vae_scale_factor}, patch size: {patch_size}")
        
        return calculate_optimal_dimensions(image, max_area, vae_scale_factor, patch_size)
        
    except Exception as e:
        logger.warning(f"Could not get pipeline parameters, using defaults: {e}")
        return calculate_optimal_dimensions(image, max_area)


# def aspect_ratio_resize(image: Image.Image, pipeline, max_area: int = 720 * 1280) -> Tuple[Image.Image, int, int]:
#     """
#     Resize image according to aspect ratio formula with pipeline-specific parameters.
    
#     Args:
#         image: Input PIL Image
#         pipeline: Loaded pipeline object
#         max_area: Maximum area constraint
    
#     Returns:
#         Tuple of (resized_image, height, width)
#     """
#     aspect_ratio = image.height / image.width
#     mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
#     height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
#     width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
#     image = image.resize((width, height))
#     return image, height, width


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
    """Clear GPU memory cache comprehensively."""
    if torch.cuda.is_available():
        # Clear PyTorch CUDA cache
        torch.cuda.empty_cache()
        
        # Collect IPC memory
        torch.cuda.ipc_collect()
        
        # Force synchronization to ensure memory is freed
        torch.cuda.synchronize()
        
        # Reset peak memory stats to free reserved memory
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("GPU memory cache cleared comprehensively")
    else:
        # Still run garbage collection even without CUDA
        import gc
        gc.collect()
        logger.info("System memory cleared (no CUDA available)")


def force_free_unallocated_memory():
    """Force free unallocated PyTorch memory more aggressively."""
    if torch.cuda.is_available():
        # Get current memory stats
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        
        logger.info(f"Before clearing - Allocated: {allocated_before/1024**3:.2f} GB, Reserved: {reserved_before/1024**3:.2f} GB")
        
        # Clear all caches
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection multiple times
        import gc
        for _ in range(3):
            gc.collect()
        
        # Additional aggressive clearing
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
        # Get memory stats after clearing
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        
        freed_allocated = allocated_before - allocated_after
        freed_reserved = reserved_before - reserved_after
        
        logger.info(f"After clearing - Allocated: {allocated_after/1024**3:.2f} GB, Reserved: {reserved_after/1024**3:.2f} GB")
        logger.info(f"Freed {freed_allocated/1024**3:.2f} GB allocated memory")
        logger.info(f"Freed {freed_reserved/1024**3:.2f} GB reserved memory")
        
        return freed_allocated, freed_reserved
    else:
        import gc
        gc.collect()
        return 0, 0


def clear_unallocated_memory():
    """Specifically target unallocated PyTorch memory."""
    if torch.cuda.is_available():
        # Get current stats
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        unallocated = reserved - allocated
        
        logger.info(f"Current unallocated memory: {unallocated/1024**3:.2f} GB")
        
        if unallocated > 1024**3:  # More than 1GB unallocated
            logger.info("Large amount of unallocated memory detected, attempting to free...")
            
            # Try to force memory release
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Reset memory pool
            torch.cuda.reset_peak_memory_stats()
            
            # Multiple garbage collection passes
            import gc
            for i in range(5):
                gc.collect()
                torch.cuda.empty_cache()
            
            # Check if we freed any memory
            new_allocated = torch.cuda.memory_allocated()
            new_reserved = torch.cuda.memory_reserved()
            new_unallocated = new_reserved - new_allocated
            
            freed = unallocated - new_unallocated
            logger.info(f"Freed {freed/1024**3:.2f} GB of unallocated memory")
            logger.info(f"Remaining unallocated: {new_unallocated/1024**3:.2f} GB")
            
            return freed
        else:
            logger.info("No significant unallocated memory to free")
            return 0
    else:
        return 0


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