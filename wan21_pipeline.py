"""
Wan2.1 I2V Pipeline
Optimized for RTX A6000 with 48GB VRAM
"""

import os
import time
import logging
import torch
import gc
from typing import Optional, Dict, Any, Union
from pathlib import Path

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

from config import (
    DEFAULT_MODEL_ID, DEFAULT_TORCH_DTYPE, DEFAULT_DEVICE,
    DEFAULT_NUM_FRAMES, DEFAULT_GUIDANCE_SCALE, DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT
)
from utils import (
    setup_directories, validate_image_path, load_and_preprocess_image,
    calculate_optimal_dimensions, get_torch_dtype, check_gpu_memory,
    clear_gpu_memory, format_time, get_output_filename, validate_model_parameters,
    log_system_info
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Wan21Pipeline:
    """
    Wan2.1 I2V Pipeline optimized for RTX A6000
    """
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        torch_dtype: str = DEFAULT_TORCH_DTYPE,
        device: str = DEFAULT_DEVICE,
        enable_optimizations: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_model_cpu_offload: bool = True,
        enable_sequential_cpu_offload: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        revision: str = "main"
    ):
        """
        Initialize Wan2.1 I2V Pipeline
        
        Args:
            model_id: Hugging Face model ID
            torch_dtype: Torch data type (bfloat16, float16, float32)
            device: Device to run on (cuda, cpu)
            enable_optimizations: Enable memory optimizations
            enable_attention_slicing: Enable attention slicing
            enable_vae_slicing: Enable VAE slicing
            enable_model_cpu_offload: Enable model CPU offloading
            enable_sequential_cpu_offload: Enable sequential CPU offloading
            cache_dir: Cache directory for model downloads
            local_files_only: Use only local files
            revision: Model revision
        """
        self.model_id = model_id
        self.torch_dtype = get_torch_dtype(torch_dtype)
        self.device = device
        self.enable_optimizations = enable_optimizations
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.revision = revision
        
        # Initialize components
        self.pipe = None
        self.image_encoder = None
        self.vae = None
        
        # Setup directories
        setup_directories()
        
        # Log system information
        log_system_info()
        
        logger.info("Wan21Pipeline initialized successfully")
    
    def _get_local_model_path(self) -> str:
        """Get local model path based on model_id."""
        # Map model IDs to local paths
        model_paths = {
            'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers': 'models/wan21-480p',
            'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers': 'models/wan21-720p'
        }
        
        local_path = model_paths.get(self.model_id)
        if not local_path:
            raise ValueError(f"Unknown model_id: {self.model_id}")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local model not found: {local_path}. Please run download_models.py first.")
        
        logger.info(f"Using local model path: {local_path}")
        return local_path
    
    def load_model(self):
        """Load the Wan2.1 I2V model components."""
        try:
            logger.info(f"Loading model: {self.model_id}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Torch dtype: {self.torch_dtype}")
            
            # Check GPU memory before loading
            gpu_info = check_gpu_memory()
            
            # Determine local model path based on model_id
            local_model_path = self._get_local_model_path()
            
            # Load image encoder
            logger.info("Loading image encoder...")
            self.image_encoder = CLIPVisionModel.from_pretrained(
                local_model_path,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # Load VAE
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                local_model_path,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # Load main pipeline
            logger.info("Loading main pipeline...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                local_model_path,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=self.torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # Apply optimizations
            if self.enable_optimizations:
                self._apply_optimizations()
            
            # Move to device (only if not using CPU offloading)
            if not self.enable_model_cpu_offload:
                self.pipe.to(self.device)
            else:
                # When using CPU offloading, keep pipeline on CPU
                # Offloading will automatically move components to GPU when needed
                logger.info("Keeping pipeline on CPU (model CPU offloading enabled)")
            
            logger.info("Model loaded successfully!")
            
            # Check GPU memory after loading
            check_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply memory optimizations to the pipeline."""
        logger.info("Applying memory optimizations...")
        
        if self.enable_model_cpu_offload:
            logger.info("Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload()
        
        if self.enable_attention_slicing:
            logger.info("Enabling attention slicing...")
            self.pipe.enable_attention_slicing()
        
        if self.enable_sequential_cpu_offload:
            logger.info("Enabling sequential CPU offload...")
            self.pipe.enable_sequential_cpu_offload()
        
        if self.enable_vae_slicing and hasattr(self.pipe.vae, 'enable_slicing'):
            logger.info("Enabling VAE slicing...")
            self.pipe.vae.enable_slicing()
        
        logger.info("Memory optimizations applied successfully")
    
    def generate_video(
        self,
        image_path: str,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_path: Optional[str] = None,
        fps: int = 16,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate video from input image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for video generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            height: Output height (auto-calculated if None)
            width: Output width (auto-calculated if None)
            output_path: Output video path (auto-generated if None)
            fps: Frames per second
            seed: Random seed for reproducibility
        
        Returns:
            Path to generated video
        """
        # Validate inputs
        if not validate_image_path(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        
        if not validate_model_parameters(num_frames, guidance_scale, num_inference_steps):
            raise ValueError("Invalid model parameters")
        
        # Load model if not loaded
        if self.pipe is None:
            self.load_model()
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            logger.info(f"Set random seed: {seed}")
        
        # Load and preprocess image
        logger.info(f"Loading image: {image_path}")
        image = load_and_preprocess_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Calculate dimensions
        if height is None or width is None:
            height, width = calculate_optimal_dimensions(image)
            logger.info(f"Calculated dimensions: {width}x{height}")
        
        # Resize image to target dimensions
        image = image.resize((width, height))
        
        # Generate output path
        if output_path is None:
            output_path = get_output_filename(image_path)
            output_path = os.path.join("output", output_path)
        
        # Generate video
        logger.info("Starting video generation...")
        start_time = time.time()
        
        try:
            with torch.no_grad():
                output = self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).frames[0]
            
            # Export video
            logger.info(f"Exporting video to: {output_path}")
            export_to_video(output, output_path, fps=fps)
            
            generation_time = time.time() - start_time
            logger.info(f"Video generation completed in {format_time(generation_time)}")
            logger.info(f"Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            raise
        finally:
            # Clear GPU memory
            clear_gpu_memory()
    

    
    def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up resources...")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Delete pipeline components
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.image_encoder is not None:
            del self.image_encoder
            self.image_encoder = None
        
        if self.vae is not None:
            del self.vae
            self.vae = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "torch_dtype": str(self.torch_dtype),
            "device": self.device,
            "enable_optimizations": self.enable_optimizations,
            "enable_attention_slicing": self.enable_attention_slicing,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload
        } 