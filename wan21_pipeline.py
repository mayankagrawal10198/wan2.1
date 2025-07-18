"""
Wan2.1 I2V Pipeline
Optimized for RTX A6000 with 48GB VRAM
"""

import os
import time
import logging
import torch
import gc
import cv2
import numpy as np
import imageio
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from PIL import Image
import peft
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanVACEPipeline as DiffusersWanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

from config import (
    DEFAULT_FPS, DEFAULT_MODEL_ID, DEFAULT_TORCH_DTYPE, DEFAULT_DEVICE,
    DEFAULT_NUM_FRAMES, DEFAULT_GUIDANCE_SCALE, DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT, LORA_GUIDANCE_SCALE, LORA_NUM_INFERENCE_STEPS,
    ENABLE_LORA, CAUSVID_LORA_PATH, CAUSVID_LORA_FILENAME, CAUSVID_ADAPTER_NAME, CAUSVID_STRENGTH
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
        revision: str = "main",
        # LoRA configuration parameters
        enable_lora: bool = ENABLE_LORA,
        lora_path: str = CAUSVID_LORA_PATH,
        lora_filename: str = CAUSVID_LORA_FILENAME,
        lora_adapter_name: str = CAUSVID_ADAPTER_NAME,
        lora_strength: float = CAUSVID_STRENGTH
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
            enable_lora: Enable LoRA loading
            lora_path: LoRA model path or ID
            lora_filename: LoRA filename (for subfolder access)
            lora_adapter_name: LoRA adapter name
            lora_strength: LoRA strength/weight (0.25-1.0)
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
        
        # LoRA configuration
        self.enable_lora = enable_lora
        self.lora_path = lora_path
        self.lora_filename = lora_filename
        self.lora_adapter_name = lora_adapter_name
        self.lora_strength = lora_strength
        self.lora_loaded_successfully = False  # Track if LoRA was successfully loaded
        
        # Initialize components
        self.pipe = None
        self.image_encoder = None
        self.vae = None
        
        # Setup directories
        setup_directories()
        
        # Log system information
        log_system_info()
        
        logger.info("Wan21Pipeline initialized successfully")
        if self.enable_lora:
            logger.info(f"LoRA enabled: {self.lora_path}/{self.lora_filename} (strength: {self.lora_strength})")
    
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
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True
            )
            
            # Load VAE
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                local_model_path,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True
            )
            
            # Load main pipeline
            logger.info("Loading main pipeline...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                local_model_path,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=self.torch_dtype,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True
            )
            
            # Load CausVid LoRA (if enabled)
            if self.enable_lora:
                logger.info(f"Loading CausVid LoRA: {self.lora_path}/{self.lora_filename}")
                try:
                    # Load LoRA weights following the reference pattern
                    self.pipe.load_lora_weights(
                        self.lora_path, 
                        weight_name=self.lora_filename,
                        adapter_name=self.lora_adapter_name
                    )
                    # Set adapter with strength as a float (not in a list)
                    self.pipe.set_adapters(self.lora_adapter_name, self.lora_strength)
                    logger.info(f"LoRA loaded successfully with strength: {self.lora_strength}")
                    self.lora_loaded_successfully = True
                except Exception as e:
                    logger.warning(f"Failed to load LoRA: {e}")
                    logger.warning("Continuing without LoRA...")
                    self.enable_lora = False
                    self.lora_loaded_successfully = False
            
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
        fps: int = DEFAULT_FPS,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate video from input image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for video generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate
            guidance_scale: Guidance scale for generation (overridden to LORA_GUIDANCE_SCALE when LoRA is enabled)
            num_inference_steps: Number of inference steps (overridden to LORA_NUM_INFERENCE_STEPS when LoRA is enabled)
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
        
        # Apply LoRA optimized defaults only when LoRA was successfully loaded
        if self.enable_lora and self.lora_loaded_successfully:
            # Override with LoRA-optimized settings from config
            guidance_scale = LORA_GUIDANCE_SCALE  # CFG Scale for LoRA
            num_inference_steps = LORA_NUM_INFERENCE_STEPS  # Inference Steps for LoRA
            logger.info(f"LoRA enabled - using optimized settings: guidance_scale={guidance_scale}, steps={num_inference_steps}")
        
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
    

    
    def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up resources...")
        
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
        info = {
            "model_id": self.model_id,
            "torch_dtype": str(self.torch_dtype),
            "device": self.device,
            "enable_optimizations": self.enable_optimizations,
            "enable_attention_slicing": self.enable_attention_slicing,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
            "enable_lora": self.enable_lora
        }
        
        if self.enable_lora:
            info.update({
                "lora_path": self.lora_path,
                "lora_filename": self.lora_filename,
                "lora_adapter_name": self.lora_adapter_name,
                "lora_strength": self.lora_strength
            })
        
        return info


class WanVACEPipelineWrapper:
    """
    Wan2.1 VACE Pipeline for video-guided generation
    """
    
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-VACE-14B-diffusers",
        torch_dtype: str = DEFAULT_TORCH_DTYPE,
        device: str = DEFAULT_DEVICE,
        enable_optimizations: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        revision: str = "main",
        # LoRA configuration parameters
        enable_lora: bool = ENABLE_LORA,
        lora_path: str = CAUSVID_LORA_PATH,
        lora_filename: str = CAUSVID_LORA_FILENAME,
        lora_adapter_name: str = CAUSVID_ADAPTER_NAME,
        lora_strength: float = CAUSVID_STRENGTH
    ):
        """
        Initialize Wan2.1 VACE Pipeline
        
        Args:
            model_id: Hugging Face model ID
            torch_dtype: Torch data type (bfloat16, float16, float32)
            device: Device to run on (cuda, cpu)
            enable_optimizations: Enable memory optimizations
            cache_dir: Cache directory for model downloads
            local_files_only: Use only local files
            revision: Model revision
            enable_lora: Enable LoRA loading
            lora_path: LoRA model path or ID
            lora_filename: LoRA filename (for subfolder access)
            lora_adapter_name: LoRA adapter name
            lora_strength: LoRA strength/weight (0.25-1.0)
        """
        self.model_id = model_id
        self.torch_dtype = get_torch_dtype(torch_dtype)
        self.device = device
        self.enable_optimizations = enable_optimizations
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.revision = revision
        self.enable_sequential_cpu_offload = False  # Will be set during optimization
        
        # LoRA configuration
        self.enable_lora = enable_lora
        self.lora_path = lora_path
        self.lora_filename = lora_filename
        self.lora_adapter_name = lora_adapter_name
        self.lora_strength = lora_strength
        self.lora_loaded_successfully = False  # Track if LoRA was successfully loaded
        
        # Initialize components
        self.pipe = None
        self.vae = None
        
        # Setup directories
        setup_directories()
        
        # Log system information
        log_system_info()
        
        logger.info("WanVACEPipelineWrapper initialized successfully")
        if self.enable_lora:
            logger.info(f"VACE LoRA enabled: {self.lora_path}/{self.lora_filename} (strength: {self.lora_strength})")
    
    def _get_local_model_path(self) -> str:
        """Get local model path for VACE model."""
        local_path = 'models/wan21-vace'
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local VACE model not found: {local_path}. Please run download_models.py first.")
        
        logger.info(f"Using local VACE model path: {local_path}")
        return local_path
    
    def load_model(self):
        """Load the Wan2.1 VACE model components."""
        try:
            logger.info(f"Loading VACE model: {self.model_id}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Torch dtype: {self.torch_dtype}")
            
            # Check GPU memory before loading
            gpu_info = check_gpu_memory()
            
            # Determine local model path
            local_model_path = self._get_local_model_path()
            
            # Load VAE
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                local_model_path,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True
            )
            
            # Load main VACE pipeline
            logger.info("Loading VACE pipeline...")
            self.pipe = DiffusersWanVACEPipeline.from_pretrained(
                local_model_path,
                vae=self.vae,
                torch_dtype=self.torch_dtype,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True
            )
            
            # Load CausVid LoRA for VACE (if enabled)
            if self.enable_lora:
                logger.info(f"Loading CausVid LoRA for VACE: {self.lora_path}/{self.lora_filename}")
                try:
                    # Load LoRA weights following the reference pattern
                    self.pipe.load_lora_weights(
                        self.lora_path, 
                        weight_name=self.lora_filename,
                        adapter_name=self.lora_adapter_name
                    )
                    # Set adapter with strength as a float (not in a list)
                    self.pipe.set_adapters(self.lora_adapter_name, self.lora_strength)
                    logger.info(f"VACE LoRA loaded successfully with strength: {self.lora_strength}")
                    self.lora_loaded_successfully = True
                except Exception as e:
                    logger.warning(f"Failed to load VACE LoRA: {e}")
                    logger.warning("Continuing VACE without LoRA...")
                    self.enable_lora = False
                    self.lora_loaded_successfully = False
            
            # Apply optimizations
            if self.enable_optimizations:
                self._apply_optimizations()
            
            # Move to device (only if not using CPU offloading)
            if not hasattr(self.pipe, 'enable_model_cpu_offload') or not self.enable_optimizations:
                self.pipe.to(self.device)
                logger.info(f"VACE pipeline moved to {self.device}")
            else:
                # When using CPU offloading, keep pipeline on CPU
                # Offloading will automatically move components to GPU when needed
                logger.info("Keeping VACE pipeline on CPU (model CPU offloading enabled)")
            
            logger.info("VACE model loaded successfully!")
            
            # Check GPU memory after loading
            check_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error loading VACE model: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply memory optimizations to the VACE pipeline."""
        logger.info("Applying VACE memory optimizations...")
        
        # Enable attention slicing if available
        if hasattr(self.pipe, 'enable_attention_slicing'):
            logger.info("Enabling attention slicing...")
            self.pipe.enable_attention_slicing()
        
        # Enable VAE slicing if available
        if hasattr(self.pipe.vae, 'enable_slicing'):
            logger.info("Enabling VAE slicing...")
            self.pipe.vae.enable_slicing()
        
        # Enable model CPU offloading for VACE
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            logger.info("Enabling model CPU offload for VACE...")
            self.pipe.enable_model_cpu_offload()
        
        # Enable sequential CPU offloading for VACE
        # Disabled due to GPU compatibility issues
        # if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
        #     logger.info("Enabling sequential CPU offload for VACE...")
        #     self.pipe.enable_sequential_cpu_offload()
        #     self.enable_sequential_cpu_offload = True
        
        logger.info("VACE memory optimizations applied successfully")
    
    def extract_video_frames(self, video_path: str, num_frames: int = 81) -> List[Image.Image]:
        """Extract frames from video file using sequential extraction for reliability."""
        logger.info(f"Extracting {num_frames} frames from video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties for logging
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {total_frames} frames, {fps} fps, {width}x{height}")
        
        # Always use sequential frame extraction for reliability
        frames = []
        frame_count = 0
        max_attempts = num_frames * 3  # Try up to 3x the requested frames to ensure we get enough
        
        # Calculate skip interval for even distribution
        if total_frames > 0 and total_frames > num_frames:
            skip_interval = max(1, total_frames // num_frames)
            logger.info(f"Using skip interval of {skip_interval} frames for even distribution")
        else:
            skip_interval = 1
            logger.info("Extracting all available frames")
        
        while len(frames) < num_frames and frame_count < max_attempts:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"Reached end of video after {frame_count} frames")
                break
            
            # Add frame if we haven't reached our target count
            if len(frames) < num_frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
            
            # Skip frames for even distribution (except for the last frame)
            if skip_interval > 1 and len(frames) < num_frames:
                for _ in range(skip_interval - 1):
                    skip_ret = cap.read()
                    if not skip_ret:
                        break
                    frame_count += 1
        
        cap.release()
        
        logger.info(f"Sequential extraction: processed {frame_count} frames, extracted {len(frames)} frames")
        
        # Ensure we have at least 2 frames
        if len(frames) < 2:
            logger.warning(f"OpenCV sequential extraction failed. Trying with imageio...")
            return self._extract_frames_with_imageio(video_path, num_frames)
        
        return frames
    
    def _extract_frames_with_imageio(self, video_path: str, num_frames: int = 81) -> List[Image.Image]:
        """Extract frames using imageio with sequential approach as fallback."""
        logger.info(f"Extracting {num_frames} frames using imageio: {video_path}")
        
        try:
            # Read video with imageio
            video = imageio.get_reader(video_path)
            total_frames = len(video)
            logger.info(f"Imageio video info: {total_frames} frames")
            
            if total_frames < 2:
                raise ValueError(f"Video has too few frames: {total_frames}")
            
            # Use sequential extraction for consistency
            frames = []
            frame_count = 0
            max_attempts = min(total_frames, num_frames * 3)
            
            # Calculate skip interval for even distribution
            if total_frames > num_frames:
                skip_interval = max(1, total_frames // num_frames)
                logger.info(f"Imageio using skip interval of {skip_interval} frames")
            else:
                skip_interval = 1
                logger.info("Imageio extracting all available frames")
            
            for frame_idx in range(0, min(total_frames, max_attempts), skip_interval):
                if len(frames) >= num_frames:
                    break
                
                try:
                    frame = video.get_data(frame_idx)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)
                    frame_count += 1
                except Exception as e:
                    logger.warning(f"Failed to read frame {frame_idx}: {e}")
                    continue
            
            video.close()
            
            logger.info(f"Imageio sequential extraction: processed {frame_count} frames, extracted {len(frames)} frames")
            
            if len(frames) < 2:
                raise ValueError(f"Could not extract enough frames with imageio. Got {len(frames)} frames.")
            
            return frames
            
        except Exception as e:
            logger.error(f"Imageio extraction failed: {e}")
            raise ValueError(f"Failed to extract frames from video with both OpenCV and imageio: {e}")
    
    def prepare_video_and_mask(self, first_img: Image.Image, last_img: Image.Image, 
                              height: int, width: int, num_frames: int) -> tuple:
        """Prepare video and mask for VACE pipeline."""
        logger.info(f"Preparing video and mask: {width}x{height}, {num_frames} frames")
        
        # Resize images to target dimensions
        first_img = first_img.resize((width, height))
        last_img = last_img.resize((width, height))
        
        # Create video frames
        frames = []
        frames.append(first_img)
        
        # Add intermediate frames (gray)
        for _ in range(num_frames - 2):
            gray_frame = Image.new("RGB", (width, height), (128, 128, 128))
            frames.append(gray_frame)
        
        frames.append(last_img)
        
        # Create mask (black for first/last frame, white for intermediate)
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)
        
        mask = [mask_black]
        mask.extend([mask_white] * (num_frames - 2))
        mask.append(mask_black)
        
        return frames, mask
    
    def generate_video_with_guidance(
        self,
        image_path: str,
        video_path: Optional[str] = None,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_path: Optional[str] = None,
        fps: int = DEFAULT_FPS,
        seed: Optional[int] = None,
        conditioning_scale: float = 1.0
    ) -> str:
        """
        Generate video from input image with optional video guidance
        
        Args:
            image_path: Path to input image
            video_path: Path to guidance video (optional - if None, uses image-only generation)
            prompt: Text prompt for video generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate
            guidance_scale: Guidance scale for generation (overridden to LORA_GUIDANCE_SCALE when LoRA is enabled)
            num_inference_steps: Number of inference steps (overridden to LORA_NUM_INFERENCE_STEPS when LoRA is enabled)
            height: Output height (auto-calculated if None)
            width: Output width (auto-calculated if None)
            output_path: Output video path (auto-generated if None)
            fps: Frames per second
            seed: Random seed for reproducibility
            conditioning_scale: Conditioning scale for VACE
        
        Returns:
            Path to generated video
        """
        # Validate inputs
        if not validate_image_path(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        
        # Handle optional video path
        if video_path is not None:
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise ValueError(f"Video file is empty: {video_path}")
            
            logger.info(f"Video file size: {file_size / (1024*1024):.1f} MB")
        else:
            logger.info("No video guidance provided - using image-only generation")
        
        # Load model if not loaded
        if self.pipe is None:
            self.load_model()
        
        # Apply LoRA optimized defaults only when LoRA was successfully loaded
        if self.enable_lora and self.lora_loaded_successfully:
            # Override with LoRA-optimized settings from config
            guidance_scale = LORA_GUIDANCE_SCALE  # CFG Scale for LoRA
            num_inference_steps = LORA_NUM_INFERENCE_STEPS  # Inference Steps for LoRA
            logger.info(f"VACE LoRA enabled - using optimized settings: guidance_scale={guidance_scale}, steps={num_inference_steps}")
        
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
        
        # Memory-aware parameter adjustment for VACE
        # VACE is more memory-intensive, so reduce parameters if needed
        original_num_frames = num_frames
        if height >= 720 or width >= 720:
            # For high resolution, reduce frames to save memory
            num_frames = min(num_frames, 61)  # Reduce from 121 to 61 for high res
            logger.info(f"Reduced frames from {original_num_frames} to {num_frames} for memory optimization")
        
        # Generate output path
        if output_path is None:
            output_path = get_output_filename(image_path, suffix="_vace")
            output_path = os.path.join("output", output_path)
        
        # Set flow shift based on resolution
        flow_shift = 5.0 if height >= 720 else 3.0
        logger.info(f"Setting flow shift to {flow_shift} for {height}p resolution")
        
        # Update scheduler with flow shift
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, 
            flow_shift=flow_shift
        )
        
        # Generate video
        logger.info("Starting VACE video generation...")
        
        # Check available memory before generation
        gpu_info = check_gpu_memory()
        if gpu_info and gpu_info['free_memory'] < 5.0:  # Less than 5GB free
            logger.warning(f"Low GPU memory available: {gpu_info['free_memory']:.1f} GB")
            logger.warning("Applying additional memory optimizations...")
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                if video_path is not None:
                    # Extract frames from guidance video
                    logger.info(f"Extracting frames from guidance video: {video_path}")
                    try:
                        video_frames = self.extract_video_frames(video_path, num_frames)
                    except Exception as e:
                        logger.error(f"Failed to extract frames from video: {e}")
                        raise ValueError(f"Video processing failed. Please ensure the video file is valid and not corrupted. Error: {e}")
                    
                    if len(video_frames) < 2:
                        raise ValueError("Video must have at least 2 frames")
                    
                    # Prepare video and mask for VACE
                    first_frame = video_frames[0]
                    last_frame = video_frames[-1]
                    video, mask = self.prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)
                    
                    output = self.pipe(
                        video=video,
                        mask=mask,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        conditioning_scale=conditioning_scale,
                        generator=torch.Generator().manual_seed(seed) if seed else None
                    ).frames[0]  # frames[0] gets the first (and only) video from output
                else:
                    # Image-only generation (no video guidance)
                    logger.info("Using image-only generation without video guidance")
                    output = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        reference_images=[image],  # Pass image as reference_images
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=torch.Generator().manual_seed(seed) if seed else None
                    ).frames[0]  # frames[0] gets the first (and only) video from output
            
            # Export video
            logger.info(f"Exporting VACE video to: {output_path}")
            export_to_video(output, output_path, fps=fps)
            
            generation_time = time.time() - start_time
            logger.info(f"VACE video generation completed in {format_time(generation_time)}")
            logger.info(f"Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error during VACE video generation: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up VACE resources...")
        
        # Delete pipeline components
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.vae is not None:
            del self.vae
            self.vae = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("VACE cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get VACE model information."""
        info = {
            "model_id": self.model_id,
            "torch_dtype": str(self.torch_dtype),
            "device": self.device,
            "enable_optimizations": self.enable_optimizations,
            "enable_lora": self.enable_lora
        }
        
        if self.enable_lora:
            info.update({
                "lora_path": self.lora_path,
                "lora_filename": self.lora_filename,
                "lora_adapter_name": self.lora_adapter_name,
                "lora_strength": self.lora_strength
            })
        
        return info 