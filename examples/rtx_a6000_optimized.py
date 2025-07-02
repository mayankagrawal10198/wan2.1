#!/usr/bin/env python3
"""
RTX A6000 Optimized Example for Wan2.1 I2V Model
This example shows how to use the pipeline optimized for RTX A6000 with 48GB VRAM.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from wan21_pipeline import Wan21Pipeline
from utils import setup_directories, check_gpu_memory
from config import RTX_A6000_SETTINGS


def main():
    """RTX A6000 optimized video generation example."""
    
    # Setup directories
    setup_directories()
    
    # Check GPU memory
    gpu_info = check_gpu_memory()
    if gpu_info:
        print(f"🎮 GPU: {gpu_info['gpu_name']}")
        print(f"💾 Total VRAM: {gpu_info['total_memory']:.1f} GB")
        print(f"🆓 Free VRAM: {gpu_info['free_memory']:.1f} GB")
    
    # Example image path
    image_path = "input/example.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        print("Please place an image file in the 'input' directory.")
        return
    
    print("🚀 Starting RTX A6000 optimized video generation...")
    print(f"⚙️ Using settings: {RTX_A6000_SETTINGS}")
    
    start_time = time.time()
    
    # Initialize pipeline with RTX A6000 optimizations
    with Wan21Pipeline(
        torch_dtype="bfloat16",  # Optimal for RTX A6000
        enable_optimizations=True,
        enable_attention_slicing=True,
        enable_vae_slicing=True,
        enable_model_cpu_offload=True,
        enable_sequential_cpu_offload=False  # Not needed for RTX A6000
    ) as pipeline:
        
        # Method 1: Use the optimized method
        print("📹 Method 1: Using RTX A6000 optimized method...")
        output_path_1 = pipeline.generate_video_rtx_a6000_optimized(
            image_path=image_path,
            prompt="Cinematic quality scene with smooth motion and professional lighting",
            negative_prompt="static, blurred, low quality, distorted, ugly, bad anatomy, watermark"
        )
        
        print(f"✅ Video 1 generated: {output_path_1}")
        
        # Method 2: Custom settings for maximum quality
        print("📹 Method 2: Using custom high-quality settings...")
        output_path_2 = pipeline.generate_video(
            image_path=image_path,
            prompt="Professional cinematic scene with dynamic movement and atmospheric lighting",
            negative_prompt="static, blurred, low quality, distorted, ugly, watermark, text",
            num_frames=81,  # Full frame count for RTX A6000
            guidance_scale=6.0,  # Higher guidance for better quality
            num_inference_steps=60,  # More steps for better quality
            fps=16
        )
        
        print(f"✅ Video 2 generated: {output_path_2}")
        
        # Method 3: Custom resolution (720P)
        print("📹 Method 3: Generating 720P video...")
        output_path_3 = pipeline.generate_video(
            image_path=image_path,
            prompt="High-resolution cinematic scene with smooth camera movement",
            negative_prompt="static, blurred, low quality, distorted, ugly",
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=50,
            height=720,
            width=1280,
            fps=16
        )
        
        print(f"✅ Video 3 generated: {output_path_3}")
    
    total_time = time.time() - start_time
    print(f"⏱️ Total generation time: {total_time:.1f} seconds")
    print(f"📊 Average time per video: {total_time/3:.1f} seconds")
    
    # Show final GPU memory status
    final_gpu_info = check_gpu_memory()
    if final_gpu_info:
        print(f"🆓 Final free VRAM: {final_gpu_info['free_memory']:.1f} GB")


if __name__ == "__main__":
    main() 