#!/usr/bin/env python3
"""
Example: Wan2.1 VACE Pipeline for Video-Guided Generation
Demonstrates how to use the VACE pipeline to generate videos with movement guidance
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from wan21_pipeline import WanVACEPipelineWrapper
from config import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

def main():
    """Example usage of VACE pipeline."""
    
    # Example paths (you'll need to provide your own files)
    image_path = "examples/sample_image.jpg"  # Replace with your image
    video_path = "examples/sample_video.mp4"  # Replace with your video
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("Please provide a valid image file path")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please provide a valid video file path")
        return
    
    print("🎬 Wan2.1 VACE Pipeline Example")
    print("=" * 50)
    print(f"Input Image: {image_path}")
    print(f"Guidance Video: {video_path}")
    print(f"Prompt: {DEFAULT_PROMPT}")
    print(f"Negative Prompt: {DEFAULT_NEGATIVE_PROMPT}")
    print("=" * 50)
    
    try:
        # Initialize VACE pipeline
        print("🚀 Initializing VACE pipeline...")
        with WanVACEPipelineWrapper() as pipeline:
            print("✅ VACE pipeline initialized successfully")
            
            # Generate video with guidance
            print("🎬 Generating video with video guidance...")
            output_path = pipeline.generate_video_with_guidance(
                image_path=image_path,
                video_path=video_path,
                prompt=DEFAULT_PROMPT,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=480,  # 480p resolution
                width=832,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=30,
                conditioning_scale=1.0,
                seed=42  # For reproducibility
            )
            
            print(f"✅ Video generated successfully!")
            print(f"📁 Output saved to: {output_path}")
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.1f} MB")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎉 VACE example completed successfully!")
    print("💡 You can now use the web interface to upload images and videos for guided generation.")

if __name__ == "__main__":
    main() 