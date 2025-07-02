#!/usr/bin/env python3
"""
Simple Example for Wan2.1 I2V Model
This example shows the basic usage with default settings.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from wan21_pipeline import Wan21Pipeline
from utils import setup_directories


def main():
    """Simple example of video generation with default settings."""
    
    # Setup directories
    setup_directories()
    
    # Example image path (you need to provide your own image)
    image_path = "input/example.jpg"
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        print("Please place an image file in the 'input' directory and update the image_path variable.")
        return
    
    print("🚀 Starting simple video generation example...")
    
    # Initialize pipeline with default settings
    with Wan21Pipeline() as pipeline:
        # Generate video with default settings
        output_path = pipeline.generate_video(
            image_path=image_path,
            prompt="A beautiful scene with gentle movement and cinematic quality",
            negative_prompt="static, blurred, low quality, distorted, ugly"
        )
        
        print(f"✅ Video generated successfully!")
        print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main() 