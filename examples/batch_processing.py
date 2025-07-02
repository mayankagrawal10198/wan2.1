#!/usr/bin/env python3
"""
Batch Processing Example for Wan2.1 I2V Model
This example shows how to process multiple images in batch.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from wan21_pipeline import Wan21Pipeline
from utils import setup_directories, validate_image_path, SUPPORTED_IMAGE_FORMATS


class BatchProcessor:
    """Batch processor for multiple images."""
    
    def __init__(self, pipeline: Wan21Pipeline):
        self.pipeline = pipeline
        self.results = []
    
    def process_directory(
        self,
        input_dir: str = "input",
        output_dir: str = "output",
        prompt: str = "A beautiful scene with gentle movement",
        negative_prompt: str = "static, blurred, low quality, distorted, ugly",
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output videos
            prompt: Text prompt for all videos
            negative_prompt: Negative prompt for all videos
            num_frames: Number of frames to generate
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            **kwargs: Additional arguments for generate_video
        
        Returns:
            List of results with input path, output path, and status
        """
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            return []
        
        # Get all image files
        image_files = []
        for ext in SUPPORTED_IMAGE_FORMATS:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No supported image files found in {input_dir}")
            return []
        
        print(f"📁 Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔄 Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                # Generate output path
                output_filename = f"{image_path.stem}_video.mp4"
                output_path = os.path.join(output_dir, output_filename)
                
                # Generate video
                start_time = time.time()
                actual_output_path = self.pipeline.generate_video(
                    image_path=str(image_path),
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_path=output_path,
                    **kwargs
                )
                
                generation_time = time.time() - start_time
                
                # Record result
                result = {
                    "input_path": str(image_path),
                    "output_path": actual_output_path,
                    "status": "success",
                    "generation_time": generation_time,
                    "file_size": os.path.getsize(actual_output_path) / (1024 * 1024) if os.path.exists(actual_output_path) else 0
                }
                
                print(f"✅ Success: {image_path.name} -> {actual_output_path}")
                print(f"⏱️ Time: {generation_time:.1f}s, Size: {result['file_size']:.1f}MB")
                
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                result = {
                    "input_path": str(image_path),
                    "output_path": None,
                    "status": "error",
                    "error": str(e),
                    "generation_time": 0
                }
            
            self.results.append(result)
        
        return self.results
    
    def print_summary(self):
        """Print summary of batch processing results."""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Successful: {len(successful)}/{len(self.results)}")
        print(f"❌ Failed: {len(failed)}/{len(self.results)}")
        
        if successful:
            total_time = sum(r["generation_time"] for r in successful)
            total_size = sum(r["file_size"] for r in successful)
            avg_time = total_time / len(successful)
            
            print(f"⏱️ Total time: {total_time:.1f}s")
            print(f"📊 Average time per video: {avg_time:.1f}s")
            print(f"💾 Total output size: {total_size:.1f}MB")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"  - {Path(result['input_path']).name}: {result.get('error', 'Unknown error')}")


def main():
    """Main function for batch processing example."""
    
    # Setup directories
    setup_directories()
    
    # Check if input directory has images
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please create an 'input' directory and add some images.")
        return
    
    # Count images
    image_count = 0
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_count += len(list(Path(input_dir).glob(f"*{ext}")))
        image_count += len(list(Path(input_dir).glob(f"*{ext.upper()}")))
    
    if image_count == 0:
        print(f"❌ No supported images found in {input_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}")
        return
    
    print(f"🚀 Starting batch processing of {image_count} images...")
    
    # Initialize pipeline
    with Wan21Pipeline(
        torch_dtype="bfloat16",
        enable_optimizations=True
    ) as pipeline:
        
        # Create batch processor
        processor = BatchProcessor(pipeline)
        
        # Process all images with different prompts
        prompts = [
            "A beautiful scene with gentle movement and cinematic quality",
            "Dynamic scene with smooth camera motion and professional lighting",
            "Atmospheric scene with subtle movement and artistic composition"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🎬 Batch {i}/3: {prompt}")
            
            results = processor.process_directory(
                input_dir=input_dir,
                output_dir="output",
                prompt=prompt,
                negative_prompt="static, blurred, low quality, distorted, ugly, watermark",
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=50
            )
        
        # Print summary
        processor.print_summary()


if __name__ == "__main__":
    main() 