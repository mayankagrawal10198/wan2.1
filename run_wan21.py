#!/usr/bin/env python3
"""
Main script for running Wan2.1 I2V Model
Usage: python run_wan21.py --image path/to/image.jpg --prompt "Your description"
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from wan21_pipeline import Wan21Pipeline
from utils import log_system_info, check_gpu_memory
from config import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Wan2.1 I2V Model for image-to-video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_wan21.py --image input.jpg --prompt "A beautiful scene with gentle movement"
  python run_wan21.py --image input.jpg --prompt "Cinematic scene" --num_frames 81 --guidance_scale 5.0
  python run_wan21.py --image input.jpg --prompt "High quality scene" --rtx_a6000
  python run_wan21.py --image input.jpg --prompt "Scene" --output output_video.mp4
        """
    )

    # Required arguments
    parser.add_argument("--image", "-i", type=str, help="Path to input image file")

    # Optional arguments
    parser.add_argument("--prompt", "-p", type=str, default=DEFAULT_PROMPT, help=f"Text prompt (default: '{DEFAULT_PROMPT}')")
    parser.add_argument("--negative_prompt", "-np", type=str, default=DEFAULT_NEGATIVE_PROMPT, help=f"Negative prompt (default: '{DEFAULT_NEGATIVE_PROMPT}')")
    parser.add_argument("--output", "-o", type=str, help="Output video path")
    parser.add_argument("--num_frames", "-f", type=int, default=81, help="Number of frames (default: 81)")
    parser.add_argument("--guidance_scale", "-g", type=float, default=5.0, help="Guidance scale (default: 5.0)")
    parser.add_argument("--num_inference_steps", "-s", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--height", "-hi", type=int, help="Output height")
    parser.add_argument("--width", "-wi", type=int, help="Output width")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second (default: 16)")
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--torch_dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Torch dtype (default: bfloat16)")
    parser.add_argument("--disable_optimizations", action="store_true", help="Disable memory optimizations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--check_system", action="store_true", help="Check system requirements and exit")

    return parser.parse_args()


def check_system_requirements():
    """Check if system meets requirements."""
    logger.info("=== System Requirements Check ===")

    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        logger.error(f"❌ Python version {python_version.major}.{python_version.minor} not supported. Need Python 3.8+")
        return False

    # Check PyTorch and CUDA
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.version.cuda}")
            logger.info(f"✅ GPU: {torch.cuda.get_device_name()}")

            # Check GPU memory
            gpu_info = check_gpu_memory()
            if gpu_info and gpu_info["total_memory"] >= 6:
                logger.info(f"✅ GPU memory: {gpu_info['total_memory']:.1f} GB (sufficient)")
            else:
                logger.warning(f"⚠️ GPU memory: {gpu_info['total_memory']:.1f} GB (may need optimizations)")
        else:
            logger.warning("⚠️ CUDA not available - using CPU (very slow)")

    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False

    # Check required packages
    required_packages = {
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "ftfy": "ftfy",
        "imageio": "imageio",
        "numpy": "numpy",
        "pillow": "PIL",  # Pillow is imported as PIL
        "opencv-python": "cv2",  # OpenCV is imported as cv2
    }

    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not installed")

    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False

    logger.info("✅ All system requirements met!")
    return True


def main():
    """Main function."""
    args = parse_arguments()

    # Enforce --image unless running in --check_system mode
    if not args.check_system and not args.image:
        logger.error("❌ You must provide --image unless using --check_system")
        sys.exit(1)

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check system requirements if requested
    if args.check_system:
        if check_system_requirements():
            sys.exit(0)
        else:
            sys.exit(1)

    # Log system information
    log_system_info()

    # Validate input image
    if not os.path.exists(args.image):
        logger.error(f"Input image not found: {args.image}")
        sys.exit(1)

    try:
        # Initialize pipeline
        logger.info("Initializing Wan2.1 I2V Pipeline...")

        pipeline_kwargs = {
            "torch_dtype": args.torch_dtype,
            "enable_optimizations": not args.disable_optimizations,
        }

        with Wan21Pipeline(**pipeline_kwargs) as pipeline:
            # Generate video with user-provided settings
            output_path = pipeline.generate_video(
                image_path=args.image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                output_path=args.output,
                fps=args.fps,
                seed=args.seed
            )

            logger.info(f"✅ Video generation completed successfully!")
            logger.info(f"📁 Output saved to: {output_path}")

            # Show file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                logger.info(f"📊 File size: {file_size:.1f} MB")

    except KeyboardInterrupt:
        logger.info("⚠️ Video generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during video generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
