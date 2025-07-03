#!/usr/bin/env python3
"""
Download Wan2.1 I2V Models Locally
Downloads models to local directories to avoid SSL issues and improve performance
"""

import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    '480p': {
        'model_id': 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
        'local_path': 'models/wan21-480p'
    },
    '720p': {
        'model_id': 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers', 
        'local_path': 'models/wan21-720p'
    }
}

def download_model(model_id: str, local_path: str):
    """Download a model to local directory."""
    try:
        logger.info(f"Downloading {model_id} to {local_path}")
        
        # Create directory if it doesn't exist
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"✅ Successfully downloaded {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_id}: {e}")
        return False

def main():
    """Download all models."""
    logger.info("🚀 Starting Wan2.1 Model Download")
    logger.info("=" * 50)
    
    success_count = 0
    total_models = len(MODELS)
    
    for resolution, config in MODELS.items():
        logger.info(f"\n📦 Downloading {resolution} model...")
        if download_model(config['model_id'], config['local_path']):
            success_count += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Download Summary: {success_count}/{total_models} models downloaded successfully")
    
    if success_count == total_models:
        logger.info("🎉 All models downloaded successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Update your app.py to use local model paths")
        logger.info("2. Restart your Flask application")
    else:
        logger.error("⚠️ Some models failed to download. Check the logs above.")

if __name__ == "__main__":
    main() 