#!/usr/bin/env python3
"""
FastAPI Web Application for Wan2.1 I2V Model
Provides a web interface for video generation
"""

import os
import json
import logging
import torch
import gc
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn

from wan21_pipeline import Wan21Pipeline, WanVACEPipelineWrapper
from utils import setup_directories, clear_gpu_memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wan2.1 Video Generation API",
    description="API for generating videos from images using Wan2.1 I2V and VACE models",
    version="1.0.0"
)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Resolution presets with model IDs
RESOLUTION_PRESETS = {
    '480p': {
        'width': 320, 
        'height': 400,
        'model_id': 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers'
    },
    '720p': {
        'width': 576, 
        'height': 720,
        'model_id': 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers'
    }
}

# Pydantic models for request/response
class VideoGenerationRequest(BaseModel):
    resolution: str = "480p"
    positive_prompt: str = "A beautiful scene with gentle movement"
    negative_prompt: str = "static, blurred, low quality, distorted, ugly"

class GeneratedVideo(BaseModel):
    filename: str
    file_size_mb: float
    model_type: str
    description: str

class VideoGenerationResponse(BaseModel):
    success: bool
    message: str
    videos: List[GeneratedVideo]
    resolution: str
    width: int
    height: int
    model_id: str
    video_guidance: bool

class HealthResponse(BaseModel):
    status: str
    message: str

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def setup_app_directories():
    """Setup necessary directories for the web app."""
    directories = ['uploads', 'output', 'templates', 'static']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def save_upload_file(upload_file: UploadFile, folder: str) -> str:
    """Save uploaded file and return the file path."""
    filename = f"{uuid.uuid4()}_{upload_file.filename}"
    file_path = os.path.join(folder, filename)
    
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    
    return file_path

@app.get("/", response_class=JSONResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    resolution: str = Form("480p"),
    positive_prompt: str = Form("A beautiful scene with gentle movement"),
    negative_prompt: str = Form("static, blurred, low quality, distorted, ugly")
):
    """Generate video from uploaded image and prompts."""
    try:
        # Validate image file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        if not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file type. Allowed: PNG, JPG, JPEG, BMP, TIFF, WEBP"
            )
        
        # Check file size
        if image.size and image.size > MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="Image file too large. Max 50MB")
        
        # Handle optional video file
        video_path = None
        if video and video.filename:
            if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid video file type. Allowed: MP4, AVI, MOV, MKV"
                )
            
            if video.size and video.size > MAX_CONTENT_LENGTH:
                raise HTTPException(status_code=400, detail="Video file too large. Max 50MB")
            
            # Save uploaded video file
            video_path = save_upload_file(video, UPLOAD_FOLDER)
            logger.info(f"Processing video: {video_path}")
        
        # Validate resolution
        if resolution not in RESOLUTION_PRESETS:
            raise HTTPException(status_code=400, detail="Invalid resolution. Choose 480p or 720p")
        
        # Get resolution settings and model ID
        width = RESOLUTION_PRESETS[resolution]['width']
        height = RESOLUTION_PRESETS[resolution]['height']
        model_id = RESOLUTION_PRESETS[resolution]['model_id']
        
        # Save uploaded image file
        upload_path = save_upload_file(image, UPLOAD_FOLDER)
        
        logger.info(f"Processing image: {upload_path}")
        logger.info(f"Resolution: {resolution} ({width}x{height})")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Positive prompt: {positive_prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Video guidance: {'Yes' if video_path else 'No'}")
        
        # Generate videos based on whether video guidance is provided
        generated_videos = []
        
        if video_path:
            # Use VACE pipeline for video-guided generation
            logger.info("Using VACE pipeline for video-guided generation")
            vace_output_filename = f"generated_vace_{uuid.uuid4()}.mp4"
            vace_output_path = os.path.join(OUTPUT_FOLDER, vace_output_filename)
            
            with WanVACEPipelineWrapper() as pipeline:
                result_path = pipeline.generate_video_with_guidance(
                    image_path=upload_path,
                    video_path=video_path,
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    output_path=vace_output_path
                )
            
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            generated_videos.append(GeneratedVideo(
                filename=vace_output_filename,
                file_size_mb=round(file_size, 1),
                model_type="VACE (Video-Guided)",
                description="Video-guided generation using VACE pipeline"
            ))
            
            logger.info(f"VACE video generated successfully: {result_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            
            # Clear GPU memory after VACE generation
            clear_gpu_memory()
            
        else:
            # Generate both I2V and VACE videos
            logger.info("Generating both I2V and VACE videos")
            
            # Generate I2V video
            i2v_output_filename = f"generated_i2v_{uuid.uuid4()}.mp4"
            i2v_output_path = os.path.join(OUTPUT_FOLDER, i2v_output_filename)
            
            with Wan21Pipeline(model_id=model_id) as pipeline:
                i2v_result_path = pipeline.generate_video(
                    image_path=upload_path,
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    output_path=i2v_output_path
                )
            
            i2v_file_size = os.path.getsize(i2v_result_path) / (1024 * 1024)  # MB
            generated_videos.append(GeneratedVideo(
                filename=i2v_output_filename,
                file_size_mb=round(i2v_file_size, 1),
                model_type="I2V (Image-to-Video)",
                description="Standard image-to-video generation"
            ))
            
            logger.info(f"I2V video generated successfully: {i2v_result_path}")
            logger.info(f"File size: {i2v_file_size:.1f} MB")
            
            # Clear GPU memory after I2V generation
            clear_gpu_memory()
            
            # Additional memory management for VACE
            import time
            logger.info("Waiting for memory cleanup before VACE generation...")
            time.sleep(2)  # Give time for memory cleanup
            clear_gpu_memory()  # Clear again after delay
            
            # Check memory before VACE generation
            from utils import check_gpu_memory
            gpu_info = check_gpu_memory()
            if gpu_info and gpu_info['free_memory'] < 3.0:  # Less than 3GB free
                logger.warning(f"Insufficient GPU memory for VACE: {gpu_info['free_memory']:.1f} GB available")
                logger.warning("Skipping VACE generation to prevent OOM error")
                vace_skip = True
            else:
                vace_skip = False

            # Generate VACE video (without video guidance)
            vace_output_filename = f"generated_vace_{uuid.uuid4()}.mp4"
            vace_output_path = os.path.join(OUTPUT_FOLDER, vace_output_filename)
            
            if not vace_skip:
                try:
                    with WanVACEPipelineWrapper() as pipeline:
                        vace_result_path = pipeline.generate_video_with_guidance(
                            image_path=upload_path,
                            video_path=None,  # No video guidance
                            prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            output_path=vace_output_path
                        )
                    
                    vace_file_size = os.path.getsize(vace_result_path) / (1024 * 1024)  # MB
                    generated_videos.append(GeneratedVideo(
                        filename=vace_output_filename,
                        file_size_mb=round(vace_file_size, 1),
                        model_type="VACE (Image-Only)",
                        description="Image-only generation using VACE pipeline"
                    ))
                    
                    logger.info(f"VACE video generated successfully: {vace_result_path}")
                    logger.info(f"File size: {vace_file_size:.1f} MB")
                    
                except Exception as e:
                    logger.error(f"Failed to generate VACE video: {e}")
                    # Continue with just I2V video if VACE fails
            else:
                logger.info("VACE generation skipped due to insufficient memory")
            
            # Clear GPU memory after VACE generation
            clear_gpu_memory()
        
        return VideoGenerationResponse(
            success=True,
            message=f'Generated {len(generated_videos)} video(s) successfully!',
            videos=generated_videos,
            resolution=resolution,
            width=width,
            height=height,
            model_id=model_id,
            video_guidance=video_path is not None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download generated video file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='video/mp4'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve video file for preview (without download)."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            media_type='video/mp4'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving video file {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Wan2.1 Web API is running"
    )

@app.on_event("startup")
async def startup_event():
    """Setup on application startup."""
    setup_app_directories()
    setup_directories()  # Setup Wan2.1 directories
    logger.info("Starting Wan2.1 Web Application...")
    logger.info("Access the web interface at: http://localhost:8080")
    logger.info("Access the API documentation at: http://localhost:8080/docs")

if __name__ == '__main__':
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    ) 