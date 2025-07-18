#!/usr/bin/env python3
"""
FastAPI Web Application for Wan2.1 I2V Model
Provides a web interface for video generation
"""

import os
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
from contextlib import asynccontextmanager

from wan21_pipeline import Wan21Pipeline, WanVACEPipelineWrapper
from utils import setup_directories, clear_gpu_memory, check_gpu_memory, force_free_unallocated_memory
from config import ENABLE_VACE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Pydantic models
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

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_app_directories()
    setup_directories()
    logger.info("Starting Wan2.1 Web Application...")
    logger.info("Access the web interface at: http://localhost:8080")
    logger.info("Access the API documentation at: http://localhost:8080/docs")
    yield

# FastAPI app instance
app = FastAPI(
    title="Wan2.1 Video Generation API",
    description="API for generating videos from images using Wan2.1 I2V and VACE models",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Utility functions
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def setup_app_directories():
    for directory in ['uploads', 'output', 'templates', 'static']:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def save_upload_file(upload_file: UploadFile, folder: str) -> str:
    filename = f"{uuid.uuid4()}_{upload_file.filename}"
    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_path

# Routes
@app.get("/", response_class=JSONResponse)
async def index(request: Request):
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
    try:
        if not image.filename or not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invalid or missing image file.")

        if image.size and image.size > MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="Image file too large. Max 50MB")

        video_path = None
        if video and video.filename:
            if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
                raise HTTPException(status_code=400, detail="Invalid video file type")
            if video.size and video.size > MAX_CONTENT_LENGTH:
                raise HTTPException(status_code=400, detail="Video file too large. Max 50MB")
            video_path = save_upload_file(video, UPLOAD_FOLDER)

        if resolution not in RESOLUTION_PRESETS:
            raise HTTPException(status_code=400, detail="Invalid resolution. Choose 480p or 720p")

        width = RESOLUTION_PRESETS[resolution]['width']
        height = RESOLUTION_PRESETS[resolution]['height']
        model_id = RESOLUTION_PRESETS[resolution]['model_id']
        upload_path = save_upload_file(image, UPLOAD_FOLDER)

        logger.info(f"Processing image: {upload_path}")
        logger.info(f"Video guidance: {'Yes' if video_path else 'No'}")

        generated_videos = []

        if video_path:

            
            logger.info("Using VACE pipeline (guided)")
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
            size = os.path.getsize(result_path) / (1024 * 1024)
            generated_videos.append(GeneratedVideo(
                filename=vace_output_filename,
                file_size_mb=round(size, 1),
                model_type="VACE (Video-Guided)",
                description="Video-guided generation using VACE pipeline"
            ))
            clear_gpu_memory()

        else:
            logger.info("Generating I2V and optionally VACE")
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
            i2v_size = os.path.getsize(i2v_result_path) / (1024 * 1024)
            generated_videos.append(GeneratedVideo(
                filename=i2v_output_filename,
                file_size_mb=round(i2v_size, 1),
                model_type="I2V (Image-to-Video)",
                description="Standard image-to-video generation"
            ))
            clear_gpu_memory()
            
            # Force free unallocated memory more aggressively
            force_free_unallocated_memory()
            
            # Small delay to ensure CUDA cache is fully cleared
            import time
            time.sleep(2)
            
            if ENABLE_VACE:
                try:
                    vace_output_filename = f"generated_vace_{uuid.uuid4()}.mp4"
                    vace_output_path = os.path.join(OUTPUT_FOLDER, vace_output_filename)
                    with WanVACEPipelineWrapper() as pipeline:
                        vace_result_path = pipeline.generate_video_with_guidance(
                            image_path=upload_path,
                            video_path=None,
                            prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            output_path=vace_output_path
                        )
                    vace_size = os.path.getsize(vace_result_path) / (1024 * 1024)
                    generated_videos.append(GeneratedVideo(
                        filename=vace_output_filename,
                        file_size_mb=round(vace_size, 1),
                        model_type="VACE (Image-Only)",
                        description="Image-only generation using VACE pipeline"
                    ))
                except Exception as e:
                    logger.error(f"VACE generation failed: {e}")

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
        logger.error(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='video/mp4')

@app.get("/video/{filename}")
async def serve_video(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type='video/mp4')

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Wan2.1 Web API is running"
    )

if __name__ == '__main__':
    # Set PyTorch CUDA allocation configuration to reduce memory fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
