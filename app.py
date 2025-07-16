#!/usr/bin/env python3
"""
Flask Web Application for Wan2.1 I2V Model
Provides a web interface for video generation
"""

import os
import json
import logging
import torch
import gc
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path

from wan21_pipeline import Wan21Pipeline, WanVACEPipelineWrapper
from utils import setup_directories, clear_gpu_memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size for videos
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

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

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    """Check if video file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def setup_app_directories():
    """Setup necessary directories for the web app."""
    directories = ['uploads', 'output', 'templates', 'static']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video from uploaded image and prompts."""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid image file type. Allowed: PNG, JPG, JPEG, BMP, TIFF, WEBP'}), 400
        
        # Check for optional video file
        video_file = request.files.get('video')
        video_path = None
        
        if video_file and video_file.filename != '':
            if not allowed_video_file(video_file.filename):
                return jsonify({'error': 'Invalid video file type. Allowed: MP4, AVI, MOV, MKV'}), 400
            
            # Save uploaded video file
            video_filename = secure_filename(video_file.filename)
            video_unique_filename = f"{uuid.uuid4()}_{video_filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_unique_filename)
            video_file.save(video_path)
            logger.info(f"Processing video: {video_path}")
        
        # Get form data
        resolution = request.form.get('resolution', '480p')
        positive_prompt = request.form.get('positive_prompt', 'A beautiful scene with gentle movement')
        negative_prompt = request.form.get('negative_prompt', 'static, blurred, low quality, distorted, ugly')
        
        # Validate resolution
        if resolution not in RESOLUTION_PRESETS:
            return jsonify({'error': 'Invalid resolution. Choose 480p or 720p'}), 400
        
        # Get resolution settings and model ID
        width = RESOLUTION_PRESETS[resolution]['width']
        height = RESOLUTION_PRESETS[resolution]['height']
        model_id = RESOLUTION_PRESETS[resolution]['model_id']
        
        # Save uploaded image file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        logger.info(f"Processing image: {upload_path}")
        logger.info(f"Resolution: {resolution} ({width}x{height})")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Positive prompt: {positive_prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Video guidance: {'Yes' if video_path else 'No'}")
        
        # Generate output filename
        output_filename = f"generated_{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Generate videos based on whether video guidance is provided
        generated_videos = []
        
        if video_path:
            # Use VACE pipeline for video-guided generation
            logger.info("Using VACE pipeline for video-guided generation")
            vace_output_filename = f"generated_vace_{uuid.uuid4()}.mp4"
            vace_output_path = os.path.join(app.config['OUTPUT_FOLDER'], vace_output_filename)
            
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
            generated_videos.append({
                'filename': vace_output_filename,
                'file_size_mb': round(file_size, 1),
                'model_type': "VACE (Video-Guided)",
                'description': "Video-guided generation using VACE pipeline"
            })
            
            logger.info(f"VACE video generated successfully: {result_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            
            # Clear GPU memory after VACE generation
            clear_gpu_memory()
            
        else:
            # Generate both I2V and VACE videos
            logger.info("Generating both I2V and VACE videos")
            
            # Generate I2V video
            i2v_output_filename = f"generated_i2v_{uuid.uuid4()}.mp4"
            i2v_output_path = os.path.join(app.config['OUTPUT_FOLDER'], i2v_output_filename)
            
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
            generated_videos.append({
                'filename': i2v_output_filename,
                'file_size_mb': round(i2v_file_size, 1),
                'model_type': "I2V (Image-to-Video)",
                'description': "Standard image-to-video generation"
            })
            
            logger.info(f"I2V video generated successfully: {i2v_result_path}")
            logger.info(f"File size: {i2v_file_size:.1f} MB")
            
            # Clear GPU memory after I2V generation
            clear_gpu_memory()

            # Generate VACE video (without video guidance)
            vace_output_filename = f"generated_vace_{uuid.uuid4()}.mp4"
            vace_output_path = os.path.join(app.config['OUTPUT_FOLDER'], vace_output_filename)
            
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
                generated_videos.append({
                    'filename': vace_output_filename,
                    'file_size_mb': round(vace_file_size, 1),
                    'model_type': "VACE (Image-Only)",
                    'description': "Image-only generation using VACE pipeline"
                })
                
                logger.info(f"VACE video generated successfully: {vace_result_path}")
                logger.info(f"File size: {vace_file_size:.1f} MB")
                
            except Exception as e:
                logger.error(f"Failed to generate VACE video: {e}")
                # Continue with just I2V video if VACE fails
            
            # Clear GPU memory after VACE generation
            clear_gpu_memory()
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(generated_videos)} video(s) successfully!',
            'videos': generated_videos,
            'resolution': resolution,
            'width': width,
            'height': height,
            'model_id': model_id,
            'video_guidance': video_path is not None
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': f'Error generating video: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_video(filename):
    """Download generated video file."""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video file for preview (without download)."""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')
    except Exception as e:
        logger.error(f"Error serving video file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Wan2.1 Web API is running'})

if __name__ == '__main__':
    # Setup directories
    setup_app_directories()
    setup_directories()  # Setup Wan2.1 directories
    
    logger.info("Starting Wan2.1 Web Application...")
    logger.info("Access the web interface at: http://localhost:8080")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False) 