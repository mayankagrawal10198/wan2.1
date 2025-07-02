#!/usr/bin/env python3
"""
Flask Web Application for Wan2.1 I2V Model
Provides a web interface for video generation
"""

import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path

from wan21_pipeline import Wan21Pipeline
from utils import setup_directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF, WEBP'}), 400
        
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
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        logger.info(f"Processing image: {upload_path}")
        logger.info(f"Resolution: {resolution} ({width}x{height})")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Positive prompt: {positive_prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        
        # Generate output filename
        output_filename = f"generated_{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Initialize pipeline with specific model ID and generate video
        with Wan21Pipeline(model_id=model_id) as pipeline:
            result_path = pipeline.generate_video(
                image_path=upload_path,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                output_path=output_path
            )
        
        # Get file size
        file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
        
        logger.info(f"Video generated successfully: {result_path}")
        logger.info(f"File size: {file_size:.1f} MB")
        
        return jsonify({
            'success': True,
            'message': 'Video generated successfully!',
            'output_file': output_filename,
            'file_size_mb': round(file_size, 1),
            'resolution': resolution,
            'width': width,
            'height': height,
            'model_id': model_id
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