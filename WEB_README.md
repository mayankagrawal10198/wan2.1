# Wan2.1 Web Interface

A beautiful web interface for the Wan2.1 I2V (Image-to-Video) model that allows you to generate videos from images through a user-friendly web application.

## 🌟 Features

- **Modern Web Interface**: Clean, responsive design that works on desktop and mobile
- **Resolution Selection**: Choose between 480p (320×400) and 720p (576×720)
- **Drag & Drop Upload**: Easy image upload with drag & drop support
- **Real-time Feedback**: Loading states and progress indicators
- **Automatic Download**: Generated videos are automatically available for download
- **Multiple Format Support**: PNG, JPG, JPEG, BMP, TIFF, WEBP
- **🎬 Video Guidance**: Optional video upload for movement guidance using VACE pipeline

## 🎬 Video Guidance Feature

The web interface now supports **video-guided generation** using the Wan2.1 VACE (Video-Aware Conditional Editing) pipeline. This allows you to provide a reference video to guide the movement and motion of your generated video.

### How It Works

1. **Standard I2V**: Upload only an image → Uses Wan2.1 I2V pipeline
2. **Video-Guided**: Upload image + video → Uses Wan2.1 VACE pipeline

### Video Guidance Benefits

- **Motion Control**: The generated video follows the movement patterns from your reference video
- **Temporal Consistency**: Better frame-to-frame coherence
- **Creative Control**: More precise control over the final video's motion

### Supported Video Formats

- **Input Videos**: MP4, AVI, MOV, MKV (max 50MB)
- **Output**: MP4 format with 16 FPS

### Usage Steps

1. **Upload Image**: Select your input image (required)
2. **Upload Video** (Optional): Select a reference video for movement guidance
3. **Enter Prompts**: Describe what you want to see
4. **Select Resolution**: Choose 480p or 720p
5. **Generate**: The system automatically chooses the appropriate pipeline

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Flask and web dependencies
pip install flask werkzeug
```

### 2. Start the Web Application
```bash
# Method 1: Using the startup script (recommended)
python start_web.py

# Method 2: Direct Flask execution
python app.py
```

### 3. Access the Web Interface
Open your browser and go to: **http://localhost:8080**

## 📱 How to Use

### Step 1: Select Resolution
- **480p**: 320×400 pixels (uses Wan2.1-I2V-14B-480P-Diffusers model)
- **720p**: 576×720 pixels (uses Wan2.1-I2V-14B-720P-Diffusers model)

### Step 2: Upload Image
- Click the upload area or drag & drop an image file
- Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP
- Maximum file size: 16MB

### Step 3: Enter Prompts
- **Positive Prompt**: Describe what you want to see in the video
- **Negative Prompt**: Describe what you don't want to see

### Step 4: Generate Video
- Click "Generate Video" and wait for processing
- The video will be automatically available for download when complete

## 🔧 API Endpoints

### POST `/generate`
Generate a video from uploaded image and prompts.

**Request:**
- `image`: Image file (multipart/form-data) - **Required**
- `video`: Video file (multipart/form-data) - **Optional** for video guidance
- `resolution`: "480p" or "720p"
- `positive_prompt`: Text description of desired video
- `negative_prompt`: Text description of unwanted elements

**Response:**
```json
{
    "success": true,
    "message": "Video generated successfully!",
    "output_file": "generated_uuid.mp4",
    "file_size_mb": 15.2,
    "resolution": "720p",
    "width": 576,
    "height": 720,
    "model_id": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    "model_type": "VACE (Video-Guided)",
    "video_guidance": true
}
```

### GET `/download/<filename>`
Download a generated video file.

### GET `/health`
Health check endpoint.

## 📁 File Structure

```
wan2.1/
├── app.py                 # Flask web application
├── start_web.py          # Startup script
├── templates/
│   └── index.html        # Web interface HTML
├── uploads/              # Temporary uploaded images
├── output/               # Generated videos
└── WEB_README.md         # This file
```

## ⚙️ Configuration

### Resolution Presets
```python
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
```

### File Upload Settings
- Maximum file size: 50MB (increased for video files)
- Allowed image extensions: PNG, JPG, JPEG, BMP, TIFF, WEBP
- Allowed video extensions: MP4, AVI, MOV, MKV

## 🛠️ Development

### Running in Development Mode
```bash
# Set Flask to development mode
export FLASK_ENV=development
python app.py
```

### Customizing the Interface
Edit `templates/index.html` to modify the web interface:
- Colors and styling in the `<style>` section
- JavaScript functionality in the `<script>` section
- HTML structure in the body

### Adding New Features
1. Modify `app.py` to add new API endpoints
2. Update `templates/index.html` for UI changes
3. Test with different image formats and resolutions

## 🔒 Security Considerations

- File upload validation prevents malicious files
- Secure filename handling with UUID generation
- Maximum file size limits prevent abuse
- Input validation for all form fields

## 🐛 Troubleshooting

### Common Issues

1. **Port 8080 already in use**
   ```bash
   # Change port in app.py
   app.run(host='0.0.0.0', port=8081, debug=False)
   ```

2. **Flask not installed**
   ```bash
   pip install flask werkzeug
   ```

3. **Permission denied for uploads**
   ```bash
   # Ensure upload directories exist and are writable
   mkdir -p uploads output
   chmod 755 uploads output
   ```

4. **Large files not uploading**
   - Check file size limit (16MB)
   - Ensure proper file format

### Debug Mode
```bash
# Enable debug mode for detailed error messages
export FLASK_DEBUG=1
python app.py
```

## 📊 Performance

### Expected Generation Times
- **480p**: 2-3 minutes
- **720p**: 4-5 minutes

### Memory Usage
- Upload processing: ~50MB
- Video generation: Varies by resolution and GPU
- Temporary files are cleaned up automatically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This web interface is part of the Wan2.1 project and follows the same license terms.

---

**Enjoy creating amazing videos with Wan2.1! 🎬** 