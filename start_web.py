#!/usr/bin/env python3
"""
Startup script for Wan2.1 Web Application
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['flask', 'werkzeug']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main startup function."""
    print("🚀 Starting Wan2.1 Web Application...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected.")
        print("   It's recommended to activate the virtual environment first:")
        print("   source wan21-env/bin/activate")
        print()
    
    # Start the Flask application
    try:
        from app import app
        print("✅ Web application loaded successfully!")
        print("🌐 Access the web interface at: http://localhost:8080")
        print("📱 The interface supports both 480p and 720p video generation")
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        app.run(host='0.0.0.0', port=8080, debug=False)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Web application stopped.")
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 