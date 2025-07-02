"""
Setup script for Wan2.1 I2V Model
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="wan21-i2v",
    version="1.0.0",
    author="Wan2.1 I2V Team",
    description="Python package for running Wan2.1 Image-to-Video (I2V) model locally",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/wan21-i2v",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "wan21=run_wan21:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
    keywords="ai, video-generation, image-to-video, wan2.1, diffusers, transformers",
    project_urls={
        "Bug Reports": "https://github.com/your-username/wan21-i2v/issues",
        "Source": "https://github.com/your-username/wan21-i2v",
        "Documentation": "https://github.com/your-username/wan21-i2v#readme",
    },
) 