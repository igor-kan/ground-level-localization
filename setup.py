#!/usr/bin/env python3
"""
Setup script for ground-level visual localization system.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ground-level-localization",
    version="1.0.0",
    author="Ground Localization Team",
    author_email="your-email@example.com",
    description="AI-powered geolocation from street-level images using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/ground-level-localization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "streamlit-folium>=0.13.0",
            "plotly>=5.15.0",
            "folium>=0.14.0",
        ],
        "gpu": [
            "torch[gpu]>=2.0.0",
            "torchvision[gpu]>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ground-localize=api.main:main",
            "ground-localize-web=web.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)