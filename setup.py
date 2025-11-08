"""
Setup script for Face Recognition System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-recognition-svm",
    version="1.0.0",
    author="Educational Team",
    description="Face Recognition System using Support Vector Machine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "opencv-python-headless>=4.8.0",
        "matplotlib>=3.8.0",
        "streamlit>=1.28.0",
        "Pillow>=10.0.0",
        "joblib>=1.3.0",
        "dlib>=19.24.0",
        "scikit-image>=0.22.0",
    ],
)
