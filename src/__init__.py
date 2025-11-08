"""
Face Recognition System using SVM
Educational project for learning face recognition and SVM
"""

__version__ = "1.0.0"
__author__ = "Educational Team"

from .face_detector import FaceDetector
from .feature_extractor import FeatureExtractor
from .utils import load_model, save_model

__all__ = [
    'FaceDetector',
    'FeatureExtractor',
    'load_model',
    'save_model'
]
