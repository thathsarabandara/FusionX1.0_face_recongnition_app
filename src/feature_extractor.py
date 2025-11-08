"""
Feature Extraction Module
Extracts facial features using HOG and other techniques
"""

import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure


class FeatureExtractor:
    """
    Extract facial features from face images
    """
    
    def __init__(self, feature_type='hog', image_size=(128, 128)):
        """
        Initialize feature extractor
        
        Args:
            feature_type: Type of features ('hog', 'raw', 'combined')
            image_size: Size to resize faces to
        """
        self.feature_type = feature_type
        self.image_size = image_size
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image
        
        Args:
            face_image: Input face image
        
        Returns:
            Preprocessed face image
        """
        # Resize to standard size
        face = cv2.resize(face_image, self.image_size)
        
        # Convert to grayscale if needed
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization for better contrast
        face = cv2.equalizeHist(face)
        
        return face
    
    def extract_hog_features(self, face_image):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            face_image: Input face image (grayscale)
        
        Returns:
            HOG feature vector as a NumPy array
        """
        try:
            face = self.preprocess_face(face_image)
            
            # Ensure face is 2D (grayscale)
            if len(face.shape) > 2:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Extract HOG features
            features = hog(
                face,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False,
                block_norm='L2-Hys',
                feature_vector=True  # Ensure we get a 1D feature vector
            )
            
            # Ensure we return a numpy array
            return np.asarray(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in extract_hog_features: {e}")
            # Return an empty array with a standard size if feature extraction fails
            return np.zeros(1764, dtype=np.float32)  # Default size for 128x128 image with these HOG params
    
    def extract_raw_features(self, face_image):
        """
        Extract raw pixel features
        
        Args:
            face_image: Input face image
        
        Returns:
            Flattened pixel vector
        """
        face = self.preprocess_face(face_image)
        # Normalize pixel values
        features = face.astype(np.float32) / 255.0
        return features.flatten()
    
    def extract_combined_features(self, face_image):
        """
        Extract combined features (HOG + raw pixels)
        
        Args:
            face_image: Input face image
        
        Returns:
            Combined feature vector
        """
        hog_features = self.extract_hog_features(face_image)
        raw_features = self.extract_raw_features(face_image)
        
        # Normalize and combine
        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-6)
        raw_features = raw_features / (np.linalg.norm(raw_features) + 1e-6)
        
        combined = np.concatenate([hog_features, raw_features])
        return combined
    
    def extract_features(self, face_image):
        """
        Extract features based on configured type
        
        Args:
            face_image: Input face image
        
        Returns:
            Feature vector
        """
        if self.feature_type == 'hog':
            return self.extract_hog_features(face_image)
        elif self.feature_type == 'raw':
            return self.extract_raw_features(face_image)
        elif self.feature_type == 'combined':
            return self.extract_combined_features(face_image)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def extract_batch_features(self, face_images):
        """
        Extract features from multiple face images
        
        Args:
            face_images: List of face images
        
        Returns:
            Array of feature vectors
        """
        features_list = []
        for face_image in face_images:
            features = self.extract_features(face_image)
            features_list.append(features)
        
        return np.array(features_list)
