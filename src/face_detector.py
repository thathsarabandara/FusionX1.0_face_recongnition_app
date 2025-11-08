"""
Face Detection Module
Handles face detection using Haar Cascade and dlib
"""

import cv2
import numpy as np
import os


class FaceDetector:
    """
    Face detection using Haar Cascade classifier
    """
    
    def __init__(self, cascade_path=None):
        """
        Initialize face detector
        
        Args:
            cascade_path: Path to Haar Cascade XML file
        """
        if cascade_path is None:
            # Use default OpenCV cascade
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            print(f"Using default cascade path: {cascade_path}")
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
            
        print(f"Loading cascade classifier from: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
            
        print("Cascade classifier loaded successfully")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            scale_factor: Scale factor for image pyramid
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
        
        Returns:
            List of face rectangles (x, y, w, h)
        """
        # Debug: Save input image
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already grayscale
            
        # Debug: Save grayscale image
        debug_gray_path = os.path.join(debug_dir, "detect_faces_gray.jpg")
        cv2.imwrite(debug_gray_path, gray)
        
        print(f"\nDetecting faces with params: scale_factor={scale_factor}, min_neighbors={min_neighbors}, min_size={min_size}")
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Convert to list of tuples if faces were found, otherwise return empty list
        if len(faces) > 0:
            # Convert from numpy array to list of tuples
            faces = [tuple(face) for face in faces]
            print(f"Found {len(faces)} faces")
            
            # Debug: Draw rectangles on the original image
            if len(image.shape) == 3:  # Color image
                debug_img = image.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                debug_path = os.path.join(debug_dir, "detected_faces.jpg")
                cv2.imwrite(debug_path, debug_img)
                print(f"Saved debug image with detected faces to: {debug_path}")
            return faces
        else:
            print("No faces detected")
            return []
    
    def extract_face_roi(self, image, face_rect, padding=10):
        """
        Extract face region of interest from image
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, w, h)
            padding: Padding around face
        
        Returns:
            Face ROI image
        """
        x, y, w, h = face_rect
        x = max(0, x - padding)
        y = max(0, y - padding)
        
        face_roi = image[y:y+h+padding*2, x:x+w+padding*2]
        return face_roi
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces
        
        Args:
            image: Input image
            faces: List of face rectangles
            color: Rectangle color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with drawn rectangles
        """
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, thickness)
        return image_copy
