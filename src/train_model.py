"""
Model Training Script
Train SVM model for face recognition
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from utils import save_model, evaluate_model, print_metrics, normalize_features


def load_training_data(data_dir, detector, extractor):
    """
    Load training data from directory structure
    Expected structure: data_dir/person_name/image1.jpg, image2.jpg, ...
    
    Args:
        data_dir: Root directory containing person folders
        detector: FaceDetector instance
        extractor: FeatureExtractor instance
    
    Returns:
        features array, labels array, label encoder
    """
    features_list = []
    labels_list = []
    label_encoder = LabelEncoder()
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(person_dirs)} person(s) in dataset")
    
    for person_name in person_dirs:
        person_path = os.path.join(data_dir, person_name)
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing {person_name}: {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"  ⚠ Failed to load {img_file}")
                    continue
                
                # Detect faces
                faces = detector.detect_faces(image)
                
                if len(faces) == 0:
                    print(f"  ⚠ No face detected in {img_file}")
                    continue
                
                # Use the largest face
                face_rect = max(faces, key=lambda r: r[2] * r[3])
                face_roi = detector.extract_face_roi(image, face_rect)
                
                # Extract features
                features = extractor.extract_features(face_roi)
                
                features_list.append(features)
                labels_list.append(person_name)
                print(f"  ✓ Processed {img_file}")
                
            except Exception as e:
                print(f"  ✗ Error processing {img_file}: {str(e)}")
                continue
    
    if len(features_list) == 0:
        raise ValueError("No training data found!")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Fit label encoder
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)
    
    print(f"\nTotal samples loaded: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X, y_encoded, label_encoder


def train_svm_model(X, y, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train SVM model
    
    Args:
        X: Training features
        y: Training labels
        kernel: SVM kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
    
    Returns:
        Trained SVM model
    """
    print("\nTraining SVM model...")
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    print(f"  Gamma: {gamma}")
    
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=42,
        verbose=1
    )
    
    model.fit(X, y)
    print("✓ Model training completed")
    
    return model


def main():
    """
    Main training pipeline
    """
    # Configuration
    DATA_DIR = 'data/raw'
    MODEL_DIR = 'models'
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize components
    detector = FaceDetector()
    extractor = FeatureExtractor(feature_type='hog', image_size=(128, 128))
    
    # Load training data
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    X, y, label_encoder = load_training_data(DATA_DIR, detector, extractor)
    
    # Normalize features
    print("\nNormalizing features...")
    X = normalize_features(X)
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    model = train_svm_model(X, y, kernel='rbf', C=100.0, gamma='scale')
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    
    # Save model
    print("="*60)
    print("SAVING MODEL")
    print("="*60)
    model_path = os.path.join(MODEL_DIR, 'svm_face_recognition.pkl')
    
    metadata = {
        'label_encoder': label_encoder,
        'feature_type': 'hog',
        'image_size': (128, 128),
        'accuracy': metrics['accuracy'],
        'classes': list(label_encoder.classes_)
    }
    
    save_model(model, model_path, metadata)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
