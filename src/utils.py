"""
Utility Functions
Helper functions for model management and data processing
"""

import os
import joblib
import numpy as np
import cv2
from pathlib import Path


def save_model(model, model_path, metadata=None):
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        model_path: Path to save model
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    if metadata:
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"Metadata saved to {metadata_path}")


def load_model(model_path):
    """
    Load trained model from disk
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_metadata(model_path):
    """
    Load model metadata
    
    Args:
        model_path: Path to model file
    
    Returns:
        Metadata dictionary
    """
    metadata_path = model_path.replace('.pkl', '_metadata.pkl')
    if os.path.exists(metadata_path):
        return joblib.load(metadata_path)
    return None


def create_directory(directory_path):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)
    print(f"Created directory: {directory_path}")
    return directory_path


def create_directories(base_path):
    """
    Create necessary project directories
    
    Args:
        base_path: Base project path
    """
    directories = [
        os.path.join(base_path, 'models'),
        os.path.join(base_path, 'data', 'raw'),
        os.path.join(base_path, 'data', 'processed'),
    ]
    
    for directory in directories:
        create_directory(directory)


def load_image(image_path):
    """
    Load image from file
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image array (BGR format)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def save_image(image, output_path):
    """
    Save image to file
    
    Args:
        image: Image array
        output_path: Path to save image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


def normalize_features(features):
    """
    Normalize feature vectors
    
    Args:
        features: Feature array
    
    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (features - mean) / std


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X: Feature array
        y: Label array
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*50)
    print("Model Evaluation Metrics")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("="*50 + "\n")
