import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime

from .feature_extractor import FeatureExtractor
from .face_detector import FaceDetector

def load_training_data(data_dir="data/raw", target_size=(128, 128), min_samples=1):
    """
    Load training data from the specified directory.
    
    Args:
        data_dir: Directory containing subdirectories of images for each person
        target_size: Target size for resizing images
        min_samples: Minimum number of samples required per class
        
    Returns:
        X: Array of feature vectors
        y: Array of corresponding labels
        class_names: List of unique class names
    """
    print(f"Loading training data from {os.path.abspath(data_dir)}")
    
    # Initialize lists to store features and labels
    X_list = []
    y_list = []
    
    # Initialize face detector and feature extractor
    detector = FaceDetector()
    extractor = FeatureExtractor(feature_type='hog', image_size=target_size)
    
    # Create debug directory if it doesn't exist
    os.makedirs("debug", exist_ok=True)
    
    # Count samples per class
    samples_per_class = {}
    
    # Get list of person directories
    try:
        person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(person_dirs)} person directories: {person_dirs}")
        
        if not person_dirs:
            print(f"No person directories found in {data_dir}")
            return np.array([]), np.array([]), []
            
    except Exception as e:
        print(f"Error reading person directories: {e}")
        return np.array([]), np.array([]), []
    
    # Process each person's directory
    for person in person_dirs:
        person_dir = os.path.join(data_dir, person)
        print(f"\nProcessing person: {person} in {person_dir}")
        
        # Initialize sample count for this person
        samples_per_class[person] = 0
        
        # Get list of image files
        try:
            image_files = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(image_files)} images for {person}")
            
            if not image_files:
                print(f"No images found for {person}")
                continue
                
        except Exception as e:
            print(f"Error reading images for {person}: {e}")
            continue
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            print(f"\nProcessing image: {img_path}")
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                print(f"Image shape: {img.shape}")
                
                # Save original image for debugging
                debug_img_path = os.path.join("debug", os.path.basename(img_path))
                cv2.imwrite(debug_img_path, img)
                print(f"Saved debug image to: {debug_img_path}")
                
                # Detect faces in the image
                faces = detector.detect_faces(img)
                
                if not faces:
                    print(f"No faces detected in {img_path}")
                    continue
                    
                print(f"Detected {len(faces)} face(s)")
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    try:
                        # Extract face region and convert to RGB for feature extraction
                        face_bgr = img[y:y+h, x:x+w]
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Save the detected face for debugging
                        face_debug_path = os.path.join("debug", f"face_{os.path.basename(img_path)}")
                        cv2.imwrite(face_debug_path, face_bgr)
                        print(f"Saved detected face to: {face_debug_path}")
                        
                        # Extract features from the face (in RGB format)
                        try:
                            # Extract features
                            features = extractor.extract_features(face_rgb)
                            
                            # Ensure features is a numpy array and 1D
                            features = np.asarray(features, dtype=np.float32).flatten()
                            
                            # Store the features and label
                            X_list.append(features)
                            y_list.append(person)
                            samples_per_class[person] += 1
                            print(f"Successfully processed {img_path}")
                                
                        except Exception as e:
                            print(f"Error extracting features from {img_path}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Print detailed sample counts
    print("\nSample counts per person:")
    for cls, count in samples_per_class.items():
        print(f"- {cls}: {count} samples")
    
    # Remove classes with insufficient samples
    valid_classes = [cls for cls, count in samples_per_class.items() if count >= min_samples]
    
    if not valid_classes:
        print(f"\nERROR: No valid classes found with at least {min_samples} samples each!")
        print("This could be due to:")
        print("1. No faces detected in any images")
        print("2. Face detection parameters are too strict")
        print("3. Image quality issues (blurry, too dark, etc.)")
        print("\nCheck the debug/ directory for processed images and faces")
        raise ValueError(f"No valid classes found with at least {min_samples} samples each")
    
    print(f"\nFound {len(valid_classes)} valid classes with sufficient samples")
    
    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    # Create a mask for valid classes
    valid_mask = np.isin(y, valid_classes)
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    
    print(f"\nFinal dataset size: {len(X_filtered)} samples from {len(valid_classes)} people")
    print("Samples per person:")
    for person, count in sorted(samples_per_class.items()):
        if count > 0:
            print(f"- {person}: {count}")
    
    return X_filtered, y_filtered, valid_classes

def train_face_recognition_model(data_dir="data/raw", model_save_path="models/face_recognition_model.pkl", 
                               test_size=0.2, random_state=42, min_samples=1):
    """
    Train the face recognition model.
    
    Args:
        data_dir: Directory containing training data
        model_save_path: Path to save the trained model
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        min_samples: Minimum number of samples required per class
        
    Returns:
        model: Trained model
        le: Label encoder
        test_accuracy: Accuracy on test set
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load and preprocess data
    print("Loading training data...")
    X, y, classes = load_training_data(data_dir, min_samples=min_samples)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = SVC(kernel='linear', probability=True, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training set
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs('reports/figures', exist_ok=True)
    cm_path = 'reports/figures/confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    
    # Prepare the model data
    model_data = {
        'model': model,
        'label_encoder': le,
        'feature_scaler': scaler,
        'classes': le.classes_.tolist(),
        'feature_extractor': 'hog',  
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_accuracy': test_accuracy,
        'num_classes': len(le.classes_),
        'min_samples': min_samples,
        'input_shape': X_train.shape[1:],
    }
    
    joblib.dump(model_data, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    return model, le, test_accuracy

def predict_face(image, model_path="models/face_recognition_model.pkl", confidence_threshold=0.5):
    """
    Predict the identity of a face in the given image.
    
    Args:
        image: Input image (numpy array)
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence threshold for prediction
        
    Returns:
        prediction: Predicted class label
        confidence: Prediction confidence
        face_roi: Detected face region (for visualization)
    """
    # Load model and components
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model_data = joblib.load(model_path)
    model = model_data['model']
    le = model_data['label_encoder']
    scaler = model_data['feature_scaler']
    
    # Initialize detector and extractor
    detector = FaceDetector()
    extractor = FeatureExtractor()
    
    # Detect faces
    faces = detector.detect_faces(image)
    if not faces:
        return None, 0.0, None
    
    # Process the first face found
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]
    
    # Extract features
    features = extractor.extract_features(face_roi).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    proba = model.predict_proba(features_scaled)[0]
    max_prob = np.max(proba)
    
    if max_prob < confidence_threshold:
        return "Unknown", max_prob, face_roi
    
    pred_class = le.inverse_transform([np.argmax(proba)])[0]
    
    return pred_class, max_prob, face_roi

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a face recognition model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing training data')
    parser.add_argument('--model_path', type=str, default='models/face_recognition_model.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("Starting face recognition model training...")
    print(f"Data directory: {args.data_dir}")
    print(f"Model will be saved to: {args.model_path}")
    
    try:
        model, le, test_accuracy = train_face_recognition_model(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            test_size=args.test_size,
            random_state=args.random_state
        )
        print(f"\nTraining completed successfully! Test accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
