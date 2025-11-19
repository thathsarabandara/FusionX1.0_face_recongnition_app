"""
LBPH (Local Binary Pattern Histogram) Face Recognition
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_dataset(data_path, target_size=(100, 100)):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    for person_name in os.listdir(data_path):
        person_dir = os.path.join(data_path, person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        faces.append(img)
                        labels.append(current_label)
            current_label += 1
    
    return faces, np.array(labels), label_dict

def train_lbph(faces, labels, radius=1, neighbors=8, grid_x=8, grid_y=8):
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y
    )
    model.train(faces, labels)
    return model

def evaluate_lbph(model, test_faces, test_labels):
    correct = 0
    y_true = []
    y_pred = []
    
    for i, face in enumerate(test_faces):
        label, confidence = model.predict(face)
        y_true.append(test_labels[i])
        y_pred.append(label)
        
        if label == test_labels[i]:
            correct += 1
    
    accuracy = correct / len(test_faces)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy

if __name__ == "__main__":
    # Update this path to your dataset
    data_path = "path_to_your_dataset"
    
    # Load dataset
    print("Loading dataset...")
    faces, labels, label_dict = load_dataset(data_path)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train LBPH
    print("Training LBPH model...")
    model = train_lbph(X_train, y_train)
    
    # Save model
    model.save("lbph_face_recognizer.yml")
    print("Model saved as 'lbph_face_recognizer.yml'")
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy = evaluate_lbph(model, X_test, y_test)
