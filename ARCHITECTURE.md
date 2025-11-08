# System Architecture

## Overview

The Face Recognition System is built using a modular architecture with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web App                      │
│                      (app.py)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐  ┌──────────┐  ┌────────────┐
   │  Upload │  │ Webcam   │  │   About    │
   │  Image  │  │  Mode    │  │   Info     │
   └────┬────┘  └──────────┘  └────────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │   Face Detection Module         │
   │   (face_detector.py)            │
   │   - Haar Cascade Classifier     │
   │   - Face ROI Extraction         │
   └────┬────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │   Feature Extraction Module     │
   │   (feature_extractor.py)        │
   │   - HOG Features                │
   │   - Image Preprocessing         │
   │   - Normalization               │
   └────┬────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │   SVM Classifier                │
   │   (Trained Model)               │
   │   - Prediction                  │
   │   - Confidence Scores           │
   └────┬────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │   Results & Visualization       │
   │   - Bounding Boxes              │
   │   - Person Names                │
   │   - Confidence Scores           │
   └─────────────────────────────────┘
```

## Component Details

### 1. Face Detection Module (`src/face_detector.py`)

**Purpose**: Detect faces in images

**Key Classes**:
- `FaceDetector`: Main face detection class

**Key Methods**:
- `detect_faces()`: Detect all faces in image
- `extract_face_roi()`: Extract region of interest
- `draw_faces()`: Visualize detected faces

**Technology**: Haar Cascade Classifier

### 2. Feature Extraction Module (`src/feature_extractor.py`)

**Purpose**: Extract facial features for classification

**Key Classes**:
- `FeatureExtractor`: Feature extraction class

**Key Methods**:
- `extract_hog_features()`: Extract HOG features
- `extract_raw_features()`: Extract pixel features
- `extract_combined_features()`: Combine multiple features
- `preprocess_face()`: Preprocess face image

**Features**:
- HOG (Histogram of Oriented Gradients)
- Raw pixel features
- Combined features

### 3. Training Module (`src/train_model.py`)

**Purpose**: Train SVM classifier on labeled face data

**Process**:
1. Load training data from `data/raw/`
2. Detect faces in each image
3. Extract features
4. Normalize features
5. Train SVM model
6. Evaluate on test set
7. Save model and metadata

**Output**:
- `models/svm_face_recognition.pkl`: Trained model
- `models/metadata.pkl`: Model metadata

### 4. Utility Module (`src/utils.py`)

**Purpose**: Helper functions for common tasks

**Key Functions**:
- `save_model()`: Save trained model
- `load_model()`: Load trained model
- `evaluate_model()`: Evaluate model performance
- `normalize_features()`: Normalize feature vectors
- `split_data()`: Train-test split

### 5. Web Application (`app.py`)

**Purpose**: Interactive web interface using Streamlit

**Features**:
- Image upload and recognition
- Real-time webcam support (local)
- Model information display
- Educational content

**Modes**:
- Upload Image: Recognize faces in uploaded images
- Webcam: Real-time face recognition
- About: Project information

## Data Flow

### Training Pipeline

```
Raw Images
    │
    ▼
Face Detection
    │
    ▼
Face ROI Extraction
    │
    ▼
Feature Extraction (HOG)
    │
    ▼
Feature Normalization
    │
    ▼
Train-Test Split
    │
    ▼
SVM Training
    │
    ▼
Model Evaluation
    │
    ▼
Save Model & Metadata
```

### Inference Pipeline

```
Input Image
    │
    ▼
Face Detection
    │
    ▼
Face ROI Extraction
    │
    ▼
Feature Extraction (HOG)
    │
    ▼
Feature Normalization
    │
    ▼
SVM Prediction
    │
    ▼
Get Confidence Scores
    │
    ▼
Display Results
```

## File Structure

```
face_recognition_system/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # Project documentation
├── QUICK_START.md                  # Quick start guide
├── GETTING_STARTED.md              # Detailed setup guide
├── DEPLOYMENT.md                   # Deployment instructions
├── ARCHITECTURE.md                 # This file
│
├── src/                            # Source code
│   ├── __init__.py                 # Package initialization
│   ├── face_detector.py            # Face detection module
│   ├── feature_extractor.py        # Feature extraction module
│   ├── train_model.py              # Training script
│   └── utils.py                    # Utility functions
│
├── notebooks/                      # Jupyter notebooks
│   └── svm_face_recognition.ipynb  # Educational notebook
│
├── models/                         # Trained models
│   ├── svm_face_recognition.pkl    # Trained SVM model
│   └── metadata.pkl                # Model metadata
│
├── data/                           # Data directory
│   ├── raw/                        # Raw training images
│   │   ├── person1/
│   │   ├── person2/
│   │   └── ...
│   └── processed/                  # Processed features
│
└── .streamlit/                     # Streamlit configuration
    └── config.toml                 # App configuration
```

## Technology Stack

### Core Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| scikit-learn | SVM classifier | >=1.3.0 |
| OpenCV | Face detection | >=4.8.0 |
| NumPy | Numerical computing | >=1.26.0 |
| scikit-image | Image processing | >=0.22.0 |

### Web Framework

| Library | Purpose | Version |
|---------|---------|---------|
| Streamlit | Web application | >=1.28.0 |
| Pillow | Image handling | >=10.0.0 |
| Matplotlib | Visualization | >=3.8.0 |

### Development Tools

| Library | Purpose | Version |
|---------|---------|---------|
| Jupyter | Interactive notebooks | >=1.0.0 |
| joblib | Model serialization | >=1.3.0 |

## Performance Characteristics

### Face Detection
- **Speed**: ~50-100ms per image
- **Accuracy**: ~95% on frontal faces
- **Limitations**: Works best with frontal faces

### Feature Extraction
- **HOG Feature Dimension**: 3780 (for 128x128 images)
- **Extraction Time**: ~10-20ms per face

### SVM Classification
- **Prediction Time**: ~1-5ms per face
- **Memory**: ~10-50MB (depends on training data)
- **Accuracy**: 85-95% (depends on training data quality)

## Scalability Considerations

### For More Persons
- Add more subdirectories in `data/raw/`
- Retrain model with new data
- Model size increases linearly with number of classes

### For More Training Data
- Increase training time
- May improve accuracy
- Consider using GPU for faster training

### For Real-time Processing
- Optimize feature extraction
- Use model caching
- Consider using GPU acceleration

## Security Considerations

1. **Data Privacy**: Training data stored locally
2. **Model Security**: Save model with restricted permissions
3. **Input Validation**: Validate uploaded images
4. **Resource Limits**: Set upload size limits

## Future Enhancements

1. **Deep Learning**: Use CNN-based features
2. **GPU Support**: CUDA acceleration
3. **Real-time Webcam**: Improved streaming
4. **Multiple Faces**: Batch processing
5. **Face Verification**: 1-to-1 matching
6. **Emotion Detection**: Additional classification
7. **Age/Gender**: Multi-task learning
