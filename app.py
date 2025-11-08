"""
Face Recognition System - Streamlit Demo Application
Uses Support Vector Machine (SVM) to recognize faces
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
from src.face_detector import FaceDetector
from src.feature_extractor import FeatureExtractor
from src.utils import load_model, save_model, create_directory
from src.guided_capture import GuidedSelfieCapture

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f0f0;
        margin: 0.5rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None


@st.cache_resource
def load_components():
    """Load model and components"""
    try:
        model_path = 'models/svm_face_recognition.pkl'
        
        if not os.path.exists(model_path):
            return None, None, None, None
        
        model = load_model(model_path)
        metadata = load_metadata(model_path)
        detector = FaceDetector()
        extractor = FeatureExtractor(
            feature_type=metadata.get('feature_type', 'hog'),
            image_size=metadata.get('image_size', (128, 128))
        )
        
        return model, detector, extractor, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


def recognize_face(image, model, detector, extractor, metadata):
    """
    Recognize face in image
    
    Args:
        image: Input image (BGR)
        model: Trained SVM model
        detector: FaceDetector instance
        extractor: FeatureExtractor instance
        metadata: Model metadata
    
    Returns:
        Dictionary with results
    """
    # Detect faces
    faces = detector.detect_faces(image)
    
    if len(faces) == 0:
        return {
            'success': False,
            'message': 'No face detected in image',
            'faces': []
        }
    
    results = []
    
    for face_rect in faces:
        try:
            # Extract face ROI
            face_roi = detector.extract_face_roi(image, face_rect)
            
            # Extract features
            features = extractor.extract_features(face_roi)
            features = features.reshape(1, -1)
            
            # Predict
            label_idx = model.predict(features)[0]
            confidence = np.max(model.predict_proba(features))
            
            # Get label name
            label_encoder = metadata['label_encoder']
            person_name = label_encoder.inverse_transform([label_idx])[0]
            
            results.append({
                'face_rect': face_rect,
                'person_name': person_name,
                'confidence': confidence,
                'face_roi': face_roi
            })
        except Exception as e:
            st.warning(f"Error processing face: {str(e)}")
            continue
    
    return {
        'success': len(results) > 0,
        'message': f'Detected {len(results)} face(s)',
        'faces': results
    }


def draw_results(image, results):
    """
    Draw recognition results on image
    
    Args:
        image: Input image
        results: Recognition results
    
    Returns:
        Annotated image
    """
    image_copy = image.copy()
    
    for result in results['faces']:
        x, y, w, h = result['face_rect']
        person_name = result['person_name']
        confidence = result['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{person_name} ({confidence:.2f})"
        cv2.putText(
            image_copy, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
    
    return image_copy


# Main app
st.markdown("<div class='main-header'>👤 Face Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Using Support Vector Machine (SVM)</div>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Add Student", "Train Model", "Face Recognition"]
)

if page == "Train Model":
    st.title("🚀 Train Face Recognition Model")
    
    st.markdown("""
    Train a new face recognition model using the collected face images.
    The model will learn to recognize the faces of all students in the dataset.
    """)
    
    # Model parameters
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
        random_state = st.number_input("Random seed", 0, 100, 42)
    
    with col2:
        min_samples = st.number_input("Minimum samples per person", 1, 20, 1,
                                    help="People with fewer samples will be excluded. Set to 1 for testing with small datasets.")
    
    # Training options
    st.subheader("Training Options")
    save_model_file = st.checkbox("Save trained model", value=True)
    show_details = st.checkbox("Show detailed training information", value=True)
    
    # Start training button
    if st.button("🎯 Start Training", type="primary"):
        if not os.path.exists("data/raw") or not any(os.scandir("data/raw")):
            st.error("No training data found. Please add students first.")
        else:
            with st.spinner("Training in progress. This may take a few minutes..."):
                try:
                    # Create a placeholder for training output
                    output = st.empty()
                    
                    # Redirect stdout to capture training output
                    import sys
                    from io import StringIO
                    
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    # Import and run training
                    from src.train_face_recognition import train_face_recognition_model
                    
                    model_path = "models/face_recognition_model.pkl"
                    model, le, test_accuracy = train_face_recognition_model(
                        data_dir="data/raw",
                        model_save_path=model_path if save_model_file else None,
                        test_size=test_size,
                        random_state=random_state,
                        min_samples=min_samples
                    )
                    
                    # Get the captured output
                    training_output = sys.stdout.getvalue()
                    sys.stdout = old_stdout
                    
                    # Display training output
                    if show_details:
                        with st.expander("Training Details", expanded=True):
                            st.code(training_output)
                    
                    # Show success message
                    st.success(f"✅ Model trained successfully! Test accuracy: {test_accuracy:.2%}")
                    
                    # Show confusion matrix if available
                    cm_path = "reports/figures/confusion_matrix.png"
                    if os.path.exists(cm_path):
                        st.image(cm_path, caption="Confusion Matrix", 
                                use_column_width=True)
                    
                    # Update session state
                    st.session_state.model_trained = True
                    st.session_state.model_accuracy = test_accuracy
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging
    
    # Show model info if available
    model_path = "models/face_recognition_model.pkl"
    if os.path.exists(model_path):
        st.subheader("Existing Model")
        
        try:
            import joblib
            from datetime import datetime
            
            model_data = joblib.load(model_path)
            train_date = model_data.get('training_date', 'Unknown')
            accuracy = model_data.get('test_accuracy', 0)
            num_classes = model_data.get('num_classes', 0)
            
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px;">
                <h4>Current Model Information</h4>
                <p>📅 <strong>Training Date:</strong> {train_date}</p>
                <p>🎯 <strong>Test Accuracy:</strong> {accuracy:.2%}</p>
                <p>👥 <strong>Number of People:</strong> {num_classes}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Could not load model information: {e}")
    
    # Add some space at the bottom
    st.markdown("\n\n---\n")
    st.markdown("💡 **Tip:** For best results, provide at least 5-10 images per person with different expressions and lighting conditions.")
elif page == "Add Student":
    st.title("👤 Add New Student")
    
    st.markdown("""
    Add a new student to the face recognition system by capturing multiple face images.
    Follow the on-screen instructions to capture images from different angles.
    """)
    
    # Create student directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of existing students
    existing_students = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    # Student info form
    with st.form("student_form"):
        st.subheader("Student Information")
        
        # Student ID (required)
        student_id = st.text_input("Student ID*", help="Unique identifier for the student")
        
        # Student name (optional)
        student_name = st.text_input("Student Name", help="Full name of the student (optional)")
        
        # Number of images to capture
        num_images = st.slider("Number of images to capture", 5, 50, 20, 5,
                             help="More images will improve recognition accuracy")
        
        # Start capture button
        submitted = st.form_submit_button("Start Face Capture")
    
    # Process form submission
    if submitted:
        if not student_id:
            st.error("Please enter a student ID")
        elif student_id in existing_students:
            st.warning(f"Student ID '{student_id}' already exists. Please use a different ID.")
        else:
            # Create student directory
            student_dir = data_dir / student_id
            student_dir.mkdir(exist_ok=True)
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Create placeholders
            image_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Capture loop
            images_captured = 0
            last_capture_time = time.time()
            
            while images_captured < num_images:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Convert to RGB for face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = faces.detectMultiScale(gray, 1.3, 5)
                
                # Draw face rectangle and instructions
                for (x, y, w, h) in face_rects:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Check if face is centered and large enough
                    frame_center = (frame.shape[1]//2, frame.shape[0]//2)
                    face_center = (x + w//2, y + h//2)
                    
                    # Calculate distance from center
                    x_diff = abs(face_center[0] - frame_center[0])
                    y_diff = abs(face_center[1] - frame_center[1])
                    
                    # Check if face is centered
                    if x_diff < 50 and y_diff < 50 and w > 200:
                        # Face is centered, check if enough time has passed since last capture
                        current_time = time.time()
                        if current_time - last_capture_time > 1.0:  # 1 second between captures
                            # Save the face image
                            face_img = rgb_frame[y:y+h, x:x+w]
                            img_path = student_dir / f"{student_id}_{images_captured:03d}.jpg"
                            cv2.imwrite(str(img_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                            images_captured += 1
                            last_capture_time = current_time
                            progress_bar.progress(images_captured / num_images)
                
                # Display instructions
                if len(face_rects) == 0:
                    status_placeholder.warning("No face detected. Please position your face in the frame.")
                elif images_captured < num_images:
                    status_placeholder.info(f"Capturing images... {images_captured}/{num_images}")
                    
                    # Add guidance text
                    cv2.putText(frame, "Position your face in the center", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Keep your face straight and look at the camera", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                image_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Check if we've captured enough images
                if images_captured >= num_images:
                    break
                    
                # Add a small delay to reduce CPU usage
                time.sleep(0.05)
            
            # Release the webcam
            cap.release()
            
            if images_captured > 0:
                st.success(f"✅ Successfully captured {images_captured} images for {student_name or student_id}")
                st.balloons()
                
                # Show sample of captured images
                st.subheader("Sample of Captured Images")
                cols = st.columns(4)
                for i, img_path in enumerate(sorted(student_dir.glob("*.jpg"))[:4]):
                    if i < len(cols):
                        cols[i].image(str(img_path), use_column_width=True)
            else:
                st.error("No images were captured. Please try again.")

elif page == "Face Recognition":
    st.title("👤 Face Recognition")
    
    # Check if model exists
    model_path = "models/face_recognition_model.pkl"
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning("⚠️ No trained model found. Please train the model first.")
        if st.button("Go to Training Page"):
            st.session_state.page = "Train Model"
            st.experimental_rerun()
    
    st.markdown("Test the face recognition system using your webcam or upload an image.")
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])
    
    with tab1:
        st.subheader("Webcam Recognition")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.6, 
            step=0.05,
            help="Higher values make the recognition more strict"
        )
        
        # Start webcam
        if st.button("🎥 Start Webcam", type="primary"):
            st.session_state.webcam_active = True
        
        if st.button("⏹️ Stop Webcam"):
            if 'webcam_active' in st.session_state:
                st.session_state.webcam_active = False
        
        # Webcam feed
        if st.session_state.get('webcam_active', False):
            import av
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            from src.face_detector import FaceDetector
            from src.feature_extractor import FeatureExtractor
            import joblib
            
            # Load model
            try:
                model_data = joblib.load(model_path)
                model = model_data['model']
                le = model_data['label_encoder']
                scaler = model_data['feature_scaler']
                
                # Initialize detector and extractor
                detector = FaceDetector()
                extractor = FeatureExtractor()
                
                # WebRTC configuration
                RTC_CONFIGURATION = RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
                
                def process_frame(frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Mirror the frame
                    img = cv2.flip(img, 1)
                    
                    # Detect faces
                    faces = detector.detect_faces(img)
                    
                    for (x, y, w, h) in faces:
                        # Extract face ROI
                        face_roi = img[y:y+h, x:x+w]
                        
                        try:
                            # Extract features
                            features = extractor.extract_features(face_roi).reshape(1, -1)
                            
                            # Scale features
                            features_scaled = scaler.transform(features)
                            
                            # Predict
                            proba = model.predict_proba(features_scaled)[0]
                            max_prob = np.max(proba)
                            
                            if max_prob >= confidence_threshold:
                                pred_class = le.inverse_transform([np.argmax(proba)])[0]
                                label = f"{pred_class} ({max_prob:.2f})"
                                color = (0, 255, 0)  # Green
                            else:
                                label = f"Unknown ({max_prob:.2f})"
                                color = (0, 0, 255)  # Red
                                
                            # Draw rectangle and label
                            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(img, label, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                        except Exception as e:
                            print(f"Error processing face: {e}")
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # Start WebRTC stream
                webrtc_ctx = webrtc_streamer(
                    key="face-recognition",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_frame_callback=process_frame,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    with tab2:
        st.subheader("Image Upload")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.6, 
            step=0.05,
            key="upload_threshold",
            help="Higher values make the recognition more strict"
        )
        
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=False)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 3:  # BGR to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Create a copy for drawing
            output_img = img_array.copy()
            
            if st.button("🔍 Detect and Recognize Faces"):
                with st.spinner("Processing image..."):
                    try:
                        # Load model
                        model_data = joblib.load(model_path)
                        model = model_data['model']
                        le = model_data['label_encoder']
                        scaler = model_data['feature_scaler']
                        
                        # Initialize detector and extractor
                        detector = FaceDetector()
                        extractor = FeatureExtractor()
                        
                        # Detect faces
                        faces = detector.detect_faces(img_array)
                        
                        if not faces:
                            st.warning("No faces detected in the image.")
                        else:
                            results = []
                            
                            for i, (x, y, w, h) in enumerate(faces):
                                # Extract face ROI
                                face_roi = img_array[y:y+h, x:x+w]
                                
                                try:
                                    # Extract features
                                    features = extractor.extract_features(face_roi).reshape(1, -1)
                                    
                                    # Scale features
                                    features_scaled = scaler.transform(features)
                                    
                                    # Predict
                                    proba = model.predict_proba(features_scaled)[0]
                                    max_prob = np.max(proba)
                                    
                                    if max_prob >= confidence_threshold:
                                        pred_class = le.inverse_transform([np.argmax(proba)])[0]
                                        label = f"{pred_class} ({max_prob:.2f})"
                                        color = (0, 255, 0)  # Green
                                    else:
                                        pred_class = "Unknown"
                                        label = f"{pred_class} ({max_prob:.2f})"
                                        color = (0, 0, 255)  # Red
                                    
                                    # Add to results
                                    results.append({
                                        "face_id": i + 1,
                                        "identity": pred_class,
                                        "confidence": f"{max_prob:.2f}",
                                        "status": "Recognized" if pred_class != "Unknown" else "Unknown"
                                    })
                                    
                                    # Draw rectangle and label
                                    cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
                                    cv2.putText(output_img, label, (x, y-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                    
                                except Exception as e:
                                    st.error(f"Error processing face {i+1}: {e}")
                            
                            # Show results
                            st.subheader("Recognition Results")
                            
                            # Display the image with detections
                            st.image(output_img, caption="Detected Faces", use_column_width=True)
                            
                            # Show results in a table
                            if results:
                                st.dataframe(results)
                                
                                # Add download button for the annotated image
                                from io import BytesIO
                                
                                # Convert the image to bytes
                                buffered = BytesIO()
                                Image.fromarray(output_img).save(buffered, format="JPEG")
                                
                                st.download_button(
                                    label="📥 Download Annotated Image",
                                    data=buffered.getvalue(),
                                    file_name="recognized_faces.jpg",
                                    mime="image/jpeg"
                                )
                            
                    except Exception as e:
                        st.error(f"An error occurred during recognition: {e}")
                        st.exception(e)  # Show full traceback for debugging
            else:
                # Just show the uploaded image without processing
                st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add some space at the bottom
    st.markdown("\n\n---\n")
    st.markdown("💡 **Tip:** For best results, ensure good lighting and that faces are clearly visible.")

elif page == "Home":
    st.title("🏠 Welcome to Face Recognition System")
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin-top: 0;">Get Started</h3>
        <p>This system allows you to recognize faces using computer vision and machine learning.</p>
        <div style="display: flex; justify-content: space-between; margin-top: 20px;">
            <div style="text-align: center; padding: 10px; background: white; border-radius: 8px; width: 30%;">
                <h4>1. Add Students</h4>
                <p>Capture images of students' faces</p>
            </div>
            <div style="text-align: center; padding: 10px; background: white; border-radius: 8px; width: 30%;">
                <h4>2. Train Model</h4>
                <p>Train the face recognition model</p>
            </div>
            <div style="text-align: center; padding: 10px; background: white; border-radius: 8px; width: 30%;">
                <h4>3. Recognize</h4>
                <p>Test with webcam or images</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Start")
    st.markdown("""
    1. Go to **Add Student** to capture face images
    2. Train the model with the **Train Model** page
    3. Test recognition with the **Face Recognition** page
    """)
    
    st.markdown("### System Requirements")
    st.markdown("""
    - Python 3.8+
    - Webcam
    - Good lighting conditions
    - Modern web browser
    """)
    
    st.markdown("### Need Help?")
    st.markdown("""
    Check out the [Documentation](https://github.com/thathsarabandara/FusionX1.0_face_recongnition_app.git) 
    or open an issue on GitHub.
    """)

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is an educational face recognition system using Support Vector Machine (SVM).
    
    ### 📚 Key Concepts
    
    **Support Vector Machine (SVM):**
    - Finds optimal hyperplane to separate classes
    - Maximizes margin between classes
    - Effective for high-dimensional data
    
    **Face Recognition Pipeline:**
    1. **Face Detection** - Detect faces using Haar Cascade
    2. **Face Alignment** - Align faces to standard orientation
    3. **Feature Extraction** - Extract HOG features
    4. **Classification** - Use SVM to classify faces
    
    ### 🛠️ Technologies
    
    - **scikit-learn** - SVM implementation
    - **OpenCV** - Face detection and image processing
    - **scikit-image** - HOG feature extraction
    - **Streamlit** - Web application framework
    
    ### 📊 Model Information
    """)
    
    if 'metadata' in st.session_state and st.session_state.metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.metadata.get('accuracy', 0):.2%}")
        
        with col2:
            st.metric("Classes", len(st.session_state.metadata.get('classes', [])))
        
        st.write("**Recognized Persons:**")
        for person in st.session_state.metadata.get('classes', []):
            st.write(f"- {person}")
    
    st.markdown("""
    ### 📖 Learning Resources
    
    - [SVM Tutorial](https://scikit-learn.org/stable/modules/svm.html)
    - [Face Recognition Basics](https://en.wikipedia.org/wiki/Facial_recognition_system)
    - [HOG Features](https://scikit-image.org/docs/stable/api/skimage.feature.html#hog)
    
    ### 🚀 Next Steps
    
    1. Prepare your own training dataset
    2. Train the model with your data
    3. Deploy to Streamlit Cloud
    4. Share with others!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Face Recognition System | Educational Project | SVM-based Classification
</div>
""", unsafe_allow_html=True)
