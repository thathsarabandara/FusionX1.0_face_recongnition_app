import cv2
import numpy as np
import os
import time
from typing import Tuple, Optional, List

class GuidedSelfieCapture:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detected = False
        self.good_pose = False
        self.capture_count = 0
        self.required_captures = 5  # Number of images to capture per person
        self.last_capture_time = 0
        
    def check_face_pose(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if face is in good position for capture"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.face_detected = False
            self.good_pose = False
            return False, frame
        
        self.face_detected = True
        x, y, w, h = faces[0]
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate face position and size
        height, width = frame.shape[:2]
        face_center_x = x + w//2
        face_center_y = y + h//2
        
        # Draw center point
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 255), -1)
        
        # Draw target center
        target_center = (width//2, height//2)
        cv2.circle(frame, target_center, 3, (0, 0, 255), -1)
        
        # Check if face is centered and properly sized
        x_margin = width * 0.2
        y_margin = height * 0.2
        min_face_size = min(width, height) * 0.15
        max_face_size = min(width, height) * 0.5
        
        is_centered = (abs(face_center_x - target_center[0]) < x_margin and 
                      abs(face_center_y - target_center[1]) < y_margin)
        good_size = (min_face_size < w < max_face_size and 
                    min_face_size < h < max_face_size)
        
        # Check face angle (simple check using face dimensions)
        aspect_ratio = w / h
        good_angle = 0.7 < aspect_ratio < 1.3  # Allow some angle variation
        
        # Check if face is looking straight
        looking_straight = True
        if w < h * 0.8 or w > h * 1.2:  # Face is tilted
            looking_straight = False
        
        self.good_pose = is_centered and good_size and good_angle and looking_straight
        
        # Draw guidance
        if not is_centered:
            # Draw arrow to center
            cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                          target_center, (0, 0, 255), 2, tipLength=0.5)
            cv2.putText(frame, "Center your face", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not good_size:
            if w < min_face_size or h < min_face_size:
                cv2.putText(frame, "Move closer", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Move back a bit", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not good_angle or not looking_straight:
            cv2.putText(frame, "Face the camera directly", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Good! Stay still...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show countdown before capture
            current_time = time.time()
            if self.last_capture_time == 0:
                self.last_capture_time = current_time
            
            elapsed = current_time - self.last_capture_time
            if elapsed < 1.0:  # 1 second delay before capture
                countdown = 1 - int(elapsed * 10) / 10
                cv2.putText(frame, f"Capturing in {countdown:.1f}s...", 
                          (width//2 - 100, height - 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return self.good_pose, frame
    
    def capture_selfie(self, frame: np.ndarray, user_id: str) -> Tuple[bool, str]:
        """Capture and save selfie if conditions are met"""
        if not self.good_pose:
            return False, ""
            
        current_time = time.time()
        if current_time - self.last_capture_time < 1.0:  # 1 second between captures
            return False, ""
            
        # Create user directory if it doesn't exist
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(current_time * 1000)
        filename = os.path.join(user_dir, f"{user_id}_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(filename, frame)
        
        self.capture_count += 1
        self.last_capture_time = current_time
        remaining = max(0, self.required_captures - self.capture_count)
        
        if remaining > 0:
            return False, f"Captured {self.capture_count}/{self.required_captures}. {remaining} more needed."
        else:
            return True, f"Successfully captured {self.required_captures} images!"
    
    def reset(self):
        """Reset capture state"""
        self.face_detected = False
        self.good_pose = False
        self.capture_count = 0
        self.last_capture_time = 0
