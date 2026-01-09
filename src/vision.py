"""
Face recognition and computer vision module with webcam interface
"""

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import threading
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

class FaceScanner:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        
        # ROI (Region of Interest) settings
        self.roi_size = 300
        self.roi_color_red = (0, 0, 255)  # Red in BGR
        self.roi_color_green = (0, 255, 0)  # Green in BGR
        self.current_roi_color = self.roi_color_red
        
        # Webcam settings
        self.cap = None
        self.running = False
        self.scanned_face = None
        
    def calculate_roi_position(self, frame_width, frame_height):
        """Calculate the center position for the ROI box"""
        x = (frame_width - self.roi_size) // 2
        y = (frame_height - self.roi_size) // 2
        return x, y
    
    def is_face_in_roi(self, face_bbox, roi_x, roi_y):
        """Check if detected face is within the ROI box"""
        # face_bbox format: [x, y, width, height]
        face_center_x = face_bbox[0] + face_bbox[2] // 2
        face_center_y = face_bbox[1] + face_bbox[3] // 2
        
        # Check if face center is within ROI boundaries
        return (roi_x <= face_center_x <= roi_x + self.roi_size and
                roi_y <= face_center_y <= roi_y + self.roi_size)
    
    def create_overlay_mask(self, frame_shape, roi_x, roi_y):
        """Create a mask to darken the background outside the ROI"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        # Create white rectangle for ROI area
        mask[roi_y:roi_y + self.roi_size, roi_x:roi_x + self.roi_size] = 255
        
        # Invert mask to darken outside ROI
        mask_inv = cv2.bitwise_not(mask)
        
        return mask_inv
    
    def apply_dark_overlay(self, frame, roi_x, roi_y):
        """Apply dark overlay to areas outside the ROI"""
        # Create overlay with darkened background
        overlay = frame.copy()
        overlay = cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0.7, 0)
        
        # Create mask for ROI
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[roi_y:roi_y + self.roi_size, roi_x:roi_x + self.roi_size] = 255
        
        # Combine original frame (ROI) with darkened background
        result = np.where(mask[..., np.newaxis] == 255, frame, overlay)
        
        return result
    
    def extract_face_embedding(self, image, face_detection_result):
        """Extract face embedding for recognition"""
        # Placeholder for face embedding extraction
        # In a real implementation, you would use a model like FaceNet or ArcFace
        return np.random.rand(128)  # Dummy embedding
    
    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        """Compare two face embeddings"""
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance < threshold
    
    def register_face(self, user_id, embedding):
        """Register a new face in the system"""
        self.known_faces[user_id] = embedding
        return True
    
    def recognize_face(self, embedding):
        """Recognize a face from known faces"""
        best_match = None
        best_distance = float('inf')
        
        for user_id, known_embedding in self.known_faces.items():
            distance = np.linalg.norm(embedding - known_embedding)
            if distance < best_distance and distance < 0.6:
                best_distance = distance
                best_match = user_id
        
        return best_match
    
    def start_webcam_scanner(self):
        """Start the webcam scanner interface"""
        self.running = True
        self.scanned_face = None
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("Webcam scanner started. Press 'S' to scan face, 'Q' to quit.")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate ROI position
            roi_x, roi_y = self.calculate_roi_position(frame_width, frame_height)
            
# Apply dark overlay outside ROI
            frame = self.apply_dark_overlay(frame, roi_x, roi_y)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            self.current_roi_color = self.roi_color_red
            face_detected_in_roi = False

            for (x, y, w, h) in faces:
                face_bbox = [x, y, w, h]

                if self.is_face_in_roi(face_bbox, roi_x, roi_y):
                    face_detected_in_roi = True
                    self.current_roi_color = self.roi_color_green
            
            # Draw ROI box
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x + self.roi_size, roi_y + self.roi_size), 
                         self.current_roi_color, 3)
            
            # Add instructions text
            cv2.putText(frame, "Position face in box", (roi_x, roi_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'S' to scan, 'Q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show status
            if face_detected_in_roi:
                cv2.putText(frame, "Face Detected!", (roi_x, roi_y + self.roi_size + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Face Scanner', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                self.running = False
                break
            elif key == ord('s') or key == ord('S'):
                if face_detected_in_roi:
                    # Crop the face from ROI
                    cropped_face = frame[roi_y:roi_y + self.roi_size, 
                                       roi_x:roi_x + self.roi_size]
                    self.scanned_face = cropped_face.copy()
                    print("Face scanned successfully!")
                    
                    # Save the scanned face (optional)
                    cv2.imwrite('scanned_face.jpg', cropped_face)
                    
                    # Break the loop after successful scan
                    self.running = False
                    break
                else:
                    print("No face detected in ROI. Please position your face in the green box.")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        return self.scanned_face
    
    def stop_scanner(self):
        """Stop the webcam scanner"""
        self.running = False
        if self.cap:
            self.cap.release()
cv2.destroyAllWindows()

# Initialize face scanner (lazy loads MediaPipe)
face_scanner = None

def get_face_scanner():
    def get_or_create_face_scanner_instance():
        global face_scanner
        if face_scanner is None:
            face_scanner = FaceScanner()
        return face_scanner
    
    return get_or_create_face_scanner_instance()

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}

    def detect_face(self, image):
        """Detect faces in an image"""
        detection_start = time.time()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detection_time = (time.time() - detection_start) * 1000

        if len(faces) > 0:
            logger.info(f"   Face detected: {detection_time:.0f}ms")

        return faces

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """Detect faces in uploaded image"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array

        faces = face_system.detect_face(image_cv)

        if len(faces) > 0:
            return {
                "success": True,
                "face_count": len(faces),
                "message": f"Detected {len(faces)} face(s)"
            }
        else:
            return {
                "success": False,
                "face_count": 0,
                "message": "No faces detected"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register-face")
async def register_face(user_id: str, file: UploadFile = File(...)):
    """Register a user's face"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array

        faces = face_system.detect_face(image_cv)

        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        if len(faces) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with only one face.")

        embedding = face_scanner.extract_face_embedding(image_cv, faces)

        success = face_scanner.register_face(user_id, embedding)

        if success:
            return {
                "success": True,
                "message": f"Face registered successfully for user {user_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register face")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize a face from uploaded image"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array

        faces = face_system.detect_face(image_cv)

        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        if len(faces) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with only one face.")

        embedding = face_scanner.extract_face_embedding(image_cv, faces)

        user_id = face_scanner.recognize_face(embedding)

        if user_id:
            return {
                "success": True,
                "user_id": user_id,
                "message": f"Face recognized for user {user_id}"
            }
        else:
            return {
                "success": False,
                "user_id": None,
                "message": "Face not recognized"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan-face")
async def scan_face_from_webcam():
    """Scan face using webcam interface"""
    try:
        # Start webcam scanner in a separate thread
        def scan_thread():
            return face_scanner.start_webcam_scanner()
        
        # Run scanner (this will block until user presses 'S' or 'Q')
        scanned_face = face_scanner.start_webcam_scanner()
        
        if scanned_face is not None:
            # Convert scanned face to base64 for response
            _, buffer = cv2.imencode('.jpg', scanned_face)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "message": "Face scanned successfully",
                "face_image": img_base64
            }
        else:
            return {
                "success": False,
                "message": "No face was scanned"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

router = app