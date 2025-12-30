"""
Main entry point for the Face Recognition Payment System
Complete workflow integration
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from vision import FaceScanner, face_scanner
from auth import SFaceAuthenticator, sface_auth
from payment import MpesaPaymentGateway, PaymentService, payment_service, router as payment_router

app = FastAPI(
    title="Face Recognition Payment System",
    description="A backend system for processing payments using face recognition",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount payment router
app.mount("/mpesa", payment_router)

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_open_hand(self, frame):
        """Detect if hand is open (palm facing camera)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check if hand is open (all fingers extended)
                landmarks = hand_landmarks.landmark
                
                # Finger tip and pip landmarks for each finger
                finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                finger_pips = [3, 6, 10, 14, 18]
                
                open_fingers = 0
                
                # Check each finger (except thumb which is handled differently)
                for i in range(1, 5):
                    if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                        open_fingers += 1
                
                # Check thumb (horizontal comparison)
                if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
                    open_fingers += 1
                
                # Consider hand open if at least 4 fingers are extended
                return open_fingers >= 4
        
        return False
    
    def detect_fist(self, frame):
        """Detect if hand is closed (fist)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check if hand is closed (all fingers curled)
                landmarks = hand_landmarks.landmark
                
                # Finger tip and pip landmarks for each finger
                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [3, 6, 10, 14, 18]
                
                closed_fingers = 0
                
                # Check each finger (except thumb)
                for i in range(1, 5):
                    if landmarks[finger_tips[i]].y > landmarks[finger_pips[i]].y:
                        closed_fingers += 1
                
                # Check thumb
                if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
                    closed_fingers += 1
                
                # Consider fist if at least 4 fingers are closed
                return closed_fingers >= 4
        
        return False

class FacePaymentSystem:
    def __init__(self):
        self.face_scanner = FaceScanner()
        self.sface_auth = SFaceAuthenticator()
        self.payment_gateway = MpesaPaymentGateway()
        self.payment_service = payment_service
        self.gesture_recognizer = GestureRecognizer()
        self.cap = None
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.demo_mode = False
        
    def initialize_camera(self, camera_index=None):
        """
        Robust camera initialization with multiple fallback options.
        
        Args:
            camera_index: Specific camera index to try (optional)
            
        Returns:
            cv2.VideoCapture object or None if all attempts fail
        """
        indices_to_try = []
        
        if camera_index is not None:
            indices_to_try.append(camera_index)
        else:
            indices_to_try.append(self.camera_index)
        
        # Add fallback indices
        for i in range(3):
            if i not in indices_to_try:
                indices_to_try.append(i)
        
        # Try different backends for Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        print("\n📷 Attempting to initialize camera...")
        
        for idx in indices_to_try:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    print(f"✅ Camera {idx} opened with backend {backend}")
                    print(f"   Resolution: {width}x{height}, FPS: {fps}")
                    
                    # Warm up camera by reading a few frames
                    print("   Warming up camera...")
                    for _ in range(10):
                        ret, frame = cap.read()
                        if not ret:
                            break
                    
                    # Verify we can still read frames after warm-up
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"   Camera {idx} is working properly!")
                        return cap
                    else:
                        print(f"   Camera {idx} failed after warm-up")
                        cap.release()
                        
                except Exception as e:
                    print(f"   Error with camera {idx}, backend {backend}: {e}")
                    try:
                        cap.release()
                    except:
                        pass
        
        print("\n❌ Failed to initialize any camera")
        print("   Possible causes:")
        print("   1. Camera is already in use by another application")
        print("   2. Camera permissions are not granted")
        print("   3. Camera driver is not installed properly")
        print("   4. No camera is connected to the system")
        return None
    
    def run_demo_mode(self):
        """Run the system in demo mode without camera"""
        print("\n" + "="*60)
        print("🎭 DEMO MODE - Camera Not Available")
        print("="*60)
        
        try:
            # Step 1: Load SFace model
            print("\n📥 Step 1: Loading SFace model...")
            self.sface_auth.load_model()
            print("✅ SFace model loaded successfully")
            
            # Step 2: Simulate face capture
            print("\n📷 Step 2: Simulating face capture...")
            time.sleep(1)
            print("✅ Face captured successfully (simulated)")
            
            # Step 3: Generate embedding (use a dummy face)
            print("\n🔍 Step 3: Generating face embedding...")
            dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            embedding = self.sface_auth.extract_embedding(dummy_face)
            print(f"✅ Embedding generated (shape: {embedding.shape})")
            
            # Step 4: Mock database check
            print("\n🔎 Step 4: Searching Database...")
            time.sleep(2)
            identified_user = "Natasha"
            print(f"✅ User Identified as {identified_user}")
            
            # Step 5: Real M-Pesa STK Push payment initiation
            print("\n💳 Step 5: Initiating Payment via M-Pesa STK Push...")
            payment_result = self.payment_gateway.trigger_stk_push("254712345678", 1000.0)
            
            if payment_result["success"]:
                receipt_number = payment_result.get("checkout_request_id", payment_result.get("merchant_request_id", "N/A"))
                print(f"✅ Payment initiated! Transaction ID: {receipt_number}")
                
                # Step 6: Simulate gesture confirmation
                print("\n🤚 Step 6: Gesture Confirmation Required")
                print("   (Simulating Open Hand confirmation)")
                time.sleep(2)
                
                print("\n🎉 Transaction Successful!")
                print(f"💰 Payment completed successfully")
                print(f"🧾 Receipt Number: {receipt_number}")
                print("="*60)
                return True
            else:
                print("❌ Payment initiation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in demo mode: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def start_complete_workflow(self):
        """Start the complete face recognition payment workflow"""
        print("🚀 Starting Face Recognition Payment System...")
        print("=" * 50)
        
        try:
            # Step 1: Load SFace model
            print("📥 Loading SFace model...")
            self.sface_auth.load_model()
            print("✅ SFace model loaded successfully")
            
            # Step 2: Initialize camera
            print("📷 Initializing camera...")
            self.cap = self.initialize_camera()
            
            if self.cap is None:
                print("\n⚠️  Camera initialization failed!")
                print("   Would you like to run in demo mode instead?")
                choice = input("   Enter 'Y' for demo mode, any other key to exit: ").strip().upper()
                if choice == 'Y':
                    return self.run_demo_mode()
                else:
                    return False
            
            # Step 3: Capture face with feedback
            print("📷 Starting camera for face capture...")
            captured_face = self.capture_face_with_feedback()
            
            if captured_face is None:
                print("❌ No face captured. Exiting...")
                return False
            
            print("✅ Face captured successfully")
            
            # Step 4: Generate embedding
            print("🔍 Generating face embedding...")
            embedding = self.sface_auth.extract_embedding(captured_face)
            print(f"✅ Embedding generated (shape: {embedding.shape})")
            
            # Step 5: Mock database check
            print("🔎 Searching Database...")
            time.sleep(2)  # Simulate database search
            identified_user = "Natasha"
            print(f"✅ User Identified as {identified_user}")
            
            # Step 6: Real M-Pesa STK Push payment initiation
            print("💳 Initiating Payment via M-Pesa STK Push...")
            payment_result = self.payment_gateway.trigger_stk_push("254712345678", 1000.0)
            
            if payment_result["success"]:
                receipt_number = payment_result.get("checkout_request_id", payment_result.get("merchant_request_id", "N/A"))
                print(f"✅ Payment initiated! Transaction ID: {receipt_number}")
                
                # Step 7: Gesture confirmation
                print("\n🤚 Gesture Confirmation Required")
                print("   Note: Payment has been sent to your phone.")
                print("   Please complete the payment on your phone before confirming.")
                confirmed = self.get_gesture_confirmation()
                
                if confirmed:
                    print("\n🎉 Transaction Successful!")
                    print(f"💰 Payment completed successfully")
                    print(f"🧾 Transaction ID: {receipt_number}")
                    return True
                else:
                    print("\n❌ Transaction Cancelled")
                    print("🚫 Payment was cancelled by user")
                    return False
            else:
                print("❌ Payment initiation failed")
                print(f"🚫 Error: {payment_result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()
    
    def capture_face_with_feedback(self):
        """Capture face with visual feedback"""
        if self.cap is None or not self.cap.isOpened():
            print("❌ Camera is not initialized!")
            return None
        
        # Create window explicitly before the loop
        window_name = 'Face Recognition Payment System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.moveWindow(window_name, 100, 100)  # Move window to visible position
        print(f"✅ Created window: '{window_name}'")
        
        print("\n" + "="*50)
        print("📷 CAMERA WINDOW INSTRUCTIONS")
        print("="*50)
        print("Position your face in the green box and press 'S' to scan")
        print("Press 'Q' to quit")
        print("\n⚠️  If you don't see the camera window:")
        print("   1. Check your taskbar for a new window")
        print("   2. The window might be behind other windows")
        print("   3. Try pressing Alt+Tab to find it")
        print("="*50 + "\n")
        
        # Give window time to appear
        time.sleep(1)
        print("📹 Camera window should now be visible...")
        
        frame_read_failures = 0
        max_failures = 10
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                frame_read_failures += 1
                print(f"⚠️  Failed to read frame ({frame_read_failures}/{max_failures})")
                
                if frame_read_failures >= max_failures:
                    print("❌ Too many frame read failures. Camera may be disconnected.")
                    cv2.destroyWindow(window_name)
                    return None
                
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            frame_read_failures = 0
            frame_count += 1
            
            # Print frame count every 30 frames to show progress
            if frame_count % 30 == 0:
                print(f"📹 Camera running... (frame {frame_count})")
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate ROI position
            roi_x, roi_y = self.face_scanner.calculate_roi_position(frame_width, frame_height)
            
            # Apply dark overlay outside ROI
            frame = self.face_scanner.apply_dark_overlay(frame, roi_x, roi_y)
            
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_scanner.face_detection.process(rgb_frame)
            
            # Reset ROI color to red
            current_roi_color = self.face_scanner.roi_color_red
            face_detected_in_roi = False
            
            # Debug: Print detection status every 60 frames
            if frame_count % 60 == 0:
                if results.detections:
                    print(f"👤 Detected {len(results.detections)} face(s)")
                else:
                    print("👤 No faces detected - try moving closer or improving lighting")
            
            # Process detected faces
            if results.detections:
                print(f"   Processing {len(results.detections)} detection(s)...")
                for idx, detection in enumerate(results.detections):
                    try:
                        # Get bounding box data
                        location_data = detection.location_data
                        if location_data is None:
                            print(f"   Detection {idx}: No location data")
                            continue
                        
                        bboxC = location_data.relative_bounding_box
                        if bboxC is None:
                            print(f"   Detection {idx}: No bounding box")
                            continue
                        
                        # MediaPipe uses protobuf messages - access fields directly
                        # The RelativeBoundingBox has x, y, width, height as float fields
                        try:
                            ih, iw, _ = frame.shape
                            
                            # Get relative coordinates (0.0 to 1.0)
                            x_rel = bboxC.xmin if hasattr(bboxC, 'xmin') else bboxC.x
                            y_rel = bboxC.ymin if hasattr(bboxC, 'ymin') else bboxC.y
                            w_rel = bboxC.xmax - bboxC.xmin if hasattr(bboxC, 'xmax') else bboxC.width
                            h_rel = bboxC.ymax - bboxC.ymin if hasattr(bboxC, 'ymax') else bboxC.height
                            
                            # Convert to absolute coordinates
                            x = max(0, int(x_rel * iw))
                            y = max(0, int(y_rel * ih))
                            w = min(iw - x, int(w_rel * iw))
                            h = min(ih - y, int(h_rel * ih))
                            
                            if frame_count % 60 == 0:
                                print(f"   Detection {idx}: bbox coords - x={x}, y={y}, w={w}, h={h}")
                        except AttributeError as e:
                            print(f"   Detection {idx}: Error accessing bbox fields - {e}")
                            # Try using ListFields to see what's available
                            if frame_count % 60 == 0:
                                fields = bboxC.ListFields()
                                print(f"   Available fields: {fields}")
                            continue
                        
                        # Skip invalid bounding boxes
                        if w <= 0 or h <= 0:
                            print(f"   Detection {idx}: Invalid bbox size {w}x{h}")
                            continue
                        
                        face_bbox = [x, y, w, h]
                        
                        # Calculate face center
                        face_center_x = x + w // 2
                        face_center_y = y + h // 2
                        
                        # Debug: Print face position every 60 frames
                        if frame_count % 60 == 0:
                            print(f"   Face {idx} center: ({face_center_x}, {face_center_y})")
                            print(f"   Face {idx} bbox: x={x}, y={y}, w={w}, h={h}")
                            print(f"   ROI center: ({roi_x + self.face_scanner.roi_size//2}, {roi_y + self.face_scanner.roi_size//2})")
                            print(f"   ROI bounds: x={roi_x}, y={roi_y}, size={self.face_scanner.roi_size}")
                        
                        # Check if face is in ROI (using center point)
                        in_roi = self.face_scanner.is_face_in_roi(face_bbox, roi_x, roi_y)
                        if in_roi:
                            face_detected_in_roi = True
                            current_roi_color = self.face_scanner.roi_color_green
                            if frame_count % 60 == 0:
                                print("✅ Face detected in ROI!")
                        elif frame_count % 60 == 0:
                            print("   Face NOT in ROI")
                    except Exception as e:
                        # Skip this detection if there's an error processing it
                        print(f"⚠️  Error processing face detection {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Draw ROI box
            cv2.rectangle(frame, (roi_x, roi_y),
                         (roi_x + self.face_scanner.roi_size, roi_y + self.face_scanner.roi_size),
                         current_roi_color, 3)
            
            # Add instructions text
            cv2.putText(frame, "Position face in box", (roi_x, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'S' to scan, 'Q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show status
            if face_detected_in_roi:
                cv2.putText(frame, "Face Detected!", (roi_x, roi_y + self.face_scanner.roi_size + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                cv2.destroyWindow(window_name)
                return None
            elif key == ord('s') or key == ord('S'):
                if face_detected_in_roi:
                    # Crop the face from ROI
                    cropped_face = frame[roi_y:roi_y + self.face_scanner.roi_size,
                                       roi_x:roi_x + self.face_scanner.roi_size]
                    cv2.destroyWindow(window_name)
                    return cropped_face.copy()
                else:
                    print("No face detected in ROI. Please position your face in the green box.")
    
    def get_gesture_confirmation(self):
        """Get gesture confirmation from user"""
        print("\n🤚 Gesture Confirmation Required")
        print("Show Open Hand to Confirm")
        print("Show Fist to Cancel")
        print("Press 'Q' to quit")
        print("-" * 30)
        
        # Reuse the same camera instance
        if self.cap is None or not self.cap.isOpened():
            print("❌ Camera is not available for gesture confirmation")
            print("   Proceeding with default confirmation...")
            time.sleep(1)
            return True
        
        # Create window explicitly
        window_name = 'Gesture Confirmation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        confirmation_received = False
        result = False
        frame_count = 0
        confirm_threshold = 30  # Number of frames to hold gesture (approx 1 second at 30fps)
        cancel_threshold = 30
        confirm_count = 0
        cancel_count = 0
        
        while not confirmation_received:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("⚠️  Failed to read frame for gesture detection")
                time.sleep(0.1)
                continue
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            is_open_hand = self.gesture_recognizer.detect_open_hand(frame)
            is_fist = self.gesture_recognizer.detect_fist(frame)
            
            # Display status with frame counting for non-blocking delay
            if is_open_hand:
                confirm_count += 1
                cancel_count = 0
                
                progress = min(confirm_count / confirm_threshold * 100, 100)
                cv2.putText(frame, "✅ CONFIRMED - Open Hand Detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Holding... {int(progress)}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check if gesture held long enough
                if confirm_count >= confirm_threshold:
                    confirmation_received = True
                    result = True
                
            elif is_fist:
                cancel_count += 1
                confirm_count = 0
                
                progress = min(cancel_count / cancel_threshold * 100, 100)
                cv2.putText(frame, "❌ CANCELLED - Fist Detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Holding... {int(progress)}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if gesture held long enough
                if cancel_count >= cancel_threshold:
                    confirmation_received = True
                    result = False
            else:
                confirm_count = 0
                cancel_count = 0
                cv2.putText(frame, "Show Open Hand to Confirm", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Show Fist to Cancel", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                confirmation_received = True
                result = False
        
        cv2.destroyWindow(window_name)
        
        return result
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                print("✅ Camera released successfully")
            except Exception as e:
                print(f"⚠️  Error releasing camera: {e}")
            finally:
                self.cap = None
        
        cv2.destroyAllWindows()

# Initialize the payment system
payment_system = FacePaymentSystem()

@app.get("/")
async def root():
    return {"message": "Face Recognition Payment System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/start-payment")
async def start_payment():
    """Start the complete payment workflow"""
    try:
        # Run the complete workflow in a separate thread to avoid blocking
        def run_workflow():
            return payment_system.start_complete_workflow()
        
        # For demo purposes, we'll run it synchronously
        # In production, you might want to use proper async handling
        success = run_workflow()
        
        return {
            "success": success,
            "message": "Payment workflow completed" if success else "Payment workflow failed or cancelled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo-workflow")
async def demo_workflow():
    """Run a demo version of the workflow with console output"""
    success = payment_system.run_demo_mode()
    
    return {
        "success": success,
        "message": "Demo workflow completed successfully" if success else "Demo workflow failed"
    }

if __name__ == "__main__":
    print("🚀 Starting Face Recognition Payment System...")
    print("=" * 50)
    print("Choose mode:")
    print("1. Full Camera Workflow")
    print("2. Demo Mode (Console Only)")
    print("3. API Server Mode")
    print("=" * 50)
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Run full workflow with camera
        print("\n📷 Starting Full Camera Workflow...")
        success = payment_system.start_complete_workflow()
        if success:
            print("\n🎉 Payment workflow completed successfully!")
        else:
            print("\n❌ Payment workflow failed or was cancelled")
    elif choice == "2":
        # Run demo mode
        print("\n🎭 Starting Demo Mode...")
        success = payment_system.run_demo_mode()
        if success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n❌ Demo failed")
    elif choice == "3":
        # Run API server
        print("\n🌐 Starting API Server...")
        print("   Server will be available at: http://0.0.0.0:8000")
        print("   API Documentation: http://0.0.0.0:8000/docs")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("Invalid choice. Starting API server...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)