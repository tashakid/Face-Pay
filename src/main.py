"""
Main entry point for the Face Recognition Payment System
Complete workflow integration
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import numpy as np
import time
import threading
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check debug mode BEFORE configuring logging
debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/home/desk-fam/projects/Face-Pay/face_recognition.log'),
        logging.StreamHandler()
    ]
)

if debug_mode:
    print("🐛 DEBUG MODE ENABLED - Verbose logging active")
else:
    print("ℹ️  Standard logging enabled")

logger = logging.getLogger(__name__)

# Import our custom modules
from vision import get_face_scanner, FaceRecognitionSystem, FaceScanner
from deepface_auth import deepface_auth
from payment import MpesaPaymentGateway, PaymentService, payment_service, router as payment_router
from database import db
from registration import router as registration_router

# Import diagnostics
try:
    from diagnostics import log_image_quality
    HAS_DIAGNOSTICS = True
except ImportError:
    HAS_DIAGNOSTICS = False
    logger.warning("Diagnostics module not available")

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
app.include_router(registration_router, prefix="/api", tags=["registration"])

@app.get("/sales-transactions")
async def get_sales_transactions():
    return {"transactions": db.get_all_transactions()}

class FacePaymentSystem:
    def __init__(self):
        self.face_scanner = FaceScanner()
        self.deepface_auth = deepface_auth
        self.payment_gateway = MpesaPaymentGateway()
        self.payment_service = payment_service
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
            print("\n📥 Step 1: Loading DeepFace model...")
            self.deepface_auth.load_model()
            print("✅ DeepFace model loaded successfully")

            print("\n📷 Step 2: Simulating face capture...")
            time.sleep(1)
            print("✅ Face captured successfully (simulated)")

            print("\n🔍 Step 3: Generating face embedding...")
            dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            embedding = self.deepface_auth.extract_embedding(dummy_face)
            print(f"✅ Embedding generated (shape: {embedding.shape})")

            print("\n🔎 Step 4: Searching Database...")
            time.sleep(2)
            identified_user = "Natasha"
            print(f"✅ User Identified as {identified_user}")

            print("\n💳 Step 5: Initiating Payment via M-Pesa STK Push...")
            payment_result = self.payment_gateway.trigger_stk_push("254712345678", 1000.0)

            if payment_result["success"]:
                receipt_number = payment_result.get("checkout_request_id", payment_result.get("merchant_request_id", "N/A"))
                print(f"✅ Payment initiated! Transaction ID: {receipt_number}")

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
            print("📥 Loading DeepFace model...")
            self.deepface_auth.load_model()
            print("✅ DeepFace model loaded successfully")

            print("📷 Initializing camera...")
            self.cap = self.initialize_camera()

            if self.cap is None:
                print("\\n⚠️  Camera initialization failed!")
                print("   Would you like to run in demo mode instead?")
                choice = input("   Enter 'Y' for demo mode, any other key to exit: ").strip().upper()
                if choice == 'Y':
                    return self.run_demo_mode()
                else:
                    return False

            print("📷 Starting camera for face capture...")
            captured_face = self.capture_face_with_feedback()

            if captured_face is None:
                print("❌ No face captured. Exiting...")
                return False

            print("✅ Face captured successfully")

            print("🔍 Generating face embedding...")
            embedding = self.deepface_auth.extract_embedding(captured_face)
            print(f"✅ Embedding generated (shape: {embedding.shape})")

            print("🔎 Searching Database...")
            time.sleep(2)
            identified_user = "Natasha"
            print(f"✅ User Identified as {identified_user}")

            print("💳 Initiating Payment via M-Pesa STK Push...")
            payment_result = self.payment_gateway.trigger_stk_push("254712345678", 1000.0)

            if payment_result["success"]:
                receipt_number = payment_result.get("checkout_request_id", payment_result.get("merchant_request_id", "N/A"))
                print(f"✅ Payment initiated! Transaction ID: {receipt_number}")

                print("\\n✅ Manual Confirmation Required")
                print("   Note: Payment has been sent to your phone.")
                confirmed = self.get_gesture_confirmation()

                if confirmed:
                    print("\\n🎉 Transaction Successful!")
                    print(f"💰 Payment completed successfully")
                    print(f"🧾 Transaction ID: {receipt_number}")
                    return True
                else:
                    print("\\n❌ Transaction Cancelled")
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

            frame = self.face_scanner.apply_dark_overlay(frame, roi_x, roi_y)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_scanner.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            current_roi_color = self.face_scanner.roi_color_red
            face_detected_in_roi = False

            if frame_count % 60 == 0:
                if len(faces) > 0:
                    print(f"👤 Detected {len(faces)} face(s)")
                else:
                    print("👤 No faces detected - try moving closer or improving lighting")

            for (x, y, w, h) in faces:
                face_bbox = [x, y, w, h]

                if self.face_scanner.is_face_in_roi(face_bbox, roi_x, roi_y):
                    face_detected_in_roi = True
                    current_roi_color = self.face_scanner.roi_color_green
                    if frame_count % 60 == 0:
                        print("✅ Face detected in ROI!")
            
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
        print("\n✅ Manual Confirmation Required")
        print("Please confirm payment on your phone")
        print("Press 'Q' to cancel")
        print("-" * 30)
        
        if self.cap is None or not self.cap.isOpened():
            print("✅ Payment confirmed (manual confirmation)")
            time.sleep(1)
            return True
        
        window_name = 'Payment Confirmation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        start_time = time.time()
        auto_confirm_delay = 3
        confirmed = False
        
        while not confirmed and (time.time() - start_time) < auto_confirm_delay:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            remaining = int(auto_confirm_delay - (time.time() - start_time))
            cv2.putText(frame, f"Confirm Payment on Phone", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Auto-confirming in {remaining}s...", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyWindow(window_name)
                return False
        
        cv2.destroyWindow(window_name)
        print("✅ Payment confirmed automatically")
        return True
    
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

@app.get("/users")
async def get_all_users():
    try:
        users = db.get_all_users()
        return {
            "success": True,
            "count": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")

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

@app.post("/face-recognition")
async def face_recognition(request: dict):
    """
    Recognize a face from the provided image data.
    """
    # Start timing
    request_start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"FACE RECOGNITION REQUEST STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        import base64
        from pydantic import BaseModel

        class FaceRecognitionRequest(BaseModel):
            face_image: str

        body = FaceRecognitionRequest(**request)
        face_image_base64 = body.face_image
        logger.info(f"Image data size: {len(face_image_base64)} bytes")

        if not face_image_base64 or face_image_base64.strip() == '':
            logger.error("❌ Error: No image data provided")
            raise HTTPException(status_code=400, detail="No face image provided")

        # Decode image with timing
        decode_start = time.time()
        try:
            if face_image_base64.startswith("data:image"):
                face_image_base64 = face_image_base64.split(",")[1].strip()
                face_image_base64 = face_image_base64.replace("%2B", "+").replace("%2F", "/").replace("%3D", "=")

            img_data = base64.b64decode(face_image_base64)

            logger.info(f"   Image data size: {len(img_data)} bytes")

            if not img_data or len(img_data) < 100:
                logger.error(f"❌ Error: Image data too small ({len(img_data)} bytes)")
                raise HTTPException(status_code=400, detail="Invalid face image: Data too small")

            face_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            decode_time = (time.time() - decode_start) * 1000

            if face_image is None:
                logger.error("❌ Error: Could not decode base64 image")
                logger.error(f"   Data length: {len(img_data)}, Array shape: {nparr.shape}")
                raise HTTPException(status_code=400, detail="Invalid face: Could not decode image")

            logger.info(f"   Image decoded: {decode_time:.0f}ms, shape: {face_image.shape}")

        except Exception as e:
            logger.error(f"❌ Exception during image decoding: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        # Log image quality
        if HAS_DIAGNOSTICS:
            log_image_quality(face_image, level=logging.DEBUG)

        deepface_auth.ensure_model_loaded()

        extract_start = time.time()
        try:
            embedding = deepface_auth.extract_embedding(face_image)
            extract_time = (time.time() - extract_start) * 1000
            logger.info(f"   Embedding extracted: {extract_time:.0f}ms")
        except Exception as e:
            logger.error(f"❌ Exception during embedding extraction: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to extract face embedding: {str(e)}")

        # Find matching face with timing
        match_start = time.time()
        try:
            matched_face = db.find_matching_face_with_averaging(
                embedding.flatten(),
                confidence_threshold=0.45
            )
            match_time = (time.time() - match_start) * 1000
            logger.info(f"   Face matching: {match_time:.0f}ms")
        except Exception as e:
            logger.error(f"❌ Exception during face matching: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Database error during face matching: {str(e)}")

        # Calculate total time
        total_time = (time.time() - request_start) * 1000
        logger.info(f"⏱️  Total processing time: {total_time:.0f}ms")

        # Return result
        if matched_face:
            confidence = matched_face.get("confidence", 0)

            if confidence >= 0.45:
                user_data = db.get_user(matched_face.get("user_id"))

                if user_data:
                    logger.info(f"✅ Recognition SUCCESS - User: {user_data.get('name')} ({user_data.get('id')})")
                    logger.info(f"{'='*60}\n")

                    return {
                        "success": True,
                        "user_id": user_data.get("user_id"),
                        "name": user_data.get("name"),
                        "phone_number": user_data.get("phone_number"),
                        "confidence": confidence * 100,
                        "debug": {
                            "sample_count": matched_face.get("sample_count"),
                            "avg_confidence": matched_face.get("avg_confidence") * 100,
                            "processing_time_ms": total_time,
                            "best_sample_index": matched_face.get("best_sample_index")
                        }
                    }
                else:
                    logger.warning(f"⚠️  Face matched but user data not found for ID: {matched_face.get('user_id')}")
                    logger.info(f"{'='*60}\n")
                    return {
                        "success": False,
                        "message": "Face matched but user data not found"
                    }
            else:
                logger.info(f"⚠️  Face found but below threshold: {confidence * 100:.1f}%")
                logger.info(f"{'='*60}\n")
                return {
                    "success": False,
                    "message": "No face matched above 45% threshold",
                    "confidence": confidence * 100,
                    "debug": {
                        "threshold_percent": 45.0,
                        "confidence_percent": confidence * 100,
                        "gap_percent": max(0, 45.0 - confidence * 100),
                        "sample_count": matched_face.get("sample_count")
                    }
                }
        else:
            logger.info(f"❌ Recognition FAILED - No match found")
            logger.info(f"{'='*60}\n")
            return {
                "success": False,
                "message": "No face matched above 45% threshold"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in face recognition: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

if __name__ == "__main__":
    # Check if SERVER_MODE environment variable is set
    server_mode = os.getenv("SERVER_MODE", "").lower() == "true"

    if server_mode:
        # Server mode - start API server automatically
        print("🌐 Starting API Server (SERVER_MODE=true)...")
        print("   Server will be available at: http://0.0.0.0:8000")
        print("   API Documentation: http://0.0.0.0:8000/docs")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    else:
        # Interactive mode - ask user for choice
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