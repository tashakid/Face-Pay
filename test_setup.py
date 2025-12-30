"""
Test script to verify the Face Recognition Payment System setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn: {uvicorn.__version__}")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow: {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import firebase_admin
        print(f"✅ Firebase Admin: {firebase_admin.__version__}")
    except ImportError as e:
        print(f"❌ Firebase Admin import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access"""
    print("\n📷 Testing camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is accessible")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera frame captured (resolution: {frame.shape})")
            else:
                print("⚠️  Camera accessible but frame capture failed")
            
            cap.release()
            return True
        else:
            print("❌ Camera not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_project_files():
    """Test if all project files exist"""
    print("\n📁 Testing project files...")
    
    required_files = [
        "src/__init__.py",
        "src/main.py",
        "src/vision.py",
        "src/auth.py",
        "src/payment.py",
        "requirements.txt",
        ".env.example",
        ".env",
        "README.md",
        "SETUP_INSTRUCTIONS.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_model_file():
    """Test if SFace model file exists"""
    print("\n🤖 Testing SFace model file...")
    
    model_paths = [
        "src/face_recognition_sface_2021dec.onnx",
        "face_recognition_sface_2021dec.onnx",
        "models/face_recognition_sface_2021dec.onnx"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ SFace model found: {model_path}")
            return True
    
    print("⚠️  SFace model file not found")
    print("   Download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface")
    return False

def test_environment_variables():
    """Test environment variables"""
    print("\n🔧 Testing environment variables...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            # MPESA credentials no longer needed for mock system
            # "MPESA_KEY",
            # "MPESA_SECRET",
            "FIREBASE_CREDENTIALS",
            "JWT_SECRET",
            "MOCK_PAYMENT_ENABLED"
        ]
        
        all_set = True
        for var in required_vars:
            value = os.getenv(var)
            if value:
                if "your_" in value.lower() or "here" in value.lower():
                    print(f"⚠️  {var}: {value} (needs to be updated)")
                else:
                    print(f"✅ {var}: Set")
            else:
                print(f"❌ {var}: Not set")
                all_set = False
        
        return all_set
        
    except ImportError:
        print("❌ python-dotenv not installed")
        return False

def main():
    """Run all tests"""
    print("🚀 Face Recognition Payment System - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Files", test_project_files),
        ("Environment Variables", test_environment_variables),
        ("SFace Model", test_model_file),
        ("Camera Access", test_camera)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready.")
        print("\n🚀 Next steps:")
        print("1. Update .env file with your actual credentials")
        print("2. Download SFace model if not present")
        print("3. Run: python src/main.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n📚 For help, see SETUP_INSTRUCTIONS.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)