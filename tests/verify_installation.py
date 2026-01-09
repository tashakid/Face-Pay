#!/usr/bin/env python
"""
Verification script for Face-Pay dependency installation
Tests all critical components to ensure system is operational
"""

import sys

print("=" * 60)
print("FACE-Pay Installation Verification")
print("=" * 60)

errors = []
warnings = []

# Test 1: TensorFlow
print("\n1️⃣  Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow: {tf.__version__}")
    print(f"   - GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except Exception as e:
    errors.append(f"TensorFlow: {e}")
    print(f"   ❌ TensorFlow FAILED: {e}")
    sys.exit(1)

# Test 2: DeepFace
print("\n2️⃣  Testing DeepFace...")
try:
    from deepface import DeepFace
    print("   ✅ DeepFace loaded successfully")
except Exception as e:
    errors.append(f"DeepFace: {e}")
    print(f"   ❌ DeepFace FAILED: {e}")
    sys.exit(1)

# Test 3: Firebase
print("\n3️⃣  Testing Firebase...")
try:
    import firebase_admin
    from firebase_admin import firestore
    print(f"   ✅ Firebase Admin: {firebase_admin.__version__}")
except Exception as e:
    errors.append(f"Firebase: {e}")
    print(f"   ❌ Firebase FAILED: {e}")
    sys.exit(1)

# Test 4: gRPC (used by Firebase internally)
print("\n4️⃣  Testing gRPC...")
try:
    import grpc
    print(f"   ✅ gRPC: {grpc.__version__}")
except Exception as e:
    errors.append(f"gRPC: {e}")
    print(f"   ❌ gRPC FAILED: {e}")
    sys.exit(1)

# Test 5: Protobuf
print("\n5️⃣  Testing Protobuf...")
try:
    import google.protobuf
    print(f"   ✅ Protobuf: {google.protobuf.__version__}")
    google.protobuf.__version__
except Exception as e:
    errors.append(f"Protobuf: {e}")
    print(f"   ❌ Protobuf FAILED: {e}")
    sys.exit(1)

# Test 6: OpenCV
print("\n6️⃣  Testing OpenCV...")
try:
    import cv2
    print(f"   ✅ OpenCV: {cv2.__version__}")
except Exception as e:
    errors.append(f"OpenCV: {e}")
    print(f"   ❌ OpenCV FAILED: {e}")
    sys.exit(1)

# Test 7: NumPy
print("\n7️⃣  Testing NumPy...")
try:
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
except Exception as e:
    errors.append(f"NumPy: {e}")
    print(f"   ❌ NumPy FAILED: {e}")
    sys.exit(1)

# Test 8: MediaPipe should NOT be available
print("\n8️⃣  Testing MediaPipe (should be removed)...")
try:
    import mediapipe
    print(f"   ⚠️  MediaPipe: {mediapipe.__version__} (should be removed)")
    warnings.append("MediaPipe is still installed but should not be used")
except ImportError:
    print("   ✅ MediaPipe: Not found (as expected)")

# Test 9: tf-keras
print("\n9️⃣  Testing tf-keras...")
try:
    import tf_keras
    print(f"   ✅ tf-keras: {tf_keras.__version__}")
except Exception as e:
    warnings.append(f"tf-keras: {e}")
    print(f"   ⚠️  tf-keras warning: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("❌ VERIFICATION FAILED")
    for error in errors:
        print(f"   - {error}")
    sys.exit(1)
else:
    print("🚀 ALL SYSTEMS OPERATIONAL - Ready for production!")
    if warnings:
        print("\n⚠️  Non-critical warnings:")
        for warning in warnings:
            print(f"   - {warning}")
print("=" * 60)