#!/usr/bin/env python
"""
Test different DeepFace models to find which one produces 1024 dimensions
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Models to test and their expected dimensions
MODELS_TO_TEST = {
    "VGG-Face": 2622,      # 2622 dimensions
    "Facenet": 128,        # 128 dimensions  
    "Facenet512": 512,     # 512 dimensions
    "ArcFace": 512,        # 512 dimensions
    "SFace": 128,          # 128 dimensions
    "Dlib": 128,           # 128 dimensions
    "DeepFace": 4096,      # 4096 dimensions
    "DeepID": 512,         # 512 dimensions
    "OpenFace": 128,       # 128 dimensions
    "GhostFaceNet": 512,   # 512 dimensions
}

def test_model(model_name):
    """Test a single model and return its embedding dimensions"""
    try:
        from deepface import DeepFace

        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = DeepFace.represent(
            img_path=test_image,
            model_name=model_name,
            detector_backend="yunet",
            enforce_detection=False,
            align=True
        )

        if result and len(result) > 0:
            embedding = np.array(result[0]["embedding"])
            return len(embedding), embedding.shape

        return None, None

    except Exception as e:
        return None, str(e)


print("🧪 Testing DeepFace Models to find 1024-dim producer...")
print("=" * 80)

found_1024 = False

for model_name, expected_dims in sorted(MODELS_TO_TEST.items()):
    print(f"\n🔍 Testing {model_name} (expected: {expected_dims} dims)...")

    try:
        dims, shape = test_model(model_name)

        if isinstance(dims, int):
            print(f"   ✅ Embedded with: {dims} dimensions")

            if dims == 1024:
                print(f"   🎯 FOUND! This model produces 1024 dimensions!")
                print(f"   📋 Set in .env: DEEPFACE_MODEL={model_name}")
                found_1024 = True
                break
            else:
                print(f"   ℹ️  Not 1024 dims, but {dims}")
        else:
            print(f"   ❌ Failed: {shape}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)

if found_1024:
    print("✅ FOUND A 1024-dim MODEL ABOVE!")
    print("\n🔧 Update your .env file:")
    print("   DEEPFACE_MODEL=<model_name_shown_above>")
    print("   DEEPFACE_DISTANCE_METRIC=euclidean_l2")
    print("   DEEPFACE_DETECTOR=yunet")
    print("\nThen:")
    print("   python3 clear_cache.py")
    print("   python3 clear_face_embeddings.py")
    print("   # Restart your application")
else:
    print("⚠️  No standard DeepFace model produces exactly 1024 dimensions")
    print("\n💡 Your system might be using:")
    print("   - A custom/modified model")
    print("   - FaceNet (128 dims) being saved incorrectly")
    print("   - An older version of DeepFace with different models")
    print("\n🔧 Solution: Use ArcFace (512 dims) which works correctly")
    print("   1. Stop your application")
    print("   2. Run: python3 clear_face_embeddings.py")
    print("   3. Run: python3 clear_cache.py")
    print("   4. Start fresh application")
    print("   5. Re-register faces (will be 512 dims)")

print("=" * 80)