#!/usr/bin/env python
"""
Test which DeepFace model is being used and what dimensions it produces
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def test_deepface_model():
    try:
        print("🎭 Testing DeepFace Model Configuration...")
        print("=" * 60)

        # Check environment variables
        model_name = os.getenv("DEEPFACE_MODEL", "ArcFace")
        detector = os.getenv("DEEPFACE_DETECTOR", "yunet")
        metric = os.getenv("DEEPFACE_DISTANCE_METRIC", "euclidean_l2")

        print(f"📋 Environment Configuration:")
        print(f"   Model: {model_name}")
        print(f"   Detector: {detector}")
        print(f"   Distance Metric: {metric}")
        print()

        # Import DeepFace auth
        from src.deepface_auth import deepface_auth

        print(f"🤖 DeepFaceAuthenticator State:")
        print(f"   Model Name: {deepface_auth.model_name}")
        print(f"   Detector: {deepface_auth.detector_backend}")
        print(f"   Distance Metric: {deepface_auth.distance_metric}")
        print(f"   Model Loaded: {deepface_auth._model_loaded}")
        print(f"   GPU Enabled: {deepface_auth.gpu_enabled}")
        print()

        # Load the model
        print("📥 Loading DeepFace model...")
        deepface_auth.load_model()
        print()

        # Test with a dummy image
        print("🧪 Testing embedding extraction...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        embedding = deepface_auth.extract_embedding(test_image)

        if embedding is not None:
            print(f"✅ Embedding extracted successfully!")
            print(f"   Shape: {embedding.shape}")
            print(f"   Dimensions: {len(embedding)}")

            # Check if dimensions match expected
            if len(embedding) == 512:
                print(f"   ✅ CORRECT: 512 dimensions (ArcFace)")
            elif len(embedding) == 128:
                print(f"   ⚠️  WARNING: 128 dimensions (FaceNet)")
            elif len(embedding) == 1024:
                print(f"   ❌ WRONG: 1024 dimensions (VGG-Face or old model)")
            elif len(embedding) == 2622:
                print(f"   ❌ WRONG: 2622 dimensions (Facenet512)")
            else:
                print(f"   ❓ UNKNOWN: {len(embedding)} dimensions")

            print()
            print("💡 Expected ArcFace: 512 dimensions")
            print("💡 Your model: {} dimensions".format(len(embedding)))

            if len(embedding) != 512:
                print()
                print("❌ PROBLEM: Using wrong model!")
                print()
                print("🔧 To fix:")
                print("1. Check your .env file has: DEEPFACE_MODEL=ArcFace")
                print("2. Restart your application")
                print("3. Clear cache: python3 clear_cache.py")

        else:
            print("❌ Failed to extract embedding")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_deepface_model()