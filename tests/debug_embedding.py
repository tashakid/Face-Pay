#!/usr/bin/env python
"""
Debug: Check what model the running application is ACTUALLY using
by examining intermediate embedding values
"""

import numpy as np
import os

def test_embedding_extraction():
    try:
        from src.deepface_auth import deepface_auth

        print("🔍 Debug: Checking Active Model Configuration")
        print("=" * 60)

        print(f"Environment DEEPFACE_MODEL: {os.getenv('DEEPFACE_MODEL')}")
        print(f"Authenticator model_name: {deepface_auth.model_name}")
        print(f"Model loaded: {deepface_auth._model_loaded}")
        print()

        print("📥 Loading model if not already loaded...")
        deepface_auth.load_model()
        print()

        # Test with actual image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        print("🔧 Extracting embedding from test image...")
        embedding = deepface_auth.extract_embedding(test_image)

        if embedding is not None:
            print(f"✅ Embedding extracted")
            print(f"   Shape: {embedding.shape}")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            print(f"   Min: {embedding.min():.6f}")
            print(f"   Max: {embedding.max():.6f}")
            print(f"   Mean: {embedding.mean():.6f}")
            print()

            # Now check what gets saved to database
            print("💾 Testing serialization to database format...")
            from src.database import db

            embedding_bytes = embedding.tobytes()
            base64_bytes = __import__('base64').b64encode(embedding_bytes)
            embedding_str = base64_bytes.decode('utf-8')

            # Deserialize back
            deserialized = db.deserialize_array(embedding_str)

            print(f"   Original dims: {len(embedding)}")
            print(f"   Deserialized dims: {len(deserialized)}")
            print(f"   Match: {len(embedding) == len(deserialized)}")

            print()
            if len(embedding) == 512:
                print("✅ ArcFace model is working correctly (512 dims)")
            elif len(embedding) == 1024:
                print("❌ PROBLEM: System is producing 1024 dimensions")
                print()
                print("🔍 This means:")
                print("   - ArcFace is NOT being used for face registration")
                print("   - There might be a fallback to another model")
                print("   - The registration code path is different")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding_extraction()