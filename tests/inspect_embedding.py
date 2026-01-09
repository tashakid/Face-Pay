#!/usr/bin/env python
"""
Inspect a face embedding directly from database to see what model created it
"""

import numpy as np
import base64

def inspect_embedding_dimensions():
    try:
        from src.database import db

        print("🔍 Inspecting Face Embedding from Database...")
        print("=" * 60)

        faces = db.db.collection("faces").get()

        if not faces:
            print("❌ No face embeddings found in database")
            return

        face_doc = list(faces)[0]
        face_data = face_doc.to_dict()
        embedding_str = face_data.get("embedding")

        if not embedding_str:
            print("❌ No embedding data found")
            return

        base64_bytes = embedding_str.encode("utf-8")
        array_bytes = base64.b64decode(base64_bytes)
        array = np.frombuffer(array_bytes, dtype=np.float32)

        print(f"📊 Embedding Analysis:")
        print(f"   User ID: {face_data.get('user_id')}")
        print(f"   Dimensions: {len(array)}")
        print(f"   Shape: {array.shape}")
        print(f"   Data type: {array.dtype}")
        print()

        print(f"📉 Statistics:")
        print(f"   Min value: {array.min():.6f}")
        print(f"   Max value: {array.max():.6f}")
        print(f"   Mean value: {array.mean():.6f}")
        print(f"   Std deviation: {array.std():.6f}")
        print()

        print("🎯 Model Identification:")
        if len(array) == 512:
            print("   ✅ ArcFace (512 dims) - CORRECT")
        elif len(array) == 128:
            print("   ℹ️  FaceNet (128 dims)")
        elif len(array) == 160:
            print("   ℹ️  DeepID (160 dims)")
        elif len(array) == 2622:
            print("   ❌ VGG-Face (2622 dims) - OLD MODEL")
        elif len(array) == 1024:
            print("   ⚠️  Custom/Unknown model (1024 dims)")
            print()
            print("   💡 This dimension is NOT from standard DeepFace models!")
            print("   💡 Possible causes:")
            print("      - App still running with cached model")
            print("      - Custom model configuration")
            print("      - Serialization issue")
        elif len(array) == 4096:
            print("   ❌ DeepFace (4096 dims) - OLD MODEL")
        else:
            print(f"   ❓ Unknown: {len(array)} dims")

        print()
        print("💡 Recommendations:")
        if len(array) != 512:
            print("   ⚠️  Dimension mismatch detected!")
            print()
            print("   Step 1: COMPLETELY STOP your application:")
            print("      ps aux | grep python")
            print("      kill <PID>")
            print()
            print("   Step 2: Clear database:")
            print("      python3 clear_face_embeddings.py")
            print()
            print("   Step 3: Clear cache:")
            print("      python3 clear_cache.py")
            print()
            print("   Step 4: Start fresh application")
            print()
            print("   Step 5: Re-register face (should be 512 dims)")
        else:
            print("   ✅ Embeddings are correct (512-dim ArcFace)")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_embedding_dimensions()