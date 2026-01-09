#!/usr/bin/env python
"""
Clear all in-memory caches in the Face-Pay system
Run this when you suspect caching issues
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def clear_all_caches():
    """Clear all in-memory caches"""

    print("🧹 Clearing all in-memory caches...")
    print("=" * 60)

    # Clear DeepFace authenticator cache
    try:
        # Clear the module-level instance's cache
        from src.deepface_auth import deepface_auth

        if hasattr(deepface_auth, 'known_faces'):
            cleared_count = len(deepface_auth.known_faces)
            deepface_auth.known_faces.clear()
            print(f"✅ Cleared DeepFace in-memory cache ({cleared_count} faces removed)")
        else:
            print("✅ No DeepFace cache found")
    except Exception as e:
        print(f"⚠️  Error clearing DeepFace cache: {e}")

    # Clear TensorFlow/Keras caches if present
    try:
        import gc
        gc.collect()
        print("✅ Python garbage collector ran")
    except Exception as e:
        print(f"⚠️  Error running garbage collector: {e}")

    # Optional: Clear DeepFace models from memory (requires reload)
    print("\n💡 Tip: To completely reset DeepFace, restart your application")
    print("   DeepFace models are loaded lazily and will re-download on restart")

    print("\n" + "=" * 60)
    print("✅ All in-memory caches cleared")
    print("=" * 60)

if __name__ == "__main__":
    clear_all_caches()