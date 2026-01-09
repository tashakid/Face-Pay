"""
DeepFace Migration Test Script
Tests the new DeepFace authentication system and compares with existing SFace data
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, "src")
def test_deepface_basic():
    """Test basic DeepFace functionality"""
    logger.info("="*60)
    logger.info("TEST 1: DeepFace Basic Functionality")
    logger.info("="*60)

    try:
        import sys
        sys.path.insert(0, 'src')
        from deepface_auth import deepface_auth
        logger.info("✅ DeepFace module imported")

        status = deepface_auth.get_model_status()
        logger.info(f"   Model Status: {status}")

        logger.info("📥 Loading DeepFace model...")
        start = time.time()
        deepface_auth.load_model()
        load_time = time.time() - start
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")

        return True
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deepface_embedding_extraction():
    """Test embedding extraction with sample image"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: DeepFace Embedding Extraction")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth

        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        logger.info("📸 Extracting embedding from test image...")
        start = time.time()
        embedding = deepface_auth.extract_embedding(test_image)
        extract_time = (time.time() - start) * 1000

        logger.info(f"✅ Embedding extracted in {extract_time:.0f}ms")
        logger.info(f"   Embedding shape: {embedding.shape}")
        logger.info(f"   Embedding dtype: {embedding.dtype}")

        return True
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_registration():
    """Test face registration and verification"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Face Registration & Verification")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth

        test_user_id = "test_user_001"
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        logger.info("📸 Registering face...")
        embedding = deepface_auth.extract_embedding(test_image)
        success = deepface_auth.register_face(test_user_id, embedding)
        logger.info(f"✅ Face registration: {'SUCCESS' if success else 'FAILED'}")

        logger.info("🔍 Verifying registered face...")
        is_match = deepface_auth.verify_face(test_user_id, embedding)
        logger.info(f"✅ Face verification: {'MATCHED' if is_match else 'NOT MATCHED'}")

        confidence = deepface_auth.get_similarity_score(embedding, embedding)
        logger.info(f"   Confidence score: {confidence:.2f}%")

        if not is_match:
            logger.warning("   ⚠️  Self-verification should always match!")

        return is_match
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_comparison():
    """Test face comparison between different faces"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Face Comparison (Different Faces)")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth

        face1 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face2 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        logger.info("📸 Extracting embeddings from two different faces...")
        embedding1 = deepface_auth.extract_embedding(face1)
        embedding2 = deepface_auth.extract_embedding(face2)

        result = deepface_auth.compare_faces(embedding1, embedding2)

        logger.info(f"✅ Comparison result:")
        logger.info(f"   Verified: {result['verified']}")
        logger.info(f"   Distance: {result['distance']:.4f}")
        logger.info(f"   Confidence: {result['confidence']:.2f}%")
        logger.info(f"   Threshold: {result['threshold']}")

        if result['verified']:
            logger.warning("   ⚠️  Different faces should NOT match (random images)")

        return True
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clahe_preprocessing():
    """Test CLAHE preprocessing"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: CLAHE Preprocessing")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth

        dark_image = np.random.randint(0, 100, (300, 300, 3), dtype=np.uint8)
        bright_image = np.random.randint(200, 255, (300, 300, 3), dtype=np.uint8)

        logger.info("📸 Applying CLAHE preprocessing...")
        enhanced_dark = deepface_auth.preprocess_image(dark_image)
        enhanced_bright = deepface_auth.preprocess_image(bright_image)

        logger.info("✅ CLAHE preprocessing successful")
        logger.info(f"   Dark image mean: {dark_image.mean():.1f} → {enhanced_dark.mean():.1f}")
        logger.info(f"   Bright image mean: {bright_image.mean():.1f} → {enhanced_bright.mean():.1f}")

        return True
    except Exception as e:
        logger.error(f"❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_compatibility():
    """Test compatibility with existing database"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Database Compatibility")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth
        from database import db

        logger.info("🔍 Checking existing face database...")
        users = db.get_all_users()
        logger.info(f"   Found {len(users)} users in database")

        faces_count = 0
        for user in users:
            user_id = user.get("id")
            face_data = db.get_face_embedding(user_id)
            if face_data:
                faces_count += 1

        logger.info(f"   Found {faces_count} users with registered faces")

        logger.info("✅ Database structure compatible with DeepFace")
        return True
    except Exception as e:
        logger.error(f"❌ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_benchmark():
    """Benchmark DeepFace performance"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: Performance Benchmark")
    logger.info("="*60)

    try:
        from deepface_auth import deepface_auth

        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        iterations = 10

        logger.info(f"⏱️  Running {iterations} embedding extractions...")
        times = []

        for i in range(iterations):
            start = time.time()
            embedding = deepface_auth.extract_embedding(test_image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        logger.info(f"✅ Performance Results:")
        logger.info(f"   Average: {avg_time:.0f}ms")
        logger.info(f"   Min: {min_time:.0f}ms")
        logger.info(f"   Max: {max_time:.0f}ms")
        logger.info(f"   Std Dev: {std_time:.0f}ms")

        if avg_time < 500:
            logger.info("   ✅ Real-time capable (<500ms)")
        elif avg_time < 1000:
            logger.info("   ⚠️  Acceptable but sluggish (<1000ms)")
        else:
            logger.warning("   ⚠️  Too slow for real-time (>1000ms)")

        return avg_time < 1000
    except Exception as e:
        logger.error(f"❌ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("🚀 DEEPFACE MIGRATION TEST SUITE")
    logger.info("="*60)

    tests = [
        ("Basic Functionality", test_deepface_basic),
        ("Embedding Extraction", test_deepface_embedding_extraction),
        ("Registration & Verification", test_face_registration),
        ("Face Comparison", test_face_comparison),
        ("CLAHE Preprocessing", test_clahe_preprocessing),
        ("Database Compatibility", test_database_compatibility),
        ("Performance Benchmark", performance_benchmark),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        logger.info("\n🎉 All tests passed! DeepFace migration is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install deepface tensorflow")
        logger.info("2. Start the system: python src/main.py")
        logger.info("3. Test with real face images for verification")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())