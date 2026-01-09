#!/usr/bin/env python
"""
Check embedding dimensions in Firebase database
Identifies dimension mismatches that cause recognition errors
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def check_embedding_dimensions():
    """Check all face embeddings in database for dimension consistency"""

    try:
        from src.database import db

        print("🔍 Checking Face Embedding Dimensions...")
        print("=" * 60)

        faces = db.db.collection("faces").get()
        total_faces = len(list(faces))

        if total_faces == 0:
            print("✅ No face embeddings found in database")
            return

        print(f"📊 Total face embeddings: {total_faces}\n")

        faces = db.db.collection("faces").get()

        dimensions_by_user = {}
        all_dimensions = set()

        for face_doc in faces:
            face_data = face_doc.to_dict()
            user_id = face_data.get("user_id")
            embedding_str = face_data.get("embedding")

            if embedding_str:
                try:
                    embedding = db.deserialize_array(embedding_str)
                    dim = len(embedding)
                    all_dimensions.add(dim)

                    if user_id not in dimensions_by_user:
                        dimensions_by_user[user_id] = set()

                    dimensions_by_user[user_id].add(dim)

                except Exception as e:
                    print(f"❌ Error reading embedding: {e}")

        print("📊 Dimensions in Database:")
        print("-" * 60)
        for dim in sorted(all_dimensions):
            count = sum(1 for dims in dimensions_by_user.values() if dim in dims)
            print(f"   {dim} dimensions: {count} users")

        print("\n📊 Dimensions by User:")
        print("-" * 60)

        for user_id, dims in sorted(dimensions_by_user.items()):
            dims_str = ", ".join(str(d) for d in sorted(dims))

            if len(dims) == 1:
                dim = list(dims)[0]
                print(f"✅ {user_id}: {dim} dimensions (consistent)")
            else:
                print(f"⚠️  {user_id}: Mixed - {dims_str} (INCONSISTENT)")

        if not dimensions_by_user:
            print("No embeddings found")

        print("\n" + "=" * 60)

        if len(all_dimensions) > 1:
            print("⚠️  WARNING: Multiple embedding dimensions found!")
            print("   This will cause recognition issues.")
            print("   Clear database and re-register with a single model.")
        elif len(all_dimensions) == 1:
            dim = list(all_dimensions)[0]
            print(f"✅ All embeddings consistent: {dim} dimensions")

            if dim in [512, 128, 2622, 160]:
                print("   ✅ Recognized dimension")
            elif dim == 1024:
                print("   ✅ Custom model dimension (1024 dims)")
                print("   💡 To prevent errors:")
                print("      The system will now adapt to 1024-dim embeddings")

        # Check for in-memory cache issues
        print("\n🧹 Checking in-memory cache...")
        try:
            from deepface_auth import deepface_auth

            if deepface_auth.known_faces:
                print(f"   ⚠️  In-memory cache has {len(deepface_auth.known_faces)} faces")
                print("      → Run: python clear_cache.py to clear")
                print("      → This cache can cause dimension mismatches")
            else:
                print("   ✅ In-memory cache is empty")
        except Exception as e:
            print(f"   ⚠️  Could not check in-memory cache: {e}")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Error checking dimensions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_embedding_dimensions()