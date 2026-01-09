#!/bin/env python
"""
Clear all face embeddings from Firebase 'faces' collection
This fixes the 1024-dimension embedding problem
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def clear_face_embeddings():
    try:
        from src.database import db

        print("🗑️  Clearing Face Embeddings...")
        print("=" * 60)
        print("⚠️  This will DELETE all face embeddings from Firebase!")
        print("📋 Collection to be cleared: faces")
        print()

        docs = db.db.collection("faces").get()

        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
            print(f"   Deleted face: {doc.id}")

        print()
        print(f"✅ Face embeddings cleared successfully!")
        print(f"   Total face embeddings deleted: {deleted_count}")
        print()
        print("💡 Next steps:")
        print("   1. Clear in-memory cache: python3 clear_cache.py")
        print("   2. Restart your application")
        print("   3. Re-register all user faces with ArcFace model")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error clearing face embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    clear_face_embeddings()