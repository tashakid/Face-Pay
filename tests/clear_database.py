#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def clear_database():
    try:
        from src.database import db

        print("⚠️  WARNING: This will DELETE all user data from Firebase!")
        print("📋 Collection to be cleared:", os.getenv("FIREBASE_COLLECTION", "users"))
        print("🔧 Auto-confirming for embedding dimension fix...")

        print("\n🗑️  Clearing database...")

        collection_name = os.getenv("FIREBASE_COLLECTION", "users")
        docs = db.db.collection(collection_name).get()

        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
            print(f"   Deleted: {doc.id}")

        print(f"\n✅ Database cleared successfully!")
        print(f"   Total documents deleted: {deleted_count}")

    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_database()