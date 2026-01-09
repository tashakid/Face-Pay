#!/usr/bin/env python
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

def inspect_database():
    try:
        from database import db

        print("🔍 Inspecting Firebase Database...")
        print("=" * 60)

        collection_name = os.getenv("FIREBASE_COLLECTION", "users")
        print(f"Collection: {collection_name}")
        print("-" * 60)

        docs = db.db.collection(collection_name).get()
        total_docs = len(list(docs))

        if total_docs == 0:
            print("✅ Database is empty!")
            return
        else:
            print(f"📊 Found {total_docs} document(s)")
            print()

        docs = db.db.collection(collection_name).get()
        embedding_dimensions = set()

        for doc in docs:
            data = doc.data()
            print(f"\n👤 User ID: {doc.id}")
            print(f"   Name: {data.get('name', 'N/A')}")
            print(f"   Phone: {data.get('phone_number', 'N/A')}")

            face_samples = data.get('face_samples', [])
            print(f"   Face samples: {len(face_samples)}")

            if face_samples:
                sample = face_samples[0]
                embedding = sample.get('embedding', [])
                dim = len(embedding)
                embedding_dimensions.add(dim)
                print(f"   Embedding dimension: {dim}")

                print(f"   Sample embedding (first 5 values): {embedding[:5]}")

        print("\n" + "=" * 60)
        print(f"📊 Summary:")
        print(f"   Total users: {total_docs}")
        print(f"   Embedding dimensions found: {embedding_dimensions}")

        if 1024 in embedding_dimensions:
            print("\n⚠️  WARNING: 1024-dim embeddings found (old data)")
            print("   You need to clear the database and re-register users")
        elif 512 in embedding_dimensions:
            print("\n✅ 512-dim embeddings found (ArcFace compatible)")
        else:
            print("\n❓ Unknown embedding dimensions found")

    except Exception as e:
        print(f"❌ Error inspecting database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    inspect_database()