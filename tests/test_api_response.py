#!/usr/bin/env python
"""
Test the face recognition API endpoint directly and show full response
"""

import base64
import requests
import json

def test_face_recognition_api():
    # Test with a recent base64 image or create a test one
    print("🧪 Testing Face Recognition API Endpoint")
    print("=" * 60)

    # Create a minimal test image (red square)
    import numpy as np
    import cv2

    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    test_image[:, :] = [0, 0, 255]  # Red

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    print("📡 Calling API: http://localhost:8000/face-recognition")
    print()

    try:
        response = requests.post(
            "http://localhost:8000/face-recognition",
            json={"face_image": img_base64},
            timeout=10
        )

        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        print()

        try:
            data = response.json()
            print("📦 Response Body:")
            print(json.dumps(data, indent=2))
            print()

            if data.get("success"):
                print("✅ SUCCESS - Face recognized!")
                print(f"   User ID: {data.get('user_id')}")
                print(f"   Name: {data.get('name')}")
                print(f"   Confidence: {data.get('confidence')}")

                if not data.get("user_id"):
                    print()
                    print("⚠️  WARNING: user_id is missing or None!")
                    print("   This is why the frontend fails - it checks result.user_id")
            else:
                print("❌ FAILURE - Face not recognized")
                if "message" in data:
                    print(f"   Message: {data['message']}")

        except Exception as e:
            print("⚠️  Could not parse JSON response")
            print(f"   Raw response: {response.text[:500]}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to API")
        print("   Make sure the backend is running on localhost:8000")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("=" * 60)

if __name__ == "__main__":
    test_face_recognition_api()