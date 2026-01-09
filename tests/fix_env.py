#!/usr/bin/env python
"""
Add DeepFace configuration to .env file if missing
"""

import os

def fix_env_file():
    env_path = ".env"

    if not os.path.exists(env_path):
        print("❌ .env file not found!")
        return False

    print("🔧 Fixing .env file...")
    print("=" * 60)

    required_settings = {
        "DEEPFACE_MODEL": "ArcFace",
        "DEEPFACE_DETECTOR": "yunet",
        "DEEPFACE_DISTANCE_METRIC": "euclidean_l2",
        "PAYMENT_THRESHOLD": "0.35"
    }

    with open(env_path, 'r') as f:
        lines = f.readlines()

    updated = False
    new_lines = []
    existing_keys = set()

    for line in lines:
        line = line.strip()

        if '=' in line and not line.startswith('#'):
            key = line.split('=')[0]

            existing_keys.add(key)

            if key in required_settings:
                new_value = required_settings[key]
                new_line = f"{key}={new_value}\n"
                if new_line != f"{line}\n":
                    print(f"   ✅ Updated: {key} = {new_value}")
                    updated = True
                    new_lines.append(new_line)
                else:
                    print(f"   ✓ Already set: {key} = {new_value}")
                    new_lines.append(line + '\n')
            else:
                new_lines.append(line + '\n')
        else:
            new_lines.append(line + '\n' if line and not line.endswith('\n') else line)

    for key, value in required_settings.items():
        if key not in existing_keys:
            print(f"   ➕ Added: {key} = {value}")
            new_lines.append(f"{key}={value}\n")
            updated = True

    if updated:
        with open(env_path, 'w') as f:
            f.writelines(new_lines)

        print()
        print("✅ .env file updated successfully!")
        print()
        print("📋 Configuration:")
        print("   DEEPFACE_MODEL=ArcFace")
        print("   DEEPFACE_DETECTOR=yunet")
        print("   DEEPFACE_DISTANCE_METRIC=euclidean_l2")
        print("   PAYMENT_THRESHOLD=0.35")
        print()
        print("🔧 Next steps:")
        print("   1. Stop your application: kill <PID>")
        print("   2. Clear cache: python3 clear_cache.py")
        print("   3. Clear faces: python3 clear_face_embeddings.py")
        print("   4. Start application fresh")
        print("   5. Re-register face (will be 512-dim now)")
    else:
        print("✅ .env file already has correct configuration!")

    print("=" * 60)
    return updated

if __name__ == "__main__":
    fix_env_file()