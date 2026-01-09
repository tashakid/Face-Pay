# Face-Pay Caching and Dimension Issues - Fix Guide

## 🔍 Problems Identified

### 1. In-Memory Caching (NOT Database Cache)
**Location**: `src/deepface_auth.py` line 86

```python
self.known_faces = {}  # Stores embeddings in memory
```

**The Issue**:
- Face embeddings are stored in a Python dictionary (`known_faces`)
- This dictionary persists globally because `deepface_auth` is initialized at module level
- **Firebase Firestore does NOT cache** - the cache is in your application's memory
- When you restart your app, old embeddings might remain in memory

**Why it causes dimension errors**:
- If you registered faces with an older model (1024 dimensions)
- Then switched to ArcFace (512 dimensions)
- The old 1024-dim embeddings stay in memory
- When comparing 512-dim new embeddings with 1024-dim cached ones → **Dimension Mismatch Error**

### 2. Database Embedding Dimension Mismatch
**Issue**: Face embeddings in Firebase Firestore may have different dimensions

**ArcFace (Current)**: 512 dimensions ✅
**Old SFace/DeepFace Models**: 1024 dimensions ⚠️

**What happens**:
```python
embedding_new = [512 values]     # ArcFace
embedding_old = [1024 values]    # Old model in DB

# Shape mismatch when comparing
score = cosine_similarity(embedding_new, embedding_old)  # ERROR!
```

---

## 🛠️ Solutions Implemented

### Solution 1: Cache Clearing Script
**File**: `clear_cache.py`

Clears all in-memory caches:
```bash
python3 clear_cache.py
```

**What it does**:
- Clears `deepface_auth.known_faces` dictionary
- Runs Python garbage collector
- Allows fresh embedding loads

### Solution 2: Dimension Checker
**File**: `check_dimensions.py`

Diagnoses dimension issues:
```bash
python3 check_dimensions.py
```

**What it checks**:
- All face embeddings in Firebase Firestore
- Identifies 512 vs 1024 dimension embeddings
- Detects mixed dimensions per user
- Checks in-memory cache status

### Solution 3: Database Dimension Validation
**File**: `src/database.py` (updated)

Added warning when encountering wrong dimensions:
```python
def deserialize_array(self, array_str: str, expected_dim: int = 512):
    array = np.frombuffer(array_bytes, dtype=np.float32)

    actual_dim = len(array)
    if actual_dim != expected_dim:
        logger.warning(f"⚠️  Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
```

### Solution 4: API Cache Clear Endpoint
**Endpoint**: `POST /api/auth/clear-cache`

Clear cache via API:
```bash
curl -X POST http://localhost:8000/api/auth/clear-cache
```

---

## 🚀 How to Fix Your System

### Step 1: Check Your Current State

```bash
# Activate your virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run dimension checker
python3 check_dimensions.py
```

**Look for warnings**:
- `⚠️ Old (1024-dim) embeddings found` → Database has old embeddings
- `⚠️ Mixed dimension embeddings found` → User has both 512 and 1024
- `⚠️ In-memory cache has X faces` → Clear the cache

### Step 2: Clear In-Memory Cache

```bash
# Option A: Clear cache via script
python3 clear_cache.py

# Option B: Clear cache via API (if app is running)
curl -X POST http://localhost:8000/api/auth/clear-cache

# Option C: Restart your application (simplest)
# Stop the app and restart it
```

### Step 3: Fix Database Dimension Issues

**If you see 1024-dim embeddings in database**:

```bash
# Option A: Wipe the database (cleanest fix)
python3 clear_database.py

# Option B: Manually update specific users (more complex)
# Use Firebase Console to delete specific face documents
```

Then re-register all users' faces.

### Step 4: Re-Register Faces

After clearing database:
1. Start your application
2. For each user, register their face again
3. Verify embeddings are now 512 dimensions

---

## 🔬 Verify the Fix

### Run Dimension Check Again:
```bash
python3 check_dimensions.py
```

**Expected output**:
```
✅ All embeddings have correct 512-dimensions
✅ In-memory cache is empty
```

### Test Face Recognition:
1. Register a new user face
2. Try to recognize that user
3. Should work without dimension errors

---

## 💡 Preventing Future Issues

### 1. Use the Clear Cache Regularly
```bash
# Clear cache before testing major changes
python3 clear_cache.py
```

### 2. Monitor Dimensions
```bash
# Check dimensions after registering users
python3 check_dimensions.py
```

### 3. Restart Application After Code Changes
- Always restart after changing face recognition models
- In-memory caches need refresh

### 4. Keep Consistent Model Configuration
Don't switch between different DeepFace models:
```python
# Keep this consistent in .env
DEEPFACE_MODEL=ArcFace  # Don't change to VGG-Face, etc.
```

---

## 🐛 Common Error Messages

### `ValueError: operands could not be broadcast together with shapes`
**Cause**: Dimension mismatch (512 vs 1024)

**Fix**:
```bash
python3 clear_database.py
# Re-register all faces
```

### `RuntimeError: dimensions must be equal`
**Cause**: In-memory cache has old dimensions

**Fix**:
```bash
python3 clear_cache.py
# Or restart application
```

### Faces Not Recognizing Despite Correct Registration
**Cause**: In-memory cache outdated

**Fix**:
```bash
python3 clear_cache.py
```

---

## 📊 Understanding the Cache System

```
┌─────────────────────────────────────┐
│  Your Application (Python)          │
│                                     │
│  ┌────────────────────────────────┐ │
│  │ deepface_auth.known_faces {}   ││ ← IN-MEMORY CACHE (your problem!)
│  │ user_id → [512 embedding]     ││
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
                 ↕
            (through API)
                 ↕
┌─────────────────────────────────────┐
│  Firebase Firestore (Database)      │
│                                     │
│  Collection: faces                  │
│  {                                  │
│    "user_id": "...",                ││ ← DATABASE (NOT cached)
│    "embedding": "base64..."         ││
│  }                                  │
│                                     │
└─────────────────────────────────────┘
```

**Key Insights**:
1. **Firebase does NOT cache** - data is always fresh from Firestore
2. **Python dictionary DOES cache** - persists until cleared
3. Both can have different dimensions → **Conflict!**

---

## 🎯 Summary Checklist

- [ ] Run `python3 check_dimensions.py` to diagnose issues
- [ ] If old dimensions found → `python3 clear_database.py`
- [ ] Always clear cache: `python3 clear_cache.py`
- [ ] Or restart application after changes
- [ ] Re-register face embeddings
- [ ] Verify with `python3 check_dimensions.py` again
- [ ] Test face recognition

---

## 🆘 Still Having Issues?

1. Check your `.env` file:
   ```bash
   DEEPFACE_MODEL=ArcFace
   DEEPFACE_DETECTOR=yunet
   DEEPFACE_DISTANCE_METRIC=euclidean_l2
   PAYMENT_THRESHOLD=0.35
   ```

2. Verify DeepFace is working:
   ```bash
   curl http://localhost:8000/api/auth/model-status
   ```
   Should show: `model_loaded: true`

3. Check application logs for specific dimension errors

4. Try the API clear cache endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/auth/clear-cache
   ```

---

## 📝 Technical Details

### Why 512 vs 1024 Dimensions?

- **ArcFace (512)**: Modern, optimized model for payments ✅
- **SFace (128/512)**: OpenCV's model
- **VGG-Face (4096)**: Larger, slower
- **FaceNet (128)**: Older model

**Your system uses ArcFace → 512 dimensions**
**Any 1024-dim embeddings are from older model → Incompatible**

### Why In-Memory Cache Exists?

Performance optimization:
- Avoids repeated database reads
- Speeds up face recognition
- SHOULD be cleared after model changes
- SHOULD be cleared after database updates

---

Need help? Check the logs and run the diagnostic scripts above! 🚀