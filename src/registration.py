"""
User Registration and Face Enrollment APIs
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List
import json
import base64
import cv2
import numpy as np
import logging
from database import db
from auth import auth_service, get_current_user

logger = logging.getLogger(__name__)

app = APIRouter()

class UserFaceRegistrationRequest(BaseModel):
    email: str
    password: str
    name: str
    phone_number: str

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    phone_number: Optional[str] = None

@app.post("/register-with-face")
async def register_user_with_face(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    phone_number: str = Form(...),
    face_image: List[UploadFile] = Form(...),
    use_multi_sample: bool = Form(False)
):
    """Register a user with their face in one operation

    Args:
        use_multi_sample: If True, register using multi-sample capture (recommended)
        face_image: List of face image files for multi-sample enrollment
    """
    try:
        user_data = {
            "email": email,
            "name": name,
            "phone_number": phone_number,
            "password": password,
            "face_registered": False,
            "enrollment_type": "multi_sample" if use_multi_sample else "single_sample"
        }

        custom_id = name.lower().replace(" ", "_")
        result = db.create_user(user_data, custom_id=custom_id)
        db_user_id = result.get("user_id")

        from deepface_auth import deepface_auth
        deepface_auth.load_model()

        embeddings_saved = 0

        if use_multi_sample:
            previous_embedding = None
            for idx, img_file in enumerate(face_image, 1):
                try:
                    contents = await img_file.read()
                    nparr = np.frombuffer(contents, np.uint8)
                    face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if face_img is None:
                        logger.warning(f"Warning: Failed to decode image {img_file.filename}")
                        continue

                    try:
                        from diagnostics import log_image_quality, format_score
                        log_image_quality(face_img, level=logging.DEBUG)

                        embedding = deepface_auth.extract_embedding(face_img)

                        if previous_embedding is not None:
                            similarity = db.cosine_similarity(embedding.flatten(), previous_embedding.flatten())
                            if similarity < 0.3:
                                logger.warning(f"⚠️  Sample {idx} shows low similarity ({format_score(similarity)}) to previous sample")
                                logger.warning(f"   This may indicate poor lighting, different angle, or blurry image")
                            else:
                                logger.debug(f"   Similarity to previous sample: {format_score(similarity)}")
                    except ImportError:
                        embedding = deepface_auth.extract_embedding(face_img)

                    if embedding is not None:
                        db.save_face_embedding(db_user_id, embedding)
                        previous_embedding = embedding
                        embeddings_saved += 1
                        logger.info(f"📸 Sample {idx}/{len(face_image)} captured for user: {name}")
                except Exception as e:
                    logger.error(f"Error processing image {img_file.filename}: {e}")
                    continue

            db.update_user(db_user_id, {
                "face_registered": True,
                "total_embeddings": embeddings_saved
            })

            logger.info(f"✅ User {name} registered with {embeddings_saved} face samples")

            if embeddings_saved == 0:
                raise HTTPException(status_code=400, detail="No valid face images were processed")

            print(f"User {name} registered with {embeddings_saved} face samples")

            return {
                "success": True,
                "message": f"User registered with {embeddings_saved} face samples",
                "user_id": db_user_id,
                "email": email,
                "name": name,
                "samples_captured": embeddings_saved
            }
        else:
            contents = await face_image[0].read()
            nparr = np.frombuffer(contents, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if face_img is None:
                raise HTTPException(status_code=400, detail="Invalid face image format")

            embedding = deepface_auth.extract_embedding(face_img)

            db.save_face_embedding(db_user_id, embedding)

            db.update_user(db_user_id, {"face_registered": True})

            return {
                "success": True,
                "message": "User registered successfully with face",
                "user_id": db_user_id,
                "email": email,
                "name": name
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/register-face-only")
async def register_face_only(
    current_user,
    face_image: UploadFile = File(...)
):
    """
    Register or update face for an existing user
    """
    try:
        user_data = db.get_user_by_email(current_user.email)

        if not user_data:
            raise HTTPException(status_code=404, detail="User not found in database")

        user_id = user_data.get("id")

        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None:
            raise HTTPException(status_code=400, detail="Invalid face image format")

        from deepface_auth import deepface_auth
        deepface_auth.load_model()
        embedding = deepface_auth.extract_embedding(face_img)

        existing_face = db.get_face_embedding(user_id)
        if existing_face:
            embedding_list = embedding.flatten().tolist()
            db.update_user(user_id, {
                "face_registered": True,
                "face_id": existing_face.get("id")
            })
        else:
            db.save_face_embedding(user_id, contents, embedding)

        return {
            "success": True,
            "message": "Face registered successfully",
            "user_id": user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face registration failed: {str(e)}")


@app.get("/verify-face")
async def verify_user_face(
    current_user,
    face_image: UploadFile = File(...)
):
    """
    Verify if uploaded face matches the registered face
    """
    try:
        user_data = db.get_user_by_email(current_user.email)

        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = user_data.get("id")

        if not user_data.get("face_registered", False):
            raise HTTPException(status_code=400, detail="No face registered for this user")

        registered_face_data = db.get_face_embedding(user_id)
        if not registered_face_data:
            raise HTTPException(status_code=404, detail="No face data found")

        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None:
            raise HTTPException(status_code=400, detail="Invalid face image format")

        from deepface_auth import deepface_auth
        deepface_auth.load_model()
        uploaded_embedding = deepface_auth.extract_embedding(face_img)
        registered_embedding = registered_face_data.get("embedding")

        similarity = db.cosine_similarity(uploaded_embedding.flatten(), registered_embedding)

        threshold = 0.363
        is_match = similarity < threshold

        return {
            "success": True,
            "verified": is_match,
            "similarity_score": float(similarity),
            "threshold": threshold,
            "message": "Face verified successfully" if is_match else "Face verification failed"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")


@app.put("/update-user/{user_id}")
async def update_user_information(user_id: str, update_data: UserUpdateRequest):
    """
    Update user information
    """
    try:
        user_data = {}
        if update_data.name:
            user_data['name'] = update_data.name
        if update_data.phone_number:
            user_data['phone_number'] = update_data.phone_number

        if not user_data:
            raise HTTPException(status_code=400, detail="No data provided for update")

        success = db.update_user(user_id, user_data)

        if success:
            return {"success": True, "message": "User updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update user")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.get("/users")
async def get_all_users():
    """Get all registered users"""
    try:
        users = db.get_all_users()
        return {
            "success": True,
            "count": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")


router = app