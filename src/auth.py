"""
Authentication and authorization module using DeepFace ArcFace
Replaces OpenCV SFace with payment-grade face recognition
"""

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
import cv2
import numpy as np
import json
from typing import Optional, Dict, List
from database import db
from deepface_auth import deepface_auth

load_dotenv()

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "")

app = FastAPI()
security = HTTPBearer()

# Initialize Firebase Admin
if not firebase_admin._apps:
    try:
        cred_path = os.getenv("FIREBASE_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"Firebase initialization error in auth.py: {e}")

class User(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class FaceRegistrationRequest(BaseModel):
    user_id: str
    face_image: str  # base64 encoded image

class FaceVerificationRequest(BaseModel):
    user_id: str
    face_image: str  # base64 encoded image



class AuthService:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "your-secret-key-here")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def create_user(self, email: str, password: str):
        """Create a new user in Firebase"""
        try:
            user = auth.create_user(
                email=email,
                password=password
            )
            return user
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def authenticate_user(self, email: str, password: str):
        try:
            if not FIREBASE_API_KEY:
                raise HTTPException(
                    status_code=500,
                    detail="Firebase API key not configured. Please set FIREBASE_API_KEY in .env file."
                )

            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": data.get("localId"),
                    "email": data.get("email"),
                    "id_token": data.get("idToken"),
                    "refresh_token": data.get("refreshToken")
                }
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Authentication failed")
                raise HTTPException(status_code=401, detail=f"Authentication failed: {error_message}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")

auth_service = AuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = auth.get_user(user_id)
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/register", response_model=UserResponse)
async def register(user: User, name: str = None, phone_number: str = None):
    try:
        firebase_user = auth.create_user(
            email=user.email,
            password=user.password
        )

        user_data = {
            "email": firebase_user.email,
            "name": name or "",
            "phone_number": phone_number or "",
            "face_registered": False
        }

        db_user_id = db.create_user(user_data)

        return UserResponse(
            id=db_user_id,
            email=firebase_user.email,
            created_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login", response_model=LoginResponse)
async def login(user: User):
    try:
        user_data = await auth_service.authenticate_user(user.email, user.password)
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": user_data["id"], "email": user_data["email"]}
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user_data["id"],
                email=user_data["email"],
                created_at=datetime.now()
            )
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.get("/me", response_model=UserResponse)
async def get_me(current_user = Depends(get_current_user)):
    user_data = db.get_user_by_email(current_user.email)
    if user_data:
        return {
            "id": user_data.get("id"),
            "email": current_user.email,
            "name": user_data.get("name", ""),
            "phone_number": user_data.get("phone_number", ""),
            "face_registered": user_data.get("face_registered", False),
            "created_at": user_data.get("created_at")
        }
    else:
        return UserResponse(
            id=current_user.uid,
            email=current_user.email,
            created_at=current_user.user_metadata.creation_timestamp
        )

@app.post("/logout")
async def logout():
    """Logout user (client-side token removal)"""
    return {"message": "Successfully logged out"}

# Face Authentication Endpoints

@app.post("/register-face")
async def register_face_endpoint(request: FaceRegistrationRequest):
    """Register a user's face using DeepFace ArcFace"""
    try:
        import base64
        img_data = base64.b64decode(request.face_image)
        nparr = np.frombuffer(img_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid face image")

        embedding = deepface_auth.extract_embedding(face_image)
        success = deepface_auth.register_face(request.user_id, embedding)

        if success:
            return {
                "success": True,
                "message": f"Face registered successfully for user {request.user_id}",
                "embedding_shape": embedding.shape
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register face")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-face")
async def verify_face_endpoint(request: FaceVerificationRequest):
    """Verify a user's face using DeepFace ArcFace"""
    try:
        import base64
        img_data = base64.b64decode(request.face_image)
        nparr = np.frombuffer(img_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid face image")

        embedding = deepface_auth.extract_embedding(face_image)
        is_match = deepface_auth.verify_face(request.user_id, embedding)

        if request.user_id in deepface_auth.known_faces:
            confidence = deepface_auth.get_similarity_score(embedding, deepface_auth.known_faces[request.user_id])
        else:
            confidence = 0.0

        return {
            "success": True,
            "verified": is_match,
            "confidence": confidence,
            "threshold": deepface_auth.payment_threshold,
            "message": "Face verified successfully" if is_match else "Face verification failed"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-faces")
async def compare_faces_endpoint(face_image1: str = File(...), face_image2: str = File(...)):
    """Compare two face images and return confidence score"""
    try:
        import base64

        img_data1 = base64.b64decode(face_image1)
        nparr1 = np.frombuffer(img_data1, np.uint8)
        face_image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        img_data2 = base64.b64decode(face_image2)
        nparr2 = np.frombuffer(img_data2, np.uint8)
        face_image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        if face_image1 is None or face_image2 is None:
            raise HTTPException(status_code=400, detail="Invalid face images")

        embedding1 = deepface_auth.extract_embedding(face_image1)
        embedding2 = deepface_auth.extract_embedding(face_image2)

        result = deepface_auth.compare_faces(embedding1, embedding2)

        return {
            "success": True,
            "match": result["verified"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "message": "Faces match" if result["verified"] else "Faces do not match"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load-model")
async def load_model_endpoint():
    """Load the DeepFace model"""
    try:
        success = deepface_auth.load_model()
        return {
            "success": success,
            "message": "DeepFace model loaded successfully" if success else "Failed to load DeepFace model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
async def model_status():
    """Check DeepFace model status"""
    return deepface_auth.get_model_status()

@app.post("/clear-cache")
async def clear_cache():
    """Clear all in-memory face caches"""
    try:
        if hasattr(deepface_auth, 'known_faces'):
            cleared_count = len(deepface_auth.known_faces)
            deepface_auth.known_faces.clear()

            return {
                "success": True,
                "message": f"Cleared {cleared_count} faces from in-memory cache",
                "cleared_count": cleared_count
            }
        else:
            return {
                "success": True,
                "message": "No cache to clear",
                "cleared_count": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

router = app