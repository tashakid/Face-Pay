"""
Authentication and authorization module using OpenCV SFace
"""

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
import cv2
import numpy as np
import json
from typing import Optional, Dict, List

load_dotenv()

app = FastAPI()
security = HTTPBearer()

# Initialize Firebase Admin
try:
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    else:
        # For development - initialize with default credentials
        firebase_admin.initialize_app()
except Exception as e:
    print(f"Firebase initialization error: {e}")

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

class SFaceAuthenticator:
    def __init__(self):
        self.model = None
        self.model_path = "src/face_recognition_sface_2021dec.onnx"
        self.known_faces = {}  # Store user_id -> embedding mapping
        self.cosine_similarity_threshold = 0.363
    
    def load_model(self):
        """Load the SFace model from ONNX file"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"SFace model file not found: {self.model_path}")
            
            # Load the SFace model using OpenCV's FaceRecognizerSF
            self.model = cv2.FaceRecognizerSF_create(
                self.model_path, 
                config=""
            )
            
            print("SFace model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading SFace model: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load face recognition model: {str(e)}"
            )
    
    def ensure_model_loaded(self):
        """Ensure the model is loaded before processing"""
        if self.model is None:
            self.load_model()
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for SFace model"""
        try:
            self.ensure_model_loaded()
            
            # Ensure image is in the correct format (BGR)
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2BGR)
            
            # Resize to standard face size if needed (SFace expects specific size)
            # SFace typically works with 112x112 faces
            if face_image.shape[0] != 112 or face_image.shape[1] != 112:
                face_image = cv2.resize(face_image, (112, 112))
            
            return face_image
            
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Face preprocessing failed: {str(e)}"
            )
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract biometric embedding from cropped face image"""
        try:
            self.ensure_model_loaded()
            
            # Preprocess the face image
            processed_face = self.preprocess_face(face_image)
            
            # Extract face embedding using SFace
            embedding = self.model.feature(processed_face)
            
            # Ensure embedding is a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Face embedding extraction failed: {str(e)}"
            )
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """Compare two face embeddings using cosine similarity"""
        try:
            self.ensure_model_loaded()
            
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity using SFace's built-in method
            cosine_similarity = self.model.match(embedding1, embedding2, cv2.FaceRecognizerSF_COSINE)
            
            # Return True if similarity is above threshold
            return cosine_similarity > self.cosine_similarity_threshold
            
        except Exception as e:
            print(f"Error comparing faces: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Face comparison failed: {str(e)}"
            )
    
    def register_face(self, user_id: str, embedding: np.ndarray) -> bool:
        """Register a face embedding for a user"""
        try:
            self.known_faces[user_id] = embedding
            return True
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def verify_face(self, user_id: str, embedding: np.ndarray) -> bool:
        """Verify a face against registered embedding"""
        try:
            if user_id not in self.known_faces:
                return False
            
            stored_embedding = self.known_faces[user_id]
            return self.compare_faces(embedding, stored_embedding)
            
        except Exception as e:
            print(f"Error verifying face: {e}")
            return False
    
    def get_similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Get the actual cosine similarity score between two embeddings"""
        try:
            self.ensure_model_loaded()
            
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = self.model.match(embedding1, embedding2, cv2.FaceRecognizerSF_COSINE)
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

# Initialize SFace authenticator
sface_auth = SFaceAuthenticator()

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
        """Authenticate user with Firebase"""
        try:
            # Note: Firebase Admin SDK doesn't provide direct password verification
            # In a real implementation, you would use Firebase REST API for sign-in
            # For now, we'll return a mock response
            return {"id": "mock_user_id", "email": email}
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid credentials")

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
async def register(user: User):
    """Register a new user"""
    try:
        firebase_user = await auth_service.create_user(user.email, user.password)
        
        return UserResponse(
            id=firebase_user.uid,
            email=firebase_user.email,
            created_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login", response_model=LoginResponse)
async def login(user: User):
    """Login user and return access token"""
    try:
        # Authenticate user (mock implementation)
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
    """Get current user information"""
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
    """Register a user's face using SFace"""
    try:
        # Decode base64 image
        import base64
        img_data = base64.b64decode(request.face_image)
        nparr = np.frombuffer(img_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid face image")
        
        # Extract embedding
        embedding = sface_auth.extract_embedding(face_image)
        
        # Register face
        success = sface_auth.register_face(request.user_id, embedding)
        
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
    """Verify a user's face using SFace"""
    try:
        # Decode base64 image
        import base64
        img_data = base64.b64decode(request.face_image)
        nparr = np.frombuffer(img_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid face image")
        
        # Extract embedding
        embedding = sface_auth.extract_embedding(face_image)
        
        # Verify face
        is_match = sface_auth.verify_face(request.user_id, embedding)
        
        # Get similarity score for debugging
        if request.user_id in sface_auth.known_faces:
            similarity = sface_auth.get_similarity_score(embedding, sface_auth.known_faces[request.user_id])
        else:
            similarity = 0.0
        
        return {
            "success": True,
            "verified": is_match,
            "similarity_score": similarity,
            "threshold": sface_auth.cosine_similarity_threshold,
            "message": "Face verified successfully" if is_match else "Face verification failed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-faces")
async def compare_faces_endpoint(face_image1: str = File(...), face_image2: str = File(...)):
    """Compare two face images and return similarity score"""
    try:
        import base64
        
        # Decode first face image
        img_data1 = base64.b64decode(face_image1)
        nparr1 = np.frombuffer(img_data1, np.uint8)
        face_image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        
        # Decode second face image
        img_data2 = base64.b64decode(face_image2)
        nparr2 = np.frombuffer(img_data2, np.uint8)
        face_image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        
        if face_image1 is None or face_image2 is None:
            raise HTTPException(status_code=400, detail="Invalid face images")
        
        # Extract embeddings
        embedding1 = sface_auth.extract_embedding(face_image1)
        embedding2 = sface_auth.extract_embedding(face_image2)
        
        # Compare faces
        is_match = sface_auth.compare_faces(embedding1, embedding2)
        similarity = sface_auth.get_similarity_score(embedding1, embedding2)
        
        return {
            "success": True,
            "match": is_match,
            "similarity_score": similarity,
            "threshold": sface_auth.cosine_similarity_threshold,
            "message": "Faces match" if is_match else "Faces do not match"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load-model")
async def load_model_endpoint():
    """Load the SFace model"""
    try:
        success = sface_auth.load_model()
        return {
            "success": success,
            "message": "SFace model loaded successfully" if success else "Failed to load SFace model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
async def model_status():
    """Check if SFace model is loaded"""
    return {
        "model_loaded": sface_auth.model is not None,
        "model_path": sface_auth.model_path,
        "threshold": sface_auth.cosine_similarity_threshold,
        "registered_faces": len(sface_auth.known_faces)
    }

router = app