"""
DeepFace Authentication Module
Replaces OpenCV SFace with DeepFace ArcFace for payment-grade face recognition
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

logger = logging.getLogger(__name__)


def configure_gpu_optimization():
    """
    Configure TensorFlow GPU optimization for Quadro K620 (2GB VRAM)
    Falls back to CPU if GPU is insufficient
    """
    try:
        import tensorflow as tf

        # Check if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            if os.getenv("FORCE_CPU", "false").lower() == "true":
                logger.info("💻 FORCE_CPU=true - Using CPU despite GPU availability")
                return False

            try:
                # Optimize GPU memory usage to fit in 2GB VRAM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Set memory limit to 1.8GB (leave 200MB buffer)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=1800
                        )]
                    )

                logger.info(f"✅ GPU Optimized for DeepFace ({len(gpus)} GPU(s) detected)")
                logger.info("   Memory limit: 1.8GB (Quadro K620 compatible)")
                return True

            except RuntimeError as e:
                logger.warning(f"⚠️ GPU optimization failed: {e}")
                logger.info("   Falling back to CPU")
                return False
        else:
            logger.info("💻 No GPU detected. Using CPU (still works well with ArcFace)")
            return False

    except ImportError:
        logger.warning("⚠️ TensorFlow not installed. Running without optimizations")
        return False


class DeepFaceAuthenticator:
    """
    DeepFace-based face authentication using ArcFace model
    Optimized for payment systems with CLAHE preprocessing
    """

    def __init__(self):
        # Configuration from environment variables
        self.model_name = os.getenv("DEEPFACE_MODEL", "ArcFace")
        self.detector_backend = os.getenv("DEEPFACE_DETECTOR", "yunet")
        self.distance_metric = os.getenv("DEEPFACE_DISTANCE_METRIC", "euclidean_l2")
        self.payment_threshold = float(os.getenv("PAYMENT_THRESHOLD", "0.35"))

        # Initialize GPU optimization
        self.gpu_enabled = configure_gpu_optimization()

        # Face storage
        self.known_faces = {}  # user_id -> embedding

        # Model loading state
        self._model_loaded = False

        logger.info(f"🎭 DeepFace Authenticator Initialized")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Detector: {self.detector_backend}")
        logger.info(f"   Distance Metric: {self.distance_metric}")
        logger.info(f"   Payment Threshold: {self.payment_threshold}")
        logger.info(f"   GPU: {'Enabled' if self.gpu_enabled else 'CPU Mode'}")

    def load_model(self):
        """
        Load DeepFace model (lazy initialization)
        Downloads models on first use
        """
        if self._model_loaded:
            return True

        try:
            logger.info("📥 Loading DeepFace models (may download on first run)...")
            from deepface import DeepFace

            # Test to ensure models are loaded
            test_embedding = DeepFace.represent(
                img_path=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )

            self._model_loaded = True
            logger.info(f"✅ DeepFace model loaded: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading DeepFace model: {e}")
            raise Exception(f"Failed to load face recognition model: {str(e)}")

    def ensure_model_loaded(self):
        """Ensure model is loaded before processing"""
        if not self._model_loaded:
            self.load_model()

    def preprocess_image(self, image):
        """
        Apply CLAHE preprocessing for better lighting handling

        Args:
            image: Face image (numpy array or file path)

        Returns:
            Preprocessed image
        """
        try:
            # Load image if it's a path
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"Could not load image from path: {image}")

            # Ensure image is in BGR format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            # Apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE parameters optimized for face recognition
            clahe = cv2.createCLAHE(
                clipLimit=3.0,  # Good balance between contrast and noise
                tileGridSize=(8, 8)  # Localized contrast enhancement
            )

            l_enhanced = clahe.apply(l)

            # Merge back and convert to BGR
            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            return enhanced

        except Exception as e:
            logger.warning(f"⚠️ Error preprocessing image: {e}")
            # Return original image if preprocessing fails
            return image

    def extract_embedding(self, face_image):
        """
        Extract face embedding using DeepFace

        Args:
            face_image: Face image (numpy array or file path)

        Returns:
            Face embedding as numpy array
        """
        try:
            self.ensure_model_loaded()
            from deepface import DeepFace

            # Preprocess image
            preprocessed = self.preprocess_image(face_image)

            # Extract embedding
            embeddings = DeepFace.represent(
                img_path=preprocessed,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                normalization="base"
            )

            if embeddings and len(embeddings) > 0:
                embedding = np.array(embeddings[0]["embedding"])
                logger.debug(f"✅ Embedding extracted: shape={embedding.shape}")
                return embedding
            else:
                raise Exception("No face detected in image")

        except Exception as e:
            logger.error(f"❌ Error extracting embedding: {e}")
            raise Exception(f"Face embedding extraction failed: {str(e)}")

    def compare_faces(self, embedding1, embedding2):
        """
        Compare two face embeddings

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Dictionary with verification results
        """
        try:
            self.ensure_model_loaded()

            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)

            # Calculate distance based on metric
            if self.distance_metric == "euclidean_l2":
                distance = np.linalg.norm(embedding1 - embedding2) / np.linalg.norm([embedding1, embedding2])
            elif self.distance_metric == "euclidean":
                distance = np.linalg.norm(embedding1 - embedding2)
            elif self.distance_metric == "cosine":
                distance = 1 - np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8)
            elif self.distance_metric == "angular":
                distance = np.arccos(np.clip(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8), -1, 1))
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")

            # Determine if match
            is_match = distance <= self.payment_threshold

            # Calculate confidence (percentage)
            if is_match:
                confidence = 100 - (distance / self.payment_threshold) * 100
            else:
                confidence = 0

            return {
                "verified": is_match,
                "distance": float(distance),
                "threshold": self.payment_threshold,
                "confidence": float(max(0, confidence))
            }

        except Exception as e:
            logger.error(f"❌ Error comparing faces: {e}")
            raise Exception(f"Face comparison failed: {str(e)}")

    def register_face(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Register a face embedding for a user

        Args:
            user_id: User identifier
            embedding: Face embedding

        Returns:
            True if successful
        """
        try:
            self.known_faces[user_id] = embedding
            logger.info(f"✅ Face registered for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error registering face: {e}")
            return False

    def verify_face(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Verify a face against registered embedding

        Args:
            user_id: User identifier
            embedding: Face embedding to verify

        Returns:
            True if face matches
        """
        try:
            if user_id not in self.known_faces:
                logger.warning(f"⚠️ User not found: {user_id}")
                return False

            result = self.compare_faces(embedding, self.known_faces[user_id])

            if result["verified"]:
                logger.info(f"✅ Face verified for user: {user_id} (confidence: {result['confidence']:.2f}%)")
            else:
                logger.info(f"❌ Face verification failed for user: {user_id} (distance: {result['distance']:.4f})")

            return result["verified"]

        except Exception as e:
            logger.error(f"❌ Error verifying face: {e}")
            return False

    def get_similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Get confidence score between two embeddings

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Confidence score (0-100)
        """
        try:
            result = self.compare_faces(embedding1, embedding2)
            return result["confidence"]
        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            return 0.0

    def get_model_status(self) -> Dict:
        """
        Get current model status

        Returns:
            Dictionary with model information
        """
        return {
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "detector_backend": self.detector_backend,
            "distance_metric": self.distance_metric,
            "threshold": self.payment_threshold,
            "registered_faces": len(self.known_faces),
            "gpu_enabled": self.gpu_enabled
        }


# Initialize global DeepFace authenticator
deepface_auth = DeepFaceAuthenticator()