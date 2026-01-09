"""
Firebase Firestore Database Module for Face-Pay
Handles user data, face embeddings, and transactions
"""

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from google.api_core import exceptions as gcp_exceptions
import numpy as np
import base64
import json
import os
import logging
import time

logger = logging.getLogger(__name__)


class Database:
    """Singleton database wrapper for Firebase Firestore operations"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_firebase()
            Database._initialized = True

    def _initialize_firebase(self):
        try:
            creds_path = os.getenv("FIREBASE_CREDENTIALS")

            if not creds_path:
                raise ValueError(
                    "FIREBASE_CREDENTIALS environment variable not set"
                )

            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Firebase credentials file not found: {creds_path}"
                )

            cred_block = credentials.Certificate(json.load(open(creds_path)))

            if not firebase_admin._apps:
                app = firebase_admin.initialize_app(cred_block)
            else:
                app = firebase_admin.get_app()

            self.db = firestore.client(app)

            print(f"✅ Firebase initialized successfully")

        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            raise

    def serialize_array(self, array: np.ndarray) -> str:
        """Serialize numpy array to base64 string for storage"""
        array_float32 = array.astype(np.float32)
        array_bytes = array_float32.tobytes()
        base64_bytes = base64.b64encode(array_bytes)
        return base64_bytes.decode("utf-8")

    def deserialize_array(self, array_str: str, expected_dim: int = None) -> np.ndarray:
        """Deserialize base64 string back to numpy array with dimension validation"""
        base64_bytes = array_str.encode("utf-8")
        array_bytes = base64.b64decode(base64_bytes)
        array = np.frombuffer(array_bytes, dtype=np.float32)

        actual_dim = len(array)

        if expected_dim is not None and actual_dim != expected_dim:
            logger.warning(
                f"⚠️  Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
            logger.warning("   Using actual dimensions for comparison")

        return array

    def create_user(self, user_data: dict, custom_id: str = None) -> dict:
        """Create a new user in Firestore

        Args:
            user_data: Dictionary with user information
                      (name, email, password, phone_number, etc.)
            custom_id: Optional custom user ID (default uses Firestore auto-generated ID)

        Returns:
            dict: Created user data with generated ID
        """
        try:
            from datetime import datetime

            user_data["created_at"] = datetime.now()

            if custom_id:
                doc_ref = self.db.collection("users").document(custom_id)
                doc_ref.set(user_data)
                user_id = custom_id
            else:
                doc_ref = self.db.collection("users").add(user_data)
                user_id = doc_ref[1].id

            user_data["user_id"] = user_id

            print(f"✅ User created: {user_id}")
            return {"success": True, "user_id": user_id, "data": user_data}

        except gcp_exceptions.GoogleAPIError as e:
            print(f"❌ Firebase error creating user: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return {"success": False, "error": str(e)}

    def get_user(self, user_id: str) -> dict:
        """Get user by ID

        Args:
            user_id: Firebase document ID

        Returns:
            dict: User data or None if not found
        """
        try:
            doc_ref = self.db.collection("users").document(user_id)
            doc = doc_ref.get()

            if doc.exists:
                user_data = doc.to_dict()
                user_data["user_id"] = doc.id
                return user_data

            print(f"❌ User not found: {user_id}")
            return None

        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None

    def get_user_by_email(self, email: str) -> dict:
        """Get user by email address

        Args:
            email: User's email address

        Returns:
            dict: User data or None if not found
        """
        try:
            users = (
                self.db.collection("users")
                .where("email", "==", email)
                .limit(1)
                .get()
            )

            if users:
                user_data = users[0].to_dict()
                user_data["user_id"] = users[0].id
                user_data["id"] = users[0].id
                return user_data

            return None

        except Exception as e:
            print(f"❌ Error getting user by email: {e}")
            return None

    def get_all_users(self) -> list:
        """Get all users from Firestore

        Returns:
            list: List of all users with their data
        """
        try:
            users_ref = self.db.collection("users").get()
            users_list = []

            for doc in users_ref:
                user_data = doc.to_dict()
                user_data["user_id"] = doc.id
                users_list.append(user_data)

            print(f"✅ Retrieved {len(users_list)} users")
            return users_list

        except Exception as e:
            print(f"❌ Error getting all users: {e}")
            return []

    def update_user(self, user_id: str, update_data: dict) -> dict:
        """Update user information

        Args:
            user_id: Firebase document ID
            update_data: Dictionary with fields to update

        Returns:
            dict: Updated user data
        """
        try:
            doc_ref = self.db.collection("users").document(user_id)
            doc_ref.update(update_data)

            updated = self.get_user(user_id)
            return {"success": True, "data": updated}

        except gcp_exceptions.NotFound:
            return {"success": False, "error": "User not found"}
        except Exception as e:
            print(f"❌ Error updating user: {e}")
            return {"success": False, "error": str(e)}

    def delete_user(self, user_id: str) -> dict:
        """Delete a user from Firestore

        Args:
            user_id: Firebase document ID

        Returns:
            dict: Success/failure status
        """
        try:
            self.db.collection("users").document(user_id).delete()

            print(f"✅ User deleted: {user_id}")
            return {"success": True}

        except gcp_exceptions.NotFound:
            return {"success": False, "error": "User not found"}
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            return {"success": False, "error": str(e)}

    def save_face_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        face_image: str = None
    ) -> dict:
        """Save face embedding for a user

        Args:
            user_id: Firebase document ID
            embedding: NumPy array of face embedding
            face_image: Optional base64 encoded face image

        Returns:
            dict: Success/failure status
        """
        try:
            embedding_str = self.serialize_array(embedding)

            face_data = {
                "user_id": user_id,
                "embedding": embedding_str,
                "created_at": firestore.SERVER_TIMESTAMP
            }

            if face_image:
                face_data["face_image"] = face_image

            doc_ref = self.db.collection("faces").add(face_data)
            face_id = doc_ref[1].id

            print(f"✅ Face embedding saved: {face_id} for user {user_id}")
            return {"success": True, "face_id": face_id}

        except Exception as e:
            print(f"❌ Error saving face embedding: {e}")
            return {"success": False, "error": str(e)}

    def get_face_embedding(self, user_id: str) -> dict:
        """Get the most recent face embedding for a user

        Args:
            user_id: Firebase document ID

        Returns:
            dict: Face data with embedding or None
        """
        try:
            faces = (
                self.db.collection("faces")
                .where("user_id", "==", user_id)
                .order_by("created_at", direction=firestore.DESCENDING)
                .limit(1)
                .get()
            )

            if faces:
                face_data = faces[0].to_dict()
                embedding_str = face_data.get("embedding")
                if embedding_str:
                    face_data["embedding"] = self.deserialize_array(embedding_str, expected_dim=None)
                    face_data["id"] = faces[0].id
                return face_data

            return None

        except Exception as e:
            print(f"❌ Error getting face embedding: {e}")
            return None

    def get_user_faces(self, user_id: str) -> list:
        """Get all face embeddings for a user

        Args:
            user_id: Firebase document ID

        Returns:
            list: List of face documents
        """
        try:
            faces = (
                self.db.collection("faces").where(
                    "user_id", "==", user_id).get()
            )

            return [face.to_dict() for face in faces]

        except Exception as e:
            print(f"❌ Error getting user faces: {e}")
            return []

    def find_matching_face(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.3
    ) -> dict:
        """Find the best matching face in the database

        Args:
            query_embedding: NumPy array of query face embedding
            threshold: Minimum similarity score to consider a match

        Returns:
            dict: Matching face data or None if no match found
        """
        try:
            faces = self.db.collection("faces").get()

            best_match = None
            best_score = 0

            query_embedding = query_embedding.reshape(-1)

            for face_doc in faces:
                face_data = face_doc.to_dict()
                stored_embedding_str = face_data.get("embedding")

                if stored_embedding_str:
                    stored_embedding = self.deserialize_array(
                        stored_embedding_str
                    ).reshape(-1)

                    score = self.cosine_similarity(query_embedding, stored_embedding)

                    if score > best_score:
                        best_score = score
                        face_data["score"] = score
                        face_data["face_id"] = face_doc.id
                        best_match = face_data

            if best_match and best_score >= threshold:
                print(
                    f"✅ Face matched with score: {best_score:.4f} (threshold: {threshold})"
                )
                return best_match

            print(f"❌ No face matched. Best score: {best_score:.4f} (threshold: {threshold})")
            return None

        except Exception as e:
            print(f"❌ Error finding matching face: {e}")
            return None

    def find_matching_face_with_averaging(
        self,
        query_embedding: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> dict:
        """Find the best matching face by averaging all embeddings per user.
        This provides better recognition accuracy by using all enrolled samples.

        Args:
            query_embedding: NumPy array of query face embedding
            confidence_threshold: Minimum confidence (0.0-1.0) to consider a match. Default 0.8 (80%)

        Returns:
            dict: Matching face data including user_id, confidence_score, sample_count or None
        """
        try:
            import diagnostics
            from diagnostics import calculate_score_distribution, format_score

            query_start = time.time()

            faces = self.db.collection("faces").get()

            # Group embeddings by user_id
            user_embeddings = {}
            for face_doc in faces:
                face_data = face_doc.to_dict()
                user_id = face_data.get("user_id")
                stored_embedding_str = face_data.get("embedding")

                if user_id and stored_embedding_str:
                    stored_embedding = self.deserialize_array(stored_embedding_str).reshape(-1)
                    if user_id not in user_embeddings:
                        user_embeddings[user_id] = []
                    user_embeddings[user_id].append(stored_embedding)

            if not user_embeddings:
                logger.warning("❌ No face embeddings found in database")
                return None

            query_embedding = query_embedding.reshape(-1)
            best_match = None
            best_confidence = 0

            query_time = (time.time() - query_start) * 1000
            logger.info(f"   Database query: {query_time:.0f}ms")

            # Calculate average similarity for each user
            logger.debug(f"\n🔍 Scanning {len(user_embeddings)} registered users...")

            for user_id, embeddings in user_embeddings.items():
                similarities = []

                # Calculate similarity for each sample
                for idx, embedding in enumerate(embeddings):
                    score = self.cosine_similarity(query_embedding, embedding)
                    similarities.append(score)

                # Get statistics
                stats = calculate_score_distribution(similarities)
                max_idx = int(np.argmax(similarities))
                max_similarity = similarities[max_idx]

                # Detailed logging for this user
                logger.debug(f"\n👤 User: {user_id} ({len(embeddings)} samples)")

                # Log all scores
                for idx, score in enumerate(similarities):
                    marker = ""
                    if score == max_similarity:
                        marker = " ⭐ BEST"
                    elif score > 0.5:
                        marker = " ⚠️  HIGH"
                    elif score > 0.3:
                        marker = " ✓ Elevated"

                    logger.debug(f"   Sample {idx+1:2d}: {format_score(score)}{marker}")

                # Log statistics
                logger.debug(f"\n   Statistics:")
                logger.debug(f"      Best match: Sample {max_idx+1} with {format_score(max_similarity)}")
                logger.debug(f"      Average: {format_score(stats['mean'])}")
                logger.debug(f"      Median: {format_score(stats['median'])}")
                logger.debug(f"      Std dev: {format_score(stats['std'])}")
                logger.debug(f"      Min/Max: {format_score(stats['min'])} / {format_score(stats['max'])}")
                logger.debug(f"      Positive scores: {stats['positive_count']}/{stats['count']} ({stats['positive_count']/stats['count']*100:.0f}%)")

                # Log distribution breakdown
                above_50 = sum(1 for s in similarities if s > 0.5)
                above_30 = sum(1 for s in similarities if s > 0.3)
                above_20 = sum(1 for s in similarities if s > 0.2)
                above_0 = stats['positive_count']

                logger.debug(f"      Score distribution:")
                logger.debug(f"         > 50%: {above_50}/{stats['count']} ({above_50/stats['count']*100:.0f}%)")
                logger.debug(f"         > 30%: {above_30}/{stats['count']} ({above_30/stats['count']*100:.0f}%)")
                logger.debug(f"         > 20%: {above_20}/{stats['count']} ({above_20/stats['count']*100:.0f}%)")
                logger.debug(f"         >  0%: {above_0}/{stats['count']} ({above_0/stats['count']*100:.0f}%)")

                if max_similarity > best_confidence:
                    best_confidence = max_similarity
                    best_match = {
                        "user_id": user_id,
                        "confidence": max_similarity,
                        "avg_confidence": stats['mean'],
                        "median_confidence": stats['median'],
                        "std_confidence": stats['std'],
                        "sample_count": len(embeddings),
                        "all_scores": similarities,
                        "best_sample_index": max_idx,
                        "statistics": stats
                    }

            total_time = (time.time() - query_start) * 1000
            logger.info(f"   Total matching time: {total_time:.0f}ms")

            # Final decision
            logger.info(f"\n{'='*60}")
            logger.info(f"FINAL RECOGNITION DECISION:")

            if best_match and best_confidence >= confidence_threshold:
                # Convert confidence to percentage display (0.0-1.0 -> 0-100)
                confidence_percent = best_confidence * 100
                logger.info(f"✅ Face matched: {best_match['user_id']}")
                logger.info(f"   Best similarity: {format_score(best_match['confidence'])}")
                logger.info(f"   Best sample: #{best_match['best_sample_index'] + 1}")
                logger.info(f"   Avg similarity: {format_score(best_match['avg_confidence'])}")
                logger.info(f"   Median similarity: {format_score(best_match['median_confidence'])}")
                logger.info(f"   Sample count: {best_match['sample_count']}")
                logger.info(f"   Threshold: {format_score(confidence_threshold)}")
                logger.info(f"   Gap: +{format_score(best_confidence - confidence_threshold)}")
                logger.info(f"{'='*60}\n")
                return best_match

            logger.info(f"❌ No face matched above threshold")
            if best_match:
                gap = confidence_threshold - best_confidence
                logger.info(f"   Best match: {best_match['user_id']} with {format_score(best_match['confidence'])}")
                logger.info(f"   Best sample: #{best_match['best_sample_index'] + 1}")
                logger.info(f"   Threshold required: {format_score(confidence_threshold)}")
                logger.info(f"   Gap: {format_score(gap)}")
                logger.info(f"   Performance insight: {gap * 100:.1f}% improvement needed")
                logger.info(f"{'='*60}\n")
            else:
                logger.info(f"   No viable match found")
                logger.info(f"{'='*60}\n")

            return None

        except Exception as e:
            logger.error(f"❌ Error finding matching face with averaging: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def create_transaction(self, transaction_data: dict) -> dict:
        """Create a new transaction record

        Args:
            transaction_data: Transaction details
                             (amount, phone_number, status, etc.)

        Returns:
            dict: Created transaction data with ID
        """
        try:
            doc_ref = self.db.collection("transactions").add(transaction_data)
            transaction_id = doc_ref[1].id

            transaction_data["transaction_id"] = transaction_id

            print(
                f"✅ Transaction created: {transaction_id}"
            )
            return {"success": True, "transaction_id": transaction_id, "data": transaction_data}

        except Exception as e:
            print(f"❌ Error creating transaction: {e}")
            return {"success": False, "error": str(e)}

    def get_transaction(self, transaction_id: str) -> dict:
        """Get transaction by ID

        Args:
            transaction_id: Firebase document ID

        Returns:
            dict: Transaction data or None if not found
        """
        try:
            doc_ref = self.db.collection("transactions").document(transaction_id)
            doc = doc_ref.get()

            if doc.exists:
                transaction_data = doc.to_dict()
                transaction_data["transaction_id"] = doc.id
                return transaction_data

            print(f"❌ Transaction not found: {transaction_id}")
            return None

        except Exception as e:
            print(f"❌ Error getting transaction: {e}")
            return None

    def update_transaction(
        self,
        transaction_id: str,
        update_data: dict
    ) -> dict:
        """Update transaction status

        Args:
            transaction_id: Firebase document ID or checkout_request_id
            update_data: Dictionary with fields to update

        Returns:
            dict: Success/failure status
        """
        try:
            transactions = (
                self.db.collection("transactions")
                .where("checkout_request_id", "==", transaction_id)
                .limit(1)
                .get()
            )

            doc_ref = None
            if transactions:
                doc_ref = transactions[0].reference
            else:
                doc_ref = self.db.collection("transactions").document(transaction_id)

            doc_ref.update(update_data)

            print(f"✅ Transaction updated: {transaction_id}")
            return {"success": True}

        except gcp_exceptions.NotFound:
            return {"success": False, "error": "Transaction not found"}
        except Exception as e:
            print(f"❌ Error updating transaction: {e}")
            return {"success": False, "error": str(e)}

    def get_user_transactions(self, user_id: str) -> list:
        """Get all transactions for a user

        Args:
            user_id: User's Firebase document ID

        Returns:
            list: List of transaction documents
        """
        try:
            transactions = (
                self.db.collection("transactions").where(
                    "user_id", "==", user_id
                ).get()
            )

            result = [tx.to_dict() for tx in transactions]
            result.sort(key=lambda x: x.get("created_at", 0), reverse=True)

            return result

        except Exception as e:
            print(f"❌ Error getting user transactions: {e}")
            return []

    def get_all_transactions(self) -> list:
        try:
            transactions = self.db.collection("transactions").get()
            result = [tx.to_dict() for tx in transactions]
            result.sort(key=lambda x: (x.get("created_at", 0), x.get("user_id", "")))
            return result
        except Exception as e:
            print(f"❌ Error getting all transactions: {e}")
            return []

    def get_transaction_by_reference(
        self,
        reference: str
    ) -> dict:
        """Get transaction by M-Pesa reference (merchant_request_id or checkout_request_id)

        Args:
            reference: M-Pesa transaction reference

        Returns:
            dict: Transaction data or None if not found
        """
        try:
            transactions = (
                self.db.collection("transactions")
                .where("merchant_request_id", "==", reference)
                .limit(1)
                .get()
            )

            if transactions:
                tx_data = transactions[0].to_dict()
                tx_data["transaction_id"] = transactions[0].id
                return tx_data

            return None

        except Exception as e:
            print(f"❌ Error getting transaction by reference: {e}")
            return None

    def health_check(self) -> dict:
        """Check database connection health

        Returns:
            dict: Health status
        """
        try:
            self.db.collection("_health").limit(1).get()
            return {"status": "healthy", "database": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


db = Database()