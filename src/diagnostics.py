"""
Diagnostic utilities for face recognition system.
Provides image quality analysis, statistical calculations, and logging helpers.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculate image brightness as percentage (0-100).

    Args:
        image: OpenCV image (BGR format)

    Returns:
        Brightness as percentage (0-100)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate mean brightness (0-255)
    mean_brightness = np.mean(gray)
    # Convert to percentage
    return (mean_brightness / 255.0) * 100.0


def detect_blur(image: np.ndarray) -> Dict[str, float]:
    """
    Detect image blur using variance of Laplacian.

    Args:
        image: OpenCV image (BGR format)

    Returns:
        Dictionary with blur metrics:
        - variance: Laplacian variance score (higher = sharper)
        - is_blurry: Boolean indicating if image is too blurry
        - quality: String description (good/fair/poor)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Thresholds based on typical values:
    # < 100: too blurry
    # 100-200: fair quality
    # > 200: good quality
    if variance < 100:
        quality = "poor"
        is_blurry = True
    elif variance < 200:
        quality = "fair"
        is_blurry = False
    else:
        quality = "good"
        is_blurry = False

    return {
        "variance": float(variance),
        "is_blurry": is_blurry,
        "quality": quality
    }


def get_face_statistics(faces: List, image_width: int, image_height: int) -> Dict:
    """
    Get statistics about detected faces.

    Args:
        faces: List of face detections (MediaPipe format)
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Dictionary with face statistics
    """
    if not faces:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "avg_size": (0, 0),
            "positions": []
        }

    confidences = []
    sizes = []
    positions = []

    for face in faces:
        confidences.append(face.get("confidence", 0))
        positions.append(face.get("bbox", [0, 0, 0, 0]))

        bbox = face.get("bbox", [0, 0, 0, 0])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        sizes.append((width, height))

    avg_width = np.mean([s[0] for s in sizes]) if sizes else 0
    avg_height = np.mean([s[1] for s in sizes]) if sizes else 0

    return {
        "count": len(faces),
        "avg_confidence": float(np.mean(confidences)),
        "avg_size": (float(avg_width), float(avg_height)),
        "positions": positions
    }


def analyze_embedding_quality(embedding: np.ndarray) -> Dict:
    """
    Analyze face embedding for quality metrics.

    Args:
        embedding: Face embedding vector

    Returns:
        Dictionary with embedding quality metrics
    """
    norm = np.linalg.norm(embedding)
    mean_value = np.mean(embedding)

    # Check for potential issues
    warnings = []
    if norm == 0:
        warnings.append("Zero norm - invalid embedding")
    if np.isnan(embedding).any():
        warnings.append("Contains NaN values")
    if np.isinf(embedding).any():
        warnings.append("Contains infinite values")

    return {
        "norm": float(norm),
        "dimension": len(embedding),
        "mean": float(mean_value),
        "std": float(np.std(embedding)),
        "warnings": warnings
    }


def calculate_score_distribution(scores: List[float]) -> Dict:
    """
    Calculate statistical distribution of similarity scores.

    Args:
        scores: List of similarity scores

    Returns:
        Dictionary with statistical measures
    """
    if not scores:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "percentile_25": 0.0,
            "percentile_75": 0.0,
            "positive_count": 0,
            "negative_count": 0
        }

    scores_arr = np.array(scores)

    return {
        "count": len(scores),
        "min": float(np.min(scores_arr)),
        "max": float(np.max(scores_arr)),
        "mean": float(np.mean(scores_arr)),
        "median": float(np.median(scores_arr)),
        "std": float(np.std(scores_arr)),
        "percentile_25": float(np.percentile(scores_arr, 25)),
        "percentile_75": float(np.percentile(scores_arr, 75)),
        "positive_count": int(np.sum(scores_arr > 0)),
        "negative_count": int(np.sum(scores_arr < 0))
    }


def log_image_quality(image: np.ndarray, level: int = logging.DEBUG) -> None:
    """
    Log image quality metrics.

    Args:
        image: OpenCV image (BGR format)
        level: Logging level (default: DEBUG)
    """
    try:
        brightness = calculate_brightness(image)
        blur = detect_blur(image)

        logger.log(level, "🖼️  Image Quality Analysis:")
        logger.log(level, f"   Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
        logger.log(level, f"   Brightness: {brightness:.1f}% (optimal: 40-60%)")
        logger.log(level, f"   Blur score: {blur['variance']:.1f} (quality: {blur['quality']})")

        if blur['is_blurry']:
            logger.log(level, "   ⚠️  Warning: Image appears blurry")
        if brightness < 30:
            logger.log(level, "   ⚠️  Warning: Image too dark")
        elif brightness > 70:
            logger.log(level, "   ⚠️  Warning: Image too bright")

    except Exception as e:
        logger.error(f"Failed to analyze image quality: {e}")


def format_score(score: float, show_percentage: bool = True) -> str:
    """
    Format a similarity score for logging.

    Args:
        score: Similarity score
        show_percentage: Whether to show percentage

    Returns:
        Formatted string
    """
    if show_percentage:
        percentage = score * 100
        return f"{score:.4f} ({percentage:.1f}%)"
    else:
        return f"{score:.4f}"


def highlight_best_scores(scores: List[float], max_count: int = 3) -> List[int]:
    """
    Get indices of best (highest) scores.

    Args:
        scores: List of scores
        max_count: Maximum number of indices to return

    Returns:
        List of indices sorted by score (highest first)
    """
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in indexed_scores[:max_count]]