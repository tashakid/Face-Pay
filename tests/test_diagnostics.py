"""
Test script to verify diagnostic functions are working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

try:
    from diagnostics import (
        calculate_brightness,
        detect_blur,
        calculate_score_distribution,
        log_image_quality,
        format_score
    )
    print("✅ Successfully imported diagnostics module\n")
except ImportError as e:
    print(f"❌ Failed to import diagnostics module: {e}")
    sys.exit(1)

print("="*60)
print("DIAGNOSTICS MODULE TEST")
print("="*60)

try:
    print("\n1. Testing image quality analysis...")
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    log_image_quality(test_image, level=logging.INFO)

    print("\n2. Testing with different brightness levels...")
    dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 200

    brightness_dark = calculate_brightness(dark_image)
    brightness_neutral = calculate_brightness(test_image)
    brightness_bright = calculate_brightness(bright_image)

    print(f"   Dark image brightness: {brightness_dark:.1f}%")
    print(f"   Neutral image brightness: {brightness_neutral:.1f}%")
    print(f"   Bright image brightness: {brightness_bright:.1f}%")

    print("\n3. Testing blur detection...")
    blur_dark = detect_blur(dark_image)
    blur_neutral = detect_blur(test_image)
    blur_bright = detect_blur(bright_image)

    print(f"   Dark blur score: {blur_dark['variance']:.1f} (quality: {blur_dark['quality']})")
    print(f"   Neutral blur score: {blur_neutral['variance']:.1f} (quality: {blur_neutral['quality']})")
    print(f"   Bright blur score: {blur_bright['variance']:.1f} (quality: {blur_bright['quality']})")

    print("\n4. Testing score distribution...")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9, -0.1, 0.2, 0.4, 0.6, 0.8]
    stats = calculate_score_distribution(test_scores)
    print(f"   Scores: {[format_score(s)[:7] for s in test_scores[:5]]}...")
    print(f"   Mean: {format_score(stats['mean'])}")
    print(f"   Median: {format_score(stats['median'])}")
    print(f"   Std: {format_score(stats['std'])}")
    print(f"   Positive: {stats['positive_count']}/{stats['count']}")

    print("\n5. Testing score formatting...")
    print(f"   High score: {format_score(0.8532)}")
    print(f"   Low score: {format_score(0.1234)}")
    print(f"   Negative: {format_score(-0.2345)}")
    print(f"   No percentage: {format_score(0.8532, show_percentage=False)}")

    print("\n7. Testing highlight_best_scores...")
    from diagnostics import highlight_best_scores
    best_indices = highlight_best_scores(test_scores, max_count=3)
    print(f"   Top 3 score indices: {best_indices}")

    print("\n" + "="*60)
    print("✅ All diagnostic tests passed!")
    print("="*60)

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)