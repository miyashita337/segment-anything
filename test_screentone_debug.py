#!/usr/bin/env python3
"""
スクリーントーン検出のデバッグスクリプト
"""

import numpy as np
import cv2

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/evaluation'))

from utils.enhanced_screentone_detector import (
    EnhancedScreentoneDetector,
    ScreentoneFeatureExtractor,
    ScreentonePatternClassifier,
    detect_screentone_enhanced,
)


def create_dot_pattern(height: int, width: int, dot_size: int, spacing: int) -> np.ndarray:
    """ドットパターン画像の生成"""
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            cv2.circle(image, (x, y), dot_size // 2, 0, -1)
    
    return image

def debug_screentone_detection():
    """スクリーントーン検出のデバッグ"""
    
    # テスト画像作成
    dot_image = create_dot_pattern(128, 128, dot_size=4, spacing=8)
    
    print("=== Enhanced Screentone Detection Debug ===")
    print(f"Test image shape: {dot_image.shape}")
    print(f"Test image stats: min={np.min(dot_image)}, max={np.max(dot_image)}, mean={np.mean(dot_image):.1f}")
    
    # 特徴量抽出器のテスト
    feature_extractor = ScreentoneFeatureExtractor()
    
    # FFT特徴量
    fft_features = feature_extractor.extract_fft_features(dot_image)
    print(f"\nFFT Features:")
    for key, value in fft_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Gabor特徴量
    gabor_features = feature_extractor.extract_gabor_features(dot_image)
    print(f"\nGabor Features:")
    for key, value in gabor_features.items():
        print(f"  {key}: {value:.4f}")
    
    # 空間特徴量
    spatial_features = feature_extractor.extract_spatial_features(dot_image)
    print(f"\nSpatial Features:")
    for key, value in spatial_features.items():
        print(f"  {key}: {value:.4f}")
    
    # 特徴量統合
    combined_features = {**fft_features, **gabor_features, **spatial_features}
    
    # パターン分類器のテスト
    classifier = ScreentonePatternClassifier()
    pattern_type, confidence = classifier.classify_pattern(combined_features)
    
    print(f"\nPattern Classification:")
    print(f"  Type: {pattern_type.value}")
    print(f"  Confidence: {confidence:.4f}")
    
    # 検出器のテスト
    detector = EnhancedScreentoneDetector()
    result = detector.detect_screentone(dot_image)
    
    print(f"\nDetection Result:")
    print(f"  Has screentone: {result.has_screentone}")
    print(f"  Type: {result.screentone_type.value}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Quality score: {result.quality_score:.4f}")
    print(f"  Coverage ratio: {result.coverage_ratio:.4f}")
    print(f"  Reasoning: {result.reasoning}")
    
    # 閾値との比較
    min_confidence = detector.detection_params['min_confidence']
    print(f"\nThreshold Analysis:")
    print(f"  Min confidence threshold: {min_confidence}")
    print(f"  Pattern confidence: {confidence:.4f}")
    print(f"  Meets threshold: {confidence > min_confidence}")

if __name__ == "__main__":
    debug_screentone_detection()