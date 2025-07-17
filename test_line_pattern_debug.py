#!/usr/bin/env python3
"""
線パターン分類のデバッグスクリプト
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/evaluation'))

from utils.enhanced_screentone_detector import (
    EnhancedScreentoneDetector,
    ScreentoneFeatureExtractor,
    ScreentonePatternClassifier,
    ScreentoneType
)

def create_line_pattern(height: int, width: int, line_width: int, spacing: int) -> np.ndarray:
    """線パターン画像の生成"""
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    for x in range(0, width, spacing):
        cv2.rectangle(image, (x, 0), (x + line_width, height), 0, -1)
    
    return image

def create_dot_pattern(height: int, width: int, dot_size: int, spacing: int) -> np.ndarray:
    """ドットパターン画像の生成"""
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            cv2.circle(image, (x, y), dot_size // 2, 0, -1)
    
    return image

def debug_pattern_classification():
    """パターン分類のデバッグ"""
    
    # テスト画像作成
    line_image = create_line_pattern(128, 128, line_width=2, spacing=6)
    dot_image = create_dot_pattern(128, 128, dot_size=4, spacing=8)
    
    print("=== Pattern Classification Debug ===")
    
    # 特徴量抽出器とパターン分類器
    feature_extractor = ScreentoneFeatureExtractor()
    classifier = ScreentonePatternClassifier()
    
    for name, test_image in [("Line Pattern", line_image), ("Dot Pattern", dot_image)]:
        print(f"\n{name} Analysis:")
        print(f"Image stats: min={np.min(test_image)}, max={np.max(test_image)}, mean={np.mean(test_image):.1f}")
        
        # 特徴量抽出
        fft_features = feature_extractor.extract_fft_features(test_image)
        gabor_features = feature_extractor.extract_gabor_features(test_image)
        wavelet_features = feature_extractor.extract_wavelet_features(test_image)
        spatial_features = feature_extractor.extract_spatial_features(test_image)
        
        combined_features = {**fft_features, **gabor_features, **wavelet_features, **spatial_features}
        
        # 個別パターン評価
        dot_score = classifier._evaluate_dot_pattern(combined_features)
        line_score = classifier._evaluate_line_pattern(combined_features)
        gradient_score = classifier._evaluate_gradient_pattern(combined_features)
        noise_score = classifier._evaluate_noise_pattern(combined_features)
        
        print(f"Pattern Scores:")
        print(f"  Dot: {dot_score:.4f}")
        print(f"  Line: {line_score:.4f}")
        print(f"  Gradient: {gradient_score:.4f}")
        print(f"  Noise: {noise_score:.4f}")
        
        # 最終分類
        pattern_type, confidence = classifier.classify_pattern(combined_features)
        print(f"Final Classification: {pattern_type.value}, confidence: {confidence:.4f}")
        
        # 線パターンの重要特徴量
        if name == "Line Pattern":
            print(f"Key Line Features:")
            print(f"  edge_density: {combined_features.get('edge_density', 0):.4f}")
            print(f"  wavelet_directionality: {combined_features.get('wavelet_directionality', 0):.4f}")
            print(f"  orientation_uniformity: {combined_features.get('orientation_uniformity', 0):.4f}")
            print(f"  wavelet_ratio_h: {combined_features.get('wavelet_ratio_h', 0):.4f}")
            print(f"  wavelet_ratio_v: {combined_features.get('wavelet_ratio_v', 0):.4f}")
        
        # ドットパターンの重要特徴量
        if name == "Dot Pattern":
            print(f"Key Dot Features:")
            print(f"  periodicity_score: {combined_features.get('periodicity_score', 0):.4f}")
            print(f"  regularity_score: {combined_features.get('regularity_score', 0):.4f}")
            print(f"  gabor_energy: {combined_features.get('gabor_energy', 0):.1f}")
            print(f"  uniform_ratio: {combined_features.get('uniform_ratio', 0):.4f}")

if __name__ == "__main__":
    debug_pattern_classification()