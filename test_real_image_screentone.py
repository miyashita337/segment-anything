#!/usr/bin/env python3
"""
実画像でのスクリーントーン検出テスト
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/evaluation'))

from utils.enhanced_screentone_detector import detect_screentone_enhanced

def test_real_image_screentone():
    """実画像でのスクリーントーン検出テスト"""
    
    # テスト画像の確認
    test_image_path = Path("test_small/img001.jpg")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    # 画像読み込み
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"Failed to load image: {test_image_path}")
        return
    
    print(f"=== Real Image Screentone Detection Test ===")
    print(f"Image: {test_image_path}")
    print(f"Shape: {image.shape}")
    
    # スクリーントーン検出
    result = detect_screentone_enhanced(image)
    
    print(f"\nDetection Result:")
    print(f"  Has screentone: {result.has_screentone}")
    print(f"  Type: {result.screentone_type.value}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Quality score: {result.quality_score:.4f}")
    print(f"  Coverage ratio: {result.coverage_ratio:.4f}")
    print(f"  Pattern density: {result.pattern_density:.4f}")
    print(f"  Dominant frequency: {result.dominant_frequency:.4f}")
    print(f"  Reasoning: {result.reasoning}")
    
    # マスクの統計情報
    mask_pixels = np.sum(result.mask > 0)
    total_pixels = result.mask.size
    print(f"\nMask Statistics:")
    print(f"  Mask pixels: {mask_pixels}")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Mask ratio: {mask_pixels / total_pixels:.4f}")

if __name__ == "__main__":
    test_real_image_screentone()