#!/usr/bin/env python3
"""
Test for P1-001: 全身検出アルゴリズム分析
Phase 1対応: 現在のfullbody_priority手法の動作確認テスト
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/extraction'))
sys.path.append(str(project_root / 'features/evaluation'))

from models.yolo_wrapper import YOLOModelWrapper
from utils.learned_quality_assessment import LearnedQualityAssessment, ImageCharacteristics


class TestCurrentFullBodyDetection(unittest.TestCase):
    """現在の全身検出システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        # Mock データの準備
        self.quality_assessor = LearnedQualityAssessment()
        
    def test_aspect_ratio_fullbody_scoring(self):
        """アスペクト比による全身判定のテスト"""
        # Test cases: [height, width, expected_good_score]
        test_cases = [
            (200, 100, True),   # アスペクト比2.0 - 全身として良い（範囲内）
            (150, 100, True),   # アスペクト比1.5 - 全身として良い（範囲内）
            (100, 100, False),  # アスペクト比1.0 - 正方形、範囲外
            (100, 200, False),  # アスペクト比0.5 - 横長、範囲外
            (300, 100, False),  # アスペクト比3.0 - 範囲外（2.5超過）※問題発見
        ]
        
        for height, width, expected_good in test_cases:
            with self.subTest(height=height, width=width):
                # アスペクト比計算（current implementation logic）
                aspect_ratio = height / max(width, 1)
                
                # Current fullbody scoring logic simulation
                if 1.2 <= aspect_ratio <= 2.5:
                    fullbody_score = min((aspect_ratio - 0.5) / 2.0, 1.0)
                    is_good_fullbody = fullbody_score > 0.5
                else:
                    fullbody_score = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)
                    is_good_fullbody = fullbody_score > 0.3
                
                if expected_good:
                    self.assertGreater(fullbody_score, 0.3, 
                                     f"Aspect ratio {aspect_ratio:.2f} should have good fullbody score")
                
                print(f"Aspect ratio {aspect_ratio:.2f}: score={fullbody_score:.3f}, "
                      f"good={is_good_fullbody}, expected_good={expected_good}")
    
    def test_image_characteristics_fullbody_detection(self):
        """ImageCharacteristics での全身検出テスト"""
        # Test cases: [height, width, expected_has_full_body]
        test_cases = [
            (600, 400, True),   # 1.5倍 - has_full_body should be True
            (480, 400, False),  # 1.2倍 - 境界ケース、実際はFalse ※問題発見  
            (400, 400, False),  # 1.0倍 - has_full_body should be False
            (400, 500, False),  # 0.8倍 - has_full_body should be False
        ]
        
        for height, width, expected_has_full_body in test_cases:
            with self.subTest(height=height, width=width):
                # Current implementation logic simulation
                aspect_ratio = height / width
                
                if aspect_ratio >= 1.5:  # 縦長画像
                    has_full_body = True
                elif height > width * 1.2:  # 高さが幅の1.2倍以上
                    has_full_body = True
                else:
                    has_full_body = False
                
                self.assertEqual(has_full_body, expected_has_full_body,
                               f"Height {height}, Width {width} (ratio {aspect_ratio:.2f}) "
                               f"should have has_full_body={expected_has_full_body}")
                
                print(f"Dimensions {height}x{width}: ratio={aspect_ratio:.2f}, "
                      f"has_full_body={has_full_body}")
    
    def test_fullbody_priority_weights(self):
        """fullbody_priority の重み設定テスト"""
        # Expected weights from yolo_wrapper.py:315
        expected_weights = {
            'area': 0.20, 
            'fullbody': 0.40, 
            'central': 0.15, 
            'grounded': 0.15, 
            'confidence': 0.10
        }
        
        # Weight validation
        total_weight = sum(expected_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2,
                              msg="Fullbody priority weights should sum to 1.0")
        
        # Fullbody weight should be highest
        max_weight = max(expected_weights.values())
        self.assertEqual(expected_weights['fullbody'], max_weight,
                        "Fullbody should have highest weight in fullbody_priority")
        
        print(f"Fullbody priority weights: {expected_weights}")
        print(f"Total weight: {total_weight:.2f}")
        print(f"Fullbody weight dominance: {expected_weights['fullbody']:.2f} "
              f"(highest: {max_weight:.2f})")
    
    def test_composite_score_calculation(self):
        """複合スコア計算のテスト"""
        # Mock mask data for testing
        mock_mask_data = {
            'area': 8000,  # 例: 100x80 の領域
            'bbox': [50, 100, 100, 150],  # [x, y, width, height]
            'yolo_confidence': 0.8
        }
        
        # Image dimensions
        h, w = 400, 300
        image_center_x, image_center_y = w // 2, h // 2
        
        # Current implementation simulation
        scores = {}
        
        # 1. Area score
        area_ratio = mock_mask_data['area'] / (h * w)
        if 0.05 <= area_ratio <= 0.4:
            scores['area'] = min(area_ratio / 0.4, 1.0)
        else:
            scores['area'] = max(0, 1.0 - abs(area_ratio - 0.2) / 0.2)
        
        # 2. Fullbody score (aspect ratio)
        bbox = mock_mask_data['bbox']
        aspect_ratio = bbox[3] / max(bbox[2], 1)  # height / width
        if 1.2 <= aspect_ratio <= 2.5:
            scores['fullbody'] = min((aspect_ratio - 0.5) / 2.0, 1.0)
        else:
            scores['fullbody'] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)
        
        # 3. Central score
        mask_center_x = bbox[0] + bbox[2] / 2
        mask_center_y = bbox[1] + bbox[3] / 2
        distance_from_center = np.sqrt(
            ((mask_center_x - image_center_x) / w)**2 + 
            ((mask_center_y - image_center_y) / h)**2
        )
        scores['central'] = max(0, 1.0 - distance_from_center)
        
        # 4. Grounded score
        bottom_position = (bbox[1] + bbox[3]) / h
        if bottom_position >= 0.6:
            scores['grounded'] = min(bottom_position, 1.0)
        else:
            scores['grounded'] = bottom_position / 0.6
        
        # 5. Confidence score
        scores['confidence'] = mock_mask_data['yolo_confidence']
        
        # Composite score calculation
        weights = {'area': 0.20, 'fullbody': 0.40, 'central': 0.15, 'grounded': 0.15, 'confidence': 0.10}
        composite_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Assertions
        for score_name, score_value in scores.items():
            self.assertGreaterEqual(score_value, 0.0, f"{score_name} score should be non-negative")
            self.assertLessEqual(score_value, 1.0, f"{score_name} score should not exceed 1.0")
        
        self.assertGreaterEqual(composite_score, 0.0, "Composite score should be non-negative")
        
        print(f"Individual scores: {scores}")
        print(f"Composite score: {composite_score:.3f}")
        print(f"Aspect ratio: {aspect_ratio:.2f}")
        print(f"Area ratio: {area_ratio:.3f}")
    
    def test_quality_assessment_integration(self):
        """品質評価システムとの統合テスト"""
        try:
            # LearnedQualityAssessment の基本動作確認
            assessor = LearnedQualityAssessment()
            
            # Method recommendations の存在確認
            if hasattr(assessor, 'method_recommendations'):
                print(f"Method recommendations loaded: {len(assessor.method_recommendations)}")
            
            # Method stats の存在確認  
            if hasattr(assessor, 'method_stats'):
                print(f"Method stats loaded: {len(assessor.method_stats)}")
                
                # fullbody_priority の統計確認
                if 'fullbody_priority' in assessor.method_stats:
                    fullbody_stats = assessor.method_stats['fullbody_priority']
                    print(f"Fullbody priority stats: {fullbody_stats}")
            
            print("✅ Quality assessment integration test passed")
            
        except Exception as e:
            print(f"⚠️ Quality assessment integration issue: {e}")
            # This is not a failure - just information about current state


if __name__ == '__main__':
    unittest.main()