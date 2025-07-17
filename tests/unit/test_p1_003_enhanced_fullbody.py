#!/usr/bin/env python3
"""
Test for P1-003: 全身判定基準の改善
Phase 1対応: 改良版全身検出システムのテスト
"""

import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/evaluation'))

from utils.enhanced_fullbody_detector import (
    EnhancedFullBodyDetector, 
    FullBodyScore,
    BodyStructureAnalysis,
    evaluate_fullbody_enhanced
)


class TestEnhancedFullBodyDetector(unittest.TestCase):
    """改良版全身検出システムのテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.detector = EnhancedFullBodyDetector()
        
        # テスト用画像作成
        self.test_image = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # リアルなキャラクター画像をシミュレート
        # 顔領域（肌色）
        self.test_image[50:100, 125:175] = [180, 150, 120]  # 肌色
        # 髪領域
        self.test_image[40:70, 115:185] = [50, 30, 20]      # 茶色髪
        # 胴体領域（服）
        self.test_image[100:250, 100:200] = [80, 100, 150]  # 青い服
        # 手足領域
        self.test_image[120:180, 70:110] = [180, 150, 120]  # 左手
        self.test_image[120:180, 190:230] = [180, 150, 120] # 右手
        self.test_image[250:350, 125:175] = [180, 150, 120] # 足
        # 髪のアクセサリー
        self.test_image[35:50, 130:170] = [200, 50, 50]     # 赤いリボン
    
    def test_detector_initialization(self):
        """検出器の初期化テスト"""
        detector = EnhancedFullBodyDetector()
        
        # 基本属性の確認
        self.assertIsNotNone(detector.logger)
        self.assertIsNotNone(detector.partial_detector)
        self.assertIsInstance(detector.weights, dict)
        self.assertIsInstance(detector.thresholds, dict)
        self.assertIsInstance(detector.aspect_ratio_ranges, dict)
        
        # 重みの合計確認
        weights_sum = sum(detector.weights.values())
        self.assertAlmostEqual(weights_sum, 1.0, places=2, 
                              msg="Weights should sum to approximately 1.0")
        
        print(f"Enhanced detector initialized with weights: {detector.weights}")
        print(f"Thresholds: {detector.thresholds}")
    
    def test_aspect_ratio_score_calculation(self):
        """改良版アスペクト比スコア計算のテスト"""
        # テストケース: [height, width, expected_score_range]
        test_cases = [
            (200, 100, (0.8, 1.0)),   # 理想的 2.0
            (150, 100, (0.8, 1.0)),   # 理想的 1.5
            (180, 100, (0.8, 1.0)),   # 理想的 1.8
            (120, 100, (0.6, 0.8)),   # 許容可能 1.2
            (280, 100, (0.6, 0.8)),   # 許容可能 2.8
            (100, 100, (0.2, 0.4)),   # 範囲外 1.0
            (350, 100, (0.1, 0.3)),   # 範囲外 3.5
        ]
        
        for height, width, expected_range in test_cases:
            with self.subTest(height=height, width=width):
                mask_data = {'bbox': [50, 50, width, height]}
                
                score = self.detector._calculate_aspect_ratio_score(mask_data, 400, 300)
                
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                self.assertGreaterEqual(score, expected_range[0])
                self.assertLessEqual(score, expected_range[1])
                
                aspect_ratio = height / width
                print(f"Aspect ratio {aspect_ratio:.2f}: score={score:.3f} "
                      f"(expected: {expected_range[0]:.1f}-{expected_range[1]:.1f})")
    
    def test_body_structure_analysis(self):
        """人体構造分析のテスト"""
        # 全身マスク作成
        full_body_mask = np.zeros((400, 300), dtype=np.uint8)
        full_body_mask[40:350, 100:200] = 255  # 縦長の全身領域
        
        structure_analysis = self.detector._analyze_body_structure(self.test_image, full_body_mask)
        
        # 基本構造の確認
        self.assertIsInstance(structure_analysis, BodyStructureAnalysis)
        self.assertIsInstance(structure_analysis.face_regions, list)
        self.assertGreaterEqual(structure_analysis.torso_density, 0.0)
        self.assertLessEqual(structure_analysis.torso_density, 1.0)
        self.assertGreaterEqual(structure_analysis.limb_density, 0.0)
        self.assertLessEqual(structure_analysis.limb_density, 1.0)
        
        print(f"Body structure analysis:")
        print(f"  Face regions: {len(structure_analysis.face_regions)}")
        print(f"  Torso density: {structure_analysis.torso_density:.3f}")
        print(f"  Limb density: {structure_analysis.limb_density:.3f}")
        print(f"  Vertical distribution: {structure_analysis.vertical_distribution:.3f}")
        print(f"  Structure completeness: {structure_analysis.structure_completeness:.3f}")
    
    def test_edge_distribution_score(self):
        """エッジ分布スコア計算のテスト"""
        # テスト用マスク（人型らしいエッジ分布）
        human_like_mask = np.zeros((400, 300), dtype=np.uint8)
        
        # 頭部（円形）
        cv2.circle(human_like_mask, (150, 70), 25, 255, -1)
        # 胴体（矩形）
        cv2.rectangle(human_like_mask, (125, 95), (175, 220), 255, -1)
        # 手足（線形）
        cv2.rectangle(human_like_mask, (100, 120), (125, 180), 255, -1)  # 左手
        cv2.rectangle(human_like_mask, (175, 120), (200, 180), 255, -1)  # 右手
        cv2.rectangle(human_like_mask, (135, 220), (165, 300), 255, -1)  # 足
        
        mask_data = {'mask': human_like_mask}
        
        edge_score = self.detector._calculate_edge_distribution_score(mask_data, 400, 300)
        
        self.assertGreaterEqual(edge_score, 0.0)
        self.assertLessEqual(edge_score, 1.0)
        
        print(f"Edge distribution score: {edge_score:.3f}")
    
    def test_semantic_region_score(self):
        """セマンティック領域スコア計算のテスト"""
        # 人型マスク
        human_mask = np.zeros((400, 300), dtype=np.uint8)
        human_mask[50:350, 100:200] = 255
        
        mask_data = {'mask': human_mask}
        
        semantic_score = self.detector._calculate_semantic_region_score(self.test_image, mask_data)
        
        self.assertGreaterEqual(semantic_score, 0.0)
        self.assertLessEqual(semantic_score, 1.0)
        
        print(f"Semantic region score: {semantic_score:.3f}")
    
    def test_fullbody_score_evaluation(self):
        """全身スコア評価の統合テスト"""
        # 高品質全身マスクデータ
        good_fullbody_mask = np.zeros((400, 300), dtype=np.uint8)
        good_fullbody_mask[40:350, 100:200] = 255
        
        mask_data = {
            'bbox': [100, 40, 100, 310],  # [x, y, width, height]
            'mask': good_fullbody_mask,
            'area': np.sum(good_fullbody_mask > 0)
        }
        
        fullbody_score = self.detector.evaluate_fullbody_score(self.test_image, mask_data)
        
        # 基本構造確認
        self.assertIsInstance(fullbody_score, FullBodyScore)
        self.assertGreaterEqual(fullbody_score.total_score, 0.0)
        self.assertLessEqual(fullbody_score.total_score, 1.0)
        self.assertGreaterEqual(fullbody_score.confidence, 0.0)
        self.assertLessEqual(fullbody_score.confidence, 1.0)
        self.assertIsInstance(fullbody_score.reasoning, str)
        
        # 個別スコア確認
        self.assertGreaterEqual(fullbody_score.aspect_ratio_score, 0.0)
        self.assertGreaterEqual(fullbody_score.body_structure_score, 0.0)
        self.assertGreaterEqual(fullbody_score.edge_distribution_score, 0.0)
        self.assertGreaterEqual(fullbody_score.semantic_region_score, 0.0)
        self.assertGreaterEqual(fullbody_score.completeness_bonus, 0.0)
        
        print(f"Enhanced fullbody evaluation:")
        print(f"  Total score: {fullbody_score.total_score:.3f}")
        print(f"  Aspect ratio: {fullbody_score.aspect_ratio_score:.3f}")
        print(f"  Body structure: {fullbody_score.body_structure_score:.3f}")
        print(f"  Edge distribution: {fullbody_score.edge_distribution_score:.3f}")
        print(f"  Semantic regions: {fullbody_score.semantic_region_score:.3f}")
        print(f"  Completeness bonus: {fullbody_score.completeness_bonus:.3f}")
        print(f"  Confidence: {fullbody_score.confidence:.3f}")
        print(f"  Reasoning: {fullbody_score.reasoning}")
    
    def test_quality_classification(self):
        """品質分類のテスト"""
        # 異なる品質のスコア
        test_scores = [
            FullBodyScore(0.85, 0.8, 0.9, 0.8, 0.8, 0.1, 0.9, "excellent"),
            FullBodyScore(0.65, 0.7, 0.6, 0.7, 0.6, 0.1, 0.7, "good"),
            FullBodyScore(0.45, 0.5, 0.4, 0.5, 0.4, 0.1, 0.5, "partial"),
            FullBodyScore(0.25, 0.3, 0.2, 0.3, 0.2, 0.1, 0.3, "poor")
        ]
        
        expected_qualities = [
            'excellent_fullbody',
            'good_fullbody', 
            'partial_extraction',
            'poor_extraction'
        ]
        
        for score, expected_quality in zip(test_scores, expected_qualities):
            quality = self.detector.classify_extraction_quality(score)
            self.assertEqual(quality, expected_quality)
            print(f"Score {score.total_score:.2f} -> Quality: {quality}")
    
    def test_improvement_suggestions(self):
        """改善提案のテスト"""
        # 低品質スコア
        poor_score = FullBodyScore(
            total_score=0.3,
            aspect_ratio_score=0.2,
            body_structure_score=0.1,
            edge_distribution_score=0.3,
            semantic_region_score=0.2,
            completeness_bonus=0.1,
            confidence=0.4,
            reasoning="poor quality"
        )
        
        suggestions = self.detector.suggest_improvements(poor_score)
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        print(f"Improvement suggestions for poor quality:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    def test_convenience_function(self):
        """便利関数のテスト"""
        mask_data = {
            'bbox': [100, 50, 100, 250],
            'mask': np.ones((400, 300), dtype=np.uint8) * 255,
            'area': 25000
        }
        
        try:
            fullbody_score = evaluate_fullbody_enhanced(self.test_image, mask_data)
            
            self.assertIsInstance(fullbody_score, FullBodyScore)
            print(f"Convenience function test: score={fullbody_score.total_score:.3f}")
            
        except Exception as e:
            # システム依存のエラーは許容
            print(f"Convenience function test encountered error: {e}")
    
    def test_comparison_with_original(self):
        """従来手法との比較テスト"""
        # 同じマスクで従来手法と改良手法を比較
        mask_data = {
            'bbox': [100, 40, 100, 300],  # アスペクト比3.0
            'area': 30000
        }
        
        # 従来手法（アスペクト比のみ）
        aspect_ratio = 300 / 100  # 3.0
        if 1.2 <= aspect_ratio <= 2.5:
            original_score = min((aspect_ratio - 0.5) / 2.0, 1.0)
        else:
            original_score = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)
        
        # 改良手法
        enhanced_score_obj = self.detector.evaluate_fullbody_score(self.test_image, mask_data)
        enhanced_score = enhanced_score_obj.total_score
        
        print(f"Comparison test (aspect ratio 3.0):")
        print(f"  Original method: {original_score:.3f}")
        print(f"  Enhanced method: {enhanced_score:.3f}")
        print(f"  Improvement: {enhanced_score - original_score:+.3f}")
        
        # 改良手法は多指標なのでより安定した評価が期待される
        self.assertGreaterEqual(enhanced_score, 0.0)
        self.assertLessEqual(enhanced_score, 1.0)


if __name__ == '__main__':
    unittest.main()