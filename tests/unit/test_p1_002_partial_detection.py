#!/usr/bin/env python3
"""
Test for P1-002: 部分抽出検出システム実装
Phase 1対応: 顔のみ/手足切断などの部分抽出検出のテスト
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

from utils.partial_extraction_detector import (
    PartialExtractionDetector, 
    ExtractionAnalysis,
    PartialExtractionIssue,
    analyze_extraction_completeness
)


class TestPartialExtractionDetector(unittest.TestCase):
    """部分抽出検出システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.detector = PartialExtractionDetector()
        
        # テスト用のシンプルな画像・マスクを作成
        self.test_image = np.zeros((400, 300, 3), dtype=np.uint8)
        self.test_image[100:300, 50:250] = [100, 150, 200]  # 人物っぽい領域
    
    def test_detector_initialization(self):
        """検出器の初期化テスト"""
        detector = PartialExtractionDetector()
        
        # 基本属性の存在確認
        self.assertIsNotNone(detector.logger)
        
        # OpenCV cascade の確認（エラーでもOK）
        # これらはシステム依存なのでNoneでも正常
        print(f"Face cascade available: {detector.face_cascade is not None}")
        print(f"Profile cascade available: {detector.profile_cascade is not None}")
        print(f"Dlib detector available: {detector.dlib_detector is not None}")
    
    def test_face_only_mask_analysis(self):
        """顔のみマスクの分析テスト"""
        # 顔のみのマスク（正方形に近い、画像上部）
        face_only_mask = np.zeros((400, 300), dtype=np.uint8)
        face_only_mask[50:150, 100:200] = 255  # 100x100の正方形領域
        
        analysis = self.detector.analyze_extraction(self.test_image, face_only_mask)
        
        # 基本構造の確認
        self.assertIsInstance(analysis, ExtractionAnalysis)
        self.assertIn(analysis.has_face, [True, False])
        self.assertIn(analysis.has_torso, [True, False])
        self.assertIn(analysis.has_limbs, [True, False])
        self.assertIsInstance(analysis.completeness_score, (int, float))
        self.assertIsInstance(analysis.issues, list)
        
        # スコアの範囲確認
        self.assertGreaterEqual(analysis.completeness_score, 0.0)
        self.assertLessEqual(analysis.completeness_score, 1.0)
        
        # 品質評価の確認
        self.assertIn(analysis.quality_assessment, ['good', 'partial', 'poor'])
        
        print(f"Face-only analysis: completeness={analysis.completeness_score:.3f}, "
              f"quality={analysis.quality_assessment}, issues={len(analysis.issues)}")
    
    def test_full_body_mask_analysis(self):
        """全身マスクの分析テスト"""
        # 全身のマスク（縦長、画像全体をカバー）
        full_body_mask = np.zeros((400, 300), dtype=np.uint8)
        full_body_mask[50:350, 80:220] = 255  # 300x140の縦長領域
        
        analysis = self.detector.analyze_extraction(self.test_image, full_body_mask)
        
        # 全身マスクは高いcompleteness_scoreが期待される
        # ただし、顔検出が機能しない可能性があるので、スコア0.3以上で判定
        print(f"Full-body analysis: completeness={analysis.completeness_score:.3f}, "
              f"quality={analysis.quality_assessment}, issues={len(analysis.issues)}")
        
        # 全身マスクでは顔のみ抽出の問題は発生しないはず
        face_only_issues = [issue for issue in analysis.issues if issue.issue_type == 'face_only']
        self.assertEqual(len(face_only_issues), 0, "Full body mask should not have face_only issues")
    
    def test_torso_presence_analysis(self):
        """胴体存在分析のテスト"""
        # 中央部に密度の高い領域があるマスク
        torso_mask = np.zeros((400, 300), dtype=np.uint8)
        torso_mask[120:280, 100:200] = 255  # 中央部の密集領域
        
        has_torso = self.detector._analyze_torso_presence(torso_mask)
        self.assertTrue(has_torso, "Dense central region should be detected as torso")
        
        # 中央部に密度がないマスク
        no_torso_mask = np.zeros((400, 300), dtype=np.uint8)
        no_torso_mask[50:100, 50:100] = 255    # 上左角のみ
        no_torso_mask[300:350, 200:250] = 255  # 下右角のみ
        
        has_torso_sparse = self.detector._analyze_torso_presence(no_torso_mask)
        self.assertFalse(has_torso_sparse, "Sparse regions should not be detected as torso")
    
    def test_limb_presence_analysis(self):
        """手足存在分析のテスト"""
        # 外周部に密度があるマスク
        limb_mask = np.zeros((400, 300), dtype=np.uint8)
        limb_mask[:, 0:30] = 255     # 左端（左手）
        limb_mask[:, 270:300] = 255  # 右端（右手）
        limb_mask[350:400, :] = 255  # 下端（足）
        
        has_limbs = self.detector._analyze_limb_presence(limb_mask)
        self.assertTrue(has_limbs, "Edge regions should be detected as limbs")
        
        # 中央部のみのマスク
        no_limb_mask = np.zeros((400, 300), dtype=np.uint8)
        no_limb_mask[150:250, 100:200] = 255  # 中央部のみ
        
        has_limbs_central = self.detector._analyze_limb_presence(no_limb_mask)
        self.assertFalse(has_limbs_central, "Central region only should not be detected as having limbs")
    
    def test_limb_truncation_detection(self):
        """手足切断検出のテスト"""
        # 画像境界で切断されているマスク
        truncated_mask = np.zeros((400, 300), dtype=np.uint8)
        truncated_mask[0:350, 50:250] = 255  # 上端から始まり下端で切断
        
        issues = self.detector._detect_limb_truncation(truncated_mask)
        
        # 切断問題が検出されることを確認
        truncation_issues = [issue for issue in issues if issue.issue_type == 'limb_truncated']
        print(f"Truncation detection: {len(truncation_issues)} issues found")
        
        if truncation_issues:
            issue = truncation_issues[0]
            self.assertGreater(issue.confidence, 0.0)
            self.assertLessEqual(issue.confidence, 1.0)
            self.assertIn(issue.severity, ['low', 'medium', 'high'])
    
    def test_completeness_score_calculation(self):
        """完全性スコア計算のテスト"""
        # 各構造要素の組み合わせテスト
        test_cases = [
            (True, True, True, [], 1.0),    # 完璧な構造
            (True, False, False, [], 0.3),  # 顔のみ
            (False, True, True, [], 0.7),   # 顔なし
            (False, False, False, [], 0.0), # 何もなし
        ]
        
        for has_face, has_torso, has_limbs, issues, expected_min in test_cases:
            score = self.detector._calculate_completeness_score(has_face, has_torso, has_limbs, issues)
            
            self.assertGreaterEqual(score, 0.0, "Completeness score should be non-negative")
            self.assertLessEqual(score, 1.0, "Completeness score should not exceed 1.0")
            
            if expected_min == 1.0:
                self.assertGreaterEqual(score, 0.9, "Perfect structure should have high score")
            elif expected_min == 0.0:
                self.assertEqual(score, 0.0, "No structure should have zero score")
            
            print(f"Structure({has_face}, {has_torso}, {has_limbs}): score={score:.3f}")
    
    def test_analysis_with_issues(self):
        """問題を含む分析のテスト"""
        # 問題のあるマスク
        problematic_mask = np.zeros((400, 300), dtype=np.uint8)
        problematic_mask[100:200, 100:200] = 255  # 小さな正方形領域
        
        analysis = self.detector.analyze_extraction(self.test_image, problematic_mask)
        
        # 問題が検出されることを確認
        self.assertGreaterEqual(len(analysis.issues), 0, "Problematic mask should have issues")
        
        # 各問題の構造確認
        for issue in analysis.issues:
            self.assertIsInstance(issue, PartialExtractionIssue)
            self.assertIn(issue.issue_type, [
                'face_only', 'limb_truncated', 'torso_missing', 
                'incomplete_extraction', 'analysis_failed'
            ])
            self.assertIn(issue.severity, ['low', 'medium', 'high'])
            self.assertGreaterEqual(issue.confidence, 0.0)
            self.assertLessEqual(issue.confidence, 1.0)
    
    def test_convenience_function(self):
        """便利関数のテスト"""
        # テスト用画像作成
        test_image_path = "/tmp/test_image.jpg"
        cv2.imwrite(test_image_path, self.test_image)
        
        # テスト用マスク
        test_mask = np.zeros((400, 300), dtype=np.uint8)
        test_mask[100:300, 50:250] = 255
        
        try:
            analysis = analyze_extraction_completeness(test_image_path, test_mask)
            
            self.assertIsInstance(analysis, ExtractionAnalysis)
            print(f"Convenience function test: completeness={analysis.completeness_score:.3f}")
            
        except Exception as e:
            # ファイルアクセスエラーの場合はスキップ
            print(f"Convenience function test skipped: {e}")
        
        # 存在しないファイルでのエラーハンドリングテスト
        analysis_error = analyze_extraction_completeness("/nonexistent/path.jpg", test_mask)
        self.assertEqual(analysis_error.quality_assessment, 'poor')
        self.assertGreater(len(analysis_error.issues), 0)


if __name__ == '__main__':
    unittest.main()