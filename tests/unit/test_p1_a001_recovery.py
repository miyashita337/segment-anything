#!/usr/bin/env python3
"""
P1-A001改善コード復旧システムのテスト
deprecatedから復旧されたコンポーネントの動作確認
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# テスト対象モジュール
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.evaluation.utils.true_success_analyzer import TrueSuccessAnalyzer, TrueVerificationResult
from features.evaluation.utils.visual_verification_system import VisualVerificationSystem, VerificationResult
from tools.core.enhanced_sam_pipeline import PerformanceMonitor, QualityEvaluator, TextDetector


class TestTrueSuccessAnalyzer(unittest.TestCase):
    """真の成功率分析システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir
        
        # テストデータ作成
        self.workspace = self.project_root / "workspace" / "P1-A001"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # サンプルAI結果
        self.ai_results = {
            "test_image_1": {
                "bbox": [100, 100, 200, 300],
                "iou": 0.85,
                "success": True,
                "quality_score": 0.9,
                "confidence": 0.8
            },
            "test_image_2": {
                "bbox": [50, 50, 150, 250],
                "iou": 0.6,
                "success": False,
                "quality_score": 0.5,
                "confidence": 0.4
            }
        }
        
        # サンプル人間ラベル
        self.human_labels = {
            "test_image_1": {
                "bbox": [95, 95, 205, 305],
                "character_description": "主人公キャラクター"
            },
            "test_image_2": {
                "bbox": [45, 45, 155, 255],
                "character_description": "サブキャラクター"
            }
        }
        
        self.analyzer = TrueSuccessAnalyzer(self.project_root)
        self.analyzer.ai_results = self.ai_results
        self.analyzer.human_labels = self.human_labels
    
    def test_analyzer_initialization(self):
        """アナライザー初期化テスト"""
        self.assertIsInstance(self.analyzer, TrueSuccessAnalyzer)
        self.assertEqual(self.analyzer.project_root, self.project_root)
        self.assertTrue(self.analyzer.workspace.exists())
    
    def test_calculate_iou(self):
        """IoU計算テスト"""
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)
        
        iou = self.analyzer._calculate_iou(bbox1, bbox2)
        
        # 期待値計算: bbox1=(100,100,200,200), bbox2=(150,150,250,250)
        # intersection=(150,150,200,200) -> area=50*50=2500
        # area1=100*100=10000, area2=100*100=10000
        # union=10000+10000-2500=17500, iou=2500/17500=1/7
        expected_iou = 2500 / 17500
        self.assertAlmostEqual(iou, expected_iou, places=3)
    
    def test_coordinate_match_check(self):
        """座標一致度チェックテスト"""
        # 高い一致度
        bbox1 = (100, 100, 200, 200)
        bbox2 = (105, 105, 195, 195)
        self.assertTrue(self.analyzer._check_coordinate_match(bbox1, bbox2, threshold=0.7))
        
        # 低い一致度
        bbox3 = (300, 300, 400, 400)
        self.assertFalse(self.analyzer._check_coordinate_match(bbox1, bbox3, threshold=0.7))
    
    def test_verify_single_result(self):
        """単一結果検証テスト"""
        image_id = "test_image_1"
        ai_result = self.ai_results[image_id]
        human_label = self.human_labels[image_id]
        
        verification = self.analyzer._verify_single_result(image_id, ai_result, human_label)
        
        self.assertIsInstance(verification, TrueVerificationResult)
        self.assertEqual(verification.image_id, image_id)
        self.assertTrue(verification.coordinate_match)  # 高いIoUなので一致
        self.assertTrue(verification.visual_content_match)  # 高品質・高信頼度
        self.assertTrue(verification.true_success)
    
    def test_analyze_true_success_rate(self):
        """真の成功率分析テスト"""
        analysis = self.analyzer.analyze_true_success_rate()
        
        self.assertIn('total_analyzed', analysis)
        self.assertIn('true_success_rate', analysis)
        self.assertIn('verification_results', analysis)
        
        # 2件のテストデータを処理
        self.assertEqual(analysis['total_analyzed'], 2)
        self.assertIsInstance(analysis['true_success_rate'], float)
        self.assertEqual(len(analysis['verification_results']), 2)


class TestVisualVerificationSystem(unittest.TestCase):
    """視覚的検証システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir
        
        # テストデータ作成
        self.workspace = self.project_root / "workspace" / "P1-A001"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # サンプルデータ（TrueSuccessAnalyzerと同じ構造）
        self.ai_results = {
            "test_image_1": {
                "bbox": [100, 100, 200, 300],
                "iou": 0.85,
                "quality_score": 0.9,
                "confidence": 0.8,
                "character_description": "主人公"
            }
        }
        
        self.human_labels = {
            "test_image_1": {
                "bbox": [95, 95, 205, 305],
                "character_description": "主人公キャラクター"
            }
        }
        
        self.verifier = VisualVerificationSystem(self.project_root)
        self.verifier.ai_results = self.ai_results
        self.verifier.human_labels = self.human_labels
    
    def test_verifier_initialization(self):
        """検証システム初期化テスト"""
        self.assertIsInstance(self.verifier, VisualVerificationSystem)
        self.assertEqual(self.verifier.project_root, self.project_root)
        self.assertTrue(self.verifier.workspace.exists())
    
    def test_extract_bbox(self):
        """境界ボックス抽出テスト"""
        bbox_list = [100, 100, 200, 200]
        result = self.verifier._extract_bbox(bbox_list)
        self.assertEqual(result, (100, 100, 200, 200))
        
        # 不正なデータ
        invalid_bbox = [100, 100]  # 要素不足
        result = self.verifier._extract_bbox(invalid_bbox)
        self.assertEqual(result, (0, 0, 0, 0))
    
    def test_calculate_size_ratio(self):
        """サイズ比率計算テスト"""
        bbox1 = (0, 0, 100, 100)  # 面積: 10000
        bbox2 = (0, 0, 50, 50)    # 面積: 2500
        
        ratio = self.verifier._calculate_size_ratio(bbox1, bbox2)
        expected_ratio = 2500 / 10000  # 0.25
        self.assertAlmostEqual(ratio, expected_ratio, places=3)
    
    def test_judge_visual_match(self):
        """視覚的一致度判定テスト"""
        # 高品質データ
        high_quality_ai = {
            "quality_score": 0.9,
            "confidence": 0.8,
            "iou": 0.85,
            "bbox": [100, 100, 200, 200]
        }
        human_label = {"bbox": [95, 95, 205, 205]}
        
        match = self.verifier._judge_visual_match(high_quality_ai, human_label)
        self.assertTrue(match)
        
        # 低品質データ
        low_quality_ai = {
            "quality_score": 0.3,
            "confidence": 0.2,
            "iou": 0.3,
            "bbox": [100, 100, 200, 200]
        }
        
        match = self.verifier._judge_visual_match(low_quality_ai, human_label)
        self.assertFalse(match)


class TestPerformanceMonitor(unittest.TestCase):
    """パフォーマンス監視システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_initialization(self):
        """監視システム初期化テスト"""
        self.assertIsInstance(self.monitor, PerformanceMonitor)
        self.assertIsNone(self.monitor.start_time)
        self.assertEqual(self.monitor.stage_times, {})
    
    def test_start_monitoring(self):
        """監視開始テスト"""
        with patch('builtins.print'):  # print出力を抑制
            self.monitor.start_monitoring()
        
        self.assertIsNotNone(self.monitor.start_time)
        self.assertEqual(self.monitor.stage_times, {})
    
    def test_stage_timing(self):
        """ステージタイミングテスト"""
        with patch('builtins.print'):  # print出力を抑制
            self.monitor.start_monitoring()
            self.monitor.start_stage("テストステージ")
            
            # 少し待機
            import time
            time.sleep(0.01)
            
            self.monitor.end_stage()
        
        self.assertIn("テストステージ", self.monitor.stage_times)
        self.assertIsInstance(self.monitor.stage_times["テストステージ"], float)
        self.assertGreater(self.monitor.stage_times["テストステージ"], 0)


class TestQualityEvaluator(unittest.TestCase):
    """品質評価システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.evaluator = QualityEvaluator()
        
        # テスト用画像・マスク作成
        self.test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.test_mask = np.zeros((300, 400), dtype=np.uint8)
        self.test_mask[50:250, 50:350] = 255  # 矩形マスク
        self.test_bbox = (50, 50, 350, 250)
    
    def test_evaluator_initialization(self):
        """評価システム初期化テスト"""
        self.assertIsInstance(self.evaluator, QualityEvaluator)
        self.assertEqual(len(self.evaluator.evaluation_methods), 5)
        self.assertIn('balanced', self.evaluator.evaluation_methods)
    
    def test_area_ratio_calculation(self):
        """面積比率計算テスト"""
        ratio = self.evaluator._calculate_area_ratio(self.test_mask)
        
        # 計算検証
        total_pixels = 300 * 400
        mask_pixels = 200 * 300  # 矩形マスクの面積
        expected_ratio = mask_pixels / total_pixels
        
        self.assertAlmostEqual(ratio, expected_ratio, places=3)
    
    def test_compactness_calculation(self):
        """コンパクトネス計算テスト"""
        compactness = self.evaluator._calculate_compactness(self.test_mask)
        
        # 矩形の場合、コンパクトネスは円形より低い
        self.assertGreater(compactness, 0)
        self.assertLess(compactness, 1)
    
    def test_balanced_evaluation(self):
        """バランス評価テスト"""
        result = self.evaluator._evaluate_balanced(self.test_image, self.test_mask, self.test_bbox)
        
        self.assertIn('quality_score', result)
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'balanced')
        self.assertGreater(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 1)
    
    def test_all_evaluation_methods(self):
        """全評価手法テスト"""
        methods = ['balanced', 'confidence_priority', 'size_priority', 
                  'fullbody_priority', 'central_priority']
        
        for method in methods:
            result = self.evaluator.evaluate_extraction_quality(
                self.test_image, self.test_mask, self.test_bbox, method
            )
            
            self.assertIn('quality_score', result)
            self.assertIn('method', result)
            self.assertEqual(result['method'], method)
            self.assertIsInstance(result['quality_score'], float)


class TestTextDetector(unittest.TestCase):
    """テキスト検出システムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.detector = TextDetector()
        self.test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """検出システム初期化テスト"""
        self.assertIsInstance(self.detector, TextDetector)
        # OCRの利用可能性は環境依存
        self.assertIsInstance(self.detector.ocr_available, bool)
    
    @patch('tools.core.enhanced_sam_pipeline.OCR_AVAILABLE', False)
    def test_text_detection_without_ocr(self):
        """OCR非利用時のテキスト検出テスト"""
        with patch('tools.core.enhanced_sam_pipeline.TextDetector.__init__') as mock_init:
            mock_init.return_value = None
            detector_no_ocr = TextDetector()
            detector_no_ocr.ocr_available = False
            
            regions = detector_no_ocr.detect_text_regions(self.test_image)
            
            # OCRが利用できない場合は空リストを返す
            self.assertEqual(regions, [])
    
    def test_create_text_mask(self):
        """テキストマスク作成テスト"""
        text_regions = [(50, 50, 150, 100), (200, 200, 300, 250)]
        mask = self.detector.create_text_mask(self.test_image, text_regions)
        
        self.assertEqual(mask.shape, (300, 400))
        self.assertEqual(mask.dtype, np.uint8)
        
        # 指定領域にマスクが作成されているか確認
        self.assertGreater(np.sum(mask[45:155, 45:155]), 0)  # マージン考慮
        self.assertGreater(np.sum(mask[195:255, 195:305]), 0)


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2)