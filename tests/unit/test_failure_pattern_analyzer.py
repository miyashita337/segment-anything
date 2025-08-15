#!/usr/bin/env python3
"""
QCC-011: 失敗パターン分析システムのユニットテスト
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from features.analysis.failure_pattern_analyzer import FailurePatternAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


class TestFailurePatternAnalyzer(unittest.TestCase):
    """失敗パターン分析システムのテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        if not ANALYZER_AVAILABLE:
            self.skipTest("FailurePatternAnalyzer が利用できません")
        
        self.analyzer = FailurePatternAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """分析器の初期化テスト"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer.failure_patterns, dict)
        self.assertIsInstance(self.analyzer.analysis_results, dict)
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_extract_image_features(self, mock_cvtColor, mock_imread):
        """画像特徴抽出テスト"""
        # モック画像データ
        mock_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_cvtColor.return_value = mock_img
        
        # テスト画像パス
        test_image_path = self.temp_path / "test.jpg"
        test_image_path.touch()
        
        # 特徴抽出実行
        features = self.analyzer.extract_image_features(test_image_path)
        
        # 結果検証
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 20)  # 期待される特徴量数
        self.assertTrue(np.all(np.isfinite(features)))  # 有効な数値
    
    def test_extract_image_features_invalid_path(self):
        """無効な画像パスでの特徴抽出テスト"""
        invalid_path = self.temp_path / "nonexistent.jpg"
        
        # 無効パスでの特徴抽出
        features = self.analyzer.extract_image_features(invalid_path)
        
        # デフォルト特徴量が返されることを確認
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 20)
        self.assertTrue(np.all(features == 0))  # ゼロベクトル
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.Canny')
    @patch('cv2.calcHist')
    def test_analyze_failure_patterns_no_failed_images(self, mock_calcHist, mock_Canny, mock_cvtColor, mock_imread):
        """失敗画像なしの場合のテスト"""
        # 空の失敗画像ディレクトリ
        failed_dir = self.temp_path / "failed"
        failed_dir.mkdir()
        
        # 分析実行
        results = self.analyzer.analyze_failure_patterns(failed_dir)
        
        # 結果が空であることを確認
        self.assertEqual(results, {})
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.Canny')
    @patch('cv2.calcHist')
    def test_analyze_failure_patterns_with_images(self, mock_calcHist, mock_Canny, mock_cvtColor, mock_imread):
        """失敗画像ありの場合のテスト"""
        # モック設定
        mock_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_cvtColor.return_value = mock_img[:,:,0]  # グレースケール
        mock_Canny.return_value = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        mock_calcHist.return_value = np.random.random((8, 1)) * 100
        
        # 失敗画像ディレクトリ作成
        failed_dir = self.temp_path / "failed"
        failed_dir.mkdir()
        
        # テスト画像作成
        for i in range(5):
            (failed_dir / f"failed_{i}.jpg").touch()
        
        # 分析実行
        results = self.analyzer.analyze_failure_patterns(failed_dir)
        
        # 結果検証
        self.assertIsInstance(results, dict)
        self.assertIn("timestamp", results)
        self.assertIn("total_failed_images", results)
        self.assertIn("clustering", results)
        self.assertIn("visualization", results)
        self.assertIn("anomalies", results)
        self.assertEqual(results["total_failed_images"], 5)
    
    def test_determine_pattern_type(self):
        """パターンタイプ決定のテスト"""
        # 暗い画像のパターン
        dark_features = np.zeros(20)
        dark_features[9] = 30  # brightness_idx = 9
        pattern_type = self.analyzer._determine_pattern_type(np.array([dark_features]))
        self.assertEqual(pattern_type, "dark_image")
        
        # 過露出画像のパターン
        bright_features = np.zeros(20)
        bright_features[9] = 220  # brightness_idx = 9
        pattern_type = self.analyzer._determine_pattern_type(np.array([bright_features]))
        self.assertEqual(pattern_type, "overexposed")
        
        # 極端なアスペクト比
        aspect_features = np.zeros(20)
        aspect_features[2] = 3.0  # aspect_ratio_idx = 2
        aspect_features[9] = 120  # normal brightness
        pattern_type = self.analyzer._determine_pattern_type(np.array([aspect_features]))
        self.assertEqual(pattern_type, "extreme_aspect_ratio")
    
    def test_generate_report(self):
        """レポート生成テスト"""
        # 分析結果のモックデータ
        self.analyzer.analysis_results = {
            "timestamp": "2025-08-09T19:00:00",
            "total_failed_images": 10,
            "clustering": {
                "n_clusters": 3,
                "n_noise": 2,
                "clusters": {
                    "pattern_0": {"size": 4, "percentage": 40.0},
                    "pattern_1": {"size": 3, "percentage": 30.0},
                    "noise": {"size": 2, "percentage": 20.0}
                }
            },
            "anomalies": {
                "n_anomalies": 3,
                "anomaly_rate": 30.0
            },
            "pattern_classification": {
                "cluster_0": {
                    "pattern_type": "dark_image",
                    "size": 4,
                    "characteristics": {
                        "mean_brightness": 25.0,
                        "mean_edge_density": 0.02
                    }
                }
            }
        }
        
        # レポート生成
        report_path = self.temp_path / "test_report.txt"
        report_text = self.analyzer.generate_report(report_path)
        
        # 結果検証
        self.assertIsInstance(report_text, str)
        self.assertIn("失敗パターン分析レポート", report_text)
        self.assertIn("分析対象: 10枚", report_text)
        self.assertTrue(report_path.exists())
        
        # JSONファイルも作成されることを確認
        json_path = report_path.with_suffix(".json")
        self.assertTrue(json_path.exists())
        
        # JSON内容確認
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.assertEqual(json_data["total_failed_images"], 10)


class TestFailurePatternAnalyzerIntegration(unittest.TestCase):
    """統合品質チェックシステムとの統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tools.core.unified_quality_checker.FailurePatternAnalyzer')
    def test_unified_quality_checker_integration(self, mock_analyzer_class):
        """統合品質チェッカーとの統合テスト"""
        try:
            from tools.core.unified_quality_checker import UnifiedQualityChecker
        except ImportError:
            self.skipTest("UnifiedQualityChecker が利用できません")
        
        # モック分析器設定
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_failure_patterns.return_value = {
            "total_failed_images": 5,
            "clustering": {"n_clusters": 2},
            "anomalies": {"n_anomalies": 1, "anomaly_rate": 20.0},
            "pattern_classification": {
                "cluster_0": {"pattern_type": "dark_image", "size": 3}
            }
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        # テストデータ作成
        results_file = self.temp_path / "test_results.json"
        test_data = {
            "total_images": 10,
            "success_count": 5,
            "results_directory": str(self.temp_path)
        }
        
        with open(results_file, 'w') as f:
            json.dump(test_data, f)
        
        # 失敗画像ディレクトリ作成
        failed_dir = self.temp_path / "failed"
        failed_dir.mkdir()
        (failed_dir / "failed_1.jpg").touch()
        
        # 品質チェック実行
        checker = UnifiedQualityChecker()
        with patch('tools.core.unified_quality_checker.FAILURE_ANALYSIS_AVAILABLE', True):
            report = checker.check_extraction_results(str(results_file))
        
        # パターン分析が実行されたことを確認
        self.assertIsNotNone(report)


def main():
    """テストメイン関数"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()