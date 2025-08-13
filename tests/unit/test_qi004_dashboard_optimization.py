"""
QI-004ダッシュボード最適化システムの単体テスト

テスト項目:
- ImageQualityAnalyzer画像品質解析機能
- DashboardOptimizer最適化機能
- QI004DashboardOptimizationSystem統合機能
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import cv2
import numpy as np

from features.evaluation.qi004_dashboard_optimization_system import (
    ImageQualityAnalyzer,
    DashboardOptimizer,
    QI004DashboardOptimizationSystem,
    QI004OptimizationResult,
    create_qi004_optimized_dashboard
)


class TestImageQualityAnalyzer(unittest.TestCase):
    """ImageQualityAnalyzer単体テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.analyzer = ImageQualityAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        
        # テスト用画像作成
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        test_image = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_image_quality_success(self):
        """画像品質解析成功テスト"""
        result = self.analyzer.analyze_image_quality(self.test_image_path)
        
        # 基本フィールドの存在確認
        required_fields = [
            'overall_score', 'resolution_score', 'aspect_score',
            'background_quality', 'crop_precision', 'dimensions',
            'aspect_ratio', 'file_size_mb'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # スコア範囲確認（0-1の範囲内）
        self.assertGreaterEqual(result['overall_score'], 0.0)
        self.assertLessEqual(result['overall_score'], 1.0)
        
        # 次元確認
        self.assertEqual(len(result['dimensions']), 2)
        self.assertGreater(result['dimensions'][0], 0)
        self.assertGreater(result['dimensions'][1], 0)
    
    def test_analyze_image_quality_file_not_found(self):
        """存在しないファイルのテスト"""
        result = self.analyzer.analyze_image_quality("nonexistent.jpg")
        
        # エラー結果の確認
        self.assertEqual(result['overall_score'], 0.0)
        self.assertIn('error', result)
    
    def test_calculate_resolution_score(self):
        """解像度スコア計算テスト"""
        # フルHD以上
        score = self.analyzer._calculate_resolution_score(1920, 1080)
        self.assertEqual(score, 1.0)
        
        # HD
        score = self.analyzer._calculate_resolution_score(1280, 720)
        self.assertEqual(score, 0.8)
        
        # VGA
        score = self.analyzer._calculate_resolution_score(640, 480)
        self.assertEqual(score, 0.6)
        
        # 低解像度
        score = self.analyzer._calculate_resolution_score(320, 240)
        self.assertEqual(score, 0.3)
    
    def test_calculate_aspect_score(self):
        """アスペクト比スコア計算テスト"""
        # 標準アスペクト比（4:3）
        score = self.analyzer._calculate_aspect_score(4/3)
        self.assertGreater(score, 0.9)
        
        # 横長（16:9）
        score = self.analyzer._calculate_aspect_score(16/9)
        self.assertGreater(score, 0.8)
        
        # 正方形（1:1）
        score = self.analyzer._calculate_aspect_score(1.0)
        self.assertGreater(score, 0.7)
    
    def test_evaluate_background_removal(self):
        """背景除去品質評価テスト"""
        # テスト用画像（均一な背景）
        uniform_bg = np.full((400, 400, 3), 128, dtype=np.uint8)
        score = self.analyzer._evaluate_background_removal(uniform_bg)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_evaluate_crop_precision(self):
        """切り抜き精度評価テスト"""
        # テスト用画像
        test_image = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
        score = self.analyzer._evaluate_crop_precision(test_image)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestDashboardOptimizer(unittest.TestCase):
    """DashboardOptimizer単体テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.optimizer = DashboardOptimizer()
        self.temp_dir = tempfile.mkdtemp()
        
        # テスト用画像ファイル作成
        self.test_images = []
        for i in range(3):
            image_path = os.path.join(self.temp_dir, f"test_{i}.jpg")
            test_image = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
            cv2.imwrite(image_path, test_image)
            self.test_images.append(image_path)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_optimize_dashboard_performance(self):
        """ダッシュボードパフォーマンス最適化テスト"""
        result = self.optimizer.optimize_dashboard_performance(
            self.test_images, self.temp_dir
        )
        
        # 必須フィールドの確認
        required_fields = [
            'optimized_paths', 'size_optimization', 'cache_strategy',
            'responsive_optimization', 'optimization_time_seconds',
            'total_images_processed'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # 処理画像数の確認
        self.assertEqual(result['total_images_processed'], len(self.test_images))
        
        # 最適化時間の確認（正の値）
        self.assertGreater(result['optimization_time_seconds'], 0.0)
    
    def test_optimize_image_paths(self):
        """画像パス最適化テスト"""
        workspace_paths = [
            "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-004/extraction/test1.jpg",
            "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-004/extraction/test2.jpg"
        ]
        
        result = self.optimizer._optimize_image_paths(workspace_paths, self.temp_dir)
        
        # 相対パス変換の確認
        self.assertEqual(len(result), 2)
        for path in result:
            self.assertTrue(path.startswith('/'))
    
    def test_optimize_image_sizes(self):
        """画像サイズ最適化テスト"""
        result = self.optimizer._optimize_image_sizes(self.test_images)
        
        # 必須フィールドの確認
        required_fields = [
            'total_size_mb', 'average_size_mb', 'size_distribution',
            'recommended_optimization'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # サイズ分布の確認
        distribution = result['size_distribution']
        total_count = sum(distribution.values())
        self.assertEqual(total_count, len(self.test_images))
    
    def test_optimize_cache_strategy(self):
        """キャッシュ戦略最適化テスト"""
        result = self.optimizer._optimize_cache_strategy(self.test_images)
        
        # キャッシュ設定の確認
        self.assertTrue(result['cache_enabled'])
        self.assertEqual(result['cache_duration_hours'], 24)
        self.assertEqual(result['preload_strategy'], 'lazy')
        self.assertTrue(result['compression_enabled'])
        self.assertGreater(result['estimated_cache_size_mb'], 0)
    
    def test_optimize_responsive_design(self):
        """レスポンシブデザイン最適化テスト"""
        result = self.optimizer._optimize_responsive_design()
        
        # レスポンシブ設定の確認
        self.assertIn('breakpoints', result)
        self.assertIn('grid_columns', result)
        self.assertEqual(result['image_sizing'], 'object-contain')
        self.assertEqual(result['max_height'], '400px')


class TestQI004DashboardOptimizationSystem(unittest.TestCase):
    """QI004DashboardOptimizationSystem統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.extraction_dir = os.path.join(self.temp_dir, "extraction")
        os.makedirs(self.extraction_dir)
        
        # テスト用抽出画像作成
        self.test_images = []
        for i in range(2):
            image_path = os.path.join(self.extraction_dir, f"extracted_{i}.jpg")
            test_image = np.random.randint(0, 256, (600, 400, 3), dtype=np.uint8)
            cv2.imwrite(image_path, test_image)
            self.test_images.append(image_path)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('features.evaluation.qi004_dashboard_optimization_system.PushoverImageSender')
    @patch('features.evaluation.qi004_dashboard_optimization_system.StandardDashboardGenerator')
    def test_run_complete_optimization(self, mock_dashboard_gen, mock_pushover):
        """完全最適化プロセステスト"""
        # モック設定
        mock_dashboard_gen.return_value.generate_standard_dashboard.return_value = Path("test_dashboard.html")
        mock_pushover.return_value.send_extraction_complete_with_images.return_value = None
        
        # 実際のダッシュボードファイル作成（サイズ測定用）
        dashboard_path = Path(self.temp_dir) / "dashboard" / "dashboard.html"
        dashboard_path.parent.mkdir(exist_ok=True)
        dashboard_path.write_text("<html><body>Test Dashboard</body></html>")
        
        # モックの戻り値を実際のファイルに変更
        mock_dashboard_gen.return_value.generate_standard_dashboard.return_value = dashboard_path
        
        system = QI004DashboardOptimizationSystem()
        result = system.run_complete_optimization(
            "QI-004", self.extraction_dir, self.temp_dir
        )
        
        # 結果の型確認
        self.assertIsInstance(result, QI004OptimizationResult)
        
        # 基本統計の確認
        self.assertEqual(result.total_images, len(self.test_images))
        self.assertEqual(len(result.image_quality_scores), len(self.test_images))
        self.assertGreater(result.dashboard_size_mb, 0)
        self.assertGreater(result.load_time_seconds, 0)
        
        # パフォーマンス指標の確認
        self.assertIn('total_time_seconds', result.performance_metrics)
        self.assertIn('images_per_second', result.performance_metrics)
        self.assertIn('optimization_efficiency', result.performance_metrics)
    
    def test_collect_extracted_images(self):
        """抽出画像収集テスト"""
        system = QI004DashboardOptimizationSystem()
        
        # 正常ケース
        images = system._collect_extracted_images(self.extraction_dir)
        self.assertEqual(len(images), len(self.test_images))
        
        # 存在しないディレクトリ
        images = system._collect_extracted_images("/nonexistent")
        self.assertEqual(len(images), 0)
    
    def test_analyze_all_images(self):
        """全画像品質解析テスト"""
        system = QI004DashboardOptimizationSystem()
        quality_scores = system._analyze_all_images(self.test_images)
        
        self.assertEqual(len(quality_scores), len(self.test_images))
        for score in quality_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_measure_dashboard_load_time(self):
        """ダッシュボード読み込み時間測定テスト"""
        system = QI004DashboardOptimizationSystem()
        
        # テスト用HTMLファイル作成
        test_html = os.path.join(self.temp_dir, "test_dashboard.html")
        with open(test_html, 'w', encoding='utf-8') as f:
            f.write("<html><body>Test</body></html>")
        
        load_time = system._measure_dashboard_load_time(test_html)
        self.assertGreater(load_time, 0.0)
        
        # 存在しないファイル
        load_time = system._measure_dashboard_load_time("/nonexistent.html")
        self.assertEqual(load_time, 1.0)  # デフォルト値


class TestQI004EntryPoint(unittest.TestCase):
    """QI-004エントリーポイント関数テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.extraction_dir = os.path.join(self.temp_dir, "extraction")
        os.makedirs(self.extraction_dir)
        
        # テスト用画像作成
        image_path = os.path.join(self.extraction_dir, "test.jpg")
        test_image = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
        cv2.imwrite(image_path, test_image)
    
    def tearDown(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('features.evaluation.qi004_dashboard_optimization_system.QI004DashboardOptimizationSystem')
    def test_create_qi004_optimized_dashboard_success(self, mock_system_class):
        """QI-004最適化ダッシュボード生成成功テスト"""
        # モック設定
        mock_system = Mock()
        mock_result = QI004OptimizationResult(
            total_images=1,
            image_quality_scores=[0.8],
            dashboard_size_mb=2.5,
            load_time_seconds=0.5,
            optimization_improvements={},
            image_path_references=["/test.jpg"],
            performance_metrics={'total_time_seconds': 1.0}
        )
        mock_system.run_complete_optimization.return_value = mock_result
        mock_system_class.return_value = mock_system
        
        result = create_qi004_optimized_dashboard(
            "QI-004", self.extraction_dir, self.temp_dir
        )
        
        self.assertTrue(result)
        mock_system.run_complete_optimization.assert_called_once_with(
            "QI-004", self.extraction_dir, self.temp_dir
        )
    
    @patch('features.evaluation.qi004_dashboard_optimization_system.QI004DashboardOptimizationSystem')
    def test_create_qi004_optimized_dashboard_failure(self, mock_system_class):
        """QI-004最適化ダッシュボード生成失敗テスト"""
        # モック設定（例外発生）
        mock_system_class.side_effect = Exception("テストエラー")
        
        result = create_qi004_optimized_dashboard(
            "QI-004", self.extraction_dir, self.temp_dir
        )
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()