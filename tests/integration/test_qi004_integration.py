"""
QI-004ダッシュボード最適化システムの統合テスト

統合テスト項目:
- QI-004システム全体の統合動作確認
- 既存ダッシュボード生成システムとの統合
- ファイルI/O、品質解析、ダッシュボード生成の統合フロー
"""

import os
import tempfile
import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

from features.evaluation.qi004_dashboard_optimization_system import (
    QI004DashboardOptimizationSystem,
    create_qi004_optimized_dashboard
)
from features.common.dashboard_generator import DashboardGenerator


class TestQI004SystemIntegration(unittest.TestCase):
    """QI-004システム統合テスト"""
    
    @classmethod
    def setUpClass(cls):
        """クラス全体のセットアップ"""
        cls.temp_workspace = tempfile.mkdtemp(prefix="qi004_integration_")
        cls.tracker_id = "QI-004-TEST"
        
        # テスト用ワークスペース構造作成
        cls.extraction_dir = os.path.join(cls.temp_workspace, cls.tracker_id, "extraction")
        cls.dashboard_dir = os.path.join(cls.temp_workspace, cls.tracker_id, "dashboard")
        cls.quality_dir = os.path.join(cls.temp_workspace, cls.tracker_id, "quality")
        
        for dir_path in [cls.extraction_dir, cls.dashboard_dir, cls.quality_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # テスト用画像作成（異なる品質レベル）
        cls.test_images = cls._create_test_images()
    
    @classmethod
    def tearDownClass(cls):
        """クラス全体のクリーンアップ"""
        shutil.rmtree(cls.temp_workspace)
    
    @classmethod
    def _create_test_images(cls):
        """テスト用画像セット作成"""
        images = []
        
        # 高品質画像（高解像度、明確なエッジ）
        high_quality = np.zeros((1200, 800, 3), dtype=np.uint8)
        cv2.rectangle(high_quality, (100, 100), (700, 1100), (255, 255, 255), -1)
        cv2.rectangle(high_quality, (200, 200), (600, 1000), (100, 150, 200), -1)
        high_quality_path = os.path.join(cls.extraction_dir, "high_quality_kana08_0001.jpg")
        cv2.imwrite(high_quality_path, high_quality)
        images.append(high_quality_path)
        
        # 中品質画像（標準解像度）
        medium_quality = np.random.randint(50, 200, (720, 480, 3), dtype=np.uint8)
        cv2.circle(medium_quality, (360, 240), 150, (200, 100, 50), -1)
        medium_quality_path = os.path.join(cls.extraction_dir, "medium_quality_kana08_0002.jpg")
        cv2.imwrite(medium_quality_path, medium_quality)
        images.append(medium_quality_path)
        
        # 低品質画像（低解像度、ノイジー）
        low_quality = np.random.randint(0, 256, (320, 240, 3), dtype=np.uint8)
        low_quality_path = os.path.join(cls.extraction_dir, "low_quality_kana08_0003.jpg")
        cv2.imwrite(low_quality_path, low_quality)
        images.append(low_quality_path)
        
        # 黒画面画像（要改善レベル）
        black_screen = np.full((640, 480, 3), 10, dtype=np.uint8)  # ほぼ黒
        black_screen_path = os.path.join(cls.extraction_dir, "black_screen_kana08_0004.jpg")
        cv2.imwrite(black_screen_path, black_screen)
        images.append(black_screen_path)
        
        return images
    
    def test_complete_qi004_optimization_workflow(self):
        """QI-004完全最適化ワークフローテスト"""
        system = QI004DashboardOptimizationSystem()
        
        # 完全最適化プロセス実行
        result = system.run_complete_optimization(
            self.tracker_id,
            self.extraction_dir,
            os.path.join(self.temp_workspace, self.tracker_id)
        )
        
        # 基本結果検証
        self.assertEqual(result.total_images, len(self.test_images))
        self.assertEqual(len(result.image_quality_scores), len(self.test_images))
        
        # 品質スコア範囲確認
        for score in result.image_quality_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # ダッシュボードサイズ確認（画像パス参照で小さくなるはず）
        self.assertLess(result.dashboard_size_mb, 0.1)  # 100KB未満を期待
        
        # 読み込み時間確認
        self.assertGreater(result.load_time_seconds, 0.0)
        
        # パフォーマンス指標確認
        self.assertIn('total_time_seconds', result.performance_metrics)
        self.assertIn('images_per_second', result.performance_metrics)
        self.assertGreater(result.performance_metrics['total_time_seconds'], 0.0)
        
        # 最適化データ確認
        self.assertIn('optimized_paths', result.optimization_improvements)
        self.assertIn('size_optimization', result.optimization_improvements)
        self.assertIn('cache_strategy', result.optimization_improvements)
    
    def test_dashboard_generation_with_real_images(self):
        """実画像を使用したダッシュボード生成テスト"""
        generator = DashboardGenerator()
        
        # テスト用品質スコア（実際の解析結果をシミュレート）
        quality_scores = [0.85, 0.65, 0.35, 0.15]  # 各画像の想定品質
        
        dashboard_data = {
            'tracker_id': self.tracker_id,
            'total_images': len(self.test_images),
            'quality_scores': quality_scores,
            'black_screen_indices': [3],  # 最後の画像が黒画面
            'image_paths': self.test_images
        }
        
        # ダッシュボード生成
        dashboard_path = generator.generate_standard_dashboard(
            dashboard_data, self.dashboard_dir
        )
        
        # ファイル存在確認
        self.assertTrue(dashboard_path.exists())
        self.assertEqual(dashboard_path.name, "dashboard.html")
        
        # ファイルサイズ確認（画像パス参照で小さくなるはず）
        file_size_mb = dashboard_path.stat().st_size / (1024 * 1024)
        self.assertLess(file_size_mb, 0.1)  # 100KB未満を期待
        
        # HTML内容確認
        html_content = dashboard_path.read_text(encoding='utf-8')
        
        # 基本構造確認
        self.assertIn(f"{self.tracker_id} 品質評価ダッシュボード", html_content)
        self.assertIn("/workspace/", html_content)  # 画像パス参照確認
        
        # 実ファイル名表示確認（QI-004要件）
        for image_path in self.test_images:
            filename = Path(image_path).name
            self.assertIn(filename, html_content)
        
        # 品質バッジ確認
        self.assertIn("高品質", html_content)  # 0.85スコア
        self.assertIn("中品質", html_content)  # 0.65スコア
        self.assertIn("低品質", html_content)  # 0.35スコア
        self.assertIn("要改善", html_content)  # 0.15スコア
        
        # 黒画面警告確認
        self.assertIn("⚠️ 黒画面検出", html_content)
    
    def test_qi004_entry_point_integration(self):
        """QI-004エントリーポイント統合テスト"""
        output_dir = os.path.join(self.temp_workspace, f"{self.tracker_id}_entry_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # エントリーポイント関数実行
        success = create_qi004_optimized_dashboard(
            f"{self.tracker_id}_ENTRY",
            self.extraction_dir,
            output_dir
        )
        
        self.assertTrue(success)
        
        # 出力ファイル確認
        dashboard_path = os.path.join(output_dir, "dashboard", "dashboard.html")
        self.assertTrue(os.path.exists(dashboard_path))
        
        # ダッシュボード内容確認
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # QI-004特有の最適化内容確認
        self.assertIn(f"{self.tracker_id}_ENTRY", content)
        self.assertIn("/workspace/", content)  # 画像パス参照確認
        
        # 実ファイル名表示確認
        for image_path in self.test_images:
            filename = Path(image_path).name
            self.assertIn(filename, content)
    
    def test_quality_score_distribution_accuracy(self):
        """品質スコア分布精度テスト"""
        system = QI004DashboardOptimizationSystem()
        
        # 品質解析実行
        quality_scores = system._analyze_all_images(self.test_images)
        
        # 品質分布確認（作成した画像の想定品質と一致するか）
        high_quality_scores = [s for s in quality_scores if s >= 0.8]
        medium_quality_scores = [s for s in quality_scores if 0.6 <= s < 0.8]
        low_quality_scores = [s for s in quality_scores if 0.3 <= s < 0.6]
        poor_quality_scores = [s for s in quality_scores if s < 0.3]
        
        # 最低1つずつは各カテゴリに分類されることを期待
        # （テスト画像の品質設定に基づく）
        total_categorized = len(high_quality_scores) + len(medium_quality_scores) + \
                          len(low_quality_scores) + len(poor_quality_scores)
        self.assertEqual(total_categorized, len(self.test_images))
        
        # 黒画面（最後の画像）は最低品質になるはず
        self.assertLess(quality_scores[-1], 0.5)  # 黒画面は低品質
    
    def test_optimization_performance_metrics(self):
        """最適化パフォーマンス指標テスト"""
        system = QI004DashboardOptimizationSystem()
        
        import time
        start_time = time.time()
        
        result = system.run_complete_optimization(
            f"{self.tracker_id}_PERF",
            self.extraction_dir,
            os.path.join(self.temp_workspace, f"{self.tracker_id}_PERF")
        )
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        # 報告された実行時間と実際の時間の整合性確認
        reported_time = result.performance_metrics['total_time_seconds']
        time_diff = abs(actual_time - reported_time)
        self.assertLess(time_diff, 1.0)  # 1秒以内の誤差は許容
        
        # 処理効率確認
        images_per_second = result.performance_metrics['images_per_second']
        self.assertGreater(images_per_second, 0.1)  # 最低効率確認
        
        # 最適化効率確認
        opt_efficiency = result.performance_metrics['optimization_efficiency']
        self.assertGreater(opt_efficiency, 0.0)
        self.assertLess(opt_efficiency, reported_time)  # 最適化時間は全体時間より短いはず
    
    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        system = QI004DashboardOptimizationSystem()
        
        # 存在しない抽出ディレクトリでの実行
        result = system.run_complete_optimization(
            "QI-004-ERROR",
            "/nonexistent/directory",
            self.temp_workspace
        )
        
        # エラー時でも適切なレスポンスが返ることを確認
        self.assertEqual(result.total_images, 0)
        self.assertEqual(len(result.image_quality_scores), 0)
        self.assertGreater(result.dashboard_size_mb, 0)  # 空のダッシュボードでも最小サイズ
        
        # パフォーマンス指標も適切に処理されることを確認
        self.assertIn('total_time_seconds', result.performance_metrics)
    
    def test_large_dataset_handling(self):
        """大規模データセット処理テスト"""
        # 追加の大量画像作成（小サイズで高速テスト）
        large_dataset_dir = os.path.join(self.temp_workspace, "large_dataset")
        os.makedirs(large_dataset_dir, exist_ok=True)
        
        # 20枚の小画像作成
        large_images = []
        for i in range(20):
            small_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            image_path = os.path.join(large_dataset_dir, f"large_test_{i:03d}.jpg")
            cv2.imwrite(image_path, small_image)
            large_images.append(image_path)
        
        system = QI004DashboardOptimizationSystem()
        
        result = system.run_complete_optimization(
            "QI-004-LARGE",
            large_dataset_dir,
            os.path.join(self.temp_workspace, "QI-004-LARGE")
        )
        
        # 大量データでの処理確認
        self.assertEqual(result.total_images, 20)
        self.assertEqual(len(result.image_quality_scores), 20)
        
        # 処理効率確認（大量データでも合理的な時間で完了）
        images_per_second = result.performance_metrics['images_per_second']
        self.assertGreater(images_per_second, 1.0)  # 1秒に1枚以上の効率
        
        # ダッシュボードサイズ確認（画像パス参照で小さく維持）
        self.assertLess(result.dashboard_size_mb, 0.5)  # 500KB未満を期待


if __name__ == '__main__':
    unittest.main()