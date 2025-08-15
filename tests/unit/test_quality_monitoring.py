#!/usr/bin/env python3
"""
P1-B001: 統合品質チェック自動実行システム単体テスト
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# プロジェクトルート設定
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.common.quality_monitoring import (
    IntegratedQualityMonitor,
    QualityMonitoringConfig,
    QualityResult,
    run_integrated_quality_check
)


class TestQualityMonitoringConfig(unittest.TestCase):
    """品質監視設定テスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = QualityMonitoringConfig()
        
        self.assertTrue(config.enabled)
        self.assertTrue(config.auto_check_after_extraction)
        self.assertEqual(config.quality_threshold, 0.7)
        self.assertTrue(config.alert_on_degradation)
        self.assertEqual(config.degradation_threshold, 0.1)
        self.assertTrue(config.dashboard_generation)
        self.assertTrue(config.notification_enabled)
        self.assertEqual(config.baseline_update_frequency, 10)
    
    def test_config_serialization(self):
        """設定シリアライゼーションテスト"""
        config = QualityMonitoringConfig(
            enabled=True,
            quality_threshold=0.8,
            degradation_threshold=0.05
        )
        
        # to_dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict['quality_threshold'], 0.8)
        self.assertEqual(config_dict['degradation_threshold'], 0.05)
        
        # from_dict
        restored_config = QualityMonitoringConfig.from_dict(config_dict)
        self.assertEqual(restored_config.quality_threshold, 0.8)
        self.assertEqual(restored_config.degradation_threshold, 0.05)


class TestQualityResult(unittest.TestCase):
    """品質結果テスト"""
    
    def test_quality_result_creation(self):
        """品質結果作成テスト"""
        result = QualityResult(
            timestamp="2025-08-02T15:30:00",
            success_rate=0.85,
            avg_quality_score=0.75,
            total_processed=20,
            quality_grades={"A": 10, "B": 7, "C": 2, "D": 1}
        )
        
        self.assertEqual(result.success_rate, 0.85)
        self.assertEqual(result.avg_quality_score, 0.75)
        self.assertEqual(result.total_processed, 20)
        self.assertEqual(result.quality_grades["A"], 10)
        self.assertFalse(result.degradation_detected)
    
    def test_quality_result_serialization(self):
        """品質結果シリアライゼーションテスト"""
        result = QualityResult(
            timestamp="2025-08-02T15:30:00",
            success_rate=0.85,
            avg_quality_score=0.75,
            total_processed=20,
            quality_grades={"A": 10, "B": 7, "C": 2, "D": 1}
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict['success_rate'], 0.85)
        self.assertEqual(result_dict['total_processed'], 20)
        self.assertIn('quality_grades', result_dict)


class TestIntegratedQualityMonitor(unittest.TestCase):
    """統合品質監視システムテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_path = self.temp_dir / "workspace"
        self.workspace_path.mkdir(parents=True)
        
        # テスト用画像ファイル作成（空ファイル）
        self.extraction_dir = self.workspace_path / "extraction"
        self.extraction_dir.mkdir()
        
        for i in range(5):
            test_image = self.extraction_dir / f"test_{i}.jpg"
            test_image.write_text("fake image data")
    
    def tearDown(self):
        """テスト後処理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_initialization(self):
        """監視システム初期化テスト"""
        config = QualityMonitoringConfig()
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        self.assertEqual(monitor.workspace_path, self.workspace_path)
        self.assertEqual(monitor.config.quality_threshold, 0.7)
        self.assertTrue(monitor.quality_history_file.name == "quality_history.json")
        self.assertTrue(monitor.baseline_file.name == "quality_baseline.json")
    
    def test_quality_check_with_mock(self):
        """品質チェック実行テスト（モック使用）"""
        # UnifiedQualityCheckerのモック設定
        mock_checker = MagicMock()
        mock_checker.evaluate_extracted_image.return_value = {
            'overall_score': 0.8,
            'grade': 'B'
        }
        
        config = QualityMonitoringConfig()
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        monitor.quality_checker = mock_checker
        
        # 品質チェック実行
        result = monitor.run_quality_check(self.extraction_dir)
        
        # 結果検証
        self.assertIsInstance(result, QualityResult)
        self.assertEqual(result.total_processed, 5)
        self.assertGreater(result.avg_quality_score, 0)
        self.assertEqual(result.quality_grades["B"], 5)
        
        # メソッドが正しく呼ばれたか確認
        self.assertEqual(mock_checker.evaluate_extracted_image.call_count, 5)
    
    def test_simple_quality_check(self):
        """簡易品質チェックテスト"""
        config = QualityMonitoringConfig()
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        # UnifiedQualityCheckerが利用できない状況をシミュレート
        monitor.quality_checker = None
        
        # 実際のPIL画像ファイル作成
        from PIL import Image
        test_image_path = self.extraction_dir / "real_test.jpg"
        
        # 200x300の白い画像作成
        img = Image.new('RGB', (200, 300), color='white')
        img.save(test_image_path)
        
        # 簡易品質チェック実行
        score = monitor._simple_quality_check(test_image_path)
        
        # 基本的な品質スコアが返されることを確認
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_score_to_grade_conversion(self):
        """スコアからグレード変換テスト"""
        config = QualityMonitoringConfig()
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        # 各スコア範囲のテスト
        self.assertEqual(monitor._score_to_grade(0.95), "A")
        self.assertEqual(monitor._score_to_grade(0.85), "B")
        self.assertEqual(monitor._score_to_grade(0.75), "C")
        self.assertEqual(monitor._score_to_grade(0.65), "D")
        self.assertEqual(monitor._score_to_grade(0.55), "E")
        self.assertEqual(monitor._score_to_grade(0.35), "F")
    
    def test_degradation_detection(self):
        """品質劣化検出テスト"""
        config = QualityMonitoringConfig(degradation_threshold=0.1)
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        # ベースライン作成
        baseline = {
            'success_rate': 0.8,
            'avg_quality_score': 0.75
        }
        
        with open(monitor.baseline_file, 'w') as f:
            json.dump(baseline, f)
        
        # 現在の結果（劣化あり）
        current = QualityResult(
            timestamp="2025-08-02T15:30:00",
            success_rate=0.65,  # 0.15ポイント低下
            avg_quality_score=0.60,  # 0.15ポイント低下
            total_processed=10,
            quality_grades={"B": 6, "C": 4}
        )
        
        # 劣化検出実行
        degradation_detected, comparison = monitor._check_degradation(current)
        
        self.assertTrue(degradation_detected)
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison['baseline_success_rate'], 0.8)
        self.assertEqual(comparison['current_success_rate'], 0.65)
    
    def test_quality_history_management(self):
        """品質履歴管理テスト"""
        config = QualityMonitoringConfig()
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        # テスト結果作成
        result = QualityResult(
            timestamp="2025-08-02T15:30:00",
            success_rate=0.8,
            avg_quality_score=0.75,
            total_processed=10,
            quality_grades={"A": 5, "B": 3, "C": 2}
        )
        
        # 履歴保存
        monitor._save_quality_history(result)
        
        # 履歴ファイル確認
        self.assertTrue(monitor.quality_history_file.exists())
        
        with open(monitor.quality_history_file, 'r') as f:
            history = json.load(f)
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['success_rate'], 0.8)
        self.assertEqual(history[0]['total_processed'], 10)
    
    def test_baseline_update(self):
        """ベースライン更新テスト"""
        config = QualityMonitoringConfig(baseline_update_frequency=3)
        monitor = IntegratedQualityMonitor(self.workspace_path, config)
        
        # 3つのテスト結果作成
        results = [
            {'success_rate': 0.8, 'avg_quality_score': 0.75},
            {'success_rate': 0.85, 'avg_quality_score': 0.78},
            {'success_rate': 0.82, 'avg_quality_score': 0.76}
        ]
        
        # 履歴保存
        with open(monitor.quality_history_file, 'w') as f:
            json.dump(results, f)
        
        # 新しい結果でベースライン更新をトリガー
        new_result = QualityResult(
            timestamp="2025-08-02T15:30:00",
            success_rate=0.83,
            avg_quality_score=0.77,
            total_processed=10,
            quality_grades={"A": 5, "B": 5}
        )
        
        monitor._update_baseline_if_needed(new_result)
        
        # ベースラインファイル確認
        self.assertTrue(monitor.baseline_file.exists())
        
        with open(monitor.baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # 平均値が正しく計算されているか確認
        expected_success_rate = (0.8 + 0.85 + 0.82) / 3
        expected_avg_score = (0.75 + 0.78 + 0.76) / 3
        
        self.assertAlmostEqual(baseline['success_rate'], expected_success_rate, places=3)
        self.assertAlmostEqual(baseline['avg_quality_score'], expected_avg_score, places=3)


class TestQualityMonitoringIntegration(unittest.TestCase):
    """品質監視統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_path = self.temp_dir / "workspace"
        self.workspace_path.mkdir(parents=True)
    
    def tearDown(self):
        """テスト後処理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integrated_quality_check_function(self):
        """統合品質チェック関数テスト"""
        # テスト用画像ファイル作成
        extraction_dir = self.workspace_path / "extraction"
        extraction_dir.mkdir()
        
        from PIL import Image
        for i in range(3):
            test_image = extraction_dir / f"test_{i}.jpg"
            img = Image.new('RGB', (150, 200), color='white')
            img.save(test_image)
        
        # 統合品質チェック実行
        result = run_integrated_quality_check(self.workspace_path, "P1-B001")
        
        # 結果検証
        self.assertIsNotNone(result)
        self.assertIsInstance(result, QualityResult)
        self.assertEqual(result.total_processed, 3)
        self.assertGreaterEqual(result.success_rate, 0.0)
        self.assertLessEqual(result.success_rate, 1.0)
    
    def test_integrated_quality_check_empty_directory(self):
        """空ディレクトリでの統合品質チェックテスト"""
        # 空の抽出ディレクトリ
        extraction_dir = self.workspace_path / "extraction"
        extraction_dir.mkdir()
        
        # 統合品質チェック実行
        result = run_integrated_quality_check(self.workspace_path, "P1-B001")
        
        # 結果検証
        self.assertIsNotNone(result)
        self.assertEqual(result.total_processed, 0)
        self.assertEqual(result.success_rate, 0.0)
    
    def test_integrated_quality_check_no_extraction_dir(self):
        """抽出ディレクトリなしでの統合品質チェックテスト"""
        # ワークスペース直下に画像ファイル配置
        from PIL import Image
        test_image = self.workspace_path / "test.jpg"
        img = Image.new('RGB', (150, 200), color='white')
        img.save(test_image)
        
        # 統合品質チェック実行（extractionディレクトリなし）
        result = run_integrated_quality_check(self.workspace_path, "P1-B001")
        
        # 結果検証
        self.assertIsNotNone(result)
        self.assertEqual(result.total_processed, 1)


if __name__ == '__main__':
    unittest.main()