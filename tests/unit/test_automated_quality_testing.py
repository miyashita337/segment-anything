#!/usr/bin/env python3
"""
自動品質テストシステム単体テスト
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import pytest

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tools.core.automated_quality_testing import (
    AutomatedQualityTesting, QualityBaseline, TestResult
)


class TestAutomatedQualityTesting(unittest.TestCase):
    """自動品質テストシステムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # CI環境では一時ディレクトリを使用、ローカル環境では本来のパスを使用
        if os.getenv('CI_ENVIRONMENT') == 'true' or not os.path.exists('/mnt/c'):
            self.base_workspace_path = self.temp_path
            self.is_ci_environment = True
        else:
            self.base_workspace_path = Path("/tmp/test_workspace")
            self.is_ci_environment = False
        
        # 設定ファイル作成（CI環境対応）
        self.config_path = self.temp_path / "test_config.json"
        
        # CI環境では一時ディレクトリを、ローカルでは実際のパスを使用
        if self.is_ci_environment:
            input_path = str(self.temp_path / "input")
        else:
            input_path = str(self.base_workspace_path / "input")
            
        test_config = {
            "test_datasets": [
                {
                    "name": "test_dataset",
                    "input_path": input_path,
                    "baseline_file": "test_baseline.json",
                    "degradation_thresholds": {
                        "ab_evaluation_rate": -5.0,
                        "sci_score": -0.05,
                        "pla_score": -0.05,
                        "ple_score": -0.05,
                        "success_rate": -10.0
                    }
                }
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # システム初期化（CI環境対応）
        with patch('tools.core.automated_quality_testing.project_root', self.base_workspace_path):
            self.system = AutomatedQualityTesting(config_path=self.config_path)
            # CI環境では一時ディレクトリ配下を使用
            self.system.baseline_dir = self.base_workspace_path / "baselines"
            self.system.test_results_dir = self.base_workspace_path / "test_results" 
            self.system.workspace_dir = self.base_workspace_path / "workspace"
            
            # ディレクトリ作成（CI環境では権限問題を回避）
            for dir_path in [self.system.baseline_dir, self.system.test_results_dir, self.system.workspace_dir]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    # CI環境で権限エラーが発生した場合は一時ディレクトリにフォールバック
                    if self.is_ci_environment:
                        relative_path = dir_path.name
                        fallback_path = self.temp_path / relative_path
                        fallback_path.mkdir(parents=True, exist_ok=True)
                        # システムの参照を更新
                        if dir_path == self.system.baseline_dir:
                            self.system.baseline_dir = fallback_path
                        elif dir_path == self.system.test_results_dir:
                            self.system.test_results_dir = fallback_path
                        elif dir_path == self.system.workspace_dir:
                            self.system.workspace_dir = fallback_path
                    else:
                        raise
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def test_config_loading(self):
        """設定読み込みテスト"""
        self.assertIsNotNone(self.system.config)
        self.assertEqual(len(self.system.config["test_datasets"]), 1)
        self.assertEqual(self.system.config["test_datasets"][0]["name"], "test_dataset")
    
    def test_baseline_creation(self):
        """ベースライン作成テスト"""
        # モック設定
        with patch.object(self.system, '_run_extraction_pipeline', return_value=True), \
             patch.object(self.system, '_evaluate_quality', return_value={
                 'ab_evaluation_rate': 75.0,
                 'sci_score': 0.85,
                 'pla_score': 0.80,
                 'ple_score': 0.78,
                 'total_processed': 20,
                 'success_count': 15,
                 'failure_count': 5,
                 'avg_processing_time': 3.5,
                 'grade_distribution': {'A': 8, 'B': 7, 'C': 3, 'D': 2}
             }):
            
            baseline = self.system.create_baseline("test_dataset", force_update=True)
            
            # 検証
            self.assertIsInstance(baseline, QualityBaseline)
            self.assertEqual(baseline.dataset, "test_dataset")
            self.assertEqual(baseline.ab_evaluation_rate, 75.0)
            self.assertEqual(baseline.sci_score, 0.85)
            self.assertEqual(baseline.total_processed, 20)
            self.assertEqual(baseline.success_count, 15)
    
    def test_degradation_detection(self):
        """劣化検出テスト"""
        # ベースライン作成
        baseline = QualityBaseline(
            dataset="test",
            timestamp="2025-01-01T00:00:00",
            ab_evaluation_rate=80.0,
            sci_score=0.85,
            pla_score=0.80,
            ple_score=0.78,
            total_processed=20,
            success_count=18,
            failure_count=2,
            avg_processing_time=3.0,
            quality_grade_distribution={'A': 10, 'B': 8}
        )
        
        # 劣化したデータ
        degraded = QualityBaseline(
            dataset="test",
            timestamp="2025-01-02T00:00:00",
            ab_evaluation_rate=70.0,  # 10%低下
            sci_score=0.78,  # 0.07低下
            pla_score=0.78,  # 0.02低下（閾値内）
            ple_score=0.70,  # 0.08低下
            total_processed=20,
            success_count=14,  # 成功率低下
            failure_count=6,
            avg_processing_time=4.0,
            quality_grade_distribution={'A': 6, 'B': 8}
        )
        
        thresholds = {
            "ab_evaluation_rate": -5.0,
            "sci_score": -0.05,
            "pla_score": -0.05,
            "ple_score": -0.05,
            "success_rate": -10.0
        }
        
        # 劣化検出実行
        is_degraded, details = self.system._detect_degradation(baseline, degraded, thresholds)
        
        # 検証
        self.assertTrue(is_degraded)
        self.assertGreater(len(details), 0)
        
        # 詳細確認
        ab_degraded = any("A/B評価率低下" in detail for detail in details)
        sci_degraded = any("SCI スコア低下" in detail for detail in details)
        pla_degraded = any("PLA スコア低下" in detail for detail in details)
        ple_degraded = any("PLE スコア低下" in detail for detail in details)
        
        self.assertTrue(ab_degraded)
        self.assertTrue(sci_degraded)
        self.assertFalse(pla_degraded)  # 閾値内
        self.assertTrue(ple_degraded)
    
    def test_recommendation_generation(self):
        """推奨事項生成テスト"""
        # 劣化詳細
        degradation_details = [
            "A/B評価率低下: 80.0% → 70.0% (差分: -10.0%)",
            "SCI スコア低下: 0.85 → 0.78 (差分: -0.07)"
        ]
        
        # ダミーの現在品質
        current = QualityBaseline(
            dataset="test",
            timestamp="2025-01-02T00:00:00",
            ab_evaluation_rate=70.0,
            sci_score=0.78,
            pla_score=0.80,
            ple_score=0.78,
            total_processed=20,
            success_count=14,
            failure_count=6,
            avg_processing_time=3.0,
            quality_grade_distribution={'A': 6, 'B': 8}
        )
        
        # 推奨事項生成
        recommendation = self.system._generate_recommendation(degradation_details, current)
        
        # 検証
        self.assertIn("品質劣化が検出されました", recommendation)
        self.assertIn("YOLO検出精度の調整", recommendation)
        self.assertIn("セマンティック完全性の改善", recommendation)
    
    def test_fallback_quality_evaluation(self):
        """フォールバック品質評価テスト"""
        # テスト用出力ディレクトリ作成
        output_path = self.temp_path / "test_output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # テスト画像ファイル作成
        for i in range(5):
            (output_path / f"test_{i}.png").touch()
        
        # フォールバック評価実行
        result = self.system._fallback_quality_evaluation(output_path)
        
        # 検証
        self.assertEqual(result['success_count'], 5)
        self.assertEqual(result['total_processed'], 5)
        self.assertGreater(result['ab_evaluation_rate'], 0)
        self.assertGreater(result['sci_score'], 0)
    
    def test_test_result_serialization(self):
        """テスト結果シリアライゼーションテスト"""
        # テストデータ作成
        baseline = QualityBaseline(
            dataset="test",
            timestamp="2025-01-01T00:00:00",
            ab_evaluation_rate=80.0,
            sci_score=0.85,
            pla_score=0.80,
            ple_score=0.78,
            total_processed=20,
            success_count=18,
            failure_count=2,
            avg_processing_time=3.0,
            quality_grade_distribution={'A': 10, 'B': 8}
        )
        
        current = QualityBaseline(
            dataset="test",
            timestamp="2025-01-02T00:00:00",
            ab_evaluation_rate=75.0,
            sci_score=0.82,
            pla_score=0.78,
            ple_score=0.76,
            total_processed=20,
            success_count=16,
            failure_count=4,
            avg_processing_time=3.2,
            quality_grade_distribution={'A': 8, 'B': 8}
        )
        
        test_result = TestResult(
            test_id="test_001",
            timestamp="2025-01-02T00:00:00",
            dataset="test",
            baseline=baseline,
            current=current,
            degradation_detected=True,
            degradation_details=["A/B評価率低下: 80.0% → 75.0%"],
            recommendation="YOLO検出精度の調整を検討してください",
            status="WARNING"
        )
        
        # 保存テスト
        self.system._save_test_result(test_result)
        
        # ファイル確認
        result_file = self.system.test_results_dir / f"{test_result.test_id}.json"
        self.assertTrue(result_file.exists())
        
        # 内容確認
        with open(result_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['test_id'], "test_001")
        self.assertEqual(saved_data['status'], "WARNING")
        self.assertTrue(saved_data['degradation_detected'])
    
    @patch('subprocess.run')
    def test_notification_sending(self, mock_subprocess):
        """通知送信テスト"""
        # テスト結果作成
        test_result = TestResult(
            test_id="test_001",
            timestamp="2025-01-02T00:00:00",
            dataset="test",
            baseline=Mock(),
            current=Mock(),
            degradation_detected=True,
            degradation_details=["テスト劣化"],
            recommendation="テスト推奨事項",
            status="FAIL"
        )
        
        # 通知送信
        self.system._send_notification(test_result)
        
        # 検証
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        self.assertEqual(call_args[0], "windows-notify")
        self.assertIn("品質テスト失敗", call_args[2])


class TestQualityBaseline(unittest.TestCase):
    """品質ベースラインテスト"""
    
    def test_baseline_creation(self):
        """ベースライン作成テスト"""
        baseline = QualityBaseline(
            dataset="test",
            timestamp="2025-01-01T00:00:00",
            ab_evaluation_rate=80.0,
            sci_score=0.85,
            pla_score=0.80,
            ple_score=0.78,
            total_processed=20,
            success_count=18,
            failure_count=2,
            avg_processing_time=3.0,
            quality_grade_distribution={'A': 10, 'B': 8}
        )
        
        self.assertEqual(baseline.dataset, "test")
        self.assertEqual(baseline.ab_evaluation_rate, 80.0)
        self.assertEqual(baseline.total_processed, 20)
        self.assertEqual(baseline.success_count, 18)


class TestTestResult(unittest.TestCase):
    """テスト結果テスト"""
    
    def test_result_creation(self):
        """結果作成テスト"""
        baseline = Mock()
        current = Mock()
        
        result = TestResult(
            test_id="test_001",
            timestamp="2025-01-01T00:00:00",
            dataset="test",
            baseline=baseline,
            current=current,
            degradation_detected=True,
            degradation_details=["テスト劣化"],
            recommendation="テスト推奨事項",
            status="WARNING"
        )
        
        self.assertEqual(result.test_id, "test_001")
        self.assertEqual(result.status, "WARNING")
        self.assertTrue(result.degradation_detected)


def main():
    """テスト実行"""
    # テストスイート作成
    test_suite = unittest.TestSuite()
    
    # テストケース追加
    test_classes = [
        TestAutomatedQualityTesting,
        TestQualityBaseline,
        TestTestResult
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果表示
    print(f"\n{'='*60}")
    print("自動品質テストシステム 単体テスト結果")
    print(f"{'='*60}")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n失敗詳細:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nエラー詳細:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)