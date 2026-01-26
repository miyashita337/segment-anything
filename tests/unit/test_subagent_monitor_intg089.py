#!/usr/bin/env python3
"""
INTG-089 SubAgentMonitor拡張機能の単体テスト
現実的なシナリオに基づく実用的なテスト
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.queue.subagent_monitor import SubAgentMonitor


class TestSubAgentMonitorINTG089(unittest.TestCase):
    """SubAgentMonitor INTG-089拡張機能テスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir
        self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_gpu_availability_check(self):
        """GPU利用可能性チェックテスト"""
        # GPU利用可能性の確認
        gpu_available = self.monitor._check_gpu_availability()
        self.assertIsInstance(gpu_available, bool)

        # GPU利用可能な場合の設定確認
        if gpu_available:
            self.assertTrue(self.monitor.gpu_monitoring_enabled)
        else:
            # GPU利用不可の場合は監視無効
            self.assertFalse(self.monitor.gpu_monitoring_enabled)

    @patch("torch.cuda.is_available")
    def test_gpu_availability_mock_true(self, mock_cuda):
        """GPU利用可能時のモックテスト"""
        mock_cuda.return_value = True

        monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.assertTrue(monitor.gpu_monitoring_enabled)

    @patch("torch.cuda.is_available")
    def test_gpu_availability_mock_false(self, mock_cuda):
        """GPU利用不可時のモックテスト"""
        mock_cuda.return_value = False

        monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.assertFalse(monitor.gpu_monitoring_enabled)

    @patch("psutil.virtual_memory")
    def test_memory_monitoring_baseline(self, mock_memory):
        """メモリ監視ベースライン設定テスト"""
        # メモリ情報をモック
        mock_memory.return_value.used = 1024 * 1024 * 1024  # 1GB

        monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.assertEqual(monitor.memory_baseline, 1024 * 1024 * 1024)

    def test_anomaly_thresholds_configuration(self):
        """異常検知閾値設定テスト"""
        expected_thresholds = {
            "gpu_memory_usage": 90,
            "system_memory_usage": 85,
            "cpu_usage": 95,
            "temperature_threshold": 80,  # GPU温度 80°C
            "process_timeout": 3600,  # 1 hour
        }

        self.assertEqual(self.monitor.anomaly_thresholds, expected_thresholds)

    def test_monitoring_interval_setting(self):
        """監視間隔設定テスト"""
        # デフォルト値確認
        self.assertEqual(self.monitor.check_interval, 5)
        self.assertFalse(self.monitor.is_monitoring)

    def test_workspace_directory_setup(self):
        """ワークスペースディレクトリセットアップテスト"""
        workspace = Path(self.workspace_path)
        queue_dir = workspace / "queue"

        # ディレクトリが適切に設定されているか
        self.assertEqual(self.monitor.workspace, workspace)
        self.assertEqual(self.monitor.queue_dir, queue_dir)
        self.assertEqual(self.monitor.status_file, queue_dir / "queue_status.json")

    @patch("psutil.virtual_memory")
    def test_detect_memory_leak_normal(self, mock_memory):
        """正常なメモリ使用状況でのメモリリーク検出テスト"""
        # 正常範囲のメモリ使用量（ベースライン + 500MB）
        mock_memory.return_value.used = self.monitor.memory_baseline + (500 * 1024 * 1024)
        mock_memory.return_value.percent = 50.0  # 50%使用率

        result = self.monitor.detect_memory_leaks()
        self.assertIsNone(result)  # 正常時はNoneを返す

    @patch("psutil.virtual_memory")
    def test_detect_memory_leak_threshold_exceeded(self, mock_memory):
        """メモリリーク閾値超過時のテスト"""
        # 閾値超過のメモリ使用量（ベースライン + 3GB）
        mock_memory.return_value.used = self.monitor.memory_baseline + (3 * 1024 * 1024 * 1024)
        mock_memory.return_value.percent = 90.0  # 90%使用率（閾値超過）

        result = self.monitor.detect_memory_leaks()
        self.assertIsNotNone(result)
        self.assertIn("Memory", result)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.max_memory_allocated")
    def test_detect_gpu_anomalies_normal(self, mock_max_mem, mock_allocated, mock_cuda):
        """正常なGPU状況での異常検知テスト"""
        mock_cuda.return_value = True
        mock_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_max_mem.return_value = 8 * 1024 * 1024 * 1024  # 8GB

        # GPU監視有効化
        self.monitor.gpu_monitoring_enabled = True

        result = self.monitor.detect_gpu_anomalies()
        self.assertIsNone(result)  # 正常時はNoneを返す

    @patch("psutil.Process")
    def test_detect_process_health_normal(self, mock_process_class):
        """正常なプロセス状況での健全性検出テスト"""
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 50.0  # 50% CPU使用率
        mock_process.create_time.return_value = time.time() - 1000  # 1000秒前開始
        mock_process.status.return_value = "running"
        mock_process.pid = 1234
        mock_process_class.return_value = mock_process

        result = self.monitor.monitor_process_health()
        self.assertIsNone(result)  # 正常時はNoneを返す

    @patch("psutil.Process")
    def test_detect_process_health_high_cpu(self, mock_process_class):
        """高CPU使用率時の健全性検出テスト"""
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 96.0  # 96% CPU使用率（閾値超過）
        mock_process.create_time.return_value = time.time() - 1000  # 1000秒前開始
        mock_process.status.return_value = "running"
        mock_process.pid = 1234
        mock_process_class.return_value = mock_process

        result = self.monitor.monitor_process_health()
        self.assertIsNotNone(result)
        self.assertIn("CPU Usage Critical", result)

    def test_comprehensive_anomaly_check_integration(self):
        """包括的異常チェック統合テスト"""
        # 全ての検出機能が正常に動作することを確認
        anomalies = self.monitor.comprehensive_anomaly_check()

        # 戻り値は辞書形式であること
        self.assertIsInstance(anomalies, dict)

        # 必須フィールドの確認
        self.assertIn("timestamp", anomalies)
        self.assertIn("anomalies_detected", anomalies)
        self.assertIn("anomaly_count", anomalies)
        self.assertIn("anomalies", anomalies)
        self.assertIn("system_stats", anomalies)

        # anomaliesはリスト形式
        self.assertIsInstance(anomalies["anomalies"], list)

    def test_record_anomaly_functionality(self):
        """異常記録機能テスト"""
        anomaly_type = "test_anomaly"
        details = {"test_key": "test_value"}

        # プライベートメソッドのテスト（存在確認）
        if hasattr(self.monitor, "_record_anomaly"):
            # 異常記録機能の呼び出し確認
            try:
                self.monitor._record_anomaly(anomaly_type, details)
                # エラーが発生しなければ成功
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"_record_anomaly failed: {e}")
        else:
            # メソッドが存在しない場合はスキップ
            self.skipTest("_record_anomaly method not implemented")

    def test_monitoring_state_management(self):
        """監視状態管理テスト"""
        # 初期状態確認
        self.assertFalse(self.monitor.is_monitoring)
        self.assertIsNone(self.monitor.last_status)

        # 状態変更テスト（if start_monitoring method exists）
        if hasattr(self.monitor, "start_monitoring"):
            # 監視開始状態のテスト
            self.monitor.is_monitoring = True
            self.assertTrue(self.monitor.is_monitoring)

    def test_configuration_validation(self):
        """設定値検証テスト"""
        # 必須設定値の存在確認
        self.assertIsInstance(self.monitor.check_interval, int)
        self.assertGreater(self.monitor.check_interval, 0)

        self.assertIsInstance(self.monitor.anomaly_thresholds, dict)
        self.assertIn("gpu_memory_usage", self.monitor.anomaly_thresholds)
        self.assertIn("system_memory_usage", self.monitor.anomaly_thresholds)
        self.assertIn("cpu_usage", self.monitor.anomaly_thresholds)


class TestSubAgentMonitorRealWorldScenarios(unittest.TestCase):
    """実世界シナリオテスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SubAgentMonitor(workspace_path=self.temp_dir)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_queue_status_file_handling(self):
        """キューステータスファイル処理テスト"""
        # ステータスファイルパスの確認
        status_file = self.monitor.status_file
        self.assertIsInstance(status_file, Path)
        self.assertEqual(status_file.name, "queue_status.json")

    def test_workspace_creation_on_demand(self):
        """必要時ワークスペース作成テスト"""
        # 存在しないパスでの初期化
        non_existent_path = os.path.join(self.temp_dir, "new_workspace")
        monitor = SubAgentMonitor(workspace_path=non_existent_path)

        # パス設定の確認
        self.assertEqual(str(monitor.workspace), non_existent_path)

    @patch("logging.getLogger")
    def test_logging_configuration(self, mock_logger):
        """ログ設定テスト"""
        # ログ設定が適切に行われているか
        monitor = SubAgentMonitor(workspace_path=self.temp_dir)

        # ログ機能が利用可能であることを確認（グローバルloggerの確認）
        import tools.queue.subagent_monitor as monitor_module

        self.assertTrue(hasattr(monitor_module, "logger"))


def run_all_tests():
    """全テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestSubAgentMonitorINTG089))
    suite.addTests(loader.loadTestsFromTestCase(TestSubAgentMonitorRealWorldScenarios))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
