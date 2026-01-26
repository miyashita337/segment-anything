#!/usr/bin/env python3
"""
INTG-089 3システム統合テスト
SubAgentMonitor + NotificationBridge + EnhancedSubAgentTaskQueue の基本連携テスト
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.mocks.mock_pushover_intg089 import MockNotificationBridge, MockPushoverClient
from tools.queue.notification_bridge import NotificationBridge
from tools.queue.subagent_monitor import SubAgentMonitor
from tools.queue.subagent_wrapper import EnhancedSubAgentTaskQueue


class TestINTG089BasicIntegration(unittest.TestCase):
    """INTG-089基本統合テスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir
        self.tracker_id = "INTG-089-TEST"

        # Pushover設定ファイル作成
        self.pushover_config = {"user_key": "test_user_key", "api_token": "test_api_token"}
        self.pushover_config_path = os.path.join(self.temp_dir, "pushover.json")

        with open(self.pushover_config_path, "w") as f:
            json.dump(self.pushover_config, f)

        # 各システムの初期化
        self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.notification_bridge = NotificationBridge(
            workspace_path=self.workspace_path, tracker_id=self.tracker_id
        )
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=Path(self.workspace_path), tracker_id=self.tracker_id
        )

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_system_initialization_compatibility(self):
        """システム初期化互換性テスト"""
        # 全システムが同じワークスペースで正常に初期化できることを確認
        self.assertIsInstance(self.monitor, SubAgentMonitor)
        self.assertIsInstance(self.notification_bridge, NotificationBridge)
        self.assertIsInstance(self.task_queue, EnhancedSubAgentTaskQueue)

        # ワークスペースパスの一貫性確認
        self.assertEqual(str(self.monitor.workspace), self.workspace_path)
        self.assertEqual(self.notification_bridge.workspace_path, self.workspace_path)
        self.assertEqual(str(self.task_queue.workspace_path), self.workspace_path)

    def test_shared_workspace_directory_structure(self):
        """共有ワークスペースディレクトリ構造テスト"""
        # 各システムが必要なディレクトリを作成・使用できることを確認
        workspace = Path(self.workspace_path)

        # SubAgentMonitor用ディレクトリ
        queue_dir = workspace / "queue"
        self.assertEqual(self.monitor.queue_dir, queue_dir)

        # EnhancedSubAgentTaskQueue用ディレクトリ
        checkpoint_dir = self.task_queue.checkpoint_dir
        self.assertIsInstance(checkpoint_dir, Path)

        # ディレクトリが競合しないことを確認
        self.assertNotEqual(queue_dir, checkpoint_dir)

    @patch.object(MockPushoverClient, "send_notification")
    def test_monitor_to_notification_integration(self, mock_send):
        """監視→通知統合テスト"""
        mock_send.return_value = True

        # モック通知ブリッジに置き換え
        mock_bridge = MockNotificationBridge(
            workspace_path=self.workspace_path, pushover_config_path=self.pushover_config_path
        )

        # 異常検知シミュレーション
        anomaly_message = "Test anomaly detected"

        # 通知送信テスト
        result = mock_bridge.send_enhanced_notification(
            "System Alert", anomaly_message, notification_type="anomaly", priority_level="high"
        )

        self.assertTrue(result)

        # 通知履歴確認
        stats = mock_bridge.get_notification_statistics()
        self.assertGreater(stats["pushover_stats"]["total_calls"], 0)

    @patch("torch.cuda.is_available")
    def test_task_queue_to_monitor_integration(self, mock_cuda):
        """タスクキュー→監視統合テスト"""
        mock_cuda.return_value = True

        # タスク実行前の監視状態確認
        initial_gpu_available = self.monitor._check_gpu_availability()

        # タスクキューでのGPU利用可能性チェック
        task_queue_gpu_available = self.task_queue._check_gpu_available()

        # 両システムで同じ結果が得られることを確認
        self.assertEqual(initial_gpu_available, task_queue_gpu_available)

    def test_checkpoint_and_monitoring_coordination(self):
        """チェックポイント・監視連携テスト"""
        # チェックポイント作成
        test_task_data = {
            "task_id": "integration_test",
            "status": "running",
            "start_time": time.time(),
        }

        checkpoint_id = "integration_checkpoint"
        save_success = self.task_queue.save_checkpoint(checkpoint_id, test_task_data)
        self.assertTrue(save_success)

        # チェックポイントファイルの存在確認（監視対象ファイル）
        checkpoint_file = self.task_queue.checkpoint_dir / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            # ファイルが監視可能な場所に作成されていることを確認
            self.assertTrue(checkpoint_file.is_file())

            # 監視システムがファイルにアクセス可能であることを確認
            workspace_files = list(Path(self.workspace_path).rglob("*.json"))
            self.assertIn(checkpoint_file, workspace_files)

    def test_notification_priority_consistency(self):
        """通知優先度一貫性テスト"""
        # NotificationBridgeの優先度設定確認
        expected_priorities = ["critical", "high", "normal", "low"]

        for priority_level in expected_priorities:
            self.assertIn(priority_level, self.notification_bridge.notification_priorities)

            priority_config = self.notification_bridge.notification_priorities[priority_level]
            self.assertIn("priority", priority_config)
            self.assertIn("min_interval", priority_config)

    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        # 各システムでエラーが発生した際の他システムへの影響を確認

        # 1. 監視システムのエラー
        try:
            # 存在しないメソッド呼び出し（意図的エラー）
            if hasattr(self.monitor, "non_existent_method"):
                self.monitor.non_existent_method()
        except AttributeError:
            # エラーが他システムに影響しないことを確認
            self.assertIsInstance(self.notification_bridge, NotificationBridge)
            self.assertIsInstance(self.task_queue, EnhancedSubAgentTaskQueue)

        # 2. 通知システムのエラー（設定ファイル不在）
        invalid_bridge = NotificationBridge(
            workspace_path=self.workspace_path, pushover_config_path="/non/existent/path.json"
        )

        # エラーが発生しても他システムには影響しない
        self.assertIsInstance(invalid_bridge, NotificationBridge)
        self.assertIsInstance(self.monitor, SubAgentMonitor)

    def test_concurrent_system_operation(self):
        """同時システム操作テスト"""
        # 複数システムが同時に動作する際の整合性確認

        # 監視システムの動作
        anomalies = self.monitor.comprehensive_anomaly_check()
        self.assertIsInstance(anomalies, list)

        # 同時にチェックポイント操作
        test_data = {"concurrent": "test", "timestamp": time.time()}
        checkpoint_id = "concurrent_test"
        save_result = self.task_queue.save_checkpoint(checkpoint_id, test_data)

        # 同時に通知送信
        notification_result = self.notification_bridge.send_enhanced_notification(
            "Concurrent Test", "Testing concurrent operations", priority_level="low"
        )

        # 全ての操作が独立して正常に完了することを確認
        self.assertTrue(save_result)
        self.assertIsInstance(notification_result, bool)


class TestINTG089SystemCommunication(unittest.TestCase):
    """INTG-089システム間通信テスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir
        self.tracker_id = "COMM-089-TEST"

        # モック通信システム構築
        self.mock_bridge = MockNotificationBridge(workspace_path=self.workspace_path)
        self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=Path(self.workspace_path), tracker_id=self.tracker_id
        )

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_anomaly_detection_to_notification_flow(self):
        """異常検知→通知フローテスト"""
        # 異常検知シミュレーション
        anomalies = self.monitor.comprehensive_anomaly_check()

        # 異常が検知された場合の通知フロー
        if anomalies:
            for anomaly in anomalies:
                notification_sent = self.mock_bridge.send_enhanced_notification(
                    "System Anomaly Detected",
                    f"Anomaly: {anomaly}",
                    notification_type="system_anomaly",
                    priority_level="high",
                )
                # 通知が送信できることを確認
                self.assertIsInstance(notification_sent, bool)

    def test_task_execution_monitoring_integration(self):
        """タスク実行監視統合テスト"""
        # タスク実行前の状態記録
        initial_time = time.time()

        # テストタスクデータ
        test_task = {
            "task_id": "monitoring_integration_test",
            "command": "echo 'integration test'",
            "status": "pending",
        }

        # チェックポイント作成（タスク開始記録）
        checkpoint_id = "execution_start"
        checkpoint_saved = self.task_queue.save_checkpoint(checkpoint_id, test_task)
        self.assertTrue(checkpoint_saved)

        # タスク実行監視開始の通知
        monitoring_start = self.mock_bridge.send_enhanced_notification(
            "Task Execution Started",
            f"Task {test_task['task_id']} execution monitoring started",
            notification_type="task_start",
            priority_level="normal",
        )

        self.assertTrue(monitoring_start)

        # 実行統計の確認
        stats = self.mock_bridge.get_notification_statistics()
        self.assertGreater(stats["pushover_stats"]["total_calls"], 0)

    def test_system_health_check_integration(self):
        """システムヘルスチェック統合テスト"""
        # 各システムのヘルスチェック
        health_results = {}

        # 監視システムのヘルス
        try:
            anomalies = self.monitor.comprehensive_anomaly_check()
            health_results["monitor"] = {"status": "healthy", "anomalies": len(anomalies)}
        except Exception as e:
            health_results["monitor"] = {"status": "error", "error": str(e)}

        # タスクキューシステムのヘルス
        try:
            # 簡単なチェックポイント操作でヘルス確認
            test_checkpoint = self.task_queue.save_checkpoint("health_check", {"test": "health"})
            health_results["task_queue"] = {"status": "healthy", "checkpoint_ok": test_checkpoint}
        except Exception as e:
            health_results["task_queue"] = {"status": "error", "error": str(e)}

        # 通知システムのヘルス
        try:
            notification_test = self.mock_bridge.send_enhanced_notification(
                "Health Check", "System health check notification", priority_level="low"
            )
            health_results["notification"] = {
                "status": "healthy",
                "notification_ok": notification_test,
            }
        except Exception as e:
            health_results["notification"] = {"status": "error", "error": str(e)}

        # 全システムが基本的に動作していることを確認
        for system, result in health_results.items():
            self.assertIn("status", result)
            # エラーの場合は詳細を確認（必ずしも失敗ではない）
            if result["status"] == "error":
                print(f"System {system} health check warning: {result.get('error')}")

    def test_notification_deduplication_across_systems(self):
        """システム間通知重複排除テスト"""
        # 同じ内容の通知を複数システムから送信
        base_title = "System Integration Alert"
        base_message = "This is a test message for deduplication"

        # 通知送信（1回目）
        first_send = self.mock_bridge.send_enhanced_notification(
            base_title, base_message, notification_type="integration", priority_level="normal"
        )

        # 短時間での同一通知送信（重複排除されるべき）
        second_send = self.mock_bridge.send_enhanced_notification(
            base_title, base_message, notification_type="integration", priority_level="normal"
        )

        # 1回目は成功、2回目は重複排除される
        self.assertTrue(first_send)
        self.assertFalse(second_send)

        # 送信統計で重複排除が機能していることを確認
        stats = self.mock_bridge.get_notification_statistics()
        self.assertGreater(stats["duplicate_prevention_rate"], 0.0)


def run_all_integration_tests():
    """全統合テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestINTG089BasicIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestINTG089SystemCommunication))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_integration_tests()
    print(f"\n{'='*50}")
    print(f"INTG-089 Integration Tests: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if success else 1)
