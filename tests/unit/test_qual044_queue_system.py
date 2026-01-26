#!/usr/bin/env python3
"""
QUAL-044 Queue System Unit Tests
長時間処理キューシステムの単体テスト

各コンポーネントの機能テスト・境界値テスト・エラー処理テスト
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.queue.long_task_manager import LongTaskQueue, QueueTask, TaskStatus
from tools.queue.notification_bridge import (
    NotificationBridge,
    PushoverNotifier,
    TaskFailureEscalator,
)
from tools.queue.subagent_monitor import SubAgentIntegration, SubAgentMonitor
from tools.queue.task_integration import TaskIntegration, TaskOrchestrator


class TestLongTaskQueue(unittest.TestCase):
    """LongTaskQueue単体テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.queue = LongTaskQueue(str(self.workspace))

    def tearDown(self):
        """テストクリーンアップ"""
        if hasattr(self, "queue"):
            self.queue.stop_execution()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_queue_initialization(self):
        """キュー初期化テスト"""
        self.assertIsInstance(self.queue, LongTaskQueue)
        self.assertEqual(len(self.queue.task_queue), 0)
        self.assertIsNone(self.queue.current_task)
        self.assertTrue(self.workspace.exists())

    def test_enqueue_task(self):
        """タスクキューイングテスト"""
        command = "echo 'test'"
        task_type = "test"

        task_id = self.queue.enqueue_task(command, task_type)

        self.assertIsInstance(task_id, str)
        self.assertTrue(task_id.startswith(f"{task_type}_"))
        self.assertEqual(len(self.queue.task_queue), 1)

        # タスク内容確認
        task = self.queue.task_queue[0]
        self.assertEqual(task.command, command)
        self.assertEqual(task.task_type, task_type)
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_queue_status(self):
        """キュー状態取得テスト"""
        # 空キュー
        status = self.queue.get_queue_status()
        self.assertEqual(status["queue_length"], 0)
        self.assertListEqual(status["tasks"], [])

        # タスク追加後
        self.queue.enqueue_task("echo 'test1'", "test")
        self.queue.enqueue_task("echo 'test2'", "test")

        status = self.queue.get_queue_status()
        self.assertEqual(status["queue_length"], 2)
        self.assertEqual(len(status["tasks"]), 2)

    def test_task_serialization(self):
        """タスクシリアライゼーションテスト"""
        # QueueTask辞書変換
        task = QueueTask(
            task_id="test_001",
            command="echo 'test'",
            task_type="test",
            status=TaskStatus.PENDING,
            created_at="2025-08-30T10:00:00",
        )

        task_dict = task.to_dict()
        self.assertEqual(task_dict["task_id"], "test_001")
        self.assertEqual(task_dict["status"], "pending")

        # 辞書からインスタンス復元
        restored_task = QueueTask.from_dict(task_dict)
        self.assertEqual(restored_task.task_id, task.task_id)
        self.assertEqual(restored_task.status, task.status)

    def test_task_cancellation(self):
        """タスクキャンセルテスト"""
        task_id = self.queue.enqueue_task("sleep 10", "test")

        # キュー内タスクのキャンセル
        cancelled = self.queue.cancel_task(task_id)
        self.assertTrue(cancelled)
        self.assertEqual(len(self.queue.task_queue), 0)

        # 存在しないタスクのキャンセル
        cancelled = self.queue.cancel_task("nonexistent")
        self.assertFalse(cancelled)


class TestSubAgentMonitor(unittest.TestCase):
    """SubAgentMonitor単体テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.workspace.mkdir(parents=True)
        self.monitor = SubAgentMonitor(str(self.workspace))

    def tearDown(self):
        """テストクリーンアップ"""
        if hasattr(self, "monitor"):
            self.monitor.stop_monitoring()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_monitor_initialization(self):
        """監視システム初期化テスト"""
        self.assertIsInstance(self.monitor, SubAgentMonitor)
        self.assertEqual(self.monitor.check_interval, 5)
        self.assertFalse(self.monitor.is_monitoring)
        self.assertIsNone(self.monitor.last_status)

    def test_status_file_read(self):
        """状態ファイル読み込みテスト"""
        # ファイルが存在しない場合
        status = self.monitor.read_status_file()
        self.assertIsNone(status)

        # ファイル作成・読み込みテスト
        status_data = {
            "task_id": "test_001",
            "status": "task_running",
            "timestamp": "2025-08-30T10:00:00",
        }

        status_file = self.workspace / "queue" / "queue_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)

        with open(status_file, "w") as f:
            json.dump(status_data, f)

        status = self.monitor.read_status_file()
        self.assertEqual(status["task_id"], "test_001")
        self.assertEqual(status["status"], "task_running")

    def test_callback_registration(self):
        """コールバック登録テスト"""

        def mock_complete_callback(status):
            return {"action": "completed"}

        def mock_failed_callback(status):
            return {"action": "failed"}

        self.monitor.register_callbacks(
            on_complete=mock_complete_callback, on_failed=mock_failed_callback
        )

        self.assertEqual(self.monitor.on_task_complete, mock_complete_callback)
        self.assertEqual(self.monitor.on_task_failed, mock_failed_callback)

    @patch("time.sleep")  # sleepをモック化してテスト高速化
    def test_monitoring_workflow(self, mock_sleep):
        """監視ワークフローテスト（シミュレーション）"""
        # 状態ファイル準備
        status_file = self.workspace / "queue" / "queue_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)

        # 完了状態を書き込み
        status_data = {
            "task_id": "test_001",
            "status": "task_completed",
            "timestamp": "2025-08-30T10:00:00",
        }

        with open(status_file, "w") as f:
            json.dump(status_data, f)

        # コールバック設定
        callback_called = False

        def completion_callback(status):
            nonlocal callback_called
            callback_called = True
            return {"next_action": "analyze_results"}

        self.monitor.register_callbacks(on_complete=completion_callback)

        # 監視開始（短時間で完了するように調整）
        def stop_monitoring():
            time.sleep(0.1)  # 短い待機
            self.monitor.stop_monitoring()

        stop_thread = threading.Thread(target=stop_monitoring)
        stop_thread.start()

        result = self.monitor.start_monitoring("test_001")
        stop_thread.join()

        # 結果確認
        self.assertIn("task_id", result)
        self.assertEqual(result["task_id"], "test_001")
        self.assertTrue(callback_called)


class TestTaskIntegration(unittest.TestCase):
    """TaskIntegration単体テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"

        # TaskIntegrationのworkspace_base設定を一時的に変更
        with patch("tools.queue.task_integration.TaskIntegration.__init__") as mock_init:
            mock_init.return_value = None
            self.integration = TaskIntegration.__new__(TaskIntegration)

            self.integration.tracker_id = "TEST-001"
            self.integration.workspace_base = Path(self.temp_dir)
            self.integration.workspace = self.workspace

            # Mock queue and integration
            self.integration.queue = Mock()
            self.integration.integration = Mock()

    def tearDown(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pytest_command_generation(self):
        """pytestコマンド生成テスト"""
        # デフォルト設定
        task_id = self.integration.execute_pytest("tests/")

        self.integration.queue.enqueue_task.assert_called_once()
        args, kwargs = self.integration.queue.enqueue_task.call_args
        command, task_type = args

        self.assertIn("sam-env/bin/python3 -m pytest tests/", command)
        self.assertIn("--cov=.", command)
        self.assertEqual(task_type, "pytest")

    def test_extract_character_command_generation(self):
        """extract_characterコマンド生成テスト"""
        input_dir = "/test/input"
        output_dir = "/test/output"

        task_id = self.integration.execute_extract_character(
            input_dir, output_dir, batch=True, max_files=5
        )

        self.integration.queue.enqueue_task.assert_called_once()
        args, kwargs = self.integration.queue.enqueue_task.call_args
        command, task_type = args

        self.assertIn("features/extraction/commands/extract_character.py", command)
        self.assertIn(input_dir, command)
        self.assertIn(output_dir, command)
        self.assertIn("--batch", command)
        self.assertIn("--max-files 5", command)
        self.assertEqual(task_type, "extract_character")

    def test_custom_command_registration(self):
        """カスタムコマンド登録テスト"""
        custom_command = "echo 'custom test'"
        task_type = "custom_test"

        task_id = self.integration.execute_custom_command(custom_command, task_type)

        self.integration.queue.enqueue_task.assert_called_once()
        args, kwargs = self.integration.queue.enqueue_task.call_args
        command, task_type_arg = args

        self.assertEqual(command, custom_command)
        self.assertEqual(task_type_arg, task_type)

    def test_result_parsing(self):
        """結果解析テスト"""
        # pytest結果解析テスト
        pytest_output = """
        ============================= test session starts ==============================
        collected 5 items
        
        tests/test_example.py::test_function PASSED
        tests/test_example.py::test_another FAILED
        
        =========================== 2 passed, 1 failed in 1.23s ===========================
        """

        # 一時ファイル作成
        output_file = self.temp_dir + "/pytest_output.log"
        with open(output_file, "w") as f:
            f.write(pytest_output)

        results = self.integration.parse_pytest_results(output_file)

        self.assertEqual(results["passed"], 2)
        self.assertEqual(results["failed"], 1)
        self.assertEqual(results["duration"], 1.23)


class TestNotificationBridge(unittest.TestCase):
    """NotificationBridge単体テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.workspace.mkdir(parents=True)

        # PushoverNotifier初期化をモック化
        with patch("tools.queue.notification_bridge.PushoverNotifier.__init__", return_value=None):
            self.bridge = NotificationBridge(str(self.workspace), "TEST-001")
            self.bridge.pushover = Mock()
            self.bridge.escalator = Mock()

    def tearDown(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_task_completion_handling(self):
        """タスク完了ハンドリングテスト"""
        task_id = "test_001"
        task_type = "test"
        results = {"success": True, "score": 0.95}

        self.bridge.handle_task_completion(task_id, task_type, results)

        # Pushover通知が呼ばれたことを確認
        self.bridge.pushover.send_task_completed.assert_called_once_with(
            task_id=task_id, task_type=task_type, details={"results": results}
        )

    def test_task_failure_handling(self):
        """タスク失敗ハンドリングテスト"""
        task_id = "test_001"
        task_type = "test"
        error = "Test error"
        retry_count = 2
        command = "echo 'test'"

        # TaskFailureEscalatorの戻り値をモック
        self.bridge.escalator.create_escalation.return_value = {
            "task_id": task_id,
            "escalation_id": "esc_001",
        }

        escalation = self.bridge.handle_task_failure(
            task_id, task_type, error, retry_count, command
        )

        # Pushover通知とTaskFailureEscalationが呼ばれたことを確認
        self.bridge.pushover.send_task_failed.assert_called_once()
        self.bridge.escalator.create_escalation.assert_called_once()
        self.assertIn("task_id", escalation)

    def test_queue_update_handling(self):
        """キュー更新ハンドリングテスト"""
        queue_status = {"queue_length": 3, "current_task": {"task_id": "test_001"}}

        self.bridge.handle_queue_update(queue_status)

        # Pushover状態通知が呼ばれたことを確認
        self.bridge.pushover.send_queue_status.assert_called_once_with(3, "test_001")


class TestTaskFailureEscalator(unittest.TestCase):
    """TaskFailureEscalator単体テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.workspace.mkdir(parents=True)
        self.escalator = TaskFailureEscalator(str(self.workspace))

    def tearDown(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_task_failure_escalation_creation(self):
        """タスク失敗エスカレーション作成テスト"""
        escalation = self.escalator.create_escalation(
            task_id="test_001",
            task_type="pytest",
            error="ImportError: No module named 'test_module'",
            retry_count=2,
            command="python -m pytest tests/",
        )

        self.assertEqual(escalation["task_id"], "test_001")
        self.assertEqual(escalation["task_type"], "pytest")
        self.assertEqual(escalation["retry_count"], 2)
        self.assertEqual(escalation["status"], "pending_review")
        self.assertIn("suggested_actions", escalation)

    def test_suggested_actions_pytest(self):
        """推奨アクション生成テスト（pytest）"""
        suggestions = self.escalator._suggest_actions("pytest", "ImportError")

        self.assertIn("Verify test dependencies are installed", suggestions)
        self.assertIn("Check for import errors in test files", suggestions)

    def test_suggested_actions_extract_character(self):
        """推奨アクション生成テスト（extract_character）"""
        suggestions = self.escalator._suggest_actions("extract_character", "CUDA out of memory")

        self.assertIn("Verify input images exist and are valid", suggestions)
        self.assertIn("Check CUDA/GPU availability", suggestions)
        self.assertIn("Increase available memory or reduce batch size", suggestions)

    def test_task_failure_escalation_prompt_generation(self):
        """タスク失敗エスカレーションプロンプト生成テスト"""
        escalation = {
            "task_id": "test_001",
            "task_type": "pytest",
            "error": "Test error",
            "retry_count": 2,
            "command": "pytest tests/",
            "created_at": "2025-08-30T10:00:00",
            "suggested_actions": ["Action 1", "Action 2"],
        }

        prompt = self.escalator.get_escalation_prompt(escalation)

        self.assertIn("test_001", prompt)
        self.assertIn("pytest", prompt)
        self.assertIn("Test error", prompt)
        self.assertIn("Action 1", prompt)
        self.assertIn("Action 2", prompt)


class TestTaskOrchestrator(unittest.TestCase):
    """TaskOrchestrator統合テスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()

        # TaskOrchestratorの初期化をモック化
        with patch("tools.queue.task_integration.TaskIntegration") as mock_integration:
            self.orchestrator = TaskOrchestrator("TEST-001")
            self.orchestrator.integration = Mock()
            self.orchestrator.active_tasks = {}

    def tearDown(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pytest_workflow(self):
        """pytestワークフローテスト"""
        # mock設定
        self.orchestrator.integration.execute_pytest.return_value = "test_task_001"
        self.orchestrator.integration.start_monitoring.return_value = {
            "final_status": "completed",
            "task_id": "test_task_001",
        }
        self.orchestrator.integration.generate_summary_report.return_value = "Test Report"

        task_id, result = self.orchestrator.run_pytest_with_monitoring("tests/")

        # 実行確認
        self.orchestrator.integration.execute_pytest.assert_called_once()
        self.orchestrator.integration.start_queue_processing.assert_called_once()
        self.orchestrator.integration.start_monitoring.assert_called_once()

        self.assertEqual(task_id, "test_task_001")
        self.assertEqual(result["final_status"], "completed")
        self.assertIn("summary_report", result)

    def test_extraction_workflow(self):
        """抽出ワークフローテスト"""
        # mock設定
        self.orchestrator.integration.execute_extract_character.return_value = "extract_task_001"
        self.orchestrator.integration.start_monitoring.return_value = {
            "final_status": "completed",
            "task_id": "extract_task_001",
        }

        task_id, result = self.orchestrator.run_extraction_with_monitoring("/test/input")

        # 実行確認
        self.orchestrator.integration.execute_extract_character.assert_called_once()
        self.orchestrator.integration.start_queue_processing.assert_called_once()
        self.orchestrator.integration.start_monitoring.assert_called_once()

        self.assertEqual(task_id, "extract_task_001")
        self.assertEqual(result["final_status"], "completed")


if __name__ == "__main__":
    # テスト実行設定
    unittest.main(argv=[""], verbosity=2, exit=False, buffer=True)  # テスト出力をバッファリング
