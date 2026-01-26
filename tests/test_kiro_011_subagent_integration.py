#!/usr/bin/env python3
"""
KIRO-011 SubAgent-ワークフロー統合システムテスト
"""

import json
import os

# テスト対象のインポート
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.workflow.state_manager import WorkflowStateManager
from tools.workflow.subagent_command_handler import SubAgentCommandHandler
from tools.workflow.subagent_lock_manager import SubAgentLockManager
from tools.workflow.subagent_monitor import SubAgentMonitor, SubAgentProcess, SubAgentStatus


class TestSubAgentMonitor(unittest.TestCase):
    """SubAgent状態監視システムテスト"""

    def setUp(self):
        """テスト前準備"""
        # 一時ディレクトリでテスト実行
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_subagent.db")
        self.monitor = SubAgentMonitor(self.db_path)

    def tearDown(self):
        """テスト後クリーンアップ"""
        self.monitor.stop_monitoring()
        # 一時ファイル削除
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_subagent(self):
        """SubAgent登録テスト"""
        tracker_id = "TEST-001"
        process_type = "extraction"
        command = "echo test"
        workspace_path = os.path.join(self.temp_dir, "workspace")

        # SubAgent登録
        success = self.monitor.register_subagent(
            tracker_id=tracker_id,
            process_type=process_type,
            command=command,
            workspace_path=workspace_path,
        )

        self.assertTrue(success)

        # 登録確認
        process_key = f"{tracker_id}_{process_type}"
        self.assertIn(process_key, self.monitor.active_processes)

        process = self.monitor.active_processes[process_key]
        self.assertEqual(process.tracker_id, tracker_id)
        self.assertEqual(process.process_type, process_type)
        self.assertEqual(process.command, command)
        self.assertEqual(process.workspace_path, workspace_path)

    def test_check_subagent_status(self):
        """SubAgent状態確認テスト"""
        tracker_id = "TEST-002"
        process_type = "extraction"

        # 未登録プロセスの状態確認
        status = self.monitor.check_subagent_status(tracker_id, process_type)
        self.assertEqual(status, SubAgentStatus.NOT_STARTED)

        # プロセス登録
        self.monitor.register_subagent(
            tracker_id=tracker_id,
            process_type=process_type,
            command="echo test",
            workspace_path=os.path.join(self.temp_dir, "workspace"),
        )

        # 登録後の状態確認
        status = self.monitor.check_subagent_status(tracker_id, process_type)
        self.assertEqual(status, SubAgentStatus.NOT_STARTED)

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_auto_retry_logic(self, mock_process, mock_pid_exists):
        """自動再実行ロジックテスト"""
        tracker_id = "TEST-003"
        process_type = "extraction"
        workspace_path = os.path.join(self.temp_dir, "workspace")

        # ワークスペース作成
        os.makedirs(workspace_path, exist_ok=True)
        os.makedirs(os.path.join(workspace_path, "extraction"), exist_ok=True)

        # SubAgent登録
        self.monitor.register_subagent(
            tracker_id=tracker_id,
            process_type=process_type,
            command="echo test",
            workspace_path=workspace_path,
        )

        process_key = f"{tracker_id}_{process_type}"
        process = self.monitor.active_processes[process_key]

        # 失敗状態に設定
        process.status = SubAgentStatus.FAILED
        process.start_time = self.monitor._get_past_time(600)  # 10分前に開始

        # 0件判定テスト（抽出ファイルなし）
        should_retry = self.monitor.should_auto_retry(tracker_id, process_type)
        self.assertTrue(should_retry, "0件で失敗状態の場合は自動再実行すべき")

        # 部分成功テスト（抽出ファイルあり）
        test_file = os.path.join(workspace_path, "extraction", "extracted_001.jpg")
        with open(test_file, "w") as f:
            f.write("test")

        should_retry = self.monitor.should_auto_retry(tracker_id, process_type)
        self.assertFalse(should_retry, "部分成功の場合は自動再実行しない")

    def test_timeout_detection(self):
        """タイムアウト検出テスト"""
        tracker_id = "TEST-004"
        process_type = "extraction"

        # プロセス登録
        self.monitor.register_subagent(
            tracker_id=tracker_id,
            process_type=process_type,
            command="sleep 1",
            workspace_path=os.path.join(self.temp_dir, "workspace"),
            expected_duration=2,  # 2秒でタイムアウト
        )

        # タイムアウト前
        is_timeout = self.monitor.is_timeout_reached(tracker_id, process_type)
        self.assertFalse(is_timeout)

        # 開始時刻を過去に設定
        process_key = f"{tracker_id}_{process_type}"
        process = self.monitor.active_processes[process_key]
        process.start_time = self.monitor._get_past_time(5)  # 5秒前

        # タイムアウト後
        is_timeout = self.monitor.is_timeout_reached(tracker_id, process_type)
        self.assertTrue(is_timeout)

    def _get_past_time(self, seconds_ago):
        """過去の時刻を取得（テスト用ヘルパー）"""
        from datetime import datetime, timedelta

        return datetime.now() - timedelta(seconds=seconds_ago)


class TestSubAgentLockManager(unittest.TestCase):
    """SubAgent二重起動防止システムテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.lock_manager = SubAgentLockManager(self.temp_dir)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lock_acquisition(self):
        """ロック取得テスト"""
        tracker_id = "TEST-001"
        process_type = "extraction"

        # 初回ロック取得
        success = self.lock_manager.acquire_lock(tracker_id, process_type)
        self.assertTrue(success, "初回ロック取得は成功すべき")

        # 二重ロック取得（失敗すべき）
        success = self.lock_manager.acquire_lock(tracker_id, process_type)
        self.assertFalse(success, "二重ロック取得は失敗すべき")

        # ロック解放
        success = self.lock_manager.release_lock(tracker_id, process_type)
        self.assertTrue(success, "ロック解放は成功すべき")

        # 解放後の再取得
        success = self.lock_manager.acquire_lock(tracker_id, process_type)
        self.assertTrue(success, "解放後の再取得は成功すべき")

    def test_duplicate_execution_risk(self):
        """二重実行リスク判定テスト"""
        tracker_id = "TEST-002"
        process_type = "extraction"

        # リスクなし状態
        risk = self.lock_manager.is_duplicate_execution_risk(tracker_id, process_type)
        self.assertFalse(risk)

        # ロック取得後はリスクあり
        self.lock_manager.acquire_lock(tracker_id, process_type)
        risk = self.lock_manager.is_duplicate_execution_risk(tracker_id, process_type)
        self.assertTrue(risk)

    def test_lock_cleanup(self):
        """ロッククリーンアップテスト"""
        tracker_id = "TEST-003"
        process_type = "extraction"

        # 複数ロック作成
        self.lock_manager.acquire_lock(tracker_id, process_type)
        self.lock_manager.acquire_lock("TEST-004", process_type)

        # 特定トラッカーのクリーンアップ
        cleaned = self.lock_manager.force_cleanup_locks(tracker_id)
        self.assertEqual(cleaned, 2)  # 個別 + グローバル

        # リスク確認
        risk = self.lock_manager.is_duplicate_execution_risk(tracker_id, process_type)
        self.assertFalse(risk, "クリーンアップ後はリスクなし")


class TestSubAgentCommandHandler(unittest.TestCase):
    """SubAgentコマンドハンドラーテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("tools.workflow.subagent_command_handler.get_workspace_config")
    @patch("tools.workflow.subagent_command_handler.get_subagent_monitor")
    @patch("tools.workflow.subagent_command_handler.get_lock_manager")
    def test_extraction_command(self, mock_lock_manager, mock_monitor, mock_workspace_config):
        """SubAgent抽出コマンドテスト"""
        # モック設定
        mock_workspace_config.return_value.get_workspace_config.return_value = {
            "workspace_path": os.path.join(self.temp_dir, "workspace"),
            "author_name": "test_user",
        }
        mock_workspace_config.return_value.get_input_directory.return_value = os.path.join(
            self.temp_dir, "input"
        )

        mock_lock_manager.return_value.is_duplicate_execution_risk.return_value = False
        mock_monitor.return_value.register_subagent.return_value = True
        mock_monitor.return_value.start_subagent.return_value = True

        # コマンドハンドラー作成
        handler = SubAgentCommandHandler()
        handler.workspace_config = mock_workspace_config.return_value
        handler.lock_manager = mock_lock_manager.return_value
        handler.subagent_monitor = mock_monitor.return_value

        # 抽出処理実行
        success = handler.handle_subagent_extraction("TEST-001")
        self.assertTrue(success)

        # 呼び出し確認
        mock_monitor.return_value.register_subagent.assert_called_once()
        mock_monitor.return_value.start_subagent.assert_called_once()

    @patch("tools.workflow.subagent_command_handler.get_subagent_monitor")
    def test_status_command(self, mock_monitor):
        """SubAgent状態確認コマンドテスト"""
        # モック設定
        mock_monitor.return_value.check_subagent_status.return_value = SubAgentStatus.RUNNING
        mock_monitor.return_value.get_all_active_processes.return_value = [
            {
                "tracker_id": "TEST-001",
                "process_type": "extraction",
                "status": "running",
                "pid": 12345,
                "elapsed_seconds": 300,
                "expected_duration": 1800,
                "retry_count": 0,
            }
        ]

        # コマンドハンドラー作成
        handler = SubAgentCommandHandler()
        handler.subagent_monitor = mock_monitor.return_value

        # 状態確認実行
        success = handler.handle_subagent_status("TEST-001")
        self.assertTrue(success)

        # 呼び出し確認
        mock_monitor.return_value.check_subagent_status.assert_called_once_with(
            "TEST-001", "extraction"
        )


class TestWorkflowStateIntegration(unittest.TestCase):
    """ワークフロー状態統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_workflow.db")
        self.state_manager = WorkflowStateManager(self.db_path)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_waiting_for_subagent_state(self):
        """waiting_for_subagent状態テスト"""
        tracker_id = "TEST-001"

        # ワークフロー作成
        self.state_manager.create_tracker_workflow(tracker_id)

        # 現在のステップ確認
        current_step = self.state_manager.get_current_step(tracker_id)
        self.assertEqual(current_step, "branch_verification")

        # ステップ進行テスト（手動で進行）
        steps = [
            "branch_verification",
            "sam_env_check",
            "google_sheets_sync",
            "sow_creation",
            "implementation",
            "testing",
            "subagent_extraction",
        ]

        for expected_step in steps:
            current = self.state_manager.get_current_step(tracker_id)
            self.assertEqual(current, expected_step)
            self.state_manager.advance_to_next_step(tracker_id)

        # SubAgent関連ステップ確認
        current_step = self.state_manager.get_current_step(tracker_id)
        self.assertEqual(current_step, "waiting_for_subagent")

        # さらに進行
        self.state_manager.advance_to_next_step(tracker_id)
        current_step = self.state_manager.get_current_step(tracker_id)
        self.assertEqual(current_step, "subagent_validation")


class TestIntegrationScenario(unittest.TestCase):
    """統合シナリオテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_subagent_workflow(self):
        """完全なSubAgentワークフローテスト"""
        tracker_id = "INTEGRATION-001"

        # 1. ワークフロー状態管理開始
        state_manager = WorkflowStateManager(os.path.join(self.temp_dir, "workflow.db"))
        state_manager.create_tracker_workflow(tracker_id)

        # 2. SubAgent監視システム初期化
        monitor = SubAgentMonitor(os.path.join(self.temp_dir, "subagent.db"))

        # 3. 二重起動防止システム初期化
        lock_manager = SubAgentLockManager(os.path.join(self.temp_dir, "locks"))

        # 4. ワークスペース準備
        workspace_path = os.path.join(self.temp_dir, "workspace")
        os.makedirs(workspace_path, exist_ok=True)
        os.makedirs(os.path.join(workspace_path, "extraction"), exist_ok=True)

        # 5. SubAgent登録
        command = "echo 'test extraction' > extraction/test.txt"
        success = monitor.register_subagent(
            tracker_id=tracker_id,
            process_type="extraction",
            command=command,
            workspace_path=workspace_path,
        )
        self.assertTrue(success)

        # 6. 二重起動防止確認
        risk = lock_manager.is_duplicate_execution_risk(tracker_id, "extraction")
        self.assertFalse(risk)

        # 7. ロック取得
        lock_acquired = lock_manager.acquire_lock(tracker_id, "extraction")
        self.assertTrue(lock_acquired)

        # 8. SubAgent状態確認
        status = monitor.check_subagent_status(tracker_id, "extraction")
        self.assertEqual(status, SubAgentStatus.NOT_STARTED)

        # 9. クリーンアップ
        monitor.stop_monitoring()
        lock_manager.release_lock(tracker_id, "extraction")


if __name__ == "__main__":
    # テスト実行
    unittest.main(verbosity=2)
