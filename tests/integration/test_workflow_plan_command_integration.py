#!/usr/bin/env python3
"""
ワークフロー計画・起票システム統合テスト
KIRO-006 Phase 2: plan→createワークフローの統合テスト

PlanCommandHandlerとCreateCommandHandlerの連携、
CLI統合、エンドツーエンドワークフローの包括的テストを提供します。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools.workflow.create_command_handler import CreateCommandHandler
from tools.workflow.plan_command_handler import PlanCommandHandler


class TestWorkflowPlanCommandIntegration(unittest.TestCase):
    """ワークフロー計画・起票システム統合テストクラス"""

    def setUp(self):
        """テスト前準備"""
        self.plan_handler = PlanCommandHandler()
        self.create_handler = CreateCommandHandler()

        # モック設定
        self.plan_handler.progress_manager = Mock()
        self.create_handler.workflow_controller = Mock()

    def test_plan_to_create_workflow_success(self):
        """plan→createワークフロー成功テスト"""
        tracker_id = "TEST-001"
        summary = "統合テスト概要"
        details = "統合テスト詳細説明"

        # 1. planコマンド実行（Google Sheets起票）
        mock_task = Mock()
        mock_task.tracker_id = tracker_id
        mock_task.status = Mock()
        mock_task.status.value = "着手前"
        mock_task.created_date = "2025-01-06 10:00:00"

        # planハンドラーのモック設定
        self.plan_handler.progress_manager.get_task.side_effect = [None, mock_task]
        self.plan_handler.progress_manager.create_task.return_value = mock_task

        plan_success, plan_message = self.plan_handler.execute_plan_command(
            tracker_id, summary, details
        )

        self.assertTrue(plan_success)
        self.assertIn("Google Sheetsにトラッカーを起票しました", plan_message)

        # 2. createコマンド実行（ワークフロー状態管理開始）
        # createハンドラーのモック設定
        self.create_handler.workflow_controller.get_workflow_status.side_effect = [
            {"error": "Tracker not found"},  # 既存確認時
            {  # 作成後確認時
                "current_phase": "Phase 0.5",
                "current_step": "branch_verification",
                "can_proceed": True,
            },
        ]
        self.create_handler.workflow_controller.create_tracker_workflow.return_value = True

        create_success, create_message = self.create_handler.execute_create_command(tracker_id)

        self.assertTrue(create_success)
        self.assertIn("ワークフロー状態管理を開始しました", create_message)

        # 3. 統合確認
        self.assertTrue(plan_success and create_success)

        # planコマンドでGoogle Sheetsに起票されたことを確認
        self.plan_handler.progress_manager.create_task.assert_called_once()

        # createコマンドでワークフロー状態が作成されたことを確認
        self.create_handler.workflow_controller.create_tracker_workflow.assert_called_once_with(
            tracker_id
        )

    def test_plan_existing_tracker_error(self):
        """plan実行時の既存トラッカーエラーテスト"""
        tracker_id = "EXISTING-001"

        # 既存トラッカーありのモック設定
        existing_task = Mock()
        existing_task.tracker_id = tracker_id
        existing_task.status = Mock()
        existing_task.status.value = "着手中"
        existing_task.created_date = "2025-01-05"
        existing_task.description = "既存のタスク"

        self.plan_handler.progress_manager.get_task.return_value = existing_task

        success, message = self.plan_handler.execute_plan_command(tracker_id, "概要", "詳細")

        self.assertFalse(success)
        self.assertIn("トラッカーが既に存在します", message)
        self.assertIn(tracker_id, message)

    def test_create_existing_workflow_error(self):
        """create実行時の既存ワークフローエラーテスト"""
        tracker_id = "EXISTING-002"

        # 既存ワークフローありのモック設定
        existing_status = {
            "current_phase": "Phase 1",
            "current_step": "sow_creation",
            "can_proceed": False,
            "completed_steps": [{"step_id": "step1"}],
            "pending_approvals": [],
        }

        self.create_handler.workflow_controller.get_workflow_status.return_value = existing_status

        success, message = self.create_handler.execute_create_command(tracker_id)

        self.assertFalse(success)
        self.assertIn("ワークフロー状態が既に存在します", message)
        self.assertIn(tracker_id, message)

    def test_input_validation_integration(self):
        """入力検証統合テスト"""
        # 無効なトラッカーID
        invalid_tracker_ids = ["", "invalid", "tracker-001", "TRACKER_001"]

        for tracker_id in invalid_tracker_ids:
            with self.subTest(tracker_id=tracker_id):
                # planコマンド検証
                plan_success, plan_message = self.plan_handler.execute_plan_command(
                    tracker_id, "概要", "詳細"
                )
                self.assertFalse(plan_success)

                # createコマンド検証
                create_success, create_message = self.create_handler.execute_create_command(
                    tracker_id
                )
                self.assertFalse(create_success)

    def test_details_length_validation(self):
        """詳細文字数制限統合テスト"""
        tracker_id = "LENGTH-001"
        summary = "文字数制限テスト"

        # 制限を超える詳細
        long_details = "a" * (PlanCommandHandler.MAX_DETAILS_LENGTH + 1)

        success, message = self.plan_handler.execute_plan_command(tracker_id, summary, long_details)

        self.assertFalse(success)
        self.assertIn("詳細が文字数制限を超えています", message)
        self.assertIn(f"{PlanCommandHandler.MAX_DETAILS_LENGTH + 1:,}文字", message)

    def test_priority_handling_integration(self):
        """優先度処理統合テスト"""
        tracker_id = "PRIORITY-001"
        summary = "優先度テスト"
        details = "優先度処理のテスト"

        # 高優先度でのplan実行
        mock_task = Mock()
        mock_task.tracker_id = tracker_id
        mock_task.status = Mock()
        mock_task.status.value = "着手前"
        mock_task.created_date = "2025-01-06 10:00:00"

        self.plan_handler.progress_manager.get_task.side_effect = [None, mock_task]
        self.plan_handler.progress_manager.create_task.return_value = mock_task

        success, message = self.plan_handler.execute_plan_command(
            tracker_id, summary, details, "high"
        )

        self.assertTrue(success)
        self.assertIn("優先度高", message)

    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        tracker_id = "ERROR-001"

        # planコマンドでのエラー
        self.plan_handler.progress_manager = None
        plan_success, plan_message = self.plan_handler.execute_plan_command(tracker_id, "概要", "詳細")
        self.assertFalse(plan_success)
        self.assertIn("Google Sheets連携が利用できません", plan_message)

        # createコマンドでのエラー
        self.create_handler.workflow_controller = None
        create_success, create_message = self.create_handler.execute_create_command(tracker_id)
        self.assertFalse(create_success)
        self.assertIn("ワークフロー状態管理が利用できません", create_message)


class TestCLIIntegration(unittest.TestCase):
    """CLI統合テスト"""

    def setUp(self):
        """テスト前準備"""
        self.cli_path = "tools/workflow/workflow_cli.py"

    def test_cli_help_commands(self):
        """CLIヘルプコマンドテスト"""
        # メインヘルプ
        result = subprocess.run(
            ["python", self.cli_path, "--help"], capture_output=True, text=True, cwd=current_dir
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("plan", result.stdout)
        self.assertIn("create", result.stdout)
        self.assertIn("Google Sheetsにトラッカーを起票", result.stdout)
        self.assertIn("SQLite専用", result.stdout)

        # planコマンドヘルプ
        result = subprocess.run(
            ["python", self.cli_path, "plan", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("tracker_id", result.stdout)
        self.assertIn("summary", result.stdout)
        self.assertIn("details", result.stdout)
        self.assertIn("priority", result.stdout)

        # createコマンドヘルプ
        result = subprocess.run(
            ["python", self.cli_path, "create", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("tracker_id", result.stdout)

    def test_cli_guide_command(self):
        """CLIガイドコマンドテスト"""
        result = subprocess.run(
            ["python", self.cli_path, "guide"], capture_output=True, text=True, cwd=current_dir
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("統合ワークフローガイド", result.stdout)
        self.assertIn("planコマンド新設", result.stdout)
        self.assertIn("createコマンド変更", result.stdout)
        self.assertIn("plan→createの順序", result.stdout)

    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    def test_cli_plan_command_integration(self, mock_handler_class):
        """CLI planコマンド統合テスト"""
        # モック設定
        mock_handler = Mock()
        mock_handler.execute_plan_command.return_value = (True, "成功メッセージ")
        mock_handler_class.return_value = mock_handler

        # CLIコマンド実行（実際には実行せず、モックで確認）
        from tools.workflow.workflow_cli import plan_tracker

        success = plan_tracker("TEST-001", "概要", "詳細", "high")

        self.assertTrue(success)
        mock_handler.execute_plan_command.assert_called_once_with("TEST-001", "概要", "詳細", "high")

    @patch("tools.workflow.create_command_handler.CreateCommandHandler")
    def test_cli_create_command_integration(self, mock_handler_class):
        """CLI createコマンド統合テスト"""
        # モック設定
        mock_handler = Mock()
        mock_handler.execute_create_command.return_value = (True, "成功メッセージ")
        mock_handler_class.return_value = mock_handler

        # CLIコマンド実行（実際には実行せず、モックで確認）
        from tools.workflow.workflow_cli import create_tracker

        success = create_tracker("TEST-001")

        self.assertTrue(success)
        mock_handler.execute_create_command.assert_called_once_with("TEST-001")


class TestBackwardCompatibility(unittest.TestCase):
    """後方互換性テスト"""

    def test_existing_functionality_preserved(self):
        """既存機能の保持確認テスト"""
        # 既存のCLIコマンドが利用可能であることを確認
        existing_commands = [
            "status",
            "instructions",
            "step",
            "approvals",
            "process",
            "sheets",
            "template",
            "guide",
        ]

        for command in existing_commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    ["python", "tools/workflow/workflow_cli.py", command, "--help"],
                    capture_output=True,
                    text=True,
                    cwd=current_dir,
                )
                # ヘルプが表示されることを確認（機能が存在することの証明）
                self.assertEqual(result.returncode, 0)

    def test_no_breaking_changes(self):
        """破壊的変更がないことの確認テスト"""
        # 既存のワークフローコマンドが引き続き動作することを確認
        # （実際の実行はせず、インポートと基本構造の確認のみ）

        try:
            from tools.workflow.workflow_cli import (
                attempt_step,
                check_process,
                check_sheets_status,
                generate_template,
                get_instructions,
                get_status,
                list_approvals,
                show_guide,
            )

            # インポートが成功すれば既存機能が保持されている
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"既存機能のインポートに失敗: {e}")


if __name__ == "__main__":
    # ログレベル設定
    import logging

    logging.basicConfig(level=logging.WARNING)

    # テスト実行
    unittest.main(verbosity=2)
