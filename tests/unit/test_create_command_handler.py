#!/usr/bin/env python3
"""
CreateCommandHandler テストスイート
KIRO-006 Phase 2: ワークフロー計画・起票システムのテスト

CreateCommandHandlerのSQLite専用機能、入力検証、
エラーハンドリングの包括的テストを提供します。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools.workflow.create_command_handler import CreateCommandHandler


class TestCreateCommandHandler(unittest.TestCase):
    """CreateCommandHandlerテストクラス"""

    def setUp(self):
        """テスト前準備"""
        self.handler = CreateCommandHandler()
        # ワークフローコントローラーをモック化
        self.handler.workflow_controller = Mock()

    def test_validate_tracker_id_success(self):
        """トラッカーID検証成功テスト"""
        valid_ids = ["TRACKER-001", "KIRO-006", "QUAL-044", "A-1", "TEST123-999"]

        for tracker_id in valid_ids:
            with self.subTest(tracker_id=tracker_id):
                is_valid, error = self.handler.validate_tracker_id(tracker_id)
                self.assertTrue(is_valid, f"Valid ID should pass: {tracker_id}")
                self.assertIsNone(error)

    def test_validate_tracker_id_invalid_format(self):
        """トラッカーID形式検証（異常）"""
        invalid_ids = [
            "tracker-001",  # 小文字
            "TRACKER_001",  # アンダースコア
            "TRACKER-",  # 番号なし
            "-001",  # プレフィックスなし
            "TRACKER001",  # ハイフンなし
            "TRACKER-ABC",  # 番号が文字
            "123-TRACKER",  # 数字から開始
        ]

        for tracker_id in invalid_ids:
            with self.subTest(tracker_id=tracker_id):
                is_valid, error = self.handler.validate_tracker_id(tracker_id)
                self.assertFalse(is_valid, f"Invalid ID should fail: {tracker_id}")
                self.assertIn("トラッカーID形式が無効です", error)

    def test_validate_tracker_id_empty(self):
        """トラッカーID空文字テスト"""
        empty_ids = ["", "   ", None]

        for tracker_id in empty_ids:
            with self.subTest(tracker_id=tracker_id):
                is_valid, error = self.handler.validate_tracker_id(tracker_id)
                self.assertFalse(is_valid)
                self.assertIn("トラッカーIDが指定されていません", error)

    def test_check_existing_workflow_not_exists(self):
        """既存ワークフロー確認（存在しない）"""
        # get_workflow_statusがエラーを返すようにモック設定
        self.handler.workflow_controller.get_workflow_status.return_value = {
            "error": "Tracker not found"
        }

        exists, info = self.handler.check_existing_workflow("TRACKER-001")

        self.assertFalse(exists)
        self.assertIsNone(info)
        self.handler.workflow_controller.get_workflow_status.assert_called_once_with("TRACKER-001")

    def test_check_existing_workflow_exists(self):
        """既存ワークフロー確認（存在する）"""
        # モックワークフロー状態
        mock_status = {
            "current_phase": "Phase 1",
            "current_step": "branch_verification",
            "can_proceed": True,
            "completed_steps": [{"step_id": "step1", "completed_at": "2025-01-06"}],
            "pending_approvals": [],
        }

        self.handler.workflow_controller.get_workflow_status.return_value = mock_status

        exists, info = self.handler.check_existing_workflow("TRACKER-001")

        self.assertTrue(exists)
        self.assertIsNotNone(info)
        self.assertEqual(info["tracker_id"], "TRACKER-001")
        self.assertEqual(info["current_phase"], "Phase 1")
        self.assertEqual(info["current_step"], "branch_verification")
        self.assertTrue(info["can_proceed"])
        self.assertEqual(info["completed_steps"], 1)
        self.assertEqual(info["pending_approvals"], 0)

    def test_check_existing_workflow_error(self):
        """既存ワークフロー確認（エラー）"""
        # get_workflow_statusでエラーが発生するようにモック設定
        self.handler.workflow_controller.get_workflow_status.side_effect = Exception(
            "Database Error"
        )

        exists, info = self.handler.check_existing_workflow("TRACKER-001")

        self.assertFalse(exists)
        self.assertIsNone(info)

    def test_check_existing_workflow_no_controller(self):
        """既存ワークフロー確認（コントローラーなし）"""
        self.handler.workflow_controller = None

        exists, info = self.handler.check_existing_workflow("TRACKER-001")

        self.assertFalse(exists)
        self.assertIsNone(info)

    def test_create_workflow_state_success(self):
        """ワークフロー状態作成成功テスト"""
        # create_tracker_workflowが成功を返すようにモック設定
        self.handler.workflow_controller.create_tracker_workflow.return_value = True

        # get_workflow_statusが作成後の状態を返すようにモック設定
        mock_status = {
            "current_phase": "Phase 0.5",
            "current_step": "branch_verification",
            "can_proceed": True,
        }
        self.handler.workflow_controller.get_workflow_status.return_value = mock_status

        success, message = self.handler.create_workflow_state("TRACKER-001")

        self.assertTrue(success)
        self.assertIn("ワークフロー状態管理を開始しました", message)
        self.assertIn("TRACKER-001", message)
        self.assertIn("Phase 0.5", message)
        self.assertIn("branch_verification", message)

        # メソッドが正しく呼ばれたか確認
        self.handler.workflow_controller.create_tracker_workflow.assert_called_once_with(
            "TRACKER-001"
        )
        self.handler.workflow_controller.get_workflow_status.assert_called_once_with("TRACKER-001")

    def test_create_workflow_state_creation_failed(self):
        """ワークフロー状態作成失敗テスト"""
        # create_tracker_workflowが失敗を返すようにモック設定
        self.handler.workflow_controller.create_tracker_workflow.return_value = False

        success, message = self.handler.create_workflow_state("TRACKER-001")

        self.assertFalse(success)
        self.assertIn("ワークフロー状態作成に失敗しました", message)
        self.assertIn("TRACKER-001", message)

    def test_create_workflow_state_no_controller(self):
        """ワークフロー状態作成（コントローラーなし）"""
        self.handler.workflow_controller = None

        success, message = self.handler.create_workflow_state("TRACKER-001")

        self.assertFalse(success)
        self.assertIn("ワークフロー状態管理が利用できません", message)

    def test_create_workflow_state_exception(self):
        """ワークフロー状態作成（例外発生）"""
        # create_tracker_workflowで例外が発生するようにモック設定
        self.handler.workflow_controller.create_tracker_workflow.side_effect = Exception(
            "Database Connection Error"
        )

        success, message = self.handler.create_workflow_state("TRACKER-001")

        self.assertFalse(success)
        self.assertIn("ワークフロー状態作成でエラーが発生しました", message)
        self.assertIn("Database Connection Error", message)

    def test_execute_create_command_success(self):
        """createコマンド実行成功テスト"""
        # 既存ワークフローなし
        self.handler.workflow_controller.get_workflow_status.return_value = {
            "error": "Tracker not found"
        }

        # 作成成功
        self.handler.workflow_controller.create_tracker_workflow.return_value = True
        mock_status = {
            "current_phase": "Phase 0.5",
            "current_step": "branch_verification",
            "can_proceed": True,
        }
        # get_workflow_statusの2回目の呼び出しで作成後の状態を返す
        self.handler.workflow_controller.get_workflow_status.side_effect = [
            {"error": "Tracker not found"},  # 既存確認時
            mock_status,  # 作成後確認時
        ]

        success, message = self.handler.execute_create_command("TRACKER-001")

        self.assertTrue(success)
        self.assertIn("ワークフロー状態管理を開始しました", message)

    def test_execute_create_command_validation_error(self):
        """createコマンド実行（入力検証エラー）"""
        success, message = self.handler.execute_create_command("")

        self.assertFalse(success)
        self.assertIn("トラッカーIDが指定されていません", message)

    def test_execute_create_command_existing_workflow(self):
        """createコマンド実行（既存ワークフロー）"""
        # 既存ワークフローあり
        mock_status = {
            "current_phase": "Phase 1",
            "current_step": "sow_creation",
            "can_proceed": False,
            "completed_steps": [{"step_id": "step1"}],
            "pending_approvals": [{"approval_id": "approval1"}],
        }
        self.handler.workflow_controller.get_workflow_status.return_value = mock_status

        success, message = self.handler.execute_create_command("TRACKER-001")

        self.assertFalse(success)
        self.assertIn("ワークフロー状態が既に存在します", message)
        self.assertIn("TRACKER-001", message)
        self.assertIn("Phase 1", message)
        self.assertIn("sow_creation", message)

    def test_get_usage_help(self):
        """使用方法ヘルプテスト"""
        help_text = self.handler.get_usage_help()

        self.assertIn("createコマンド使用方法", help_text)
        self.assertIn("TRACKER_ID", help_text)
        self.assertIn("SQLiteベース", help_text)
        self.assertIn("Google Sheets機能削除", help_text)
        self.assertIn("planコマンド分離", help_text)
        self.assertIn("使用例", help_text)

    def test_japanese_characters_in_tracker_id(self):
        """日本語文字を含むトラッカーIDテスト"""
        # 日本語文字は無効
        is_valid, error = self.handler.validate_tracker_id("トラッカー-001")

        self.assertFalse(is_valid)
        self.assertIn("トラッカーID形式が無効です", error)

    def test_special_characters_in_tracker_id(self):
        """特殊文字を含むトラッカーIDテスト"""
        invalid_ids = ["TRACKER@001", "TRACKER#001", "TRACKER$001", "TRACKER%001", "TRACKER&001"]

        for tracker_id in invalid_ids:
            with self.subTest(tracker_id=tracker_id):
                is_valid, error = self.handler.validate_tracker_id(tracker_id)
                self.assertFalse(is_valid)
                self.assertIn("トラッカーID形式が無効です", error)

    def test_boundary_tracker_id_values(self):
        """境界値トラッカーIDテスト"""
        # 最小有効値
        is_valid, error = self.handler.validate_tracker_id("A-1")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # 長い有効値
        is_valid, error = self.handler.validate_tracker_id("VERYLONGPREFIX123-999999")
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class TestCreateCommandHandlerIntegration(unittest.TestCase):
    """CreateCommandHandler統合テスト"""

    @patch("tools.interface.workflow_controller.get_workflow_controller")
    def test_initialization_with_controller(self, mock_get_controller):
        """ワークフローコントローラーでの初期化テスト"""
        # モック設定
        mock_controller = Mock()
        mock_get_controller.return_value = mock_controller

        handler = CreateCommandHandler()

        self.assertIsNotNone(handler.workflow_controller)
        self.assertEqual(handler.workflow_controller, mock_controller)
        mock_get_controller.assert_called_once()

    @patch("tools.interface.workflow_controller.get_workflow_controller")
    def test_initialization_error_handling(self, mock_get_controller):
        """初期化エラーハンドリングテスト"""
        # 初期化でエラーが発生するようにモック設定
        mock_get_controller.side_effect = Exception("Controller Error")

        handler = CreateCommandHandler()

        self.assertIsNone(handler.workflow_controller)

    def test_no_google_sheets_functionality(self):
        """Google Sheets機能が削除されていることの確認テスト"""
        handler = CreateCommandHandler()

        # Google Sheets関連のメソッドが存在しないことを確認
        self.assertFalse(hasattr(handler, "create_google_sheets_tracker"))
        self.assertFalse(hasattr(handler, "progress_manager"))

        # ヘルプメッセージにGoogle Sheets機能削除の記載があることを確認
        help_text = handler.get_usage_help()
        self.assertIn("Google Sheets機能削除", help_text)
        self.assertIn("SQLite専用", help_text)
        self.assertIn("planコマンド分離", help_text)


if __name__ == "__main__":
    # ログレベル設定
    import logging

    logging.basicConfig(level=logging.WARNING)

    # テスト実行
    unittest.main(verbosity=2)
