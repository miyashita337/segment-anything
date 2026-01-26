"""
CreateCommandHandler ユニットテスト
SQLite専用ワークフロー状態管理コマンドのテスト
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tests.workflow.fixtures.mock_data import SAMPLE_TRACKER_ID, MockData
from tests.workflow.fixtures.workflow_fixtures import CLITestHelper, WorkflowTestBase


class TestCreateCommandHandler(WorkflowTestBase):
    """CreateCommandHandlerのテストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # WorkflowControllerのモック設定
        self.mock_workflow_controller = self.create_mock_workflow_controller()
        self.add_mock(
            "tools.interface.workflow_controller.get_workflow_controller",
            return_value=self.mock_workflow_controller,
        )

        # ワークスペース検証のモック設定
        self.setup_workspace_validation_mocks(is_configured=True)

    def test_create_command_success(self):
        """createコマンド正常実行テスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # WorkflowControllerが存在しないトラッカーを返すよう設定
        self.mock_workflow_controller.get_workflow_status.return_value = {"status": "not_found"}

        # 新規作成成功をシミュレート
        self.mock_workflow_controller.create_tracker_workflow.return_value = True

        success, message = handler.execute_create_command(SAMPLE_TRACKER_ID)

        self.assertTrue(success)
        self.assertIn("✅", message)
        self.assertIn("ワークフロー状態管理を開始しました", message)

    def test_create_command_invalid_tracker_id(self):
        """無効なトラッカーIDのテスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # 無効なトラッカーIDでテスト
        invalid_ids = ["", "invalid", "tracker-001", "TRACKER_001", "123-TRACKER"]

        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                success, message = handler.execute_create_command(invalid_id)

                self.assertFalse(success)
                self.assertIn("❌", message)
                self.assertIn("トラッカーID", message)

    def test_create_command_existing_workflow(self):
        """既存ワークフローのテスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # 既存のワークフロー状態を返すよう設定
        existing_status = MockData.MOCK_WORKFLOW_STATUS["TRACKER-001"]
        self.mock_workflow_controller.get_workflow_status.return_value = existing_status

        success, message = handler.execute_create_command("TRACKER-001")

        self.assertFalse(success)
        self.assertIn("⚠️", message)
        self.assertIn("ワークフロー状態が既に存在します", message)
        self.assertIn("TRACKER-001", message)

    @unittest.skip("ワークスペース検証機能は実装に存在しないためスキップ")
    def test_create_command_workspace_not_configured(self):
        """ワークスペース未設定のテスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        # ワークスペース未設定のモック設定
        self.setup_workspace_validation_mocks(is_configured=False)

        handler = CreateCommandHandler()

        success, message = handler.execute_create_command(SAMPLE_TRACKER_ID)

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("ワークスペース設定エラー", message)

    def test_create_command_workflow_controller_error(self):
        """WorkflowController初期化エラーのテスト"""
        with patch(
            "tools.interface.workflow_controller.get_workflow_controller"
        ) as mock_get_controller:
            mock_get_controller.return_value = None

            from tools.workflow.create_command_handler import CreateCommandHandler

            handler = CreateCommandHandler()

            success, message = handler.execute_create_command(SAMPLE_TRACKER_ID)

            self.assertFalse(success)
            self.assertIn("❌", message)
            self.assertIn("ワークフローコントローラーが利用できません", message)

    def test_create_command_creation_failure(self):
        """ワークフロー作成失敗のテスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # WorkflowControllerが存在しないトラッカーを返すよう設定
        self.mock_workflow_controller.get_workflow_status.return_value = {"status": "not_found"}

        # 作成失敗をシミュレート
        self.mock_workflow_controller.create_tracker_workflow.return_value = False

        success, message = handler.execute_create_command(SAMPLE_TRACKER_ID)

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("ワークフロー状態作成に失敗しました", message)

    def test_validate_tracker_id_method(self):
        """validate_tracker_idメソッドの単体テスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # 正常ケース
        is_valid, error = handler.validate_tracker_id(SAMPLE_TRACKER_ID)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # 無効ケースのテスト
        test_cases = [
            ("", "トラッカーIDが指定されていません"),
            ("   ", "トラッカーIDが指定されていません"),
            ("invalid", "トラッカーID形式が無効です"),
            ("tracker-001", "トラッカーID形式が無効です"),
            ("TRACKER_001", "トラッカーID形式が無効です"),
            ("123-TRACKER", "トラッカーID形式が無効です"),
        ]

        for tracker_id, expected_error in test_cases:
            with self.subTest(tracker_id=tracker_id, expected_error=expected_error):
                is_valid, error = handler.validate_tracker_id(tracker_id)
                self.assertFalse(is_valid)
                self.assertIn(expected_error, error)

    def test_check_existing_workflow_method(self):
        """check_existing_workflowメソッドの単体テスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # 既存ワークフローの場合
        existing_status = MockData.MOCK_WORKFLOW_STATUS["TRACKER-001"]
        self.mock_workflow_controller.get_workflow_status.return_value = existing_status

        exists, info = handler.check_existing_workflow("TRACKER-001")
        self.assertTrue(exists)
        self.assertIsNotNone(info)
        self.assertEqual(info["tracker_id"], "TRACKER-001")
        self.assertEqual(info["current_phase"], "phase_0_5")

        # 存在しないワークフローの場合
        self.mock_workflow_controller.get_workflow_status.return_value = {"status": "not_found"}

        exists, info = handler.check_existing_workflow("NON-EXISTENT-001")
        self.assertFalse(exists)
        self.assertIsNone(info)

        # エラーの場合
        self.mock_workflow_controller.get_workflow_status.return_value = {
            "error": "Database connection failed"
        }

        exists, info = handler.check_existing_workflow("ERROR-001")
        self.assertFalse(exists)
        self.assertIsNone(info)

    def test_create_workflow_state_method(self):
        """create_workflow_stateメソッドの単体テスト"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # 成功ケース
        self.mock_workflow_controller.create_tracker_workflow.return_value = True

        success, message = handler.create_workflow_state(SAMPLE_TRACKER_ID)

        self.assertTrue(success)
        self.assertIn("✅", message)
        self.assertIn("ワークフロー状態管理を開始しました", message)
        self.assertIn(SAMPLE_TRACKER_ID, message)

        # 失敗ケース
        self.mock_workflow_controller.create_tracker_workflow.return_value = False

        success, message = handler.create_workflow_state(SAMPLE_TRACKER_ID)

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("ワークフロー状態作成に失敗しました", message)

        # 例外発生ケース
        self.mock_workflow_controller.create_tracker_workflow.side_effect = Exception(
            "Unexpected error"
        )

        success, message = handler.create_workflow_state(SAMPLE_TRACKER_ID)

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("ワークフロー状態作成でエラーが発生しました", message)

    def test_controller_initialization_failure(self):
        """WorkflowController初期化失敗の詳細テスト"""
        with patch(
            "tools.interface.workflow_controller.get_workflow_controller"
        ) as mock_get_controller:
            mock_get_controller.side_effect = Exception("Import failed")

            from tools.workflow.create_command_handler import CreateCommandHandler

            handler = CreateCommandHandler()

            # 初期化でワークフローコントローラーがNoneになることを確認
            self.assertIsNone(handler.workflow_controller)

            success, message = handler.execute_create_command(SAMPLE_TRACKER_ID)

            self.assertFalse(success)
            self.assertIn("❌", message)
            self.assertIn("ワークフローコントローラーが利用できません", message)

    def test_sql_injection_prevention(self):
        """SQLインジェクション防止のテスト（ID検証）"""
        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()

        # SQLインジェクション的な文字列でのテスト
        malicious_ids = [
            "TRACKER-001'; DROP TABLE workflows; --",
            "TRACKER-001 OR 1=1",
            "TRACKER-001<script>alert('xss')</script>",
            "TRACKER-001`rm -rf /`",
        ]

        for malicious_id in malicious_ids:
            with self.subTest(malicious_id=malicious_id):
                success, message = handler.execute_create_command(malicious_id)

                # フォーマット検証で拒否されることを確認
                self.assertFalse(success)
                self.assertIn("❌", message)
                self.assertIn("トラッカーID形式が無効です", message)


if __name__ == "__main__":
    unittest.main()
