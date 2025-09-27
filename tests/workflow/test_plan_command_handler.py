"""
PlanCommandHandler ユニットテスト
Google Sheets起票コマンドのテスト
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tests.workflow.fixtures.workflow_fixtures import WorkflowTestBase, CLITestHelper
from tests.workflow.fixtures.mock_data import MockData, SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_DETAILS, SAMPLE_LONG_DETAILS, SAMPLE_AUTHOR_NAME


class TestPlanCommandHandler(WorkflowTestBase):
    """PlanCommandHandlerのテストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # PlanCommandHandlerのモック設定
        self.mock_progress_manager = Mock()

        # 作成されるタスクのモック
        mock_task = Mock()
        mock_task.tracker_id = SAMPLE_TRACKER_ID
        mock_task.status.value = "planning"
        mock_task.created_date = "2025-09-27 12:00:00"

        # get_task: 最初はNone（既存なし）、2回目は作成されたタスク（作成後確認）
        self.mock_progress_manager.get_task.side_effect = [None, mock_task]
        self.mock_progress_manager.create_task.return_value = mock_task

        # configとclientのモック
        mock_config = Mock()
        mock_config.sheet_url = "https://docs.google.com/spreadsheets/d/test/edit"
        self.mock_progress_manager.config = mock_config

        mock_client = Mock()
        self.mock_progress_manager.client = mock_client

        self.add_mock('tools.workflow.plan_command_handler.ProgressManager',
                     return_value=self.mock_progress_manager)

        # get_default_configのモック設定（ローカルインポート用）
        self.add_mock('tools.progress_tracker.config.get_default_config',
                     return_value=Mock())

    def test_plan_command_success(self):
        """planコマンド正常実行テスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 正常実行
        success, message = handler.execute_plan_command(
            SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_DETAILS
        )

        self.assertTrue(success)
        self.assertIn("✅", message)
        self.assertIn("Google Sheetsにトラッカーを起票しました", message)
        self.assertIn(SAMPLE_TRACKER_ID, message)

    def test_plan_command_invalid_tracker_id(self):
        """無効なトラッカーIDのテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 無効なトラッカーIDでテスト
        invalid_ids = ["", "invalid", "tracker-001", "TRACKER_001", "123-TRACKER"]

        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                success, message = handler.execute_plan_command(
                    invalid_id, SAMPLE_SUMMARY, SAMPLE_DETAILS
                )

                self.assertFalse(success)
                self.assertIn("❌", message)
                self.assertIn("トラッカーID", message)

    def test_plan_command_empty_summary(self):
        """空の概要のテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        success, message = handler.execute_plan_command(
            SAMPLE_TRACKER_ID, "", SAMPLE_DETAILS
        )

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("概要が指定されていません", message)

    def test_plan_command_empty_details(self):
        """空の詳細のテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        success, message = handler.execute_plan_command(
            SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, ""
        )

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("詳細が指定されていません", message)

    def test_plan_command_long_details(self):
        """詳細文字数制限テスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        success, message = handler.execute_plan_command(
            SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_LONG_DETAILS
        )

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("詳細が文字数制限を超えています", message)
        self.assertIn("20,000", message)

    def test_plan_command_existing_tracker(self):
        """既存トラッカーIDのテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 既存のトラッカーを返すようにモック設定
        existing_task = MockData.MOCK_TASKS["TRACKER-001"]
        self.mock_progress_manager.get_task.return_value = existing_task

        success, message = handler.execute_plan_command(
            "TRACKER-001", SAMPLE_SUMMARY, SAMPLE_DETAILS
        )

        self.assertFalse(success)
        self.assertIn("⚠️", message)
        self.assertIn("トラッカーが既に存在します", message)
        self.assertIn("TRACKER-001", message)

    def test_plan_command_google_sheets_error(self):
        """Google Sheetsエラーのテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # Google Sheetsエラーを発生させる
        self.mock_progress_manager.create_task.side_effect = Exception("Google Sheets API Error")

        success, message = handler.execute_plan_command(
            "NEW-001", SAMPLE_SUMMARY, SAMPLE_DETAILS
        )

        self.assertFalse(success)
        self.assertIn("❌", message)
        self.assertIn("Google Sheets起票に失敗しました", message)

    def test_plan_command_priority_setting(self):
        """優先度設定のテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        priorities = ["highest", "high", "medium", "low"]

        for priority in priorities:
            with self.subTest(priority=priority):
                success, message = handler.execute_plan_command(
                    f"TEST-{priority.upper()}", SAMPLE_SUMMARY, SAMPLE_DETAILS, priority
                )

                self.assertTrue(success)
                self.assertIn("✅", message)
                self.assertIn(f"優先度: {priority}", message)

    def test_validate_inputs_method(self):
        """validate_inputsメソッドの単体テスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 正常ケース
        is_valid, error = handler.validate_inputs(SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_DETAILS)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # 無効ケースのテスト
        test_cases = [
            ("", SAMPLE_SUMMARY, SAMPLE_DETAILS, "トラッカーIDが指定されていません"),
            (SAMPLE_TRACKER_ID, "", SAMPLE_DETAILS, "概要が指定されていません"),
            (SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, "", "詳細が指定されていません"),
            ("invalid-id", SAMPLE_SUMMARY, SAMPLE_DETAILS, "トラッカーID形式が無効です"),
            (SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_LONG_DETAILS, "詳細が文字数制限を超えています")
        ]

        for tracker_id, summary, details, expected_error in test_cases:
            with self.subTest(tracker_id=tracker_id, expected_error=expected_error):
                is_valid, error = handler.validate_inputs(tracker_id, summary, details)
                self.assertFalse(is_valid)
                self.assertIn(expected_error, error)

    def test_tracker_id_format_validation(self):
        """トラッカーID形式検証の単体テスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 有効な形式
        valid_ids = ["TRACKER-001", "KIRO-006", "QUAL-044", "TEST-123", "ABC-999"]
        for valid_id in valid_ids:
            with self.subTest(valid_id=valid_id):
                self.assertTrue(handler._validate_tracker_id_format(valid_id))

        # 無効な形式
        invalid_ids = ["", "tracker-001", "TRACKER_001", "123-TRACKER", "TRACKER", "TRACKER-", "-001"]
        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                self.assertFalse(handler._validate_tracker_id_format(invalid_id))

    def test_check_existing_tracker_method(self):
        """check_existing_trackerメソッドの単体テスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()

        # 既存トラッカーの場合
        existing_task = MockData.MOCK_TASKS["TRACKER-001"]
        self.mock_progress_manager.get_task.return_value = existing_task

        exists, info = handler.check_existing_tracker("TRACKER-001")
        self.assertTrue(exists)
        self.assertIsNotNone(info)
        self.assertEqual(info["tracker_id"], "TRACKER-001")

        # 存在しないトラッカーの場合
        self.mock_progress_manager.get_task.return_value = None

        exists, info = handler.check_existing_tracker("NON-EXISTENT-001")
        self.assertFalse(exists)
        self.assertIsNone(info)

    def test_progress_manager_initialization_failure(self):
        """ProgressManager初期化失敗のテスト"""
        with patch('tools.workflow.plan_command_handler.ProgressManager') as mock_pm_class:
            mock_pm_class.side_effect = Exception("Initialization failed")

            from tools.workflow.plan_command_handler import PlanCommandHandler

            handler = PlanCommandHandler()

            success, message = handler.execute_plan_command(
                SAMPLE_TRACKER_ID, SAMPLE_SUMMARY, SAMPLE_DETAILS, SAMPLE_AUTHOR_NAME
            )

            self.assertFalse(success)
            self.assertIn("❌", message)
            self.assertIn("Google Sheets連携が利用できません", message)


    def test_get_usage_help_method(self):
        """get_usage_helpメソッドのテスト"""
        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()
        help_text = handler.get_usage_help()

        self.assertIn("planコマンド使用方法", help_text)
        self.assertIn("python tools/workflow/workflow_cli.py plan", help_text)
        self.assertIn("20,000", help_text)  # 文字数制限の記載確認


if __name__ == '__main__':
    unittest.main()