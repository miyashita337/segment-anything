"""
ワークフローCLI統合テスト
workflow_cli.py のメインエントリーポイントとコマンド解析のテスト
"""

import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tests.workflow.fixtures.mock_data import (
    SAMPLE_AUTHOR_NAME,
    SAMPLE_DETAILS,
    SAMPLE_SUMMARY,
    SAMPLE_TRACKER_ID,
    MockData,
)
from tests.workflow.fixtures.workflow_fixtures import CLITestHelper, WorkflowTestBase


class TestWorkflowCLI(WorkflowTestBase):
    """ワークフローCLI統合テストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # 全コマンドハンドラーのモック設定
        self.setup_all_command_mocks()

    def setup_all_command_mocks(self):
        """全コマンドハンドラーのモック設定"""
        # Plan/Create/SubAgentCommandHandlerのモック
        self.setup_plan_command_mocks()
        self.setup_create_command_mocks()
        self.setup_workspace_validation_mocks(is_configured=True)

        # WorkflowControllerのモック
        self.mock_workflow_controller = self.create_mock_workflow_controller()
        self.add_mock(
            "tools.interface.workflow_controller.get_workflow_controller",
            return_value=self.mock_workflow_controller,
        )

        # SubAgentHandlerのモック
        self.mock_subagent_handler = self.create_mock_subagent_handler()
        self.add_mock(
            "tools.workflow.workflow_cli.SubAgentCommandHandler",
            return_value=self.mock_subagent_handler,
        )

        # ProgressManagerのモック（sheets用）
        self.mock_progress_manager = self.create_mock_progress_manager()
        self.add_mock(
            "tools.workflow.workflow_cli.ProgressManager", return_value=self.mock_progress_manager
        )

    @patch("sys.argv")
    def test_main_plan_command(self, mock_argv):
        """main関数 - planコマンドテスト"""
        # planコマンドの引数を設定
        mock_argv.__getitem__.side_effect = [
            "workflow_cli.py",
            "plan",
            SAMPLE_TRACKER_ID,
            SAMPLE_SUMMARY,
            SAMPLE_DETAILS,
            SAMPLE_AUTHOR_NAME,
        ]
        mock_argv.__len__.return_value = 6

        from tools.workflow.workflow_cli import main

        # メイン関数実行
        with patch("sys.exit") as mock_exit:
            main()

            # 成功で終了することを確認
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_create_command(self, mock_argv):
        """main関数 - createコマンドテスト"""
        # createコマンドの引数を設定
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "create", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_status_command(self, mock_argv):
        """main関数 - statusコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "status", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_instructions_command(self, mock_argv):
        """main関数 - instructionsコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "instructions", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_step_command(self, mock_argv):
        """main関数 - stepコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "step", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_approvals_command(self, mock_argv):
        """main関数 - approvalsコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "approvals"]
        mock_argv.__len__.return_value = 2

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_process_command(self, mock_argv):
        """main関数 - processコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "process", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_sheets_command(self, mock_argv):
        """main関数 - sheetsコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "sheets", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_template_command(self, mock_argv):
        """main関数 - templateコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "template", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        # ファイル操作をモック
        with patch("builtins.open"), patch("os.makedirs"), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_guide_command(self, mock_argv):
        """main関数 - guideコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "guide"]
        mock_argv.__len__.return_value = 2

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_subagent_extraction_command(self, mock_argv):
        """main関数 - subagent-extractionコマンドテスト"""
        mock_argv.__getitem__.side_effect = [
            "workflow_cli.py",
            "subagent-extraction",
            SAMPLE_TRACKER_ID,
        ]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_subagent_status_command(self, mock_argv):
        """main関数 - subagent-statusコマンドテスト"""
        mock_argv.__getitem__.side_effect = [
            "workflow_cli.py",
            "subagent-status",
            SAMPLE_TRACKER_ID,
        ]
        mock_argv.__len__.return_value = 3

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch("sys.argv")
    def test_main_no_command(self, mock_argv):
        """main関数 - コマンドなしテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py"]
        mock_argv.__len__.return_value = 1

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit, patch(
            "argparse.ArgumentParser.print_help"
        ) as mock_help:
            main()
            mock_help.assert_called_once()
            mock_exit.assert_called_once_with(1)

    @patch("sys.argv")
    def test_main_invalid_command(self, mock_argv):
        """main関数 - 無効なコマンドテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "invalid-command"]
        mock_argv.__len__.return_value = 2

        from tools.workflow.workflow_cli import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

    @patch("sys.argv")
    def test_main_keyboard_interrupt(self, mock_argv):
        """main関数 - KeyboardInterruptテスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "status", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        # get_statusでKeyboardInterruptを発生させる
        self.add_mock("tools.workflow.workflow_cli.get_status", side_effect=KeyboardInterrupt())

        from tools.workflow.workflow_cli import main

        output = CLITestHelper.capture_output(main)

        # KeyboardInterruptが適切にハンドリングされることを確認
        self.assertEqual(output["result"], 1)

    @patch("sys.argv")
    def test_main_unexpected_exception(self, mock_argv):
        """main関数 - 予期しない例外テスト"""
        mock_argv.__getitem__.side_effect = ["workflow_cli.py", "status", SAMPLE_TRACKER_ID]
        mock_argv.__len__.return_value = 3

        # get_statusで例外を発生させる
        self.add_mock(
            "tools.workflow.workflow_cli.get_status", side_effect=Exception("Unexpected error")
        )

        from tools.workflow.workflow_cli import main

        output = CLITestHelper.capture_output(main)

        # 例外が適切にハンドリングされることを確認
        self.assertEqual(output["result"], 1)

    def test_command_function_mapping(self):
        """コマンド関数マッピングのテスト"""
        from tools.workflow import workflow_cli

        # 各コマンドに対応する関数が存在することを確認
        command_functions = {
            "plan": "plan_tracker",
            "create": "create_tracker",
            "status": "get_status",
            "instructions": "get_instructions",
            "step": "attempt_step",
            "approvals": "list_approvals",
            "process": "check_process",
            "sheets": "check_sheets_status",
            "template": "generate_template",
            "guide": "show_guide",
        }

        for command, function_name in command_functions.items():
            with self.subTest(command=command, function=function_name):
                self.assertTrue(
                    hasattr(workflow_cli, function_name),
                    f"Function {function_name} not found for command {command}",
                )

    def test_subagent_command_function_mapping(self):
        """SubAgentコマンド関数マッピングのテスト"""
        from tools.workflow import workflow_cli

        # SubAgentコマンドに対応する関数が存在することを確認
        subagent_functions = {
            "subagent_extraction",
            "subagent_status",
            "subagent_wait",
            "subagent_retry",
            "subagent_terminate",
            "subagent_cleanup",
            "subagent_locks_status",
            "subagent_cleanup_all",
            "subagent_auto_retry_check",
            "subagent_auto_retry",
            "subagent_auto_retry_all",
        }

        for function_name in subagent_functions:
            with self.subTest(function=function_name):
                self.assertTrue(
                    hasattr(workflow_cli, function_name),
                    f"SubAgent function {function_name} not found",
                )

    def test_virtual_environment_check(self):
        """仮想環境チェック関数のテスト"""
        from tools.workflow.workflow_cli import check_virtual_environment

        # CI環境でのテスト
        with patch("os.environ.get") as mock_env_get:
            mock_env_get.side_effect = (
                lambda key, default=None: "true" if key == "GITHUB_ACTIONS" else default
            )

            result = check_virtual_environment()
            self.assertTrue(result)

        # 正常な仮想環境でのテスト
        with patch("os.environ.get") as mock_env_get:
            mock_env_get.side_effect = (
                lambda key, default=None: "/path/to/sam-env" if key == "VIRTUAL_ENV" else default
            )

            output = CLITestHelper.capture_output(check_virtual_environment)
            self.assertTrue(output["result"])
            self.assertIn("✅", output["stdout"])

        # 仮想環境未設定でのテスト
        with patch("os.environ.get") as mock_env_get:
            mock_env_get.return_value = None

            output = CLITestHelper.capture_output(check_virtual_environment)
            self.assertFalse(output["result"])
            self.assertIn("❌", output["stdout"])

    def test_argument_parsing_plan_command(self):
        """引数解析 - planコマンドテスト"""
        from tools.workflow.workflow_cli import main

        test_args = [
            "workflow_cli.py",
            "plan",
            "TEST-001",
            "概要テスト",
            "詳細テスト",
            "test_author",
            "--priority",
            "high",
        ]

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_argument_parsing_subagent_extraction(self):
        """引数解析 - subagent-extractionコマンドテスト"""
        from tools.workflow.workflow_cli import main

        test_args = [
            "workflow_cli.py",
            "subagent-extraction",
            "TEST-001",
            "/custom/path",
            "--max-files",
            "50",
        ]

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_argument_parsing_subagent_wait(self):
        """引数解析 - subagent-waitコマンドテスト"""
        from tools.workflow.workflow_cli import main

        test_args = ["workflow_cli.py", "subagent-wait", "TEST-001", "--timeout", "30"]

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_argument_parsing_subagent_terminate(self):
        """引数解析 - subagent-terminateコマンドテスト"""
        from tools.workflow.workflow_cli import main

        test_args = ["workflow_cli.py", "subagent-terminate", "TEST-001", "--force"]

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_help_text_generation(self):
        """ヘルプテキスト生成のテスト"""
        from tools.workflow.workflow_cli import main

        with patch("sys.argv", ["workflow_cli.py", "--help"]), patch(
            "sys.exit"
        ) as mock_exit, patch("argparse.ArgumentParser.print_help") as mock_help:
            try:
                main()
            except SystemExit:
                pass

            # ヘルプが表示されることを確認
            mock_help.assert_called_once()

    def test_error_handling_in_command_execution(self):
        """コマンド実行でのエラーハンドリングテスト"""
        # planコマンドでエラーを発生させる
        self.add_mock("tools.workflow.workflow_cli.plan_tracker", return_value=False)

        from tools.workflow.workflow_cli import main

        test_args = ["workflow_cli.py", "plan", "TEST-001", "概要", "詳細", "author"]

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
