"""
管理コマンドのユニットテスト
approvals, process, sheets, template, guide コマンドのテスト
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, mock_open, patch

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tests.workflow.fixtures.mock_data import SAMPLE_TRACKER_ID, MockData
from tests.workflow.fixtures.workflow_fixtures import CLITestHelper, WorkflowTestBase


class TestManagementCommands(WorkflowTestBase):
    """管理コマンドのテストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # WorkflowControllerのモック設定
        self.mock_workflow_controller = self.create_mock_workflow_controller()
        self.add_mock(
            "tools.interface.workflow_controller.get_workflow_controller",
            return_value=self.mock_workflow_controller,
        )

    def test_list_approvals_success(self):
        """approvalsコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import list_approvals

        # 承認待ちリストを設定
        pending_approvals = [
            {
                "approval_id": "APP-001",
                "tracker_id": "KIRO-006",
                "step_name": "implementation",
                "priority": "high",
                "requested_at": "2025-09-27 10:00:00",
                "time_remaining_hours": 23.5,
                "approval_criteria": ["設計レビュー完了", "技術仕様確認"],
            },
            {
                "approval_id": "APP-002",
                "tracker_id": "TRACKER-001",
                "step_name": "quality_check",
                "priority": "medium",
                "requested_at": "2025-09-27 11:00:00",
                "time_remaining_hours": 22.0,
                "approval_criteria": ["品質確認", "テスト結果レビュー"],
            },
        ]

        self.mock_workflow_controller.approval_controller.list_pending_approvals.return_value = (
            pending_approvals
        )

        output = CLITestHelper.capture_output(list_approvals)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        # 期待される出力を確認
        self.assertIn("⏳ 承認待ち", stdout)
        self.assertIn("📋 APP-001", stdout)
        self.assertIn("📋 APP-002", stdout)
        self.assertIn("KIRO-006", stdout)
        self.assertIn("TRACKER-001", stdout)
        self.assertIn("優先度: high", stdout)
        self.assertIn("優先度: medium", stdout)

    def test_list_approvals_empty(self):
        """approvalsコマンド - 承認待ちなしテスト"""
        from tools.workflow.workflow_cli import list_approvals

        # 空の承認待ちリストを設定
        self.mock_workflow_controller.approval_controller.list_pending_approvals.return_value = []

        output = CLITestHelper.capture_output(list_approvals)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        self.assertIn("✅ 承認待ちはありません", stdout)

    def test_list_approvals_controller_unavailable(self):
        """approvalsコマンド - コントローラー利用不可テスト"""
        from tools.workflow.workflow_cli import list_approvals

        # approval_controllerが利用できない状態
        self.mock_workflow_controller.approval_controller = None

        output = CLITestHelper.capture_output(list_approvals)

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("承認コントローラーが利用できません", stdout)

    def test_check_process_success(self):
        """processコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import check_process

        # プロセス情報を設定
        process_status = {
            "status": "running",
            "step": "extraction",
            "started_at": "2025-09-27 10:30:00",
            "pid": 12345,
            "log_file": "/tmp/test.log",
        }

        self.mock_workflow_controller.executor.check_process_status.return_value = process_status

        output = CLITestHelper.capture_output(check_process, SAMPLE_TRACKER_ID)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        # 期待される出力を確認
        self.assertIn("🔄", stdout)
        self.assertIn(SAMPLE_TRACKER_ID, stdout)
        self.assertIn("バックグラウンドプロセス", stdout)
        self.assertIn("状態: running", stdout)
        self.assertIn("PID: 12345", stdout)
        self.assertIn("ログ: /tmp/test.log", stdout)

    def test_check_process_completed(self):
        """processコマンド - 完了プロセステスト"""
        from tools.workflow.workflow_cli import check_process

        # 完了したプロセス情報を設定
        process_status = {
            "status": "completed",
            "step": "extraction",
            "started_at": "2025-09-27 10:30:00",
            "completed_at": "2025-09-27 11:00:00",
            "return_code": 0,
            "validation": "success",
        }

        self.mock_workflow_controller.executor.check_process_status.return_value = process_status

        output = CLITestHelper.capture_output(check_process, SAMPLE_TRACKER_ID)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        self.assertIn("状態: completed", stdout)
        self.assertIn("完了時刻", stdout)
        self.assertIn("リターンコード: 0", stdout)
        self.assertIn("検証: success", stdout)

    def test_check_process_not_running(self):
        """processコマンド - プロセス未実行テスト"""
        from tools.workflow.workflow_cli import check_process

        # プロセスが実行されていない状態
        self.mock_workflow_controller.executor.check_process_status.return_value = None

        output = CLITestHelper.capture_output(check_process, SAMPLE_TRACKER_ID)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        self.assertIn("ℹ️", stdout)
        self.assertIn("バックグラウンドプロセスは実行されていません", stdout)

    def test_check_process_executor_unavailable(self):
        """processコマンド - エグゼキューター利用不可テスト"""
        from tools.workflow.workflow_cli import check_process

        # executorが利用できない状態
        self.mock_workflow_controller.executor = None

        output = CLITestHelper.capture_output(check_process, SAMPLE_TRACKER_ID)

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("エグゼキューターが利用できません", stdout)

    def test_check_sheets_status_success(self):
        """sheetsコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import check_sheets_status

        # ProgressManagerのモック設定
        mock_progress_manager = self.create_mock_progress_manager()
        mock_task = MockData.MOCK_TASKS[SAMPLE_TRACKER_ID]
        mock_progress_manager.get_task.return_value = mock_task

        with patch(
            "tools.workflow.workflow_cli.ProgressManager", return_value=mock_progress_manager
        ):
            output = CLITestHelper.capture_output(check_sheets_status, SAMPLE_TRACKER_ID)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        # 期待される出力を確認
        self.assertIn("📊 Google Sheets状態", stdout)
        self.assertIn(SAMPLE_TRACKER_ID, stdout)
        self.assertIn("説明:", stdout)
        self.assertIn("ステータス:", stdout)

    def test_check_sheets_status_not_found(self):
        """sheetsコマンド - トラッカー未発見テスト"""
        from tools.workflow.workflow_cli import check_sheets_status

        # ProgressManagerのモック設定
        mock_progress_manager = Mock()
        mock_progress_manager.get_task.return_value = None

        with patch(
            "tools.workflow.workflow_cli.ProgressManager", return_value=mock_progress_manager
        ):
            output = CLITestHelper.capture_output(check_sheets_status, "NON-EXISTENT-001")

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("Google Sheetsにトラッカーが見つかりません", stdout)

    def test_check_sheets_status_error(self):
        """sheetsコマンド - Google Sheetsエラーテスト"""
        from tools.workflow.workflow_cli import check_sheets_status

        # ProgressManagerでエラーが発生
        with patch("tools.workflow.workflow_cli.ProgressManager") as mock_pm_class:
            mock_pm_class.side_effect = Exception("Google Sheets connection failed")

            output = CLITestHelper.capture_output(check_sheets_status, SAMPLE_TRACKER_ID)

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("Google Sheets確認でエラーが発生", stdout)

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_generate_template_success(self, mock_makedirs, mock_file):
        """templateコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import generate_template

        # ワークフロー状態とインストラクションのモック設定
        status = MockData.MOCK_WORKFLOW_STATUS[SAMPLE_TRACKER_ID]
        instructions = Mock()
        instructions.title = "ブランチ検証"
        instructions.description = "feature/TEST-001 ブランチで作業していることを確認"
        instructions.required_actions = ["ブランチ確認", "初期コミット"]
        instructions.validation_criteria = ["正しいブランチ", "コミット履歴確認"]
        instructions.approval_required = False
        instructions.blocking_reasons = []

        self.mock_workflow_controller.get_workflow_status.return_value = status
        self.mock_workflow_controller.get_current_step_instructions.return_value = instructions

        output = CLITestHelper.capture_output(generate_template, SAMPLE_TRACKER_ID)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        # 期待される出力を確認
        self.assertIn("✅", stdout)
        self.assertIn("統合テンプレートを生成しました", stdout)
        self.assertIn("workspace/", stdout)

        # ファイル書き込みが呼ばれたことを確認
        mock_file.assert_called_once()
        mock_makedirs.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_generate_template_custom_output(self, mock_makedirs, mock_file):
        """templateコマンド - カスタム出力パステスト"""
        from tools.workflow.workflow_cli import generate_template

        custom_path = "/custom/path/template.md"

        output = CLITestHelper.capture_output(generate_template, SAMPLE_TRACKER_ID, custom_path)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        self.assertIn("✅", stdout)
        self.assertIn(custom_path, stdout)

    @patch("builtins.open")
    def test_generate_template_write_error(self, mock_open_func):
        """templateコマンド - ファイル書き込みエラーテスト"""
        from tools.workflow.workflow_cli import generate_template

        # ファイル書き込みでエラーを発生させる
        mock_open_func.side_effect = IOError("Permission denied")

        output = CLITestHelper.capture_output(generate_template, SAMPLE_TRACKER_ID)

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("テンプレート生成に失敗", stdout)

    def test_generate_template_controller_unavailable(self):
        """templateコマンド - コントローラー利用不可テスト"""
        from tools.workflow.workflow_cli import generate_template

        # WorkflowControllerが利用できない状態
        self.add_mock(
            "tools.interface.workflow_controller.get_workflow_controller", return_value=None
        )

        output = CLITestHelper.capture_output(generate_template, SAMPLE_TRACKER_ID)

        self.assertFalse(output["result"])
        stdout = output["stdout"]

        self.assertIn("❌", stdout)
        self.assertIn("ワークフローコントローラーが利用できません", stdout)

    def test_show_guide(self):
        """guideコマンドテスト"""
        from tools.workflow.workflow_cli import show_guide

        output = CLITestHelper.capture_output(show_guide)

        self.assertTrue(output["result"])
        stdout = output["stdout"]

        # 期待される出力を確認
        self.assertIn("🚀", stdout)
        self.assertIn("統合ワークフローガイド", stdout)
        self.assertIn("python tools/workflow/workflow_cli.py", stdout)
        self.assertIn("plan", stdout)
        self.assertIn("create", stdout)
        self.assertIn("step", stdout)
        self.assertIn("新しいワークフロー（推奨）", stdout)

    def test_template_content_generation(self):
        """テンプレート内容生成の詳細テスト"""
        from tools.workflow.workflow_cli import generate_template

        # 複雑な状態とインストラクションを設定
        complex_status = {
            "tracker_id": SAMPLE_TRACKER_ID,
            "current_phase": "phase_2",
            "current_step": "implementation",
            "can_proceed": False,
            "completed_steps": [
                {"step_id": "planning", "completed_at": "2025-09-27 10:00:00"},
                {"step_id": "design", "completed_at": "2025-09-27 11:00:00"},
            ],
            "pending_approvals": [{"approval_id": "APP-001", "title": "実装承認"}],
            "blocked_actions": [{"action": "step_completion", "reason": "承認待ちのため進行できません"}],
        }

        complex_instructions = Mock()
        complex_instructions.title = "実装フェーズ"
        complex_instructions.description = "システムの実装を行う"
        complex_instructions.required_actions = ["コード実装", "テスト作成", "ドキュメント更新"]
        complex_instructions.validation_criteria = ["テスト通過", "コードレビュー", "品質確認"]
        complex_instructions.approval_required = True
        complex_instructions.can_proceed = False
        complex_instructions.blocking_reasons = ["承認待ち", "前提条件未達成"]

        self.mock_workflow_controller.get_workflow_status.return_value = complex_status
        self.mock_workflow_controller.get_current_step_instructions.return_value = (
            complex_instructions
        )

        with patch("builtins.open", mock_open()) as mock_file, patch("os.makedirs"):
            output = CLITestHelper.capture_output(generate_template, SAMPLE_TRACKER_ID)

            self.assertTrue(output["result"])

            # ファイルに書き込まれた内容を確認
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)

            # テンプレート内容の確認
            self.assertIn(SAMPLE_TRACKER_ID, written_content)
            self.assertIn("実装フェーズ", written_content)
            self.assertIn("コード実装", written_content)
            self.assertIn("テスト作成", written_content)
            self.assertIn("承認が必要です", written_content)
            self.assertIn("ブロック理由", written_content)


if __name__ == "__main__":
    unittest.main()
