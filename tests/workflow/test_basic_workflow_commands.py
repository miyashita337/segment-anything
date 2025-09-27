"""
基本ワークフローコマンドのユニットテスト
status, instructions, step コマンドのテスト
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
from tests.workflow.fixtures.mock_data import MockData, SAMPLE_TRACKER_ID


class TestBasicWorkflowCommands(WorkflowTestBase):
    """基本ワークフローコマンドのテストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # WorkflowControllerのモック設定
        self.mock_workflow_controller = self.create_mock_workflow_controller()
        self.add_mock('tools.interface.workflow_controller.get_workflow_controller',
                     return_value=self.mock_workflow_controller)

        # ワークスペース検証のモック設定
        self.setup_workspace_validation_mocks(is_configured=True)

    def test_get_status_success(self):
        """statusコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import get_status

        # 正常な状態を返すよう設定
        self.mock_workflow_controller.get_workflow_status.return_value = MockData.MOCK_WORKFLOW_STATUS["TRACKER-001"]

        output = CLITestHelper.capture_output(get_status, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        # 期待される出力を確認
        self.assertIn("📋", stdout)
        self.assertIn(SAMPLE_TRACKER_ID, stdout)
        self.assertIn("ワークフロー状態", stdout)
        self.assertIn("現在のフェーズ", stdout)
        self.assertIn("現在のステップ", stdout)
        self.assertIn("進行可能", stdout)

    def test_get_status_tracker_not_found(self):
        """statusコマンド - トラッカー未発見テスト"""
        from tools.workflow.workflow_cli import get_status

        # トラッカーが見つからない状態をシミュレート
        self.mock_workflow_controller.get_workflow_status.return_value = {
            "error": "Tracker not found",
            "status": "not_found"
        }

        output = CLITestHelper.capture_output(get_status, "NON-EXISTENT-001")

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("エラー", stdout)

    def test_get_status_workspace_not_configured(self):
        """statusコマンド - ワークスペース未設定テスト"""
        from tools.workflow.workflow_cli import get_status

        # ワークスペース未設定のモック設定
        self.setup_workspace_validation_mocks(is_configured=False)

        output = CLITestHelper.capture_output(get_status, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("ワークスペース設定エラー", stdout)

    def test_get_status_controller_unavailable(self):
        """statusコマンド - コントローラー利用不可テスト"""
        from tools.workflow.workflow_cli import get_status

        # WorkflowControllerが利用できない状態
        self.add_mock('tools.interface.workflow_controller.get_workflow_controller',
                     return_value=None)

        output = CLITestHelper.capture_output(get_status, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])

    def test_get_instructions_success(self):
        """instructionsコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import get_instructions

        # 正常な指示を返すよう設定
        mock_instructions = Mock()
        mock_instructions.step_id = "branch_verification"
        mock_instructions.title = "ブランチ検証"
        mock_instructions.description = "feature/TRACKER-001 ブランチで作業していることを確認"
        mock_instructions.required_actions = ["ブランチ確認", "初期コミット"]
        mock_instructions.validation_criteria = ["正しいブランチ", "コミット履歴確認"]
        mock_instructions.approval_required = False
        mock_instructions.can_proceed = True
        mock_instructions.blocking_reasons = []

        self.mock_workflow_controller.get_current_step_instructions.return_value = mock_instructions

        output = CLITestHelper.capture_output(get_instructions, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        # 期待される出力を確認
        self.assertIn("📝", stdout)
        self.assertIn(SAMPLE_TRACKER_ID, stdout)
        self.assertIn("現在のステップ指示", stdout)
        self.assertIn("ブランチ検証", stdout)
        self.assertIn("🔧 必要なアクション", stdout)
        self.assertIn("✅ 検証基準", stdout)
        self.assertIn("✅ 進行準備完了", stdout)

    def test_get_instructions_with_approval_required(self):
        """instructionsコマンド - 承認必要テスト"""
        from tools.workflow.workflow_cli import get_instructions

        # 承認が必要な指示を設定
        mock_instructions = Mock()
        mock_instructions.step_id = "implementation"
        mock_instructions.title = "実装フェーズ"
        mock_instructions.description = "ワークフロー強制実行システムの実装"
        mock_instructions.required_actions = ["コード実装", "ユニットテスト作成"]
        mock_instructions.validation_criteria = ["テスト通過", "コードレビュー"]
        mock_instructions.approval_required = True
        mock_instructions.can_proceed = False
        mock_instructions.blocking_reasons = ["承認待ち"]

        self.mock_workflow_controller.get_current_step_instructions.return_value = mock_instructions

        output = CLITestHelper.capture_output(get_instructions, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("⚠️", stdout)
        self.assertIn("このステップには人間の承認が必要です", stdout)
        self.assertIn("🚫 進行できません", stdout)
        self.assertIn("承認待ち", stdout)

    def test_get_instructions_not_found(self):
        """instructionsコマンド - 指示未発見テスト"""
        from tools.workflow.workflow_cli import get_instructions

        # 指示が見つからない状態
        self.mock_workflow_controller.get_current_step_instructions.return_value = None

        output = CLITestHelper.capture_output(get_instructions, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("トラッカーの指示が見つかりません", stdout)

    def test_attempt_step_success(self):
        """stepコマンド正常実行テスト"""
        from tools.workflow.workflow_cli import attempt_step

        # 正常なステップ完了を設定
        mock_result = Mock()
        mock_result.status = "completed"
        mock_result.message = "ステップが正常に完了しました"
        mock_result.next_step = "next_step_id"
        mock_result.approval_id = None

        self.mock_workflow_controller.attempt_step_completion.return_value = mock_result

        output = CLITestHelper.capture_output(attempt_step, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🔄", stdout)
        self.assertIn("現在のステップ完了を試行中", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("ステップが正常に完了しました", stdout)
        self.assertIn("次のステップ", stdout)

    def test_attempt_step_pending_approval(self):
        """stepコマンド - 承認待ちテスト"""
        from tools.workflow.workflow_cli import attempt_step

        # 承認待ち状態を設定
        mock_result = Mock()
        mock_result.status = "pending_approval"
        mock_result.message = "ステップには承認が必要です"
        mock_result.approval_id = "APP-001"
        mock_result.next_step = None

        self.mock_workflow_controller.attempt_step_completion.return_value = mock_result

        output = CLITestHelper.capture_output(attempt_step, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("⏳", stdout)
        self.assertIn("ステップには承認が必要です", stdout)
        self.assertIn("承認ID", stdout)
        self.assertIn("APP-001", stdout)

    def test_attempt_step_blocked(self):
        """stepコマンド - ブロック状態テスト"""
        from tools.workflow.workflow_cli import attempt_step

        # ブロック状態を設定
        mock_result = Mock()
        mock_result.status = "blocked"
        mock_result.message = "前提条件が満たされていません"
        mock_result.approval_id = None
        mock_result.next_step = None

        self.mock_workflow_controller.attempt_step_completion.return_value = mock_result

        output = CLITestHelper.capture_output(attempt_step, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("🚫", stdout)
        self.assertIn("ステップがブロックされています", stdout)

    def test_attempt_step_failed(self):
        """stepコマンド - 失敗テスト"""
        from tools.workflow.workflow_cli import attempt_step

        # 失敗状態を設定
        mock_result = Mock()
        mock_result.status = "failed"
        mock_result.message = "検証に失敗しました"
        mock_result.approval_id = None
        mock_result.next_step = None

        self.mock_workflow_controller.attempt_step_completion.return_value = mock_result

        output = CLITestHelper.capture_output(attempt_step, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("ステップが失敗しました", stdout)

    def test_attempt_step_controller_unavailable(self):
        """stepコマンド - コントローラー利用不可テスト"""
        from tools.workflow.workflow_cli import attempt_step

        # WorkflowControllerが利用できない状態
        self.add_mock('tools.interface.workflow_controller.get_workflow_controller',
                     return_value=None)

        output = CLITestHelper.capture_output(attempt_step, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])

    def test_status_with_background_process(self):
        """statusコマンド - バックグラウンドプロセス情報表示テスト"""
        from tools.workflow.workflow_cli import get_status

        # バックグラウンドプロセス情報を含む状態
        status_with_process = MockData.MOCK_WORKFLOW_STATUS["TRACKER-001"].copy()
        status_with_process["background_process"] = {
            "status": "running",
            "step": "extraction",
            "pid": 12345,
            "log_file": "/tmp/test.log"
        }

        self.mock_workflow_controller.get_workflow_status.return_value = status_with_process

        output = CLITestHelper.capture_output(get_status, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🔄 バックグラウンドプロセス", stdout)
        self.assertIn("状態: running", stdout)
        self.assertIn("PID: 12345", stdout)

    def test_status_with_pending_approvals(self):
        """statusコマンド - 承認待ち表示テスト"""
        from tools.workflow.workflow_cli import get_status

        # 承認待ちを含む状態
        status_with_approval = MockData.MOCK_WORKFLOW_STATUS["KIRO-006"]

        self.mock_workflow_controller.get_workflow_status.return_value = status_with_approval

        output = CLITestHelper.capture_output(get_status, "KIRO-006")

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("承認待ち", stdout)
        self.assertIn("⏳", stdout)
        self.assertIn("APP-001", stdout)


if __name__ == '__main__':
    unittest.main()