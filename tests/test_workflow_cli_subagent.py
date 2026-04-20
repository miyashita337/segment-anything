#!/usr/bin/env python3
"""
KIRO-024 サブエージェントトリガーロジックのユニットテスト

workflow_cli.py の step_requires_investigation 関数と
attempt_step 関数のマーカー出力をテストする

KIRO-024拡張: plan/createコマンドのinvestigation_steps追加テスト
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.workflow.workflow_cli import step_requires_investigation, _output_task_tool_marker


class TestStepRequiresInvestigation(unittest.TestCase):
    """step_requires_investigation関数のテスト"""

    def test_plan_step_requires_investigation(self):
        """KIRO-024: planステップがinvestigation_stepsに含まれることを検証"""
        result = step_requires_investigation("plan")
        self.assertTrue(
            result,
            "planは調査必須ステップのためTrueを返すべき"
        )

    def test_create_step_requires_investigation(self):
        """KIRO-024: createステップがinvestigation_stepsに含まれることを検証"""
        result = step_requires_investigation("create")
        self.assertTrue(
            result,
            "createは調査必須ステップのためTrueを返すべき"
        )

    def test_investigation_required_steps(self):
        """調査必須ステップでTrueを返すこと（plan/create含む8ステップ）"""
        investigation_steps = [
            "plan",                   # KIRO-024で追加
            "create",                 # KIRO-024で追加
            "sow_creation",
            "implementation",
            "quality_workflow",
            "subagent_validation",
            "testing",
            "dashboard_generation",
        ]

        for step in investigation_steps:
            with self.subTest(step=step):
                result = step_requires_investigation(step)
                self.assertTrue(
                    result,
                    f"{step} は調査必須ステップのためTrueを返すべき"
                )

    def test_investigation_not_required_steps(self):
        """調査不要ステップでFalseを返すこと"""
        non_investigation_steps = [
            "branch_verification",
            "sam_env_check",
            "google_sheets_sync",
            "subagent_extraction",
            "waiting_for_subagent",
            "final_validation",
            "pr_creation",
            "completion",
        ]

        for step in non_investigation_steps:
            with self.subTest(step=step):
                result = step_requires_investigation(step)
                self.assertFalse(
                    result,
                    f"{step} は調査不要ステップのためFalseを返すべき"
                )

    def test_empty_string(self):
        """空文字列でFalseを返すこと"""
        result = step_requires_investigation("")
        self.assertFalse(result, "空文字列はFalseを返すべき")

    def test_none_input(self):
        """Noneの場合Falseを返すこと"""
        # Noneを渡しても `in` 演算子はsetに対して正常に動作し、Falseを返す
        result = step_requires_investigation(None)
        self.assertFalse(result, "Noneを渡した場合はFalseを返すべき")

    def test_case_sensitivity(self):
        """大文字小文字を正しく区別すること"""
        # 大文字バリエーション
        self.assertFalse(
            step_requires_investigation("SOW_CREATION"),
            "大文字はFalseを返すべき（厳密マッチ）"
        )
        self.assertFalse(
            step_requires_investigation("Implementation"),
            "キャメルケースはFalseを返すべき（厳密マッチ）"
        )
        self.assertFalse(
            step_requires_investigation("TESTING"),
            "全大文字はFalseを返すべき（厳密マッチ）"
        )
        # KIRO-024: plan/createの大文字小文字も確認
        self.assertFalse(
            step_requires_investigation("PLAN"),
            "PLANはFalseを返すべき（厳密マッチ）"
        )
        self.assertFalse(
            step_requires_investigation("CREATE"),
            "CREATEはFalseを返すべき（厳密マッチ）"
        )

    def test_unknown_step(self):
        """未知のステップ名でFalseを返すこと"""
        unknown_steps = [
            "unknown_step",
            "random_task",
            "non_existent",
            "custom_step",
        ]

        for step in unknown_steps:
            with self.subTest(step=step):
                result = step_requires_investigation(step)
                self.assertFalse(
                    result,
                    f"{step} は未知のステップのためFalseを返すべき"
                )

    def test_total_investigation_steps_count(self):
        """KIRO-024: investigation_stepsの総数が8であることを検証"""
        all_investigation_steps = [
            "plan",
            "create",
            "sow_creation",
            "implementation",
            "testing",
            "subagent_validation",
            "quality_workflow",
            "dashboard_generation",
        ]
        # すべてのステップがTrueを返すことを確認
        for step in all_investigation_steps:
            self.assertTrue(
                step_requires_investigation(step),
                f"{step}がinvestigation_stepsに含まれていない"
            )


class TestOutputTaskToolMarker(unittest.TestCase):
    """_output_task_tool_marker()関数のテスト"""

    def test_marker_output_format(self):
        """マーカー出力フォーマットの検証"""
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            _output_task_tool_marker(
                "テスト理由",
                "テストタスク",
                "テスト詳細"
            )
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("🤖 テスト理由", output)
        self.assertIn("タスク: テストタスク", output)
        self.assertIn("調査対象: テスト詳細", output)
        self.assertIn("=" * 60, output)

    def test_marker_contains_all_required_elements(self):
        """マーカーが必須要素をすべて含むことを検証"""
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            _output_task_tool_marker(
                "Google Sheets接続エラー",
                "plan",
                "トラッカーID: TEST-001"
            )
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        required_elements = [
            "[TASK_TOOL_REQUIRED]",
            "🤖",
            "タスク:",
            "調査対象:",
        ]
        for element in required_elements:
            self.assertIn(element, output, f"必須要素 '{element}' がマーカー出力に含まれていない")


class TestPlanTrackerMarkerOutput(unittest.TestCase):
    """plan_tracker()のマーカー出力テスト"""

    @patch("config.workspace_config.get_workspace_config")
    def test_plan_workspace_config_failure_outputs_marker(self, mock_get_config):
        """ワークスペース設定失敗時にマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import plan_tracker

        mock_config = MagicMock()
        mock_config.set_workspace_config.return_value = False
        mock_get_config.return_value = mock_config

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = plan_tracker("TEST-001", "テスト概要", "テスト詳細", "zundamon")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("ワークスペース設定エラー", output)

    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    @patch("config.workspace_config.get_workspace_config")
    def test_plan_success_outputs_next_step_marker(self, mock_get_config, mock_handler_class):
        """plan成功時に次ステップ(create)のマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import plan_tracker

        mock_config = MagicMock()
        mock_config.set_workspace_config.return_value = True
        mock_get_config.return_value = mock_config

        mock_handler = MagicMock()
        mock_handler.execute_plan_command.return_value = (True, "✅ Google Sheetsに起票しました")
        mock_handler_class.return_value = mock_handler

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = plan_tracker("TEST-001", "テスト概要", "テスト詳細", "zundamon")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertTrue(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("create", output)
        self.assertIn("次のステップ", output)

    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    @patch("config.workspace_config.get_workspace_config")
    def test_plan_sheets_failure_outputs_marker(self, mock_get_config, mock_handler_class):
        """Google Sheets起票失敗時にマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import plan_tracker

        mock_config = MagicMock()
        mock_config.set_workspace_config.return_value = True
        mock_get_config.return_value = mock_config

        mock_handler = MagicMock()
        mock_handler.execute_plan_command.return_value = (False, "❌ Google Sheets接続エラー")
        mock_handler_class.return_value = mock_handler

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = plan_tracker("TEST-001", "テスト概要", "テスト詳細", "zundamon")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("Google Sheets接続エラー", output)

    @patch("config.workspace_config.get_workspace_config")
    def test_plan_exception_outputs_marker(self, mock_get_config):
        """plan実行時の例外でマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import plan_tracker

        mock_get_config.side_effect = Exception("テスト例外")

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = plan_tracker("TEST-001", "テスト概要", "テスト詳細", "zundamon")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("例外発生", output)
        self.assertIn("Exception", output)


class TestCreateTrackerMarkerOutput(unittest.TestCase):
    """create_tracker()のマーカー出力テスト"""

    @patch("config.workspace_config.validate_tracker_setup")
    def test_create_validation_failure_outputs_marker(self, mock_validate):
        """ワークスペース設定検証失敗時にマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import create_tracker

        mock_validate.return_value = {
            "is_configured": False,
            "errors": ["❌ ワークスペースが設定されていません"]
        }

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = create_tracker("TEST-001")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("ワークスペース設定が未完了", output)
        self.assertIn("planコマンドを先に実行", output)

    @patch("tools.workflow.create_command_handler.CreateCommandHandler")
    @patch("config.workspace_config.validate_tracker_setup")
    def test_create_sqlite_failure_outputs_marker(self, mock_validate, mock_handler_class):
        """SQLiteワークフロー作成失敗時にマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import create_tracker

        mock_validate.return_value = {
            "is_configured": True,
            "errors": []
        }

        mock_handler = MagicMock()
        mock_handler.execute_create_command.return_value = (False, "❌ SQLite書き込みエラー")
        mock_handler_class.return_value = mock_handler

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = create_tracker("TEST-001")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("SQLiteワークフロー作成エラー", output)

    @patch("tools.workflow.create_command_handler.CreateCommandHandler")
    @patch("config.workspace_config.validate_tracker_setup")
    def test_create_success_no_error_marker(self, mock_validate, mock_handler_class):
        """create成功時にエラーマーカーが出力されないことを検証"""
        from tools.workflow.workflow_cli import create_tracker

        mock_validate.return_value = {
            "is_configured": True,
            "errors": []
        }

        mock_handler = MagicMock()
        mock_handler.execute_create_command.return_value = (True, "✅ ワークフローを作成しました")
        mock_handler_class.return_value = mock_handler

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = create_tracker("TEST-001")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertTrue(result)
        # 成功時はエラー関連のマーカーは出力されない
        self.assertNotIn("SQLiteワークフロー作成エラー", output)
        self.assertNotIn("ワークスペース設定が未完了", output)

    @patch("config.workspace_config.validate_tracker_setup")
    def test_create_exception_outputs_marker(self, mock_validate):
        """create実行時の例外でマーカーが出力されることを検証"""
        from tools.workflow.workflow_cli import create_tracker

        mock_validate.side_effect = Exception("SQLite接続エラー")

        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            result = create_tracker("TEST-001")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertFalse(result)
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("例外発生", output)
        self.assertIn("Exception", output)


class TestAttemptStepMarkerOutput(unittest.TestCase):
    """attempt_step関数のマーカー出力テスト"""

    def setUp(self):
        """テスト前の準備"""
        self.tracker_id = "TEST-001"

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_marker_output_for_investigation_step(self, mock_get_controller):
        """調査必須ステップへ遷移時にマーカーが出力されること"""
        # モックの設定
        mock_controller = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.message = "ステップ完了"
        mock_result.next_step = "implementation"  # 調査必須ステップ
        mock_controller.attempt_step_completion.return_value = mock_result
        mock_get_controller.return_value = mock_controller

        # 標準出力をキャプチャ
        from tools.workflow.workflow_cli import attempt_step

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            attempt_step(self.tracker_id)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # マーカーが出力されていることを確認
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("サブエージェント", output)

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_no_marker_for_non_investigation_step(self, mock_get_controller):
        """調査不要ステップへ遷移時にマーカーが出力されないこと"""
        # モックの設定
        mock_controller = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.message = "ステップ完了"
        mock_result.next_step = "branch_verification"  # 調査不要ステップ
        mock_controller.attempt_step_completion.return_value = mock_result
        mock_get_controller.return_value = mock_controller

        # 標準出力をキャプチャ
        from tools.workflow.workflow_cli import attempt_step

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            attempt_step(self.tracker_id)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # マーカーが出力されていないことを確認
        self.assertNotIn("[TASK_TOOL_REQUIRED]", output)

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_no_marker_when_no_next_step(self, mock_get_controller):
        """次ステップがない場合にマーカーが出力されないこと"""
        # モックの設定
        mock_controller = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.message = "ワークフロー完了"
        mock_result.next_step = None  # 次ステップなし
        mock_controller.attempt_step_completion.return_value = mock_result
        mock_get_controller.return_value = mock_controller

        # 標準出力をキャプチャ
        from tools.workflow.workflow_cli import attempt_step

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            attempt_step(self.tracker_id)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # マーカーが出力されていないことを確認
        self.assertNotIn("[TASK_TOOL_REQUIRED]", output)

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_no_marker_when_step_blocked(self, mock_get_controller):
        """ステップがブロックされた場合にマーカーが出力されないこと"""
        # モックの設定
        mock_controller = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "blocked"
        mock_result.message = "ブロックされています"
        mock_result.next_step = "implementation"  # 調査必須ステップだが blocked
        mock_controller.attempt_step_completion.return_value = mock_result
        mock_get_controller.return_value = mock_controller

        # 標準出力をキャプチャ
        from tools.workflow.workflow_cli import attempt_step

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            attempt_step(self.tracker_id)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # マーカーが出力されていないことを確認（blockedの場合）
        self.assertNotIn("[TASK_TOOL_REQUIRED]", output)

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_marker_format(self, mock_get_controller):
        """マーカーの出力形式が正しいこと"""
        # モックの設定
        mock_controller = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.message = "ステップ完了"
        mock_result.next_step = "testing"  # 調査必須ステップ
        mock_controller.attempt_step_completion.return_value = mock_result
        mock_get_controller.return_value = mock_controller

        # 標準出力をキャプチャ
        from tools.workflow.workflow_cli import attempt_step

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            attempt_step(self.tracker_id)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # マーカーの形式を確認
        self.assertIn("[TASK_TOOL_REQUIRED]", output)
        self.assertIn("testing", output)
        self.assertIn("=" * 60, output)  # 区切り線


class TestAllInvestigationStepsMarker(unittest.TestCase):
    """全ての調査必須ステップでマーカーが出力されることを確認"""

    @patch("tools.workflow.workflow_cli.get_workflow_controller")
    def test_all_investigation_steps_trigger_marker(self, mock_get_controller):
        """全ての調査必須ステップでマーカーが出力されること（plan/create含む）"""
        investigation_steps = [
            "plan",                   # KIRO-024で追加
            "create",                 # KIRO-024で追加
            "sow_creation",
            "implementation",
            "quality_workflow",
            "subagent_validation",
            "testing",
            "dashboard_generation",
        ]

        for next_step in investigation_steps:
            with self.subTest(next_step=next_step):
                # モックの設定
                mock_controller = MagicMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.message = "ステップ完了"
                mock_result.next_step = next_step
                mock_controller.attempt_step_completion.return_value = mock_result
                mock_get_controller.return_value = mock_controller

                # 標準出力をキャプチャ
                from tools.workflow.workflow_cli import attempt_step

                captured_output = StringIO()
                sys.stdout = captured_output

                try:
                    attempt_step("TEST-001")
                finally:
                    sys.stdout = sys.__stdout__

                output = captured_output.getvalue()

                # マーカーが出力されていることを確認
                self.assertIn(
                    "[TASK_TOOL_REQUIRED]",
                    output,
                    f"{next_step} への遷移時にマーカーが出力されるべき"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
