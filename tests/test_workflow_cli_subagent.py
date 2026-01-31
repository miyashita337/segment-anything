#!/usr/bin/env python3
"""
KIRO-024 サブエージェントトリガーロジックのユニットテスト

workflow_cli.py の step_requires_investigation 関数と
attempt_step 関数のマーカー出力をテストする
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.workflow.workflow_cli import step_requires_investigation


class TestStepRequiresInvestigation(unittest.TestCase):
    """step_requires_investigation関数のテスト"""

    def test_investigation_required_steps(self):
        """調査必須ステップでTrueを返すこと"""
        investigation_steps = [
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
        """全ての調査必須ステップでマーカーが出力されること"""
        investigation_steps = [
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
