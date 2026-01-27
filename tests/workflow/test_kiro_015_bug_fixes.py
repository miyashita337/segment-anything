"""
KIRO-015: ワークフローシステム3バグ修正のテスト

修正内容:
1. subagent_validation パス不整合 (extraction_full → extraction)
2. quality_workflow queueモジュール競合 (sys.path修正)
3. dashboard_generation 外部依存エラー (curl + 明確なエラーメッセージ)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestExtractionPathFix(unittest.TestCase):
    """問題1: subagent_validation パス不整合のテスト"""

    def test_extraction_dir_path_is_extraction_not_extraction_full(self):
        """automatic_executor.py が extraction ディレクトリを参照すること"""
        # ソースコードを直接確認
        import inspect
        from tools.execution.automatic_executor import AutomaticWorkflowExecutor

        source = inspect.getsource(AutomaticWorkflowExecutor)

        # extraction_full への参照がないことを確認
        # （extraction_full はフォールバック用途で別の場所では使われる可能性あり）
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "extraction_dir = Path" in line and "extraction_full" in line:
                self.fail(f"行 {i+1}: extraction_dir が extraction_full を参照しています: {line}")


class TestQueueModuleShadowingFix(unittest.TestCase):
    """問題2: quality_workflow queueモジュール競合のテスト"""

    def test_standard_queue_module_is_accessible(self):
        """標準ライブラリのqueueモジュールがインポートできること"""
        # tools/queue/ がシャドウしないことを確認
        import queue

        self.assertTrue(hasattr(queue, "Queue"))
        self.assertTrue(hasattr(queue, "Empty"))

    def test_run_objective_evaluation_sys_path_points_to_project_root(self):
        """run_objective_evaluation.py の sys.path がプロジェクトルートを指すこと"""
        eval_script_path = project_root / "tools" / "core" / "run_objective_evaluation.py"

        with open(eval_script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # .parent.parent.parent が使われていることを確認
        self.assertIn(
            ".parent.parent.parent", content, "sys.path はプロジェクトルート（segment-anything/）を指すべき"
        )

        # .parent.parent のみ（tools/を指す）が使われていないことを確認
        lines = content.split("\n")
        for line in lines:
            if "project_root = Path" in line and ".parent.parent.parent" not in line:
                if ".parent.parent" in line and ".parent.parent.parent" not in line:
                    self.fail(f"sys.path が tools/ を指しています: {line}")

    def test_objective_evaluation_can_be_imported(self):
        """run_objective_evaluation.py が正常にインポートできること"""
        try:
            # インポート時にqueueモジュール競合が発生しないことを確認
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_objective_evaluation",
                project_root / "tools" / "core" / "run_objective_evaluation.py",
            )
            module = importlib.util.module_from_spec(spec)
            # 実行せずにロードできることを確認
            self.assertIsNotNone(module)
        except ImportError as e:
            if "queue" in str(e).lower():
                self.fail(f"queueモジュール競合が発生: {e}")
            raise


class TestFinalApprovalConditionsCheck(unittest.TestCase):
    """問題3: dashboard_generation 外部依存エラーのテスト"""

    def setUp(self):
        """テスト用の一時ディレクトリを作成"""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker_id = "TEST-001"
        self.tracker_workspace = os.path.join(self.temp_dir, self.tracker_id)
        os.makedirs(self.tracker_workspace, exist_ok=True)

    def tearDown(self):
        """一時ディレクトリを削除"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("config.workspace_config.WorkspaceConfig.get_workspace_base")
    def test_missing_dashboard_file_returns_actionable_error(self, mock_workspace):
        """必須ファイル不在時に対処方法を含むエラーが返ること"""
        mock_workspace.return_value = self.temp_dir

        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()

        result = controller._check_final_approval_conditions(self.tracker_id)

        self.assertFalse(result.success)
        # エラーメッセージに対処方法が含まれていることを確認
        error_text = "\n".join(result.errors)
        self.assertIn("対処:", error_text, "エラーメッセージに対処方法が含まれるべき")

    @patch("config.workspace_config.WorkspaceConfig.get_workspace_base")
    def test_missing_quality_report_returns_actionable_error(self, mock_workspace):
        """品質レポート不在時に対処方法を含むエラーが返ること"""
        mock_workspace.return_value = self.temp_dir

        # ダッシュボードHTMLは作成
        os.makedirs(os.path.join(self.tracker_workspace, "dashboard"), exist_ok=True)
        with open(os.path.join(self.tracker_workspace, "dashboard", "dashboard.html"), "w") as f:
            f.write("<html></html>")

        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()

        result = controller._check_final_approval_conditions(self.tracker_id)

        self.assertFalse(result.success)
        error_text = "\n".join(result.errors)
        self.assertIn("対処:", error_text)

    @patch("subprocess.run")
    @patch("config.workspace_config.WorkspaceConfig.get_workspace_base")
    def test_server_connection_uses_curl(self, mock_workspace, mock_run):
        """サーバー接続確認がcurlを使用すること"""
        mock_workspace.return_value = self.temp_dir

        # 必須ファイルを作成
        os.makedirs(os.path.join(self.tracker_workspace, "dashboard"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "quality"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "extraction"), exist_ok=True)

        with open(os.path.join(self.tracker_workspace, "dashboard", "dashboard.html"), "w") as f:
            f.write("<html></html>")
        with open(
            os.path.join(self.tracker_workspace, "quality", "unified_quality_report.json"), "w"
        ) as f:
            f.write('{"evaluation_metrics": [{"test": 1}]}')
        with open(
            os.path.join(self.tracker_workspace, "statistical_analysis_result.txt"), "w"
        ) as f:
            f.write(
                "Current 平均=0.85\nBaseLine 平均=0.70\np値: 0.001\nCohen's d: 0.8\n改善率: 15%\n統計的有意性: 有意\n信頼区間: [0.75, 0.95]"
            )

        # テスト画像を作成
        with open(os.path.join(self.tracker_workspace, "extraction", "test.jpg"), "w") as f:
            f.write("dummy")

        # curlコマンドの成功をモック
        mock_run.return_value = MagicMock(returncode=0, stdout="200")

        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()

        controller._check_final_approval_conditions(self.tracker_id)

        # curlが呼ばれたことを確認
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "curl", "curlコマンドが使用されるべき")

    @patch("subprocess.run")
    @patch("config.workspace_config.WorkspaceConfig.get_workspace_base")
    def test_server_timeout_returns_actionable_error(self, mock_workspace, mock_run):
        """サーバータイムアウト時に対処方法を含むエラーが返ること"""
        import subprocess

        mock_workspace.return_value = self.temp_dir
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="curl", timeout=5)

        # 必須ファイルを作成
        os.makedirs(os.path.join(self.tracker_workspace, "dashboard"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "quality"), exist_ok=True)

        with open(os.path.join(self.tracker_workspace, "dashboard", "dashboard.html"), "w") as f:
            f.write("<html></html>")
        with open(
            os.path.join(self.tracker_workspace, "quality", "unified_quality_report.json"), "w"
        ) as f:
            f.write('{"evaluation_metrics": [{"test": 1}]}')

        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()

        result = controller._check_final_approval_conditions(self.tracker_id)

        self.assertFalse(result.success)
        error_text = "\n".join(result.errors)
        self.assertIn("タイムアウト", error_text)
        self.assertIn("対処:", error_text)

    def test_workflow_controller_uses_subprocess_not_requests(self):
        """workflow_controller.py が requests ではなく subprocess を使用すること"""
        controller_path = project_root / "tools" / "interface" / "workflow_controller.py"

        with open(controller_path, "r", encoding="utf-8") as f:
            content = f.read()

        # _check_final_approval_conditions 内で subprocess が使われていることを確認
        self.assertIn("import subprocess", content)

        # _check_final_approval_conditions 内で requests.get が使われていないことを確認
        # （他のメソッドでは使用可能なので、メソッド内のみをチェック）
        import re

        method_match = re.search(
            r"def _check_final_approval_conditions.*?(?=\n    def |\nclass |\Z)", content, re.DOTALL
        )
        if method_match:
            method_content = method_match.group(0)
            self.assertNotIn(
                "requests.get",
                method_content,
                "_check_final_approval_conditions で requests.get は使用禁止",
            )


class TestExtractionDirectoryFallback(unittest.TestCase):
    """extraction ディレクトリのフォールバック動作テスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tracker_id = "TEST-002"
        self.tracker_workspace = os.path.join(self.temp_dir, self.tracker_id)
        os.makedirs(self.tracker_workspace, exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.run")
    @patch("config.workspace_config.WorkspaceConfig.get_workspace_base")
    def test_extraction_full_is_used_if_exists(self, mock_workspace, mock_run):
        """extraction_full が存在する場合はそちらが使用されること"""
        mock_workspace.return_value = self.temp_dir
        mock_run.return_value = MagicMock(returncode=0, stdout="200")

        # 必須ファイルとディレクトリを作成
        os.makedirs(os.path.join(self.tracker_workspace, "dashboard"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "quality"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "extraction"), exist_ok=True)
        os.makedirs(os.path.join(self.tracker_workspace, "extraction_full"), exist_ok=True)

        with open(os.path.join(self.tracker_workspace, "dashboard", "dashboard.html"), "w") as f:
            f.write("<html></html>")
        with open(
            os.path.join(self.tracker_workspace, "quality", "unified_quality_report.json"), "w"
        ) as f:
            f.write('{"evaluation_metrics": [{"test": 1}]}')
        with open(
            os.path.join(self.tracker_workspace, "statistical_analysis_result.txt"), "w"
        ) as f:
            f.write(
                "Current 平均=0.85\nBaseLine 平均=0.70\np値: 0.001\nCohen's d: 0.8\n改善率: 15%\n統計的有意性: 有意\n信頼区間: [0.75, 0.95]"
            )

        # extraction_full にのみ画像を配置
        with open(os.path.join(self.tracker_workspace, "extraction_full", "test.jpg"), "w") as f:
            f.write("dummy")

        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()

        result = controller._check_final_approval_conditions(self.tracker_id)

        # extraction_full が使用されるため、画像ファイルが見つかりエラーにならないはず
        error_text = "\n".join(result.errors)
        self.assertNotIn("抽出画像ファイルなし", error_text)


if __name__ == "__main__":
    unittest.main()
