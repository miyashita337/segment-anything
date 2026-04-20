#!/usr/bin/env python3
"""
KIRO-029: WorkflowController ワークスペースパス解決のユニットテスト

多作者対応バグ修正の検証テスト:
- _get_tracker_workspace_base()
- _check_final_approval_conditions()
- _get_step_artifacts()
- _generate_step_actions()
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# パッチ対象のモジュールパス
WC_MODULE = "tools.interface.workflow_controller.WorkflowController"


def create_mock_controller():
    """モック化されたWorkflowControllerを作成"""
    patches = [
        patch(f"{WC_MODULE}._init_state_manager"),
        patch(f"{WC_MODULE}._init_validator"),
        patch(f"{WC_MODULE}._init_approval_controller"),
        patch(f"{WC_MODULE}._init_executor"),
        patch(f"{WC_MODULE}._load_workflow_config"),
    ]
    for p in patches:
        p.start()
    from tools.interface.workflow_controller import WorkflowController

    controller = WorkflowController()
    return controller, patches


class TestGetTrackerWorkspaceBase(unittest.TestCase):
    """_get_tracker_workspace_base() メソッドのテスト"""

    def setUp(self):
        self.controller, self.patches = create_mock_controller()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_author_yado(self, mock_get_config):
        """作者yadoのトラッカーで正しいパスを返す"""
        mock_config_manager = MagicMock()
        mock_config_manager.get_workspace_config.return_value = {
            "tracker_id": "TRACKER-001",
            "author_name": "yado",
            "workspace_path": "/mnt/c/AItools/lora/train/yado/tracker-workspace",
        }
        yado_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        mock_config_manager.get_author_workspace_base.return_value = yado_base
        mock_get_config.return_value = mock_config_manager

        result = self.controller._get_tracker_workspace_base("TRACKER-001")

        self.assertEqual(result, yado_base)
        mock_config_manager.get_workspace_config.assert_called_once_with("TRACKER-001")
        mock_config_manager.get_author_workspace_base.assert_called_once_with("yado")

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_author_zundamon(self, mock_get_config):
        """作者zundamonのトラッカーで正しいパスを返す"""
        mock_config_manager = MagicMock()
        mock_config_manager.get_workspace_config.return_value = {
            "tracker_id": "KIRO-029",
            "author_name": "zundamon",
        }
        zunda_base = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"
        mock_config_manager.get_author_workspace_base.return_value = zunda_base
        mock_get_config.return_value = mock_config_manager

        result = self.controller._get_tracker_workspace_base("KIRO-029")

        self.assertEqual(result, zunda_base)
        mock_config_manager.get_workspace_config.assert_called_once_with("KIRO-029")

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_multiple_authors_isolation(self, mock_get_config):
        """複数作者が存在する場合、各トラッカーで正しい作者のパスを返す"""
        mock_config_manager = MagicMock()

        def mock_get_workspace(tracker_id):
            configs = {
                "TRACKER-001": {"author_name": "yado"},
                "KIRO-029": {"author_name": "zundamon"},
                "KIRO-030": {"author_name": "kiri"},
            }
            return configs.get(tracker_id)

        def mock_get_author_base(author_name):
            bases = {
                "yado": "/mnt/c/AItools/lora/train/yado/tracker-workspace",
                "zundamon": "/mnt/c/AItools/lora/train/zundamon/tracker-workspace",
                "kiri": "/mnt/c/AItools/lora/train/kiri/tracker-workspace",
            }
            return bases.get(author_name)

        mock_config_manager.get_workspace_config.side_effect = mock_get_workspace
        mock_config_manager.get_author_workspace_base.side_effect = mock_get_author_base
        mock_get_config.return_value = mock_config_manager

        # 各トラッカーで正しい作者のパスが返されることを確認
        yado_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        zunda_base = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"
        kiri_base = "/mnt/c/AItools/lora/train/kiri/tracker-workspace"

        self.assertEqual(self.controller._get_tracker_workspace_base("TRACKER-001"), yado_base)
        self.assertEqual(self.controller._get_tracker_workspace_base("KIRO-029"), zunda_base)
        self.assertEqual(self.controller._get_tracker_workspace_base("KIRO-030"), kiri_base)

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_fallback_kiro_prefix(self, mock_get_config):
        """設定が見つからない場合、KIROプレフィックスでフォールバック"""
        mock_config_manager = MagicMock()
        mock_config_manager.get_workspace_config.return_value = None
        mock_get_config.return_value = mock_config_manager

        result = self.controller._get_tracker_workspace_base("KIRO-999")

        self.assertEqual(result, "/mnt/c/AItools/lora/train/kiri/tracker-workspace")

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_fallback_tracker_prefix(self, mock_get_config):
        """設定が見つからない場合、TRACKERプレフィックスでフォールバック"""
        mock_config_manager = MagicMock()
        mock_config_manager.get_workspace_config.return_value = None
        mock_get_config.return_value = mock_config_manager

        result = self.controller._get_tracker_workspace_base("TRACKER-999")

        self.assertEqual(result, "/mnt/c/AItools/lora/train/yado/tracker-workspace")

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_fallback_generic(self, mock_get_config):
        """設定が見つからない場合、汎用フォールバック"""
        mock_config_manager = MagicMock()
        mock_config_manager.get_workspace_config.return_value = None
        mock_get_config.return_value = mock_config_manager

        result = self.controller._get_tracker_workspace_base("OTHER-001")

        self.assertEqual(result, "/mnt/c/AItools/lora/train/generic/tracker-workspace")

    @patch("config.workspace_config.get_workspace_config")
    def test_workspace_base_exception_handling(self, mock_get_config):
        """例外発生時のフォールバック動作"""
        mock_get_config.side_effect = Exception("Database connection failed")

        result = self.controller._get_tracker_workspace_base("KIRO-029")

        self.assertEqual(result, "/mnt/c/AItools/lora/train/kiri/tracker-workspace")


class TestCheckFinalApprovalConditions(unittest.TestCase):
    """_check_final_approval_conditions() メソッドのテスト"""

    def setUp(self):
        self.controller, self.patches = create_mock_controller()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_uses_tracker_specific_workspace(self):
        """トラッカー固有のワークスペースパスを使用することを確認"""
        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            zunda_base = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"
            mock_get_base.return_value = zunda_base

            # 関数を呼び出す（ファイルが存在しないのでエラーが出るが、パス検証が目的）
            self.controller._check_final_approval_conditions("KIRO-029")

            mock_get_base.assert_called_once_with("KIRO-029")

    def test_different_authors_different_paths(self):
        """異なる作者のトラッカーで異なるパスが使用されることを確認"""
        call_args = []

        def capture_call(tracker_id):
            call_args.append(tracker_id)
            if tracker_id == "TRACKER-001":
                return "/mnt/c/AItools/lora/train/yado/tracker-workspace"
            elif tracker_id == "KIRO-029":
                return "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"

        with patch.object(self.controller, "_get_tracker_workspace_base", side_effect=capture_call):
            self.controller._check_final_approval_conditions("TRACKER-001")
            self.controller._check_final_approval_conditions("KIRO-029")

            self.assertEqual(call_args, ["TRACKER-001", "KIRO-029"])


class TestGetStepArtifacts(unittest.TestCase):
    """_get_step_artifacts() メソッドのテスト"""

    def setUp(self):
        self.controller, self.patches = create_mock_controller()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_sow_creation_artifacts_author_yado(self):
        """sow_creationステップでyadoのアーティファクトパス"""
        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            mock_get_base.return_value = "/mnt/c/AItools/lora/train/yado/tracker-workspace"

            artifacts = self.controller._get_step_artifacts("TRACKER-001", "sow_creation")

            self.assertEqual(len(artifacts), 1)
            self.assertIn("yado", artifacts[0])
            self.assertIn("TRACKER-001", artifacts[0])
            self.assertIn("sow_document.md", artifacts[0])

    def test_sow_creation_artifacts_author_zundamon(self):
        """sow_creationステップでzundamonのアーティファクトパス"""
        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            mock_get_base.return_value = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"

            artifacts = self.controller._get_step_artifacts("KIRO-029", "sow_creation")

            self.assertEqual(len(artifacts), 1)
            self.assertIn("zundamon", artifacts[0])
            self.assertIn("KIRO-029", artifacts[0])

    def test_final_approval_artifacts_paths(self):
        """final_approvalステップで正しいパスが含まれる"""
        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            mock_get_base.return_value = "/mnt/c/AItools/lora/train/kiri/tracker-workspace"

            artifacts = self.controller._get_step_artifacts("KIRO-030", "final_approval")

            self.assertEqual(len(artifacts), 3)
            # 全てのパスにkiri作者のパスが含まれる
            for artifact in artifacts:
                self.assertIn("kiri", artifact)
                self.assertIn("KIRO-030", artifact)

    def test_no_hardcoded_yado(self):
        """ハードコードされた'yado'が使用されないことを確認"""
        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            mock_get_base.return_value = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"

            artifacts = self.controller._get_step_artifacts("KIRO-029", "sow_creation")

            # artifacts内に'yado'が含まれていないことを確認
            for artifact in artifacts:
                self.assertNotIn("/yado/", artifact)

    def test_concurrent_different_authors(self):
        """異なる作者の同時アクセスで各々正しいパスを返す"""

        def mock_get_base(tracker_id):
            bases = {
                "TRACKER-001": "/mnt/c/AItools/lora/train/yado/tracker-workspace",
                "KIRO-029": "/mnt/c/AItools/lora/train/zundamon/tracker-workspace",
            }
            return bases.get(tracker_id)

        with patch.object(
            self.controller, "_get_tracker_workspace_base", side_effect=mock_get_base
        ):
            artifacts_yado = self.controller._get_step_artifacts("TRACKER-001", "sow_creation")
            artifacts_zunda = self.controller._get_step_artifacts("KIRO-029", "sow_creation")

            self.assertIn("yado", artifacts_yado[0])
            self.assertIn("zundamon", artifacts_zunda[0])


class TestGenerateStepActions(unittest.TestCase):
    """_generate_step_actions() メソッドのテスト（quality_workflowステップ）"""

    def setUp(self):
        self.controller, self.patches = create_mock_controller()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_quality_workflow_uses_tracker_specific_path(self):
        """quality_workflowステップでトラッカー固有パスを使用"""
        from tools.interface.workflow_controller import WorkflowStep

        step_config = WorkflowStep(
            step_id="quality_workflow",
            phase="phase_2",
            title="Quality Workflow",
            description="Run quality workflow",
            prerequisites=[],
            validation_required=True,
            approval_required=False,
            auto_executable=True,
        )

        with patch.object(self.controller, "_get_tracker_workspace_base") as mock_get_base:
            zunda_base = "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"
            mock_get_base.return_value = zunda_base

            # 引数順序: step_config, tracker_id
            actions = self.controller._generate_step_actions(step_config, "KIRO-029")

            mock_get_base.assert_called_once_with("KIRO-029")
            # アクションにzundamonのパスが含まれることを確認
            actions_str = " ".join(actions)
            self.assertIn("zundamon", actions_str)

    def test_quality_workflow_different_authors(self):
        """quality_workflowステップで異なる作者に対応"""
        from tools.interface.workflow_controller import WorkflowStep

        step_config = WorkflowStep(
            step_id="quality_workflow",
            phase="phase_2",
            title="Quality Workflow",
            description="Run quality workflow",
            prerequisites=[],
            validation_required=True,
            approval_required=False,
            auto_executable=True,
        )

        def mock_get_base(tracker_id):
            if tracker_id == "TRACKER-001":
                return "/mnt/c/AItools/lora/train/yado/tracker-workspace"
            elif tracker_id == "KIRO-029":
                return "/mnt/c/AItools/lora/train/zundamon/tracker-workspace"

        with patch.object(
            self.controller, "_get_tracker_workspace_base", side_effect=mock_get_base
        ):
            # 引数順序: step_config, tracker_id
            actions_yado = self.controller._generate_step_actions(step_config, "TRACKER-001")
            actions_zunda = self.controller._generate_step_actions(step_config, "KIRO-029")

            self.assertIn("yado", " ".join(actions_yado))
            self.assertIn("zundamon", " ".join(actions_zunda))


class TestIntegrationMultiAuthor(unittest.TestCase):
    """統合テスト: 複数作者環境での動作"""

    def setUp(self):
        self.controller, self.patches = create_mock_controller()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_author_isolation(self):
        """作者Aのトラッカー処理が作者Bに影響しないことを確認"""
        call_log = []

        def mock_get_base(tracker_id):
            call_log.append(tracker_id)
            bases = {
                "TRACKER-001": "/mnt/c/AItools/lora/train/yado/tracker-workspace",
                "KIRO-029": "/mnt/c/AItools/lora/train/zundamon/tracker-workspace",
                "KIRO-030": "/mnt/c/AItools/lora/train/kiri/tracker-workspace",
            }
            return bases.get(tracker_id)

        with patch.object(
            self.controller, "_get_tracker_workspace_base", side_effect=mock_get_base
        ):
            # 複数のトラッカーを順番に処理
            a1 = self.controller._get_step_artifacts("TRACKER-001", "sow_creation")
            a2 = self.controller._get_step_artifacts("KIRO-029", "sow_creation")
            a3 = self.controller._get_step_artifacts("KIRO-030", "sow_creation")

            # 各アーティファクトが正しい作者のパスを持つ
            self.assertIn("yado", a1[0])
            self.assertIn("zundamon", a2[0])
            self.assertIn("kiri", a3[0])

            # 呼び出し順序が記録されている
            self.assertEqual(call_log, ["TRACKER-001", "KIRO-029", "KIRO-030"])

    def test_latest_update_does_not_affect_other_trackers(self):
        """最新更新トラッカーが他トラッカーのパスに影響しないことを確認

        バグの本質: 旧実装では ORDER BY updated_at DESC LIMIT 1 により
        最新更新トラッカーの作者名が返されていた。
        新実装では各トラッカーのIDから正しい作者名を取得する。
        """

        def mock_get_base(tracker_id):
            # 各トラッカーIDに対して正しい作者のパスを返す
            bases = {
                "TRACKER-001": "/mnt/c/AItools/lora/train/yado/tracker-workspace",
                "KIRO-029": "/mnt/c/AItools/lora/train/zundamon/tracker-workspace",
                "KIRO-030": "/mnt/c/AItools/lora/train/kiri/tracker-workspace",
            }
            return bases.get(tracker_id)

        with patch.object(
            self.controller, "_get_tracker_workspace_base", side_effect=mock_get_base
        ):
            # KIRO-030が最後に更新されたシナリオ
            # しかし、TRACKER-001の処理ではyado作者のパスが使用されるべき
            artifacts = self.controller._get_step_artifacts("TRACKER-001", "sow_creation")

            # 旧バグでは、ここで'kiri'（最新更新トラッカーの作者）のパスが返されていた
            # 新実装では'yado'のパスが正しく返される
            self.assertIn("yado", artifacts[0])
            self.assertNotIn("kiri", artifacts[0])


if __name__ == "__main__":
    unittest.main()
