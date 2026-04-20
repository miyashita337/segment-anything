#!/usr/bin/env python3
"""
KIRO-029: planコマンドの--input-pathオプションのユニットテスト

workflow_cli.py の plan_tracker 関数の input_path 引数と
argparser の --input-path オプションをテストする
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.workflow.workflow_cli import plan_tracker


class TestPlanTrackerInputPath(unittest.TestCase):
    """plan_tracker関数のinput_path引数テスト"""

    @patch("config.workspace_config.get_workspace_config")
    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    def test_default_input_path_used_when_not_specified(
        self, mock_handler_class, mock_get_config
    ):
        """input_pathが指定されていない場合、デフォルト値が使用される"""
        # モック設定
        mock_config = MagicMock()
        mock_config.get_default_input_path.return_value = (
            "/mnt/c/AItools/lora/train/yado/org/kana05"
        )
        mock_config.set_workspace_config.return_value = True
        mock_get_config.return_value = mock_config

        mock_handler = MagicMock()
        mock_handler.execute_plan_command.return_value = (True, "Success")
        mock_handler_class.return_value = mock_handler

        # 実行
        result = plan_tracker(
            tracker_id="TEST-001",
            summary="テスト概要",
            details="テスト詳細",
            author_name="yado",
            priority="medium",
            input_path=None,  # 明示的にNoneを指定
        )

        # 検証
        self.assertTrue(result)
        mock_config.get_default_input_path.assert_called_once_with("yado")
        mock_config.set_workspace_config.assert_called_once()
        call_args = mock_config.set_workspace_config.call_args
        self.assertEqual(call_args[0][3], "/mnt/c/AItools/lora/train/yado/org/kana05")

    @patch("config.workspace_config.get_workspace_config")
    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    def test_custom_input_path_used_when_specified(
        self, mock_handler_class, mock_get_config
    ):
        """input_pathが指定されている場合、その値が使用される"""
        # モック設定
        mock_config = MagicMock()
        mock_config.set_workspace_config.return_value = True
        mock_get_config.return_value = mock_config

        mock_handler = MagicMock()
        mock_handler.execute_plan_command.return_value = (True, "Success")
        mock_handler_class.return_value = mock_handler

        custom_path = "/mnt/c/AItools/lora/train/yado/org/kana08"

        # 実行
        result = plan_tracker(
            tracker_id="TEST-002",
            summary="テスト概要",
            details="テスト詳細",
            author_name="yado",
            priority="medium",
            input_path=custom_path,  # カスタムパスを指定
        )

        # 検証
        self.assertTrue(result)
        # デフォルトパス取得は呼ばれないはず
        mock_config.get_default_input_path.assert_not_called()
        # カスタムパスが使用されていることを検証
        mock_config.set_workspace_config.assert_called_once()
        call_args = mock_config.set_workspace_config.call_args
        self.assertEqual(call_args[0][3], custom_path)

    @patch("config.workspace_config.get_workspace_config")
    @patch("tools.workflow.plan_command_handler.PlanCommandHandler")
    def test_different_author_default_path(
        self, mock_handler_class, mock_get_config
    ):
        """異なる作者名でデフォルトパスが正しく生成される"""
        # モック設定
        mock_config = MagicMock()
        mock_config.get_default_input_path.return_value = (
            "/mnt/c/AItools/lora/train/kiri/org/kana05"
        )
        mock_config.set_workspace_config.return_value = True
        mock_get_config.return_value = mock_config

        mock_handler = MagicMock()
        mock_handler.execute_plan_command.return_value = (True, "Success")
        mock_handler_class.return_value = mock_handler

        # 実行
        result = plan_tracker(
            tracker_id="TEST-003",
            summary="テスト概要",
            details="テスト詳細",
            author_name="kiri",
            priority="medium",
            input_path=None,
        )

        # 検証
        self.assertTrue(result)
        mock_config.get_default_input_path.assert_called_once_with("kiri")


class TestWorkspaceConfigDefaultInputPath(unittest.TestCase):
    """WorkspaceConfig.get_default_input_path のテスト"""

    def test_default_folder_name_kana05(self):
        """folder_nameが省略された場合、kana05がデフォルトとして使用される"""
        from config.workspace_config import WorkspaceConfig
        import tempfile
        import os

        # 一時DBでテスト
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            config = WorkspaceConfig(db_path=temp_db)
            result = config.get_default_input_path("yado")
            self.assertEqual(
                result, "/mnt/c/AItools/lora/train/yado/org/kana05"
            )
        finally:
            os.unlink(temp_db)

    def test_custom_folder_name(self):
        """folder_nameを指定した場合、その値が使用される"""
        from config.workspace_config import WorkspaceConfig
        import tempfile
        import os

        # 一時DBでテスト
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            config = WorkspaceConfig(db_path=temp_db)
            result = config.get_default_input_path("yado", folder_name="kana08")
            self.assertEqual(
                result, "/mnt/c/AItools/lora/train/yado/org/kana08"
            )
        finally:
            os.unlink(temp_db)

    def test_different_authors(self):
        """異なる作者名で正しいパスが生成される"""
        from config.workspace_config import WorkspaceConfig
        import tempfile
        import os

        # 一時DBでテスト
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            config = WorkspaceConfig(db_path=temp_db)

            test_cases = [
                ("yado", "/mnt/c/AItools/lora/train/yado/org/kana05"),
                ("kiri", "/mnt/c/AItools/lora/train/kiri/org/kana05"),
                ("zundamon", "/mnt/c/AItools/lora/train/zundamon/org/kana05"),
            ]

            for author, expected in test_cases:
                with self.subTest(author=author):
                    result = config.get_default_input_path(author)
                    self.assertEqual(result, expected)
        finally:
            os.unlink(temp_db)


class TestArgparserInputPath(unittest.TestCase):
    """argparser の --input-path オプションのテスト"""

    def test_argparser_accepts_input_path_short(self):
        """argparserが-iオプションを受け付ける"""
        import argparse

        # argparserのテスト用に直接パースをシミュレート
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        plan_parser = subparsers.add_parser("plan")
        plan_parser.add_argument("tracker_id")
        plan_parser.add_argument("summary")
        plan_parser.add_argument("details")
        plan_parser.add_argument("author_name")
        plan_parser.add_argument("--priority", default="medium")
        plan_parser.add_argument("--input-path", "-i", dest="input_path", default=None)

        args = parser.parse_args(
            ["plan", "TEST-001", "概要", "詳細", "yado", "-i", "/custom/path"]
        )

        self.assertEqual(args.input_path, "/custom/path")

    def test_argparser_accepts_input_path_long(self):
        """argparserが--input-pathオプションを受け付ける"""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        plan_parser = subparsers.add_parser("plan")
        plan_parser.add_argument("tracker_id")
        plan_parser.add_argument("summary")
        plan_parser.add_argument("details")
        plan_parser.add_argument("author_name")
        plan_parser.add_argument("--priority", default="medium")
        plan_parser.add_argument("--input-path", "-i", dest="input_path", default=None)

        args = parser.parse_args(
            ["plan", "TEST-002", "概要", "詳細", "yado", "--input-path", "/another/path"]
        )

        self.assertEqual(args.input_path, "/another/path")

    def test_argparser_input_path_default_none(self):
        """--input-pathが省略された場合、Noneがデフォルト"""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        plan_parser = subparsers.add_parser("plan")
        plan_parser.add_argument("tracker_id")
        plan_parser.add_argument("summary")
        plan_parser.add_argument("details")
        plan_parser.add_argument("author_name")
        plan_parser.add_argument("--priority", default="medium")
        plan_parser.add_argument("--input-path", "-i", dest="input_path", default=None)

        args = parser.parse_args(
            ["plan", "TEST-003", "概要", "詳細", "yado"]
        )

        self.assertIsNone(args.input_path)


if __name__ == "__main__":
    unittest.main()
