#!/usr/bin/env python3
"""
KIRO-021 - sheets/sheetget コマンドのユニットテスト

テスト対象:
- check_sheets_status(): Google Sheets状態確認（簡易）
- get_sheet_row(): Google Sheetデータ取得（詳細）
"""

import json
import os
import pytest
import sys
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.workflow.workflow_cli import check_sheets_status, get_sheet_row


class MockTaskRecord:
    """テスト用のモックTaskRecord"""

    def __init__(
        self,
        tracker_id="TEST-001",
        description="テスト説明",
        status_value="着手前",
        priority_value="優先度中",
        created_date=None,
        updated_date=None,
        details="テスト詳細",
    ):
        self.tracker_id = tracker_id
        self.description = description
        self.status = MagicMock()
        self.status.value = status_value
        self.priority = MagicMock()
        self.priority.value = priority_value
        self.created_date = created_date or datetime(2026, 1, 29, 12, 0, 0)
        self.updated_date = updated_date or datetime(2026, 1, 29, 12, 0, 0)
        self.details = details


class TestCheckSheetsStatus:
    """check_sheets_status() のテスト"""

    @patch("tools.progress_tracker.progress_manager.ProgressManager")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_success_with_valid_tracker(self, mock_config, mock_manager_class, capsys):
        """正常ケース: 有効なトラッカーIDで成功"""
        # モック設定
        mock_config.return_value = MagicMock()
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        mock_task = MockTaskRecord(
            tracker_id="KIRO-021",
            description="テストタスク",
            status_value="着手前",
        )
        mock_manager.get_task.return_value = mock_task

        # 実行
        result = check_sheets_status("KIRO-021")

        # 検証
        assert result is True
        captured = capsys.readouterr()
        assert "KIRO-021" in captured.out
        assert "テストタスク" in captured.out

    @patch("tools.progress_tracker.progress_manager.ProgressManager")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_failure_tracker_not_found(self, mock_config, mock_manager_class, capsys):
        """失敗ケース: トラッカーが見つからない"""
        mock_config.return_value = MagicMock()
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_task.return_value = None

        result = check_sheets_status("NONEXISTENT-999")

        assert result is False
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "見つかりません" in captured.out

    @patch("tools.progress_tracker.progress_manager.ProgressManager")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_failure_api_error(self, mock_config, mock_manager_class, capsys):
        """失敗ケース: API接続エラー"""
        mock_config.return_value = MagicMock()
        mock_manager_class.side_effect = Exception("API接続失敗")

        result = check_sheets_status("TEST-001")

        assert result is False
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "エラー" in captured.out


class TestGetSheetRow:
    """get_sheet_row() のテスト"""

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_success_text_output(self, mock_config, mock_client_class, capsys):
        """正常ケース: テキスト形式出力"""
        mock_config.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_task = MockTaskRecord(
            tracker_id="KIRO-021",
            description="sheetgetテスト",
            details="詳細テキスト",
        )
        mock_client.get_all_tasks.return_value = [mock_task]

        result = get_sheet_row("KIRO-021", output_json=False)

        assert result is True
        captured = capsys.readouterr()
        assert "KIRO-021" in captured.out
        assert "sheetgetテスト" in captured.out
        assert "詳細テキスト" in captured.out

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_success_json_output(self, mock_config, mock_client_class, capsys):
        """正常ケース: JSON形式出力"""
        mock_config.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_task = MockTaskRecord(
            tracker_id="KIRO-021",
            description="JSON出力テスト",
        )
        mock_client.get_all_tasks.return_value = [mock_task]

        result = get_sheet_row("KIRO-021", output_json=True)

        assert result is True
        captured = capsys.readouterr()

        # JSON形式検証
        output_json = json.loads(captured.out)
        assert output_json["tracker_id"] == "KIRO-021"
        assert output_json["description"] == "JSON出力テスト"

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_failure_tracker_not_found(self, mock_config, mock_client_class, capsys):
        """失敗ケース: トラッカーが見つからない"""
        mock_config.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_all_tasks.return_value = []

        result = get_sheet_row("NONEXISTENT-999")

        assert result is False
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "見つかりません" in captured.out

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_failure_api_error(self, mock_config, mock_client_class, capsys):
        """失敗ケース: API接続エラー"""
        mock_config.return_value = MagicMock()
        mock_client_class.side_effect = Exception("Google Sheets API接続失敗")

        result = get_sheet_row("TEST-001")

        assert result is False
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "エラー" in captured.out

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_success_multiple_tasks_find_correct_one(self, mock_config, mock_client_class, capsys):
        """正常ケース: 複数タスクから正しいものを検索"""
        mock_config.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # 複数タスク
        tasks = [
            MockTaskRecord(tracker_id="OTHER-001", description="別タスク1"),
            MockTaskRecord(tracker_id="TARGET-002", description="ターゲットタスク"),
            MockTaskRecord(tracker_id="OTHER-003", description="別タスク3"),
        ]
        mock_client.get_all_tasks.return_value = tasks

        result = get_sheet_row("TARGET-002", output_json=False)

        assert result is True
        captured = capsys.readouterr()
        assert "TARGET-002" in captured.out
        assert "ターゲットタスク" in captured.out

    @patch("tools.progress_tracker.sheets_client.GoogleSheetsClient")
    @patch("tools.progress_tracker.config.get_default_config")
    def test_json_output_with_none_values(self, mock_config, mock_client_class, capsys):
        """JSONケース: None値の適切な処理"""
        mock_config.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_task = MockTaskRecord(
            tracker_id="TEST-001",
            description="Noneテスト",
            details="",  # 空の詳細
        )
        mock_client.get_all_tasks.return_value = [mock_task]

        result = get_sheet_row("TEST-001", output_json=True)

        assert result is True
        captured = capsys.readouterr()
        output_json = json.loads(captured.out)
        assert output_json["tracker_id"] == "TEST-001"
        # 空文字列の場合detailsはNoneになる
        assert output_json["details"] is None


class TestCLIIntegration:
    """CLI統合テスト（argparse経由）"""

    @patch("tools.workflow.workflow_cli.check_sheets_status")
    def test_cli_sheets_command(self, mock_func):
        """CLIからsheetsコマンド呼び出し"""
        mock_func.return_value = True

        import sys
        from tools.workflow.workflow_cli import main

        with patch.object(sys, "argv", ["workflow_cli.py", "sheets", "TEST-001"]):
            with patch("tools.workflow.workflow_cli.check_virtual_environment", return_value=True):
                result = main()

        mock_func.assert_called_once_with("TEST-001")

    @patch("tools.workflow.workflow_cli.get_sheet_row")
    def test_cli_sheetget_command(self, mock_func):
        """CLIからsheetgetコマンド呼び出し"""
        mock_func.return_value = True

        import sys
        from tools.workflow.workflow_cli import main

        with patch.object(sys, "argv", ["workflow_cli.py", "sheetget", "TEST-001"]):
            with patch("tools.workflow.workflow_cli.check_virtual_environment", return_value=True):
                result = main()

        mock_func.assert_called_once_with("TEST-001", False)

    @patch("tools.workflow.workflow_cli.get_sheet_row")
    def test_cli_sheetget_with_json_flag(self, mock_func):
        """CLIからsheetget --jsonコマンド呼び出し"""
        mock_func.return_value = True

        import sys
        from tools.workflow.workflow_cli import main

        with patch.object(sys, "argv", ["workflow_cli.py", "sheetget", "TEST-001", "--json"]):
            with patch("tools.workflow.workflow_cli.check_virtual_environment", return_value=True):
                result = main()

        mock_func.assert_called_once_with("TEST-001", True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
