#!/usr/bin/env python3
"""
SpreadSheet API改善のテスト
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).parent.parent))

try:
    from tools.progress_tracker.sheets_client import GoogleSheetsClient
    from tools.progress_tracker.config import get_default_config
    from tools.progress_tracker.connection_monitor import ConnectionMonitor
    from tools.progress_tracker.data_models import ProgressTrackerConfig
except ImportError as e:
    pytest.skip(f"Progress tracker modules not available: {e}", allow_module_level=True)


class TestAPIImprovements:
    """API改善のテスト"""
    
    def test_sheet_name_detection(self):
        """シート名自動検出のテスト"""
        config = ProgressTrackerConfig(
            spreadsheet_id="test_id",
            sheet_name="Progress Tracker",
            auth_file_path="test_auth.json"
        )
        
        # モックデータ
        mock_metadata = {
            'sheets': [
                {'properties': {'title': 'シート1', 'sheetId': 0}},
                {'properties': {'title': 'Sheet2', 'sheetId': 1}}
            ]
        }
        
        with patch('tools.progress_tracker.sheets_client.GoogleSheetsClient._authenticate'):
            with patch.object(GoogleSheetsClient, 'service') as mock_service:
                mock_service.spreadsheets().get().execute.return_value = mock_metadata
                
                client = GoogleSheetsClient(config)
                
                # 実際のシート名が検出されること
                assert client._actual_sheet_name == 'シート1'
    
    def test_safe_range_generation(self):
        """安全な範囲指定のテスト"""
        config = ProgressTrackerConfig(
            spreadsheet_id="test_id",
            sheet_name="Progress Tracker",
            auth_file_path="test_auth.json"
        )
        
        with patch('tools.progress_tracker.sheets_client.GoogleSheetsClient._authenticate'):
            client = GoogleSheetsClient(config)
            client._actual_sheet_name = "シート1"
            
            # 特殊文字を含むシート名の安全な処理
            safe_range = client._get_safe_range("A1:U1")
            assert "シート1" in safe_range
            assert "A1:U1" in safe_range
    
    def test_connection_monitoring(self):
        """接続監視のテスト"""
        monitor = ConnectionMonitor()
        
        # 履歴が正しく管理されること
        assert len(monitor.health_history) == 0
        assert monitor.max_history == 100
        
        # サマリー生成
        summary = monitor.get_connection_summary()
        assert summary["status"] == "no_data"
    
    def test_error_handling_improvements(self):
        """エラーハンドリング改善のテスト"""
        config = ProgressTrackerConfig(
            spreadsheet_id="test_id",
            sheet_name="Progress Tracker",
            auth_file_path="test_auth.json"
        )
        
        with patch('tools.progress_tracker.sheets_client.GoogleSheetsClient._authenticate'):
            client = GoogleSheetsClient(config)
            
            # 手動更新ガイドが表示されること
            with patch('builtins.print') as mock_print:
                client._show_manual_update_guide()
                
                # 呼び出し回数を確認
                assert mock_print.call_count > 5
                
                # 重要な情報が含まれること
                printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
                assert "PH2-002" in printed_text
                assert "実装完了" in printed_text
                assert config.spreadsheet_id in printed_text


def test_integration_workflow():
    """統合ワークフローテスト"""
    # 実際の設定で基本機能テスト
    config = get_default_config()
    
    # 設定値が正しいこと
    assert config.spreadsheet_id
    assert config.sheet_name
    assert config.auth_file_path
    
    # 接続監視機能
    monitor = ConnectionMonitor()
    assert monitor.config.spreadsheet_id == config.spreadsheet_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])