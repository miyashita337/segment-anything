#!/usr/bin/env python3
"""
実行権限管理システムのテスト
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent))

from tools.progress_tracker.execution_permission import (
    ExecutionPermissionManager,
    PermissionLevel,
    ActionType,
    PermissionViolationError,
    require_permission
)


class TestExecutionPermissionManager(unittest.TestCase):
    """ExecutionPermissionManagerのテスト"""
    
    def setUp(self):
        """テスト前処理"""
        # 一時ファイル作成
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / ".claude_execution_state.json"
        
        # 環境変数設定
        os.environ['CLAUDE_PERMISSION_ENABLED'] = 'true'
        os.environ['CLAUDE_AUTO_APPROVE'] = 'true'  # 自動承認
        
        # マネージャー作成
        self.manager = ExecutionPermissionManager(str(self.state_file))
    
    def tearDown(self):
        """テスト後処理"""
        # 一時ファイル削除
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        
        # 環境変数クリア
        os.environ.pop('CLAUDE_PERMISSION_ENABLED', None)
        os.environ.pop('CLAUDE_AUTO_APPROVE', None)
    
    def test_default_permission_level(self):
        """デフォルト権限レベルのテスト"""
        self.assertEqual(
            self.manager.get_current_level(),
            PermissionLevel.EXECUTE_FULL
        )
    
    def test_set_permission_level(self):
        """権限レベル設定のテスト"""
        self.manager.set_permission_level(PermissionLevel.READ_ONLY)
        self.assertEqual(
            self.manager.get_current_level(),
            PermissionLevel.READ_ONLY
        )
        
        # 状態ファイル確認
        self.assertTrue(self.state_file.exists())
        with open(self.state_file, 'r') as f:
            state = json.load(f)
            self.assertEqual(state['current_level'], 'READ_ONLY')
    
    def test_permission_check_read_only(self):
        """READ_ONLY権限のテスト"""
        self.manager.set_permission_level(PermissionLevel.READ_ONLY)
        
        # 読み取りは許可
        self.assertTrue(
            self.manager.check_permission(ActionType.READ)
        )
        
        # 書き込みは拒否
        self.assertFalse(
            self.manager.check_permission(ActionType.WRITE)
        )
        
        # 削除は拒否
        self.assertFalse(
            self.manager.check_permission(ActionType.DELETE)
        )
    
    def test_permission_check_plan_only(self):
        """PLAN_ONLY権限のテスト"""
        self.manager.set_permission_level(PermissionLevel.PLAN_ONLY)
        
        # 読み取りは許可
        self.assertTrue(
            self.manager.check_permission(ActionType.READ)
        )
        
        # 書き込みは拒否
        self.assertFalse(
            self.manager.check_permission(ActionType.WRITE)
        )
        
        # 実行は拒否
        self.assertFalse(
            self.manager.check_permission(ActionType.EXECUTE)
        )
    
    def test_permission_check_execute_full(self):
        """EXECUTE_FULL権限のテスト"""
        self.manager.set_permission_level(PermissionLevel.EXECUTE_FULL)
        
        # 全て許可
        for action in ActionType:
            self.assertTrue(
                self.manager.check_permission(action),
                f"{action.value} should be allowed in EXECUTE_FULL"
            )
    
    def test_enforce_permission(self):
        """権限強制チェックのテスト"""
        self.manager.set_permission_level(PermissionLevel.READ_ONLY)
        
        # 読み取りは成功
        self.manager.enforce_permission(ActionType.READ, "test.txt")
        
        # 書き込みは例外発生
        with self.assertRaises(PermissionViolationError) as cm:
            self.manager.enforce_permission(ActionType.WRITE, "test.txt")
        
        self.assertIn("権限違反", str(cm.exception))
        self.assertIn("write", str(cm.exception))  # 小文字に修正
        self.assertIn("READ_ONLY", str(cm.exception))
    
    def test_audit_log(self):
        """監査ログのテスト"""
        # アクション実行
        self.manager.check_permission(ActionType.READ, "file1.txt")
        self.manager.check_permission(ActionType.WRITE, "file2.txt")
        
        # ログ取得
        logs = self.manager.get_audit_log(limit=10)
        
        # ログエントリ確認
        self.assertGreaterEqual(len(logs), 2)
        
        # 最新ログ確認
        latest_log = logs[-1]
        self.assertEqual(latest_log['event_type'], 'permission_check')
        self.assertEqual(latest_log['data']['action'], 'write')
        self.assertEqual(latest_log['data']['target'], 'file2.txt')
    
    def test_state_persistence(self):
        """状態永続化のテスト"""
        # 権限設定
        self.manager.set_permission_level(PermissionLevel.PLAN_ONLY)
        
        # 新しいマネージャー作成（同じ状態ファイル）
        new_manager = ExecutionPermissionManager(str(self.state_file))
        
        # 権限レベルが保持されている
        self.assertEqual(
            new_manager.get_current_level(),
            PermissionLevel.PLAN_ONLY
        )
    
    def test_disabled_manager(self):
        """無効化時の動作テスト"""
        os.environ['CLAUDE_PERMISSION_ENABLED'] = 'false'
        disabled_manager = ExecutionPermissionManager(str(self.state_file))
        
        # 無効時は全て許可
        for action in ActionType:
            self.assertTrue(
                disabled_manager.check_permission(action),
                f"{action.value} should be allowed when disabled"
            )
    
    def test_require_permission_decorator(self):
        """権限デコレータのテスト"""
        
        @require_permission(ActionType.WRITE)
        def test_function(path):
            return f"Writing to {path}"
        
        # EXECUTE_FULLでは成功
        self.manager.set_permission_level(PermissionLevel.EXECUTE_FULL)
        result = test_function("test.txt")
        self.assertEqual(result, "Writing to test.txt")
        
        # READ_ONLYでは失敗
        self.manager.set_permission_level(PermissionLevel.READ_ONLY)
        with self.assertRaises(PermissionViolationError):
            test_function("test.txt")


class TestCLIIntegration(unittest.TestCase):
    """CLI統合のテスト"""
    
    def setUp(self):
        """テスト前処理"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / ".claude_execution_state.json"
        os.environ['CLAUDE_PERMISSION_ENABLED'] = 'true'
        os.environ['CLAUDE_AUTO_APPROVE'] = 'true'
    
    def tearDown(self):
        """テスト後処理"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        os.environ.pop('CLAUDE_PERMISSION_ENABLED', None)
        os.environ.pop('CLAUDE_AUTO_APPROVE', None)
    
    @patch('tools.progress_tracker.cli.ProgressManager')
    def test_cli_command_with_permission(self, mock_pm):
        """CLIコマンドの権限チェックテスト"""
        from tools.progress_tracker.cli import cmd_create_task
        
        # モック設定
        mock_manager = MagicMock()
        mock_pm.return_value = mock_manager
        mock_task = MagicMock()
        mock_task.tracker_id = "TEST-001"
        mock_task.description = "Test task"
        mock_manager.create_task.return_value = mock_task
        
        # 引数モック
        args = MagicMock()
        args.tracker_id = "TEST-001"
        args.description = "Test task"
        
        # EXECUTE_FULLでは成功
        manager = ExecutionPermissionManager(str(self.state_file))
        manager.set_permission_level(PermissionLevel.EXECUTE_FULL)
        
        with patch('tools.progress_tracker.cli.get_permission_manager', return_value=manager):
            result = cmd_create_task(args)
            self.assertEqual(result, 0)
        
        # READ_ONLYでは失敗
        manager.set_permission_level(PermissionLevel.READ_ONLY)
        
        with patch('tools.progress_tracker.cli.get_permission_manager', return_value=manager):
            with self.assertRaises(PermissionViolationError):
                cmd_create_task(args)


if __name__ == '__main__':
    unittest.main()