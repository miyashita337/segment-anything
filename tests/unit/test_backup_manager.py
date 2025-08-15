#!/usr/bin/env python3
"""
PH2-007: バックアップマネージャー単体テスト
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.common.backup_manager import BackupManager, backup_extraction_results
from features.common.backup_config import BackupConfig


class TestBackupManager(unittest.TestCase):
    """バックアップマネージャーテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_path = self.temp_dir / "workspace"
        self.workspace_path.mkdir(parents=True)
        
        # テスト用ファイル作成
        (self.workspace_path / "extraction").mkdir()
        (self.workspace_path / "quality").mkdir()
        (self.workspace_path / "extraction" / "test1.jpg").write_text("fake image data")
        (self.workspace_path / "quality" / "report.json").write_text('{"test": "data"}')
        (self.workspace_path / "test.log").write_text("log data")
        
        # テスト用設定
        self.config = BackupConfig(
            enabled=True,
            retention_days=1,
            max_backup_size_mb=100,
            backup_targets=["extraction/", "quality/", "*.log"],
            compression_enabled=True
        )
    
    def tearDown(self):
        """テスト後処理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_manager_initialization(self):
        """バックアップマネージャー初期化テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        
        self.assertEqual(manager.workspace_path, self.workspace_path)
        self.assertEqual(manager.config.retention_days, 1)
        self.assertTrue(manager.backup_dir.exists())
    
    def test_collect_backup_targets(self):
        """バックアップ対象ファイル収集テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        backup_files = manager._collect_backup_targets()
        
        # 期待されるファイル
        expected_files = {
            "extraction/test1.jpg",
            "quality/report.json", 
            "test.log"
        }
        
        actual_files = {str(f.relative_to(self.workspace_path)) for f in backup_files}
        self.assertEqual(expected_files, actual_files)
    
    def test_create_backup_success(self):
        """バックアップ作成成功テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        backup_path = manager.create_backup("test_backup")
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())
        self.assertTrue(backup_path.name.startswith("test_backup"))
        self.assertTrue(backup_path.name.endswith(".tar.gz"))
        
        # メタデータファイル確認
        meta_path = backup_path.with_suffix(backup_path.suffix + '.meta.json')
        self.assertTrue(meta_path.exists())
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["backup_name"], "test_backup")
        self.assertEqual(metadata["file_count"], 3)
    
    def test_create_backup_disabled(self):
        """バックアップ無効時のテスト"""
        self.config.enabled = False
        manager = BackupManager(self.workspace_path, self.config)
        backup_path = manager.create_backup("test_backup")
        
        self.assertIsNone(backup_path)
    
    def test_create_backup_no_files(self):
        """バックアップ対象ファイルなしテスト"""
        # 全ファイル削除
        import shutil
        shutil.rmtree(self.workspace_path / "extraction")
        shutil.rmtree(self.workspace_path / "quality")
        (self.workspace_path / "test.log").unlink()
        
        manager = BackupManager(self.workspace_path, self.config)
        backup_path = manager.create_backup("test_backup")
        
        self.assertIsNone(backup_path)
    
    def test_list_backups(self):
        """バックアップリスト取得テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        
        # バックアップ作成
        backup_path1 = manager.create_backup("backup1")
        backup_path2 = manager.create_backup("backup2")
        
        # リスト取得
        backups = manager.list_backups()
        
        self.assertEqual(len(backups), 2)
        backup_names = [b["name"] for b in backups]
        self.assertIn("backup1", backup_names)
        self.assertIn("backup2", backup_names)
    
    def test_backup_statistics(self):
        """バックアップ統計情報テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        
        # バックアップ作成
        manager.create_backup("backup1")
        manager.create_backup("backup2")
        
        # 統計取得
        stats = manager.get_backup_statistics()
        
        self.assertEqual(stats["total_backups"], 2)
        self.assertGreaterEqual(stats["total_size_mb"], 0)  # 小さなファイルの場合は0.0になる可能性がある
        self.assertIsNotNone(stats["oldest_backup"])
        self.assertIsNotNone(stats["newest_backup"])
    
    def test_restore_backup(self):
        """バックアップ復元テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        
        # バックアップ作成
        backup_path = manager.create_backup("restore_test")
        
        # 元ファイル削除
        (self.workspace_path / "test.log").unlink()
        
        # 復元
        restore_success = manager.restore_backup("restore_test")
        
        self.assertTrue(restore_success)
        self.assertTrue((self.workspace_path / "test.log").exists())
    
    def test_restore_backup_not_found(self):
        """存在しないバックアップ復元テスト"""
        manager = BackupManager(self.workspace_path, self.config)
        
        restore_success = manager.restore_backup("nonexistent_backup")
        
        self.assertFalse(restore_success)
    
    def test_backup_extraction_results_function(self):
        """バックアップフック関数テスト"""
        backup_path = backup_extraction_results(self.workspace_path, "hook_test")
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())
    
    @patch('features.common.backup_manager.BackupManager')
    def test_backup_extraction_results_error(self, mock_manager_class):
        """バックアップフック関数エラーテスト"""
        mock_manager = MagicMock()
        mock_manager.create_backup.side_effect = Exception("Test error")
        mock_manager_class.return_value = mock_manager
        
        backup_path = backup_extraction_results(self.workspace_path, "error_test")
        
        self.assertIsNone(backup_path)


class TestBackupConfig(unittest.TestCase):
    """バックアップ設定テスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = BackupConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.retention_days, 7)
        self.assertIsNotNone(config.backup_targets)
        self.assertTrue(config.compression_enabled)
    
    def test_config_validation(self):
        """設定値検証テスト"""
        config = BackupConfig(
            retention_days=0,  # 無効値
            max_backup_size_mb=5,  # 無効値
            compression_level=10,  # 無効値
            backup_targets=[]  # 無効値
        )
        
        errors = config.validate()
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("retention_days" in error for error in errors))
        self.assertTrue(any("max_backup_size_mb" in error for error in errors))
        self.assertTrue(any("compression_level" in error for error in errors))
        self.assertTrue(any("backup_targets" in error for error in errors))
    
    def test_config_file_operations(self):
        """設定ファイル操作テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "backup_config.json"
            
            # 設定保存
            config = BackupConfig(retention_days=14)
            config.save_to_file(config_path)
            
            self.assertTrue(config_path.exists())
            
            # 設定読み込み
            loaded_config = BackupConfig.load_from_file(config_path)
            
            self.assertEqual(loaded_config.retention_days, 14)


if __name__ == '__main__':
    unittest.main()