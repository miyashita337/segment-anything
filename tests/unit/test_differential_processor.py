#!/usr/bin/env python3
"""
P1-013差分処理最適化システムのテスト
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.differential_processor import (
    DifferentialProcessor,
    FileChangeInfo,
    ProcessingCache,
    DifferentialReport
)


class TestDifferentialProcessor(unittest.TestCase):
    """差分処理最適化システムテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.input_dir.mkdir()
        
        # テスト画像ファイル作成
        self.test_files = [
            self.input_dir / "test1.jpg",
            self.input_dir / "test2.png",
            self.input_dir / "test3.jpeg"
        ]
        
        for test_file in self.test_files:
            test_file.write_text(f"test image data {test_file.name}")
            
        # モックのpath_manager設定
        self.mock_path_manager = MagicMock()
        self.mock_path_manager.ensure_output_dir.return_value = Path(self.test_dir) / "output"
        
    def tearDown(self):
        """テストクリーンアップ"""
        shutil.rmtree(self.test_dir)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_initialization(self, mock_path_manager_class):
        """初期化テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        self.assertEqual(processor.tracker_id, "P1-013-TEST")
        self.assertEqual(processor.input_dir, self.input_dir)
        self.assertTrue(processor.enable_cache)
        self.assertIsInstance(processor.file_hashes, dict)
        self.assertIsInstance(processor.processing_cache, dict)
        
    def test_calculate_file_hash(self):
        """ファイルハッシュ計算テスト"""
        # テストファイル作成
        test_file = self.input_dir / "hash_test.txt"
        test_content = "test content for hash calculation"
        test_file.write_text(test_content)
        
        with patch('features.processing.differential_processor.OutputPathManager'):
            processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
            
            # ハッシュ計算
            calculated_hash = processor._calculate_file_hash(test_file)
            
            # 期待値計算
            expected_hash = hashlib.sha256(test_content.encode()).hexdigest()
            
            self.assertEqual(calculated_hash, expected_hash)
            
    def test_get_file_info(self):
        """ファイル情報取得テスト"""
        test_file = self.test_files[0]
        
        with patch('features.processing.differential_processor.OutputPathManager'):
            processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
            
            file_info = processor._get_file_info(test_file)
            
            self.assertIsNotNone(file_info)
            self.assertEqual(file_info.file_path, str(test_file))
            self.assertGreater(file_info.last_modified, 0)
            self.assertNotEqual(file_info.file_hash, "")
            self.assertGreater(file_info.file_size, 0)
            
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_detect_changes_new_files(self, mock_path_manager_class):
        """新規ファイル変更検出テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # 変更検出実行
        changes = processor.detect_changes()
        
        # 全ファイルが新規として検出されることを確認
        self.assertEqual(len(changes), 3)
        for change in changes:
            self.assertEqual(change.change_type, 'added')
            
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_detect_changes_modified_files(self, mock_path_manager_class):
        """変更ファイル検出テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # 初回変更検出
        processor.detect_changes()
        
        # ファイル変更
        modified_file = self.test_files[0]
        modified_file.write_text("modified content")
        
        # 再度変更検出
        changes = processor.detect_changes()
        
        # 変更されたファイルが検出されることを確認
        modified_changes = [c for c in changes if c.change_type == 'modified']
        self.assertEqual(len(modified_changes), 1)
        self.assertEqual(modified_changes[0].file_path, str(modified_file))
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_detect_changes_deleted_files(self, mock_path_manager_class):
        """削除ファイル検出テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # 初回変更検出
        processor.detect_changes()
        
        # ファイル削除
        deleted_file = self.test_files[0]
        deleted_file.unlink()
        
        # 再度変更検出
        changes = processor.detect_changes()
        
        # 削除されたファイルが検出されることを確認
        deleted_changes = [c for c in changes if c.change_type == 'deleted']
        self.assertEqual(len(deleted_changes), 1)
        self.assertEqual(deleted_changes[0].file_path, str(deleted_file))
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_should_process_file_cache_disabled(self, mock_path_manager_class):
        """キャッシュ無効時の処理要否判定テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir), enable_cache=False)
        
        change_info = FileChangeInfo(
            file_path=str(self.test_files[0]),
            last_modified=time.time(),
            file_hash="test_hash",
            file_size=100,
            change_type='added',
            dependencies=[]
        )
        
        # キャッシュ無効時は常に処理必要
        self.assertTrue(processor._should_process_file(str(self.test_files[0]), change_info))
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_should_process_file_cache_hit(self, mock_path_manager_class):
        """キャッシュヒット時の処理要否判定テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir), enable_cache=True)
        
        # キャッシュエントリ作成
        file_path = str(self.test_files[0])
        file_hash = "test_hash"
        output_file = Path(self.test_dir) / "output" / "test1_output.jpg"
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text("output data")
        
        cache_entry = ProcessingCache(
            input_file=file_path,
            input_hash=file_hash,
            output_files=[str(output_file)],
            processing_time=1.0,
            success=True,
            timestamp=time.time(),
            processing_params={}
        )
        processor.processing_cache[file_path] = cache_entry
        
        change_info = FileChangeInfo(
            file_path=file_path,
            last_modified=time.time(),
            file_hash=file_hash,  # 同じハッシュ
            file_size=100,
            change_type='modified',
            dependencies=[]
        )
        
        # キャッシュヒット時は処理不要
        self.assertFalse(processor._should_process_file(file_path, change_info))
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_should_process_file_cache_miss(self, mock_path_manager_class):
        """キャッシュミス時の処理要否判定テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir), enable_cache=True)
        
        change_info = FileChangeInfo(
            file_path=str(self.test_files[0]),
            last_modified=time.time(),
            file_hash="new_hash",
            file_size=100,
            change_type='modified',
            dependencies=[]
        )
        
        # キャッシュエントリなし時は処理必要
        self.assertTrue(processor._should_process_file(str(self.test_files[0]), change_info))
        
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_process_single_file_success(self, mock_subprocess, mock_path_manager_class):
        """単一ファイル処理成功テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        # subprocess.runのモック設定
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # 出力ディレクトリ作成
        output_dir = Path(self.test_dir) / "output"
        output_dir.mkdir()
        self.mock_path_manager.ensure_output_dir.return_value = output_dir
        
        # 出力ファイル作成（処理成功シミュレーション）
        output_file = output_dir / "test1_extracted.jpg"
        output_file.write_text("extracted image data")
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        change_info = FileChangeInfo(
            file_path=str(self.test_files[0]),
            last_modified=time.time(),
            file_hash="test_hash",
            file_size=100,
            change_type='added',
            dependencies=[]
        )
        
        # 単一ファイル処理実行
        result = processor._process_single_file(str(self.test_files[0]), change_info)
        
        self.assertTrue(result)
        self.assertEqual(processor.stats['processed_files'], 1)
        self.assertEqual(processor.stats['failed_files'], 0)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_process_single_file_failure(self, mock_subprocess, mock_path_manager_class):
        """単一ファイル処理失敗テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        # subprocess.runのモック設定（失敗）
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Processing error"
        mock_subprocess.return_value = mock_result
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        change_info = FileChangeInfo(
            file_path=str(self.test_files[0]),
            last_modified=time.time(),
            file_hash="test_hash",
            file_size=100,
            change_type='added',
            dependencies=[]
        )
        
        # 単一ファイル処理実行
        result = processor._process_single_file(str(self.test_files[0]), change_info)
        
        self.assertFalse(result)
        self.assertEqual(processor.stats['processed_files'], 0)
        self.assertEqual(processor.stats['failed_files'], 1)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_process_changes(self, mock_path_manager_class):
        """変更処理テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # モック変更リスト
        changes = [
            FileChangeInfo(
                file_path=str(self.test_files[0]),
                last_modified=time.time(),
                file_hash="hash1",
                file_size=100,
                change_type='added',
                dependencies=[]
            ),
            FileChangeInfo(
                file_path=str(self.test_files[1]),
                last_modified=time.time(),
                file_hash="hash2",
                file_size=200,
                change_type='modified',
                dependencies=[]
            )
        ]
        
        # 処理関数をモック化
        with patch.object(processor, '_should_process_file') as mock_should_process, \
             patch.object(processor, '_process_single_file') as mock_process_file:
            
            # 1つ目は処理、2つ目はスキップ
            mock_should_process.side_effect = [True, False]
            mock_process_file.return_value = True
            
            # 変更処理実行
            report = processor.process_changes(changes)
            
            # 結果確認
            self.assertIsInstance(report, DifferentialReport)
            self.assertEqual(report.changed_files, 2)
            
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_save_report(self, mock_path_manager_class):
        """レポート保存テスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        report_dir = Path(self.test_dir) / "reports"
        report_dir.mkdir()
        self.mock_path_manager.ensure_output_dir.return_value = report_dir
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # テストレポート作成
        report = DifferentialReport(
            total_files=10,
            changed_files=3,
            processed_files=2,
            skipped_files=1,
            failed_files=0,
            processing_time=15.5,
            cache_hits=1,
            cache_misses=2,
            change_details=[]
        )
        
        # レポート保存
        report_file = processor.save_report(report)
        
        # ファイル存在確認
        self.assertTrue(report_file.exists())
        self.assertTrue((report_dir / f"P1-013-TEST_differential_summary.md").exists())
        
        # JSON内容確認
        with open(report_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            
        self.assertEqual(saved_data['total_files'], 10)
        self.assertEqual(saved_data['changed_files'], 3)
        self.assertEqual(saved_data['tracker_id'], "P1-013-TEST")
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_cleanup_old_cache(self, mock_path_manager_class):
        """古いキャッシュクリーンアップテスト"""
        mock_path_manager_class.return_value = self.mock_path_manager
        
        processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
        
        # 古いキャッシュエントリ作成
        old_timestamp = time.time() - (35 * 24 * 3600)  # 35日前
        new_timestamp = time.time() - (10 * 24 * 3600)  # 10日前
        
        processor.processing_cache["old_file"] = ProcessingCache(
            input_file="old_file",
            input_hash="old_hash",
            output_files=[],
            processing_time=1.0,
            success=True,
            timestamp=old_timestamp,
            processing_params={}
        )
        
        processor.processing_cache["new_file"] = ProcessingCache(
            input_file="new_file",
            input_hash="new_hash",
            output_files=[],
            processing_time=1.0,
            success=True,
            timestamp=new_timestamp,
            processing_params={}
        )
        
        # クリーンアップ実行（30日以上古いものを削除）
        processor.cleanup_old_cache(max_age_days=30)
        
        # 古いエントリが削除され、新しいエントリが残ることを確認
        self.assertNotIn("old_file", processor.processing_cache)
        self.assertIn("new_file", processor.processing_cache)
        
    def test_cache_file_operations(self):
        """キャッシュファイル操作テスト"""
        with patch('features.processing.differential_processor.OutputPathManager'):
            processor = DifferentialProcessor("P1-013-TEST", str(self.input_dir))
            
            # テストデータ設定
            processor.file_hashes["test_file"] = FileChangeInfo(
                file_path="test_file",
                last_modified=time.time(),
                file_hash="test_hash",
                file_size=100,
                change_type='added',
                dependencies=[]
            )
            
            processor.processing_cache["test_file"] = ProcessingCache(
                input_file="test_file",
                input_hash="test_hash",
                output_files=["output_file"],
                processing_time=1.0,
                success=True,
                timestamp=time.time(),
                processing_params={}
            )
            
            # キャッシュ保存のモック
            with patch('builtins.open', mock_open()) as mock_file, \
                 patch('json.dump') as mock_json_dump:
                
                processor._save_caches()
                
                # ファイルが開かれたことを確認
                self.assertEqual(mock_file.call_count, 3)  # 3つのキャッシュファイル
                self.assertEqual(mock_json_dump.call_count, 3)
                

if __name__ == '__main__':
    unittest.main()