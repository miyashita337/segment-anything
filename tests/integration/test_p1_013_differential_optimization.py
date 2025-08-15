#!/usr/bin/env python3
"""
P1-013差分処理最適化システム統合テスト
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.differential_processor import DifferentialProcessor


class TestP1013DifferentialOptimizationIntegration(unittest.TestCase):
    """P1-013差分処理最適化統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.input_dir.mkdir()
        
        # テスト画像ファイル作成
        self.test_files = []
        for i in range(5):
            test_file = self.input_dir / f"test_image_{i:03d}.jpg"
            # より現実的な画像データサイズ
            test_content = f"JPEG_IMAGE_DATA_{i}" * 1000  # より大きなデータ
            test_file.write_text(test_content)
            self.test_files.append(test_file)
            
        # 出力ディレクトリ
        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir()
        
        # キャッシュディレクトリ
        self.cache_dir = Path(self.test_dir) / "cache"
        self.cache_dir.mkdir()
        
    def tearDown(self):
        """テストクリーンアップ"""
        shutil.rmtree(self.test_dir)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_full_differential_process_initial_run(self, mock_subprocess, mock_path_manager_class):
        """初回実行時の完全差分処理テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # subprocess.runのモック設定（抽出処理成功）
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # 出力ファイル作成をシミュレート
        def create_output_files(*args, **kwargs):
            # 抽出コマンドが呼ばれたら出力ファイルを作成
            if len(args) > 0 and "extract_character.py" in str(args[0]):
                input_file = Path(args[0][2])  # 3番目の引数が入力ファイル
                output_file = self.output_dir / f"{input_file.stem}_extracted.jpg"
                output_file.write_text("extracted image data")
            return mock_result
            
        mock_subprocess.side_effect = create_output_files
        
        # 差分プロセッサ初期化
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 完全差分処理実行
        report = processor.run_full_differential_process()
        
        # 結果検証
        self.assertEqual(report.total_files, 5)
        self.assertEqual(report.changed_files, 5)  # 初回なので全て新規
        self.assertEqual(report.processed_files, 5)  # 全て処理される
        self.assertEqual(report.skipped_files, 0)   # スキップなし
        self.assertEqual(report.failed_files, 0)    # 失敗なし
        
        # 抽出コマンドが5回呼ばれることを確認
        self.assertEqual(mock_subprocess.call_count, 5)
        
        # キャッシュファイルが作成されることを確認
        self.assertTrue((self.cache_dir / "P1-013-TEST_file_hashes.json").exists())
        self.assertTrue((self.cache_dir / "P1-013-TEST_processing_cache.json").exists())
        
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_differential_process_with_changes(self, mock_subprocess, mock_path_manager_class):
        """変更ファイルありの差分処理テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # subprocess.runのモック設定
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        def create_output_files(*args, **kwargs):
            if len(args) > 0 and "extract_character.py" in str(args[0]):
                input_file = Path(args[0][2])
                output_file = self.output_dir / f"{input_file.stem}_extracted.jpg"
                output_file.write_text("extracted image data")
            return mock_result
            
        mock_subprocess.side_effect = create_output_files
        
        # 差分プロセッサ初期化
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 初回実行
        processor.run_full_differential_process()
        
        # モックカウンタリセット
        mock_subprocess.reset_mock()
        
        # 一部ファイルを変更
        modified_file = self.test_files[2]
        modified_file.write_text("MODIFIED_JPEG_IMAGE_DATA" * 1000)
        
        # 新しいファイルを追加
        new_file = self.input_dir / "new_test_image.jpg"
        new_file.write_text("NEW_JPEG_IMAGE_DATA" * 1000)
        
        # 2回目実行
        report = processor.run_full_differential_process()
        
        # 結果検証
        self.assertEqual(report.total_files, 6)  # 5 + 1新規
        self.assertEqual(report.changed_files, 2)  # 1変更 + 1新規
        self.assertEqual(report.processed_files, 2)  # 変更・新規のみ処理
        self.assertEqual(report.skipped_files, 4)   # 残り4つはスキップ
        self.assertEqual(report.failed_files, 0)
        
        # 抽出コマンドが変更・新規分のみ呼ばれることを確認
        self.assertEqual(mock_subprocess.call_count, 2)
        
        # キャッシュヒット率確認
        total_cache_ops = report.cache_hits + report.cache_misses
        if total_cache_ops > 0:
            hit_rate = report.cache_hits / total_cache_ops
            self.assertGreater(hit_rate, 0.5)  # 50%以上のヒット率
            
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_differential_process_with_deleted_files(self, mock_subprocess, mock_path_manager_class):
        """削除ファイルありの差分処理テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # subprocess.runのモック設定
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # 差分プロセッサ初期化
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 初回実行
        processor.run_full_differential_process()
        initial_cache_count = len(processor.processing_cache)
        
        # ファイル削除
        deleted_file = self.test_files[1]
        deleted_file.unlink()
        
        # 2回目実行
        report = processor.run_full_differential_process()
        
        # 結果検証
        self.assertEqual(report.total_files, 4)  # 5 - 1削除
        self.assertEqual(report.changed_files, 1)  # 1削除
        
        # 削除されたファイルのキャッシュエントリが削除されることを確認
        final_cache_count = len(processor.processing_cache)
        self.assertLess(final_cache_count, initial_cache_count)
        
        # 削除変更が検出されることを確認
        deleted_changes = [c for c in report.change_details if c.change_type == 'deleted']
        self.assertEqual(len(deleted_changes), 1)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    @patch('features.processing.differential_processor.subprocess.run')
    def test_differential_process_with_failures(self, mock_subprocess, mock_path_manager_class):
        """処理失敗ありの差分処理テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # 一部失敗する設定
        call_count = 0
        def mixed_results(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_result = MagicMock()
            if call_count <= 2:
                # 最初の2つは成功
                mock_result.returncode = 0
                mock_result.stderr = ""
                # 出力ファイル作成
                if len(args) > 0 and "extract_character.py" in str(args[0]):
                    input_file = Path(args[0][2])
                    output_file = self.output_dir / f"{input_file.stem}_extracted.jpg"
                    output_file.write_text("extracted image data")
            else:
                # 残りは失敗
                mock_result.returncode = 1
                mock_result.stderr = "Processing failed"
                
            return mock_result
            
        mock_subprocess.side_effect = mixed_results
        
        # 差分プロセッサ初期化
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 完全差分処理実行
        report = processor.run_full_differential_process()
        
        # 結果検証
        self.assertEqual(report.total_files, 5)
        self.assertEqual(report.processed_files, 2)  # 2つ成功
        self.assertEqual(report.failed_files, 3)     # 3つ失敗
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_cache_persistence(self, mock_path_manager_class):
        """キャッシュ永続化テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # 1つ目のプロセッサでキャッシュ作成
        processor1 = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 変更検出（キャッシュ構築）
        changes = processor1.detect_changes()
        processor1._save_caches()
        
        # 2つ目のプロセッサで同じキャッシュ読み込み
        processor2 = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # キャッシュが正しく読み込まれることを確認
        self.assertEqual(len(processor2.file_hashes), len(processor1.file_hashes))
        
        # ファイルハッシュが一致することを確認
        for file_path, hash_info in processor1.file_hashes.items():
            self.assertIn(file_path, processor2.file_hashes)
            self.assertEqual(processor2.file_hashes[file_path].file_hash, hash_info.file_hash)
            
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_report_generation(self, mock_path_manager_class):
        """レポート生成テスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 変更検出
        changes = processor.detect_changes()
        report = processor.process_changes(changes)
        
        # レポート保存
        report_file = processor.save_report(report)
        
        # JSONレポート検証
        self.assertTrue(report_file.exists())
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            
        self.assertEqual(report_data['tracker_id'], "P1-013-TEST")
        self.assertEqual(report_data['total_files'], report.total_files)
        self.assertIn('timestamp', report_data)
        self.assertIn('cache_enabled', report_data)
        
        # Markdownレポート検証
        md_report = self.output_dir / "P1-013-TEST_differential_summary.md"
        self.assertTrue(md_report.exists())
        
        md_content = md_report.read_text(encoding='utf-8')
        self.assertIn("P1-013 差分処理最適化レポート", md_content)
        self.assertIn("処理サマリー", md_content)
        self.assertIn("P1-013-TEST", md_content)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_cache_cleanup(self, mock_path_manager_class):
        """キャッシュクリーンアップテスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 古いキャッシュエントリを手動作成
        from features.processing.differential_processor import ProcessingCache
        old_timestamp = time.time() - (35 * 24 * 3600)  # 35日前
        
        processor.processing_cache["old_file_1"] = ProcessingCache(
            input_file="old_file_1",
            input_hash="old_hash_1",
            output_files=[],
            processing_time=1.0,
            success=True,
            timestamp=old_timestamp,
            processing_params={}
        )
        
        processor.processing_cache["old_file_2"] = ProcessingCache(
            input_file="old_file_2",
            input_hash="old_hash_2",
            output_files=[],
            processing_time=1.0,
            success=True,
            timestamp=old_timestamp,
            processing_params={}
        )
        
        # 新しいキャッシュエントリ
        new_timestamp = time.time() - (10 * 24 * 3600)  # 10日前
        processor.processing_cache["new_file"] = ProcessingCache(
            input_file="new_file",
            input_hash="new_hash",
            output_files=[],
            processing_time=1.0,
            success=True,
            timestamp=new_timestamp,
            processing_params={}
        )
        
        # クリーンアップ前のエントリ数確認
        self.assertEqual(len(processor.processing_cache), 3)
        
        # クリーンアップ実行
        processor.cleanup_old_cache(max_age_days=30)
        
        # 古いエントリが削除されることを確認
        self.assertEqual(len(processor.processing_cache), 1)
        self.assertIn("new_file", processor.processing_cache)
        self.assertNotIn("old_file_1", processor.processing_cache)
        self.assertNotIn("old_file_2", processor.processing_cache)
        
    @patch('features.processing.differential_processor.OutputPathManager')
    def test_performance_with_large_dataset(self, mock_path_manager_class):
        """大規模データセットでのパフォーマンステスト"""
        # OutputPathManagerのモック設定
        mock_path_manager = MagicMock()
        mock_path_manager.ensure_output_dir.side_effect = lambda category: {
            'extraction': self.output_dir,
            'cache': self.cache_dir,
            'quality_report': self.output_dir
        }.get(category.value if hasattr(category, 'value') else str(category), self.output_dir)
        mock_path_manager_class.return_value = mock_path_manager
        
        # 大量のテストファイル作成（50個）
        large_test_files = []
        for i in range(50):
            test_file = self.input_dir / f"large_test_{i:03d}.jpg"
            test_file.write_text(f"LARGE_IMAGE_DATA_{i}" * 500)
            large_test_files.append(test_file)
            
        processor = DifferentialProcessor(
            tracker_id="P1-013-TEST",
            input_dir=str(self.input_dir),
            enable_cache=True
        )
        
        # 変更検出のパフォーマンス測定
        start_time = time.time()
        changes = processor.detect_changes()
        detection_time = time.time() - start_time
        
        # 結果検証
        self.assertEqual(len(changes), 50)
        self.assertLess(detection_time, 5.0)  # 5秒以内で完了
        
        # ハッシュ計算が正常に完了することを確認
        for change in changes:
            self.assertNotEqual(change.file_hash, "")
            self.assertGreater(change.file_size, 0)
            

if __name__ == '__main__':
    unittest.main()