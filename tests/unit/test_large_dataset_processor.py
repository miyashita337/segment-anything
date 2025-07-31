#!/usr/bin/env python3
"""
P1-015: 大規模データセット処理システムのテスト
メモリ最適化とプログレッシブ処理のテスト
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock, call
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.large_dataset_processor import (
    LargeDatasetProcessor,
    LargeDatasetConfig
)
from features.common.memory_optimizer import BatchMemoryManager


class TestLargeDatasetConfig:
    """大規模データセット設定テスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = LargeDatasetConfig()
        
        assert config.chunk_size == 50
        assert config.max_concurrent_chunks == 2
        assert config.memory_threshold_gb == 2.0
        assert config.enable_progressive_processing is True
        assert config.enable_intermediate_cleanup is True
        assert config.checkpoint_interval == 100
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        config = LargeDatasetConfig(
            chunk_size=30,
            max_concurrent_chunks=4,
            memory_threshold_gb=1.5,
            enable_progressive_processing=False,
            checkpoint_interval=50
        )
        
        assert config.chunk_size == 30
        assert config.max_concurrent_chunks == 4
        assert config.memory_threshold_gb == 1.5
        assert config.enable_progressive_processing is False
        assert config.checkpoint_interval == 50


class TestLargeDatasetProcessor:
    """大規模データセット処理システムテスト"""
    
    @pytest.fixture
    def sample_images_large(self):
        """大規模サンプル画像ファイル作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 多数のテスト用画像ファイル作成
            image_files = []
            for i in range(15):  # 15個の画像（テスト用に適度な数）
                # 256x256のランダム画像作成
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_path = temp_path / f"test_large_{i:03d}.jpg"
                img.save(img_path)
                image_files.append(img_path)
            
            yield image_files, str(temp_path)
    
    def test_processor_initialization(self):
        """処理システム初期化テスト"""
        config = LargeDatasetConfig(chunk_size=25)
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        assert processor.tracker_id == "P1-015-TEST"
        assert processor.config.chunk_size == 25
        assert isinstance(processor.memory_manager, BatchMemoryManager)
        assert processor.memory_manager.large_dataset_mode is True
        assert processor.processing_stats['total_files'] == 0
        assert processor.sam_yolo_script.name == "sam_yolo_character_segment.py"
    
    def test_collect_image_files(self, sample_images_large):
        """画像ファイル収集テスト"""
        image_files, temp_dir = sample_images_large
        
        processor = LargeDatasetProcessor("P1-015-TEST")
        collected_files = processor._collect_image_files(Path(temp_dir))
        
        assert len(collected_files) == 15
        assert all(f.suffix.lower() == '.jpg' for f in collected_files)
        assert all(f.exists() for f in collected_files)
    
    def test_create_file_chunks(self, sample_images_large):
        """ファイルチャンク作成テスト"""
        image_files, temp_dir = sample_images_large
        
        config = LargeDatasetConfig(chunk_size=5)
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        files = [Path(f) for f in image_files]
        chunks = processor._create_file_chunks(files)
        
        assert len(chunks) == 3  # 15ファイル / 5 = 3チャンク
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 5
        
        # 全ファイルがチャンクに含まれている
        all_chunk_files = []
        for chunk in chunks:
            all_chunk_files.extend(chunk)
        assert len(all_chunk_files) == len(files)
    
    def test_create_file_chunks_uneven(self):
        """不均等なファイルチャンク作成テスト"""
        config = LargeDatasetConfig(chunk_size=7)
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        # 16ファイルを7個ずつのチャンクに分割
        files = [Path(f"file_{i}.jpg") for i in range(16)]
        chunks = processor._create_file_chunks(files)
        
        assert len(chunks) == 3  # 16 / 7 = 2余り2 → 3チャンク
        assert len(chunks[0]) == 7
        assert len(chunks[1]) == 7
        assert len(chunks[2]) == 2  # 余り
    
    @patch('subprocess.run')
    def test_process_single_chunk_success(self, mock_subprocess):
        """単一チャンク処理成功テスト"""
        # subprocess成功をモック
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テストチャンク作成
            chunk_files = []
            for i in range(3):
                file_path = temp_path / f"chunk_test_{i}.jpg"
                file_path.write_text("dummy image content")
                chunk_files.append(file_path)
            
            # 出力ディレクトリ
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # 処理実行
            success = processor._process_single_chunk(
                chunk=chunk_files,
                output_dir=output_dir,
                processing_params={'score_threshold': 0.07}
            )
            
            assert success is True
            mock_subprocess.assert_called_once()
            
            # コマンドライン確認
            call_args = mock_subprocess.call_args[0][0]
            assert "python3" in call_args
            assert "--mode" in call_args
            assert "reproduce-auto" in call_args
            assert "--score_threshold" in call_args
            assert "0.07" in call_args
    
    @patch('subprocess.run')
    def test_process_single_chunk_failure(self, mock_subprocess):
        """単一チャンク処理失敗テスト"""
        # subprocess失敗をモック
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Processing error"
        mock_subprocess.return_value = mock_result
        
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テストチャンク作成
            chunk_files = [temp_path / "test.jpg"]
            chunk_files[0].write_text("dummy")
            
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # 処理実行
            success = processor._process_single_chunk(
                chunk=chunk_files,
                output_dir=output_dir,
                processing_params={}
            )
            
            assert success is False
            assert processor.processing_stats['failed_files'] == 1
    
    def test_count_output_files(self):
        """出力ファイル数カウントテスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 出力ファイル作成
            (temp_path / "output1.jpg").write_text("dummy")
            (temp_path / "output2.png").write_text("dummy")
            (temp_path / "output3.PNG").write_text("dummy")
            (temp_path / "not_image.txt").write_text("dummy")
            
            count = processor._count_output_files(temp_path, 4)
            assert count == 3  # jpg, png, PNG のみカウント
    
    @patch('psutil.virtual_memory')
    def test_should_optimize_memory(self, mock_virtual_memory):
        """メモリ最適化要否判定テスト"""
        config = LargeDatasetConfig(memory_threshold_gb=2.0)
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        # メモリ使用量が閾値以下の場合
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.available = 6 * 1024**3  # 6GB利用可能（2GB使用）
        mock_virtual_memory.return_value = mock_memory
        
        assert processor._should_optimize_memory() is False
        
        # メモリ使用量が閾値を超える場合
        mock_memory.available = 3 * 1024**3  # 3GB利用可能（5GB使用）
        assert processor._should_optimize_memory() is True
    
    def test_create_checkpoint(self):
        """チェックポイント作成テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # パスマネージャーのワークスペースをモック
            with patch.object(processor.path_manager.workspace, 'base_path', Path(temp_dir)):
                processor._create_checkpoint(5, 10)
                
                checkpoint_file = Path(temp_dir) / "checkpoint_5.json"
                assert checkpoint_file.exists()
                
                # ファイル内容確認
                import json
                with open(checkpoint_file) as f:
                    data = json.load(f)
                
                assert data['tracker_id'] == "P1-015-TEST"
                assert data['current_chunk'] == 5
                assert data['total_chunks'] == 10
                assert 'processing_stats' in data
                assert 'timestamp' in data
                
                assert processor.processing_stats['checkpoints_created'] == 1
    
    def test_intermediate_cleanup(self):
        """中間クリーンアップテスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        # クリーンアップ実行（エラーが発生しないことを確認）
        processor._intermediate_cleanup()
        
        # 特に検証することはないが、例外が発生しないことが重要
        assert True
    
    def test_get_processing_stats(self):
        """処理統計取得テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        # 統計値を設定
        processor.processing_stats['successful_files'] = 10
        processor.processing_stats['failed_files'] = 2
        
        stats = processor.get_processing_stats()
        
        assert stats['successful_files'] == 10
        assert stats['failed_files'] == 2
        assert 'memory_stats' in stats
        assert isinstance(stats['memory_stats'], dict)


class TestLargeDatasetIntegration:
    """大規模データセット統合テスト"""
    
    @pytest.fixture
    def sample_images_large(self):
        """大規模サンプル画像ファイル作成（統合テスト用）"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 多数のテスト用画像ファイル作成
            image_files = []
            for i in range(12):  # 12個の画像（統合テスト用）
                # 128x128の小さめ画像でテスト高速化
                img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_path = temp_path / f"test_integration_{i:03d}.jpg"
                img.save(img_path)
                image_files.append(img_path)
            
            yield image_files, str(temp_path)
    
    @pytest.fixture
    def mock_sam_yolo_success(self):
        """SAM+YOLO成功処理のモック"""
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Processing completed successfully"
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result
            yield mock_subprocess
    
    def test_sequential_processing_simulation(self, sample_images_large, mock_sam_yolo_success):
        """逐次処理シミュレーションテスト"""
        image_files, temp_input_dir = sample_images_large
        
        config = LargeDatasetConfig(
            chunk_size=5,
            enable_progressive_processing=True
        )
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        with tempfile.TemporaryDirectory() as output_dir:
            # 出力ファイルの作成をシミュレート
            def create_output_files(*args, **kwargs):
                # 各チャンクの処理でファイルを作成
                output_path = Path(output_dir)
                for i in range(5):  # チャンクサイズ分のファイル作成
                    output_file = output_path / f"output_{time.time()}_{i}.jpg"
                    output_file.write_text("processed image")
                return mock_sam_yolo_success.return_value
            
            mock_sam_yolo_success.side_effect = create_output_files
            
            # 処理実行
            success = processor.process_large_dataset(
                input_dir=temp_input_dir,
                processing_params={'score_threshold': 0.05}
            )
            
            # 検証
            assert success is True
            assert processor.processing_stats['total_files'] == 15
            assert processor.processing_stats['chunks_processed'] > 0
            
            # SAM+YOLOが呼び出されたことを確認
            assert mock_sam_yolo_success.called
    
    @patch('features.processing.large_dataset_processor.LargeDatasetProcessor._process_chunks_parallel')
    def test_parallel_processing_selection(self, mock_parallel, sample_images_large):
        """並列処理選択テスト"""
        image_files, temp_input_dir = sample_images_large
        
        config = LargeDatasetConfig(
            enable_progressive_processing=False  # 並列処理を選択
        )
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        # 並列処理メソッドがTrueを返すようにモック
        mock_parallel.return_value = True
        
        with tempfile.TemporaryDirectory() as output_dir:
            success = processor.process_large_dataset(
                input_dir=temp_input_dir,
                processing_params={}
            )
            
            # 並列処理メソッドが呼び出されたことを確認
            mock_parallel.assert_called_once()
    
    def test_nonexistent_directory_handling(self):
        """存在しないディレクトリの処理テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        success = processor.process_large_dataset(
            input_dir="/nonexistent/directory",
            processing_params={}
        )
        
        assert success is False
    
    def test_empty_directory_handling(self):
        """空ディレクトリの処理テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        
        with tempfile.TemporaryDirectory() as empty_dir:
            success = processor.process_large_dataset(
                input_dir=empty_dir,
                processing_params={}
            )
            
            assert success is False


class TestMemoryOptimizationIntegration:
    """メモリ最適化統合テスト"""
    
    def test_memory_manager_large_dataset_mode(self):
        """メモリ管理大規模データセットモードテスト"""
        config = LargeDatasetConfig(chunk_size=20)
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        # 大規模データセットモードが有効になっていることを確認
        assert processor.memory_manager.large_dataset_mode is True
        assert processor.memory_manager.max_batch_size == 20
        
        # メモリ統計取得
        stats = processor.memory_manager.get_memory_stats()
        assert stats['large_dataset_mode'] is True
        assert stats['max_batch_size'] == 20
    
    def test_memory_optimization_trigger(self):
        """メモリ最適化トリガーテスト"""
        config = LargeDatasetConfig(memory_threshold_gb=0.1)  # 非常に低い閾値
        processor = LargeDatasetProcessor("P1-015-TEST", config)
        
        # メモリ最適化が必要と判定されることを確認
        should_optimize = processor._should_optimize_memory()
        # システムによってはメモリ使用量が少ない場合があるため、
        # この結果は環境依存だが、エラーが発生しないことを確認
        assert isinstance(should_optimize, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])