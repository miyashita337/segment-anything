#!/usr/bin/env python3
"""
P1-015: メモリ使用最適化統合テスト
大規模データセット対応とメモリ効率化の統合動作テスト
"""

import numpy as np

import pytest

# プロジェクトルートをパスに追加
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.memory_optimizer import BatchMemoryManager
from features.processing.large_dataset_processor import LargeDatasetConfig, LargeDatasetProcessor


class TestP1015MemoryOptimization:
    """P1-015メモリ最適化統合テスト"""

    @pytest.fixture
    def large_test_dataset(self):
        """大規模テストデータセット作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 100個のテスト画像作成（実際の大規模データセットをシミュレート）
            image_files = []
            for i in range(100):
                # 小さめの画像でテスト高速化
                img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)

                img_path = temp_path / f"large_test_{i:03d}.jpg"
                img.save(img_path, quality=85)
                image_files.append(img_path)

            yield image_files, str(temp_path)

    def test_large_dataset_mode_activation(self):
        """大規模データセットモード有効化テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")

        # 初期化時に大規模データセットモードが有効になっている
        assert processor.memory_manager.large_dataset_mode is True

        # メモリ制限が適切に設定されている
        assert processor.memory_manager.max_memory_mb > 0

        # 統計情報確認
        stats = processor.memory_manager.get_memory_stats()
        assert stats["large_dataset_mode"] is True
        assert stats["current_batch_size"] >= 1
        assert stats["max_batch_size"] >= 1

    def test_memory_pressure_detection_simulation(self):
        """メモリ圧迫検知シミュレーションテスト"""
        config = LargeDatasetConfig(memory_threshold_gb=0.01)  # 極小閾値でテスト
        processor = LargeDatasetProcessor("P1-015-TEST", config)

        # メモリ圧迫検知テスト
        with patch("psutil.virtual_memory") as mock_memory:
            mock_mem = Mock()
            mock_mem.total = 8 * 1024**3  # 8GB
            mock_mem.available = 0.5 * 1024**3  # 0.5GB利用可能（高使用率）
            mock_memory.return_value = mock_mem

            should_optimize = processor._should_optimize_memory()
            assert should_optimize is True

    def test_adaptive_batch_size_mechanism(self):
        """動的バッチサイズ調整メカニズムテスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        memory_manager = processor.memory_manager

        # 初期バッチサイズ
        initial_batch_size = memory_manager.current_batch_size
        assert initial_batch_size >= 1

        # メモリ圧迫シミュレーション
        memory_manager.consecutive_pressure_count = 3
        memory_manager._adapt_batch_size()

        # バッチサイズが減少することを確認
        assert memory_manager.current_batch_size <= initial_batch_size

        # 圧迫カウントリセット確認
        assert memory_manager.consecutive_pressure_count == 0

    @patch("subprocess.run")
    def test_chunk_processing_with_memory_optimization(self, mock_subprocess, large_test_dataset):
        """メモリ最適化を伴うチャンク処理テスト"""
        image_files, temp_input_dir = large_test_dataset

        # 成功レスポンスをモック
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Processing completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        config = LargeDatasetConfig(
            chunk_size=10,  # 小さなチャンクサイズ
            memory_threshold_gb=0.5,  # 低い閾値でメモリ最適化をトリガー
            enable_progressive_processing=True,
        )
        processor = LargeDatasetProcessor("P1-015-TEST", config)

        with tempfile.TemporaryDirectory() as output_dir:
            # 出力ファイル作成をシミュレート
            def simulate_processing(*args, **kwargs):
                # 各呼び出しで少数のファイルを作成
                output_path = Path(output_dir)
                for i in range(3):  # チャンクごとに3ファイル作成
                    output_file = output_path / f"processed_{time.time()}_{i}.jpg"
                    output_file.write_text("mock processed image")
                return mock_result

            mock_subprocess.side_effect = simulate_processing

            # 処理実行
            start_time = time.time()
            success = processor.process_large_dataset(
                input_dir=temp_input_dir, processing_params={"score_threshold": 0.07}
            )
            end_time = time.time()

            # 結果検証
            assert success is True

            # 処理統計確認
            stats = processor.get_processing_stats()
            assert stats["total_files"] == 100
            assert stats["chunks_processed"] > 0
            assert stats["total_processing_time"] > 0

            # 処理時間が合理的な範囲内
            processing_time = end_time - start_time
            assert processing_time < 60.0  # 1分以内で完了

            # SAM+YOLOが複数回呼び出された
            assert mock_subprocess.call_count >= 10  # 100ファイル / 10チャンク = 10回以上

    def test_checkpoint_creation_during_processing(self):
        """処理中のチェックポイント作成テスト"""
        config = LargeDatasetConfig(checkpoint_interval=20)  # 小さい間隔でテスト
        processor = LargeDatasetProcessor("P1-015-TEST", config)

        with tempfile.TemporaryDirectory() as temp_dir:
            # パスマネージャーをモック
            with patch.object(processor.path_manager, "get_base_dir", return_value=Path(temp_dir)):
                # チェックポイント作成
                processor._create_checkpoint(2, 10)

                checkpoint_file = Path(temp_dir) / "checkpoint_2.json"
                assert checkpoint_file.exists()

                # 統計更新確認
                assert processor.processing_stats["checkpoints_created"] == 1

    def test_intermediate_cleanup_effectiveness(self):
        """中間クリーンアップ効果テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")

        # 大きなオブジェクトを作成してメモリを消費
        large_objects = []
        for i in range(10):
            large_array = np.random.rand(100, 100, 100)  # 約8MB
            large_objects.append(large_array)

        # メモリ使用量記録
        memory_before = processor.memory_manager.optimizer.monitor.get_memory_usage()

        # オブジェクト削除
        del large_objects

        # 中間クリーンアップ実行
        processor._intermediate_cleanup()

        # メモリ使用量確認（完全ではないが、エラーが発生しないことを確認）
        memory_after = processor.memory_manager.optimizer.monitor.get_memory_usage()
        assert isinstance(memory_after["ram_mb"], float)

    @patch("subprocess.run")
    def test_error_handling_during_large_dataset_processing(self, mock_subprocess):
        """大規模データセット処理中のエラーハンドリングテスト"""
        # 失敗レスポンスをモック
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Processing failed"
        mock_subprocess.return_value = mock_result

        processor = LargeDatasetProcessor("P1-015-TEST")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # テスト画像作成
            for i in range(5):
                img_path = temp_path / f"test_{i}.jpg"
                img_path.write_text("dummy image")

            # 処理実行（失敗が予想される）
            success = processor.process_large_dataset(
                input_dir=str(temp_path), processing_params={}
            )

            # 失敗が適切に処理されている
            assert success is False

            # 失敗統計が記録されている
            stats = processor.get_processing_stats()
            assert stats["failed_files"] > 0

    def test_processing_stats_accuracy(self):
        """処理統計精度テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")

        # 初期統計確認
        initial_stats = processor.get_processing_stats()
        assert initial_stats["total_files"] == 0
        assert initial_stats["processed_files"] == 0
        assert initial_stats["successful_files"] == 0
        assert initial_stats["failed_files"] == 0

        # 統計手動更新
        processor.processing_stats["total_files"] = 50
        processor.processing_stats["successful_files"] = 45
        processor.processing_stats["failed_files"] = 5

        updated_stats = processor.get_processing_stats()
        assert updated_stats["total_files"] == 50
        assert updated_stats["successful_files"] == 45
        assert updated_stats["failed_files"] == 5

        # メモリ統計も含まれている
        assert "memory_stats" in updated_stats
        assert isinstance(updated_stats["memory_stats"], dict)


class TestP1015PerformanceCharacteristics:
    """P1-015パフォーマンス特性テスト"""

    def test_memory_usage_tracking(self):
        """メモリ使用量追跡テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")
        memory_manager = processor.memory_manager

        # メモリ監視開始
        memory_manager.optimizer.monitor.start_monitoring(interval=0.1)

        # 短時間待機
        time.sleep(0.3)

        # メモリ使用量取得
        current_memory = memory_manager.optimizer.monitor.get_memory_usage()
        assert "ram_mb" in current_memory
        assert current_memory["ram_mb"] > 0

        # 監視停止
        memory_manager.optimizer.monitor.stop_monitoring()

    def test_batch_processing_efficiency(self):
        """バッチ処理効率テスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")

        def mock_processing_function(item_id):
            # 軽量な処理をシミュレート
            return {"id": item_id, "processed": True}

        # バッチ処理実行
        start_time = time.time()
        results = []
        for i in range(20):
            result = processor.memory_manager.process_batch_item(mock_processing_function, i)
            results.append(result)
        end_time = time.time()

        # 結果検証
        assert len(results) == 20
        assert all(r["processed"] for r in results)

        # 処理時間が合理的
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 5秒以内

        # メモリ管理統計確認
        stats = processor.memory_manager.get_memory_stats()
        assert stats["processed_items"] == 20

    def test_scalability_with_different_chunk_sizes(self):
        """異なるチャンクサイズでのスケーラビリティテスト"""
        chunk_sizes = [5, 10, 25, 50]

        for chunk_size in chunk_sizes:
            config = LargeDatasetConfig(chunk_size=chunk_size)
            processor = LargeDatasetProcessor(f"P1-015-TEST-{chunk_size}", config)

            # 設定が正しく適用されている
            assert processor.config.chunk_size == chunk_size
            assert processor.memory_manager.max_batch_size == chunk_size

            # 100ファイルのチャンク分割テスト
            files = [Path(f"file_{i}.jpg") for i in range(100)]
            chunks = processor._create_file_chunks(files)

            expected_chunks = (100 + chunk_size - 1) // chunk_size  # 切り上げ除算
            assert len(chunks) == expected_chunks


class TestP1015RealWorldScenarios:
    """P1-015現実的シナリオテスト"""

    @patch("subprocess.run")
    def test_mixed_success_failure_scenario(self, mock_subprocess):
        """成功・失敗混在シナリオテスト"""

        # 交互に成功・失敗を返すモック
        call_count = 0

        def alternating_result(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % 2 == 1:  # 奇数回目は成功
                result = Mock()
                result.returncode = 0
                result.stdout = "Success"
            else:  # 偶数回目は失敗
                result = Mock()
                result.returncode = 1
                result.stderr = "Failure"

            return result

        mock_subprocess.side_effect = alternating_result

        config = LargeDatasetConfig(chunk_size=5)
        processor = LargeDatasetProcessor("P1-015-TEST", config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 20個のテスト画像作成
            for i in range(20):
                img_path = temp_path / f"mixed_test_{i}.jpg"
                img_path.write_text(f"dummy image {i}")

            # 処理実行
            success = processor.process_large_dataset(
                input_dir=str(temp_path), processing_params={}
            )

            # 部分的成功の場合の処理確認
            stats = processor.get_processing_stats()
            assert stats["total_files"] == 20
            assert stats["chunks_processed"] >= 1  # 一部は成功

            # 成功率が70%未満の場合は全体として失敗
            # （この例では50%なので失敗扱い）
            assert success is False

    def test_resume_capability_simulation(self):
        """レジューム機能シミュレーションテスト"""
        processor = LargeDatasetProcessor("P1-015-TEST")

        with tempfile.TemporaryDirectory() as temp_dir:
            # チェックポイントファイルをシミュレート
            checkpoint_data = {
                "tracker_id": "P1-015-TEST",
                "current_chunk": 3,
                "total_chunks": 10,
                "processing_stats": {
                    "total_files": 100,
                    "processed_files": 30,
                    "successful_files": 28,
                    "failed_files": 2,
                },
                "timestamp": time.time(),
            }

            checkpoint_file = Path(temp_dir) / "checkpoint_3.json"

            import json

            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            # チェックポイントファイルが作成されている
            assert checkpoint_file.exists()

            # 実際のレジューム機能は将来の拡張として位置づけ
            # 現在はチェックポイントの作成・読み込み可能性を確認
            with open(checkpoint_file) as f:
                loaded_data = json.load(f)

            assert loaded_data["tracker_id"] == "P1-015-TEST"
            assert loaded_data["current_chunk"] == 3
            assert loaded_data["processing_stats"]["successful_files"] == 28


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
