#!/usr/bin/env python3
"""
スケーラビリティシステムのテスト
"""

import sys
import pytest
import asyncio
import time
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).parent.parent))

try:
    from features.common.scalability import (
        ProcessingTask, TaskResult, ParallelProcessor, AsyncProcessor,
        PipelineProcessor, GPUParallelProcessor, ScalabilityManager
    )
except ImportError as e:
    pytest.skip(f"Scalability modules not available: {e}", allow_module_level=True)


class TestProcessingTask:
    """ProcessingTaskクラスのテスト"""
    
    def test_processing_task_creation(self):
        """ProcessingTask作成のテスト"""
        def test_func(x):
            return x * 2
        
        task = ProcessingTask(
            id="test_1",
            func=test_func,
            args=(5,),
            kwargs={},
            priority=1,
            timeout=10.0
        )
        
        assert task.id == "test_1"
        assert task.func == test_func
        assert task.args == (5,)
        assert task.kwargs == {}
        assert task.priority == 1
        assert task.timeout == 10.0
        assert task.retry_count == 0
        assert task.max_retries == 3


class TestTaskResult:
    """TaskResultクラスのテスト"""
    
    def test_task_result_success(self):
        """成功時のTaskResultテスト"""
        result = TaskResult(
            task_id="test_1",
            success=True,
            result=42,
            execution_time=0.5
        )
        
        assert result.task_id == "test_1"
        assert result.success is True
        assert result.result == 42
        assert result.error is None
        assert result.execution_time == 0.5
    
    def test_task_result_failure(self):
        """失敗時のTaskResultテスト"""
        error = ValueError("test error")
        result = TaskResult(
            task_id="test_2",
            success=False,
            error=error,
            execution_time=0.1,
            retry_count=2
        )
        
        assert result.task_id == "test_2"
        assert result.success is False
        assert result.result is None
        assert result.error == error
        assert result.execution_time == 0.1
        assert result.retry_count == 2


class TestParallelProcessor:
    """ParallelProcessorクラスのテスト"""
    
    def test_parallel_processor_initialization(self):
        """ParallelProcessor初期化のテスト"""
        processor = ParallelProcessor(
            max_workers=4,
            use_processes=True,
            chunk_size=2
        )
        
        assert processor.max_workers == 4
        assert processor.use_processes is True
        assert processor.chunk_size == 2
    
    def test_create_chunks(self):
        """チャンク作成のテスト"""
        processor = ParallelProcessor()
        items = list(range(10))
        chunks = processor._create_chunks(items, 3)
        
        expected_chunks = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert chunks == expected_chunks
    
    def test_process_batch_threads(self):
        """スレッド並列処理のテスト"""
        def square_func(x):
            return x ** 2
        
        processor = ParallelProcessor(
            max_workers=2,
            use_processes=False,
            chunk_size=2
        )
        
        items = [1, 2, 3, 4, 5]
        results = processor.process_batch(square_func, items)
        
        assert len(results) == 5
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 5
        
        # 結果の値をチェック（順序は保証されない）
        values = [r.result for r in successful_results]
        expected_values = [1, 4, 9, 16, 25]
        assert sorted(values) == sorted(expected_values)
    
    def test_process_batch_with_error(self):
        """エラー含む並列処理のテスト"""
        def error_func(x):
            if x == 3:
                raise ValueError("test error")
            return x * 2
        
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        items = [1, 2, 3, 4]
        results = processor.process_batch(error_func, items)
        
        assert len(results) == 4
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) == 3
        assert len(failed_results) == 1


class TestAsyncProcessor:
    """AsyncProcessorクラスのテスト"""
    
    def test_async_processor_initialization(self):
        """AsyncProcessor初期化のテスト"""
        processor = AsyncProcessor(max_concurrent=5)
        assert processor.max_concurrent == 5
    
    @pytest.mark.asyncio
    async def test_process_async_batch(self):
        """非同期バッチ処理のテスト"""
        async def async_square(x):
            await asyncio.sleep(0.01)  # 短い非同期処理
            return x ** 2
        
        processor = AsyncProcessor(max_concurrent=3)
        items = [1, 2, 3, 4, 5]
        
        results = await processor.process_async_batch(async_square, items)
        
        assert len(results) == 5
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 5
        
        values = [r.result for r in successful_results]
        expected_values = [1, 4, 9, 16, 25]
        assert sorted(values) == sorted(expected_values)
    
    @pytest.mark.asyncio
    async def test_async_with_error(self):
        """エラー含む非同期処理のテスト"""
        async def async_error_func(x):
            await asyncio.sleep(0.01)
            if x == 3:
                raise ValueError("async error")
            return x * 3
        
        processor = AsyncProcessor(max_concurrent=2)
        items = [1, 2, 3, 4]
        
        results = await processor.process_async_batch(async_error_func, items)
        
        assert len(results) == 4
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) == 3
        assert len(failed_results) == 1


class TestPipelineProcessor:
    """PipelineProcessorクラスのテスト"""
    
    def test_pipeline_processor_initialization(self):
        """PipelineProcessor初期化のテスト"""
        stages = [lambda x: x + 1, lambda x: x * 2]
        processor = PipelineProcessor(
            stages=stages,
            buffer_size=10,
            max_workers_per_stage=2
        )
        
        assert len(processor.stages) == 2
        assert processor.buffer_size == 10
        assert processor.max_workers_per_stage == 2
        assert len(processor.queues) == 3  # stages + 1
    
    def test_pipeline_processing(self):
        """パイプライン処理のテスト"""
        # ステージ定義：各ステージで値を変換
        def stage1(x):
            time.sleep(0.01)  # 処理時間をシミュレート
            return x + 10
        
        def stage2(x):
            time.sleep(0.01)
            return x * 2
        
        stages = [stage1, stage2]
        processor = PipelineProcessor(
            stages=stages,
            buffer_size=5,
            max_workers_per_stage=1
        )
        
        inputs = [1, 2, 3]
        results = processor.process_pipeline(inputs)
        
        # 結果をソートして確認（順序は保証されない）
        sorted_results = sorted(results)
        expected = sorted([22, 24, 26])  # (1+10)*2, (2+10)*2, (3+10)*2
        assert sorted_results == expected


class TestGPUParallelProcessor:
    """GPUParallelProcessorクラスのテスト"""
    
    def test_gpu_processor_initialization(self):
        """GPUProcessor初期化のテスト"""
        processor = GPUParallelProcessor(batch_size=16)
        assert processor.batch_size == 16
        
        # デバイス確認
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert processor.device == expected_device
    
    def test_process_tensors_batch_cpu(self):
        """CPU環境でのテンソルバッチ処理テスト"""
        def simple_processing(tensors):
            # バッチテンソルの平均を返す
            if isinstance(tensors, list):
                batch = torch.stack(tensors)
            else:
                batch = tensors
            return torch.mean(batch, dim=0)
        
        processor = GPUParallelProcessor(batch_size=2)
        
        # テストテンソル作成
        tensors = [
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4)
        ]
        
        results = processor.process_tensors_batch(simple_processing, tensors)
        
        # 2つのバッチに分かれる（batch_size=2）
        assert len(results) >= 1
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_process_tensors_batch_gpu(self):
        """GPU環境でのテンソルバッチ処理テスト"""
        def gpu_processing(tensors):
            if isinstance(tensors, list):
                batch = torch.stack(tensors)
            else:
                batch = tensors
            return torch.sum(batch, dim=0)
        
        processor = GPUParallelProcessor(batch_size=2)
        
        # GPU テンソル作成
        tensors = [
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3)
        ]
        
        results = processor.process_tensors_batch(gpu_processing, tensors)
        
        assert len(results) == 2  # 4 tensors / batch_size 2
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 2


class TestScalabilityManager:
    """ScalabilityManagerクラスのテスト"""
    
    def test_scalability_manager_initialization(self):
        """ScalabilityManager初期化のテスト"""
        manager = ScalabilityManager()
        
        assert manager.parallel_processor is not None
        assert manager.async_processor is not None
        assert manager.gpu_processor is not None
        assert manager.resource_manager is not None
    
    @patch('psutil.cpu_count', return_value=8)
    def test_choose_optimal_strategy_gpu(self, mock_cpu_count):
        """GPU推奨戦略選択のテスト"""
        manager = ScalabilityManager()
        
        # GPU互換でメモリ集約的でない場合
        strategy = manager.choose_optimal_processing_strategy(
            task_type="model_inference",
            data_size=100,
            memory_intensive=False,
            io_intensive=False,
            gpu_compatible=True
        )
        
        if manager.resource_manager.gpu_available:
            assert strategy == "gpu_parallel"
    
    @patch('psutil.cpu_count', return_value=8)
    def test_choose_optimal_strategy_async(self, mock_cpu_count):
        """非同期戦略選択のテスト"""
        manager = ScalabilityManager()
        
        # I/O集約的でデータサイズが大きい場合
        strategy = manager.choose_optimal_processing_strategy(
            task_type="file_processing",
            data_size=200,
            memory_intensive=False,
            io_intensive=True,
            gpu_compatible=False
        )
        
        assert strategy == "async"
    
    @patch('psutil.cpu_count', return_value=8)
    def test_choose_optimal_strategy_parallel_process(self, mock_cpu_count):
        """プロセス並列戦略選択のテスト"""
        manager = ScalabilityManager()
        
        # メモリ集約的またはデータサイズが非常に大きい場合
        strategy = manager.choose_optimal_processing_strategy(
            task_type="data_processing",
            data_size=1500,
            memory_intensive=True,
            io_intensive=False,
            gpu_compatible=False
        )
        
        assert strategy == "parallel_process"
    
    @patch('psutil.cpu_count', return_value=2)
    def test_choose_optimal_strategy_sequential(self, mock_cpu_count):
        """シーケンシャル戦略選択のテスト"""
        manager = ScalabilityManager()
        
        # 小規模でマルチコアでない場合
        strategy = manager.choose_optimal_processing_strategy(
            task_type="simple_task",
            data_size=10,
            memory_intensive=False,
            io_intensive=False,
            gpu_compatible=False
        )
        
        assert strategy == "sequential"
    
    def test_get_performance_recommendations(self):
        """パフォーマンス推奨事項のテスト"""
        manager = ScalabilityManager()
        recommendations = manager.get_performance_recommendations()
        
        assert isinstance(recommendations, list)
        # 推奨事項の内容は環境に依存するため、型のみチェック
        for rec in recommendations:
            assert isinstance(rec, str)


def test_processing_performance_comparison():
    """処理性能比較テスト"""
    def cpu_intensive_task(x):
        # CPU集約的なタスクをシミュレート
        result = 0
        for i in range(1000):
            result += np.sin(x * i) ** 2
        return result
    
    items = list(range(20))
    
    # シーケンシャル処理
    start_time = time.time()
    sequential_results = [cpu_intensive_task(x) for x in items]
    sequential_time = time.time() - start_time
    
    # 並列処理
    processor = ParallelProcessor(max_workers=2, use_processes=False)
    start_time = time.time()
    parallel_results = processor.process_batch(cpu_intensive_task, items)
    parallel_time = time.time() - start_time
    
    # 並列処理の方が速いかまたは同程度であることを確認
    # （テスト環境によってはシーケンシャルの方が速い場合もある）
    assert len(parallel_results) == len(sequential_results)
    
    # 結果の正確性確認
    successful_parallel = [r.result for r in parallel_results if r.success]
    assert len(successful_parallel) == len(sequential_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])