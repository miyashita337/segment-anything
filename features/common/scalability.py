#!/usr/bin/env python3
"""
スケーラビリティ改善システム
PH2-002: 並列処理・分散処理・非同期処理による性能向上
"""

import torch

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import psutil
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .error_handling import (
    BaseCustomError,
    ProcessingError,
    global_error_handler,
    with_error_handling,
)
from .resource_manager import ResourceManager


@dataclass
class ProcessingTask:
    """処理タスク"""

    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """タスク結果"""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_count: int = 0


class ParallelProcessor:
    """並列処理エンジン"""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            max_workers: 最大ワーカー数（Noneの場合は自動計算）
            use_processes: プロセス使用（True）かスレッド使用（False）
            chunk_size: チャンクサイズ
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_processes = use_processes
        self.chunk_size = chunk_size or max(1, self.max_workers // 2)

        self.logger = logging.getLogger(__name__)
        self.resource_manager = ResourceManager()

        self.logger.info(
            f"Parallel processor initialized: {self.max_workers} workers, "
            f"{'processes' if use_processes else 'threads'}"
        )

    def process_batch(self, func: Callable, items: List[Any], *args, **kwargs) -> List[TaskResult]:
        """バッチ並列処理"""
        start_time = time.time()

        # アイテムをチャンクに分割
        chunks = self._create_chunks(items, self.chunk_size)

        self.logger.info(
            f"Processing {len(items)} items in {len(chunks)} chunks "
            f"with {self.max_workers} workers"
        )

        results = []

        if self.use_processes:
            # プロセス並列
            with mp.Pool(processes=self.max_workers) as pool:
                chunk_results = pool.starmap(
                    self._process_chunk, [(func, chunk, args, kwargs) for chunk in chunks]
                )

                # 結果を統合
                for chunk_result in chunk_results:
                    results.extend(chunk_result)
        else:
            # スレッド並列
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk, func, chunk, args, kwargs): chunk
                    for chunk in chunks
                }

                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_result = future.result()
                        results.extend(chunk_result)
                    except Exception as e:
                        self.logger.error(f"Chunk processing failed: {e}")

        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)

        self.logger.info(
            f"Batch processing completed: {success_count}/{len(items)} successful "
            f"in {total_time:.2f}s"
        )

        return results

    def _create_chunks(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """アイテムをチャンクに分割"""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i : i + chunk_size])
        return chunks

    def _process_chunk(
        self, func: Callable, chunk: List[Any], args: tuple, kwargs: dict
    ) -> List[TaskResult]:
        """チャンク処理"""
        results = []

        for item in chunk:
            start_time = time.time()
            task_id = f"item_{id(item)}"

            try:
                result = func(item, *args, **kwargs)
                execution_time = time.time() - start_time

                results.append(
                    TaskResult(
                        task_id=task_id, success=True, result=result, execution_time=execution_time
                    )
                )

            except Exception as e:
                execution_time = time.time() - start_time

                results.append(
                    TaskResult(
                        task_id=task_id, success=False, error=e, execution_time=execution_time
                    )
                )

        return results


class AsyncProcessor:
    """非同期処理エンジン"""

    def __init__(self, max_concurrent: int = 10):
        """
        Args:
            max_concurrent: 最大同時実行数
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)

    async def process_async_batch(
        self, async_func: Callable, items: List[Any], *args, **kwargs
    ) -> List[TaskResult]:
        """非同期バッチ処理"""
        self.logger.info(f"Processing {len(items)} items asynchronously")

        # 非同期タスクを作成
        tasks = [
            self._process_item_async(async_func, item, args, kwargs, f"item_{i}")
            for i, item in enumerate(items)
        ]

        # 全タスクを並列実行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果を変換
        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_results.append(TaskResult(task_id=f"item_{i}", success=False, error=result))
            else:
                task_results.append(result)

        success_count = sum(1 for r in task_results if r.success)
        self.logger.info(f"Async processing completed: {success_count}/{len(items)} successful")

        return task_results

    async def _process_item_async(
        self, async_func: Callable, item: Any, args: tuple, kwargs: dict, task_id: str
    ) -> TaskResult:
        """アイテム非同期処理"""
        async with self.semaphore:
            start_time = time.time()

            try:
                result = await async_func(item, *args, **kwargs)
                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task_id, success=True, result=result, execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task_id, success=False, error=e, execution_time=execution_time
                )


class PipelineProcessor:
    """パイプライン処理エンジン"""

    def __init__(
        self, stages: List[Callable], buffer_size: int = 100, max_workers_per_stage: int = 2
    ):
        """
        Args:
            stages: 処理ステージのリスト
            buffer_size: ステージ間のバッファサイズ
            max_workers_per_stage: ステージあたりの最大ワーカー数
        """
        self.stages = stages
        self.buffer_size = buffer_size
        self.max_workers_per_stage = max_workers_per_stage

        self.logger = logging.getLogger(__name__)

        # ステージ間のキュー
        self.queues = [queue.Queue(maxsize=buffer_size) for _ in range(len(stages) + 1)]

        self.logger.info(f"Pipeline processor initialized with {len(stages)} stages")

    def process_pipeline(self, inputs: List[Any]) -> List[TaskResult]:
        """パイプライン処理"""
        self.logger.info(
            f"Processing {len(inputs)} items through {len(self.stages)} stage pipeline"
        )

        # 入力キューにアイテムを追加
        for item in inputs:
            self.queues[0].put(item)

        # 終了シグナル
        for _ in range(self.max_workers_per_stage):
            self.queues[0].put(None)

        # 各ステージのワーカーを開始
        threads = []
        for stage_idx, stage_func in enumerate(self.stages):
            for worker_id in range(self.max_workers_per_stage):
                thread = threading.Thread(
                    target=self._stage_worker, args=(stage_idx, stage_func, worker_id)
                )
                thread.start()
                threads.append(thread)

        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()

        # 結果を収集
        results = []
        while not self.queues[-1].empty():
            try:
                result = self.queues[-1].get_nowait()
                if result is not None:
                    results.append(result)
            except queue.Empty:
                break

        self.logger.info(f"Pipeline processing completed: {len(results)} results")
        return results

    def _stage_worker(self, stage_idx: int, stage_func: Callable, worker_id: int):
        """ステージワーカー"""
        input_queue = self.queues[stage_idx]
        output_queue = self.queues[stage_idx + 1]

        while True:
            try:
                item = input_queue.get(timeout=1.0)
                if item is None:
                    # 終了シグナル
                    output_queue.put(None)
                    break

                # 処理実行
                start_time = time.time()
                try:
                    result = stage_func(item)
                    execution_time = time.time() - start_time

                    task_result = TaskResult(
                        task_id=f"stage_{stage_idx}_worker_{worker_id}",
                        success=True,
                        result=result,
                        execution_time=execution_time,
                    )

                    output_queue.put(task_result.result)

                except Exception as e:
                    execution_time = time.time() - start_time

                    task_result = TaskResult(
                        task_id=f"stage_{stage_idx}_worker_{worker_id}",
                        success=False,
                        error=e,
                        execution_time=execution_time,
                    )

                    # エラーは次のステージに渡さない
                    self.logger.error(f"Stage {stage_idx} error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Stage worker error: {e}")
                break


class GPUParallelProcessor:
    """GPU並列処理エンジン"""

    def __init__(self, batch_size: int = 32):
        """
        Args:
            batch_size: GPUバッチサイズ
        """
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.logger.info(f"GPU processor initialized: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.warning("GPU not available, using CPU")

    def process_tensors_batch(
        self, func: Callable, tensors: List[torch.Tensor], *args, **kwargs
    ) -> List[TaskResult]:
        """テンソルバッチ処理"""
        if not tensors:
            return []

        self.logger.info(f"Processing {len(tensors)} tensors on {self.device}")

        results = []

        # バッチごとに処理
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i : i + self.batch_size]

            start_time = time.time()

            try:
                # テンソルをGPUに転送
                batch_tensors = [t.to(self.device) for t in batch]

                # バッチ処理実行
                batch_result = func(batch_tensors, *args, **kwargs)

                execution_time = time.time() - start_time

                # 結果をCPUに戻す
                if isinstance(batch_result, torch.Tensor):
                    batch_result = batch_result.cpu()
                elif isinstance(batch_result, list):
                    batch_result = [
                        r.cpu() if isinstance(r, torch.Tensor) else r for r in batch_result
                    ]

                results.append(
                    TaskResult(
                        task_id=f"batch_{i//self.batch_size}",
                        success=True,
                        result=batch_result,
                        execution_time=execution_time,
                    )
                )

            except Exception as e:
                execution_time = time.time() - start_time

                results.append(
                    TaskResult(
                        task_id=f"batch_{i//self.batch_size}",
                        success=False,
                        error=e,
                        execution_time=execution_time,
                    )
                )

                self.logger.error(f"GPU batch processing failed: {e}")

            # GPU メモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            f"GPU processing completed: {success_count}/{len(results)} batches successful"
        )

        return results


class ScalabilityManager:
    """スケーラビリティ管理クラス"""

    def __init__(self):
        self.parallel_processor = ParallelProcessor()
        self.async_processor = AsyncProcessor()
        self.gpu_processor = GPUParallelProcessor()
        self.resource_manager = ResourceManager()

        self.logger = logging.getLogger(__name__)

    def choose_optimal_processing_strategy(
        self,
        task_type: str,
        data_size: int,
        memory_intensive: bool = False,
        io_intensive: bool = False,
        gpu_compatible: bool = False,
    ) -> str:
        """最適な処理戦略を選択"""

        # リソース状況確認
        usage = self.resource_manager.get_current_usage()
        cpu_cores = psutil.cpu_count()

        # 戦略決定ロジック
        if gpu_compatible and self.resource_manager.gpu_available and not memory_intensive:
            strategy = "gpu_parallel"
        elif io_intensive and data_size > 100:
            strategy = "async"
        elif memory_intensive or data_size > 1000:
            strategy = "parallel_process"
        elif cpu_cores > 4 and data_size > 50:
            strategy = "parallel_thread"
        else:
            strategy = "sequential"

        self.logger.info(
            f"Optimal strategy for {task_type}: {strategy} "
            f"(data_size={data_size}, memory_intensive={memory_intensive}, "
            f"io_intensive={io_intensive}, gpu_compatible={gpu_compatible})"
        )

        return strategy

    def get_performance_recommendations(self) -> List[str]:
        """パフォーマンス改善推奨事項"""
        recommendations = []

        usage = self.resource_manager.get_current_usage()
        cpu_cores = psutil.cpu_count()

        # CPU使用率チェック
        if usage.cpu_percent < 30:
            recommendations.append(f"CPU使用率が低い（{usage.cpu_percent:.1f}%）- 並列処理の活用を検討")

        # メモリ使用率チェック
        if usage.memory_percent > 80:
            recommendations.append(f"メモリ使用率が高い（{usage.memory_percent:.1f}%）- バッチサイズの縮小を検討")

        # GPU活用チェック
        if self.resource_manager.gpu_available and usage.gpu_memory_mb < 1000:
            recommendations.append("GPU メモリ使用量が少ない - GPU並列処理の活用を検討")

        # マルチコア活用チェック
        if cpu_cores > 4:
            recommendations.append(f"{cpu_cores}コアCPU活用のため並列処理を推奨")

        return recommendations


# グローバルスケーラビリティマネージャー
global_scalability_manager = ScalabilityManager()
