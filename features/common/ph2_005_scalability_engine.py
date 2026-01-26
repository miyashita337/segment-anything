#!/usr/bin/env python3
"""
PH2-005: スケーラビリティ向上システム
4種類並列処理エンジンの統合実装: ThreadPool, ProcessPool, AsyncIO, GPU並列
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .resource_manager import ResourceManager


class ProcessingEngineType(Enum):
    """処理エンジンタイプ"""

    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_IO = "async_io"
    GPU_PARALLEL = "gpu_parallel"


@dataclass
class ScalabilityConfig:
    """スケーラビリティ設定"""

    thread_pool_workers: int = 4
    process_pool_workers: int = 2
    async_concurrency: int = 10
    gpu_batch_size: int = 32
    auto_optimization: bool = True
    resource_monitoring: bool = True
    performance_logging: bool = True


@dataclass
class ProcessingMetrics:
    """処理メトリクス"""

    engine_type: ProcessingEngineType
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_time: float
    average_time_per_task: float
    throughput: float  # tasks per second
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskBatch:
    """タスクバッチ"""

    id: str
    tasks: List[Any]
    function: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    engine_type: Optional[ProcessingEngineType] = None
    priority: int = 0


class PH2005ScalabilityEngine:
    """PH2-005: 4種類並列処理エンジン統合システム"""

    def __init__(self, config: Optional[ScalabilityConfig] = None):
        """
        初期化

        Args:
            config: スケーラビリティ設定
        """
        self.config = config or ScalabilityConfig()
        self.logger = logging.getLogger(__name__)
        self.resource_manager = ResourceManager()

        # 処理エンジン初期化
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # メトリクス記録
        self.metrics_history: List[ProcessingMetrics] = []
        self.performance_stats: Dict[ProcessingEngineType, List[float]] = {
            engine_type: [] for engine_type in ProcessingEngineType
        }

        self.logger.info("PH2-005 スケーラビリティエンジン初期化完了")
        self._log_system_capabilities()

    def _log_system_capabilities(self):
        """システム性能情報ログ出力"""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_available = torch.cuda.is_available()

        self.logger.info(f"システム性能: CPU {cpu_count}コア, メモリ {memory_gb:.1f}GB")
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"GPU: {gpu_name}, VRAM {gpu_memory:.1f}GB")
        else:
            self.logger.info("GPU: 利用不可")

    def choose_optimal_engine(
        self,
        task_count: int,
        cpu_intensive: bool = False,
        io_intensive: bool = False,
        memory_intensive: bool = False,
        gpu_compatible: bool = False,
    ) -> ProcessingEngineType:
        """
        最適な処理エンジンを自動選択

        Args:
            task_count: タスク数
            cpu_intensive: CPU集約的か
            io_intensive: I/O集約的か
            memory_intensive: メモリ集約的か
            gpu_compatible: GPU対応か

        Returns:
            最適な処理エンジンタイプ
        """
        resource_usage = self.resource_manager.get_current_usage()
        cpu_cores = mp.cpu_count()

        # GPU最優先（対応タスクかつGPU利用可能）
        if gpu_compatible and torch.cuda.is_available():
            self.logger.info(f"GPU並列エンジン選択: {task_count}タスク, GPU対応")
            return ProcessingEngineType.GPU_PARALLEL

        # プロセス並列（CPU集約的または大容量メモリ使用）
        if cpu_intensive or memory_intensive:
            if task_count >= cpu_cores and resource_usage.memory_percent < 70:
                self.logger.info(f"プロセス並列エンジン選択: {task_count}タスク, CPU集約的")
                return ProcessingEngineType.PROCESS_POOL

        # I/O集約的タスクは非同期処理
        if io_intensive:
            self.logger.info(f"非同期エンジン選択: {task_count}タスク, I/O集約的")
            return ProcessingEngineType.ASYNC_IO

        # その他はスレッド並列（軽量タスク）
        if task_count > 1:
            self.logger.info(f"スレッド並列エンジン選択: {task_count}タスク, 軽量処理")
            return ProcessingEngineType.THREAD_POOL

        # フォールバック: スレッド並列
        return ProcessingEngineType.THREAD_POOL

    def process_with_thread_pool(
        self, function: Callable, tasks: List[Any], *args, **kwargs
    ) -> Tuple[List[Any], ProcessingMetrics]:
        """ThreadPoolExecutor による並列処理"""
        start_time = time.time()
        results = []
        successful_tasks = 0
        failed_tasks = 0

        with ThreadPoolExecutor(max_workers=self.config.thread_pool_workers) as executor:
            self.thread_executor = executor

            # タスクを並列実行
            future_to_task = {
                executor.submit(function, task, *args, **kwargs): task for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    successful_tasks += 1
                except Exception as e:
                    self.logger.error(f"スレッド並列処理エラー: {e}")
                    results.append(None)
                    failed_tasks += 1

        total_time = time.time() - start_time
        metrics = self._create_metrics(
            ProcessingEngineType.THREAD_POOL, len(tasks), successful_tasks, failed_tasks, total_time
        )

        self.logger.info(f"スレッド並列処理完了: {successful_tasks}/{len(tasks)} 成功, {total_time:.2f}秒")
        return results, metrics

    def process_with_process_pool(
        self, function: Callable, tasks: List[Any], *args, **kwargs
    ) -> Tuple[List[Any], ProcessingMetrics]:
        """ProcessPoolExecutor による並列処理"""
        start_time = time.time()
        results = []
        successful_tasks = 0
        failed_tasks = 0

        with ProcessPoolExecutor(max_workers=self.config.process_pool_workers) as executor:
            self.process_executor = executor

            # タスクを並列実行
            future_to_task = {
                executor.submit(function, task, *args, **kwargs): task for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    successful_tasks += 1
                except Exception as e:
                    self.logger.error(f"プロセス並列処理エラー: {e}")
                    results.append(None)
                    failed_tasks += 1

        total_time = time.time() - start_time
        metrics = self._create_metrics(
            ProcessingEngineType.PROCESS_POOL,
            len(tasks),
            successful_tasks,
            failed_tasks,
            total_time,
        )

        self.logger.info(f"プロセス並列処理完了: {successful_tasks}/{len(tasks)} 成功, {total_time:.2f}秒")
        return results, metrics

    async def process_with_async_io(
        self, async_function: Callable, tasks: List[Any], *args, **kwargs
    ) -> Tuple[List[Any], ProcessingMetrics]:
        """AsyncIO による非同期処理"""
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.config.async_concurrency)

        async def process_task(task):
            async with semaphore:
                try:
                    return await async_function(task, *args, **kwargs)
                except Exception as e:
                    self.logger.error(f"非同期処理エラー: {e}")
                    return None

        # 全タスクを非同期実行
        results = await asyncio.gather(*[process_task(task) for task in tasks])

        successful_tasks = sum(1 for r in results if r is not None)
        failed_tasks = len(results) - successful_tasks

        total_time = time.time() - start_time
        metrics = self._create_metrics(
            ProcessingEngineType.ASYNC_IO, len(tasks), successful_tasks, failed_tasks, total_time
        )

        self.logger.info(f"非同期処理完了: {successful_tasks}/{len(tasks)} 成功, {total_time:.2f}秒")
        return results, metrics

    def process_with_gpu_parallel(
        self, function: Callable, tensors: List[torch.Tensor], *args, **kwargs
    ) -> Tuple[List[Any], ProcessingMetrics]:
        """GPU並列処理"""
        if not torch.cuda.is_available():
            self.logger.warning("GPU利用不可、CPU処理にフォールバック")
            return self.process_with_thread_pool(function, tensors, *args, **kwargs)

        start_time = time.time()
        results = []
        successful_tasks = 0
        failed_tasks = 0
        batch_size = self.config.gpu_batch_size

        # バッチごとに処理
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i : i + batch_size]

            try:
                # テンソルをGPUに転送
                gpu_batch = [tensor.to(self.gpu_device) for tensor in batch]

                # GPU処理実行
                batch_result = function(gpu_batch, *args, **kwargs)

                # 結果をCPUに戻す
                if isinstance(batch_result, torch.Tensor):
                    batch_result = batch_result.cpu()
                elif isinstance(batch_result, list):
                    batch_result = [
                        r.cpu() if isinstance(r, torch.Tensor) else r for r in batch_result
                    ]

                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                successful_tasks += len(batch)

                # GPU メモリクリア
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"GPU並列処理エラー: {e}")
                results.extend([None] * len(batch))
                failed_tasks += len(batch)

        total_time = time.time() - start_time
        metrics = self._create_metrics(
            ProcessingEngineType.GPU_PARALLEL,
            len(tensors),
            successful_tasks,
            failed_tasks,
            total_time,
        )

        self.logger.info(f"GPU並列処理完了: {successful_tasks}/{len(tensors)} 成功, {total_time:.2f}秒")
        return results, metrics

    def process_batch_auto(self, task_batch: TaskBatch) -> Tuple[List[Any], ProcessingMetrics]:
        """
        自動エンジン選択によるバッチ処理

        Args:
            task_batch: 処理タスクバッチ

        Returns:
            処理結果とメトリクス
        """
        # エンジン自動選択
        if task_batch.engine_type is None:
            task_batch.engine_type = self.choose_optimal_engine(len(task_batch.tasks))

        self.logger.info(f"バッチ処理開始: {task_batch.id}, エンジン: {task_batch.engine_type.value}")

        # エンジン別処理実行
        if task_batch.engine_type == ProcessingEngineType.THREAD_POOL:
            return self.process_with_thread_pool(
                task_batch.function, task_batch.tasks, *task_batch.args, **task_batch.kwargs
            )
        elif task_batch.engine_type == ProcessingEngineType.PROCESS_POOL:
            return self.process_with_process_pool(
                task_batch.function, task_batch.tasks, *task_batch.args, **task_batch.kwargs
            )
        elif task_batch.engine_type == ProcessingEngineType.ASYNC_IO:
            # 非同期処理は別途 run_async_batch() を使用
            raise ValueError("非同期処理は run_async_batch() を使用してください")
        elif task_batch.engine_type == ProcessingEngineType.GPU_PARALLEL:
            return self.process_with_gpu_parallel(
                task_batch.function, task_batch.tasks, *task_batch.args, **task_batch.kwargs
            )
        else:
            raise ValueError(f"未対応のエンジンタイプ: {task_batch.engine_type}")

    async def run_async_batch(self, task_batch: TaskBatch) -> Tuple[List[Any], ProcessingMetrics]:
        """非同期バッチ処理実行"""
        return await self.process_with_async_io(
            task_batch.function, task_batch.tasks, *task_batch.args, **task_batch.kwargs
        )

    def benchmark_engines(
        self, test_function: Callable, test_data: List[Any], iterations: int = 3
    ) -> Dict[ProcessingEngineType, ProcessingMetrics]:
        """
        全エンジンのベンチマーク実行

        Args:
            test_function: テスト関数
            test_data: テストデータ
            iterations: 反復回数

        Returns:
            エンジン別平均メトリクス
        """
        benchmark_results = {}

        # 各エンジンでベンチマーク実行
        for engine_type in ProcessingEngineType:
            if engine_type == ProcessingEngineType.ASYNC_IO:
                continue  # 非同期は別途テスト

            engine_metrics = []

            for i in range(iterations):
                self.logger.info(f"ベンチマーク実行: {engine_type.value} 反復{i+1}/{iterations}")

                try:
                    task_batch = TaskBatch(
                        id=f"benchmark_{engine_type.value}_{i}",
                        tasks=test_data.copy(),
                        function=test_function,
                        engine_type=engine_type,
                    )

                    if engine_type == ProcessingEngineType.GPU_PARALLEL:
                        # GPU処理はテンソルが必要
                        if not all(isinstance(t, torch.Tensor) for t in test_data):
                            continue

                    _, metrics = self.process_batch_auto(task_batch)
                    engine_metrics.append(metrics)

                except Exception as e:
                    self.logger.error(f"ベンチマーク失敗 {engine_type.value}: {e}")
                    continue

            if engine_metrics:
                # 平均メトリクス計算
                avg_metrics = self._calculate_average_metrics(engine_metrics)
                benchmark_results[engine_type] = avg_metrics

        self.logger.info(f"ベンチマーク完了: {len(benchmark_results)}エンジン")
        return benchmark_results

    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        if not self.metrics_history:
            return {"error": "メトリクス履歴がありません"}

        # エンジン別統計
        engine_stats = {}
        for engine_type in ProcessingEngineType:
            engine_metrics = [m for m in self.metrics_history if m.engine_type == engine_type]
            if engine_metrics:
                avg_throughput = sum(m.throughput for m in engine_metrics) / len(engine_metrics)
                avg_success_rate = sum(
                    m.successful_tasks / m.total_tasks for m in engine_metrics
                ) / len(engine_metrics)

                engine_stats[engine_type.value] = {
                    "処理回数": len(engine_metrics),
                    "平均スループット": f"{avg_throughput:.2f} tasks/sec",
                    "平均成功率": f"{avg_success_rate:.1%}",
                    "総処理タスク数": sum(m.total_tasks for m in engine_metrics),
                }

        # システム推奨事項
        recommendations = self._generate_recommendations()

        return {
            "timestamp": datetime.now().isoformat(),
            "total_batches_processed": len(self.metrics_history),
            "engine_statistics": engine_stats,
            "system_recommendations": recommendations,
            "resource_usage": self.resource_manager.get_current_usage().__dict__,
        }

    def _create_metrics(
        self,
        engine_type: ProcessingEngineType,
        total_tasks: int,
        successful_tasks: int,
        failed_tasks: int,
        total_time: float,
    ) -> ProcessingMetrics:
        """メトリクス作成"""
        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        throughput = total_tasks / total_time if total_time > 0 else 0

        metrics = ProcessingMetrics(
            engine_type=engine_type,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            total_time=total_time,
            average_time_per_task=avg_time,
            throughput=throughput,
            resource_usage=self.resource_manager.get_current_usage().__dict__,
        )

        self.metrics_history.append(metrics)
        return metrics

    def _calculate_average_metrics(
        self, metrics_list: List[ProcessingMetrics]
    ) -> ProcessingMetrics:
        """メトリクスリストの平均計算"""
        if not metrics_list:
            raise ValueError("メトリクスリストが空です")

        avg_metrics = ProcessingMetrics(
            engine_type=metrics_list[0].engine_type,
            total_tasks=sum(m.total_tasks for m in metrics_list) // len(metrics_list),
            successful_tasks=sum(m.successful_tasks for m in metrics_list) // len(metrics_list),
            failed_tasks=sum(m.failed_tasks for m in metrics_list) // len(metrics_list),
            total_time=sum(m.total_time for m in metrics_list) / len(metrics_list),
            average_time_per_task=sum(m.average_time_per_task for m in metrics_list)
            / len(metrics_list),
            throughput=sum(m.throughput for m in metrics_list) / len(metrics_list),
        )

        return avg_metrics

    def _generate_recommendations(self) -> List[str]:
        """性能改善推奨事項生成"""
        recommendations = []

        if not self.metrics_history:
            return ["データ不足のため推奨事項を生成できません"]

        # エンジン別性能分析
        engine_performance = {}
        for engine_type in ProcessingEngineType:
            engine_metrics = [m for m in self.metrics_history if m.engine_type == engine_type]
            if engine_metrics:
                avg_throughput = sum(m.throughput for m in engine_metrics) / len(engine_metrics)
                engine_performance[engine_type] = avg_throughput

        if engine_performance:
            best_engine = max(engine_performance, key=engine_performance.get)
            recommendations.append(
                f"最高性能エンジン: {best_engine.value} ({engine_performance[best_engine]:.2f} tasks/sec)"
            )

        # リソース使用率チェック
        current_usage = self.resource_manager.get_current_usage()
        if current_usage.cpu_percent < 50:
            recommendations.append("CPU使用率が低い - より多くの並列処理を検討")
        if current_usage.memory_percent > 80:
            recommendations.append("メモリ使用率が高い - バッチサイズ縮小を検討")

        # GPU活用推奨
        if torch.cuda.is_available() and current_usage.gpu_memory_mb < 2000:
            recommendations.append("GPU活用度が低い - GPU並列処理の増加を検討")

        return recommendations

    def cleanup(self):
        """リソースクリーンアップ"""
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("PH2-005 スケーラビリティエンジン クリーンアップ完了")


# グローバルスケーラビリティエンジン
global_scalability_engine = PH2005ScalabilityEngine()


def create_scalability_engine(
    config: Optional[ScalabilityConfig] = None,
) -> PH2005ScalabilityEngine:
    """スケーラビリティエンジン作成"""
    return PH2005ScalabilityEngine(config)


# 便利関数
def auto_process_batch(
    function: Callable, tasks: List[Any], *args, **kwargs
) -> Tuple[List[Any], ProcessingMetrics]:
    """自動エンジン選択バッチ処理"""
    engine = global_scalability_engine

    task_batch = TaskBatch(
        id=f"auto_batch_{int(time.time())}",
        tasks=tasks,
        function=function,
        args=args,
        kwargs=kwargs,
    )

    return engine.process_batch_auto(task_batch)


async def auto_process_async_batch(
    async_function: Callable, tasks: List[Any], *args, **kwargs
) -> Tuple[List[Any], ProcessingMetrics]:
    """自動非同期バッチ処理"""
    engine = global_scalability_engine

    task_batch = TaskBatch(
        id=f"async_batch_{int(time.time())}",
        tasks=tasks,
        function=async_function,
        args=args,
        kwargs=kwargs,
        engine_type=ProcessingEngineType.ASYNC_IO,
    )

    return await engine.run_async_batch(task_batch)
