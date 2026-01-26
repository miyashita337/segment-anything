#!/usr/bin/env python3
"""
PH2-005 スケーラビリティエンジン統合テスト
4種類並列処理エンジンの性能評価とベンチマーク
"""

import torch

import asyncio
import json
import logging
import math

# プロジェクトルート追加
import sys
import time
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))

from features.common.ph2_005_scalability_engine import (
    PH2005ScalabilityEngine,
    ProcessingEngineType,
    ScalabilityConfig,
    TaskBatch,
    auto_process_batch,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cpu_intensive_task(x: int) -> int:
    """CPU集約的テストタスク"""
    result = 0
    for i in range(x * 1000):
        result += math.sqrt(i + 1)
    return int(result)


def io_intensive_task(delay: float) -> str:
    """I/O集約的テストタスク（スリープで模擬）"""
    time.sleep(delay)
    return f"processed_after_{delay}s"


async def async_io_task(delay: float) -> str:
    """非同期I/Oテストタスク"""
    await asyncio.sleep(delay)
    return f"async_processed_after_{delay}s"


def gpu_tensor_task(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """GPU並列テストタスク"""
    results = []
    for tensor in tensors:
        # 簡単な行列演算
        result = torch.matmul(tensor, tensor.T)
        result = torch.relu(result)
        results.append(result)
    return results


class PH2005ScalabilityTester:
    """PH2-005 スケーラビリティテスター"""

    def __init__(self):
        """初期化"""
        self.config = ScalabilityConfig(
            thread_pool_workers=4,
            process_pool_workers=2,
            async_concurrency=10,
            gpu_batch_size=16,
            auto_optimization=True,
            performance_logging=True,
        )

        self.engine = PH2005ScalabilityEngine(self.config)
        self.test_results = {}

        # 出力ディレクトリ作成
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-005")
        self.tests_dir = self.output_dir / "tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PH2-005 スケーラビリティテスター初期化完了")

    def test_thread_pool_engine(self) -> dict:
        """スレッド並列エンジンテスト"""
        logger.info("🔧 スレッド並列エンジンテスト開始")

        # 軽量CPUタスクを大量処理
        test_data = list(range(1, 21))  # 20タスク

        start_time = time.time()
        results, metrics = self.engine.process_with_thread_pool(cpu_intensive_task, test_data)
        end_time = time.time()

        test_result = {
            "engine": "ThreadPoolExecutor",
            "test_type": "lightweight_cpu_tasks",
            "task_count": len(test_data),
            "successful_tasks": metrics.successful_tasks,
            "failed_tasks": metrics.failed_tasks,
            "total_time": metrics.total_time,
            "throughput": metrics.throughput,
            "success_rate": metrics.successful_tasks / metrics.total_tasks,
            "results_sample": results[:5] if results else [],
        }

        logger.info(f"スレッド並列テスト完了: {metrics.successful_tasks}/{metrics.total_tasks} 成功")
        return test_result

    def test_process_pool_engine(self) -> dict:
        """プロセス並列エンジンテスト"""
        logger.info("🔧 プロセス並列エンジンテスト開始")

        # CPU集約的タスク
        test_data = list(range(10, 30))  # 20タスク (より重い処理)

        start_time = time.time()
        results, metrics = self.engine.process_with_process_pool(cpu_intensive_task, test_data)
        end_time = time.time()

        test_result = {
            "engine": "ProcessPoolExecutor",
            "test_type": "cpu_intensive_tasks",
            "task_count": len(test_data),
            "successful_tasks": metrics.successful_tasks,
            "failed_tasks": metrics.failed_tasks,
            "total_time": metrics.total_time,
            "throughput": metrics.throughput,
            "success_rate": metrics.successful_tasks / metrics.total_tasks,
            "results_sample": results[:5] if results else [],
        }

        logger.info(f"プロセス並列テスト完了: {metrics.successful_tasks}/{metrics.total_tasks} 成功")
        return test_result

    async def test_async_io_engine(self) -> dict:
        """非同期I/Oエンジンテスト"""
        logger.info("🔧 非同期I/Oエンジンテスト開始")

        # I/O待機時間シミュレーション
        test_data = [0.1, 0.2, 0.1, 0.3, 0.1] * 4  # 20タスク

        start_time = time.time()
        results, metrics = await self.engine.process_with_async_io(async_io_task, test_data)
        end_time = time.time()

        test_result = {
            "engine": "AsyncIO",
            "test_type": "io_intensive_tasks",
            "task_count": len(test_data),
            "successful_tasks": metrics.successful_tasks,
            "failed_tasks": metrics.failed_tasks,
            "total_time": metrics.total_time,
            "throughput": metrics.throughput,
            "success_rate": metrics.successful_tasks / metrics.total_tasks,
            "results_sample": results[:5] if results else [],
        }

        logger.info(f"非同期I/Oテスト完了: {metrics.successful_tasks}/{metrics.total_tasks} 成功")
        return test_result

    def test_gpu_parallel_engine(self) -> dict:
        """GPU並列エンジンテスト"""
        logger.info("🔧 GPU並列エンジンテスト開始")

        if not torch.cuda.is_available():
            logger.warning("GPU利用不可、テストスキップ")
            return {
                "engine": "GPU_Parallel",
                "test_type": "gpu_tensor_operations",
                "status": "skipped",
                "reason": "GPU not available",
            }

        # テンソルデータ作成
        test_tensors = [torch.randn(32, 32) for _ in range(10)]

        start_time = time.time()
        results, metrics = self.engine.process_with_gpu_parallel(gpu_tensor_task, test_tensors)
        end_time = time.time()

        test_result = {
            "engine": "GPU_Parallel",
            "test_type": "gpu_tensor_operations",
            "task_count": len(test_tensors),
            "successful_tasks": metrics.successful_tasks,
            "failed_tasks": metrics.failed_tasks,
            "total_time": metrics.total_time,
            "throughput": metrics.throughput,
            "success_rate": metrics.successful_tasks / metrics.total_tasks,
            "gpu_info": {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0),
            },
        }

        logger.info(f"GPU並列テスト完了: {metrics.successful_tasks}/{metrics.total_tasks} 成功")
        return test_result

    def test_auto_engine_selection(self) -> dict:
        """自動エンジン選択テスト"""
        logger.info("🔧 自動エンジン選択テスト開始")

        selection_tests = []

        # テストケース1: 少数軽量タスク
        engine_type = self.engine.choose_optimal_engine(
            task_count=5, cpu_intensive=False, io_intensive=False, gpu_compatible=False
        )
        selection_tests.append(
            {
                "case": "few_lightweight_tasks",
                "parameters": {
                    "task_count": 5,
                    "cpu_intensive": False,
                    "io_intensive": False,
                    "gpu_compatible": False,
                },
                "selected_engine": engine_type.value,
            }
        )

        # テストケース2: CPU集約的タスク
        engine_type = self.engine.choose_optimal_engine(
            task_count=50, cpu_intensive=True, memory_intensive=True
        )
        selection_tests.append(
            {
                "case": "cpu_intensive_tasks",
                "parameters": {"task_count": 50, "cpu_intensive": True, "memory_intensive": True},
                "selected_engine": engine_type.value,
            }
        )

        # テストケース3: I/O集約的タスク
        engine_type = self.engine.choose_optimal_engine(task_count=100, io_intensive=True)
        selection_tests.append(
            {
                "case": "io_intensive_tasks",
                "parameters": {"task_count": 100, "io_intensive": True},
                "selected_engine": engine_type.value,
            }
        )

        # テストケース4: GPU対応タスク
        engine_type = self.engine.choose_optimal_engine(task_count=20, gpu_compatible=True)
        selection_tests.append(
            {
                "case": "gpu_compatible_tasks",
                "parameters": {"task_count": 20, "gpu_compatible": True},
                "selected_engine": engine_type.value,
            }
        )

        test_result = {
            "test_type": "auto_engine_selection",
            "selection_cases": selection_tests,
            "total_cases": len(selection_tests),
        }

        logger.info(f"自動エンジン選択テスト完了: {len(selection_tests)}ケース")
        return test_result

    def run_benchmark_comparison(self) -> dict:
        """エンジンベンチマーク比較"""
        logger.info("🏁 エンジンベンチマーク比較開始")

        # 共通テストデータ
        benchmark_data = list(range(1, 16))  # 15タスク

        try:
            benchmark_results = self.engine.benchmark_engines(
                cpu_intensive_task, benchmark_data, iterations=2
            )

            # 結果整理
            comparison_results = {}
            for engine_type, metrics in benchmark_results.items():
                comparison_results[engine_type.value] = {
                    "throughput": metrics.throughput,
                    "success_rate": metrics.successful_tasks / metrics.total_tasks,
                    "average_time_per_task": metrics.average_time_per_task,
                    "total_time": metrics.total_time,
                }

            # 最適エンジン特定
            if comparison_results:
                best_engine = max(
                    comparison_results.keys(), key=lambda k: comparison_results[k]["throughput"]
                )

                return {
                    "test_type": "benchmark_comparison",
                    "engine_results": comparison_results,
                    "best_engine": best_engine,
                    "best_throughput": comparison_results[best_engine]["throughput"],
                }
            else:
                return {
                    "test_type": "benchmark_comparison",
                    "status": "failed",
                    "reason": "No successful benchmark results",
                }

        except Exception as e:
            logger.error(f"ベンチマーク実行エラー: {e}")
            return {"test_type": "benchmark_comparison", "status": "error", "error": str(e)}

    async def run_all_tests(self):
        """全テスト実行"""
        logger.info("🚀 PH2-005 全テスト実行開始")

        self.test_results = {
            "timestamp": time.time(),
            "test_summary": {"total_tests": 6, "completed_tests": 0, "failed_tests": 0},
            "engine_tests": {},
            "integration_tests": {},
        }

        try:
            # 1. スレッド並列テスト
            self.test_results["engine_tests"]["thread_pool"] = self.test_thread_pool_engine()
            self.test_results["test_summary"]["completed_tests"] += 1

            # 2. プロセス並列テスト
            self.test_results["engine_tests"]["process_pool"] = self.test_process_pool_engine()
            self.test_results["test_summary"]["completed_tests"] += 1

            # 3. 非同期I/Oテスト
            self.test_results["engine_tests"]["async_io"] = await self.test_async_io_engine()
            self.test_results["test_summary"]["completed_tests"] += 1

            # 4. GPU並列テスト
            self.test_results["engine_tests"]["gpu_parallel"] = self.test_gpu_parallel_engine()
            self.test_results["test_summary"]["completed_tests"] += 1

            # 5. 自動エンジン選択テスト
            self.test_results["integration_tests"][
                "auto_selection"
            ] = self.test_auto_engine_selection()
            self.test_results["test_summary"]["completed_tests"] += 1

            # 6. ベンチマーク比較
            self.test_results["integration_tests"]["benchmark"] = self.run_benchmark_comparison()
            self.test_results["test_summary"]["completed_tests"] += 1

            # パフォーマンスレポート生成
            self.test_results["performance_report"] = self.engine.get_performance_report()

            # 成功率計算
            success_rate = (
                self.test_results["test_summary"]["completed_tests"]
                - self.test_results["test_summary"]["failed_tests"]
            ) / self.test_results["test_summary"]["total_tests"]

            self.test_results["test_summary"]["success_rate"] = success_rate

            logger.info(
                f"✅ 全テスト完了: {self.test_results['test_summary']['completed_tests']}/{self.test_results['test_summary']['total_tests']}"
            )

        except Exception as e:
            logger.error(f"テスト実行エラー: {e}")
            self.test_results["test_summary"]["failed_tests"] += 1
            self.test_results["error"] = str(e)

        finally:
            # 結果保存
            self.save_test_results()
            self.engine.cleanup()

    def save_test_results(self):
        """テスト結果保存"""
        results_path = self.tests_dir / "ph2_005_scalability_test_results.json"

        # 結果をJSON形式で保存
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📊 テスト結果保存: {results_path}")

        # サマリーをログ出力
        self.log_test_summary()

    def log_test_summary(self):
        """テストサマリーログ出力"""
        logger.info("=" * 60)
        logger.info("🎯 PH2-005 スケーラビリティテスト サマリー")
        logger.info("=" * 60)

        summary = self.test_results["test_summary"]
        logger.info(f"総テスト数: {summary['total_tests']}")
        logger.info(f"完了テスト: {summary['completed_tests']}")
        logger.info(f"失敗テスト: {summary['failed_tests']}")
        logger.info(f"成功率: {summary.get('success_rate', 0):.1%}")

        # エンジン別結果
        logger.info("\n🔧 エンジン別テスト結果:")
        for engine_name, result in self.test_results.get("engine_tests", {}).items():
            if isinstance(result, dict) and "success_rate" in result:
                logger.info(
                    f"  {engine_name}: 成功率 {result['success_rate']:.1%}, "
                    f"スループット {result.get('throughput', 0):.2f} tasks/sec"
                )

        # ベンチマーク結果
        benchmark = self.test_results.get("integration_tests", {}).get("benchmark", {})
        if "best_engine" in benchmark:
            logger.info(
                f"\n🏆 最高性能エンジン: {benchmark['best_engine']} "
                f"({benchmark.get('best_throughput', 0):.2f} tasks/sec)"
            )

        logger.info("=" * 60)


async def main():
    """メイン実行"""
    tester = PH2005ScalabilityTester()
    await tester.run_all_tests()

    # 簡単な統合テストも実行
    logger.info("🧪 統合テスト: auto_process_batch")
    test_data = list(range(1, 11))
    results, metrics = auto_process_batch(cpu_intensive_task, test_data)
    logger.info(f"統合テスト完了: {metrics.successful_tasks}/{metrics.total_tasks} 成功")


if __name__ == "__main__":
    asyncio.run(main())
