#!/usr/bin/env python3
"""
非同期トラッカー実行システム (asyncio + プロセス分離)

完全非同期処理による無制限タイムアウト対策とClaude利用効率化
- asyncio: 真の非同期I/O処理
- プロセス分離: CPU集約的タスクの並列実行  
- 動的負荷分散: GPU/CPU リソース最適配分
- 障害回復: 個別プロセス障害の自動回復
"""

import aiofiles
import asyncio
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import psutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("async_tracker_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """タスク種別"""

    LIGHTWEIGHT = "lightweight"  # テスト、品質チェック等
    CPU_INTENSIVE = "cpu_intensive"  # 抽出パイプライン
    GPU_INTENSIVE = "gpu_intensive"  # ディープラーニング処理
    IO_INTENSIVE = "io_intensive"  # ファイル操作、ダッシュボード生成


class TaskStatus(Enum):
    """タスク実行状態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class TaskConfig:
    """タスク設定"""

    task_id: str
    task_type: TaskType
    command: str
    timeout: Optional[int] = None
    max_retries: int = 3
    retry_delay: int = 30
    requires_gpu: bool = False
    priority: int = 5  # 1(高) - 10(低)
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskResult:
    """タスク実行結果"""

    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    retry_count: int = 0

    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class SystemResourceMonitor:
    """システムリソース監視"""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_total = psutil.virtual_memory().total

    async def get_current_load(self) -> Dict[str, Any]:
        """現在のシステム負荷取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        gpu_info = {}
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                gpu_info = {
                    "available": True,
                    "memory_total": gpu_memory,
                    "memory_used": gpu_used,
                    "memory_percent": (gpu_used / gpu_memory) * 100,
                }
        except ImportError:
            gpu_info = {"available": False}

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "gpu": gpu_info,
        }

    def calculate_optimal_workers(self, task_type: TaskType) -> int:
        """タスク種別に基づく最適ワーカー数算出"""
        if task_type == TaskType.LIGHTWEIGHT:
            return min(self.cpu_count, 8)
        elif task_type == TaskType.CPU_INTENSIVE:
            return max(1, self.cpu_count - 2)
        elif task_type == TaskType.GPU_INTENSIVE:
            return 1  # GPU処理は通常1プロセス
        elif task_type == TaskType.IO_INTENSIVE:
            return min(self.cpu_count * 2, 16)
        return self.cpu_count // 2


class AsyncTaskExecutor:
    """非同期タスク実行エンジン"""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_queue = asyncio.Queue()
        self.resource_monitor = SystemResourceMonitor()
        self.shutdown_event = asyncio.Event()

    async def execute_task(self, config: TaskConfig) -> TaskResult:
        """単一タスクの非同期実行"""
        result = TaskResult(
            task_id=config.task_id, status=TaskStatus.PENDING, start_time=datetime.now()
        )

        retry_count = 0
        while retry_count <= config.max_retries:
            try:
                result.status = TaskStatus.RUNNING
                result.retry_count = retry_count

                logger.info(f"タスク開始: {config.task_id} (試行: {retry_count + 1})")

                # プロセス分離実行
                if config.task_type in [TaskType.CPU_INTENSIVE, TaskType.GPU_INTENSIVE]:
                    process_result = await self._execute_in_process(config)
                else:
                    process_result = await self._execute_async_subprocess(config)

                result.return_code = process_result["returncode"]
                result.stdout = process_result["stdout"]
                result.stderr = process_result["stderr"]
                result.end_time = datetime.now()

                if result.return_code == 0:
                    result.status = TaskStatus.COMPLETED
                    logger.info(f"タスク完了: {config.task_id}")
                    break
                else:
                    raise subprocess.CalledProcessError(
                        result.return_code, config.command, result.stderr
                    )

            except asyncio.TimeoutError:
                result.status = TaskStatus.TIMEOUT
                result.error_message = f"タイムアウト (制限: {config.timeout}秒)"
                logger.warning(f"タスクタイムアウト: {config.task_id}")

            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error_message = str(e)
                logger.error(f"タスク失敗: {config.task_id}, エラー: {e}")

            retry_count += 1
            if retry_count <= config.max_retries:
                result.status = TaskStatus.RETRYING
                logger.info(f"リトライ待機: {config.task_id} ({config.retry_delay}秒)")
                await asyncio.sleep(config.retry_delay)

        if result.status != TaskStatus.COMPLETED:
            result.end_time = datetime.now()
            logger.error(f"タスク最終失敗: {config.task_id}")

        return result

    async def _execute_async_subprocess(self, config: TaskConfig) -> Dict[str, Any]:
        """軽量タスクの非同期サブプロセス実行"""
        try:
            if config.timeout:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        config.command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd="/mnt/c/AItools/segment-anything",
                    ),
                    timeout=config.timeout,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=config.timeout
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    config.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/mnt/c/AItools/segment-anything",
                )
                stdout, stderr = await process.communicate()

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            }

        except asyncio.TimeoutError:
            if "process" in locals():
                process.kill()
                await process.wait()
            raise

    async def _execute_in_process(self, config: TaskConfig) -> Dict[str, Any]:
        """CPU/GPU集約的タスクのプロセス分離実行"""
        loop = asyncio.get_event_loop()

        # ProcessPoolExecutor使用
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.resource_monitor.calculate_optimal_workers(config.task_type)
        ) as executor:
            try:
                result = await loop.run_in_executor(
                    executor, self._run_subprocess_sync, config.command, config.timeout
                )
                return result

            except concurrent.futures.TimeoutError:
                raise asyncio.TimeoutError()

    @staticmethod
    def _run_subprocess_sync(command: str, timeout: Optional[int]) -> Dict[str, Any]:
        """同期サブプロセス実行（ProcessPoolExecutor用）"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/mnt/c/AItools/segment-anything",
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"プロセスタイムアウト: {timeout}秒"}

    async def run_parallel_tasks(self, task_configs: List[TaskConfig]) -> Dict[str, TaskResult]:
        """複数タスクの並列実行"""
        # 依存関係解決
        ordered_tasks = self._resolve_dependencies(task_configs)

        # タスクグループごとに並列実行
        all_results = {}

        for task_group in ordered_tasks:
            # 同一グループ内は並列実行
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

            async def execute_with_semaphore(config: TaskConfig) -> Tuple[str, TaskResult]:
                async with semaphore:
                    result = await self.execute_task(config)
                    return config.task_id, result

            # 並列実行
            tasks = [execute_with_semaphore(config) for config in task_group]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 結果統合
            for result in group_results:
                if isinstance(result, tuple):
                    task_id, task_result = result
                    all_results[task_id] = task_result
                else:
                    logger.error(f"タスク実行例外: {result}")

        return all_results

    def _resolve_dependencies(self, task_configs: List[TaskConfig]) -> List[List[TaskConfig]]:
        """依存関係解決とタスクグループ化"""
        # 簡易実装: 依存関係のないタスクを並列実行
        # より複雑な依存関係解決は必要に応じて拡張

        no_deps = [config for config in task_configs if not config.dependencies]
        with_deps = [config for config in task_configs if config.dependencies]

        result = []
        if no_deps:
            result.append(no_deps)
        if with_deps:
            result.append(with_deps)

        return result


class TrackerWorkflowManager:
    """トラッカー統合ワークフロー管理"""

    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)
        self.executor = AsyncTaskExecutor()

    async def execute_full_tracker_workflow(
        self, tracker_id: str, input_dir: str, skip_extraction: bool = False
    ) -> Dict[str, TaskResult]:
        """完全トラッカーワークフロー実行"""

        logger.info(f"トラッカーワークフロー開始: {tracker_id}")

        # ワークスペース準備
        workspace_dir = self.workspace_base / tracker_id
        await self._prepare_workspace(workspace_dir)

        # タスク設定生成
        task_configs = self._generate_task_configs(tracker_id, input_dir, skip_extraction)

        # 並列実行
        results = await self.executor.run_parallel_tasks(task_configs)

        # 結果レポート生成
        await self._generate_execution_report(tracker_id, results)

        logger.info(f"トラッカーワークフロー完了: {tracker_id}")
        return results

    async def _prepare_workspace(self, workspace_dir: Path):
        """ワークスペース準備"""
        for subdir in ["extraction", "quality", "dashboard", "tests"]:
            (workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _generate_task_configs(
        self, tracker_id: str, input_dir: str, skip_extraction: bool
    ) -> List[TaskConfig]:
        """タスク設定生成"""

        configs = [
            # 軽量テスト群（並列実行可能）
            TaskConfig(
                task_id="unit_tests",
                task_type=TaskType.LIGHTWEIGHT,
                command="python3 -m pytest tests/unit/test_extract.py -v",
                timeout=300,
                priority=1,
            ),
            TaskConfig(
                task_id="integration_tests",
                task_type=TaskType.LIGHTWEIGHT,
                command="python3 -m pytest tests/integration/test_extraction_pipeline.py -v",
                timeout=600,
                priority=2,
            ),
            TaskConfig(
                task_id="quality_check",
                task_type=TaskType.LIGHTWEIGHT,
                command="./bin/shell/linter.sh",
                timeout=180,
                priority=3,
            ),
        ]

        if not skip_extraction:
            # CPU集約的抽出処理（プロセス分離）
            configs.append(
                TaskConfig(
                    task_id="extraction_pipeline",
                    task_type=TaskType.CPU_INTENSIVE,
                    command=f"python3 tools/automation/batched_extraction_runner.py --input_dir '{input_dir}' --tracker_id '{tracker_id}'",
                    timeout=None,  # 無制限
                    requires_gpu=True,
                    priority=4,
                    dependencies=["unit_tests", "quality_check"],
                )
            )

        # I/O集約的処理（ダッシュボード生成等）
        configs.extend(
            [
                TaskConfig(
                    task_id="dashboard_generation",
                    task_type=TaskType.IO_INTENSIVE,
                    command=f"python3 tools/core/quality_dashboard.py --tracker_id {tracker_id}",
                    timeout=300,
                    priority=5,
                    dependencies=["extraction_pipeline"] if not skip_extraction else [],
                ),
                TaskConfig(
                    task_id="sheets_update",
                    task_type=TaskType.LIGHTWEIGHT,
                    command=f"python3 tools/progress_tracker/cli.py update --tracker_id {tracker_id} --status '/release'",
                    timeout=60,
                    priority=6,
                    dependencies=["dashboard_generation"],
                ),
            ]
        )

        return configs

    async def _generate_execution_report(self, tracker_id: str, results: Dict[str, TaskResult]):
        """実行レポート生成"""
        report_path = self.workspace_base / tracker_id / "execution_report.json"

        report_data = {
            "tracker_id": tracker_id,
            "execution_time": datetime.now().isoformat(),
            "tasks": {task_id: asdict(result) for task_id, result in results.items()},
            "summary": {
                "total_tasks": len(results),
                "completed": sum(1 for r in results.values() if r.status == TaskStatus.COMPLETED),
                "failed": sum(1 for r in results.values() if r.status == TaskStatus.FAILED),
                "total_duration": str(
                    sum((r.duration for r in results.values() if r.duration), timedelta())
                ),
            },
        }

        async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(report_data, indent=2, ensure_ascii=False))

        logger.info(f"実行レポート生成: {report_path}")


async def main():
    """メイン実行関数"""
    if len(sys.argv) < 3:
        print("使用法: python async_tracker_system.py <tracker_id> <input_dir> [--skip-extraction]")
        sys.exit(1)

    tracker_id = sys.argv[1]
    input_dir = sys.argv[2]
    skip_extraction = "--skip-extraction" in sys.argv

    manager = TrackerWorkflowManager()

    try:
        results = await manager.execute_full_tracker_workflow(
            tracker_id, input_dir, skip_extraction
        )

        # 結果サマリー表示
        print(f"\n=== {tracker_id} 実行結果 ===")
        for task_id, result in results.items():
            status_icon = "✅" if result.status == TaskStatus.COMPLETED else "❌"
            print(f"{status_icon} {task_id}: {result.status.value}")
            if result.duration:
                print(f"   実行時間: {result.duration}")

        # 失敗タスクの詳細表示
        failed_tasks = [r for r in results.values() if r.status != TaskStatus.COMPLETED]
        if failed_tasks:
            print(f"\n❌ 失敗タスク ({len(failed_tasks)}件):")
            for result in failed_tasks:
                print(f"- {result.task_id}: {result.error_message}")

    except KeyboardInterrupt:
        print("\n中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Windowsでのマルチプロセッシング対応
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)

    asyncio.run(main())
