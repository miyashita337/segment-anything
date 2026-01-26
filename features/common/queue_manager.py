"""
処理キュー管理システム (P1-011)
大量画像の効率的処理順序制御
"""

import heapq
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """タスク優先度"""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    """タスクステータス"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingMode(Enum):
    """処理モード"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class QueueTask:
    """キュータスク"""

    task_id: str
    image_path: str
    priority: TaskPriority
    created_at: datetime
    estimated_size_mb: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    result: Any = None

    def __lt__(self, other):
        """優先度比較（heapqのため）"""
        return self.priority.value < other.priority.value


@dataclass
class QueueConfig:
    """キュー設定"""

    max_queue_size: int = 1000
    max_workers: int = 4
    processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE
    memory_threshold_mb: float = 8000.0
    timeout_seconds: float = 300.0
    enable_retry: bool = True
    enable_statistics: bool = True
    auto_priority: bool = True
    batch_size: int = 10


class ProcessingQueue:
    """処理キュー管理システム"""

    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=self.config.max_queue_size)
        self.processing_tasks: Dict[str, QueueTask] = {}
        self.completed_tasks: Dict[str, QueueTask] = {}
        self.failed_tasks: Dict[str, QueueTask] = {}

        # ワーカー管理
        self.workers: List[threading.Thread] = []
        self.worker_stop_event = threading.Event()
        self.worker_lock = threading.Lock()

        # 統計情報
        self.statistics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "processing_time_total": 0.0,
            "queue_start_time": datetime.now(),
        }

        logger.info("ProcessingQueue initialized")

    def add_task(
        self,
        image_path: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        estimated_size_mb: Optional[float] = None,
    ) -> str:
        """タスク追加"""
        task_id = str(uuid.uuid4())[:8]

        # ファイルサイズベースの優先度自動調整
        if self.config.auto_priority and estimated_size_mb is None:
            estimated_size_mb = self._estimate_file_size(image_path)
            priority = self._calculate_auto_priority(estimated_size_mb)

        task = QueueTask(
            task_id=task_id,
            image_path=image_path,
            priority=priority,
            created_at=datetime.now(),
            estimated_size_mb=estimated_size_mb or 0.0,
        )

        try:
            self.task_queue.put(task, timeout=1.0)
            self.statistics["total_tasks"] += 1
            logger.info(f"Task added: {task_id} ({priority.name}) - {image_path}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            raise

    def add_batch_tasks(self, image_paths: List[str]) -> List[str]:
        """バッチタスク追加"""
        task_ids = []

        # サイズでソート（小さいファイルから処理）
        sorted_paths = self._sort_by_size(image_paths)

        for path in sorted_paths:
            try:
                task_id = self.add_task(path)
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to add batch task {path}: {e}")

        logger.info(f"Batch tasks added: {len(task_ids)}/{len(image_paths)}")
        return task_ids

    def start_workers(self, num_workers: Optional[int] = None):
        """ワーカー開始"""
        if self.workers:
            logger.warning("Workers already started")
            return

        worker_count = num_workers or self.config.max_workers
        self.worker_stop_event.clear()

        for i in range(worker_count):
            worker = threading.Thread(
                target=self._worker_loop, name=f"QueueWorker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {worker_count} queue workers")

    def stop_workers(self, timeout: float = 10.0):
        """ワーカー停止"""
        if not self.workers:
            return

        logger.info("Stopping queue workers...")
        self.worker_stop_event.set()

        # 全ワーカーの終了を待機
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))

        self.workers.clear()
        logger.info("Queue workers stopped")

    def get_task_status(self, task_id: str) -> Optional[QueueTask]:
        """タスクステータス取得"""
        # 処理中タスクから検索
        if task_id in self.processing_tasks:
            return self.processing_tasks[task_id]

        # 完了タスクから検索
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # 失敗タスクから検索
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id]

        return None

    def get_queue_status(self) -> Dict[str, Any]:
        """キューステータス取得"""
        return {
            "queue_size": self.task_queue.qsize(),
            "processing_count": len(self.processing_tasks),
            "completed_count": len(self.completed_tasks),
            "failed_count": len(self.failed_tasks),
            "workers_active": len(self.workers),
            "statistics": self.statistics.copy(),
            "config": asdict(self.config),
        }

    def cancel_task(self, task_id: str) -> bool:
        """タスクキャンセル"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task cancelled: {task_id}")
            return True
        return False

    def clear_completed_tasks(self):
        """完了タスククリア"""
        cleared_count = len(self.completed_tasks)
        self.completed_tasks.clear()
        logger.info(f"Cleared {cleared_count} completed tasks")

    def _worker_loop(self):
        """ワーカーループ"""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")

        while not self.worker_stop_event.is_set():
            try:
                # タスク取得（タイムアウト付き）
                task = self.task_queue.get(timeout=1.0)

                if task.status == TaskStatus.CANCELLED:
                    continue

                # メモリチェック
                if not self._check_memory_availability():
                    logger.warning(f"{worker_name}: Memory threshold exceeded, skipping task")
                    self.task_queue.put(task)  # キューに戻す
                    time.sleep(5.0)
                    continue

                # タスク処理開始
                with self.worker_lock:
                    self.processing_tasks[task.task_id] = task
                    task.status = TaskStatus.PROCESSING

                logger.info(f"{worker_name}: Processing task {task.task_id}")

                try:
                    # 実際の処理実行
                    start_time = time.time()
                    result = self._process_task(task)
                    processing_time = time.time() - start_time

                    # 処理成功
                    task.status = TaskStatus.COMPLETED
                    task.processing_time = processing_time
                    task.result = result

                    with self.worker_lock:
                        del self.processing_tasks[task.task_id]
                        self.completed_tasks[task.task_id] = task
                        self.statistics["completed_tasks"] += 1
                        self.statistics["processing_time_total"] += processing_time

                    logger.info(
                        f"{worker_name}: Task completed {task.task_id} in {processing_time:.2f}s"
                    )

                except Exception as e:
                    # 処理失敗
                    logger.error(f"{worker_name}: Task failed {task.task_id}: {e}")

                    task.error_message = str(e)
                    task.retry_count += 1

                    if self.config.enable_retry and task.retry_count <= task.max_retries:
                        # リトライ
                        task.status = TaskStatus.PENDING
                        self.task_queue.put(task)
                        logger.info(
                            f"{worker_name}: Task queued for retry {task.task_id} ({task.retry_count}/{task.max_retries})"
                        )
                    else:
                        # 最終失敗
                        task.status = TaskStatus.FAILED
                        with self.worker_lock:
                            del self.processing_tasks[task.task_id]
                            self.failed_tasks[task.task_id] = task
                            self.statistics["failed_tasks"] += 1

                        logger.error(f"{worker_name}: Task final failure {task.task_id}")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"{worker_name}: Worker error: {e}")

        logger.info(f"{worker_name} stopped")

    def _process_task(self, task: QueueTask) -> Any:
        """タスク処理（オーバーライド可能）"""
        # デフォルトは画像ファイル存在確認のみ
        image_path = Path(task.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {task.image_path}")

        # 実際の処理はサブクラスでオーバーライド
        return {"status": "processed", "path": task.image_path}

    def _estimate_file_size(self, image_path: str) -> float:
        """ファイルサイズ推定"""
        try:
            size_bytes = os.path.getsize(image_path)
            return size_bytes / (1024 * 1024)  # MB
        except Exception:
            return 5.0  # デフォルト値

    def _calculate_auto_priority(self, size_mb: float) -> TaskPriority:
        """サイズベース優先度計算"""
        if size_mb < 1.0:
            return TaskPriority.HIGH  # 小さいファイルは高優先度
        elif size_mb < 5.0:
            return TaskPriority.NORMAL
        else:
            return TaskPriority.LOW  # 大きいファイルは低優先度

    def _sort_by_size(self, image_paths: List[str]) -> List[str]:
        """サイズでソート"""

        def get_size(path):
            try:
                return os.path.getsize(path)
            except Exception:
                return 0

        return sorted(image_paths, key=get_size)

    def _check_memory_availability(self) -> bool:
        """メモリ利用可能性チェック"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            return available_mb > self.config.memory_threshold_mb
        except ImportError:
            return True  # psutil未インストール時は処理続行
        except Exception:
            return True  # エラー時は処理続行


class ImageProcessingQueue(ProcessingQueue):
    """画像処理特化キュー"""

    def __init__(self, config: Optional[QueueConfig] = None):
        super().__init__(config)
        logger.info("ImageProcessingQueue initialized")

    def _process_task(self, task: QueueTask) -> Any:
        """画像処理タスク実行"""
        image_path = Path(task.image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {task.image_path}")

        # 基本的な画像処理（実際の抽出処理はここに実装）
        result = {
            "status": "processed",
            "path": str(image_path),
            "size_mb": task.estimated_size_mb,
            "processing_mode": self.config.processing_mode.value,
            "task_id": task.task_id,
        }

        # 処理時間をシミュレート（実際は画像処理）
        time.sleep(0.1)

        return result


# グローバルインスタンス
default_queue = ProcessingQueue()
image_queue = ImageProcessingQueue()


def create_processing_queue(config: Optional[QueueConfig] = None) -> ProcessingQueue:
    """処理キュー作成"""
    return ProcessingQueue(config)


def create_image_queue(config: Optional[QueueConfig] = None) -> ImageProcessingQueue:
    """画像処理キュー作成"""
    return ImageProcessingQueue(config)
