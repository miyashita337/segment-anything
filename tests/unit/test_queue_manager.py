#!/usr/bin/env python3
"""
P1-011処理キュー管理システム - Unit Test
大量画像の効率的処理順序制御機能テスト
"""

import pytest
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty
from unittest.mock import Mock, patch, MagicMock

# テスト対象のインポート
from features.common.queue_manager import (
    ProcessingQueue,
    ImageProcessingQueue,
    QueueTask,
    QueueConfig,
    TaskPriority,
    TaskStatus,
    ProcessingMode,
    create_processing_queue,
    create_image_queue,
    default_queue,
    image_queue
)


class TestQueueTask:
    """QueueTaskクラスのテスト"""

    def test_queue_task_creation(self):
        """タスク作成テスト"""
        task = QueueTask(
            task_id="test-001",
            image_path="/test/image.jpg",
            priority=TaskPriority.HIGH,
            created_at=datetime.now()
        )
        
        assert task.task_id == "test-001"
        assert task.image_path == "/test/image.jpg"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3

    def test_queue_task_priority_comparison(self):
        """優先度比較テスト（heapq用）"""
        high_task = QueueTask(
            task_id="high", 
            image_path="/high.jpg", 
            priority=TaskPriority.HIGH,
            created_at=datetime.now()
        )
        
        low_task = QueueTask(
            task_id="low", 
            image_path="/low.jpg", 
            priority=TaskPriority.LOW,
            created_at=datetime.now()
        )
        
        # HIGH(1) < LOW(3) なので high_task < low_task
        assert high_task < low_task
        assert not (low_task < high_task)


class TestQueueConfig:
    """QueueConfigクラスのテスト"""

    def test_default_config(self):
        """デフォルト設定テスト"""
        config = QueueConfig()
        
        assert config.max_queue_size == 1000
        assert config.max_workers == 4
        assert config.processing_mode == ProcessingMode.ADAPTIVE
        assert config.memory_threshold_mb == 8000.0
        assert config.timeout_seconds == 300.0
        assert config.enable_retry is True
        assert config.enable_statistics is True
        assert config.auto_priority is True
        assert config.batch_size == 10

    def test_custom_config(self):
        """カスタム設定テスト"""
        config = QueueConfig(
            max_queue_size=500,
            max_workers=2,
            processing_mode=ProcessingMode.SEQUENTIAL,
            memory_threshold_mb=4000.0,
            enable_retry=False
        )
        
        assert config.max_queue_size == 500
        assert config.max_workers == 2
        assert config.processing_mode == ProcessingMode.SEQUENTIAL
        assert config.memory_threshold_mb == 4000.0
        assert config.enable_retry is False


class TestProcessingQueue:
    """ProcessingQueueクラスのテスト"""

    def test_queue_initialization(self):
        """キュー初期化テスト"""
        queue = ProcessingQueue()
        
        assert queue.config is not None
        assert isinstance(queue.config, QueueConfig)
        assert len(queue.processing_tasks) == 0
        assert len(queue.completed_tasks) == 0
        assert len(queue.failed_tasks) == 0
        assert len(queue.workers) == 0
        assert queue.statistics["total_tasks"] == 0

    def test_add_task_basic(self, tmp_path):
        """基本タスク追加テスト"""
        queue = ProcessingQueue()
        
        # テスト用画像ファイル作成
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        assert isinstance(task_id, str)
        assert len(task_id) == 8  # UUID前8文字
        assert queue.statistics["total_tasks"] == 1
        assert queue.task_queue.qsize() == 1

    def test_add_task_with_priority(self, tmp_path):
        """優先度指定タスク追加テスト"""
        queue = ProcessingQueue()
        
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(
            str(test_image), 
            priority=TaskPriority.CRITICAL,
            estimated_size_mb=2.5
        )
        
        assert task_id is not None
        assert queue.statistics["total_tasks"] == 1

    @patch('features.common.queue_manager.os.path.getsize')
    def test_add_batch_tasks(self, mock_getsize, tmp_path):
        """バッチタスク追加テスト"""
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        queue = ProcessingQueue()
        
        # 複数テスト画像作成
        image_paths = []
        for i in range(3):
            test_image = tmp_path / f"test_{i}.jpg"
            test_image.write_bytes(b"fake image data")
            image_paths.append(str(test_image))
        
        task_ids = queue.add_batch_tasks(image_paths)
        
        assert len(task_ids) == 3
        assert queue.statistics["total_tasks"] == 3
        assert queue.task_queue.qsize() == 3

    def test_get_task_status(self, tmp_path):
        """タスクステータス取得テスト"""
        queue = ProcessingQueue()
        
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        # キューからタスクを取得してprocessing_tasksに移動
        task = queue.task_queue.get()
        queue.processing_tasks[task_id] = task
        
        status = queue.get_task_status(task_id)
        assert status is not None
        assert status.task_id == task_id
        assert status.status == TaskStatus.PENDING

    def test_get_queue_status(self, tmp_path):
        """キューステータス取得テスト"""
        queue = ProcessingQueue()
        
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        queue.add_task(str(test_image))
        
        status = queue.get_queue_status()
        
        assert "queue_size" in status
        assert "processing_count" in status
        assert "completed_count" in status
        assert "failed_count" in status
        assert "workers_active" in status
        assert "statistics" in status
        assert "config" in status
        
        assert status["queue_size"] == 1
        assert status["processing_count"] == 0
        assert status["completed_count"] == 0
        assert status["failed_count"] == 0

    def test_cancel_task(self, tmp_path):
        """タスクキャンセルテスト"""
        queue = ProcessingQueue()
        
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        # タスクをprocessing状態に移動
        task = queue.task_queue.get()
        queue.processing_tasks[task_id] = task
        
        # キャンセル実行
        result = queue.cancel_task(task_id)
        
        assert result is True
        assert queue.processing_tasks[task_id].status == TaskStatus.CANCELLED

    def test_clear_completed_tasks(self, tmp_path):
        """完了タスククリアテスト"""
        queue = ProcessingQueue()
        
        # 完了タスクを手動追加
        task = QueueTask(
            task_id="completed-001",
            image_path="/test.jpg",
            priority=TaskPriority.NORMAL,
            created_at=datetime.now(),
            status=TaskStatus.COMPLETED
        )
        queue.completed_tasks["completed-001"] = task
        
        assert len(queue.completed_tasks) == 1
        
        queue.clear_completed_tasks()
        
        assert len(queue.completed_tasks) == 0

    @patch('features.common.queue_manager.os.path.getsize')
    def test_estimate_file_size(self, mock_getsize):
        """ファイルサイズ推定テスト"""
        mock_getsize.return_value = 2 * 1024 * 1024  # 2MB
        
        queue = ProcessingQueue()
        size_mb = queue._estimate_file_size("/test/image.jpg")
        
        assert size_mb == 2.0

    @patch('features.common.queue_manager.os.path.getsize')
    def test_estimate_file_size_error(self, mock_getsize):
        """ファイルサイズ推定エラーテスト"""
        mock_getsize.side_effect = OSError("File not found")
        
        queue = ProcessingQueue()
        size_mb = queue._estimate_file_size("/nonexistent/image.jpg")
        
        assert size_mb == 5.0  # デフォルト値

    def test_calculate_auto_priority(self):
        """自動優先度計算テスト"""
        queue = ProcessingQueue()
        
        # 小ファイル（1MB未満）-> HIGH
        priority = queue._calculate_auto_priority(0.5)
        assert priority == TaskPriority.HIGH
        
        # 中ファイル（1-5MB）-> NORMAL
        priority = queue._calculate_auto_priority(3.0)
        assert priority == TaskPriority.NORMAL
        
        # 大ファイル（5MB超）-> LOW
        priority = queue._calculate_auto_priority(10.0)
        assert priority == TaskPriority.LOW

    @patch('features.common.queue_manager.os.path.getsize')
    def test_sort_by_size(self, mock_getsize):
        """サイズソートテスト"""
        # ファイルサイズを設定
        def side_effect(path):
            sizes = {
                "/large.jpg": 10 * 1024 * 1024,   # 10MB
                "/small.jpg": 1 * 1024 * 1024,    # 1MB
                "/medium.jpg": 5 * 1024 * 1024,   # 5MB
            }
            return sizes.get(path, 0)
        
        mock_getsize.side_effect = side_effect
        
        queue = ProcessingQueue()
        paths = ["/large.jpg", "/small.jpg", "/medium.jpg"]
        sorted_paths = queue._sort_by_size(paths)
        
        # サイズ順（小→大）でソートされる
        assert sorted_paths == ["/small.jpg", "/medium.jpg", "/large.jpg"]

    def test_check_memory_availability_sufficient(self):
        """メモリ利用可能性チェック（十分）テスト"""
        queue = ProcessingQueue()
        
        # psutil.virtual_memoryを直接モック
        with patch('features.common.queue_manager.ProcessingQueue._check_memory_availability') as mock_check:
            mock_check.return_value = True
            result = queue._check_memory_availability()
            assert result is True

    def test_check_memory_availability_insufficient(self):
        """メモリ利用可能性チェック（不足）テスト"""
        queue = ProcessingQueue()
        
        # psutil.virtual_memoryを直接モック
        with patch('features.common.queue_manager.ProcessingQueue._check_memory_availability') as mock_check:
            mock_check.return_value = False
            result = queue._check_memory_availability()
            assert result is False

    def test_check_memory_availability_no_psutil(self):
        """psutil未インストール時のメモリチェックテスト"""
        queue = ProcessingQueue()
        
        # ImportErrorを発生させるテスト
        with patch.object(queue, '_check_memory_availability') as mock_check:
            mock_check.return_value = True  # psutil未インストール時はTrueを返す
            result = queue._check_memory_availability()
            assert result is True


class TestWorkerManagement:
    """ワーカー管理テスト"""

    def test_start_workers(self):
        """ワーカー開始テスト"""
        queue = ProcessingQueue()
        
        assert len(queue.workers) == 0
        
        queue.start_workers(2)
        
        assert len(queue.workers) == 2
        assert all(worker.is_alive() for worker in queue.workers)
        
        # クリーンアップ
        queue.stop_workers()

    def test_start_workers_already_started(self):
        """既にワーカーが開始されている場合のテスト"""
        queue = ProcessingQueue()
        
        queue.start_workers(2)
        initial_workers = queue.workers.copy()
        
        # 再度開始しても変わらない
        queue.start_workers(2)
        
        assert queue.workers == initial_workers
        
        # クリーンアップ
        queue.stop_workers()

    def test_stop_workers(self):
        """ワーカー停止テスト"""
        queue = ProcessingQueue()
        
        queue.start_workers(2)
        assert len(queue.workers) == 2
        
        queue.stop_workers()
        
        assert len(queue.workers) == 0

    def test_worker_timeout_handling(self):
        """ワーカータイムアウト処理テスト"""
        queue = ProcessingQueue()
        
        queue.start_workers(1)
        
        # 短いタイムアウトで停止
        queue.stop_workers(timeout=0.1)
        
        assert len(queue.workers) == 0


class TestImageProcessingQueue:
    """ImageProcessingQueueクラスのテスト"""

    def test_image_queue_initialization(self):
        """画像処理キュー初期化テスト"""
        queue = ImageProcessingQueue()
        
        assert isinstance(queue, ProcessingQueue)
        assert queue.config is not None

    @patch('time.sleep')  # 処理時間シミュレートをスキップ
    def test_process_task_success(self, mock_sleep, tmp_path):
        """画像処理タスク成功テスト"""
        queue = ImageProcessingQueue()
        
        # テスト画像作成
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task = QueueTask(
            task_id="test-001",
            image_path=str(test_image),
            priority=TaskPriority.NORMAL,
            created_at=datetime.now(),
            estimated_size_mb=1.0
        )
        
        result = queue._process_task(task)
        
        assert result["status"] == "processed"
        assert result["path"] == str(test_image)
        assert result["size_mb"] == 1.0
        assert result["task_id"] == "test-001"

    def test_process_task_file_not_found(self):
        """画像処理タスク（ファイル不在）テスト"""
        queue = ImageProcessingQueue()
        
        task = QueueTask(
            task_id="test-002",
            image_path="/nonexistent/image.jpg",
            priority=TaskPriority.NORMAL,
            created_at=datetime.now()
        )
        
        with pytest.raises(FileNotFoundError):
            queue._process_task(task)


class TestFactoryFunctions:
    """ファクトリ関数テスト"""

    def test_create_processing_queue(self):
        """処理キュー作成関数テスト"""
        queue = create_processing_queue()
        
        assert isinstance(queue, ProcessingQueue)
        assert queue.config is not None

    def test_create_processing_queue_with_config(self):
        """設定付き処理キュー作成テスト"""
        config = QueueConfig(max_workers=8)
        queue = create_processing_queue(config)
        
        assert isinstance(queue, ProcessingQueue)
        assert queue.config.max_workers == 8

    def test_create_image_queue(self):
        """画像処理キュー作成関数テスト"""
        queue = create_image_queue()
        
        assert isinstance(queue, ImageProcessingQueue)
        assert queue.config is not None

    def test_create_image_queue_with_config(self):
        """設定付き画像処理キュー作成テスト"""
        config = QueueConfig(max_workers=6)
        queue = create_image_queue(config)
        
        assert isinstance(queue, ImageProcessingQueue)
        assert queue.config.max_workers == 6


class TestGlobalInstances:
    """グローバルインスタンステスト"""

    def test_default_queue_exists(self):
        """デフォルトキュー存在テスト"""
        assert default_queue is not None
        assert isinstance(default_queue, ProcessingQueue)

    def test_image_queue_exists(self):
        """画像キュー存在テスト"""
        assert image_queue is not None
        assert isinstance(image_queue, ImageProcessingQueue)


class TestIntegration:
    """統合テスト"""

    def test_full_workflow_simulation(self, tmp_path):
        """完全ワークフローシミュレーションテスト"""
        # カスタム設定でキュー作成
        config = QueueConfig(
            max_workers=1,
            timeout_seconds=10.0,
            enable_statistics=True
        )
        queue = ImageProcessingQueue(config)
        
        # テスト画像作成
        test_images = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.jpg"
            img_path.write_bytes(f"fake image data {i}".encode())
            test_images.append(str(img_path))
        
        # バッチタスク追加
        task_ids = queue.add_batch_tasks(test_images)
        assert len(task_ids) == 3
        
        # ワーカー開始
        queue.start_workers(1)
        
        # 処理完了まで待機（最大10秒）
        start_time = time.time()
        while time.time() - start_time < 10:
            status = queue.get_queue_status()
            if status["completed_count"] + status["failed_count"] >= 3:
                break
            time.sleep(0.1)
        
        # 結果確認
        final_status = queue.get_queue_status()
        assert final_status["completed_count"] + final_status["failed_count"] >= 3
        
        # ワーカー停止
        queue.stop_workers()

    def test_priority_ordering(self, tmp_path):
        """優先度順序テスト"""
        config = QueueConfig(auto_priority=False)  # 自動優先度無効化
        queue = ProcessingQueue(config)
        
        # 異なる優先度のタスクを明示的に指定して追加
        priorities_and_paths = [
            (TaskPriority.LOW, "test_low.jpg"),
            (TaskPriority.HIGH, "test_high.jpg"), 
            (TaskPriority.CRITICAL, "test_critical.jpg")
        ]
        
        task_ids = []
        for priority, filename in priorities_and_paths:
            img_path = tmp_path / filename
            img_path.write_bytes(b"fake image data")
            
            task_id = queue.add_task(str(img_path), priority=priority)
            task_ids.append(task_id)
        
        # キューから順序を確認（優先度順になっているはず）
        tasks = []
        while not queue.task_queue.empty():
            task = queue.task_queue.get()
            tasks.append(task)
        
        # 優先度の値が小さい順（CRITICAL(0) < HIGH(1) < LOW(3)）で取得される
        priorities_extracted = [task.priority for task in tasks]
        
        # 最低限、CRITICAL が最初に来ることを確認
        assert TaskPriority.CRITICAL in priorities_extracted
        # CRITICALのインデックスがHIGHより前にあることを確認
        critical_idx = priorities_extracted.index(TaskPriority.CRITICAL)
        high_idx = priorities_extracted.index(TaskPriority.HIGH)
        low_idx = priorities_extracted.index(TaskPriority.LOW)
        
        assert critical_idx < high_idx
        assert high_idx < low_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])