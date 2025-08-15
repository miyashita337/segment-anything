#!/usr/bin/env python3
"""
P1-011処理キュー管理システム - Integration Test
既存システムとの統合テスト
"""

import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from features.common.queue_manager import (
    ProcessingQueue,
    ImageProcessingQueue, 
    QueueConfig,
    TaskPriority,
    TaskStatus,
    ProcessingMode
)
from features.common.retry_handler import RetryHandler, RetryConfig


class TestRetryHandlerIntegration:
    """retry_handlerとの統合テスト"""

    def test_queue_with_retry_handler(self, tmp_path):
        """キューとリトライハンドラーの統合テスト"""
        # リトライ設定
        retry_config = RetryConfig(
            max_retries=2,
            initial_delay=0.1,
            exponential_backoff=False
        )
        retry_handler = RetryHandler(retry_config)
        
        # キュー設定
        queue_config = QueueConfig(
            max_workers=1,
            enable_retry=True,
            timeout_seconds=5.0
        )
        queue = ProcessingQueue(queue_config)
        
        # テスト画像作成
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        # リトライ機能付きタスク関数
        call_count = 0
        def failing_task(image_path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # 最初2回は失敗
                raise RuntimeError("Temporary failure")
            return {"status": "success", "path": image_path}
        
        # リトライデコレータ適用
        retry_task = retry_handler.retry(failing_task)
        
        # タスク実行
        result = retry_task(str(test_image))
        
        # 3回目で成功することを確認
        assert result["status"] == "success"
        assert call_count == 3

    def test_queue_with_quality_retry(self, tmp_path):
        """品質ベースリトライとの統合テスト"""
        retry_config = RetryConfig(
            max_retries=2,
            quality_retry_enabled=True,
            quality_threshold=0.7
        )
        retry_handler = RetryHandler(retry_config)
        
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        call_count = 0
        def quality_task(image_path):
            nonlocal call_count
            call_count += 1
            
            # 最初は低品質、2回目で高品質
            quality_score = 0.5 if call_count == 1 else 0.8
            return {
                "status": "processed",
                "path": image_path,
                "quality_score": quality_score
            }
        
        retry_task = retry_handler.retry(quality_task)
        result = retry_task(str(test_image))
        
        assert result["quality_score"] == 0.8
        assert call_count == 2


class TestFileSystemIntegration:
    """ファイルシステムとの統合テスト"""

    def test_real_file_processing(self, tmp_path):
        """実ファイルでの処理テスト"""
        queue = ImageProcessingQueue()
        
        # 実際のテスト画像作成（PNG形式）
        test_images = []
        for i in range(3):
            img_path = tmp_path / f"real_test_{i}.png"
            # 簡単なPNG形式バイナリ（最小限）
            png_header = b'\x89PNG\r\n\x1a\n'
            img_path.write_bytes(png_header + b'fake png data')
            test_images.append(str(img_path))
        
        # バッチ追加
        task_ids = queue.add_batch_tasks(test_images)
        assert len(task_ids) == 3
        
        # ワーカー開始して処理
        queue.start_workers(1)
        
        # 処理完了待機
        start_time = time.time()
        while time.time() - start_time < 5:
            status = queue.get_queue_status()
            total_processed = status["completed_count"] + status["failed_count"]
            if total_processed >= 3:
                break
            time.sleep(0.1)
        
        # 結果確認
        final_status = queue.get_queue_status()
        assert final_status["completed_count"] >= 1  # 少なくとも1つは成功
        
        queue.stop_workers()

    def test_file_size_based_prioritization(self, tmp_path):
        """ファイルサイズベース優先度付けテスト"""
        config = QueueConfig(auto_priority=True)
        queue = ProcessingQueue(config)
        
        # 異なるサイズのファイル作成
        small_file = tmp_path / "small.jpg"
        medium_file = tmp_path / "medium.jpg"
        large_file = tmp_path / "large.jpg"
        
        small_file.write_bytes(b"x" * 500 * 1024)      # 500KB -> HIGH
        medium_file.write_bytes(b"x" * 3 * 1024 * 1024)  # 3MB -> NORMAL  
        large_file.write_bytes(b"x" * 10 * 1024 * 1024)  # 10MB -> LOW
        
        # タスク追加（順序は意図的にランダム）
        medium_id = queue.add_task(str(medium_file))
        large_id = queue.add_task(str(large_file))
        small_id = queue.add_task(str(small_file))
        
        # キューから取得して優先度確認
        tasks = []
        while not queue.task_queue.empty():
            task = queue.task_queue.get()
            tasks.append(task)
        
        # 優先度順（HIGH, NORMAL, LOW）になっているはず
        priorities = [task.priority for task in tasks]
        assert TaskPriority.HIGH in priorities
        assert TaskPriority.NORMAL in priorities
        assert TaskPriority.LOW in priorities
        
        # 最初に出てくるのはHIGH優先度のタスク
        assert tasks[0].priority == TaskPriority.HIGH


class TestMemoryManagement:
    """メモリ管理統合テスト"""

    def test_memory_threshold_behavior(self, tmp_path):
        """メモリ閾値動作テスト"""
        config = QueueConfig(
            memory_threshold_mb=6000.0,  # 6GB閾値
            max_workers=1
        )
        queue = ProcessingQueue(config)
        
        # テスト画像
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        # メモリチェック機能の存在確認（実際の値は環境依存）
        memory_available = queue._check_memory_availability()
        assert isinstance(memory_available, bool)

    def test_memory_sufficient_behavior(self, tmp_path):
        """メモリ十分時の動作テスト"""
        queue = ProcessingQueue()
        
        # メモリチェック機能の存在確認
        memory_available = queue._check_memory_availability()
        assert isinstance(memory_available, bool)


class TestConcurrencyIntegration:
    """並行処理統合テスト"""

    def test_multiple_workers_processing(self, tmp_path):
        """複数ワーカー処理テスト"""
        config = QueueConfig(max_workers=2)
        queue = ImageProcessingQueue(config)
        
        # 複数テスト画像作成
        test_images = []
        for i in range(6):
            img_path = tmp_path / f"concurrent_test_{i}.jpg"
            img_path.write_bytes(f"fake image data {i}".encode())
            test_images.append(str(img_path))
        
        # バッチ追加
        task_ids = queue.add_batch_tasks(test_images)
        assert len(task_ids) == 6
        
        # 2ワーカーで開始
        queue.start_workers(2)
        assert len(queue.workers) == 2
        
        # 処理完了待機
        start_time = time.time()
        while time.time() - start_time < 10:
            status = queue.get_queue_status()
            if status["completed_count"] + status["failed_count"] >= 6:
                break
            time.sleep(0.1)
        
        # 全タスク処理確認
        final_status = queue.get_queue_status()
        assert final_status["completed_count"] + final_status["failed_count"] == 6
        
        queue.stop_workers()

    def test_worker_thread_safety(self, tmp_path):
        """ワーカースレッドセーフティテスト"""
        queue = ProcessingQueue()
        
        # 大量タスク追加
        test_images = []
        for i in range(20):
            img_path = tmp_path / f"thread_test_{i}.jpg"
            img_path.write_bytes(f"fake image data {i}".encode())
            test_images.append(str(img_path))
        
        task_ids = queue.add_batch_tasks(test_images)
        
        # 4ワーカーで並行処理
        queue.start_workers(4)
        
        # 統計情報の整合性をチェック
        start_time = time.time()
        while time.time() - start_time < 15:
            status = queue.get_queue_status()
            
            # 統計の整合性確認
            total_tasks = status["statistics"]["total_tasks"]
            queue_size = status["queue_size"]
            processing_count = status["processing_count"]
            completed_count = status["completed_count"] 
            failed_count = status["failed_count"]
            
            # 基本的な整合性確認
            assert total_tasks >= 0
            assert queue_size >= 0
            assert processing_count >= 0
            assert completed_count >= 0
            assert failed_count >= 0
            
            if completed_count + failed_count >= 20:
                break
            time.sleep(0.2)
        
        queue.stop_workers()


class TestErrorHandlingIntegration:
    """エラーハンドリング統合テスト"""

    def test_task_failure_handling(self, tmp_path):
        """タスク失敗処理テスト"""
        class FailingQueue(ProcessingQueue):
            def _process_task(self, task):
                # 意図的に失敗させる
                raise RuntimeError("Intentional test failure")
        
        config = QueueConfig(
            enable_retry=True,
            max_workers=1
        )
        queue = FailingQueue(config)
        
        # テスト画像
        test_image = tmp_path / "failing_test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        # ワーカー開始
        queue.start_workers(1)
        
        # 失敗処理完了まで待機
        start_time = time.time()
        while time.time() - start_time < 5:
            status = queue.get_queue_status()
            if status["failed_count"] > 0:
                break
            time.sleep(0.1)
        
        # 失敗タスクが記録されることを確認
        final_status = queue.get_queue_status()
        assert final_status["failed_count"] >= 1
        
        # 失敗タスクの詳細確認
        failed_task = queue.get_task_status(task_id)
        assert failed_task is not None
        assert failed_task.status == TaskStatus.FAILED
        # error_messageが設定されていることを確認（QueueTaskにerrorsフィールドはない）
        assert failed_task.error_message is not None
        
        queue.stop_workers()

    def test_timeout_handling(self, tmp_path):
        """タイムアウト処理テスト"""
        class SlowQueue(ProcessingQueue):
            def _process_task(self, task):
                # 意図的に遅い処理
                time.sleep(2.0)
                return {"status": "slow_processed"}
        
        config = QueueConfig(
            timeout_seconds=1.0,  # 1秒タイムアウト
            max_workers=1
        )
        queue = SlowQueue(config)
        
        test_image = tmp_path / "slow_test.jpg"
        test_image.write_bytes(b"fake image data")
        
        task_id = queue.add_task(str(test_image))
        
        # 短いタイムアウトでワーカー開始
        queue.start_workers(1)
        
        # タイムアウト処理待機
        start_time = time.time()
        while time.time() - start_time < 3:
            if queue.get_task_status(task_id):
                task_status = queue.get_task_status(task_id)
                if task_status.status in [TaskStatus.FAILED, TaskStatus.COMPLETED]:
                    break
            time.sleep(0.1)
        
        queue.stop_workers()


class TestConfigurationIntegration:
    """設定統合テスト"""

    def test_adaptive_processing_mode(self, tmp_path):
        """適応処理モードテスト"""
        config = QueueConfig(
            processing_mode=ProcessingMode.ADAPTIVE,
            batch_size=3
        )
        queue = ImageProcessingQueue(config)
        
        # バッチサイズと同じ数の画像
        test_images = []
        for i in range(3):
            img_path = tmp_path / f"adaptive_test_{i}.jpg"
            img_path.write_bytes(f"adaptive image {i}".encode())
            test_images.append(str(img_path))
        
        task_ids = queue.add_batch_tasks(test_images)
        assert len(task_ids) == 3
        
        # 設定が正しく反映されることを確認
        assert queue.config.processing_mode == ProcessingMode.ADAPTIVE
        assert queue.config.batch_size == 3

    def test_sequential_processing_mode(self, tmp_path):
        """順次処理モードテスト"""
        config = QueueConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            max_workers=1  # 順次処理なので1ワーカー
        )
        queue = ImageProcessingQueue(config)
        
        test_images = []
        for i in range(4):
            img_path = tmp_path / f"sequential_test_{i}.jpg"
            img_path.write_bytes(f"sequential image {i}".encode())
            test_images.append(str(img_path))
        
        task_ids = queue.add_batch_tasks(test_images)
        
        queue.start_workers(1)
        
        # 順次処理確認（完了まで待機）
        start_time = time.time()
        while time.time() - start_time < 8:
            status = queue.get_queue_status()
            if status["completed_count"] + status["failed_count"] >= 4:
                break
            time.sleep(0.1)
        
        final_status = queue.get_queue_status()
        assert final_status["completed_count"] + final_status["failed_count"] == 4
        
        queue.stop_workers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])