#!/usr/bin/env python3
"""
バックグラウンド実行制御システム
CLAUDE.md準拠の自動実行管理
"""

import os
import sys
import time
import threading
import multiprocessing
import queue
import logging
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackgroundExecutor:
    """バックグラウンドタスク実行管理クラス"""
    
    def __init__(self, max_workers: int = 4):
        """
        初期化
        
        Args:
            max_workers: 最大並列実行数
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        self.task_status = {}
        self.lock = threading.Lock()
        
    def start(self):
        """バックグラウンド実行開始"""
        if self.running:
            logger.warning("Background executor is already running")
            return
            
        self.running = True
        
        # ワーカースレッド起動
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.max_workers} background workers")
        
    def stop(self):
        """バックグラウンド実行停止"""
        self.running = False
        
        # 停止シグナルをキューに送信
        for _ in range(self.max_workers):
            self.task_queue.put(None)
            
        # ワーカー終了待機
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        self.workers.clear()
        logger.info("Background executor stopped")
        
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """
        タスクをキューに追加
        
        Args:
            task_id: タスクID
            func: 実行する関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数
            
        Returns:
            タスクID
        """
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': datetime.now(),
            'status': 'pending'
        }
        
        with self.lock:
            self.task_status[task_id] = task
            
        self.task_queue.put(task)
        logger.info(f"Task {task_id} submitted")
        
        return task_id
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        タスクステータス取得
        
        Args:
            task_id: タスクID
            
        Returns:
            タスクステータス辞書
        """
        with self.lock:
            return self.task_status.get(task_id, {}).copy()
            
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """全タスクステータス取得"""
        with self.lock:
            return {k: v.copy() for k, v in self.task_status.items()}
            
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        タスク結果取得（ブロッキング）
        
        Args:
            task_id: タスクID
            timeout: タイムアウト秒数
            
        Returns:
            タスク実行結果
        """
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            
            if status.get('status') == 'completed':
                return status.get('result')
            elif status.get('status') == 'failed':
                raise Exception(f"Task {task_id} failed: {status.get('error')}")
                
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
                
            time.sleep(0.1)
            
    def _worker_loop(self):
        """ワーカーループ"""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")
        
        while self.running:
            try:
                # タスク取得（タイムアウト付き）
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:
                    break
                    
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                
        logger.info(f"{worker_name} stopped")
        
    def _execute_task(self, task: Dict[str, Any]):
        """タスク実行"""
        task_id = task['id']
        
        # ステータス更新
        with self.lock:
            task['status'] = 'running'
            task['started_at'] = datetime.now()
            self.task_status[task_id] = task
            
        logger.info(f"Executing task {task_id}")
        
        try:
            # 関数実行
            result = task['func'](*task['args'], **task['kwargs'])
            
            # 成功時の処理
            with self.lock:
                task['status'] = 'completed'
                task['completed_at'] = datetime.now()
                task['result'] = result
                self.task_status[task_id] = task
                
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # エラー時の処理
            with self.lock:
                task['status'] = 'failed'
                task['failed_at'] = datetime.now()
                task['error'] = str(e)
                self.task_status[task_id] = task
                
            logger.error(f"Task {task_id} failed: {e}")
            

class TaskManager:
    """タスク管理の高レベルインターフェース"""
    
    def __init__(self):
        self.executor = BackgroundExecutor()
        self.executor.start()
        
    def run_batch_extraction(self, input_dir: str, output_dir: str, **kwargs) -> str:
        """バッチ抽出実行"""
        from tools.sam_yolo_character_segment import process_batch
        
        task_id = f"batch_extraction_{int(time.time())}"
        
        return self.executor.submit_task(
            task_id,
            process_batch,
            input_dir=input_dir,
            output_dir=output_dir,
            **kwargs
        )
        
    def run_yolo_comparison(self, thresholds: List[float]) -> str:
        """YOLO閾値比較実行"""
        def compare_thresholds():
            results = {}
            for threshold in thresholds:
                # 実際の比較ロジック
                results[threshold] = {'dummy': 'result'}
            return results
            
        task_id = f"yolo_comparison_{int(time.time())}"
        
        return self.executor.submit_task(
            task_id,
            compare_thresholds
        )
        
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """タスクステータス取得"""
        return self.executor.get_task_status(task_id)
        
    def wait_for_completion(self, task_id: str, timeout: float = 3600) -> Any:
        """タスク完了待機"""
        return self.executor.get_result(task_id, timeout)
        
    def shutdown(self):
        """シャットダウン"""
        self.executor.stop()


# シングルトンインスタンス
_task_manager = None


def get_task_manager() -> TaskManager:
    """タスクマネージャーのシングルトンインスタンス取得"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager