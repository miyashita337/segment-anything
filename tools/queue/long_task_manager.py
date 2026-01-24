#!/usr/bin/env python3
"""
長時間処理キュー管理システム
QUAL-044: SubAgent統合による同一セッション自動化

タイムアウト制約（2分）を回避し、長時間処理（pytest、extract_character.py等）を
バックグラウンドで実行・管理するキューシステム
"""

import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import signal
import threading
import logging

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """タスク状態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class QueueTask:
    """キュータスク定義"""
    task_id: str
    command: str
    task_type: str  # "pytest", "extract_character", etc.
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    process_pid: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueTask':
        """辞書からインスタンス生成"""
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class LongTaskQueue:
    """長時間処理キュー管理クラス"""
    
    def __init__(self, workspace_path: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-044"):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
        """
        self.workspace = Path(workspace_path)
        self.queue_dir = self.workspace / "queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # 状態ファイル
        self.status_file = self.queue_dir / "queue_status.json"
        self.task_queue_file = self.queue_dir / "task_queue.json"
        
        # タスクキュー（FIFO）
        self.task_queue: deque[QueueTask] = deque()
        self.current_task: Optional[QueueTask] = None
        self.running_process: Optional[subprocess.Popen] = None
        
        # スレッド制御
        self.executor_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 初期化
        self._load_queue_state()
        logger.info(f"LongTaskQueue initialized at {self.workspace}")
    
    def _load_queue_state(self) -> None:
        """キュー状態の読み込み"""
        if self.task_queue_file.exists():
            try:
                with open(self.task_queue_file, 'r') as f:
                    data = json.load(f)
                    for task_data in data.get('tasks', []):
                        task = QueueTask.from_dict(task_data)
                        # 実行中だったタスクはPENDINGに戻す
                        if task.status == TaskStatus.RUNNING:
                            task.status = TaskStatus.PENDING
                        self.task_queue.append(task)
                logger.info(f"Loaded {len(self.task_queue)} tasks from queue")
            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")
    
    def _save_queue_state(self) -> None:
        """キュー状態の保存"""
        try:
            tasks_data = [task.to_dict() for task in self.task_queue]
            if self.current_task:
                tasks_data.insert(0, self.current_task.to_dict())
            
            with open(self.task_queue_file, 'w') as f:
                json.dump({'tasks': tasks_data}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def _update_status_file(self, status: Dict[str, Any]) -> None:
        """状態ファイル更新（SubAgent監視用）"""
        try:
            status['timestamp'] = datetime.now().isoformat()
            status['queue_length'] = len(self.task_queue)
            
            if self.current_task:
                status['current_task'] = self.current_task.to_dict()
            
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update status file: {e}")
    
    def enqueue_task(self, command: str, task_type: str) -> str:
        """
        タスクをキューに追加
        
        Args:
            command: 実行コマンド
            task_type: タスクタイプ
            
        Returns:
            task_id: タスクID
        """
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = QueueTask(
            task_id=task_id,
            command=command,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        self.task_queue.append(task)
        self._save_queue_state()
        
        logger.info(f"Enqueued task: {task_id}")
        self._update_status_file({
            'status': 'task_enqueued',
            'task_id': task_id
        })
        
        return task_id
    
    def start_background_execution(self) -> None:
        """バックグラウンド実行開始"""
        if self.executor_thread and self.executor_thread.is_alive():
            logger.warning("Executor already running")
            return
        
        self.stop_event.clear()
        self.executor_thread = threading.Thread(target=self._execute_queue)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        logger.info("Background execution started")
    
    def _execute_queue(self) -> None:
        """キュー実行ループ"""
        logger.info("🔄 Starting _execute_queue loop")
        while not self.stop_event.is_set():
            logger.debug(f"Queue loop: queue_length={len(self.task_queue)}, current_task={self.current_task is not None}")
            
            if not self.task_queue and not self.current_task:
                logger.debug("Queue empty, waiting...")
                time.sleep(5)  # キューが空の場合は待機
                continue
            
            if not self.current_task and self.task_queue:
                self.current_task = self.task_queue.popleft()
                logger.info(f"🎯 Picked up task from queue: {self.current_task.task_id}")
            
            if self.current_task and self.current_task.status == TaskStatus.PENDING:
                logger.info(f"▶️ Executing task: {self.current_task.task_id}")
                self._execute_task(self.current_task)
            elif self.current_task:
                logger.debug(f"Task {self.current_task.task_id} status: {self.current_task.status}")
            
            time.sleep(1)
    
    def _execute_task(self, task: QueueTask) -> None:
        """タスク実行"""
        logger.info(f"Executing task: {task.task_id}")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        self._save_queue_state()
        
        # 出力ファイル設定
        output_file = self.queue_dir / f"{task.task_id}_output.log"
        task.output_file = str(output_file)
        
        self._update_status_file({
            'status': 'task_running',
            'task_id': task.task_id,
            'command': task.command
        })
        
        try:
            # プロセス実行前の詳細ログ
            logger.info(f"🚀 Starting subprocess execution")
            logger.info(f"   Command: {task.command}")
            logger.info(f"   Working directory: /mnt/c/AItools/segment-anything")
            logger.info(f"   Output file: {output_file}")
            
            # 作業ディレクトリの存在確認
            cwd_path = "/mnt/c/AItools/segment-anything"
            if not os.path.exists(cwd_path):
                logger.error(f"❌ Working directory does not exist: {cwd_path}")
                raise FileNotFoundError(f"Working directory not found: {cwd_path}")
            
            logger.info(f"✅ Working directory confirmed: {cwd_path}")
            
            # プロセス実行
            with open(output_file, 'w') as f:
                logger.info(f"📝 Output file opened for writing: {output_file}")
                
                self.running_process = subprocess.Popen(
                    task.command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=cwd_path,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )
                
                task.process_pid = self.running_process.pid
                # 重要: process_pid設定後にキュー状態を保存
                self._save_queue_state()
                logger.info(f"✅ Process started successfully with PID: {self.running_process.pid}")
                
                # プロセス完了待機開始
                logger.info(f"⏳ Waiting for process completion...")
                return_code = self.running_process.wait()
                logger.info(f"🏁 Process completed with return code: {return_code}")
                
                if return_code == 0:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now().isoformat()
                    logger.info(f"Task completed: {task.task_id}")
                    
                    self._update_status_file({
                        'status': 'task_completed',
                        'task_id': task.task_id,
                        'output_file': str(output_file)
                    })
                else:
                    raise subprocess.CalledProcessError(return_code, task.command)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Task failed: {task.task_id}, return code: {e.returncode}")
            
            # リトライ処理
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                logger.info(f"Retrying task: {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                time.sleep(5)  # リトライ前に待機
                self._execute_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.error_message = f"Failed after {task.max_retries} retries"
                task.completed_at = datetime.now().isoformat()
                
                self._update_status_file({
                    'status': 'task_failed',
                    'task_id': task.task_id,
                    'error': task.error_message,
                    'requires_manual_review': True
                })
        
        except Exception as e:
            logger.error(f"Unexpected error executing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now().isoformat()
            
            self._update_status_file({
                'status': 'task_error',
                'task_id': task.task_id,
                'error': str(e)
            })
        
        finally:
            self.running_process = None

            # タスク完了時のステータス更新
            if self.current_task:
                if self.current_task.status == TaskStatus.COMPLETED:
                    logger.info(f"✅ Task completed successfully: {self.current_task.task_id}")
                    self._update_status_file({
                        'status': 'idle',
                        'last_completed_task': self.current_task.task_id,
                        'completed_at': datetime.now().isoformat()
                    })
                elif self.current_task.status == TaskStatus.FAILED:
                    logger.info(f"❌ Task failed: {self.current_task.task_id}")
                    self._update_status_file({
                        'status': 'idle',
                        'last_failed_task': self.current_task.task_id,
                        'failed_at': datetime.now().isoformat()
                    })

            self.current_task = None
            self._save_queue_state()
    
    def stop_execution(self) -> None:
        """実行停止"""
        logger.info("Stopping execution...")
        self.stop_event.set()
        
        # 実行中のプロセスを終了
        if self.running_process:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.running_process.pid), signal.SIGTERM)
                else:
                    self.running_process.terminate()
                self.running_process.wait(timeout=10)
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        # スレッド終了待機
        if self.executor_thread:
            self.executor_thread.join(timeout=10)
        
        logger.info("Execution stopped")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """キュー状態取得"""
        status = {
            'queue_length': len(self.task_queue),
            'tasks': [task.to_dict() for task in self.task_queue]
        }
        
        if self.current_task:
            status['current_task'] = self.current_task.to_dict()
        
        return status
    
    def cancel_task(self, task_id: str) -> bool:
        """タスクキャンセル"""
        # キュー内のタスクをキャンセル
        for task in self.task_queue:
            if task.task_id == task_id:
                self.task_queue.remove(task)
                task.status = TaskStatus.CANCELLED
                self._save_queue_state()
                logger.info(f"Cancelled task: {task_id}")
                return True
        
        # 実行中のタスクをキャンセル
        if self.current_task and self.current_task.task_id == task_id:
            if self.running_process:
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.running_process.pid), signal.SIGTERM)
                    else:
                        self.running_process.terminate()
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")
            
            self.current_task.status = TaskStatus.CANCELLED
            self.current_task = None
            self._save_queue_state()
            logger.info(f"Cancelled running task: {task_id}")
            return True
        
        return False


def main():
    """CLI実行用メイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python long_task_manager.py <command> [args]")
        print("Commands: enqueue, start, stop, status")
        sys.exit(1)
    
    command = sys.argv[1]
    queue = LongTaskQueue()
    
    if command == "enqueue":
        if len(sys.argv) < 4:
            print("Usage: python long_task_manager.py enqueue <command> <task_type>")
            sys.exit(1)
        
        task_command = sys.argv[2]
        task_type = sys.argv[3]
        task_id = queue.enqueue_task(task_command, task_type)
        print(f"✅ Task enqueued: {task_id}")
    
    elif command == "start":
        queue.start_background_execution()
        print("✅ Background execution started")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            queue.stop_execution()
            print("\n✅ Execution stopped")
    
    elif command == "stop":
        queue.stop_execution()
        print("✅ Execution stopped")
    
    elif command == "status":
        status = queue.get_queue_status()
        print(json.dumps(status, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()