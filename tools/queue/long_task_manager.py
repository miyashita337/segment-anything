#!/usr/bin/env python3
"""
長時間タスクキューマネージャー
QUAL-044 SubAgent統合システムの中核実装
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTaskQueue:
    """長時間タスクのキュー管理システム"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.queue_dir = self.workspace_dir / "queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self.queue_file = self.queue_dir / "task_queue.json"
        self.status_file = self.queue_dir / "queue_status.json"
        
        # タイムアウト設定（1時間）
        self.MAX_EXECUTION_TIME = 3600
        
        self._load_queue()
        logger.info(f"LongTaskQueue initialized at {self.workspace_dir}")
    
    def _load_queue(self):
        """キューを読み込み"""
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                self.queue_data = json.load(f)
                logger.info(f"Loaded {len(self.queue_data.get('tasks', []))} tasks from queue")
        else:
            self.queue_data = {"tasks": []}
            self._save_queue()
    
    def _save_queue(self):
        """キューを保存"""
        with open(self.queue_file, 'w') as f:
            json.dump(self.queue_data, f, indent=2, ensure_ascii=False)
    
    def _save_status(self, status: str, task_id: Optional[str] = None, message: str = ""):
        """ステータスを保存"""
        status_data = {
            "status": status,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "queue_length": len([t for t in self.queue_data["tasks"] if t["status"] == "pending"])
        }
        
        if task_id:
            task = self._get_task(task_id)
            if task:
                status_data["current_task"] = task
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
    
    def _get_task(self, task_id: str) -> Optional[Dict]:
        """タスクを取得"""
        for task in self.queue_data["tasks"]:
            if task["task_id"] == task_id:
                return task
        return None
    
    def _update_task(self, task_id: str, updates: Dict):
        """タスクを更新"""
        for task in self.queue_data["tasks"]:
            if task["task_id"] == task_id:
                task.update(updates)
                self._save_queue()
                return True
        return False
    
    def _check_process_alive(self, pid: int) -> bool:
        """プロセスが生きているか確認"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False
    
    def enqueue(self, command: str, task_type: str = "generic") -> str:
        """タスクをキューに追加"""
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = {
            "task_id": task_id,
            "command": command,
            "task_type": task_type,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "max_retries": 2,
            "output_file": None,
            "error_message": None,
            "process_pid": None
        }
        
        self.queue_data["tasks"].append(task)
        self._save_queue()
        self._save_status("task_enqueued", task_id, f"Task {task_id} added to queue")
        
        logger.info(f"Enqueued task: {task_id}")
        return task_id
    
    def process_next(self) -> bool:
        """次のタスクを処理"""
        # 実行中のタスクをチェック
        for task in self.queue_data["tasks"]:
            if task["status"] == "running":
                # プロセスが生きているか確認
                if task.get("process_pid") and self._check_process_alive(task["process_pid"]):
                    # タイムアウトチェック
                    if task.get("started_at"):
                        start_time = datetime.fromisoformat(task["started_at"])
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > self.MAX_EXECUTION_TIME:
                            logger.warning(f"Task {task['task_id']} timed out after {elapsed:.0f} seconds")
                            self._terminate_task(task)
                        else:
                            logger.info(f"Task {task['task_id']} still running (PID: {task['process_pid']})")
                            return False
                else:
                    # プロセスが死んでいる場合は完了とみなす
                    logger.info(f"Task {task['task_id']} process not found, marking as completed")
                    self._mark_completed(task["task_id"])
        
        # 次のpendingタスクを探す
        for task in self.queue_data["tasks"]:
            if task["status"] == "pending":
                return self._execute_task(task)
        
        logger.info("No pending tasks in queue")
        self._save_status("idle", None, "Queue is empty")
        return False
    
    def _execute_task(self, task: Dict) -> bool:
        """タスクを実行"""
        task_id = task["task_id"]
        logger.info(f"Executing task: {task_id}")
        
        try:
            # 出力ファイルパス
            output_file = self.queue_dir / f"{task_id}_output.log"
            
            # タスク開始
            self._update_task(task_id, {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "output_file": str(output_file)
            })
            
            # プロセス起動
            with open(output_file, 'w') as f:
                process = subprocess.Popen(
                    task["command"],
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd())
                )
            
            # PID保存
            self._update_task(task_id, {"process_pid": process.pid})
            self._save_status("task_running", task_id, f"Executing: {task['command'][:100]}")
            
            logger.info(f"Started process PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            self._update_task(task_id, {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat()
            })
            self._save_status("task_failed", task_id, str(e))
            return False
    
    def _terminate_task(self, task: Dict):
        """タスクを強制終了"""
        task_id = task["task_id"]
        pid = task.get("process_pid")
        
        if pid:
            try:
                process = psutil.Process(pid)
                process.terminate()
                time.sleep(2)
                if process.is_running():
                    process.kill()
                logger.info(f"Terminated task {task_id} (PID: {pid})")
            except:
                pass
        
        self._update_task(task_id, {
            "status": "timeout",
            "completed_at": datetime.now().isoformat(),
            "error_message": "Task exceeded maximum execution time"
        })
        self._save_status("task_timeout", task_id, "Task timed out")
    
    def _mark_completed(self, task_id: str):
        """タスクを完了としてマーク"""
        self._update_task(task_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        })
        self._save_status("task_completed", task_id, "Task completed successfully")
        logger.info(f"Task {task_id} marked as completed")
    
    def get_status(self) -> Dict:
        """現在のステータスを取得"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {"status": "unknown", "message": "No status file found"}
    
    def reset_stuck_tasks(self):
        """スタックしたタスクをリセット"""
        reset_count = 0
        for task in self.queue_data["tasks"]:
            if task["status"] == "running":
                # プロセスが存在しない場合はリセット
                if not task.get("process_pid") or not self._check_process_alive(task["process_pid"]):
                    logger.info(f"Resetting stuck task: {task['task_id']}")
                    self._update_task(task["task_id"], {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "error_message": "Task was stuck and has been reset"
                    })
                    reset_count += 1
        
        if reset_count > 0:
            logger.info(f"Reset {reset_count} stuck tasks")
            self._save_status("reset_complete", None, f"Reset {reset_count} stuck tasks")
        return reset_count


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python long_task_manager.py <workspace_dir> [command]")
        sys.exit(1)
    
    workspace_dir = sys.argv[1]
    queue = LongTaskQueue(workspace_dir)
    
    if len(sys.argv) >= 3:
        command = sys.argv[2]
        if command == "status":
            print(json.dumps(queue.get_status(), indent=2))
        elif command == "reset":
            count = queue.reset_stuck_tasks()
            print(f"Reset {count} stuck tasks")
        elif command == "process":
            queue.process_next()
        else:
            # コマンドとして実行
            task_id = queue.enqueue(" ".join(sys.argv[2:]))
            print(f"Enqueued task: {task_id}")
    else:
        # デフォルトはステータス表示
        print(json.dumps(queue.get_status(), indent=2))