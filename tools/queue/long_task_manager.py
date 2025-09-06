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
    
    def _check_process_alive(self, pid: int) -> bool:
        """プロセスが生きているか確認"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False
    
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
    
    def _save_status(self, status: str, task_id: Optional[str] = None, message: str = ""):
        """ステータスを保存"""
        status_data = {
            "status": status,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "queue_length": len([t for t in self.queue_data["tasks"] if t["status"] == "pending"])
        }
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
    
    def get_status(self) -> Dict:
        """現在のステータスを取得"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {"status": "unknown", "message": "No status file found"}
    
    def _mark_completed(self, task_id: str):
        """タスクを完了としてマーク"""
        self._update_task(task_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        })
        self._save_status("task_completed", task_id, "Task completed successfully")
        logger.info(f"Task {task_id} marked as completed")