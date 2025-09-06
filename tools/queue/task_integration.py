#!/usr/bin/env python3
"""
タスク統合システム
各種タスクの統合管理とエラーリカバリー
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from long_task_manager import LongTaskQueue
from subagent_monitor import SubAgentMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskIntegration:
    """タスク統合管理システム"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.workspace_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}")
        
        # コンポーネント初期化
        self.queue = LongTaskQueue(str(self.workspace_dir))
        self.monitor = SubAgentMonitor(str(self.workspace_dir))
        
        logger.info(f"TaskIntegration initialized for {tracker_id}")
    
    def check_task_status(self, task_id: str) -> Dict:
        """タスクステータス確認"""
        logger.info(f"Checking task status: {task_id}")
        
        # キューから直接タスク情報を取得
        task = self.queue._get_task(task_id)
        
        if not task:
            return {
                "status": "not_found",
                "message": f"Task {task_id} not found in queue",
                "error": "Task not found"
            }
        
        # プロセス生存確認
        if task["status"] == "running" and task.get("process_pid"):
            if self.queue._check_process_alive(task["process_pid"]):
                elapsed = 0
                if task.get("started_at"):
                    start_time = datetime.fromisoformat(task["started_at"])
                    elapsed = (datetime.now() - start_time).total_seconds()
                
                return {
                    "status": "running",
                    "message": f"Task is running (PID: {task['process_pid']}, elapsed: {elapsed:.0f}s)",
                    "pid": task["process_pid"],
                    "elapsed_seconds": elapsed
                }
            else:
                # プロセスが死んでいる場合は完了とみなす
                logger.info(f"Process {task['process_pid']} not found, marking as completed")
                self.queue._mark_completed(task_id)
                task["status"] = "completed"
        
        # SubAgentモニター情報を追加
        monitor_status = self.monitor.get_status()
        
        logger.info(f"Task status: {task['status']}")
        
        return {
            "status": task["status"],
            "message": self._get_status_message(task),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
            "error": task.get("error_message"),
            "monitor": monitor_status
        }
    
    def _get_status_message(self, task: Dict) -> str:
        """ステータスメッセージ生成"""
        status = task["status"]
        
        if status == "pending":
            return "タスクは待機中です"
        elif status == "running":
            return f"タスク実行中: {task.get('command', 'Unknown')[:100]}"
        elif status == "completed":
            return "タスクが正常に完了しました"
        elif status == "failed":
            return f"タスク失敗: {task.get('error_message', 'Unknown error')}"
        elif status == "timeout":
            return "タスクがタイムアウトしました"
        else:
            return f"Unknown status: {status}"
    
    def collect_task_results(self, task_id: str) -> Dict:
        """タスク結果収集"""
        logger.info(f"Collecting task results: {task_id}")
        
        # タスクステータス確認
        status = self.check_task_status(task_id)
        
        if status["status"] != "completed":
            return {
                "status": "not_ready",
                "message": f"タスクが未完了です。現在の状態: {status['status']}",
                "current_status": status
            }
        
        # 出力ファイル読み込み
        task = self.queue._get_task(task_id)
        output_file = task.get("output_file")
        
        results = {
            "status": "success",
            "message": "結果収集完了",
            "data": {
                "task_id": task_id,
                "completed_at": task.get("completed_at"),
                "output_file": output_file
            }
        }
        
        # 出力ファイルが存在する場合は内容を読み込み
        if output_file and Path(output_file).exists():
            try:
                with open(output_file, 'r') as f:
                    # 最後の100行を取得
                    lines = f.readlines()
                    results["data"]["output_summary"] = "".join(lines[-100:])
                    results["data"]["total_lines"] = len(lines)
            except Exception as e:
                logger.error(f"Failed to read output file: {e}")
                results["data"]["output_error"] = str(e)
        
        # 抽出結果のJSONファイルを確認
        extraction_result = self.workspace_dir / "extraction_result.json"
        if extraction_result.exists():
            try:
                with open(extraction_result, 'r') as f:
                    extraction_data = json.load(f)
                    results["data"]["extraction_summary"] = {
                        "total_images": extraction_data.get("total_images", 0),
                        "successful_extractions": extraction_data.get("successful_extractions", 0),
                        "average_quality_score": extraction_data.get("average_quality_score", 0)
                    }
            except Exception as e:
                logger.error(f"Failed to read extraction result: {e}")
        
        return results
    
    def start_extraction(self, input_dir: str, max_files: int = 10) -> str:
        """抽出タスク開始"""
        logger.info(f"Starting extraction for: {input_dir}")
        
        command = (
            f"sam-env/bin/python3 features/extraction/commands/extract_character.py "
            f"{input_dir} -o {self.workspace_dir}/extraction "
            f"--batch --max-files {max_files} --quality-method balanced --verbose"
        )
        
        task_id = self.queue.enqueue(command, "extract_character")
        
        # モニター開始
        self.monitor.start_monitoring(task_id)
        self.monitor.set_context({
            "tracker_id": self.tracker_id,
            "input_dir": input_dir,
            "max_files": max_files
        })
        
        # 処理開始
        self.queue.process_next()
        
        logger.info(f"Extraction task started: {task_id}")
        return task_id
    
    def start_dashboard_generation(self) -> str:
        """ダッシュボード生成タスク開始"""
        logger.info(f"Starting dashboard generation for: {self.tracker_id}")
        
        command = (
            f"python tools/core/dashboard_generator.py "
            f"--tracker-id {self.tracker_id} "
            f"--with-stats --with-images --markdown-report"
        )
        
        task_id = self.queue.enqueue(command, "dashboard_generation")
        
        # モニター開始
        self.monitor.start_monitoring(task_id)
        self.monitor.set_context({
            "tracker_id": self.tracker_id,
            "task_type": "dashboard"
        })
        
        # 処理開始
        self.queue.process_next()
        
        logger.info(f"Dashboard generation task started: {task_id}")
        return task_id
    
    def reset_all_stuck_tasks(self) -> int:
        """全スタックタスクをリセット"""
        logger.info("Resetting all stuck tasks")
        count = self.queue.reset_stuck_tasks()
        logger.info(f"Reset {count} stuck tasks")
        return count


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python task_integration.py <tracker_id> [command]")
        sys.exit(1)
    
    tracker_id = sys.argv[1]
    integration = TaskIntegration(tracker_id)
    
    if len(sys.argv) >= 3:
        command = sys.argv[2]
        
        if command == "reset":
            count = integration.reset_all_stuck_tasks()
            print(f"Reset {count} stuck tasks")
        elif command == "extraction":
            if len(sys.argv) < 4:
                print("Usage: python task_integration.py <tracker_id> extraction <input_dir>")
                sys.exit(1)
            input_dir = sys.argv[3]
            task_id = integration.start_extraction(input_dir)
            print(f"Started extraction task: {task_id}")
        elif command == "dashboard":
            task_id = integration.start_dashboard_generation()
            print(f"Started dashboard generation: {task_id}")
        else:
            print(f"Unknown command: {command}")
    else:
        # ステータス表示
        status = integration.queue.get_status()
        print(json.dumps(status, indent=2))