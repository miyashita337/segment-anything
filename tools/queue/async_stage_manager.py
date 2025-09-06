#!/usr/bin/env python3
"""
非同期ステージマネージャー
QUAL-044 SubAgent統合システムのステージ管理
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 同じディレクトリからインポート
from long_task_manager import LongTaskQueue
from subagent_monitor import SubAgentMonitor
from task_integration import TaskIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncStageManager:
    """非同期ステージ管理システム"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.workspace_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage_file = self.workspace_dir / "async_stage_status.json"
        
        # コンポーネント初期化
        self.queue = LongTaskQueue(str(self.workspace_dir))
        self.monitor = SubAgentMonitor(str(self.workspace_dir))
        self.integration = TaskIntegration(tracker_id)
        
        logger.info(f"AsyncStageManager initialized for {tracker_id}")
        logger.info(f"Workspace: {self.workspace_dir}")
    
    def _load_stage_status(self) -> Dict:
        """ステージステータスを読み込み"""
        if self.stage_file.exists():
            with open(self.stage_file, 'r') as f:
                return json.load(f)
        return {
            "stage": "init",
            "status": "not_started",
            "tracker_id": self.tracker_id,
            "last_updated": datetime.now().timestamp()
        }
    
    def _save_stage_status(self, data: Dict):
        """ステージステータスを保存"""
        data["last_updated"] = datetime.now().timestamp()
        with open(self.stage_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register_task(self, input_dir: str, task_type: str = "extraction") -> str:
        """タスクを登録（Stage 1）"""
        logger.info("=" * 50)
        logger.info("📝 Stage 1: Task Registration")
        logger.info("=" * 50)
        
        # 既存のスタックタスクをリセット
        self.queue.reset_stuck_tasks()
        
        # コマンド生成
        if task_type == "extraction":
            command = (
                f"sam-env/bin/python3 features/extraction/commands/extract_character.py "
                f"{input_dir} -o {self.workspace_dir}/extraction "
                f"--batch --max-files 10 --quality-method balanced --verbose"
            )
        else:
            command = f"echo 'Unknown task type: {task_type}'"
        
        # タスク登録
        task_id = self.queue.enqueue(command, task_type)
        
        # ステージ状態更新
        stage_data = {
            "stage": "register",
            "status": "registered",
            "message": f"{task_type}タスク登録完了・バックグラウンド処理開始",
            "tracker_id": self.tracker_id,
            "task_id": task_id,
            "task_type": task_type,
            "background_running": True,
            "workspace": str(self.workspace_dir),
            "input_dir": input_dir
        }
        self._save_stage_status(stage_data)
        
        logger.info(f"✅ Task registered: {task_id}")
        logger.info(f"📂 Input: {input_dir}")
        logger.info(f"📂 Output: {self.workspace_dir}/extraction")
        
        # 処理開始
        self.queue.process_next()
        
        return task_id
    
    def monitor_task(self) -> Dict:
        """タスクを監視（Stage 2）"""
        logger.info("=" * 50)
        logger.info("👁️ Stage 2: Task Monitoring")
        logger.info("=" * 50)
        
        stage_data = self._load_stage_status()
        task_id = stage_data.get("task_id")
        
        if not task_id:
            return {
                "stage": "monitor",
                "status": "error",
                "message": "No task ID found in stage status",
                "tracker_id": self.tracker_id
            }
        
        logger.info(f"🔍 Monitoring task: {task_id}")
        
        # タスク状態確認
        status = self.integration.check_task_status(task_id)
        
        logger.info(f"📊 Current status: {status['status']}")
        logger.info(f"💬 Message: {status['message']}")
        
        if status["status"] == "completed":
            logger.info("✅ Task completed successfully")
            stage_data["stage"] = "monitor"
            stage_data["status"] = "completed"
            stage_data["message"] = "タスク完了・結果収集可能"
        elif status["status"] == "failed":
            logger.info("❌ Task failed")
            stage_data["stage"] = "monitor" 
            stage_data["status"] = "failed"
            stage_data["message"] = f"タスク失敗: {status.get('error', 'Unknown error')}"
        else:
            logger.info("⏳ Task is still running in background")
            logger.info("🔄 Continue monitoring with next command")
            stage_data["stage"] = "monitor"
            stage_data["status"] = status["status"]
            stage_data["message"] = status["message"]
        
        # SubAgent監視データを追加
        monitor_status = self.monitor.get_status()
        stage_data["current_monitor"] = monitor_status
        
        self._save_stage_status(stage_data)
        return stage_data
    
    def collect_results(self) -> Dict:
        """結果を収集（Stage 3）"""
        logger.info("=" * 50)
        logger.info("📊 Stage 3: Result Collection")
        logger.info("=" * 50)
        
        stage_data = self._load_stage_status()
        task_id = stage_data.get("task_id")
        
        if not task_id:
            return {
                "stage": "collect",
                "status": "error",
                "message": "No task ID found",
                "tracker_id": self.tracker_id
            }
        
        logger.info(f"📋 Collecting results for: {task_id}")
        
        # 結果収集
        results = self.integration.collect_task_results(task_id)
        
        if results["status"] == "success":
            logger.info("✅ Results collected successfully")
            stage_data["stage"] = "collect"
            stage_data["status"] = "completed"
            stage_data["message"] = "結果収集完了"
            stage_data["results"] = results["data"]
        else:
            logger.info(f"❌ Failed to collect results: {results.get('error')}")
            stage_data["stage"] = "collect"
            stage_data["status"] = results["status"]
            stage_data["message"] = results.get("message", "結果収集失敗")
        
        self._save_stage_status(stage_data)
        return stage_data
    
    def get_status(self) -> Dict:
        """現在のステータスを取得"""
        return self._load_stage_status()


def main():
    """メインエントリーポイント"""
    if len(sys.argv) < 3:
        print("Usage: python async_stage_manager.py <command> <tracker_id> [options]")
        print("Commands: register, monitor, collect, status")
        sys.exit(1)
    
    command = sys.argv[1]
    tracker_id = sys.argv[2]
    
    manager = AsyncStageManager(tracker_id)
    
    if command == "register":
        if len(sys.argv) < 4:
            print("Usage: python async_stage_manager.py register <tracker_id> <input_dir>")
            sys.exit(1)
        input_dir = sys.argv[3]
        task_id = manager.register_task(input_dir)
        print(json.dumps({
            "stage": "register",
            "status": "success",
            "task_id": task_id,
            "tracker_id": tracker_id
        }, indent=2))
    
    elif command == "monitor":
        result = manager.monitor_task()
        print(json.dumps(result, indent=2))
    
    elif command == "collect":
        result = manager.collect_results()
        print(json.dumps(result, indent=2))
    
    elif command == "status":
        status = manager.get_status()
        stage_status = manager._load_stage_status()
        
        result = {
            "tracker_id": tracker_id,
            "workspace": str(manager.workspace_dir),
            "stage_file": str(manager.stage_file),
            "full_status": stage_status,
            "last_updated": stage_status.get("last_updated", 0)
        }
        
        # 次の推奨アクション
        current_stage = stage_status.get("stage", "init")
        current_status = stage_status.get("status", "unknown")
        
        if current_stage == "init" or current_status == "not_started":
            result["next_recommended_action"] = "register: タスクを登録してください"
        elif current_stage == "register" and current_status == "registered":
            result["next_recommended_action"] = "monitor: タスクの状態を監視してください"
        elif current_stage == "monitor" and current_status == "completed":
            result["next_recommended_action"] = "collect: 結果を収集してください"
        elif current_stage == "collect" and current_status == "completed":
            result["next_recommended_action"] = "完了: すべてのステージが完了しました"
        else:
            result["next_recommended_action"] = f"{current_stage}: 現在の状態を確認してください"
        
        print(json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()