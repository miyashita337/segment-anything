#!/usr/bin/env python3
"""
SubAgentモニター
長時間実行タスクの監視とClaude Code 2分制限対応
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubAgentMonitor:
    """SubAgent監視システム"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.queue_dir = self.workspace_dir / "queue"
        self.monitor_file = self.queue_dir / "subagent_monitor.json"
        
        # 2分制限対応のタイムアウト
        self.CLAUDE_TIMEOUT = 110  # 1分50秒で警告
        
        self.session_start = time.time()
        self.context = {}
        
        logger.info(f"SubAgentMonitor initialized for {workspace_dir}")
    
    def _save_monitor_state(self, data: Dict):
        """監視状態を保存"""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        with open(self.monitor_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_monitor_state(self) -> Dict:
        """監視状態を読み込み"""
        if self.monitor_file.exists():
            with open(self.monitor_file, 'r') as f:
                return json.load(f)
        return {}
    
    def start_monitoring(self, task_id: str):
        """監視開始"""
        monitor_data = {
            "task_id": task_id,
            "started_at": datetime.now().isoformat(),
            "session_id": f"session_{int(time.time())}",
            "status": "monitoring",
            "context": self.context,
            "checkpoints": []
        }
        self._save_monitor_state(monitor_data)
        logger.info(f"Starting monitoring for task: {task_id}")
    
    def check_timeout(self) -> bool:
        """タイムアウトチェック"""
        elapsed = time.time() - self.session_start
        if elapsed > self.CLAUDE_TIMEOUT:
            logger.warning(f"⚠️ Approaching Claude 2-minute limit: {elapsed:.0f}s elapsed")
            return True
        return False
    
    def add_checkpoint(self, checkpoint_name: str, data: Dict):
        """チェックポイント追加"""
        monitor_data = self._load_monitor_state()
        checkpoint = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        monitor_data.setdefault("checkpoints", []).append(checkpoint)
        self._save_monitor_state(monitor_data)
        logger.info(f"Checkpoint added: {checkpoint_name}")
    
    def get_status(self) -> Dict:
        """現在のステータスを取得"""
        monitor_data = self._load_monitor_state()
        
        # タイムアウトチェック
        if self.check_timeout():
            monitor_data["requires_new_session"] = True
            monitor_data["message"] = "Claude 2分制限に近づいています。新しいセッションが必要です。"
        
        return monitor_data
    
    def set_context(self, context: Dict):
        """コンテキスト設定"""
        self.context = context
        monitor_data = self._load_monitor_state()
        monitor_data["context"] = context
        self._save_monitor_state(monitor_data)
    
    def complete_monitoring(self, task_id: str, status: str = "completed"):
        """監視完了"""
        monitor_data = self._load_monitor_state()
        monitor_data["status"] = status
        monitor_data["completed_at"] = datetime.now().isoformat()
        monitor_data["elapsed_seconds"] = time.time() - self.session_start
        self._save_monitor_state(monitor_data)
        logger.info(f"Monitoring completed for task: {task_id} with status: {status}")


class SubAgentIntegration:
    """SubAgent統合システム"""
    
    def __init__(self):
        self.monitors = {}
        logger.info("SubAgentIntegration initialized")
    
    def create_monitor(self, workspace_dir: str) -> SubAgentMonitor:
        """モニター作成"""
        monitor = SubAgentMonitor(workspace_dir)
        self.monitors[workspace_dir] = monitor
        return monitor
    
    def get_monitor(self, workspace_dir: str) -> Optional[SubAgentMonitor]:
        """モニター取得"""
        return self.monitors.get(workspace_dir)
    
    def set_context(self, tracker_id: str, context: Dict):
        """コンテキスト設定"""
        workspace_dir = f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}"
        monitor = self.get_monitor(workspace_dir)
        if not monitor:
            monitor = self.create_monitor(workspace_dir)
        monitor.set_context(context)
        logger.info(f"Context set: {tracker_id}")
    
    def check_all_timeouts(self) -> Dict[str, bool]:
        """全モニターのタイムアウトチェック"""
        results = {}
        for workspace_dir, monitor in self.monitors.items():
            results[workspace_dir] = monitor.check_timeout()
        return results


if __name__ == "__main__":
    # テスト実行
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python subagent_monitor.py <workspace_dir>")
        sys.exit(1)
    
    workspace_dir = sys.argv[1]
    monitor = SubAgentMonitor(workspace_dir)
    
    # ステータス表示
    status = monitor.get_status()
    print(json.dumps(status, indent=2))