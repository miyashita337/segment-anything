#!/usr/bin/env python3
"""
INTG-087: SubAgent長時間タスクキューシステム
Claude Code SubAgent統合用の長時間タスクキュー・実行制御システム
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil


class SubAgentTaskQueue:
    """SubAgent長時間タスクキューシステム"""
    
    def __init__(self, workspace_path: Path, tracker_id: str):
        """
        SubAgentタスクキューの初期化
        
        Args:
            workspace_path: ワークスペースパス
            tracker_id: トラッカーID
        """
        self.workspace_path = Path(workspace_path)
        self.tracker_id = tracker_id
        
        # キューディレクトリセットアップ
        self.queue_dir = self.workspace_path / ".subagent_queue"
        self.queue_dir.mkdir(exist_ok=True)
        
        # ログ設定
        self.log_dir = self.queue_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # キュー状態ファイル
        self.queue_state_file = self.queue_dir / "queue_state.json"
        self.task_registry_file = self.queue_dir / "task_registry.json"
        
        # 実行制御設定
        self.max_execution_time = 3600  # 1時間
        self.max_memory_usage = 8 * 1024 * 1024 * 1024  # 8GB
        self.task_timeout = 1800  # 30分
        
        self.logger.info(f"SubAgentタスクキュー初期化完了: {tracker_id}")

    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger(f"SubAgentQueue-{self.tracker_id}")
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラー
        log_file = self.log_dir / f"subagent_queue_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger

    def enqueue_task(
        self, 
        task_id: str, 
        command: str, 
        priority: int = 1,
        estimated_duration: int = 300,
        resource_requirements: Optional[Dict] = None
    ) -> bool:
        """
        タスクをキューに追加
        
        Args:
            task_id: タスクID
            command: 実行するコマンド
            priority: 優先度（1=低, 5=高）
            estimated_duration: 予想実行時間（秒）
            resource_requirements: リソース要件
            
        Returns:
            bool: エンキュー成功フラグ
        """
        try:
            # タスク定義
            task = {
                "task_id": task_id,
                "tracker_id": self.tracker_id,
                "command": command,
                "priority": priority,
                "estimated_duration": estimated_duration,
                "resource_requirements": resource_requirements or {},
                "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "output": None,
                "error": None,
                "execution_stats": {}
            }
            
            # タスクレジストリー更新
            registry = self._load_task_registry()
            registry[task_id] = task
            self._save_task_registry(registry)
            
            # キュー状態更新
            queue_state = self._load_queue_state()
            queue_state["queued_tasks"].append(task_id)
            queue_state["total_tasks"] += 1
            self._save_queue_state(queue_state)
            
            self.logger.info(f"タスクエンキュー完了: {task_id} (優先度: {priority})")
            return True
            
        except Exception as e:
            self.logger.error(f"タスクエンキューエラー: {task_id} - {str(e)}")
            return False

    def execute_next_task(self) -> Optional[Dict]:
        """
        キューから次のタスクを実行
        
        Returns:
            Optional[Dict]: 実行結果
        """
        try:
            # キュー状態とレジストリー読み込み
            queue_state = self._load_queue_state()
            registry = self._load_task_registry()
            
            if not queue_state["queued_tasks"]:
                self.logger.info("実行待ちタスクなし")
                return None
            
            # 優先度別ソート
            queued_tasks = queue_state["queued_tasks"]
            task_priorities = [(task_id, registry[task_id]["priority"]) for task_id in queued_tasks]
            task_priorities.sort(key=lambda x: x[1], reverse=True)  # 高優先度から
            
            next_task_id = task_priorities[0][0]
            task = registry[next_task_id]
            
            self.logger.info(f"タスク実行開始: {next_task_id}")
            
            # 実行前チェック
            if not self._pre_execution_check(task):
                self.logger.warning(f"実行前チェック失敗: {next_task_id}")
                return None
            
            # タスク実行
            result = self._execute_task(task)
            
            # 実行結果の保存
            task.update(result)
            registry[next_task_id] = task
            self._save_task_registry(registry)
            
            # キュー状態更新
            queue_state["queued_tasks"].remove(next_task_id)
            if result["status"] == "completed":
                queue_state["completed_tasks"].append(next_task_id)
            else:
                queue_state["failed_tasks"].append(next_task_id)
            
            self._save_queue_state(queue_state)
            
            self.logger.info(f"タスク実行完了: {next_task_id} (ステータス: {result['status']})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"タスク実行エラー: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _execute_task(self, task: Dict) -> Dict:
        """
        実際のタスク実行
        
        Args:
            task: タスク定義
            
        Returns:
            Dict: 実行結果
        """
        start_time = time.time()
        
        try:
            # 実行開始時刻記録
            task["started_at"] = datetime.now(timezone.utc).isoformat()
            task["status"] = "running"
            
            # メモリ使用量監視開始
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # コマンド実行
            self.logger.info(f"コマンド実行: {task['command']}")
            
            result = subprocess.run(
                task["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.task_timeout,
                cwd="/mnt/c/AItools/segment-anything"
            )
            
            # 実行時間計算
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory
            
            # 実行統計
            execution_stats = {
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "cpu_percent": process.cpu_percent(),
                "return_code": result.returncode
            }
            
            # 結果判定
            if result.returncode == 0:
                status = "completed"
                output = result.stdout
                error = None
            else:
                status = "failed"
                output = result.stdout
                error = result.stderr
            
            return {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": output,
                "error": error,
                "execution_stats": execution_stats
            }
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"タスクタイムアウト: {task['task_id']}")
            return {
                "status": "timeout",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": None,
                "error": "Task execution timeout",
                "execution_stats": {
                    "execution_time": time.time() - start_time,
                    "timeout": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"タスク実行例外: {task['task_id']} - {str(e)}")
            return {
                "status": "error",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": None,
                "error": str(e),
                "execution_stats": {
                    "execution_time": time.time() - start_time,
                    "exception": True
                }
            }

    def _pre_execution_check(self, task: Dict) -> bool:
        """
        実行前チェック
        
        Args:
            task: タスク定義
            
        Returns:
            bool: 実行可能フラグ
        """
        try:
            # システムリソースチェック
            memory = psutil.virtual_memory()
            if memory.available < self.max_memory_usage * 0.3:  # 30%以下の空きメモリで警告
                self.logger.warning(f"メモリ不足警告: 使用可能メモリ {memory.available // 1024 // 1024}MB")
            
            # CPU使用率チェック
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"CPU高負荷警告: {cpu_percent}%")
            
            # 依存ファイルチェック（extract_character.pyの場合）
            if "extract_character.py" in task["command"]:
                required_files = [
                    "/mnt/c/AItools/segment-anything/features/extraction/commands/extract_character.py",
                    "/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth"
                ]
                
                for file_path in required_files:
                    if not Path(file_path).exists():
                        self.logger.error(f"必須ファイル不存在: {file_path}")
                        return False
            
            # CUDA利用可能性チェック（GPU処理の場合）
            if "gpu" in task.get("resource_requirements", {}).get("type", ""):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.logger.warning("CUDA利用不可 - CPU処理で実行")
                except ImportError:
                    self.logger.warning("PyTorch未インストール")
            
            return True
            
        except Exception as e:
            self.logger.error(f"実行前チェックエラー: {str(e)}")
            return False

    def get_queue_status(self) -> Dict:
        """
        キューステータス取得
        
        Returns:
            Dict: キュー状況
        """
        try:
            queue_state = self._load_queue_state()
            registry = self._load_task_registry()
            
            # 統計計算
            total_tasks = len(registry)
            queued_count = len(queue_state["queued_tasks"])
            completed_count = len(queue_state["completed_tasks"])
            failed_count = len(queue_state["failed_tasks"])
            
            # 実行時間統計
            completed_tasks = [registry[task_id] for task_id in queue_state["completed_tasks"]]
            avg_execution_time = 0
            if completed_tasks:
                execution_times = [
                    task["execution_stats"].get("execution_time", 0) 
                    for task in completed_tasks
                ]
                avg_execution_time = sum(execution_times) / len(execution_times)
            
            return {
                "tracker_id": self.tracker_id,
                "queue_status": {
                    "total_tasks": total_tasks,
                    "queued": queued_count,
                    "completed": completed_count,
                    "failed": failed_count,
                    "success_rate": completed_count / max(total_tasks, 1) * 100
                },
                "performance": {
                    "average_execution_time": avg_execution_time,
                    "total_execution_time": sum([
                        task["execution_stats"].get("execution_time", 0)
                        for task in registry.values()
                    ])
                },
                "next_task": queue_state["queued_tasks"][0] if queue_state["queued_tasks"] else None
            }
            
        except Exception as e:
            self.logger.error(f"キューステータス取得エラー: {str(e)}")
            return {"error": str(e)}

    def cleanup_completed_tasks(self, keep_days: int = 7) -> int:
        """
        完了タスクのクリーンアップ
        
        Args:
            keep_days: 保持日数
            
        Returns:
            int: クリーンアップしたタスク数
        """
        try:
            registry = self._load_task_registry()
            queue_state = self._load_queue_state()
            
            cutoff_date = datetime.now(timezone.utc).timestamp() - (keep_days * 24 * 3600)
            cleaned_count = 0
            
            # 古い完了タスクを削除
            to_remove = []
            for task_id, task in registry.items():
                if task["status"] == "completed" and task.get("completed_at"):
                    completed_at = datetime.fromisoformat(
                        task["completed_at"].replace('Z', '+00:00')
                    ).timestamp()
                    
                    if completed_at < cutoff_date:
                        to_remove.append(task_id)
            
            # レジストリーから削除
            for task_id in to_remove:
                del registry[task_id]
                if task_id in queue_state["completed_tasks"]:
                    queue_state["completed_tasks"].remove(task_id)
                cleaned_count += 1
            
            # 更新保存
            if cleaned_count > 0:
                self._save_task_registry(registry)
                self._save_queue_state(queue_state)
                
                self.logger.info(f"完了タスククリーンアップ: {cleaned_count}件削除")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {str(e)}")
            return 0

    def _load_queue_state(self) -> Dict:
        """キュー状態読み込み"""
        if self.queue_state_file.exists():
            with open(self.queue_state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "tracker_id": self.tracker_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "queued_tasks": [],
                "completed_tasks": [],
                "failed_tasks": [],
                "total_tasks": 0
            }

    def _save_queue_state(self, state: Dict) -> None:
        """キュー状態保存"""
        with open(self.queue_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_task_registry(self) -> Dict:
        """タスクレジストリー読み込み"""
        if self.task_registry_file.exists():
            with open(self.task_registry_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def _save_task_registry(self, registry: Dict) -> None:
        """タスクレジストリー保存"""
        with open(self.task_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)


class ExtractCharacterTaskValidator:
    """extract_character.py 直接実行防止システム"""
    
    @staticmethod
    def check_subagent_execution(command: str) -> Tuple[bool, str]:
        """
        SubAgent実行必須チェック
        
        Args:
            command: 実行されるコマンド
            
        Returns:
            Tuple[bool, str]: (実行許可, メッセージ)
        """
        if "extract_character.py" in command:
            # 直接実行かSubAgent経由かの判定
            if not ExtractCharacterTaskValidator._is_subagent_execution():
                return False, (
                    "❌ extract_character.py の直接実行は禁止されています。\n"
                    "SubAgentキューシステム経由で実行してください。\n"
                    "\n"
                    "🔧 正しい実行方法:\n"
                    "1. python tools/queue/subagent_wrapper.py enqueue extract_character\n"
                    "2. python tools/queue/subagent_wrapper.py execute\n"
                    "\n"
                    "詳細: docs/claude-code-hooks-guide.md を参照"
                )
            
        return True, "実行許可"

    @staticmethod
    def _is_subagent_execution() -> bool:
        """SubAgent実行環境の検出"""
        # 環境変数による検出
        subagent_env = [
            "SUBAGENT_TASK_ID",
            "SUBAGENT_EXECUTION",
            "CLAUDE_SUBAGENT_MODE"
        ]
        
        import os
        for env_var in subagent_env:
            if os.getenv(env_var):
                return True
        
        # プロセス階層による検出
        try:
            current_process = psutil.Process()
            for parent in current_process.parents():
                if "subagent" in parent.name().lower():
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return False


def main():
    """メイン実行フロー"""
    if len(sys.argv) < 2:
        print("使用方法: python subagent_wrapper.py <command> [args...]")
        print("コマンド:")
        print("  enqueue <task_name> <command>  - タスクをキューに追加")
        print("  execute                       - キューから次のタスクを実行")
        print("  status                        - キューステータス表示")
        print("  cleanup [days]                - 完了タスククリーンアップ")
        return
    
    command = sys.argv[1]
    
    # デフォルトワークスペース（テスト用）
    workspace_path = Path("/tmp/subagent_test")
    tracker_id = "SUBAGENT-TEST"
    
    # 実際の実行時はコマンドライン引数または環境変数から取得
    import os
    if os.getenv("TRACKER_WORKSPACE"):
        workspace_path = Path(os.getenv("TRACKER_WORKSPACE"))
    if os.getenv("TRACKER_ID"):
        tracker_id = os.getenv("TRACKER_ID")
    
    queue = SubAgentTaskQueue(workspace_path, tracker_id)
    
    if command == "enqueue":
        if len(sys.argv) < 4:
            print("使用方法: python subagent_wrapper.py enqueue <task_name> <command>")
            return
        
        task_name = sys.argv[2]
        task_command = " ".join(sys.argv[3:])
        
        success = queue.enqueue_task(
            task_id=f"{tracker_id}_{task_name}_{int(time.time())}",
            command=task_command,
            priority=3
        )
        
        if success:
            print(f"✅ タスクエンキュー成功: {task_name}")
        else:
            print(f"❌ タスクエンキュー失敗: {task_name}")
    
    elif command == "execute":
        result = queue.execute_next_task()
        if result:
            print(f"✅ タスク実行完了: {result['status']}")
            if result.get('output'):
                print(f"出力: {result['output'][:500]}...")
        else:
            print("⚠️ 実行可能なタスクなし")
    
    elif command == "status":
        status = queue.get_queue_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    elif command == "cleanup":
        keep_days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        cleaned = queue.cleanup_completed_tasks(keep_days)
        print(f"✅ クリーンアップ完了: {cleaned}件削除")
    
    else:
        print(f"❌ 不明なコマンド: {command}")


if __name__ == "__main__":
    main()