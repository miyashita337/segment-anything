#!/usr/bin/env python3
"""
INTG-087: SubAgent長時間タスクキューシステム
Claude Code SubAgent統合用の長時間タスクキュー・実行制御システム
"""

import json
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
        self.running_tasks_file = self.queue_dir / "running_tasks.json"
        
        # 実行制御設定
        self.max_execution_time = 3600  # 1時間
        self.max_memory_usage = 8 * 1024 * 1024 * 1024  # 8GB
        self.task_timeout = 1800  # 30分

        # INTG-089: 拡張機能設定
        self.checkpoint_dir = self.queue_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.gpu_fallback_enabled = True
        self.max_gpu_retry = 3
        self.checkpoint_interval = 300  # 5分間隔
        self.auto_checkpoint_enabled = True  # 自動チェックポイント有効
        self.progress_monitors = {}  # 進捗監視辞書
        self.checkpoint_metadata = {}  # チェックポイントメタデータ
        self.auto_recovery_enabled = True

        # GPUフォールバック統計
        self.gpu_fallback_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'gpu_healthy': True,
            'last_gpu_check': 0,
            'gpu_errors': [],
            'cpu_fallback_count': 0,
            'memory_fallback_count': 0
        }

        self.logger.info(f"SubAgentタスクキュー初期化完了: {tracker_id}")
        self.logger.info("INTG-089: Enhanced checkpoint and GPU fallback enabled")

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
            
            # コマンド実行（Popenでプロセス管理強化）
            self.logger.info(f"コマンド実行: {task['command']}")

            process = subprocess.Popen(
                task["command"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/mnt/c/AItools/segment-anything"
            )

            # プロセスID記録
            task["pid"] = process.pid
            self._save_running_task(task)
            
            # 自動チェックポイント開始
            if self.auto_checkpoint_enabled:
                self._start_progress_monitoring(task["task_id"], process.pid)

            # プロセス実行完了待機
            try:
                stdout, stderr = process.communicate(timeout=self.task_timeout)
                result_code = process.returncode
            except subprocess.TimeoutExpired:
                # タイムアウト時は安全に停止
                self._terminate_process(process.pid)
                stdout, stderr = process.communicate()
                result_code = -1
                
            # 進捗監視終了
            if self.auto_checkpoint_enabled and task["task_id"] in self.progress_monitors:
                self._stop_progress_monitoring(task["task_id"])
            
            # 実行時間計算
            execution_time = time.time() - start_time

            # メモリ情報取得（現在のプロセス）
            try:
                current_proc = psutil.Process()
                final_memory = current_proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                final_memory = initial_memory
            memory_delta = final_memory - initial_memory
            
            # 実行統計
            execution_stats = {
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "cpu_percent": current_proc.cpu_percent() if 'current_proc' in locals() else 0.0,
                "return_code": result_code
            }
            
            # 結果判定
            if result_code == 0:
                status = "completed"
                output = stdout
                error = None
            else:
                status = "failed"
                output = stdout
                error = stderr

            # 実行完了後はrunning_tasksから削除
            self._remove_running_task(task["task_id"])
            
            return {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": output,
                "error": error,
                "execution_stats": execution_stats
            }
            
            
        except Exception as e:
            self.logger.error(f"タスク実行例外: {task['task_id']} - {str(e)}")
            # エラー時も実行中タスクから削除
            self._remove_running_task(task["task_id"])
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

    def kill_task(self, task_id: str) -> bool:
        """
        指定したタスクを停止

        Args:
            task_id: 停止するタスクID

        Returns:
            bool: 停止成功フラグ
        """
        try:
            running_tasks = self._load_running_tasks()

            if task_id not in running_tasks:
                self.logger.warning(f"実行中タスクが見つかりません: {task_id}")
                return False

            task = running_tasks[task_id]
            pid = task.get("pid")

            if not pid:
                self.logger.error(f"タスクにプロセスIDが記録されていません: {task_id}")
                return False

            # プロセス停止実行
            success = self._terminate_process(pid)

            if success:
                # 停止後の後処理
                self._remove_running_task(task_id)

                # レジストリー更新
                registry = self._load_task_registry()
                if task_id in registry:
                    registry[task_id]["status"] = "killed"
                    registry[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    registry[task_id]["error"] = "Task killed by user"
                    self._save_task_registry(registry)

                self.logger.info(f"タスク停止成功: {task_id} (PID: {pid})")
                return True

            return False

        except Exception as e:
            self.logger.error(f"タスク停止エラー: {task_id} - {str(e)}")
            return False

    def kill_all_tasks(self) -> int:
        """
        全実行中タスクを停止

        Returns:
            int: 停止したタスク数
        """
        try:
            running_tasks = self._load_running_tasks()
            killed_count = 0

            for task_id in list(running_tasks.keys()):
                if self.kill_task(task_id):
                    killed_count += 1

            self.logger.info(f"全タスク停止完了: {killed_count}件停止")
            return killed_count

        except Exception as e:
            self.logger.error(f"全タスク停止エラー: {str(e)}")
            return 0

    def list_running_tasks(self) -> Dict:
        """
        実行中タスク一覧取得

        Returns:
            Dict: 実行中タスク情報
        """
        try:
            running_tasks = self._load_running_tasks()

            # 実際のプロセス存在確認
            active_tasks = {}

            for task_id, task in running_tasks.items():
                pid = task.get("pid")
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        task["cpu_percent"] = proc.cpu_percent()
                        task["memory_mb"] = proc.memory_info().rss // 1024 // 1024
                        task["status_detail"] = proc.status()
                        active_tasks[task_id] = task
                    except psutil.NoSuchProcess:
                        # プロセス終了済みの場合は削除
                        self._remove_running_task(task_id)
                else:
                    # PID不正または存在しない場合は削除
                    self._remove_running_task(task_id)

            return {
                "running_count": len(active_tasks),
                "tasks": active_tasks
            }

        except Exception as e:
            self.logger.error(f"実行中タスク一覧取得エラー: {str(e)}")
            return {"error": str(e)}

    def _terminate_process(self, pid: int) -> bool:
        """
        プロセスを安全に停止

        Args:
            pid: プロセスID

        Returns:
            bool: 停止成功フラグ
        """
        try:
            if not psutil.pid_exists(pid):
                self.logger.warning(f"プロセスが既に終了しています: PID {pid}")
                return True

            proc = psutil.Process(pid)

            # 1. SIGTERM送信（正常終了要求）
            self.logger.info(f"プロセス正常停止要求: PID {pid}")
            proc.terminate()

            # 2. 5秒間正常終了を待機
            try:
                proc.wait(timeout=5)
                self.logger.info(f"プロセス正常停止完了: PID {pid}")
                return True
            except psutil.TimeoutExpired:
                pass

            # 3. SIGKILL送信（強制終了）
            if proc.is_running():
                self.logger.warning(f"プロセス強制停止実行: PID {pid}")
                proc.kill()
                try:
                    proc.wait(timeout=3)
                    self.logger.info(f"プロセス強制停止完了: PID {pid}")
                except psutil.TimeoutExpired:
                    self.logger.error(f"プロセス強制停止もタイムアウト: PID {pid}")
                    # それでもTrueを返す（プロセス管理の一貫性のため）

            return True

        except psutil.NoSuchProcess:
            self.logger.info(f"プロセス既に終了: PID {pid}")
            return True
        except Exception as e:
            self.logger.error(f"プロセス停止エラー: PID {pid} - {str(e)}")
            return False

    def _save_running_task(self, task: Dict) -> None:
        """実行中タスク記録"""
        try:
            running_tasks = self._load_running_tasks()
            running_tasks[task["task_id"]] = {
                "task_id": task["task_id"],
                "pid": task.get("pid"),
                "command": task["command"],
                "started_at": task["started_at"],
                "status": "running"
            }

            with open(self.running_tasks_file, 'w') as f:
                json.dump(running_tasks, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"実行中タスク記録エラー: {str(e)}")

    def _remove_running_task(self, task_id: str) -> None:
        """実行中タスク削除"""
        try:
            running_tasks = self._load_running_tasks()
            if task_id in running_tasks:
                del running_tasks[task_id]

                with open(self.running_tasks_file, 'w') as f:
                    json.dump(running_tasks, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"実行中タスク削除エラー: {str(e)}")

    def _load_running_tasks(self) -> Dict:
        """実行中タスク読み込み"""
        if self.running_tasks_file.exists():
            try:
                with open(self.running_tasks_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"実行中タスク読み込みエラー: {str(e)}")
        return {}

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


# INTG-089: SubAgentTaskQueue拡張機能
class EnhancedSubAgentTaskQueue(SubAgentTaskQueue):
    """INTG-089拡張版SubAgentTaskQueue"""

    def __init__(self, workspace_path: Path, tracker_id: str):
        """
        拡張SubAgentTaskQueueの初期化

        Args:
            workspace_path: ワークスペースパス
            tracker_id: トラッカーID
        """
        # 親クラス初期化
        super().__init__(workspace_path, tracker_id)

        # INTG-089: 拡張機能初期化
        self.checkpoint_dir = self.queue_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # チェックポイント関連設定
        self.checkpoint_interval = 300  # 5分間隔
        self.auto_checkpoint_enabled = True
        self.last_checkpoint_time: Dict[str, float] = {}

        # GPU Fallback機能
        self.gpu_fallback_enabled = True
        self.max_gpu_retry = 3
        self.gpu_fallback_stats = {
            'gpu_attempts': 0,
            'cpu_fallbacks': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }

        # 自動復旧機能
        self.auto_recovery_enabled = True
        self.max_recovery_attempts = 3

        self.logger.info("EnhancedSubAgentTaskQueue initialized with INTG-089 extensions")

    def save_checkpoint(self, task_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        チェックポイント保存

        Args:
            task_id: タスクID
            progress_data: 進捗データ

        Returns:
            bool: 保存成功フラグ
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"

            checkpoint = {
                'task_id': task_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'progress_data': progress_data,
                'tracker_id': self.tracker_id
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            self.logger.info(f"Checkpoint saved: {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {task_id} - {e}")
            return False

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        チェックポイント読み込み

        Args:
            task_id: タスクID

        Returns:
            Optional[Dict[str, Any]]: チェックポイントデータ
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"
            if not checkpoint_file.exists():
                return None

            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            self.logger.info(f"Checkpoint loaded: {task_id}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {task_id} - {e}")
            return None

    def auto_checkpoint_if_needed(self, task_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        必要に応じて自動チェックポイント保存

        Args:
            task_id: タスクID
            progress_data: 進捗データ

        Returns:
            bool: チェックポイント実行フラグ
        """
        if not self.auto_checkpoint_enabled:
            return False

        current_time = time.time()
        last_checkpoint = self.last_checkpoint_time.get(task_id, 0)

        # チェックポイント間隔チェック
        if current_time - last_checkpoint >= self.checkpoint_interval:
            success = self.save_checkpoint(task_id, progress_data)
            if success:
                self.last_checkpoint_time[task_id] = current_time
                self.logger.info(f"Auto checkpoint saved for task: {task_id}")
            return success

        return False

    def checkpoint_task_progress(self, task_id: str, current_step: int, total_steps: int,
                               additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        タスク進捗の標準チェックポイント保存

        Args:
            task_id: タスクID
            current_step: 現在のステップ
            total_steps: 総ステップ数
            additional_data: 追加データ

        Returns:
            bool: 保存成功フラグ
        """
        progress_data = {
            'current_step': current_step,
            'total_steps': total_steps,
            'progress_percent': round((current_step / total_steps) * 100, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checkpoint_reason': 'progress_update'
        }

        if additional_data:
            progress_data['additional_data'] = additional_data

        # 自動チェックポイント判定
        return self.auto_checkpoint_if_needed(task_id, progress_data)

    def cleanup_old_checkpoints(self, keep_days: int = 7) -> int:
        """
        古いチェックポイントのクリーンアップ

        Args:
            keep_days: 保持日数

        Returns:
            int: クリーンアップされたファイル数
        """
        try:
            cutoff_time = time.time() - (keep_days * 24 * 3600)
            cleaned_count = 0

            for checkpoint_file in self.checkpoint_dir.glob('*_checkpoint.json'):
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"Cleaned old checkpoint: {checkpoint_file.name}")

            if cleaned_count > 0:
                self.logger.info(f"Cleaned {cleaned_count} old checkpoint files")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0

    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """
        チェックポイント統計情報取得

        Returns:
            dict: チェックポイント統計
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob('*_checkpoint.json'))
            total_size = sum(f.stat().st_size for f in checkpoint_files)

            stats = {
                'total_checkpoints': len(checkpoint_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'auto_checkpoint_enabled': self.auto_checkpoint_enabled,
                'checkpoint_interval_seconds': self.checkpoint_interval,
                'active_task_checkpoints': len(self.last_checkpoint_time),
                'checkpoint_directory': str(self.checkpoint_dir)
            }

            # 最新・最古のチェックポイント情報
            if checkpoint_files:
                timestamps = [f.stat().st_mtime for f in checkpoint_files]
                stats['latest_checkpoint'] = datetime.fromtimestamp(max(timestamps)).isoformat()
                stats['oldest_checkpoint'] = datetime.fromtimestamp(min(timestamps)).isoformat()

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get checkpoint statistics: {e}")
            return {'error': str(e)}

    def execute_with_integrated_features(self, task: Dict) -> Dict:
        """
        統合機能付きタスク実行（チェックポイント+GPU fallback+自動復旧）

        Args:
            task: タスク定義

        Returns:
            Dict: 実行結果
        """
        task_id = task['task_id']
        start_time = time.time()

        self.logger.info(f"Starting integrated task execution: {task_id}")

        try:
            # 1. チェックポイントからの復旧確認
            checkpoint = self.load_checkpoint(task_id)
            if checkpoint and self.auto_recovery_enabled:
                self.logger.info(f"Resuming task from checkpoint: {task_id}")
                # チェックポイントデータをタスクに統合
                task['resume_data'] = checkpoint.get('progress_data', {})

            # 2. GPU fallback付きタスク実行
            result = self.execute_with_gpu_fallback(task)

            # 3. 実行中のチェックポイント保存（成功時）
            if result.get('status') == 'completed':
                final_checkpoint = {
                    'execution_completed': True,
                    'result': result,
                    'completion_time': datetime.now(timezone.utc).isoformat()
                }
                self.save_checkpoint(task_id, final_checkpoint)

            # 4. 統計更新
            self._update_integrated_statistics(result, start_time)

            return result

        except Exception as e:
            # エラー時のチェックポイント保存
            error_checkpoint = {
                'execution_failed': True,
                'error': str(e),
                'failure_time': datetime.now(timezone.utc).isoformat(),
                'recovery_needed': True
            }
            self.save_checkpoint(task_id, error_checkpoint)

            self.logger.error(f"Integrated task execution failed: {task_id} - {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'task_id': task_id,
                'execution_time': time.time() - start_time
            }

    def _update_integrated_statistics(self, result: Dict, start_time: float) -> None:
        """
        統合実行統計の更新

        Args:
            result: 実行結果
            start_time: 開始時間
        """
        execution_time = time.time() - start_time

        if not hasattr(self, 'integrated_stats'):
            self.integrated_stats = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_execution_time': 0.0,
                'checkpoint_resumes': 0,
                'gpu_fallback_uses': 0
            }

        self.integrated_stats['total_executions'] += 1

        if result.get('status') == 'completed':
            self.integrated_stats['successful_executions'] += 1
        else:
            self.integrated_stats['failed_executions'] += 1

        # 平均実行時間の更新
        total_time = (self.integrated_stats['average_execution_time'] *
                     (self.integrated_stats['total_executions'] - 1) + execution_time)
        self.integrated_stats['average_execution_time'] = total_time / self.integrated_stats['total_executions']

        # GPU fallback使用回数
        if result.get('execution_stats', {}).get('gpu_fallback_used'):
            self.integrated_stats['gpu_fallback_uses'] += 1

    def get_integrated_system_status(self) -> Dict[str, Any]:
        """
        統合システム状態取得

        Returns:
            Dict[str, Any]: システム状態情報
        """
        status = {
            'enhanced_features_enabled': True,
            'checkpoint_system': self.get_checkpoint_statistics(),
            'gpu_fallback_stats': dict(self.gpu_fallback_stats),
            'integrated_stats': getattr(self, 'integrated_stats', {}),
            'system_health': {
                'auto_checkpoint_enabled': self.auto_checkpoint_enabled,
                'gpu_fallback_enabled': self.gpu_fallback_enabled,
                'auto_recovery_enabled': self.auto_recovery_enabled,
                'max_gpu_retry': self.max_gpu_retry,
                'checkpoint_interval': self.checkpoint_interval
            },
            'active_monitoring': {
                'active_tasks': len(self.last_checkpoint_time),
                'checkpoint_directory_size_mb': self.get_checkpoint_statistics().get('total_size_mb', 0)
            }
        }

        return status

    def _check_gpu_available(self) -> bool:
        """
        GPU利用可能性チェック（統合システム用）

        Returns:
            bool: GPU利用可能フラグ
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# INTG-089: 3システム統合クラス
class IntegratedSubAgentSystem:
    """
    SubAgentMonitor + NotificationBridge + EnhancedSubAgentTaskQueue 統合システム
    """

    def __init__(self, workspace_path: str, tracker_id: str = "INTG-089"):
        """
        統合SubAgentSystemの初期化

        Args:
            workspace_path: ワークスペースパス
            tracker_id: トラッカーID
        """
        self.workspace_path = workspace_path
        self.tracker_id = tracker_id

        # 3つのコンポーネント初期化
        from .subagent_monitor import SubAgentMonitor
        from .notification_bridge import NotificationBridge

        self.monitor = SubAgentMonitor(workspace_path)
        self.notification = NotificationBridge(workspace_path, tracker_id)
        self.task_queue = EnhancedSubAgentTaskQueue(Path(workspace_path), tracker_id)

        # 統合設定
        self.integration_enabled = True
        self.auto_monitoring = True
        self.monitoring_interval = 60  # 1分間隔

        # 統合統計
        self.integration_stats = {
            'system_start_time': datetime.now(timezone.utc).isoformat(),
            'total_monitoring_cycles': 0,
            'anomalies_detected': 0,
            'notifications_sent': 0,
            'tasks_executed': 0,
            'successful_integrations': 0,
            'failed_integrations': 0
        }

        self.logger = self._setup_integration_logger()
        self.logger.info(f"IntegratedSubAgentSystem initialized for {tracker_id}")

    def _setup_integration_logger(self):
        """統合ログ設定"""
        logger = logging.getLogger(f"IntegratedSubAgentSystem.{self.tracker_id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            log_file = Path(self.workspace_path) / ".subagent_queue" / "logs" / "integrated_system.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def execute_monitoring_cycle(self) -> Dict[str, Any]:
        """
        統合監視サイクル実行

        Returns:
            Dict[str, Any]: 監視サイクル結果
        """
        cycle_start = time.time()
        cycle_results = {
            'cycle_id': f"cycle_{int(cycle_start)}",
            'start_time': datetime.now(timezone.utc).isoformat(),
            'anomalies_detected': [],
            'notifications_sent': [],
            'tasks_processed': 0,
            'integration_success': True,
            'errors': []
        }

        try:
            self.logger.info("Starting integrated monitoring cycle")

            # 1. システム異常検知
            anomalies = self.monitor.comprehensive_anomaly_check()
            cycle_results['anomalies_detected'] = anomalies.get('anomalies', [])

            if anomalies.get('anomaly_count', 0) > 0:
                self.integration_stats['anomalies_detected'] += anomalies['anomaly_count']

                # 2. 重要度に応じた通知送信
                for anomaly in anomalies.get('anomalies', []):
                    try:
                        notification_result = self.notification.send_enhanced_notification(
                            title=f"System Anomaly Detected: {self.tracker_id}",
                            message=f"Anomaly: {anomaly}",
                            priority_level="high",
                            tracker_id=self.tracker_id
                        )
                        if notification_result:
                            cycle_results['notifications_sent'].append({
                                'anomaly': anomaly,
                                'notification_sent': True,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                            self.integration_stats['notifications_sent'] += 1
                    except Exception as e:
                        cycle_results['errors'].append(f"Notification error: {e}")

            # 3. タスクキューステータス確認とメンテナンス
            try:
                queue_status = self.task_queue.get_integrated_system_status()

                # チェックポイントクリーンアップ（必要時）
                checkpoint_stats = queue_status.get('checkpoint_system', {})
                if checkpoint_stats.get('total_checkpoints', 0) > 50:
                    cleaned = self.task_queue.cleanup_old_checkpoints(keep_days=3)
                    if cleaned > 0:
                        self.logger.info(f"Cleaned {cleaned} old checkpoints")

            except Exception as e:
                cycle_results['errors'].append(f"Task queue error: {e}")

            # 4. システム統計更新
            self.integration_stats['total_monitoring_cycles'] += 1
            if not cycle_results['errors']:
                self.integration_stats['successful_integrations'] += 1
            else:
                self.integration_stats['failed_integrations'] += 1

            # 5. サイクル完了
            cycle_results['execution_time'] = time.time() - cycle_start
            cycle_results['end_time'] = datetime.now(timezone.utc).isoformat()

            self.logger.info(f"Monitoring cycle completed in {cycle_results['execution_time']:.2f}s")
            return cycle_results

        except Exception as e:
            cycle_results['integration_success'] = False
            cycle_results['errors'].append(f"Critical integration error: {e}")
            self.integration_stats['failed_integrations'] += 1
            self.logger.error(f"Monitoring cycle failed: {e}")
            return cycle_results

    def execute_integrated_task(self, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        統合タスク実行（監視+通知+実行）

        Args:
            task_definition: タスク定義

        Returns:
            Dict[str, Any]: 実行結果
        """
        task_id = task_definition.get('task_id', f"task_{int(time.time())}")
        self.logger.info(f"Starting integrated task execution: {task_id}")

        try:
            # 1. 事前システムチェック
            pre_check = self.monitor.comprehensive_anomaly_check()
            if pre_check.get('anomaly_count', 0) > 2:
                self.notification.send_enhanced_notification(
                    title=f"Pre-execution System Warning: {self.tracker_id}",
                    message=f"System anomalies detected before task {task_id}",
                    priority_level="medium"
                )

            # 2. 統合タスク実行
            result = self.task_queue.execute_with_integrated_features(task_definition)

            # 3. 実行後通知
            if result.get('status') == 'completed':
                self.notification.send_enhanced_notification(
                    title=f"Task Completed: {task_id}",
                    message=f"Task execution successful for {self.tracker_id}",
                    priority_level="low"
                )
            elif result.get('status') == 'failed':
                self.notification.send_enhanced_notification(
                    title=f"Task Failed: {task_id}",
                    message=f"Task execution failed: {result.get('error', 'Unknown error')}",
                    priority_level="high"
                )

            # 4. 統計更新
            self.integration_stats['tasks_executed'] += 1

            return result

        except Exception as e:
            self.logger.error(f"Integrated task execution failed: {task_id} - {e}")

            # エラー通知
            self.notification.send_enhanced_notification(
                title=f"System Integration Error: {task_id}",
                message=f"Critical integration failure: {e}",
                priority_level="critical"
            )

            return {
                'status': 'integration_failed',
                'task_id': task_id,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_system_health_report(self) -> Dict[str, Any]:
        """
        システム全体健康状態レポート取得

        Returns:
            Dict[str, Any]: システム健康状態
        """
        try:
            monitor_status = self.monitor.comprehensive_anomaly_check()
            notification_stats = self.notification.get_hash_cache_statistics()
            queue_status = self.task_queue.get_integrated_system_status()

            health_report = {
                'system_overview': {
                    'integration_enabled': self.integration_enabled,
                    'auto_monitoring_enabled': self.auto_monitoring,
                    'tracker_id': self.tracker_id,
                    'workspace_path': self.workspace_path,
                    'report_timestamp': datetime.now(timezone.utc).isoformat()
                },
                'component_health': {
                    'monitor': {
                        'anomaly_count': monitor_status.get('anomaly_count', 0),
                        'system_stats': monitor_status.get('system_stats', {}),
                        'last_check': monitor_status.get('timestamp')
                    },
                    'notification': {
                        'cache_efficiency': f"{notification_stats.get('hit_rate_percent', 0)}%",
                        'total_requests': notification_stats.get('total_requests', 0),
                        'cache_size': notification_stats.get('cache_size', 0)
                    },
                    'task_queue': {
                        'checkpoint_count': queue_status.get('checkpoint_system', {}).get('total_checkpoints', 0),
                        'gpu_fallback_enabled': queue_status.get('system_health', {}).get('gpu_fallback_enabled', False),
                        'auto_recovery_enabled': queue_status.get('system_health', {}).get('auto_recovery_enabled', False)
                    }
                },
                'integration_statistics': dict(self.integration_stats),
                'overall_health_score': self._calculate_health_score(monitor_status, notification_stats, queue_status)
            }

            return health_report

        except Exception as e:
            self.logger.error(f"Failed to get system health report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _calculate_health_score(self, monitor_status: Dict, notification_stats: Dict, queue_status: Dict) -> float:
        """
        システム健康度スコア計算

        Returns:
            float: 健康度スコア (0.0-1.0)
        """
        score_components = []

        # モニター健康度 (40%)
        anomaly_count = monitor_status.get('anomaly_count', 0)
        monitor_score = max(0.0, 1.0 - (anomaly_count / 10.0))  # 10個以上の異常で0点
        score_components.append(monitor_score * 0.4)

        # 通知システム効率 (30%)
        cache_hit_rate = notification_stats.get('hit_rate_percent', 0) / 100.0
        score_components.append(cache_hit_rate * 0.3)

        # タスクキュー健康度 (30%)
        queue_health = queue_status.get('system_health', {})
        queue_features = sum([
            queue_health.get('auto_checkpoint_enabled', False),
            queue_health.get('gpu_fallback_enabled', False),
            queue_health.get('auto_recovery_enabled', False)
        ]) / 3.0
        score_components.append(queue_features * 0.3)

        return round(sum(score_components), 3)

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        チェックポイント読み込み

        Args:
            task_id: タスクID

        Returns:
            Optional[Dict[str, Any]]: チェックポイントデータ
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"

            if not checkpoint_file.exists():
                return None

            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            self.logger.info(f"Checkpoint loaded: {task_id}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {task_id} - {e}")
            return None

    def resume_task_from_checkpoint(self, task_id: str) -> Optional[Dict]:
        """
        チェックポイントからのタスク復旧

        Args:
            task_id: タスクID

        Returns:
            Optional[Dict]: 復旧結果
        """
        try:
            # チェックポイント読み込み
            checkpoint = self.load_checkpoint(task_id)
            if not checkpoint:
                self.logger.warning(f"No checkpoint found for task: {task_id}")
                return None

            # タスクレジストリー更新
            registry = self._load_task_registry()
            if task_id not in registry:
                self.logger.error(f"Task not found in registry: {task_id}")
                return None

            task = registry[task_id]

            # 復旧用コマンド生成（extract_character.py用の例）
            original_command = task.get('command', '')
            if 'extract_character.py' in original_command and '--resume' not in original_command:
                resumed_command = original_command + " --resume"
                task['command'] = resumed_command

            task['status'] = 'resumed'
            task['resumed_from_checkpoint'] = True
            task['checkpoint_data'] = checkpoint['progress_data']

            # レジストリー更新
            registry[task_id] = task
            self._save_task_registry(registry)

            self.logger.info(f"Task resumed from checkpoint: {task_id}")

            return {
                'status': 'resumed',
                'task_id': task_id,
                'checkpoint_timestamp': checkpoint['timestamp'],
                'command': task['command']
            }

        except Exception as e:
            self.logger.error(f"Failed to resume task: {task_id} - {e}")
            return None

    def execute_with_gpu_fallback(self, task: Dict) -> Dict:
        """
        GPU fallback機能付きタスク実行（統合版）

        Args:
            task: タスク定義

        Returns:
            Dict: 実行結果
        """
        start_time = time.time()
        gpu_retry_count = 0
        fallback_reason = None

        self.logger.info(f"Starting enhanced GPU fallback execution: {task['task_id']}")

        # GPU健全性事前チェック
        gpu_health = self._comprehensive_gpu_health_check()
        
        while gpu_retry_count <= self.max_gpu_retry:
            try:
                # 動的GPU利用可能性判定
                should_use_gpu = self._should_use_gpu(gpu_health, gpu_retry_count)
                
                if should_use_gpu:
                    # GPU実行試行
                    self.logger.info(f"Attempting GPU execution (attempt {gpu_retry_count + 1}) - Health: {gpu_health['overall_score']:.1f}")

                    # 最適化されたGPU環境設定
                    env = self._prepare_gpu_environment(gpu_health)
                    result = self._execute_task_with_env(task, env)

                    if result['status'] == 'completed':
                        self.gpu_fallback_stats['successes'] += 1
                        self._record_gpu_success(gpu_health)
                        self.logger.info(f"GPU execution successful: {task['task_id']}")
                        return result
                        
                    elif self._is_gpu_related_error(result.get('error', '')):
                        # GPU関連エラーの詳細解析
                        error_type = self._analyze_gpu_error(result.get('error', ''))
                        self.logger.warning(f"GPU error detected ({error_type}): {result.get('error', '')[:100]}...")
                        
                        self._record_gpu_error(error_type, result.get('error', ''))
                        gpu_retry_count += 1
                        self.gpu_fallback_stats['attempts'] += 1
                        
                        # GPU健全性を再評価
                        gpu_health = self._comprehensive_gpu_health_check()
                        continue
                    else:
                        # GPU以外のエラー
                        return result

                else:
                    # CPU fallback実行
                    fallback_reason = gpu_health.get('fallback_reason', 'GPU unavailable')
                    self.logger.info(f"Executing with CPU fallback (attempt {gpu_retry_count + 1}) - Reason: {fallback_reason}")

                    # CPU最適化環境設定
                    env = self._prepare_cpu_environment()
                    result = self._execute_task_with_env(task, env)

                    if result['status'] == 'completed':
                        self.gpu_fallback_stats['successes'] += 1
                        self.gpu_fallback_stats['cpu_fallback_count'] += 1
                        self.logger.info(f"CPU fallback execution successful: {task['task_id']}")
                        return result
                    else:
                        gpu_retry_count += 1
                        self.gpu_fallback_stats['failures'] += 1

                        if gpu_retry_count <= self.max_gpu_retry:
                            wait_time = min(2 ** gpu_retry_count, 60)  # 最大60秒のexponential backoff
                            self.logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)

                    if result['status'] == 'completed':
                        self.gpu_fallback_stats['successes'] += 1
                        self.logger.info(f"CPU fallback execution successful: {task['task_id']}")
                        return result
                    else:
                        gpu_retry_count += 1
                        self.gpu_fallback_stats['failures'] += 1

                        if gpu_retry_count <= self.max_gpu_retry:
                            wait_time = 2 ** gpu_retry_count  # 指数バックオフ
                            self.logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)

            except Exception as e:
                self.logger.error(f"GPU fallback execution error: {e}")
                gpu_retry_count += 1
                self.gpu_fallback_stats['failures'] += 1

        # 最大試行回数到達
        execution_time = time.time() - start_time

        return {
            'status': 'failed',
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'output': None,
            'error': f'GPU fallback failed after {self.max_gpu_retry + 1} attempts',
            'execution_stats': {
                'execution_time': execution_time,
                'gpu_fallback_attempts': gpu_retry_count,
                'gpu_fallback_failed': True
            }
        }

    def _execute_task_with_env(self, task: Dict, env: Dict[str, str]) -> Dict:
        """
        環境変数指定でのタスク実行

        Args:
            task: タスク定義
            env: 環境変数

        Returns:
            Dict: 実行結果
        """
        start_time = time.time()

        try:
            # 実行開始時刻記録
            task["started_at"] = datetime.now(timezone.utc).isoformat()
            task["status"] = "running"

            # コマンド実行
            process = subprocess.Popen(
                task["command"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/mnt/c/AItools/segment-anything",
                env=env
            )

            # プロセスID記録
            task["pid"] = process.pid
            self._save_running_task(task)

            # プロセス実行完了待機
            try:
                stdout, stderr = process.communicate(timeout=self.task_timeout)
                result_code = process.returncode
            except subprocess.TimeoutExpired:
                self._terminate_process(process.pid)
                stdout, stderr = process.communicate()
                result_code = -1

            # 実行時間計算
            execution_time = time.time() - start_time

            # 結果判定
            if result_code == 0:
                status = "completed"
                output = stdout
                error = None
            else:
                status = "failed"
                output = stdout
                error = stderr

            # 実行完了後はrunning_tasksから削除
            self._remove_running_task(task["task_id"])

            return {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": output,
                "error": error,
                "execution_stats": {
                    "execution_time": execution_time,
                    "return_code": result_code,
                    "env_type": "gpu" if env.get('CUDA_VISIBLE_DEVICES') != '' else "cpu"
                }
            }

        except Exception as e:
            # エラー時も実行中タスクから削除
            self._remove_running_task(task["task_id"])
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

    def _check_gpu_available(self) -> bool:
        """GPU利用可能性チェック"""
        try:
            # nvidia-smiチェック
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False

            # PyTorchチェック
            try:
                import torch
                return torch.cuda.is_available() and torch.cuda.device_count() > 0
            except ImportError:
                return False

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def get_enhanced_queue_status(self) -> Dict:
        """
        拡張キューステータス取得（INTG-089機能込み）

        Returns:
            Dict: 拡張キュー状況
        """
        base_status = self.get_queue_status()

        # INTG-089拡張情報追加
        base_status.update({
            'gpu_fallback_enabled': self.gpu_fallback_enabled,
            'gpu_fallback_stats': dict(self.gpu_fallback_stats),
            'checkpoint_count': len(list(self.checkpoint_dir.glob('*_checkpoint.json'))),
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'gpu_available': self._check_gpu_available()
        })

        return base_status

    def _start_progress_monitoring(self, task_id: str, pid: int) -> None:
        """
        進捗監視開始（自動チェックポイント）
        
        Args:
            task_id: タスクID
            pid: プロセスID
        """
        try:
            self.progress_monitors[task_id] = {
                'pid': pid,
                'start_time': time.time(),
                'last_checkpoint': time.time(),
                'checkpoint_count': 0
            }
            
            # バックグラウンドスレッドで監視開始
            import threading
            
            def monitor_progress():
                self._monitor_task_progress(task_id)
            
            thread = threading.Thread(target=monitor_progress, daemon=True)
            thread.start()
            
            self.logger.info(f"Progress monitoring started for task: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start progress monitoring: {task_id} - {e}")
    
    def _stop_progress_monitoring(self, task_id: str) -> None:
        """
        進捗監視停止
        
        Args:
            task_id: タスクID
        """
        try:
            if task_id in self.progress_monitors:
                monitor_data = self.progress_monitors.pop(task_id)
                self.logger.info(f"Progress monitoring stopped for task: {task_id} (checkpoints: {monitor_data.get('checkpoint_count', 0)})")
                
        except Exception as e:
            self.logger.error(f"Failed to stop progress monitoring: {task_id} - {e}")
    
    def _monitor_task_progress(self, task_id: str) -> None:
        """
        タスク進捗監視（定期チェックポイント作成）
        
        Args:
            task_id: タスクID
        """
        try:
            monitor_data = self.progress_monitors.get(task_id)
            if not monitor_data:
                return
            
            pid = monitor_data['pid']
            
            while task_id in self.progress_monitors:
                try:
                    # プロセス存在確認
                    if not psutil.pid_exists(pid):
                        break
                        
                    proc = psutil.Process(pid)
                    if not proc.is_running():
                        break
                    
                    # チェックポイント間隔チェック
                    current_time = time.time()
                    time_since_last = current_time - monitor_data['last_checkpoint']
                    
                    if time_since_last >= self.checkpoint_interval:
                        # 進捗データ収集
                        progress_data = self._collect_progress_data(task_id, pid, proc)
                        
                        # チェックポイント保存
                        if self.save_checkpoint(task_id, progress_data):
                            monitor_data['last_checkpoint'] = current_time
                            monitor_data['checkpoint_count'] += 1
                            
                            self.logger.info(f"Auto-checkpoint created for task: {task_id} (#{monitor_data['checkpoint_count']})")
                    
                    # 30秒間隔で監視
                    time.sleep(30)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # プロセス終了済み
                    break
                except Exception as e:
                    self.logger.error(f"Progress monitoring error for {task_id}: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Progress monitoring failed for {task_id}: {e}")
        finally:
            # 監視データクリーンアップ
            if task_id in self.progress_monitors:
                self.progress_monitors.pop(task_id, None)
    
    def _collect_progress_data(self, task_id: str, pid: int, proc: psutil.Process) -> Dict[str, Any]:
        """
        進捗データ収集
        
        Args:
            task_id: タスクID  
            pid: プロセスID
            proc: psutilプロセスオブジェクト
            
        Returns:
            Dict[str, Any]: 進捗データ
        """
        try:
            # システム情報収集
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            
            # 実行時間計算
            monitor_data = self.progress_monitors.get(task_id, {})
            start_time = monitor_data.get('start_time', time.time())
            runtime_seconds = time.time() - start_time
            
            # ディスク使用量チェック（出力ディレクトリ）
            output_size = 0
            try:
                # WorkspaceConfigManagerを使って動的パス解決
                from config.workspace_config import WorkspaceConfig
                workspace_config = WorkspaceConfig()
                config = workspace_config.get_workspace_config(self.tracker_id)
                
                if config:
                    extraction_path = f"{config['workspace_path']}/extraction/"
                    workspace_dirs = [extraction_path]
                else:
                    # フォールバック: 従来の複数パス
                    workspace_dirs = [
                        f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{self.tracker_id}/extraction/",
                        f"/mnt/c/AItools/lora/train/kiri/tracker-workspace/{self.tracker_id}/extraction/"
                    ]
                
                for output_dir in workspace_dirs:
                    if Path(output_dir).exists():
                        output_size = sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file())
                        break
            except Exception:
                pass
            
            progress_data = {
                'task_id': task_id,
                'checkpoint_time': time.time(),
                'runtime_seconds': runtime_seconds,
                'system_stats': {
                    'pid': pid,
                    'cpu_percent': cpu_percent,
                    'memory_rss_mb': memory_info.rss // 1024 // 1024,
                    'memory_vms_mb': memory_info.vms // 1024 // 1024,
                },
                'output_stats': {
                    'output_size_bytes': output_size,
                    'output_size_mb': output_size // 1024 // 1024
                },
                'checkpoint_metadata': {
                    'checkpoint_count': monitor_data.get('checkpoint_count', 0) + 1,
                    'auto_generated': True,
                    'monitoring_interval': self.checkpoint_interval
                }
            }
            
            # GPU統計（利用可能な場合）
            try:
                import torch
                if torch.cuda.is_available():
                    progress_data['gpu_stats'] = {
                        'gpu_memory_allocated': torch.cuda.memory_allocated(),
                        'gpu_memory_cached': torch.cuda.memory_reserved(),
                        'gpu_available': True
                    }
            except Exception:
                progress_data['gpu_stats'] = {'gpu_available': False}
            
            return progress_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect progress data for {task_id}: {e}")
            return {
                'task_id': task_id,
                'checkpoint_time': time.time(),
                'error': str(e),
                'checkpoint_metadata': {'auto_generated': True, 'error': True}
            }
    
    def cleanup_checkpoints(self, keep_days: int = 7) -> int:
        """
        古いチェックポイントのクリーンアップ

        Args:
            keep_days: 保持日数

        Returns:
            int: クリーンアップしたファイル数
        """
        try:
            cutoff_time = time.time() - (keep_days * 24 * 3600)
            cleaned_count = 0

            for checkpoint_file in self.checkpoint_dir.glob('*_checkpoint.json'):
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Checkpoint cleanup: {cleaned_count} files removed")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Checkpoint cleanup error: {e}")
            return 0
            
    def _comprehensive_gpu_health_check(self) -> Dict[str, Any]:
        """
        包括的GPU健全性チェック
        
        Returns:
            Dict[str, Any]: GPU健全性情報
        """
        current_time = time.time()
        
        # キャッシュされた結果を使用（30秒以内）
        if (current_time - self.gpu_fallback_stats.get('last_gpu_check', 0)) < 30:
            return {
                'gpu_available': self.gpu_fallback_stats.get('gpu_healthy', False),
                'overall_score': 0.5 if self.gpu_fallback_stats.get('gpu_healthy') else 0.0,
                'cached': True
            }
        
        health_info = {
            'gpu_available': False,
            'memory_available_mb': 0,
            'memory_total_mb': 0,
            'temperature_ok': True,
            'driver_ok': True,
            'overall_score': 0.0,
            'fallback_reason': 'Unknown',
            'detailed_checks': {}
        }
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                health_info['fallback_reason'] = 'CUDA not available'
                return health_info
                
            device_count = torch.cuda.device_count()
            if device_count == 0:
                health_info['fallback_reason'] = 'No CUDA devices found'
                return health_info
            
            # GPU 0の詳細チェック
            device_props = torch.cuda.get_device_properties(0)
            
            # メモリチェック
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = device_props.total_memory
            memory_available = memory_total - memory_reserved
            
            health_info['memory_available_mb'] = memory_available // 1024 // 1024
            health_info['memory_total_mb'] = memory_total // 1024 // 1024
            health_info['memory_usage_percent'] = (memory_reserved / memory_total) * 100
            
            # メモリ不足チェック
            if memory_available < (1024 * 1024 * 1024):  # 1GB未満
                health_info['fallback_reason'] = 'Insufficient GPU memory'
                health_info['overall_score'] = 0.2
            else:
                health_info['gpu_available'] = True
                health_info['overall_score'] = min(1.0, memory_available / (2 * 1024 * 1024 * 1024))  # 2GB基準でスコア計算
            
            # 簡易GPU計算テスト
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, result
                torch.cuda.empty_cache()
                
                health_info['detailed_checks']['computation_test'] = True
                if health_info['overall_score'] > 0:
                    health_info['overall_score'] = min(health_info['overall_score'] + 0.2, 1.0)
                    
            except Exception as e:
                health_info['detailed_checks']['computation_test'] = False
                health_info['detailed_checks']['computation_error'] = str(e)
                health_info['fallback_reason'] = f'GPU computation test failed: {str(e)[:50]}'
                health_info['overall_score'] *= 0.5
            
            # 最近のエラー履歴チェック
            recent_errors = [e for e in self.gpu_fallback_stats.get('gpu_errors', []) 
                           if (current_time - e.get('timestamp', 0)) < 300]  # 5分以内
            
            if len(recent_errors) >= 3:
                health_info['fallback_reason'] = 'Recent GPU errors detected'
                health_info['overall_score'] *= 0.3
            
            self.gpu_fallback_stats['last_gpu_check'] = current_time
            self.gpu_fallback_stats['gpu_healthy'] = health_info['overall_score'] > 0.5
            
        except ImportError:
            health_info['fallback_reason'] = 'PyTorch not available'
        except Exception as e:
            health_info['fallback_reason'] = f'GPU check error: {str(e)[:50]}'
            health_info['detailed_checks']['check_error'] = str(e)
        
        return health_info
    
    def _should_use_gpu(self, gpu_health: Dict[str, Any], retry_count: int) -> bool:
        """
        GPU使用可否判定
        
        Args:
            gpu_health: GPU健全性情報
            retry_count: リトライ回数
            
        Returns:
            bool: GPU使用可否
        """
        # 基本的なGPU利用不可条件
        if not gpu_health.get('gpu_available', False):
            return False
            
        # 健全性スコアによる判定
        health_score = gpu_health.get('overall_score', 0.0)
        
        # リトライ回数に応じて閾値を下げる
        threshold = max(0.3, 0.8 - (retry_count * 0.2))
        
        return health_score >= threshold
    
    def _prepare_gpu_environment(self, gpu_health: Dict[str, Any]) -> Dict[str, str]:
        """
        GPU環境変数準備
        
        Args:
            gpu_health: GPU健全性情報
            
        Returns:
            Dict[str, str]: 環境変数
        """
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # メモリ使用量に応じた最適化
        memory_usage = gpu_health.get('memory_usage_percent', 0)
        if memory_usage > 70:
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
        return env
    
    def _prepare_cpu_environment(self) -> Dict[str, str]:
        """
        CPU環境変数準備
        
        Returns:
            Dict[str, str]: 環境変数
        """
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['CPU_ONLY'] = '1'
        env['OMP_NUM_THREADS'] = str(min(psutil.cpu_count(), 8))  # CPU使用数制限
        
        return env
    
    def _is_gpu_related_error(self, error_message: str) -> bool:
        """
        GPU関連エラー判定
        
        Args:
            error_message: エラーメッセージ
            
        Returns:
            bool: GPU関連エラーか
        """
        if not error_message:
            return False
            
        gpu_error_keywords = [
            'cuda', 'gpu', 'out of memory', 'cublas', 'cudnn', 'nvml',
            'device-side assert', 'cuda runtime error', 'cuda driver',
            'insufficient memory', 'gpu memory'
        ]
        
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in gpu_error_keywords)
    
    def _analyze_gpu_error(self, error_message: str) -> str:
        """
        GPUエラー分析
        
        Args:
            error_message: エラーメッセージ
            
        Returns:
            str: エラータイプ
        """
        error_lower = error_message.lower()
        
        if 'out of memory' in error_lower or 'insufficient memory' in error_lower:
            return 'memory_error'
        elif 'cuda runtime error' in error_lower:
            return 'runtime_error'  
        elif 'device-side assert' in error_lower:
            return 'assertion_error'
        elif 'cuda driver' in error_lower:
            return 'driver_error'
        else:
            return 'unknown_gpu_error'
    
    def _record_gpu_error(self, error_type: str, error_message: str) -> None:
        """
        GPUエラー記録
        
        Args:
            error_type: エラータイプ
            error_message: エラーメッセージ
        """
        error_record = {
            'timestamp': time.time(),
            'error_type': error_type,
            'message': error_message[:200],  # メッセージは200文字まで
        }
        
        if 'gpu_errors' not in self.gpu_fallback_stats:
            self.gpu_fallback_stats['gpu_errors'] = []
            
        self.gpu_fallback_stats['gpu_errors'].append(error_record)
        
        # エラー履歴は最新の10件まで保持
        if len(self.gpu_fallback_stats['gpu_errors']) > 10:
            self.gpu_fallback_stats['gpu_errors'] = self.gpu_fallback_stats['gpu_errors'][-10:]
    
    def _record_gpu_success(self, gpu_health: Dict[str, Any]) -> None:
        """
        GPU成功記録
        
        Args:
            gpu_health: GPU健全性情報
        """
        # GPU健全性を回復
        self.gpu_fallback_stats['gpu_healthy'] = True
        
        # 古いエラー履歴をクリア（成功時）
        current_time = time.time()
        self.gpu_fallback_stats['gpu_errors'] = [
            e for e in self.gpu_fallback_stats.get('gpu_errors', [])
            if (current_time - e.get('timestamp', 0)) < 300  # 5分以内のエラーのみ保持
        ]


# INTG-089: 3システム統合コーディネーター
class SubAgentSystemCoordinator:
    """SubAgent 3システム統合コーディネーター（INTG-089）"""
    
    def __init__(self, workspace_path: str, tracker_id: str):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
            tracker_id: トラッカーID
        """
        self.workspace_path = workspace_path
        self.tracker_id = tracker_id
        self.logger = logging.getLogger(f"SubAgentCoordinator.{tracker_id}")
        
        # 3システム初期化
        self._initialize_systems()
        
        # 統合状態管理
        self.coordination_stats = {
            'started_at': time.time(),
            'tasks_coordinated': 0,
            'anomalies_handled': 0,
            'notifications_sent': 0,
            'checkpoints_created': 0,
            'gpu_fallbacks': 0
        }
        
        self.logger.info(f"SubAgent System Coordinator initialized for {tracker_id}")
        
    def _initialize_systems(self):
        """3システム初期化"""
        try:
            # SubAgentMonitorの初期化
            from tools.queue.subagent_monitor import SubAgentMonitor
            self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)
            
            # NotificationBridgeの初期化
            from tools.queue.notification_bridge import NotificationBridge
            self.notification_bridge = NotificationBridge(
                workspace_path=self.workspace_path, 
                tracker_id=self.tracker_id
            )
            
            # EnhancedSubAgentTaskQueueの初期化
            self.task_queue = EnhancedSubAgentTaskQueue(
                workspace_path=Path(self.workspace_path),
                tracker_id=self.tracker_id
            )
            
            self.logger.info("All 3 systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize systems: {e}")
            raise
    
    def execute_coordinated_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        3システム連携タスク実行
        
        Args:
            task: タスク定義
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        task_id = task.get('task_id', 'unknown')
        self.logger.info(f"Starting coordinated execution: {task_id}")
        
        try:
            # Phase 1: 事前異常検知
            anomalies = self.monitor.comprehensive_anomaly_check()
            
            if anomalies['anomaly_count'] > 0:
                self.coordination_stats['anomalies_handled'] += anomalies['anomaly_count']
                
                # 重要な異常の場合は通知
                if self._is_critical_anomaly(anomalies):
                    self.notification_bridge.send_anomaly_notification(anomalies)
                    self.coordination_stats['notifications_sent'] += 1
            
            # Phase 2: GPU fallback判定付きタスク実行
            if task.get('gpu_enabled', True):
                execution_result = self.task_queue.execute_with_gpu_fallback(task)
                if 'gpu_fallback' in execution_result.get('execution_stats', {}):
                    self.coordination_stats['gpu_fallbacks'] += 1
            else:
                execution_result = self.task_queue.execute_task(task)
            
            # Phase 3: 実行中監視（長時間タスクの場合）
            if execution_result.get('status') == 'running':
                self._start_coordinated_monitoring(task_id)
            
            # Phase 4: 結果処理と通知
            if execution_result.get('status') == 'completed':
                self.notification_bridge.handle_task_completion(
                    task_id=task_id,
                    task_type=task.get('task_type', 'unknown'),
                    results=execution_result.get('execution_stats', {})
                )
                self.coordination_stats['notifications_sent'] += 1
                
            elif execution_result.get('status') == 'failed':
                self.notification_bridge.handle_task_failure(
                    task_id=task_id,
                    task_type=task.get('task_type', 'unknown'),
                    error=execution_result.get('error', 'Unknown error'),
                    retry_count=task.get('retry_count', 0),
                    command=task.get('command', '')
                )
                self.coordination_stats['notifications_sent'] += 1
            
            # Phase 5: 事後異常検知
            post_anomalies = self.monitor.comprehensive_anomaly_check()
            if post_anomalies['anomaly_count'] > anomalies['anomaly_count']:
                # 新たな異常が発生
                new_anomaly_count = post_anomalies['anomaly_count'] - anomalies['anomaly_count']
                self.coordination_stats['anomalies_handled'] += new_anomaly_count
                
                self.notification_bridge.send_anomaly_notification(post_anomalies)
                self.coordination_stats['notifications_sent'] += 1
            
            self.coordination_stats['tasks_coordinated'] += 1
            
            return {
                'status': execution_result.get('status'),
                'execution_result': execution_result,
                'coordination_stats': self.coordination_stats.copy(),
                'anomalies': {
                    'pre_execution': anomalies,
                    'post_execution': post_anomalies
                }
            }
            
        except Exception as e:
            self.logger.error(f"Coordinated execution failed: {task_id} - {e}")
            
            # エラー時の通知
            self.notification_bridge.handle_task_failure(
                task_id=task_id,
                task_type=task.get('task_type', 'unknown'),
                error=str(e),
                retry_count=task.get('retry_count', 0),
                command=task.get('command', '')
            )
            
            return {
                'status': 'error',
                'error': str(e),
                'coordination_stats': self.coordination_stats.copy()
            }
    
    def _is_critical_anomaly(self, anomalies: Dict[str, Any]) -> bool:
        """
        重要異常判定
        
        Args:
            anomalies: 異常検知結果
            
        Returns:
            bool: 重要異常フラグ
        """
        if anomalies['anomaly_count'] >= 3:
            return True
            
        for anomaly in anomalies.get('anomalies', []):
            if anomaly.get('type') == 'gpu' or 'critical' in anomaly.get('message', '').lower():
                return True
                
        return False
    
    def _start_coordinated_monitoring(self, task_id: str) -> None:
        """
        連携監視開始
        
        Args:
            task_id: タスクID
        """
        try:
            import threading
            
            def coordinated_monitor():
                self._monitor_task_coordination(task_id)
            
            thread = threading.Thread(target=coordinated_monitor, daemon=True)
            thread.start()
            
            self.logger.info(f"Coordinated monitoring started for: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start coordinated monitoring: {e}")
    
    def _monitor_task_coordination(self, task_id: str) -> None:
        """
        タスク連携監視
        
        Args:
            task_id: タスクID
        """
        monitoring_start = time.time()
        last_checkpoint = time.time()
        last_anomaly_check = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # 実行中タスクが存在するかチェック
                running_tasks = self.task_queue.list_running_tasks()
                if not running_tasks.get('tasks', {}).get(task_id):
                    # タスク完了または停止
                    break
                
                # 30秒ごとに異常検知
                if current_time - last_anomaly_check >= 30:
                    anomalies = self.monitor.comprehensive_anomaly_check()
                    
                    if anomalies['anomaly_count'] > 0:
                        self.coordination_stats['anomalies_handled'] += anomalies['anomaly_count']
                        
                        # 異常通知（重複防止機能付き）
                        self.notification_bridge.send_anomaly_notification(anomalies)
                        self.coordination_stats['notifications_sent'] += 1
                    
                    last_anomaly_check = current_time
                
                # 5分ごとにチェックポイント作成
                if current_time - last_checkpoint >= 300:
                    progress_data = {
                        'monitoring_duration': current_time - monitoring_start,
                        'coordination_stats': self.coordination_stats.copy(),
                        'last_anomaly_check': last_anomaly_check,
                        'system_coordination': True
                    }
                    
                    if self.task_queue.save_checkpoint(task_id, progress_data):
                        self.coordination_stats['checkpoints_created'] += 1
                    
                    last_checkpoint = current_time
                
                # 10秒待機
                time.sleep(10)
                
        except Exception as e:
            self.logger.error(f"Coordination monitoring error for {task_id}: {e}")
            
        finally:
            self.logger.info(f"Coordination monitoring ended for: {task_id}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """
        連携状態取得
        
        Returns:
            Dict[str, Any]: 連携状態
        """
        try:
            # 各システムの個別状態取得
            monitor_status = {
                'gpu_monitoring_enabled': self.monitor.gpu_monitoring_enabled,
                'memory_baseline': self.monitor.memory_baseline,
                'anomaly_thresholds': self.monitor.anomaly_thresholds
            }
            
            queue_status = self.task_queue.get_queue_status()
            notification_stats = self.notification_bridge.get_notification_stats()
            
            coordination_runtime = time.time() - self.coordination_stats['started_at']
            
            return {
                'coordinator': {
                    'tracker_id': self.tracker_id,
                    'runtime_seconds': coordination_runtime,
                    'coordination_stats': self.coordination_stats
                },
                'systems': {
                    'monitor': monitor_status,
                    'task_queue': queue_status,
                    'notifications': notification_stats
                },
                'integration_health': {
                    'all_systems_active': True,
                    'coordination_rate': self.coordination_stats['tasks_coordinated'] / max(coordination_runtime / 3600, 1),  # tasks/hour
                    'anomaly_detection_rate': self.coordination_stats['anomalies_handled'] / max(coordination_runtime / 3600, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get coordination status: {e}")
            return {
                'error': str(e),
                'coordination_stats': self.coordination_stats
            }
    
    def shutdown_coordination(self) -> Dict[str, Any]:
        """
        連携システム停止
        
        Returns:
            Dict[str, Any]: 停止結果
        """
        try:
            shutdown_start = time.time()
            
            # 実行中タスクの安全停止
            running_tasks = self.task_queue.list_running_tasks()
            stopped_tasks = 0
            
            for task_id in running_tasks.get('tasks', {}):
                if self.task_queue.kill_task(task_id):
                    stopped_tasks += 1
            
            # 最終チェックポイント作成
            final_checkpoint_data = {
                'shutdown_time': time.time(),
                'final_coordination_stats': self.coordination_stats.copy(),
                'stopped_tasks': stopped_tasks,
                'shutdown_type': 'coordinated'
            }
            
            # システム終了処理
            shutdown_duration = time.time() - shutdown_start
            
            self.logger.info(f"Coordination shutdown completed in {shutdown_duration:.1f}s")
            
            return {
                'success': True,
                'shutdown_duration': shutdown_duration,
                'stopped_tasks': stopped_tasks,
                'final_stats': self.coordination_stats
            }
            
        except Exception as e:
            self.logger.error(f"Coordination shutdown error: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_stats': self.coordination_stats
            }


def main():
    """メイン実行フロー"""
    if len(sys.argv) < 2:
        print("使用方法: python subagent_wrapper.py <command> [args...]")
        print("コマンド:")
        print("  enqueue <task_name> <command>  - タスクをキューに追加")
        print("  execute                       - キューから次のタスクを実行")
        print("  status                        - キューステータス表示")
        print("  cleanup [days]                - 完了タスククリーンアップ")
        print("  kill <task_id>                - 指定タスクを停止")
        print("  kill-all                      - 全実行中タスクを停止")
        print("  list-running                  - 実行中タスク一覧表示")
        print("")
        print("INTG-089 統合コマンド:")
        print("  coordinate <task_name> <cmd>  - 3システム連携実行")
        print("  coord-status                  - 連携システム状態確認")
        print("  coord-shutdown                - 連携システム安全停止")
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

    elif command == "kill":
        if len(sys.argv) < 3:
            print("使用方法: python subagent_wrapper.py kill <task_id>")
            return

        task_id = sys.argv[2]
        success = queue.kill_task(task_id)

        if success:
            print(f"✅ タスク停止成功: {task_id}")
        else:
            print(f"❌ タスク停止失敗: {task_id}")

    elif command == "kill-all":
        killed_count = queue.kill_all_tasks()
        print(f"✅ 全タスク停止完了: {killed_count}件停止")

    elif command == "list-running":
        running = queue.list_running_tasks()
        print("🔄 実行中タスク一覧:")
        print(json.dumps(running, indent=2, ensure_ascii=False))

    # INTG-089: 3システム連携コマンド
    elif command == "coordinate":
        if len(sys.argv) < 4:
            print("使用方法: python subagent_wrapper.py coordinate <task_name> <command>")
            return
        
        task_name = sys.argv[2]
        task_command = " ".join(sys.argv[3:])
        
        # 3システム連携コーディネーター初期化
        coordinator = SubAgentSystemCoordinator(
            workspace_path=str(workspace_path),
            tracker_id=tracker_id
        )
        
        # 連携タスク定義
        coordinated_task = {
            'task_id': f"{tracker_id}_coord_{task_name}_{int(time.time())}",
            'command': task_command,
            'task_type': 'coordinated',
            'gpu_enabled': True,
            'priority': 5,  # 連携タスクは高優先度
            'retry_count': 0
        }
        
        print(f"🚀 3システム連携実行開始: {task_name}")
        result = coordinator.execute_coordinated_task(coordinated_task)
        
        print("📊 連携実行結果:")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        if result['status'] == 'completed':
            print(f"✅ 連携実行成功: {task_name}")
        else:
            print(f"❌ 連携実行失敗: {task_name}")
    
    elif command == "coord-status":
        try:
            coordinator = SubAgentSystemCoordinator(
                workspace_path=str(workspace_path),
                tracker_id=tracker_id
            )
            
            status = coordinator.get_coordination_status()
            print("📈 3システム連携状態:")
            print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
            
        except Exception as e:
            print(f"❌ 連携状態取得エラー: {e}")
    
    elif command == "coord-shutdown":
        try:
            coordinator = SubAgentSystemCoordinator(
                workspace_path=str(workspace_path),
                tracker_id=tracker_id
            )
            
            result = coordinator.shutdown_coordination()
            print("🛑 3システム連携停止結果:")
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            
            if result.get('success'):
                print("✅ 連携システム安全停止完了")
            else:
                print("❌ 連携システム停止エラー")
                
        except Exception as e:
            print(f"❌ 連携システム停止エラー: {e}")

    else:
        print(f"❌ 不明なコマンド: {command}")


if __name__ == "__main__":
    main()