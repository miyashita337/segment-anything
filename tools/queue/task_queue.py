#!/usr/bin/env python3
"""
INTG-087: SubAgent長時間タスクキューシステム
Claude Code SubAgent統合用の長時間タスクキュー・実行制御システム
"""

import json
import logging
import psutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "gpu_healthy": True,
            "last_gpu_check": 0,
            "gpu_errors": [],
            "cpu_fallback_count": 0,
            "memory_fallback_count": 0,
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
        formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def enqueue_task(
        self,
        task_id: str,
        command: str,
        priority: int = 1,
        estimated_duration: int = 300,
        resource_requirements: Optional[Dict] = None,
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
                "execution_stats": {},
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
                cwd="/mnt/c/AItools/segment-anything",
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
                "cpu_percent": current_proc.cpu_percent() if "current_proc" in locals() else 0.0,
                "return_code": result_code,
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
                "execution_stats": execution_stats,
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
                "execution_stats": {"execution_time": time.time() - start_time, "exception": True},
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
                    "/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth",
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
                    task["execution_stats"].get("execution_time", 0) for task in completed_tasks
                ]
                avg_execution_time = sum(execution_times) / len(execution_times)

            return {
                "tracker_id": self.tracker_id,
                "queue_status": {
                    "total_tasks": total_tasks,
                    "queued": queued_count,
                    "completed": completed_count,
                    "failed": failed_count,
                    "success_rate": completed_count / max(total_tasks, 1) * 100,
                },
                "performance": {
                    "average_execution_time": avg_execution_time,
                    "total_execution_time": sum(
                        [
                            task["execution_stats"].get("execution_time", 0)
                            for task in registry.values()
                        ]
                    ),
                },
                "next_task": queue_state["queued_tasks"][0]
                if queue_state["queued_tasks"]
                else None,
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
                        task["completed_at"].replace("Z", "+00:00")
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

            return {"running_count": len(active_tasks), "tasks": active_tasks}

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
                "status": "running",
            }

            with open(self.running_tasks_file, "w") as f:
                json.dump(running_tasks, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"実行中タスク記録エラー: {str(e)}")

    def _remove_running_task(self, task_id: str) -> None:
        """実行中タスク削除"""
        try:
            running_tasks = self._load_running_tasks()
            if task_id in running_tasks:
                del running_tasks[task_id]

                with open(self.running_tasks_file, "w") as f:
                    json.dump(running_tasks, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"実行中タスク削除エラー: {str(e)}")

    def _load_running_tasks(self) -> Dict:
        """実行中タスク読み込み"""
        if self.running_tasks_file.exists():
            try:
                with open(self.running_tasks_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"実行中タスク読み込みエラー: {str(e)}")
        return {}

    def _load_queue_state(self) -> Dict:
        """キュー状態読み込み"""
        if self.queue_state_file.exists():
            with open(self.queue_state_file, "r") as f:
                return json.load(f)
        else:
            return {
                "tracker_id": self.tracker_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "queued_tasks": [],
                "completed_tasks": [],
                "failed_tasks": [],
                "total_tasks": 0,
            }

    def _save_queue_state(self, state: Dict) -> None:
        """キュー状態保存"""
        with open(self.queue_state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_task_registry(self) -> Dict:
        """タスクレジストリー読み込み"""
        if self.task_registry_file.exists():
            with open(self.task_registry_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def _save_task_registry(self, registry: Dict) -> None:
        """タスクレジストリー保存"""
        with open(self.task_registry_file, "w") as f:
            json.dump(registry, f, indent=2)
