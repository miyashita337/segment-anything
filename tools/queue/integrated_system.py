#!/usr/bin/env python3
"""
KIRO-015: 統合システムモジュール
subagent_wrapper.pyから分割
"""

import json
import logging
import os
import psutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .enhanced_task_queue import EnhancedSubAgentTaskQueue
from .task_validator import ExtractCharacterTaskValidator


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
        from .notification_bridge import NotificationBridge
        from .subagent_monitor import SubAgentMonitor

        self.monitor = SubAgentMonitor(workspace_path)
        self.notification = NotificationBridge(workspace_path, tracker_id)
        self.task_queue = EnhancedSubAgentTaskQueue(Path(workspace_path), tracker_id)

        # 統合設定
        self.integration_enabled = True
        self.auto_monitoring = True
        self.monitoring_interval = 60  # 1分間隔

        # 統合統計
        self.integration_stats = {
            "system_start_time": datetime.now(timezone.utc).isoformat(),
            "total_monitoring_cycles": 0,
            "anomalies_detected": 0,
            "notifications_sent": 0,
            "tasks_executed": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
        }

        self.logger = self._setup_integration_logger()
        self.logger.info(f"IntegratedSubAgentSystem initialized for {tracker_id}")

    def _setup_integration_logger(self):
        """統合ログ設定"""
        logger = logging.getLogger(f"IntegratedSubAgentSystem.{self.tracker_id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            log_file = (
                Path(self.workspace_path) / ".subagent_queue" / "logs" / "integrated_system.log"
            )
            log_file.parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
            "cycle_id": f"cycle_{int(cycle_start)}",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "anomalies_detected": [],
            "notifications_sent": [],
            "tasks_processed": 0,
            "integration_success": True,
            "errors": [],
        }

        try:
            self.logger.info("Starting integrated monitoring cycle")

            # 1. システム異常検知
            anomalies = self.monitor.comprehensive_anomaly_check()
            cycle_results["anomalies_detected"] = anomalies.get("anomalies", [])

            if anomalies.get("anomaly_count", 0) > 0:
                self.integration_stats["anomalies_detected"] += anomalies["anomaly_count"]

                # 2. 重要度に応じた通知送信
                for anomaly in anomalies.get("anomalies", []):
                    try:
                        notification_result = self.notification.send_enhanced_notification(
                            title=f"System Anomaly Detected: {self.tracker_id}",
                            message=f"Anomaly: {anomaly}",
                            priority_level="high",
                            tracker_id=self.tracker_id,
                        )
                        if notification_result:
                            cycle_results["notifications_sent"].append(
                                {
                                    "anomaly": anomaly,
                                    "notification_sent": True,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            )
                            self.integration_stats["notifications_sent"] += 1
                    except Exception as e:
                        cycle_results["errors"].append(f"Notification error: {e}")

            # 3. タスクキューステータス確認とメンテナンス
            try:
                queue_status = self.task_queue.get_integrated_system_status()

                # チェックポイントクリーンアップ（必要時）
                checkpoint_stats = queue_status.get("checkpoint_system", {})
                if checkpoint_stats.get("total_checkpoints", 0) > 50:
                    cleaned = self.task_queue.cleanup_old_checkpoints(keep_days=3)
                    if cleaned > 0:
                        self.logger.info(f"Cleaned {cleaned} old checkpoints")

            except Exception as e:
                cycle_results["errors"].append(f"Task queue error: {e}")

            # 4. システム統計更新
            self.integration_stats["total_monitoring_cycles"] += 1
            if not cycle_results["errors"]:
                self.integration_stats["successful_integrations"] += 1
            else:
                self.integration_stats["failed_integrations"] += 1

            # 5. サイクル完了
            cycle_results["execution_time"] = time.time() - cycle_start
            cycle_results["end_time"] = datetime.now(timezone.utc).isoformat()

            self.logger.info(
                f"Monitoring cycle completed in {cycle_results['execution_time']:.2f}s"
            )
            return cycle_results

        except Exception as e:
            cycle_results["integration_success"] = False
            cycle_results["errors"].append(f"Critical integration error: {e}")
            self.integration_stats["failed_integrations"] += 1
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
        task_id = task_definition.get("task_id", f"task_{int(time.time())}")
        self.logger.info(f"Starting integrated task execution: {task_id}")

        try:
            # 1. 事前システムチェック
            pre_check = self.monitor.comprehensive_anomaly_check()
            if pre_check.get("anomaly_count", 0) > 2:
                self.notification.send_enhanced_notification(
                    title=f"Pre-execution System Warning: {self.tracker_id}",
                    message=f"System anomalies detected before task {task_id}",
                    priority_level="medium",
                )

            # 2. 統合タスク実行
            result = self.task_queue.execute_with_integrated_features(task_definition)

            # 3. 実行後通知
            if result.get("status") == "completed":
                self.notification.send_enhanced_notification(
                    title=f"Task Completed: {task_id}",
                    message=f"Task execution successful for {self.tracker_id}",
                    priority_level="low",
                )
            elif result.get("status") == "failed":
                self.notification.send_enhanced_notification(
                    title=f"Task Failed: {task_id}",
                    message=f"Task execution failed: {result.get('error', 'Unknown error')}",
                    priority_level="high",
                )

            # 4. 統計更新
            self.integration_stats["tasks_executed"] += 1

            return result

        except Exception as e:
            self.logger.error(f"Integrated task execution failed: {task_id} - {e}")

            # エラー通知
            self.notification.send_enhanced_notification(
                title=f"System Integration Error: {task_id}",
                message=f"Critical integration failure: {e}",
                priority_level="critical",
            )

            return {
                "status": "integration_failed",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "system_overview": {
                    "integration_enabled": self.integration_enabled,
                    "auto_monitoring_enabled": self.auto_monitoring,
                    "tracker_id": self.tracker_id,
                    "workspace_path": self.workspace_path,
                    "report_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "component_health": {
                    "monitor": {
                        "anomaly_count": monitor_status.get("anomaly_count", 0),
                        "system_stats": monitor_status.get("system_stats", {}),
                        "last_check": monitor_status.get("timestamp"),
                    },
                    "notification": {
                        "cache_efficiency": f"{notification_stats.get('hit_rate_percent', 0)}%",
                        "total_requests": notification_stats.get("total_requests", 0),
                        "cache_size": notification_stats.get("cache_size", 0),
                    },
                    "task_queue": {
                        "checkpoint_count": queue_status.get("checkpoint_system", {}).get(
                            "total_checkpoints", 0
                        ),
                        "gpu_fallback_enabled": queue_status.get("system_health", {}).get(
                            "gpu_fallback_enabled", False
                        ),
                        "auto_recovery_enabled": queue_status.get("system_health", {}).get(
                            "auto_recovery_enabled", False
                        ),
                    },
                },
                "integration_statistics": dict(self.integration_stats),
                "overall_health_score": self._calculate_health_score(
                    monitor_status, notification_stats, queue_status
                ),
            }

            return health_report

        except Exception as e:
            self.logger.error(f"Failed to get system health report: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    def _calculate_health_score(
        self, monitor_status: Dict, notification_stats: Dict, queue_status: Dict
    ) -> float:
        """
        システム健康度スコア計算

        Returns:
            float: 健康度スコア (0.0-1.0)
        """
        score_components = []

        # モニター健康度 (40%)
        anomaly_count = monitor_status.get("anomaly_count", 0)
        monitor_score = max(0.0, 1.0 - (anomaly_count / 10.0))  # 10個以上の異常で0点
        score_components.append(monitor_score * 0.4)

        # 通知システム効率 (30%)
        cache_hit_rate = notification_stats.get("hit_rate_percent", 0) / 100.0
        score_components.append(cache_hit_rate * 0.3)

        # タスクキュー健康度 (30%)
        queue_health = queue_status.get("system_health", {})
        queue_features = (
            sum(
                [
                    queue_health.get("auto_checkpoint_enabled", False),
                    queue_health.get("gpu_fallback_enabled", False),
                    queue_health.get("auto_recovery_enabled", False),
                ]
            )
            / 3.0
        )
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

            with open(checkpoint_file, "r") as f:
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
            original_command = task.get("command", "")
            if "extract_character.py" in original_command and "--resume" not in original_command:
                resumed_command = original_command + " --resume"
                task["command"] = resumed_command

            task["status"] = "resumed"
            task["resumed_from_checkpoint"] = True
            task["checkpoint_data"] = checkpoint["progress_data"]

            # レジストリー更新
            registry[task_id] = task
            self._save_task_registry(registry)

            self.logger.info(f"Task resumed from checkpoint: {task_id}")

            return {
                "status": "resumed",
                "task_id": task_id,
                "checkpoint_timestamp": checkpoint["timestamp"],
                "command": task["command"],
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
                    self.logger.info(
                        f"Attempting GPU execution (attempt {gpu_retry_count + 1}) - Health: {gpu_health['overall_score']:.1f}"
                    )

                    # 最適化されたGPU環境設定
                    env = self._prepare_gpu_environment(gpu_health)
                    result = self._execute_task_with_env(task, env)

                    if result["status"] == "completed":
                        self.gpu_fallback_stats["successes"] += 1
                        self._record_gpu_success(gpu_health)
                        self.logger.info(f"GPU execution successful: {task['task_id']}")
                        return result

                    elif self._is_gpu_related_error(result.get("error", "")):
                        # GPU関連エラーの詳細解析
                        error_type = self._analyze_gpu_error(result.get("error", ""))
                        self.logger.warning(
                            f"GPU error detected ({error_type}): {result.get('error', '')[:100]}..."
                        )

                        self._record_gpu_error(error_type, result.get("error", ""))
                        gpu_retry_count += 1
                        self.gpu_fallback_stats["attempts"] += 1

                        # GPU健全性を再評価
                        gpu_health = self._comprehensive_gpu_health_check()
                        continue
                    else:
                        # GPU以外のエラー
                        return result

                else:
                    # CPU fallback実行
                    fallback_reason = gpu_health.get("fallback_reason", "GPU unavailable")
                    self.logger.info(
                        f"Executing with CPU fallback (attempt {gpu_retry_count + 1}) - Reason: {fallback_reason}"
                    )

                    # CPU最適化環境設定
                    env = self._prepare_cpu_environment()
                    result = self._execute_task_with_env(task, env)

                    if result["status"] == "completed":
                        self.gpu_fallback_stats["successes"] += 1
                        self.gpu_fallback_stats["cpu_fallback_count"] += 1
                        self.logger.info(f"CPU fallback execution successful: {task['task_id']}")
                        return result
                    else:
                        gpu_retry_count += 1
                        self.gpu_fallback_stats["failures"] += 1

                        if gpu_retry_count <= self.max_gpu_retry:
                            wait_time = min(2**gpu_retry_count, 60)  # 最大60秒のexponential backoff
                            self.logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)

                    if result["status"] == "completed":
                        self.gpu_fallback_stats["successes"] += 1
                        self.logger.info(f"CPU fallback execution successful: {task['task_id']}")
                        return result
                    else:
                        gpu_retry_count += 1
                        self.gpu_fallback_stats["failures"] += 1

                        if gpu_retry_count <= self.max_gpu_retry:
                            wait_time = 2**gpu_retry_count  # 指数バックオフ
                            self.logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)

            except Exception as e:
                self.logger.error(f"GPU fallback execution error: {e}")
                gpu_retry_count += 1
                self.gpu_fallback_stats["failures"] += 1

        # 最大試行回数到達
        execution_time = time.time() - start_time

        return {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output": None,
            "error": f"GPU fallback failed after {self.max_gpu_retry + 1} attempts",
            "execution_stats": {
                "execution_time": execution_time,
                "gpu_fallback_attempts": gpu_retry_count,
                "gpu_fallback_failed": True,
            },
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
                env=env,
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
                    "env_type": "gpu" if env.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
                },
            }

        except Exception as e:
            # エラー時も実行中タスクから削除
            self._remove_running_task(task["task_id"])
            return {
                "status": "error",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": None,
                "error": str(e),
                "execution_stats": {"execution_time": time.time() - start_time, "exception": True},
            }

    def _check_gpu_available(self) -> bool:
        """GPU利用可能性チェック"""
        try:
            # nvidia-smiチェック
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
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
        base_status.update(
            {
                "gpu_fallback_enabled": self.gpu_fallback_enabled,
                "gpu_fallback_stats": dict(self.gpu_fallback_stats),
                "checkpoint_count": len(list(self.checkpoint_dir.glob("*_checkpoint.json"))),
                "auto_recovery_enabled": self.auto_recovery_enabled,
                "gpu_available": self._check_gpu_available(),
            }
        )

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
                "pid": pid,
                "start_time": time.time(),
                "last_checkpoint": time.time(),
                "checkpoint_count": 0,
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
                self.logger.info(
                    f"Progress monitoring stopped for task: {task_id} (checkpoints: {monitor_data.get('checkpoint_count', 0)})"
                )

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

            pid = monitor_data["pid"]

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
                    time_since_last = current_time - monitor_data["last_checkpoint"]

                    if time_since_last >= self.checkpoint_interval:
                        # 進捗データ収集
                        progress_data = self._collect_progress_data(task_id, pid, proc)

                        # チェックポイント保存
                        if self.save_checkpoint(task_id, progress_data):
                            monitor_data["last_checkpoint"] = current_time
                            monitor_data["checkpoint_count"] += 1

                            self.logger.info(
                                f"Auto-checkpoint created for task: {task_id} (#{monitor_data['checkpoint_count']})"
                            )

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

    def _collect_progress_data(
        self, task_id: str, pid: int, proc: psutil.Process
    ) -> Dict[str, Any]:
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
            start_time = monitor_data.get("start_time", time.time())
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
                        f"/mnt/c/AItools/lora/train/kiri/tracker-workspace/{self.tracker_id}/extraction/",
                    ]

                for output_dir in workspace_dirs:
                    if Path(output_dir).exists():
                        output_size = sum(
                            f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
                        )
                        break
            except Exception:
                pass

            progress_data = {
                "task_id": task_id,
                "checkpoint_time": time.time(),
                "runtime_seconds": runtime_seconds,
                "system_stats": {
                    "pid": pid,
                    "cpu_percent": cpu_percent,
                    "memory_rss_mb": memory_info.rss // 1024 // 1024,
                    "memory_vms_mb": memory_info.vms // 1024 // 1024,
                },
                "output_stats": {
                    "output_size_bytes": output_size,
                    "output_size_mb": output_size // 1024 // 1024,
                },
                "checkpoint_metadata": {
                    "checkpoint_count": monitor_data.get("checkpoint_count", 0) + 1,
                    "auto_generated": True,
                    "monitoring_interval": self.checkpoint_interval,
                },
            }

            # GPU統計（利用可能な場合）
            try:
                import torch

                if torch.cuda.is_available():
                    progress_data["gpu_stats"] = {
                        "gpu_memory_allocated": torch.cuda.memory_allocated(),
                        "gpu_memory_cached": torch.cuda.memory_reserved(),
                        "gpu_available": True,
                    }
            except Exception:
                progress_data["gpu_stats"] = {"gpu_available": False}

            return progress_data

        except Exception as e:
            self.logger.error(f"Failed to collect progress data for {task_id}: {e}")
            return {
                "task_id": task_id,
                "checkpoint_time": time.time(),
                "error": str(e),
                "checkpoint_metadata": {"auto_generated": True, "error": True},
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

            for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
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
        if (current_time - self.gpu_fallback_stats.get("last_gpu_check", 0)) < 30:
            return {
                "gpu_available": self.gpu_fallback_stats.get("gpu_healthy", False),
                "overall_score": 0.5 if self.gpu_fallback_stats.get("gpu_healthy") else 0.0,
                "cached": True,
            }

        health_info = {
            "gpu_available": False,
            "memory_available_mb": 0,
            "memory_total_mb": 0,
            "temperature_ok": True,
            "driver_ok": True,
            "overall_score": 0.0,
            "fallback_reason": "Unknown",
            "detailed_checks": {},
        }

        try:
            import torch

            if not torch.cuda.is_available():
                health_info["fallback_reason"] = "CUDA not available"
                return health_info

            device_count = torch.cuda.device_count()
            if device_count == 0:
                health_info["fallback_reason"] = "No CUDA devices found"
                return health_info

            # GPU 0の詳細チェック
            device_props = torch.cuda.get_device_properties(0)

            # メモリチェック
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = device_props.total_memory
            memory_available = memory_total - memory_reserved

            health_info["memory_available_mb"] = memory_available // 1024 // 1024
            health_info["memory_total_mb"] = memory_total // 1024 // 1024
            health_info["memory_usage_percent"] = (memory_reserved / memory_total) * 100

            # メモリ不足チェック
            if memory_available < (1024 * 1024 * 1024):  # 1GB未満
                health_info["fallback_reason"] = "Insufficient GPU memory"
                health_info["overall_score"] = 0.2
            else:
                health_info["gpu_available"] = True
                health_info["overall_score"] = min(
                    1.0, memory_available / (2 * 1024 * 1024 * 1024)
                )  # 2GB基準でスコア計算

            # 簡易GPU計算テスト
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, result
                torch.cuda.empty_cache()

                health_info["detailed_checks"]["computation_test"] = True
                if health_info["overall_score"] > 0:
                    health_info["overall_score"] = min(health_info["overall_score"] + 0.2, 1.0)

            except Exception as e:
                health_info["detailed_checks"]["computation_test"] = False
                health_info["detailed_checks"]["computation_error"] = str(e)
                health_info["fallback_reason"] = f"GPU computation test failed: {str(e)[:50]}"
                health_info["overall_score"] *= 0.5

            # 最近のエラー履歴チェック
            recent_errors = [
                e
                for e in self.gpu_fallback_stats.get("gpu_errors", [])
                if (current_time - e.get("timestamp", 0)) < 300
            ]  # 5分以内

            if len(recent_errors) >= 3:
                health_info["fallback_reason"] = "Recent GPU errors detected"
                health_info["overall_score"] *= 0.3

            self.gpu_fallback_stats["last_gpu_check"] = current_time
            self.gpu_fallback_stats["gpu_healthy"] = health_info["overall_score"] > 0.5

        except ImportError:
            health_info["fallback_reason"] = "PyTorch not available"
        except Exception as e:
            health_info["fallback_reason"] = f"GPU check error: {str(e)[:50]}"
            health_info["detailed_checks"]["check_error"] = str(e)

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
        if not gpu_health.get("gpu_available", False):
            return False

        # 健全性スコアによる判定
        health_score = gpu_health.get("overall_score", 0.0)

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
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # メモリ使用量に応じた最適化
        memory_usage = gpu_health.get("memory_usage_percent", 0)
        if memory_usage > 70:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        return env

    def _prepare_cpu_environment(self) -> Dict[str, str]:
        """
        CPU環境変数準備

        Returns:
            Dict[str, str]: 環境変数
        """
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CPU_ONLY"] = "1"
        env["OMP_NUM_THREADS"] = str(min(psutil.cpu_count(), 8))  # CPU使用数制限

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
            "cuda",
            "gpu",
            "out of memory",
            "cublas",
            "cudnn",
            "nvml",
            "device-side assert",
            "cuda runtime error",
            "cuda driver",
            "insufficient memory",
            "gpu memory",
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

        if "out of memory" in error_lower or "insufficient memory" in error_lower:
            return "memory_error"
        elif "cuda runtime error" in error_lower:
            return "runtime_error"
        elif "device-side assert" in error_lower:
            return "assertion_error"
        elif "cuda driver" in error_lower:
            return "driver_error"
        else:
            return "unknown_gpu_error"

    def _record_gpu_error(self, error_type: str, error_message: str) -> None:
        """
        GPUエラー記録

        Args:
            error_type: エラータイプ
            error_message: エラーメッセージ
        """
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "message": error_message[:200],  # メッセージは200文字まで
        }

        if "gpu_errors" not in self.gpu_fallback_stats:
            self.gpu_fallback_stats["gpu_errors"] = []

        self.gpu_fallback_stats["gpu_errors"].append(error_record)

        # エラー履歴は最新の10件まで保持
        if len(self.gpu_fallback_stats["gpu_errors"]) > 10:
            self.gpu_fallback_stats["gpu_errors"] = self.gpu_fallback_stats["gpu_errors"][-10:]

    def _record_gpu_success(self, gpu_health: Dict[str, Any]) -> None:
        """
        GPU成功記録

        Args:
            gpu_health: GPU健全性情報
        """
        # GPU健全性を回復
        self.gpu_fallback_stats["gpu_healthy"] = True

        # 古いエラー履歴をクリア（成功時）
        current_time = time.time()
        self.gpu_fallback_stats["gpu_errors"] = [
            e
            for e in self.gpu_fallback_stats.get("gpu_errors", [])
            if (current_time - e.get("timestamp", 0)) < 300  # 5分以内のエラーのみ保持
        ]


# INTG-089: 3システム統合コーディネーター
