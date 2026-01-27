#!/usr/bin/env python3
"""
KIRO-015: システムコーディネーターモジュール
subagent_wrapper.pyから分割
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .enhanced_task_queue import EnhancedSubAgentTaskQueue
from .integrated_system import IntegratedSubAgentSystem
from .task_queue import SubAgentTaskQueue


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
            "started_at": time.time(),
            "tasks_coordinated": 0,
            "anomalies_handled": 0,
            "notifications_sent": 0,
            "checkpoints_created": 0,
            "gpu_fallbacks": 0,
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
                workspace_path=self.workspace_path, tracker_id=self.tracker_id
            )

            # EnhancedSubAgentTaskQueueの初期化
            self.task_queue = EnhancedSubAgentTaskQueue(
                workspace_path=Path(self.workspace_path), tracker_id=self.tracker_id
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
        task_id = task.get("task_id", "unknown")
        self.logger.info(f"Starting coordinated execution: {task_id}")

        try:
            # Phase 1: 事前異常検知
            anomalies = self.monitor.comprehensive_anomaly_check()

            if anomalies["anomaly_count"] > 0:
                self.coordination_stats["anomalies_handled"] += anomalies["anomaly_count"]

                # 重要な異常の場合は通知
                if self._is_critical_anomaly(anomalies):
                    self.notification_bridge.send_anomaly_notification(anomalies)
                    self.coordination_stats["notifications_sent"] += 1

            # Phase 2: GPU fallback判定付きタスク実行
            if task.get("gpu_enabled", True):
                execution_result = self.task_queue.execute_with_gpu_fallback(task)
                if "gpu_fallback" in execution_result.get("execution_stats", {}):
                    self.coordination_stats["gpu_fallbacks"] += 1
            else:
                execution_result = self.task_queue.execute_task(task)

            # Phase 3: 実行中監視（長時間タスクの場合）
            if execution_result.get("status") == "running":
                self._start_coordinated_monitoring(task_id)

            # Phase 4: 結果処理と通知
            if execution_result.get("status") == "completed":
                self.notification_bridge.handle_task_completion(
                    task_id=task_id,
                    task_type=task.get("task_type", "unknown"),
                    results=execution_result.get("execution_stats", {}),
                )
                self.coordination_stats["notifications_sent"] += 1

            elif execution_result.get("status") == "failed":
                self.notification_bridge.handle_task_failure(
                    task_id=task_id,
                    task_type=task.get("task_type", "unknown"),
                    error=execution_result.get("error", "Unknown error"),
                    retry_count=task.get("retry_count", 0),
                    command=task.get("command", ""),
                )
                self.coordination_stats["notifications_sent"] += 1

            # Phase 5: 事後異常検知
            post_anomalies = self.monitor.comprehensive_anomaly_check()
            if post_anomalies["anomaly_count"] > anomalies["anomaly_count"]:
                # 新たな異常が発生
                new_anomaly_count = post_anomalies["anomaly_count"] - anomalies["anomaly_count"]
                self.coordination_stats["anomalies_handled"] += new_anomaly_count

                self.notification_bridge.send_anomaly_notification(post_anomalies)
                self.coordination_stats["notifications_sent"] += 1

            self.coordination_stats["tasks_coordinated"] += 1

            return {
                "status": execution_result.get("status"),
                "execution_result": execution_result,
                "coordination_stats": self.coordination_stats.copy(),
                "anomalies": {"pre_execution": anomalies, "post_execution": post_anomalies},
            }

        except Exception as e:
            self.logger.error(f"Coordinated execution failed: {task_id} - {e}")

            # エラー時の通知
            self.notification_bridge.handle_task_failure(
                task_id=task_id,
                task_type=task.get("task_type", "unknown"),
                error=str(e),
                retry_count=task.get("retry_count", 0),
                command=task.get("command", ""),
            )

            return {
                "status": "error",
                "error": str(e),
                "coordination_stats": self.coordination_stats.copy(),
            }

    def _is_critical_anomaly(self, anomalies: Dict[str, Any]) -> bool:
        """
        重要異常判定

        Args:
            anomalies: 異常検知結果

        Returns:
            bool: 重要異常フラグ
        """
        if anomalies["anomaly_count"] >= 3:
            return True

        for anomaly in anomalies.get("anomalies", []):
            if anomaly.get("type") == "gpu" or "critical" in anomaly.get("message", "").lower():
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
                if not running_tasks.get("tasks", {}).get(task_id):
                    # タスク完了または停止
                    break

                # 30秒ごとに異常検知
                if current_time - last_anomaly_check >= 30:
                    anomalies = self.monitor.comprehensive_anomaly_check()

                    if anomalies["anomaly_count"] > 0:
                        self.coordination_stats["anomalies_handled"] += anomalies["anomaly_count"]

                        # 異常通知（重複防止機能付き）
                        self.notification_bridge.send_anomaly_notification(anomalies)
                        self.coordination_stats["notifications_sent"] += 1

                    last_anomaly_check = current_time

                # 5分ごとにチェックポイント作成
                if current_time - last_checkpoint >= 300:
                    progress_data = {
                        "monitoring_duration": current_time - monitoring_start,
                        "coordination_stats": self.coordination_stats.copy(),
                        "last_anomaly_check": last_anomaly_check,
                        "system_coordination": True,
                    }

                    if self.task_queue.save_checkpoint(task_id, progress_data):
                        self.coordination_stats["checkpoints_created"] += 1

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
                "gpu_monitoring_enabled": self.monitor.gpu_monitoring_enabled,
                "memory_baseline": self.monitor.memory_baseline,
                "anomaly_thresholds": self.monitor.anomaly_thresholds,
            }

            queue_status = self.task_queue.get_queue_status()
            notification_stats = self.notification_bridge.get_notification_stats()

            coordination_runtime = time.time() - self.coordination_stats["started_at"]

            return {
                "coordinator": {
                    "tracker_id": self.tracker_id,
                    "runtime_seconds": coordination_runtime,
                    "coordination_stats": self.coordination_stats,
                },
                "systems": {
                    "monitor": monitor_status,
                    "task_queue": queue_status,
                    "notifications": notification_stats,
                },
                "integration_health": {
                    "all_systems_active": True,
                    "coordination_rate": self.coordination_stats["tasks_coordinated"]
                    / max(coordination_runtime / 3600, 1),  # tasks/hour
                    "anomaly_detection_rate": self.coordination_stats["anomalies_handled"]
                    / max(coordination_runtime / 3600, 1),
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get coordination status: {e}")
            return {"error": str(e), "coordination_stats": self.coordination_stats}

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

            for task_id in running_tasks.get("tasks", {}):
                if self.task_queue.kill_task(task_id):
                    stopped_tasks += 1

            # 最終チェックポイント作成
            final_checkpoint_data = {
                "shutdown_time": time.time(),
                "final_coordination_stats": self.coordination_stats.copy(),
                "stopped_tasks": stopped_tasks,
                "shutdown_type": "coordinated",
            }

            # システム終了処理
            shutdown_duration = time.time() - shutdown_start

            self.logger.info(f"Coordination shutdown completed in {shutdown_duration:.1f}s")

            return {
                "success": True,
                "shutdown_duration": shutdown_duration,
                "stopped_tasks": stopped_tasks,
                "final_stats": self.coordination_stats,
            }

        except Exception as e:
            self.logger.error(f"Coordination shutdown error: {e}")
            return {"success": False, "error": str(e), "partial_stats": self.coordination_stats}


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
            task_id=f"{tracker_id}_{task_name}_{int(time.time())}", command=task_command, priority=3
        )

        if success:
            print(f"✅ タスクエンキュー成功: {task_name}")
        else:
            print(f"❌ タスクエンキュー失敗: {task_name}")

    elif command == "execute":
        result = queue.execute_next_task()
        if result:
            print(f"✅ タスク実行完了: {result['status']}")
            if result.get("output"):
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
            workspace_path=str(workspace_path), tracker_id=tracker_id
        )

        # 連携タスク定義
        coordinated_task = {
            "task_id": f"{tracker_id}_coord_{task_name}_{int(time.time())}",
            "command": task_command,
            "task_type": "coordinated",
            "gpu_enabled": True,
            "priority": 5,  # 連携タスクは高優先度
            "retry_count": 0,
        }

        print(f"🚀 3システム連携実行開始: {task_name}")
        result = coordinator.execute_coordinated_task(coordinated_task)

        print("📊 連携実行結果:")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

        if result["status"] == "completed":
            print(f"✅ 連携実行成功: {task_name}")
        else:
            print(f"❌ 連携実行失敗: {task_name}")

    elif command == "coord-status":
        try:
            coordinator = SubAgentSystemCoordinator(
                workspace_path=str(workspace_path), tracker_id=tracker_id
            )

            status = coordinator.get_coordination_status()
            print("📈 3システム連携状態:")
            print(json.dumps(status, indent=2, ensure_ascii=False, default=str))

        except Exception as e:
            print(f"❌ 連携状態取得エラー: {e}")

    elif command == "coord-shutdown":
        try:
            coordinator = SubAgentSystemCoordinator(
                workspace_path=str(workspace_path), tracker_id=tracker_id
            )

            result = coordinator.shutdown_coordination()
            print("🛑 3システム連携停止結果:")
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

            if result.get("success"):
                print("✅ 連携システム安全停止完了")
            else:
                print("❌ 連携システム停止エラー")

        except Exception as e:
            print(f"❌ 連携システム停止エラー: {e}")

    else:
        print(f"❌ 不明なコマンド: {command}")


if __name__ == "__main__":
    main()
