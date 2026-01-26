#!/usr/bin/env python3
"""
KIRO-015: 拡張タスクキューモジュール
subagent_wrapper.pyから分割
"""

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_queue import SubAgentTaskQueue


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
            "gpu_attempts": 0,
            "cpu_fallbacks": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
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
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "progress_data": progress_data,
                "tracker_id": self.tracker_id,
            }

            with open(checkpoint_file, "w") as f:
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

            with open(checkpoint_file, "r") as f:
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

    def checkpoint_task_progress(
        self,
        task_id: str,
        current_step: int,
        total_steps: int,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
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
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percent": round((current_step / total_steps) * 100, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_reason": "progress_update",
        }

        if additional_data:
            progress_data["additional_data"] = additional_data

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

            for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
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
            checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.json"))
            total_size = sum(f.stat().st_size for f in checkpoint_files)

            stats = {
                "total_checkpoints": len(checkpoint_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "auto_checkpoint_enabled": self.auto_checkpoint_enabled,
                "checkpoint_interval_seconds": self.checkpoint_interval,
                "active_task_checkpoints": len(self.last_checkpoint_time),
                "checkpoint_directory": str(self.checkpoint_dir),
            }

            # 最新・最古のチェックポイント情報
            if checkpoint_files:
                timestamps = [f.stat().st_mtime for f in checkpoint_files]
                stats["latest_checkpoint"] = datetime.fromtimestamp(max(timestamps)).isoformat()
                stats["oldest_checkpoint"] = datetime.fromtimestamp(min(timestamps)).isoformat()

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get checkpoint statistics: {e}")
            return {"error": str(e)}

    def execute_with_integrated_features(self, task: Dict) -> Dict:
        """
        統合機能付きタスク実行（チェックポイント+GPU fallback+自動復旧）

        Args:
            task: タスク定義

        Returns:
            Dict: 実行結果
        """
        task_id = task["task_id"]
        start_time = time.time()

        self.logger.info(f"Starting integrated task execution: {task_id}")

        try:
            # 1. チェックポイントからの復旧確認
            checkpoint = self.load_checkpoint(task_id)
            if checkpoint and self.auto_recovery_enabled:
                self.logger.info(f"Resuming task from checkpoint: {task_id}")
                # チェックポイントデータをタスクに統合
                task["resume_data"] = checkpoint.get("progress_data", {})

            # 2. GPU fallback付きタスク実行
            result = self.execute_with_gpu_fallback(task)

            # 3. 実行中のチェックポイント保存（成功時）
            if result.get("status") == "completed":
                final_checkpoint = {
                    "execution_completed": True,
                    "result": result,
                    "completion_time": datetime.now(timezone.utc).isoformat(),
                }
                self.save_checkpoint(task_id, final_checkpoint)

            # 4. 統計更新
            self._update_integrated_statistics(result, start_time)

            return result

        except Exception as e:
            # エラー時のチェックポイント保存
            error_checkpoint = {
                "execution_failed": True,
                "error": str(e),
                "failure_time": datetime.now(timezone.utc).isoformat(),
                "recovery_needed": True,
            }
            self.save_checkpoint(task_id, error_checkpoint)

            self.logger.error(f"Integrated task execution failed: {task_id} - {e}")
            return {
                "status": "failed",
                "error": str(e),
                "task_id": task_id,
                "execution_time": time.time() - start_time,
            }

    def _update_integrated_statistics(self, result: Dict, start_time: float) -> None:
        """
        統合実行統計の更新

        Args:
            result: 実行結果
            start_time: 開始時間
        """
        execution_time = time.time() - start_time

        if not hasattr(self, "integrated_stats"):
            self.integrated_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0,
                "checkpoint_resumes": 0,
                "gpu_fallback_uses": 0,
            }

        self.integrated_stats["total_executions"] += 1

        if result.get("status") == "completed":
            self.integrated_stats["successful_executions"] += 1
        else:
            self.integrated_stats["failed_executions"] += 1

        # 平均実行時間の更新
        total_time = (
            self.integrated_stats["average_execution_time"]
            * (self.integrated_stats["total_executions"] - 1)
            + execution_time
        )
        self.integrated_stats["average_execution_time"] = (
            total_time / self.integrated_stats["total_executions"]
        )

        # GPU fallback使用回数
        if result.get("execution_stats", {}).get("gpu_fallback_used"):
            self.integrated_stats["gpu_fallback_uses"] += 1

    def get_integrated_system_status(self) -> Dict[str, Any]:
        """
        統合システム状態取得

        Returns:
            Dict[str, Any]: システム状態情報
        """
        status = {
            "enhanced_features_enabled": True,
            "checkpoint_system": self.get_checkpoint_statistics(),
            "gpu_fallback_stats": dict(self.gpu_fallback_stats),
            "integrated_stats": getattr(self, "integrated_stats", {}),
            "system_health": {
                "auto_checkpoint_enabled": self.auto_checkpoint_enabled,
                "gpu_fallback_enabled": self.gpu_fallback_enabled,
                "auto_recovery_enabled": self.auto_recovery_enabled,
                "max_gpu_retry": self.max_gpu_retry,
                "checkpoint_interval": self.checkpoint_interval,
            },
            "active_monitoring": {
                "active_tasks": len(self.last_checkpoint_time),
                "checkpoint_directory_size_mb": self.get_checkpoint_statistics().get(
                    "total_size_mb", 0
                ),
            },
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
