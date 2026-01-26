#!/usr/bin/env python3
"""
Mock Progress Manager for workflow testing

進捗管理をモックしてテスト可能にするシステム
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TaskStatus(Enum):
    """タスクステータス"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """タスク優先度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ProgressTask:
    """進捗タスク データクラス"""

    task_id: str
    tracker_id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: int = 0
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    assignee: Optional[str] = None
    dependencies: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返却"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data


class MockProgressManager:
    """Mock進捗管理システム"""

    def __init__(self):
        self.tasks: Dict[str, ProgressTask] = {}
        self.progress_log_file = Path("tests/fixtures/mock_progress_log.json")
        self.task_counter = 0
        self.notification_callbacks = []

    def create_task(
        self,
        tracker_id: str,
        title: str,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        estimated_hours: Optional[float] = None,
        assignee: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProgressTask:
        """
        タスク作成

        Args:
            tracker_id: トラッカーID
            title: タスクタイトル
            description: タスク説明
            priority: 優先度
            estimated_hours: 予想作業時間
            assignee: 担当者
            dependencies: 依存タスクID一覧
            tags: タグ一覧
            metadata: メタデータ

        Returns:
            作成されたタスク
        """
        self.task_counter += 1
        task_id = f"TASK-{self.task_counter:04d}"

        task = ProgressTask(
            task_id=task_id,
            tracker_id=tracker_id,
            title=title,
            description=description,
            status=TaskStatus.NOT_STARTED,
            priority=priority,
            created_at=datetime.now(),
            estimated_hours=estimated_hours,
            assignee=assignee,
            dependencies=dependencies or [],
            tags=tags or [],
            metadata=metadata or {},
        )

        self.tasks[task_id] = task
        self._save_progress_log()

        self._notify_callbacks("task_created", task)
        return task

    def start_task(self, task_id: str, assignee: Optional[str] = None) -> bool:
        """
        タスク開始

        Args:
            task_id: タスクID
            assignee: 担当者（指定時は更新）

        Returns:
            開始成功フラグ
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.NOT_STARTED:
            return False

        # 依存関係チェック
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.BLOCKED
                    self._save_progress_log()
                    return False

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        if assignee:
            task.assignee = assignee

        self._save_progress_log()
        self._notify_callbacks("task_started", task)
        return True

    def update_progress(self, task_id: str, percentage: int) -> bool:
        """
        進捗更新

        Args:
            task_id: タスクID
            percentage: 進捗率（0-100）

        Returns:
            更新成功フラグ
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.IN_PROGRESS:
            return False

        task.progress_percentage = max(0, min(100, percentage))

        # 100%完了時は自動的に完了状態に移行
        if task.progress_percentage == 100:
            return self.complete_task(task_id)

        self._save_progress_log()
        self._notify_callbacks("progress_updated", task)
        return True

    def complete_task(self, task_id: str, actual_hours: Optional[float] = None) -> bool:
        """
        タスク完了

        Args:
            task_id: タスクID
            actual_hours: 実際の作業時間

        Returns:
            完了成功フラグ
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status not in [TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED]:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.progress_percentage = 100
        if actual_hours:
            task.actual_hours = actual_hours

        self._save_progress_log()
        self._notify_callbacks("task_completed", task)

        # 依存関係のあるタスクのブロック状態を解除
        self._unblock_dependent_tasks(task_id)

        return True

    def fail_task(self, task_id: str, reason: str) -> bool:
        """
        タスク失敗

        Args:
            task_id: タスクID
            reason: 失敗理由

        Returns:
            失敗設定成功フラグ
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status not in [TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED]:
            return False

        task.status = TaskStatus.FAILED
        if not task.metadata:
            task.metadata = {}
        task.metadata["failure_reason"] = reason
        task.metadata["failed_at"] = datetime.now().isoformat()

        self._save_progress_log()
        self._notify_callbacks("task_failed", task)
        return True

    def cancel_task(self, task_id: str, reason: str = "") -> bool:
        """
        タスクキャンセル

        Args:
            task_id: タスクID
            reason: キャンセル理由

        Returns:
            キャンセル成功フラグ
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.COMPLETED:
            return False

        task.status = TaskStatus.CANCELLED
        if not task.metadata:
            task.metadata = {}
        task.metadata["cancellation_reason"] = reason
        task.metadata["cancelled_at"] = datetime.now().isoformat()

        self._save_progress_log()
        self._notify_callbacks("task_cancelled", task)
        return True

    def get_task(self, task_id: str) -> Optional[ProgressTask]:
        """タスク取得"""
        return self.tasks.get(task_id)

    def get_tasks_by_tracker(self, tracker_id: str) -> List[ProgressTask]:
        """トラッカーID別タスク取得"""
        return [task for task in self.tasks.values() if task.tracker_id == tracker_id]

    def get_tasks_by_status(self, status: TaskStatus) -> List[ProgressTask]:
        """ステータス別タスク取得"""
        return [task for task in self.tasks.values() if task.status == status]

    def get_tasks_by_assignee(self, assignee: str) -> List[ProgressTask]:
        """担当者別タスク取得"""
        return [task for task in self.tasks.values() if task.assignee == assignee]

    def get_active_tasks(self) -> List[ProgressTask]:
        """アクティブなタスク取得"""
        active_statuses = [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED]
        return [task for task in self.tasks.values() if task.status in active_statuses]

    def calculate_tracker_progress(self, tracker_id: str) -> Dict[str, Any]:
        """
        トラッカー全体の進捗計算

        Args:
            tracker_id: トラッカーID

        Returns:
            進捗情報の辞書
        """
        tasks = self.get_tasks_by_tracker(tracker_id)
        if not tasks:
            return {"tracker_id": tracker_id, "progress": 0, "tasks": 0}

        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in tasks if task.status == TaskStatus.FAILED)
        in_progress_tasks = sum(1 for task in tasks if task.status == TaskStatus.IN_PROGRESS)

        # 進捗率計算（完了タスク + 進行中タスクの進捗）
        total_progress = completed_tasks * 100  # 完了タスクは100%
        for task in tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                total_progress += task.progress_percentage

        overall_progress = total_progress / (total_tasks * 100) * 100 if total_tasks > 0 else 0

        return {
            "tracker_id": tracker_id,
            "overall_progress": round(overall_progress, 1),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "not_started_tasks": total_tasks - completed_tasks - failed_tasks - in_progress_tasks,
            "tasks": [task.to_dict() for task in tasks],
        }

    def get_overdue_tasks(self) -> List[ProgressTask]:
        """期限切れタスク取得（簡易実装）"""
        # 実際のシステムでは期限を持つが、モックでは予想時間から推定
        overdue_tasks = []
        now = datetime.now()

        for task in self.tasks.values():
            if task.status != TaskStatus.IN_PROGRESS:
                continue

            if not task.started_at or not task.estimated_hours:
                continue

            expected_completion = task.started_at + timedelta(
                hours=task.estimated_hours * 1.2
            )  # 20%のバッファ
            if now > expected_completion:
                overdue_tasks.append(task)

        return overdue_tasks

    def get_progress_statistics(self) -> Dict[str, Any]:
        """進捗統計情報取得"""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return {"total_tasks": 0}

        stats = {"total_tasks": total_tasks}

        # ステータス別集計
        status_counts = {}
        for status in TaskStatus:
            count = sum(1 for task in self.tasks.values() if task.status == status)
            status_counts[status.value] = count
        stats["by_status"] = status_counts

        # 優先度別集計
        priority_counts = {}
        for priority in TaskPriority:
            count = sum(1 for task in self.tasks.values() if task.priority == priority)
            priority_counts[priority.value] = count
        stats["by_priority"] = priority_counts

        # 作業時間統計
        completed_tasks = [
            task
            for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED and task.actual_hours
        ]
        if completed_tasks:
            actual_hours = [task.actual_hours for task in completed_tasks]
            stats["average_completion_hours"] = sum(actual_hours) / len(actual_hours)

        # 進捗率統計
        active_tasks = [
            task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS
        ]
        if active_tasks:
            progress_percentages = [task.progress_percentage for task in active_tasks]
            stats["average_progress_percentage"] = sum(progress_percentages) / len(
                progress_percentages
            )

        return stats

    def add_notification_callback(self, callback):
        """通知コールバック追加"""
        self.notification_callbacks.append(callback)

    def clear_all_tasks(self):
        """全タスククリア"""
        self.tasks.clear()
        self.task_counter = 0
        if self.progress_log_file.exists():
            self.progress_log_file.unlink()

    def _unblock_dependent_tasks(self, completed_task_id: str):
        """依存タスクのブロック解除"""
        for task in self.tasks.values():
            if (
                task.status == TaskStatus.BLOCKED
                and task.dependencies
                and completed_task_id in task.dependencies
            ):
                # 全ての依存関係が完了しているかチェック
                all_deps_completed = True
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        all_deps_completed = False
                        break

                if all_deps_completed:
                    task.status = TaskStatus.NOT_STARTED  # 開始可能状態に戻す

    def _notify_callbacks(self, event_type: str, task: ProgressTask):
        """通知コールバック実行"""
        for callback in self.notification_callbacks:
            try:
                callback(event_type, task)
            except Exception:
                pass  # コールバック実行エラーは無視

    def _save_progress_log(self):
        """進捗ログファイル保存"""
        self.progress_log_file.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "statistics": self.get_progress_statistics(),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.progress_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


# グローバルモックインスタンス（シングルトン）
_mock_progress_instance = None


def get_mock_progress_manager() -> MockProgressManager:
    """Mock進捗管理システムのシングルトン取得"""
    global _mock_progress_instance
    if _mock_progress_instance is None:
        _mock_progress_instance = MockProgressManager()
    return _mock_progress_instance


def reset_mock_progress_manager():
    """Mock進捗管理システムのリセット"""
    global _mock_progress_instance
    if _mock_progress_instance:
        _mock_progress_instance.clear_all_tasks()
    _mock_progress_instance = None
