#!/usr/bin/env python3
"""
タスクの日付を更新するスクリプト
"""

import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.progress_manager import ProgressManager


def update_task_dates():
    """タスクの日付を更新"""
    config = get_default_config()
    manager = ProgressManager(config)

    # Phase 1タスクの完了日付を設定
    phase1_completion_date = datetime(2025, 7, 26)
    phase1_tasks = ["PH1-001", "PH1-002", "PH1-003", "PH1-004"]

    for task_id in phase1_tasks:
        task = manager.get_task(task_id)
        if task:
            task.created_date = phase1_completion_date
            task.updated_date = phase1_completion_date
            manager.client.update_task(task)
            print(f"✅ {task_id}の日付を2025-07-26に更新")

    # PH2-001の日付を設定
    task = manager.get_task("PH2-001")
    if task:
        task.created_date = datetime(2025, 7, 26)
        task.updated_date = datetime(2025, 7, 27)
        manager.client.update_task(task)
        print(f"✅ PH2-001の日付を登録:2025-07-26、更新:2025-07-27に設定")

    # PH2-003の日付を設定
    task = manager.get_task("PH2-003")
    if task:
        task.created_date = datetime(2025, 7, 27)
        task.updated_date = datetime(2025, 7, 27)
        manager.client.update_task(task)
        print(f"✅ PH2-003の日付を2025-07-27に更新")


if __name__ == "__main__":
    update_task_dates()
