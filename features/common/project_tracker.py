#!/usr/bin/env python3
"""
Project Progress Tracker - 人間ラベル学習プロジェクト進捗管理
Integrated TODO management system with automatic progress tracking
"""

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """タスク項目"""

    id: str
    content: str
    status: str  # pending, in_progress, completed
    priority: str  # high, medium, low
    phase: str  # phase0, phase1, phase2, phase3, phase4
    created_at: str
    updated_at: str
    completion_date: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PhaseProgress:
    """Phase進捗情報"""

    phase_id: str
    phase_name: str
    description: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    pending_tasks: int
    progress_percentage: float
    estimated_completion: Optional[str] = None
    key_milestones: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.key_milestones is None:
            self.key_milestones = []


class ProjectTracker:
    """プロジェクト進捗追跡システム"""

    def __init__(self, project_root: Path):
        """
        初期化

        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        self.progress_dir = project_root / "project_progress"
        self.progress_dir.mkdir(exist_ok=True)

        self.tasks_file = self.progress_dir / "tasks.json"
        self.phases_file = self.progress_dir / "phases.json"
        self.milestones_file = self.progress_dir / "milestones.json"

        self.tasks = self.load_tasks()
        self.phases = self.initialize_phases()

    def load_tasks(self) -> List[TaskItem]:
        """タスクデータ読み込み"""
        try:
            if self.tasks_file.exists():
                with open(self.tasks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [TaskItem(**task) for task in data]
            return []
        except Exception as e:
            logger.error(f"タスクデータ読み込みエラー: {e}")
            return []

    def save_tasks(self):
        """タスクデータ保存"""
        try:
            with open(self.tasks_file, "w", encoding="utf-8") as f:
                data = [asdict(task) for task in self.tasks]
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"タスクデータ保存エラー: {e}")

    def initialize_phases(self) -> Dict[str, PhaseProgress]:
        """Phase情報初期化"""
        phases_config = {
            "phase0": {
                "phase_name": "ルールベースベンチマーク",
                "description": "現状性能の定量化と改善基準の確立",
                "key_milestones": [
                    {"name": "ベースライン精度測定完了", "completed": False},
                    {"name": "評価指標システム構築", "completed": False},
                    {"name": "101ファイル性能測定完了", "completed": False},
                ],
            },
            "phase1": {
                "phase_name": "コマ検出ネット構築",
                "description": "「一番大きいコマ」の自動判別システム構築",
                "key_milestones": [
                    {"name": "Mask R-CNN転移学習完了", "completed": False},
                    {"name": "コマ検出mIoU 80%達成", "completed": False},
                    {"name": "推論速度2秒/画像達成", "completed": False},
                ],
            },
            "phase2": {
                "phase_name": "キャラ検出+サイズランキング",
                "description": "コマ内での高精度キャラクター検出",
                "key_milestones": [
                    {"name": "Detectron2+YOLOv8アンサンブル完了", "completed": False},
                    {"name": "mAP@0.5: 85%達成", "completed": False},
                    {"name": "Largest-Character Accuracy 75%達成", "completed": False},
                ],
            },
            "phase3": {
                "phase_name": "主人公判定ヘッド（オプション）",
                "description": "複数キャラクター中の主人公自動識別",
                "key_milestones": [
                    {"name": "CLIP-ViT特徴抽出システム完了", "completed": False},
                    {"name": "主人公識別精度80%達成", "completed": False},
                    {"name": "処理時間増加1秒以下達成", "completed": False},
                ],
            },
            "phase4": {
                "phase_name": "End-to-End微調整",
                "description": "全体最適化による最終精度向上",
                "key_milestones": [
                    {"name": "DETR-type Transformer構築完了", "completed": False},
                    {"name": "Largest-Character Accuracy 85%達成", "completed": False},
                    {"name": "A/B評価率75%達成", "completed": False},
                ],
            },
        }

        phases = {}
        for phase_id, config in phases_config.items():
            phase_tasks = [t for t in self.tasks if t.phase == phase_id]

            total_tasks = len(phase_tasks)
            completed_tasks = len([t for t in phase_tasks if t.status == "completed"])
            in_progress_tasks = len([t for t in phase_tasks if t.status == "in_progress"])
            pending_tasks = len([t for t in phase_tasks if t.status == "pending"])

            progress_percentage = (completed_tasks / max(total_tasks, 1)) * 100

            phases[phase_id] = PhaseProgress(
                phase_id=phase_id,
                phase_name=config["phase_name"],
                description=config["description"],
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                in_progress_tasks=in_progress_tasks,
                pending_tasks=pending_tasks,
                progress_percentage=progress_percentage,
                key_milestones=config["key_milestones"],
            )

        return phases

    def add_task(
        self,
        task_id: str,
        content: str,
        phase: str,
        priority: str = "medium",
        estimated_hours: float = None,
        dependencies: List[str] = None,
    ) -> TaskItem:
        """
        新しいタスクを追加

        Args:
            task_id: タスクID
            content: タスク内容
            phase: 所属Phase
            priority: 優先度
            estimated_hours: 推定工数
            dependencies: 依存タスクID

        Returns:
            作成されたタスク
        """
        now = datetime.datetime.now().isoformat()

        task = TaskItem(
            id=task_id,
            content=content,
            status="pending",
            priority=priority,
            phase=phase,
            created_at=now,
            updated_at=now,
            estimated_hours=estimated_hours,
            dependencies=dependencies or [],
        )

        self.tasks.append(task)
        self.save_tasks()
        self.update_phase_progress()

        logger.info(f"新規タスク追加: {task_id} - {content}")
        return task

    def update_task_status(self, task_id: str, status: str, actual_hours: float = None) -> bool:
        """
        タスクステータス更新

        Args:
            task_id: タスクID
            status: 新しいステータス
            actual_hours: 実際の作業時間

        Returns:
            更新成功フラグ
        """
        for task in self.tasks:
            if task.id == task_id:
                old_status = task.status
                task.status = status
                task.updated_at = datetime.datetime.now().isoformat()

                if actual_hours is not None:
                    task.actual_hours = actual_hours

                if status == "completed":
                    task.completion_date = task.updated_at

                self.save_tasks()
                self.update_phase_progress()

                logger.info(f"タスクステータス更新: {task_id} {old_status} → {status}")
                return True

        logger.warning(f"タスクが見つかりません: {task_id}")
        return False

    def update_phase_progress(self):
        """Phase進捗情報を更新"""
        self.phases = self.initialize_phases()

        # Phase進捗をファイルに保存
        try:
            phases_data = {k: asdict(v) for k, v in self.phases.items()}
            with open(self.phases_file, "w", encoding="utf-8") as f:
                json.dump(phases_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Phase進捗保存エラー: {e}")

    def get_current_phase(self) -> Optional[str]:
        """現在アクティブなPhaseを取得"""
        for phase_id, phase in self.phases.items():
            if phase.in_progress_tasks > 0:
                return phase_id
            if phase.pending_tasks > 0 and phase.completed_tasks == 0:
                return phase_id
        return None

    def get_next_tasks(self, limit: int = 5) -> List[TaskItem]:
        """次に実行すべきタスクを取得"""
        # 依存関係を考慮した実行可能タスク
        executable_tasks = []

        for task in self.tasks:
            if task.status != "pending":
                continue

            # 依存タスクが全て完了しているかチェック
            dependencies_completed = True
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.tasks if t.id == dep_id), None)
                if dep_task is None or dep_task.status != "completed":
                    dependencies_completed = False
                    break

            if dependencies_completed:
                executable_tasks.append(task)

        # 優先度順でソート
        priority_order = {"high": 0, "medium": 1, "low": 2}
        executable_tasks.sort(key=lambda t: priority_order.get(t.priority, 999))

        return executable_tasks[:limit]

    def generate_progress_summary(self) -> Dict[str, Any]:
        """進捗サマリー生成"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == "completed"])
        in_progress_tasks = len([t for t in self.tasks if t.status == "in_progress"])

        overall_progress = (completed_tasks / max(total_tasks, 1)) * 100

        phase_summaries = {}
        for phase_id, phase in self.phases.items():
            phase_summaries[phase_id] = {
                "name": phase.phase_name,
                "progress": phase.progress_percentage,
                "completed_milestones": len([m for m in phase.key_milestones if m["completed"]]),
                "total_milestones": len(phase.key_milestones),
            }

        return {
            "project_name": "人間ラベルデータ活用キャラクター抽出学習",
            "overall_progress": overall_progress,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "current_phase": self.get_current_phase(),
            "phases": phase_summaries,
            "next_tasks": [
                {"id": t.id, "content": t.content, "priority": t.priority}
                for t in self.get_next_tasks(3)
            ],
            "last_updated": datetime.datetime.now().isoformat(),
        }


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)

    # プロジェクトトラッカー初期化
    project_root = Path("/mnt/c/AItools/segment-anything")
    tracker = ProjectTracker(project_root)

    # 既存タスクがない場合は初期タスクを設定
    if not tracker.tasks:
        # Phase 0 タスク
        tracker.add_task("phase0-benchmark", "既存YOLO/SAM+面積最大選択でベースライン測定", "phase0", "high", 8.0)
        tracker.add_task(
            "phase0-metrics",
            "Largest-Character Accuracy指標の確立",
            "phase0",
            "high",
            4.0,
            ["phase0-benchmark"],
        )
        tracker.add_task(
            "phase0-evaluation", "101ファイルでの性能測定システム構築", "phase0", "high", 6.0, ["phase0-metrics"]
        )

        # Phase 1 タスク
        tracker.add_task("phase1-data-prep", "疑似ラベル+人手修正によるデータ拡張", "phase1", "high", 16.0)
        tracker.add_task(
            "phase1-model-setup",
            "Mask R-CNN/YOLOv8-seg転移学習環境構築",
            "phase1",
            "high",
            12.0,
            ["phase0-evaluation"],
        )
        tracker.add_task(
            "phase1-training",
            "COCO→Manga109→自前データの3段階学習",
            "phase1",
            "high",
            24.0,
            ["phase1-data-prep", "phase1-model-setup"],
        )

        print("✅ 初期タスクセットアップ完了")

    # 進捗サマリー表示
    summary = tracker.generate_progress_summary()

    print(f"\n📊 {summary['project_name']}")
    print(f"全体進捗: {summary['overall_progress']:.1f}%")
    print(f"タスク状況: {summary['completed_tasks']}/{summary['total_tasks']} 完了")
    print(f"現在Phase: {summary['current_phase']}")

    print(f"\n🎯 次のタスク:")
    for task in summary["next_tasks"]:
        print(f"  [{task['priority']}] {task['content']}")

    print(f"\n📋 Phase別進捗:")
    for phase_id, phase_data in summary["phases"].items():
        print(f"  {phase_id}: {phase_data['name']} ({phase_data['progress']:.1f}%)")


if __name__ == "__main__":
    main()
