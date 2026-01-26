#!/usr/bin/env python3
"""
Progress Reporter - プロジェクト進捗可視化レポート生成
Generates comprehensive progress reports and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt

import datetime
import json
import logging
from features.common.project_tracker import ProjectTracker
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressReporter:
    """進捗レポート生成クラス"""

    def __init__(self, project_root: Path):
        """
        初期化

        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        self.tracker = ProjectTracker(project_root)
        self.reports_dir = project_root / "project_progress" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self) -> str:
        """
        Markdownフォーマットの進捗レポート生成

        Returns:
            Markdownテキスト
        """
        summary = self.tracker.generate_progress_summary()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # ヘッダー
        md_content = f"""# 📊 人間ラベルデータ活用プロジェクト 進捗レポート

**生成日時**: {current_date}  
**全体進捗**: {summary['overall_progress']:.1f}%  
**現在Phase**: {summary['current_phase'] or 'N/A'}

---

## 🎯 プロジェクト概要

### 目標
101ファイルの人間ラベルデータを活用して「一番大きいコマのキャラクター抽出」精度を根本的に向上

### 成功基準
- **Largest-Character Accuracy**: 80%以上
- **A/B評価率**: 70%以上
- **処理速度**: 5秒/画像以下

---

## 📈 全体進捗

**タスク完了状況**: {summary['completed_tasks']}/{summary['total_tasks']} ({summary['overall_progress']:.1f}%)
- ✅ 完了: {summary['completed_tasks']}タスク
- 🔄 進行中: {summary['in_progress_tasks']}タスク
- 📋 未着手: {summary['total_tasks'] - summary['completed_tasks'] - summary['in_progress_tasks']}タスク

---

## 🚀 Phase別進捗状況

"""

        # Phase別詳細
        for phase_id, phase_data in summary["phases"].items():
            phase_info = self.tracker.phases[phase_id]

            md_content += f"""### {phase_id.upper()}: {phase_data['name']}

**進捗**: {phase_data['progress']:.1f}%  
**マイルストーン**: {phase_data['completed_milestones']}/{phase_data['total_milestones']} 完了

**説明**: {phase_info.description}

**主要マイルストーン**:
"""
            for milestone in phase_info.key_milestones:
                status = "✅" if milestone["completed"] else "📋"
                md_content += f"- {status} {milestone['name']}\n"

            md_content += "\n"

        # 次のタスク
        md_content += f"""---

## 🎯 次に実行すべきタスク

"""
        for i, task in enumerate(summary["next_tasks"], 1):
            priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📝"}.get(task["priority"], "📝")
            md_content += (
                f"{i}. {priority_emoji} **[{task['priority'].upper()}]** {task['content']}\n"
            )

        # 完了済みタスク
        completed_tasks = [t for t in self.tracker.tasks if t.status == "completed"]
        if completed_tasks:
            md_content += f"""
---

## ✅ 完了済みタスク ({len(completed_tasks)}件)

"""
            for task in completed_tasks[-5:]:  # 最新5件のみ表示
                completion_date = datetime.datetime.fromisoformat(task.completion_date).strftime(
                    "%m/%d"
                )
                md_content += f"- **{completion_date}** {task.content}\n"

        # 技術的メトリクス（利用可能な場合）
        metrics = self.get_technical_metrics()
        if metrics:
            md_content += f"""
---

## 📊 技術的メトリクス

### 現在の性能指標
- **Largest-Character Accuracy**: {metrics.get('largest_char_accuracy', 'N/A')}
- **A/B評価率**: {metrics.get('ab_eval_rate', 'N/A')}
- **平均処理時間**: {metrics.get('avg_processing_time', 'N/A')}

### データセット状況
- **学習データ**: {metrics.get('training_data_count', 'N/A')}ファイル
- **検証データ**: {metrics.get('validation_data_count', 'N/A')}ファイル
"""

        md_content += f"""
---

## 📋 プロジェクト管理情報

### ファイル構成
- **メインプラン**: `HUMAN_LABEL_LEARNING_PROJECT.md`
- **ラベルデータ**: `extracted_labels.json` (101ファイル)
- **分析結果**: `label_analysis/analysis_report.json`
- **進捗データ**: `project_progress/`

### 更新履歴
- **前回更新**: {summary['last_updated'][:16]}
- **次回更新予定**: 次タスク完了時

---

*自動生成レポート - Project Tracker v1.0*
"""

        return md_content

    def get_technical_metrics(self) -> Dict[str, Any]:
        """
        技術的メトリクスの取得

        Returns:
            メトリクス辞書
        """
        try:
            # 各種結果ファイルから性能指標を取得
            metrics = {}

            # ラベル分析結果
            analysis_file = self.project_root / "label_analysis" / "analysis_report.json"
            if analysis_file.exists():
                with open(analysis_file, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
                    metrics["training_data_count"] = (
                        analysis.get("analysis_summary", {})
                        .get("dataset_summary", {})
                        .get("total_labels", "N/A")
                    )

            # 評価結果（存在すれば）
            eval_files = [
                self.project_root / "batch_evaluation_results.json",
                self.project_root / "phase0_evaluation_results.json",  # Phase 0完了後に作成予定
            ]

            for eval_file in eval_files:
                if eval_file.exists():
                    with open(eval_file, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                        metrics.update(
                            {
                                "largest_char_accuracy": f"{eval_data.get('largest_char_accuracy', 0)*100:.1f}%",
                                "ab_eval_rate": f"{eval_data.get('ab_eval_rate', 0)*100:.1f}%",
                                "avg_processing_time": f"{eval_data.get('avg_processing_time', 0):.1f}秒",
                            }
                        )
                        break

            return metrics

        except Exception as e:
            logger.error(f"技術的メトリクス取得エラー: {e}")
            return {}

    def create_progress_visualization(self):
        """進捗可視化チャート作成"""
        try:
            summary = self.tracker.generate_progress_summary()

            # Phase別進捗バーチャート
            plt.figure(figsize=(12, 8))

            phases = list(summary["phases"].keys())
            progress = [summary["phases"][phase]["progress"] for phase in phases]
            phase_names = [summary["phases"][phase]["name"] for phase in phases]

            # カラーマップ
            colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7"]

            plt.subplot(2, 1, 1)
            bars = plt.barh(phase_names, progress, color=colors[: len(phases)])
            plt.xlabel("進捗率 (%)")
            plt.title("Phase別進捗状況")
            plt.xlim(0, 100)

            # 進捗率ラベル
            for i, (bar, prog) in enumerate(zip(bars, progress)):
                plt.text(prog + 2, i, f"{prog:.1f}%", va="center")

            # タスクステータス円グラフ
            plt.subplot(2, 1, 2)
            task_counts = [
                summary["completed_tasks"],
                summary["in_progress_tasks"],
                summary["total_tasks"] - summary["completed_tasks"] - summary["in_progress_tasks"],
            ]
            task_labels = ["完了", "進行中", "未着手"]
            task_colors = ["#2ecc71", "#f39c12", "#95a5a6"]

            plt.pie(
                task_counts,
                labels=task_labels,
                colors=task_colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("タスク状況")

            plt.tight_layout()
            plt.savefig(self.reports_dir / "progress_chart.png", dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("進捗可視化チャート作成完了")

        except Exception as e:
            logger.error(f"可視化チャート作成エラー: {e}")

    def generate_full_report(self) -> Path:
        """
        完全な進捗レポート生成

        Returns:
            生成されたレポートファイルパス
        """
        try:
            # Markdownレポート生成
            md_content = self.generate_markdown_report()

            # ファイル保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            report_file = self.reports_dir / f"progress_report_{timestamp}.md"

            with open(report_file, "w", encoding="utf-8") as f:
                f.write(md_content)

            # 最新レポートとしてもコピー
            latest_report = self.reports_dir / "latest_progress_report.md"
            with open(latest_report, "w", encoding="utf-8") as f:
                f.write(md_content)

            # 進捗可視化チャート作成
            self.create_progress_visualization()

            logger.info(f"完全進捗レポート生成完了: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"完全レポート生成エラー: {e}")
            return None

    def update_master_plan(self):
        """マスタープランファイルの進捗セクション更新"""
        try:
            master_plan_file = self.project_root / "HUMAN_LABEL_LEARNING_PROJECT.md"

            if not master_plan_file.exists():
                logger.warning("マスタープランファイルが見つかりません")
                return

            # 現在の進捗情報取得
            summary = self.tracker.generate_progress_summary()

            # 進捗セクションの更新内容生成
            progress_update = f"""
### 完了済み ✅
"""
            completed_tasks = [t for t in self.tracker.tasks if t.status == "completed"]
            for task in completed_tasks:
                progress_update += f"- [x] {task.content}\n"

            progress_update += f"""
### 進行中 🔄
"""
            in_progress_tasks = [t for t in self.tracker.tasks if t.status == "in_progress"]
            for task in in_progress_tasks:
                progress_update += f"- [ ] {task.content} (進行中)\n"

            progress_update += f"""
### 今後の予定 📋
"""
            next_tasks = self.tracker.get_next_tasks(5)
            for task in next_tasks:
                progress_update += f"- [ ] {task.content}\n"

            logger.info("マスタープラン進捗セクション更新準備完了")

        except Exception as e:
            logger.error(f"マスタープラン更新エラー: {e}")


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)

    # プロジェクトルート
    project_root = Path("/mnt/c/AItools/segment-anything")

    # レポーター初期化
    reporter = ProgressReporter(project_root)

    # 完全レポート生成
    report_file = reporter.generate_full_report()

    if report_file:
        print(f"📊 進捗レポート生成完了: {report_file}")
        print(f"📈 可視化チャート: {reporter.reports_dir / 'progress_chart.png'}")

        # サマリー表示
        summary = reporter.tracker.generate_progress_summary()
        print(f"\n🎯 プロジェクト進捗: {summary['overall_progress']:.1f}%")
        print(f"現在Phase: {summary['current_phase']}")
        print(f"次のタスク: {summary['next_tasks'][0]['content'] if summary['next_tasks'] else 'なし'}")
    else:
        print("❌ レポート生成に失敗しました")


if __name__ == "__main__":
    main()
