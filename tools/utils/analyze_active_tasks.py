#!/usr/bin/env python3
"""
アクティブタスク分析ツール
優先度高トラッカーの分析と実装順序提案
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.workspace_config import WorkspaceConfig
from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.data_models import PriorityLevel, TaskStatus
from tools.progress_tracker.progress_manager import ProgressManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ActiveTaskAnalyzer:
    """アクティブタスク分析クラス"""

    def __init__(self):
        """初期化"""
        try:
            self.config = get_default_config()
            self.manager = ProgressManager(self.config)
            self.workspace_config = WorkspaceConfig()
            logger.info("ActiveTaskAnalyzer初期化完了")
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            raise

    def get_high_priority_tasks(self) -> List[Dict]:
        """優先度高・最高のタスク取得"""
        try:
            all_tasks = self.manager.get_all_tasks()

            high_priority_tasks = []
            for task in all_tasks:
                # 優先度最高または高のタスクを抽出
                if hasattr(task, "priority") and task.priority in [
                    PriorityLevel.HIGHEST,
                    PriorityLevel.HIGH,
                ]:
                    task_dict = {
                        "tracker_id": task.tracker_id,
                        "priority": task.priority.value if hasattr(task, "priority") else "unknown",
                        "status": task.status.value,
                        "description": task.description or "",
                        "created_date": task.created_date.strftime("%Y-%m-%d")
                        if task.created_date
                        else "",
                        "updated_date": task.updated_date.strftime("%Y-%m-%d")
                        if task.updated_date
                        else "",
                        "implementation_status": self._analyze_implementation_status(task),
                    }
                    high_priority_tasks.append(task_dict)

            # 優先度順でソート（最高 > 高）
            priority_order = {"優先度最高": 0, "優先度高": 1}
            high_priority_tasks.sort(key=lambda x: priority_order.get(x["priority"], 999))

            logger.info(f"優先度高以上のタスク取得完了: {len(high_priority_tasks)}件")
            return high_priority_tasks

        except Exception as e:
            logger.error(f"優先度高タスク取得エラー: {e}")
            return []

    def _analyze_implementation_status(self, task) -> Dict[str, str]:
        """実装状況分析"""
        status = {
            "workspace_exists": "unknown",
            "has_code": "unknown",
            "has_tests": "unknown",
            "estimated_progress": "0%",
        }

        try:
            # ワークスペース存在確認
            workspace_path = self.workspace_config.get_tracker_workspace(task.tracker_id)
            if workspace_path.exists():
                status["workspace_exists"] = "yes"

                # 実装コード確認
                if (workspace_path / "extraction").exists():
                    status["has_code"] = "yes"
                    status["estimated_progress"] = "30%"

                # テスト確認
                if (workspace_path / "tests").exists():
                    status["has_tests"] = "yes"
                    status["estimated_progress"] = "60%"

                # ダッシュボード確認
                if (workspace_path / "dashboard" / "dashboard.html").exists():
                    status["estimated_progress"] = "90%"
            else:
                status["workspace_exists"] = "no"

        except Exception as e:
            logger.warning(f"実装状況分析エラー ({task.tracker_id}): {e}")

        return status

    def generate_implementation_roadmap(self, high_priority_tasks: List[Dict]) -> List[Dict]:
        """実装ロードマップ生成"""
        roadmap = []

        # 依存関係とビジネス価値に基づく実装順序
        implementation_priority = {
            "P1-A003": {"order": 1, "reason": "自動テスト強化 - 他の品質保証基盤となるため最優先"},
            "P1-A001": {"order": 2, "reason": "改善コード復旧 - 既存機能の安定化"},
            "P1-A002": {"order": 3, "reason": "品質基準統一 - 評価基準の標準化"},
            "P1-A004": {"order": 4, "reason": "ドキュメント整備 - 全体の文書化完成"},
        }

        for task in high_priority_tasks:
            tracker_id = task["tracker_id"]
            priority_info = implementation_priority.get(
                tracker_id, {"order": 999, "reason": "標準実装順序"}
            )

            # 作業時間見積もり
            estimated_hours = self._estimate_implementation_time(task)

            roadmap_item = {
                "tracker_id": tracker_id,
                "implementation_order": priority_info["order"],
                "priority_reason": priority_info["reason"],
                "current_status": task["status"],
                "estimated_hours": estimated_hours,
                "estimated_progress": task["implementation_status"]["estimated_progress"],
                "next_actions": self._generate_next_actions(task),
                "dependencies": self._analyze_dependencies(tracker_id),
                "risk_level": self._assess_risk_level(task),
            }
            roadmap.append(roadmap_item)

        # 実装順序でソート
        roadmap.sort(key=lambda x: x["implementation_order"])

        logger.info(f"実装ロードマップ生成完了: {len(roadmap)}件")
        return roadmap

    def _estimate_implementation_time(self, task: Dict) -> int:
        """実装時間見積もり（時間）"""
        base_hours = {
            "P1-A001": 16,  # 改善コード復旧
            "P1-A002": 24,  # 品質基準統一
            "P1-A003": 32,  # 自動テスト強化
            "P1-A004": 20,  # ドキュメント整備
        }

        tracker_id = task["tracker_id"]
        estimated = base_hours.get(tracker_id, 20)

        # 進捗による調整
        progress = task["implementation_status"]["estimated_progress"]
        if progress == "30%":
            estimated = int(estimated * 0.7)
        elif progress == "60%":
            estimated = int(estimated * 0.4)
        elif progress == "90%":
            estimated = int(estimated * 0.1)

        return estimated

    def _generate_next_actions(self, task: Dict) -> List[str]:
        """次のアクション生成"""
        tracker_id = task["tracker_id"]
        status = task["status"]

        if status == "着手前":
            actions = [
                f"./tools/scripts/run_quality_workflow.sh {tracker_id} でワークスペース初期化",
                f"要件定義とアーキテクチャ設計",
                f"実装コード作成開始",
            ]
        elif status == "実装中":
            actions = [f"実装継続", f"単体テスト作成", f"統合テスト実行"]
        elif status == "実装完了":
            actions = [f"品質評価実行", f"ダッシュボード生成確認", f"Google Sheets更新: /release"]
        else:
            actions = [f"ステータス確認が必要"]

        return actions

    def _analyze_dependencies(self, tracker_id: str) -> List[str]:
        """依存関係分析"""
        dependencies = {
            "P1-A001": [],  # 依存なし
            "P1-A002": ["P1-A001"],  # 改善コード復旧後に品質基準統一
            "P1-A003": [],  # 独立して実装可能
            "P1-A004": ["P1-A001", "P1-A002", "P1-A003"],  # 全機能実装後にドキュメント化
        }

        return dependencies.get(tracker_id, [])

    def _assess_risk_level(self, task: Dict) -> str:
        """リスクレベル評価"""
        tracker_id = task["tracker_id"]

        # 複雑さとリスクの分析
        risk_factors = {
            "P1-A001": "medium",  # deprecated復旧はテスト必要
            "P1-A002": "high",  # 品質基準統一は影響範囲大
            "P1-A003": "high",  # 自動テスト強化は設計重要
            "P1-A004": "low",  # ドキュメント整備は相対的に低リスク
        }

        return risk_factors.get(tracker_id, "medium")

    def generate_analysis_report(
        self,
        high_priority_tasks: List[Dict],
        roadmap: List[Dict],
        output_path: Optional[str] = None,
    ) -> str:
        """分析レポート生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            output_path = f"analyze_active_tasks_report_{timestamp}.md"

        report_content = f"""# アクティブタスク分析レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析対象**: 優先度高以上のトラッカー

## 📊 概要サマリー

- **優先度最高**: {sum(1 for task in high_priority_tasks if task['priority'] == '優先度最高')}件
- **優先度高**: {sum(1 for task in high_priority_tasks if task['priority'] == '優先度高')}件
- **着手前**: {sum(1 for task in high_priority_tasks if task['status'] == '着手前')}件
- **実装中**: {sum(1 for task in high_priority_tasks if task['status'] == '実装中')}件
- **完了**: {sum(1 for task in high_priority_tasks if task['status'] == '完了')}件

## 🎯 優先度高タスク一覧

"""

        for task in high_priority_tasks:
            report_content += f"""### {task['tracker_id']} - {task['priority']}

- **ステータス**: {task['status']}
- **概要**: {task['description']}
- **登録日**: {task['created_date']}
- **更新日**: {task['updated_date'] or '未更新'}
- **進捗推定**: {task['implementation_status']['estimated_progress']}
- **ワークスペース**: {'✅' if task['implementation_status']['workspace_exists'] == 'yes' else '❌'}

"""

        report_content += f"""## 🗺️ 実装ロードマップ

推奨実装順序:

"""

        for item in roadmap:
            report_content += f"""### {item['implementation_order']}. {item['tracker_id']}

- **理由**: {item['priority_reason']}
- **現在ステータス**: {item['current_status']}
- **推定作業時間**: {item['estimated_hours']}時間
- **リスクレベル**: {item['risk_level']}
- **依存関係**: {', '.join(item['dependencies']) if item['dependencies'] else 'なし'}

**次のアクション**:
"""
            for action in item["next_actions"]:
                report_content += f"- {action}\n"

            report_content += "\n"

        report_content += f"""## 💡 推奨事項

### 即座に実装すべきトラッカー
1. **{roadmap[0]['tracker_id']}**: {roadmap[0]['priority_reason']}

### 実装手順
1. `python3 tools/utils/analyze_active_tasks.py --start {roadmap[0]['tracker_id']}` でタスク開始
2. 品質ワークフロー実行: `./tools/scripts/run_quality_workflow.sh {roadmap[0]['tracker_id']}`
3. 実装完了後にGoogle Sheets更新: `/release`

### 総合的な作業時間見積もり
- **合計**: {sum(item['estimated_hours'] for item in roadmap)}時間
- **並行実装**: 一部タスクは並行実装可能
- **推定期間**: 2-3週間（フルタイム）

---

*このレポートは `tools/utils/analyze_active_tasks.py` により自動生成されました*
"""

        # ファイル出力
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"分析レポート出力完了: {output_path}")
        except Exception as e:
            logger.error(f"レポート出力エラー: {e}")

        return report_content

    def start_tracker_implementation(self, tracker_id: str) -> bool:
        """トラッカー実装開始"""
        try:
            # ステータスを実装中に更新
            task = self.manager.update_task_status(tracker_id, TaskStatus.IN_PROGRESS)

            # ワークスペース初期化
            workspace_path = self.workspace_config.get_tracker_workspace(tracker_id)
            workspace_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"トラッカー実装開始: {tracker_id}")
            return True

        except Exception as e:
            logger.error(f"トラッカー実装開始エラー ({tracker_id}): {e}")
            return False

    def print_console_summary(self, high_priority_tasks: List[Dict], roadmap: List[Dict]):
        """コンソールサマリー表示"""
        print("\n" + "=" * 60)
        print("🎯 アクティブタスク分析サマリー")
        print("=" * 60)

        print(f"\n📊 優先度高以上のタスク: {len(high_priority_tasks)}件")

        # 優先度別表示
        highest_tasks = [t for t in high_priority_tasks if t["priority"] == "優先度最高"]
        high_tasks = [t for t in high_priority_tasks if t["priority"] == "優先度高"]

        if highest_tasks:
            print(f"\n🚨 優先度最高 ({len(highest_tasks)}件):")
            for task in highest_tasks:
                status_emoji = (
                    "🔴" if task["status"] == "着手前" else "🟡" if task["status"] == "実装中" else "🟢"
                )
                print(f"  {status_emoji} {task['tracker_id']}: {task['description'][:50]}...")

        if high_tasks:
            print(f"\n⚠️  優先度高 ({len(high_tasks)}件):")
            for task in high_tasks:
                status_emoji = (
                    "🔴" if task["status"] == "着手前" else "🟡" if task["status"] == "実装中" else "🟢"
                )
                print(f"  {status_emoji} {task['tracker_id']}: {task['description'][:50]}...")

        print(f"\n🗺️  推奨実装順序:")
        for i, item in enumerate(roadmap[:3], 1):  # 上位3件表示
            print(
                f"  {i}. {item['tracker_id']} ({item['estimated_hours']}h) - {item['priority_reason']}"
            )

        if len(roadmap) > 3:
            print(f"  ... 他{len(roadmap)-3}件")

        print(f"\n💡 次のアクション:")
        if roadmap:
            next_tracker = roadmap[0]["tracker_id"]
            print(f"  python3 tools/utils/analyze_active_tasks.py --start {next_tracker}")
            print(f"  ./tools/scripts/run_quality_workflow.sh {next_tracker}")

        print("=" * 60)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="アクティブタスク分析ツール")
    parser.add_argument(
        "--priority", choices=["highest", "high", "all"], default="all", help="分析対象の優先度"
    )
    parser.add_argument("--output", "-o", help="レポート出力ファイルパス")
    parser.add_argument(
        "--format", choices=["console", "markdown", "json"], default="console", help="出力形式"
    )
    parser.add_argument("--start", help="指定トラッカーの実装開始")

    args = parser.parse_args()

    try:
        analyzer = ActiveTaskAnalyzer()

        # トラッカー実装開始
        if args.start:
            success = analyzer.start_tracker_implementation(args.start)
            if success:
                print(f"✅ {args.start} の実装を開始しました")
                return 0
            else:
                print(f"❌ {args.start} の実装開始に失敗しました")
                return 1

        # 高優先度タスク取得
        high_priority_tasks = analyzer.get_high_priority_tasks()

        if not high_priority_tasks:
            print("優先度高以上のタスクが見つかりませんでした")
            return 0

        # 実装ロードマップ生成
        roadmap = analyzer.generate_implementation_roadmap(high_priority_tasks)

        # 出力形式に応じた処理
        if args.format == "console":
            analyzer.print_console_summary(high_priority_tasks, roadmap)

        elif args.format == "markdown":
            report = analyzer.generate_analysis_report(high_priority_tasks, roadmap, args.output)
            if not args.output:
                print(report)

        elif args.format == "json":
            output_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "high_priority_tasks": high_priority_tasks,
                "implementation_roadmap": roadmap,
            }

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"JSON出力完了: {args.output}")
            else:
                print(json.dumps(output_data, indent=2, ensure_ascii=False))

        return 0

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
