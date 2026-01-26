#!/usr/bin/env python3
"""
実際のGoogle Sheets APIを使ったデータ読み取りツール
batch_task_ticketing.pyと同じAPI接続方法を使用
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))

try:
    from google_sheets_updater import GoogleSheetsUpdater
    from progress_tracker.data_models import PriorityLevel, TaskRecord, TaskStatus

    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ 必要なライブラリが利用できません: {e}")
    DEPS_AVAILABLE = False
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)  # エラーのみ表示
logger = logging.getLogger(__name__)


class SheetsReader:
    """Google Sheetsデータ読み取りクラス（実際のAPI使用）"""

    def __init__(self):
        """初期化"""
        if not DEPS_AVAILABLE:
            raise RuntimeError("必要なライブラリが利用できません")

        # GoogleSheetsUpdater を使用（batch_task_ticketing.pyと同じ方法）
        try:
            self.updater = GoogleSheetsUpdater()
            if not self.updater.service:
                print("❌ Google Sheets API接続失敗")
                print("💡 ヒント: batch_task_ticketing.pyが動作していれば同じ設定を使用します")
                sys.exit(1)

            print(f"✅ Google Sheets API接続成功")
            print(
                f"📊 スプレッドシートURL: https://docs.google.com/spreadsheets/d/{self.updater.spreadsheet_id}/edit"
            )

        except Exception as e:
            print(f"❌ Google Sheets API接続失敗: {e}")
            sys.exit(1)

    def get_all_data(self) -> List[List[str]]:
        """全シートデータを取得"""
        try:
            data = self.updater.get_all_sheet_data()
            print(f"📋 取得データ: {len(data)}行")
            return data
        except Exception as e:
            print(f"❌ データ取得エラー: {e}")
            return []

    def get_filtered_data(self, column_index: int, filter_value: str) -> List[List[str]]:
        """フィルタリング済みデータを取得"""
        try:
            data = self.updater.get_filtered_data(column_index, filter_value)
            print(f"🔍 フィルタリング結果: {len(data)-1}件該当")
            return data
        except Exception as e:
            print(f"❌ フィルタリングエラー: {e}")
            return []

    def get_task_by_id(self, tracker_id: str) -> Optional[List[str]]:
        """特定タスクを取得"""
        try:
            task_data = self.updater.get_task_by_id(tracker_id)
            if task_data:
                print(f"✅ タスク取得成功: {tracker_id}")
            else:
                print(f"⚠️ タスクが見つかりません: {tracker_id}")
            return task_data
        except Exception as e:
            print(f"❌ タスク取得エラー: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        try:
            stats = self.updater.get_statistics()
            print(f"📊 統計計算完了: {stats.get('total_tasks', 0)}件のタスク")
            return stats
        except Exception as e:
            print(f"❌ 統計計算エラー: {e}")
            return {"error": str(e)}

    def display_table(
        self, data: List[List[str]], title: str = "Google Sheetsデータ", limit: Optional[int] = None
    ) -> None:
        """テーブル形式で表示"""
        if not data:
            print("📝 表示するデータがありません")
            return

        display_data = data[:limit] if limit else data

        print(f"\\n📊 {title}")
        print("=" * 120)

        # ヘッダー表示
        if display_data and len(display_data[0]) >= 6:
            print(f"{'ID':<12} {'優先度':<8} {'ステータス':<12} {'登録日':<12} {'概要':<60}")
            print("=" * 120)

            # データ行表示
            for row in display_data[1:]:  # ヘッダー行をスキップ
                if len(row) >= 6:
                    tracker_id = row[0][:11] if len(row[0]) > 11 else row[0]
                    priority = row[1][:7] if len(row[1]) > 7 else row[1]
                    status = row[2][:11] if len(row[2]) > 11 else row[2]
                    date = row[3][:11] if len(row[3]) > 11 else row[3]
                    description = row[5][:57] + "..." if len(row[5]) > 60 else row[5]

                    print(
                        f"{tracker_id:<12} {priority:<8} {status:<12} {date:<12} {description:<60}"
                    )

            if limit and len(data) > limit:
                print(f"\\n... 他 {len(data) - limit} 件があります")

        print("=" * 120)

    def display_task_detail(self, task_data: List[str]) -> None:
        """タスクの詳細表示"""
        print("\\n" + "=" * 80)
        print(f"📋 タスク詳細: {task_data[0] if task_data else 'unknown'}")
        print("=" * 80)

        headers = [
            "トラッカーID",
            "優先度",
            "ステータス",
            "登録日付",
            "更新日付",
            "概要",
            "動作確認",
            "テストUNIT",
            "品質評価",
            "統合実行スクリプト",
            "ダッシュボード生成",
            "抽出パイプライン",
        ]

        for i, header in enumerate(headers):
            value = task_data[i] if i < len(task_data) and task_data[i] else "未設定"
            print(f"{header:<15}: {value}")

        print("=" * 80)

    def display_statistics(self, stats: Dict[str, Any]) -> None:
        """統計情報を表示"""
        print("\\n📊 Google Sheets統計情報（実際のデータ）")
        print("=" * 60)

        if "error" in stats:
            print(f"❌ エラー: {stats['error']}")
            return

        print(f"総タスク数: {stats.get('total_tasks', 0)}")
        print(f"取得日時: {stats.get('timestamp', '不明')}")

        # 優先度別統計
        priority_dist = stats.get("priority_distribution", {})
        if priority_dist:
            print("\\n🎯 優先度別統計:")
            total = stats.get("total_tasks", 1)
            for priority in ["優先度最高", "優先度高", "優先度中", "優先度低"]:
                count = priority_dist.get(priority, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  {priority:<10}: {count:>3}件 ({percentage:>5.1f}%)")

        # ステータス別統計
        status_dist = stats.get("status_distribution", {})
        if status_dist:
            print("\\n📈 ステータス別統計:")
            for status, count in sorted(status_dist.items()):
                percentage = (count / stats.get("total_tasks", 1)) * 100
                print(f"  {status:<15}: {count:>3}件 ({percentage:>5.1f}%)")

        # Phase別統計
        phase_dist = stats.get("phase_distribution", {})
        if phase_dist:
            print("\\n🔄 Phase別統計:")
            for phase, count in sorted(phase_dist.items()):
                percentage = (count / stats.get("total_tasks", 1)) * 100
                print(f"  {phase:<10}: {count:>3}件 ({percentage:>5.1f}%)")

        print("=" * 60)

    def export_to_json(self, data: List[List[str]], output_path: str) -> None:
        """JSON形式でエクスポート"""
        try:
            if not data:
                print("❌ エクスポートするデータがありません")
                return

            # ヘッダー行をキーとして使用
            headers = data[0] if data else []
            json_data = []

            for row in data[1:]:  # ヘッダー行をスキップ
                row_dict = {}
                for i, header in enumerate(headers):
                    value = row[i] if i < len(row) else ""
                    row_dict[header] = value
                json_data.append(row_dict)

            output_data = {
                "metadata": {
                    "source": "Google Sheets API (Real Data)",
                    "spreadsheet_id": self.updater.spreadsheet_id,
                    "exported_at": datetime.now().isoformat(),
                    "total_records": len(json_data),
                },
                "data": json_data,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✅ JSON出力完了: {output_path} ({len(json_data)}件)")

        except Exception as e:
            print(f"❌ JSON出力エラー: {e}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Google Sheets実データ読み取りツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python read_sheets_with_api.py --all --limit 10          # 全データ表示（10件まで）
  python read_sheets_with_api.py --priority 優先度最高     # 優先度フィルタ
  python read_sheets_with_api.py --status 実装完了        # ステータスフィルタ
  python read_sheets_with_api.py --tracker-id PH2-002     # 特定タスク取得
  python read_sheets_with_api.py --stats                  # 統計情報
  python read_sheets_with_api.py --all --json real.json   # JSON出力

注意: 実際のGoogle Sheets APIを使用してリアルタイムでデータを取得します。
        """,
    )

    # 表示オプション
    parser.add_argument("--all", action="store_true", help="全データ表示")
    parser.add_argument("--tracker-id", help="特定トラッカーID検索")
    parser.add_argument("--priority", help="優先度フィルタ（優先度最高/高/中/低）")
    parser.add_argument("--status", help="ステータスフィルタ")
    parser.add_argument("--stats", action="store_true", help="統計情報表示")

    # 出力オプション
    parser.add_argument("--limit", type=int, help="表示件数制限")
    parser.add_argument("--json", help="JSON出力ファイルパス")
    parser.add_argument("--detail", action="store_true", help="詳細表示")

    args = parser.parse_args()

    # 引数チェック
    if not any([args.all, args.tracker_id, args.priority, args.status, args.stats]):
        parser.print_help()
        return

    # SheetsReader初期化
    try:
        reader = SheetsReader()
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        sys.exit(1)

    # データ取得・表示
    try:
        if args.tracker_id:
            # 特定タスク取得
            task_data = reader.get_task_by_id(args.tracker_id)
            if task_data:
                if args.detail:
                    reader.display_task_detail(task_data)
                else:
                    # 1行データをテーブル形式で表示
                    headers = ["トラッカーID", "優先度", "ステータス", "登録日付", "更新日付", "概要"]
                    fake_table = [headers, task_data]
                    reader.display_table(fake_table, f"タスク: {args.tracker_id}")

                if args.json:
                    # 1タスクのJSONエクスポート
                    headers = ["トラッカーID", "優先度", "ステータス", "登録日付", "更新日付", "概要"]
                    fake_table = [headers, task_data]
                    reader.export_to_json(fake_table, args.json)

        else:
            # 複数データ取得
            data = []
            title = "Google Sheetsデータ"

            if args.priority:
                # 優先度フィルタ（B列：インデックス1）
                data = reader.get_filtered_data(1, args.priority)
                title = f"優先度「{args.priority}」のタスク"
            elif args.status:
                # ステータスフィルタ（C列：インデックス2）
                data = reader.get_filtered_data(2, args.status)
                title = f"ステータス「{args.status}」のタスク"
            elif args.all:
                # 全データ取得
                data = reader.get_all_data()
                title = "全タスクデータ"

            # 統計情報表示
            if args.stats:
                stats = reader.get_statistics()
                reader.display_statistics(stats)

                if args.json:
                    with open(args.json, "w", encoding="utf-8") as f:
                        json.dump(stats, f, ensure_ascii=False, indent=2)
                    print(f"✅ 統計情報をJSON出力: {args.json}")
                return

            # データ表示
            if data:
                reader.display_table(data, title, args.limit)

                if args.json:
                    reader.export_to_json(data, args.json)
            else:
                print("📝 該当データがありません")

    except KeyboardInterrupt:
        print("\\n🛑 処理を中断しました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
