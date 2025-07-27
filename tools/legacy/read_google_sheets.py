#!/usr/bin/env python3
"""
Google Sheetsデータ取得スクリプト
進捗管理スプレッドシートの内容を読み取り・表示

既存のGoogle Sheets API設定を使用してスプレッドシートの内容を取得し、
コマンドラインから直接確認できるツール。
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import Counter
from datetime import datetime

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))

try:
    from progress_tracker.config import get_default_config, check_configuration
    from progress_tracker.data_models import TaskRecord, TaskStatus, PriorityLevel
    from progress_tracker.sheets_client import GoogleSheetsClient
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"❌ progress_tracker未利用可能: {e}")
    PROGRESS_TRACKER_AVAILABLE = False
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)  # エラーのみ表示
logger = logging.getLogger(__name__)


class SheetsReader:
    """Google Sheetsデータ読み取りクラス"""
    
    def __init__(self):
        """初期化"""
        if not PROGRESS_TRACKER_AVAILABLE:
            raise RuntimeError("progress_tracker未利用可能")
        
        # 設定チェック
        self.config = get_default_config()
        status = check_configuration()
        
        if not status['config_valid']:
            print("❌ Google Sheets設定に問題があります:")
            for issue in status['issues']:
                print(f"   - {issue}")
            sys.exit(1)
        
        # APIクライアント初期化
        try:
            self.client = GoogleSheetsClient(self.config)
            print(f"✅ Google Sheets API接続成功")
            print(f"📊 スプレッドシートURL: https://docs.google.com/spreadsheets/d/{self.config.spreadsheet_id}/edit")
        except Exception as e:
            print(f"❌ Google Sheets API接続失敗: {e}")
            sys.exit(1)
    
    def get_all_tasks(self) -> List[TaskRecord]:
        """全タスク取得"""
        try:
            tasks = self.client.get_all_tasks()
            print(f"📋 取得タスク数: {len(tasks)}件")
            return tasks
        except Exception as e:
            print(f"❌ タスク取得エラー: {e}")
            return []
    
    def get_task_by_id(self, tracker_id: str) -> Optional[TaskRecord]:
        """トラッカーIDで特定タスク取得"""
        try:
            task = self.client.get_task(tracker_id)
            if task:
                print(f"✅ タスク取得成功: {tracker_id}")
            else:
                print(f"⚠️ タスクが見つかりません: {tracker_id}")
            return task
        except Exception as e:
            print(f"❌ タスク取得エラー: {e}")
            return None
    
    def filter_tasks_by_priority(self, tasks: List[TaskRecord], priority: str) -> List[TaskRecord]:
        """優先度でフィルタリング"""
        try:
            priority_level = PriorityLevel(priority)
            filtered = [task for task in tasks if task.priority == priority_level]
            print(f"🔍 優先度「{priority}」のタスク: {len(filtered)}件")
            return filtered
        except ValueError:
            print(f"❌ 無効な優先度: {priority}")
            print(f"有効な優先度: {[p.value for p in PriorityLevel]}")
            return []
    
    def filter_tasks_by_status(self, tasks: List[TaskRecord], status: str) -> List[TaskRecord]:
        """ステータスでフィルタリング"""
        try:
            status_enum = TaskStatus(status)
            filtered = [task for task in tasks if task.status == status_enum]
            print(f"🔍 ステータス「{status}」のタスク: {len(filtered)}件")
            return filtered
        except ValueError:
            print(f"❌ 無効なステータス: {status}")
            print(f"有効なステータス: {[s.value for s in TaskStatus]}")
            return []
    
    def display_tasks_table(self, tasks: List[TaskRecord], limit: Optional[int] = None) -> None:
        """タスクをテーブル形式で表示"""
        if not tasks:
            print("📝 表示するタスクがありません")
            return
        
        # 制限適用
        display_tasks = tasks[:limit] if limit else tasks
        
        # テーブルヘッダー
        print("\\n" + "="*120)
        print(f"{'ID':<12} {'優先度':<8} {'ステータス':<12} {'登録日':<12} {'概要':<60}")
        print("="*120)
        
        # タスク表示
        for task in display_tasks:
            tracker_id = task.tracker_id[:11] if len(task.tracker_id) > 11 else task.tracker_id
            priority = task.priority.value[:7] if len(task.priority.value) > 7 else task.priority.value
            status = task.status.value[:11] if len(task.status.value) > 11 else task.status.value
            
            # 登録日フォーマット
            if task.created_date:
                date_str = task.created_date.strftime("%Y-%m-%d")
            else:
                date_str = "未設定"
            
            # 概要（60文字制限）
            description = task.description[:57] + "..." if len(task.description) > 60 else task.description
            
            print(f"{tracker_id:<12} {priority:<8} {status:<12} {date_str:<12} {description:<60}")
        
        if limit and len(tasks) > limit:
            print(f"\\n... 他 {len(tasks) - limit} 件のタスクがあります")
        
        print("="*120)
    
    def display_task_detail(self, task: TaskRecord) -> None:
        """タスクの詳細表示"""
        print("\\n" + "="*80)
        print(f"📋 タスク詳細: {task.tracker_id}")
        print("="*80)
        
        print(f"優先度     : {task.priority.value}")
        print(f"ステータス : {task.status.value}")
        print(f"登録日     : {task.created_date.strftime('%Y-%m-%d %H:%M:%S') if task.created_date else '未設定'}")
        print(f"更新日     : {task.updated_date.strftime('%Y-%m-%d %H:%M:%S') if task.updated_date else '未設定'}")
        print(f"概要       : {task.description}")
        
        # コンポーネントステータス
        print("\\n📊 コンポーネントステータス:")
        print(f"  動作確認           : {task.operation_check.value if task.operation_check else '未設定'}")
        print(f"  テストUNIT         : {task.test_unit.value if task.test_unit else '未設定'}")
        print(f"  品質評価           : {task.quality_evaluation.value if task.quality_evaluation else '未設定'}")
        print(f"  統合実行スクリプト : {task.integrated_script.value if task.integrated_script else '未設定'}")
        print(f"  ダッシュボード生成 : {task.dashboard_generation.value if task.dashboard_generation else '未設定'}")
        print(f"  抽出パイプライン   : {task.extraction_pipeline.value if task.extraction_pipeline else '未設定'}")
        
        # メトリクス情報
        if task.metrics:
            print("\\n📈 メトリクス:")
            print(f"  LCA               : {task.metrics.lca if task.metrics.lca is not None else '未設定'}")
            print(f"  A/B評価率         : {task.metrics.ab_evaluation_rate if task.metrics.ab_evaluation_rate is not None else '未設定'}")
            print(f"  FPS               : {task.metrics.fps if task.metrics.fps is not None else '未設定'}")
            print(f"  C以上評価率       : {task.metrics.c_plus_rate if task.metrics.c_plus_rate is not None else '未設定'}")
            print(f"  平均カバレッジ率   : {task.metrics.avg_coverage_rate if task.metrics.avg_coverage_rate is not None else '未設定'}")
            print(f"  平均コンパクトネス : {task.metrics.avg_compactness if task.metrics.avg_compactness is not None else '未設定'}")
            print(f"  平均フィル率       : {task.metrics.avg_fill_rate if task.metrics.avg_fill_rate is not None else '未設定'}")
            print(f"  SCI               : {task.metrics.sci if task.metrics.sci is not None else '未設定'}")
            print(f"  PLA               : {task.metrics.pla if task.metrics.pla is not None else '未設定'}")
            print(f"  PLE               : {task.metrics.ple if task.metrics.ple is not None else '未設定'}")
        
        print("="*80)
    
    def export_to_json(self, tasks: List[TaskRecord], output_path: str) -> None:
        """JSON形式でエクスポート"""
        try:
            data = []
            for task in tasks:
                task_dict = {
                    "tracker_id": task.tracker_id,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "created_date": task.created_date.isoformat() if task.created_date else None,
                    "updated_date": task.updated_date.isoformat() if task.updated_date else None,
                    "description": task.description,
                    "operation_check": task.operation_check.value if task.operation_check else None,
                    "test_unit": task.test_unit.value if task.test_unit else None,
                    "quality_evaluation": task.quality_evaluation.value if task.quality_evaluation else None,
                    "integrated_script": task.integrated_script.value if task.integrated_script else None,
                    "dashboard_generation": task.dashboard_generation.value if task.dashboard_generation else None,
                    "extraction_pipeline": task.extraction_pipeline.value if task.extraction_pipeline else None,
                }
                
                # メトリクス追加
                if task.metrics:
                    task_dict["metrics"] = {
                        "lca": task.metrics.lca,
                        "ab_evaluation_rate": task.metrics.ab_evaluation_rate,
                        "fps": task.metrics.fps,
                        "c_plus_rate": task.metrics.c_plus_rate,
                        "avg_coverage_rate": task.metrics.avg_coverage_rate,
                        "avg_compactness": task.metrics.avg_compactness,
                        "avg_fill_rate": task.metrics.avg_fill_rate,
                        "sci": task.metrics.sci,
                        "pla": task.metrics.pla,
                        "ple": task.metrics.ple
                    }
                else:
                    task_dict["metrics"] = None
                
                data.append(task_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ JSON出力完了: {output_path} ({len(tasks)}件)")
            
        except Exception as e:
            print(f"❌ JSON出力エラー: {e}")
    
    def export_to_csv(self, tasks: List[TaskRecord], output_path: str) -> None:
        """CSV形式でエクスポート"""
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # ヘッダー
                headers = [
                    "トラッカーID", "優先度", "ステータス", "登録日", "更新日", "概要",
                    "動作確認", "テストUNIT", "品質評価", "統合実行スクリプト", 
                    "ダッシュボード生成", "抽出パイプライン",
                    "LCA", "A/B評価率", "FPS", "C以上評価率", "平均カバレッジ率",
                    "平均コンパクトネス", "平均フィル率", "SCI", "PLA", "PLE"
                ]
                writer.writerow(headers)
                
                # データ行
                for task in tasks:
                    row = [
                        task.tracker_id,
                        task.priority.value,
                        task.status.value,
                        task.created_date.strftime('%Y-%m-%d') if task.created_date else '',
                        task.updated_date.strftime('%Y-%m-%d') if task.updated_date else '',
                        task.description,
                        task.operation_check.value if task.operation_check else '',
                        task.test_unit.value if task.test_unit else '',
                        task.quality_evaluation.value if task.quality_evaluation else '',
                        task.integrated_script.value if task.integrated_script else '',
                        task.dashboard_generation.value if task.dashboard_generation else '',
                        task.extraction_pipeline.value if task.extraction_pipeline else '',
                    ]
                    
                    # メトリクス
                    if task.metrics:
                        row.extend([
                            task.metrics.lca or '',
                            task.metrics.ab_evaluation_rate or '',
                            task.metrics.fps or '',
                            task.metrics.c_plus_rate or '',
                            task.metrics.avg_coverage_rate or '',
                            task.metrics.avg_compactness or '',
                            task.metrics.avg_fill_rate or '',
                            task.metrics.sci or '',
                            task.metrics.pla or '',
                            task.metrics.ple or ''
                        ])
                    else:
                        row.extend([''] * 10)
                    
                    writer.writerow(row)
            
            print(f"✅ CSV出力完了: {output_path} ({len(tasks)}件)")
            
        except Exception as e:
            print(f"❌ CSV出力エラー: {e}")
    
    def show_statistics(self, tasks: List[TaskRecord]) -> None:
        """統計情報表示"""
        if not tasks:
            print("📊 統計データなし")
            return
        
        print("\\n" + "="*60)
        print("📊 タスク統計情報")
        print("="*60)
        
        # 基本統計
        print(f"総タスク数: {len(tasks)}")
        
        # 優先度別統計
        priority_counts = Counter(task.priority.value for task in tasks)
        print("\\n🎯 優先度別統計:")
        for priority in ["優先度最高", "優先度高", "優先度中", "優先度低"]:
            count = priority_counts.get(priority, 0)
            percentage = (count / len(tasks)) * 100 if tasks else 0
            print(f"  {priority:<10}: {count:>3}件 ({percentage:>5.1f}%)")
        
        # ステータス別統計
        status_counts = Counter(task.status.value for task in tasks)
        print("\\n📈 ステータス別統計:")
        for status, count in status_counts.most_common():
            percentage = (count / len(tasks)) * 100
            print(f"  {status:<15}: {count:>3}件 ({percentage:>5.1f}%)")
        
        # Phase別統計
        phase_counts = Counter()
        for task in tasks:
            if task.tracker_id.startswith('PH'):
                phase = task.tracker_id.split('-')[0]
                phase_counts[phase] += 1
            elif task.tracker_id.startswith('P1'):
                phase_counts['P1'] += 1
            elif task.tracker_id.startswith('T-'):
                phase_counts['T-series'] += 1
            else:
                phase_counts['Other'] += 1
        
        print("\\n🔄 Phase別統計:")
        for phase, count in sorted(phase_counts.items()):
            percentage = (count / len(tasks)) * 100
            print(f"  {phase:<10}: {count:>3}件 ({percentage:>5.1f}%)")
        
        # 最新更新日
        updated_tasks = [task for task in tasks if task.updated_date]
        if updated_tasks:
            latest_update = max(task.updated_date for task in updated_tasks)
            print(f"\\n🕒 最新更新: {latest_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("="*60)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Google Sheetsデータ取得・表示ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python read_google_sheets.py --all                      # 全タスク表示
  python read_google_sheets.py --all --limit 10           # 最初の10件のみ表示
  python read_google_sheets.py --priority 優先度最高       # 最高優先度のみ
  python read_google_sheets.py --status 実装完了          # 実装完了のみ
  python read_google_sheets.py --tracker-id PH2-002       # 特定ID検索
  python read_google_sheets.py --all --json tasks.json    # JSON出力
  python read_google_sheets.py --all --csv tasks.csv      # CSV出力
  python read_google_sheets.py --stats                    # 統計情報
        """
    )
    
    # 表示オプション
    parser.add_argument("--all", action="store_true", help="全タスク表示")
    parser.add_argument("--tracker-id", help="特定トラッカーID検索")
    parser.add_argument("--priority", help="優先度フィルタ（優先度最高/高/中/低）")
    parser.add_argument("--status", help="ステータスフィルタ")
    parser.add_argument("--stats", action="store_true", help="統計情報表示")
    
    # 出力オプション
    parser.add_argument("--limit", type=int, help="表示件数制限")
    parser.add_argument("--json", help="JSON出力ファイルパス")
    parser.add_argument("--csv", help="CSV出力ファイルパス")
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
            task = reader.get_task_by_id(args.tracker_id)
            if task:
                if args.detail:
                    reader.display_task_detail(task)
                else:
                    reader.display_tasks_table([task])
                    
                # 出力処理
                if args.json:
                    reader.export_to_json([task], args.json)
                if args.csv:
                    reader.export_to_csv([task], args.csv)
        
        else:
            # 複数タスク取得
            tasks = reader.get_all_tasks()
            
            # フィルタリング
            if args.priority:
                tasks = reader.filter_tasks_by_priority(tasks, args.priority)
            
            if args.status:
                tasks = reader.filter_tasks_by_status(tasks, args.status)
            
            # 表示・出力
            if args.stats:
                reader.show_statistics(tasks)
            
            if args.all or args.priority or args.status:
                if args.detail and len(tasks) == 1:
                    reader.display_task_detail(tasks[0])
                else:
                    reader.display_tasks_table(tasks, args.limit)
                
                # 出力処理
                if args.json:
                    reader.export_to_json(tasks, args.json)
                if args.csv:
                    reader.export_to_csv(tasks, args.csv)
        
    except KeyboardInterrupt:
        print("\\n🛑 処理を中断しました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()