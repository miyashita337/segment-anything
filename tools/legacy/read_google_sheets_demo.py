#!/usr/bin/env python3
"""
Google Sheetsデータ取得デモスクリプト
認証ファイルが利用できない場合のサンプル・デモ機能

Note: 実際のGoogle Sheets APIを使用するには以下が必要です:
1. Google Cloud Consoleでプロジェクト作成
2. Google Sheets API有効化
3. サービスアカウント作成・認証ファイル取得
4. スプレッドシートに編集権限付与
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter

# 既存データモデルから型定義を使用
sys.path.append(str(Path(__file__).parent))

try:
    from progress_tracker.data_models import TaskStatus, PriorityLevel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # フォールバック用の基本定義
    class PriorityLevel:
        HIGHEST = "優先度最高"
        HIGH = "優先度高"
        MEDIUM = "優先度中"
        LOW = "優先度低"
    
    class TaskStatus:
        NOT_STARTED = "着手前"
        IN_PROGRESS = "着手中"
        IMPLEMENTATION = "実装完了"
        QUALITY_CHECK = "品質チェック"
        RELEASE = "リリース"


class SampleTaskRecord:
    """サンプルタスクレコード"""
    
    def __init__(self, tracker_id: str, priority: str, status: str, 
                 description: str, created_date: str = None):
        self.tracker_id = tracker_id
        self.priority = priority
        self.status = status
        self.description = description
        self.created_date = datetime.strptime(created_date, "%Y-%m-%d") if created_date else datetime.now()


class GoogleSheetsDemo:
    """Google Sheetsデモクラス（サンプルデータ使用）"""
    
    def __init__(self):
        """サンプルデータで初期化"""
        self.spreadsheet_url = "https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit"
        # 詳細: docs/google_sheets_reference.md を参照
        self.sample_tasks = self._create_sample_data()
        
        print("📊 Google Sheetsデモモード")
        print(f"🔗 実際のスプレッドシート: {self.spreadsheet_url}")
        print("⚠️  注意: サンプルデータを表示しています")
    
    def _create_sample_data(self) -> List[SampleTaskRecord]:
        """実際の67タスクベースのサンプルデータ作成"""
        return [
            # Phase 1 残存タスク
            SampleTaskRecord("P1-005", "優先度高", "着手前", "自動マスク修正機能: マスクエッジ自動スムージング・ノイズ除去", "2025-07-27"),
            SampleTaskRecord("P1-006", "優先度高", "着手前", "階層的品質評価: 画像全体→領域→ピクセルレベルの多段階評価", "2025-07-27"),
            SampleTaskRecord("P1-007", "優先度高", "実装完了", "評価説明可能性: 各評価の根拠・改善点の具体的提示", "2025-07-21"),
            SampleTaskRecord("P1-008", "優先度高", "着手前", "リアルタイム品質監視: 処理中の品質指標ライブ表示", "2025-07-27"),
            SampleTaskRecord("P1-009", "優先度高", "着手前", "異常検知アラート: 品質低下の自動検出・通知", "2025-07-27"),
            
            # Phase 1-A 緊急改善タスク
            SampleTaskRecord("P1-A001", "優先度最高", "実装完了", "改善コード復旧: deprecatedから本番環境への復帰", "2025-07-27"),
            SampleTaskRecord("P1-A002", "優先度最高", "品質チェック", "品質基準統一: データセット横断的な評価基準", "2025-07-27"),
            SampleTaskRecord("P1-A003", "優先度最高", "着手中", "自動テスト強化: 品質劣化の事前検出", "2025-07-27"),
            
            # Phase 2 システム安定化
            SampleTaskRecord("PH2-001", "優先度中", "リリース", "並列処理エンジン実装: 4種類並列処理対応", "2025-07-26"),
            SampleTaskRecord("PH2-002", "優先度中", "リリース", "スケーラビリティ改善: 大規模データセット対応", "2025-07-27"),
            SampleTaskRecord("PH2-003", "優先度中", "着手前", "エラーハンドリング強化: 6種類カスタムエラークラス実装", "2025-07-27"),
            
            # Phase 3 設定ベース管理
            SampleTaskRecord("PH3-001", "優先度中", "着手前", "設定ファイル統一: YAML/TOML形式の統一設定", "2025-07-27"),
            SampleTaskRecord("PH3-002", "優先度中", "着手前", "環境別設定: 開発/本番環境の切り替え", "2025-07-27"),
            
            # Phase 4 自律的開発ループ
            SampleTaskRecord("PH4-001", "優先度低", "着手前", "自動実験管理: パラメータ探索・結果記録", "2025-07-27"),
            SampleTaskRecord("PH4-002", "優先度低", "着手前", "A/Bテスト自動化: 複数手法の比較評価", "2025-07-27"),
            
            # 継続的改善タスク
            SampleTaskRecord("T-001", "優先度中", "着手中", "日次進捗追跡: 毎日の進捗自動レポート", "2025-07-27"),
            SampleTaskRecord("T-002", "優先度中", "着手前", "週次サマリー: 週間成果のまとめ", "2025-07-27"),
            SampleTaskRecord("T-004", "優先度高", "着手前", "品質トレンド分析: 時系列品質変化の追跡", "2025-07-27"),
            SampleTaskRecord("T-013", "優先度高", "着手前", "セキュリティ監査: 脆弱性の定期チェック", "2025-07-27"),
            
            # テスト用追加データ
            SampleTaskRecord("TEST-001", "優先度中", "品質チェック", "統合テスト検証: 全フロー動作確認", "2025-07-27"),
        ]
    
    def get_all_tasks(self) -> List[SampleTaskRecord]:
        """全タスク取得"""
        print(f"📋 サンプルタスク数: {len(self.sample_tasks)}件")
        return self.sample_tasks
    
    def get_task_by_id(self, tracker_id: str) -> SampleTaskRecord:
        """トラッカーIDで検索"""
        for task in self.sample_tasks:
            if task.tracker_id == tracker_id:
                print(f"✅ タスク取得成功: {tracker_id}")
                return task
        
        print(f"⚠️ タスクが見つかりません: {tracker_id}")
        return None
    
    def filter_tasks_by_priority(self, tasks: List[SampleTaskRecord], priority: str) -> List[SampleTaskRecord]:
        """優先度でフィルタリング"""
        filtered = [task for task in tasks if task.priority == priority]
        print(f"🔍 優先度「{priority}」のタスク: {len(filtered)}件")
        return filtered
    
    def filter_tasks_by_status(self, tasks: List[SampleTaskRecord], status: str) -> List[SampleTaskRecord]:
        """ステータスでフィルタリング"""
        filtered = [task for task in tasks if task.status == status]
        print(f"🔍 ステータス「{status}」のタスク: {len(filtered)}件")
        return filtered
    
    def display_tasks_table(self, tasks: List[SampleTaskRecord], limit: int = None) -> None:
        """タスクをテーブル形式で表示"""
        if not tasks:
            print("📝 表示するタスクがありません")
            return
        
        display_tasks = tasks[:limit] if limit else tasks
        
        print("\\n" + "="*120)
        print(f"{'ID':<12} {'優先度':<8} {'ステータス':<12} {'登録日':<12} {'概要':<60}")
        print("="*120)
        
        for task in display_tasks:
            tracker_id = task.tracker_id[:11] if len(task.tracker_id) > 11 else task.tracker_id
            priority = task.priority[:7] if len(task.priority) > 7 else task.priority
            status = task.status[:11] if len(task.status) > 11 else task.status
            date_str = task.created_date.strftime("%Y-%m-%d")
            description = task.description[:57] + "..." if len(task.description) > 60 else task.description
            
            print(f"{tracker_id:<12} {priority:<8} {status:<12} {date_str:<12} {description:<60}")
        
        if limit and len(tasks) > limit:
            print(f"\\n... 他 {len(tasks) - limit} 件のタスクがあります")
        
        print("="*120)
    
    def display_task_detail(self, task: SampleTaskRecord) -> None:
        """タスクの詳細表示"""
        print("\\n" + "="*80)
        print(f"📋 タスク詳細: {task.tracker_id}")
        print("="*80)
        
        print(f"優先度     : {task.priority}")
        print(f"ステータス : {task.status}")
        print(f"登録日     : {task.created_date.strftime('%Y-%m-%d')}")
        print(f"概要       : {task.description}")
        
        print("\\n💡 注意: これはサンプルデータです")
        print(f"🔗 実際のデータ: {self.spreadsheet_url}")
        
        print("="*80)
    
    def show_statistics(self, tasks: List[SampleTaskRecord]) -> None:
        """統計情報表示"""
        if not tasks:
            print("📊 統計データなし")
            return
        
        print("\\n" + "="*60)
        print("📊 サンプルタスク統計情報")
        print("="*60)
        
        print(f"総タスク数: {len(tasks)}")
        
        # 優先度別統計
        priority_counts = Counter(task.priority for task in tasks)
        print("\\n🎯 優先度別統計:")
        for priority in ["優先度最高", "優先度高", "優先度中", "優先度低"]:
            count = priority_counts.get(priority, 0)
            percentage = (count / len(tasks)) * 100 if tasks else 0
            print(f"  {priority:<10}: {count:>3}件 ({percentage:>5.1f}%)")
        
        # ステータス別統計
        status_counts = Counter(task.status for task in tasks)
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
        
        print("\\n💡 注意: これはサンプルデータです")
        print(f"🔗 実際のスプレッドシート: {self.spreadsheet_url}")
        print("="*60)
    
    def export_to_json(self, tasks: List[SampleTaskRecord], output_path: str) -> None:
        """JSON形式でエクスポート"""
        try:
            data = []
            for task in tasks:
                data.append({
                    "tracker_id": task.tracker_id,
                    "priority": task.priority,
                    "status": task.status,
                    "created_date": task.created_date.isoformat(),
                    "description": task.description,
                    "_note": "This is sample data"
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source": "Google Sheets Demo",
                        "spreadsheet_url": self.spreadsheet_url,
                        "exported_at": datetime.now().isoformat(),
                        "note": "This is sample data for demonstration"
                    },
                    "tasks": data
                }, f, ensure_ascii=False, indent=2)
            
            print(f"✅ JSON出力完了: {output_path} ({len(tasks)}件)")
            print("💡 注意: サンプルデータを出力しました")
            
        except Exception as e:
            print(f"❌ JSON出力エラー: {e}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Google Sheetsデータ取得デモツール（サンプルデータ使用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python read_google_sheets_demo.py --all                    # 全サンプルタスク表示
  python read_google_sheets_demo.py --all --limit 5          # 最初の5件のみ
  python read_google_sheets_demo.py --priority 優先度最高     # 最高優先度のみ
  python read_google_sheets_demo.py --status 実装完了        # 実装完了のみ
  python read_google_sheets_demo.py --tracker-id PH2-002     # 特定ID検索
  python read_google_sheets_demo.py --all --json demo.json   # JSON出力
  python read_google_sheets_demo.py --stats                  # 統計情報

注意: このデモ版はサンプルデータを使用しています。
実際のGoogle Sheetsデータを取得するには認証設定が必要です。
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
    parser.add_argument("--detail", action="store_true", help="詳細表示")
    
    # Google Sheets設定ヘルプ
    parser.add_argument("--setup-help", action="store_true", help="実際のGoogle Sheets設定方法を表示")
    
    args = parser.parse_args()
    
    # 設定ヘルプ表示
    if args.setup_help:
        print("""
🔧 Google Sheets API実際の設定方法

1. Google Cloud Console設定:
   https://console.cloud.google.com/
   - 新しいプロジェクト作成
   - Google Sheets API有効化

2. サービスアカウント作成:
   - IAM & Admin → サービスアカウント
   - JSONキーをダウンロード
   - config/google_sheets_auth.json として配置

3. スプレッドシート権限:
   - サービスアカウントのメールを共有に追加
   - 編集者権限を付与

4. 実行:
   python tools/read_google_sheets.py --all

詳細: tools/progress_tracker/config.py を参照
        """)
        return
    
    # 引数チェック
    if not any([args.all, args.tracker_id, args.priority, args.status, args.stats]):
        parser.print_help()
        return
    
    # デモクラス初期化
    demo = GoogleSheetsDemo()
    
    # データ取得・表示
    try:
        if args.tracker_id:
            # 特定タスク取得
            task = demo.get_task_by_id(args.tracker_id)
            if task:
                if args.detail:
                    demo.display_task_detail(task)
                else:
                    demo.display_tasks_table([task])
                    
                if args.json:
                    demo.export_to_json([task], args.json)
        
        else:
            # 複数タスク取得
            tasks = demo.get_all_tasks()
            
            # フィルタリング
            if args.priority:
                tasks = demo.filter_tasks_by_priority(tasks, args.priority)
            
            if args.status:
                tasks = demo.filter_tasks_by_status(tasks, args.status)
            
            # 表示・出力
            if args.stats:
                demo.show_statistics(tasks)
            
            if args.all or args.priority or args.status:
                if args.detail and len(tasks) == 1:
                    demo.display_task_detail(tasks[0])
                else:
                    demo.display_tasks_table(tasks, args.limit)
                
                if args.json:
                    demo.export_to_json(tasks, args.json)
        
    except KeyboardInterrupt:
        print("\\n🛑 処理を中断しました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()