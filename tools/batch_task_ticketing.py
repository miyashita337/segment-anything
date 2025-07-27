#!/usr/bin/env python3
"""
残存タスク一括起票スクリプト
PROGRESS_TRACKER.mdから抽出した67個のタスクをGoogle Sheetsに登録
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))
from progress_tracker.data_models import TaskRecord, TaskStatus, PriorityLevel
from google_sheets_updater import GoogleSheetsUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 残存タスクリスト（PROGRESS_TRACKER.mdから抽出）
REMAINING_TASKS = [
    # Phase 1 残存タスク
    {"id": "P1-005", "priority": "HIGH", "title": "自動マスク修正機能", "desc": "マスクエッジ自動スムージング・ノイズ除去"},
    {"id": "P1-006", "priority": "HIGH", "title": "階層的品質評価", "desc": "画像全体→領域→ピクセルレベルの多段階評価"},
    {"id": "P1-007", "priority": "HIGH", "title": "評価説明可能性", "desc": "各評価の根拠・改善点の具体的提示"},
    {"id": "P1-008", "priority": "HIGH", "title": "リアルタイム品質監視", "desc": "処理中の品質指標ライブ表示"},
    {"id": "P1-009", "priority": "HIGH", "title": "異常検知アラート", "desc": "品質低下の自動検出・通知"},
    {"id": "P1-010", "priority": "HIGH", "title": "自動リトライ機能", "desc": "失敗時の自動再実行（最大3回）"},
    {"id": "P1-011", "priority": "HIGH", "title": "処理キュー管理", "desc": "大量画像の効率的処理順序制御"},
    {"id": "P1-012", "priority": "HIGH", "title": "部分処理再開", "desc": "中断箇所からの処理継続"},
    {"id": "P1-013", "priority": "HIGH", "title": "差分処理最適化", "desc": "変更箇所のみの再処理"},
    {"id": "P1-014", "priority": "HIGH", "title": "マルチGPU対応", "desc": "複数GPU並列処理"},
    {"id": "P1-015", "priority": "HIGH", "title": "メモリ使用最適化", "desc": "大規模データセット対応"},
    {"id": "P1-016", "priority": "HIGH", "title": "処理速度改善", "desc": "ボトルネック特定・最適化"},
    
    # Phase 1-A 緊急改善タスク
    {"id": "P1-A001", "priority": "HIGHEST", "title": "改善コード復旧", "desc": "deprecatedから本番環境への復帰"},
    {"id": "P1-A002", "priority": "HIGHEST", "title": "品質基準統一", "desc": "データセット横断的な評価基準"},
    {"id": "P1-A003", "priority": "HIGHEST", "title": "自動テスト強化", "desc": "品質劣化の事前検出"},
    {"id": "P1-A004", "priority": "HIGHEST", "title": "ドキュメント整備", "desc": "実装と仕様の同期"},
    
    # Phase 1-B 品質監視自動化
    {"id": "P1-B001", "priority": "HIGH", "title": "継続的品質監視", "desc": "24時間365日の自動監視"},
    {"id": "P1-B002", "priority": "HIGH", "title": "品質ダッシュボード", "desc": "リアルタイム可視化"},
    {"id": "P1-B003", "priority": "HIGH", "title": "自動改善提案", "desc": "AIによる改善策生成"},
    
    # Phase 2 システム安定化
    {"id": "PH2-003", "priority": "MEDIUM", "title": "エラーハンドリング強化", "desc": "6種類カスタムエラークラス実装"},
    {"id": "PH2-004", "priority": "MEDIUM", "title": "リソース管理最適化", "desc": "CPU/メモリ/GPU使用率改善"},
    {"id": "PH2-005", "priority": "MEDIUM", "title": "スケーラビリティ向上", "desc": "4種類並列処理エンジン"},
    {"id": "PH2-006", "priority": "MEDIUM", "title": "監視システム構築", "desc": "性能メトリクス収集・分析"},
    {"id": "PH2-007", "priority": "MEDIUM", "title": "バックアップ機能", "desc": "処理結果の自動バックアップ"},
    {"id": "PH2-008", "priority": "MEDIUM", "title": "復旧機能強化", "desc": "障害時の自動復旧"},
    
    # Phase 3 設定ベース管理
    {"id": "PH3-001", "priority": "MEDIUM", "title": "設定ファイル統一", "desc": "YAML/TOML形式の統一設定"},
    {"id": "PH3-002", "priority": "MEDIUM", "title": "環境別設定", "desc": "開発/本番環境の切り替え"},
    {"id": "PH3-003", "priority": "MEDIUM", "title": "動的設定更新", "desc": "再起動不要の設定変更"},
    {"id": "PH3-004", "priority": "MEDIUM", "title": "設定バリデーション", "desc": "設定値の妥当性検証"},
    {"id": "PH3-005", "priority": "LOW", "title": "設定履歴管理", "desc": "変更履歴の追跡"},
    {"id": "PH3-006", "priority": "LOW", "title": "設定テンプレート", "desc": "用途別設定プリセット"},
    {"id": "PH3-007", "priority": "LOW", "title": "設定マイグレーション", "desc": "バージョン間の設定移行"},
    {"id": "PH3-008", "priority": "LOW", "title": "設定ドキュメント", "desc": "全設定項目の詳細説明"},
    
    # Phase 4 自律的開発ループ
    {"id": "PH4-001", "priority": "LOW", "title": "自動実験管理", "desc": "パラメータ探索・結果記録"},
    {"id": "PH4-002", "priority": "LOW", "title": "A/Bテスト自動化", "desc": "複数手法の比較評価"},
    {"id": "PH4-003", "priority": "LOW", "title": "ハイパーパラメータ最適化", "desc": "ベイズ最適化による自動調整"},
    {"id": "PH4-004", "priority": "LOW", "title": "モデル自動選択", "desc": "データセット特性に応じた最適モデル"},
    {"id": "PH4-005", "priority": "LOW", "title": "自動特徴エンジニアリング", "desc": "新特徴量の自動生成・評価"},
    {"id": "PH4-006", "priority": "LOW", "title": "継続的学習", "desc": "新データによるモデル更新"},
    {"id": "PH4-007", "priority": "LOW", "title": "ドリフト検出", "desc": "データ分布変化の監視"},
    {"id": "PH4-008", "priority": "LOW", "title": "自動レポート生成", "desc": "週次/月次の性能レポート"},
    {"id": "PH4-009", "priority": "LOW", "title": "知識グラフ構築", "desc": "処理結果の関係性可視化"},
    {"id": "PH4-010", "priority": "LOW", "title": "推論説明機能", "desc": "判断根拠の可視化"},
    {"id": "PH4-011", "priority": "LOW", "title": "フィードバックループ", "desc": "ユーザー評価の自動反映"},
    {"id": "PH4-012", "priority": "LOW", "title": "自己診断機能", "desc": "システム健全性の自動チェック"},
    {"id": "PH4-013", "priority": "LOW", "title": "自動コード生成", "desc": "改善案の自動実装"},
    {"id": "PH4-014", "priority": "LOW", "title": "テスト自動生成", "desc": "カバレッジ向上のためのテスト"},
    {"id": "PH4-015", "priority": "LOW", "title": "ドキュメント自動更新", "desc": "コード変更に追従する文書"},
    {"id": "PH4-016", "priority": "LOW", "title": "完全自律化", "desc": "人間介入最小化の実現"},
    
    # 継続的改善タスク
    {"id": "T-001", "priority": "MEDIUM", "title": "日次進捗追跡", "desc": "毎日の進捗自動レポート"},
    {"id": "T-002", "priority": "MEDIUM", "title": "週次サマリー", "desc": "週間成果のまとめ"},
    {"id": "T-003", "priority": "MEDIUM", "title": "月次マイルストーン", "desc": "月間目標達成状況"},
    {"id": "T-004", "priority": "HIGH", "title": "品質トレンド分析", "desc": "時系列品質変化の追跡"},
    {"id": "T-005", "priority": "HIGH", "title": "パフォーマンス監視", "desc": "処理速度・リソース使用率"},
    {"id": "T-006", "priority": "MEDIUM", "title": "アラート最適化", "desc": "誤検知削減・精度向上"},
    {"id": "T-007", "priority": "MEDIUM", "title": "PDCA自動化", "desc": "改善サイクルの自動実行"},
    {"id": "T-008", "priority": "LOW", "title": "ベンチマーク更新", "desc": "最新手法との比較"},
    {"id": "T-009", "priority": "LOW", "title": "技術調査", "desc": "新技術の評価・導入検討"},
    {"id": "T-010", "priority": "LOW", "title": "コミュニティ連携", "desc": "オープンソース貢献"},
    
    # 追加の運用タスク
    {"id": "T-011", "priority": "MEDIUM", "title": "バックアップ自動化", "desc": "定期バックアップの実行"},
    {"id": "T-012", "priority": "MEDIUM", "title": "ログ分析", "desc": "エラーパターンの特定"},
    {"id": "T-013", "priority": "HIGH", "title": "セキュリティ監査", "desc": "脆弱性の定期チェック"},
    {"id": "T-014", "priority": "MEDIUM", "title": "依存関係更新", "desc": "ライブラリの最新化"},
    {"id": "T-015", "priority": "LOW", "title": "コスト最適化", "desc": "クラウドリソース費用削減"},
    {"id": "T-016", "priority": "LOW", "title": "ユーザビリティ改善", "desc": "UI/UXの継続的向上"},
    {"id": "T-017", "priority": "MEDIUM", "title": "API設計改善", "desc": "RESTful API標準化"},
    {"id": "T-018", "priority": "LOW", "title": "国際化対応", "desc": "多言語サポート"},
]


def convert_priority(priority_str: str) -> PriorityLevel:
    """優先度文字列をPriorityLevelに変換"""
    mapping = {
        "HIGHEST": PriorityLevel.HIGHEST,
        "HIGH": PriorityLevel.HIGH,
        "MEDIUM": PriorityLevel.MEDIUM,
        "LOW": PriorityLevel.LOW
    }
    return mapping.get(priority_str, PriorityLevel.MEDIUM)


def create_tasks_from_list() -> List[TaskRecord]:
    """タスクリストからTaskRecordを作成"""
    tasks = []
    today = datetime.now()
    
    for task_def in REMAINING_TASKS:
        task = TaskRecord(
            tracker_id=task_def["id"],
            priority=convert_priority(task_def["priority"]),
            status=TaskStatus.NOT_STARTED,
            created_date=today,
            description=f"{task_def['title']}: {task_def['desc']}"
        )
        tasks.append(task)
    
    return tasks


def batch_register_tasks():
    """タスクを一括でGoogle Sheetsに登録"""
    
    print("残存タスク一括起票")
    print("=" * 60)
    
    try:
        # Google Sheets接続
        updater = GoogleSheetsUpdater()
        if not updater.service:
            print("❌ Google Sheets API未認証")
            return False
        
        # タスクリスト作成
        tasks = create_tasks_from_list()
        print(f"\n起票対象タスク数: {len(tasks)}")
        
        # 優先度別集計
        priority_counts = {}
        for task in tasks:
            priority = task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print("\n優先度別内訳:")
        for priority, count in sorted(priority_counts.items()):
            print(f"  {priority}: {count}件")
        
        # 既存レコードチェック
        print("\n既存レコードチェック中...")
        existing_count = 0
        new_count = 0
        
        sheet_name = "シート1"
        
        for i, task in enumerate(tasks, 1):
            # 進捗表示
            if i % 10 == 0:
                print(f"  処理中... {i}/{len(tasks)}")
            
            # 既存チェック
            existing_row = updater.find_existing_record(task.tracker_id)
            
            # シート更新
            values = [task.to_sheets_row()]
            body = {'values': values}
            
            try:
                if existing_row:
                    # 既存レコードはスキップ（上書きしない）
                    existing_count += 1
                    continue
                else:
                    # 新規レコード追加
                    range_name = f"{sheet_name}!A:V"
                    result = updater.service.spreadsheets().values().append(
                        spreadsheetId=updater.spreadsheet_id,
                        range=range_name,
                        valueInputOption='RAW',
                        body=body
                    ).execute()
                    new_count += 1
                    
            except Exception as e:
                logger.error(f"タスク {task.tracker_id} の登録エラー: {e}")
                continue
        
        # 結果サマリー
        print("\n" + "=" * 60)
        print("起票結果サマリー")
        print("=" * 60)
        print(f"✅ 新規登録: {new_count}件")
        print(f"⏭️ スキップ（既存）: {existing_count}件")
        print(f"📊 合計処理: {len(tasks)}件")
        
        print("\n📊 Google Sheetsで確認:")
        print(f"https://docs.google.com/spreadsheets/d/{updater.spreadsheet_id}/edit")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="残存タスク一括起票")
    parser.add_argument("--dry-run", action="store_true", help="実行せずにタスクリストを表示")
    parser.add_argument("--filter-priority", choices=["HIGHEST", "HIGH", "MEDIUM", "LOW"], 
                       help="特定優先度のタスクのみ起票")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # ドライラン
        tasks = create_tasks_from_list()
        print(f"起票予定タスク数: {len(tasks)}\n")
        
        for task in tasks:
            print(f"{task.tracker_id} [{task.priority.value}] {task.description}")
        
        return
    
    # 実行
    success = batch_register_tasks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()