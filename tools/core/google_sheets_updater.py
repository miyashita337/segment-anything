#!/usr/bin/env python3
"""
Google Sheets 自動更新システム
品質チェック、抽出パイプライン、評価結果をSpreadsheetに反映
既存progress_tracker設定を活用した統合版
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# progress_tracker設定を利用
sys.path.append(str(Path(__file__).parent.parent))
try:
    from progress_tracker.config import get_default_config, check_configuration
    from progress_tracker.data_models import MetricsRecord, TaskRecord, ComponentStatus, TaskStatus, PriorityLevel
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False
    logging.warning("progress_tracker未利用可能")

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    logging.warning("Google Sheets API未インストール: pip install google-api-python-client google-auth")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleSheetsUpdater:
    """Google Spreadsheet自動更新システム（progress_tracker統合版）"""
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config: ProgressTrackerConfig（省略時はデフォルト設定使用）
        """
        if not PROGRESS_TRACKER_AVAILABLE or not GOOGLE_SHEETS_AVAILABLE:
            logger.error("必要なライブラリが利用できません")
            self.service = None
            return
            
        # progress_tracker設定使用
        self.config = config or get_default_config()
        self.spreadsheet_id = self.config.spreadsheet_id
        self.sheet_name = self.config.sheet_name
        self.service = None
        
        # 設定チェック
        status = check_configuration()
        if not status['config_valid']:
            logger.warning(f"設定に問題があります: {status['issues']}")
            # 認証ファイルが見つからない場合も継続（progress_trackerの設定を試行）
            
        # サービスアカウント認証
        self._authenticate()
        
        # progress_trackerのシート構造を使用
    
    def _authenticate(self):
        """Google Sheets API認証（progress_tracker設定使用）"""
        try:
            # progress_tracker設定から認証ファイルパス取得
            auth_file_path = Path(self.config.auth_file_path)
            
            if not auth_file_path.exists():
                logger.error(f"認証ファイルが見つかりません: {auth_file_path}")
                logger.info("progress_tracker設定手順に従って認証ファイルを配置してください")
                return
            
            # 認証
            scopes = ['https://www.googleapis.com/auth/spreadsheets']
            credentials = Credentials.from_service_account_file(
                str(auth_file_path), scopes=scopes)
            
            self.service = build('sheets', 'v4', credentials=credentials)
            logger.info(f"Google Sheets API認証成功: {self.spreadsheet_id}")
            
        except Exception as e:
            logger.error(f"Google Sheets API認証エラー: {e}")
    
    def find_existing_record(self, tracker_id: str) -> Optional[int]:
        """
        既存レコードの行番号を検索
        
        Args:
            tracker_id: 検索するトラッカーID
            
        Returns:
            Optional[int]: 見つかった行番号（1始まり）、見つからなければNone
        """
        if not self.service:
            return None
            
        try:
            # A列（トラッカーID）を検索
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A:A"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            # トラッカーIDを検索
            for row_index, row in enumerate(values):
                if row and len(row) > 0 and row[0] == tracker_id:
                    return row_index + 1  # 1始まりの行番号
                    
            return None
            
        except Exception as e:
            logger.error(f"既存レコード検索エラー: {e}")
            return None
    
    def get_all_sheet_data(self) -> List[List[str]]:
        """
        全シートデータを取得（23列形式）
        
        Returns:
            List[List[str]]: 全行データ（ヘッダー含む）
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return []
            
        try:
            # 23列（A-W）の全データを取得
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A:W"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            logger.info(f"全シートデータ取得完了: {len(values)}行")
            return values
            
        except Exception as e:
            logger.error(f"全シートデータ取得エラー: {e}")
            return []
    
    def get_filtered_data(self, column_index: int, filter_value: str) -> List[List[str]]:
        """
        指定列でフィルタリングしたデータを取得
        
        Args:
            column_index: フィルタ対象列インデックス（0始まり）
            filter_value: フィルタ値
            
        Returns:
            List[List[str]]: フィルタリング結果（ヘッダー含む）
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return []
            
        try:
            all_data = self.get_all_sheet_data()
            if not all_data:
                return []
            
            # ヘッダー行を保持
            filtered_data = [all_data[0]] if all_data else []
            
            # データ行をフィルタリング
            for row in all_data[1:]:  # ヘッダー行をスキップ
                if (len(row) > column_index and 
                    row[column_index] == filter_value):
                    filtered_data.append(row)
            
            logger.info(f"フィルタリング完了: {len(filtered_data)-1}件該当 (列{column_index}='{filter_value}')")
            return filtered_data
            
        except Exception as e:
            logger.error(f"フィルタリングエラー: {e}")
            return []
    
    def get_task_by_id(self, tracker_id: str) -> Optional[List[str]]:
        """
        トラッカーIDで特定のタスクデータを取得
        
        Args:
            tracker_id: 検索するトラッカーID
            
        Returns:
            Optional[List[str]]: タスクデータ行（見つからない場合はNone）
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return None
            
        try:
            row_number = self.find_existing_record(tracker_id)
            if row_number is None:
                logger.info(f"タスクが見つかりません: {tracker_id}")
                return None
            
            # 特定行のデータを取得（23列）
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A{row_number}:W{row_number}"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if values and values[0]:
                logger.info(f"タスクデータ取得成功: {tracker_id}")
                return values[0]
            
            return None
            
        except Exception as e:
            logger.error(f"タスクデータ取得エラー: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        スプレッドシートデータの統計情報を取得
        
        Returns:
            Dict[str, Any]: 統計情報辞書
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return {}
            
        try:
            all_data = self.get_all_sheet_data()
            if not all_data or len(all_data) <= 1:  # ヘッダーのみ
                return {"total_tasks": 0}
            
            # ヘッダー行を除くデータ行
            data_rows = all_data[1:]
            
            # 基本統計
            stats = {
                "total_tasks": len(data_rows),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 優先度別統計（B列：インデックス1）
            priority_counts = {}
            for row in data_rows:
                if len(row) > 1 and row[1]:
                    priority = row[1]
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
            stats["priority_distribution"] = priority_counts
            
            # ステータス別統計（C列：インデックス2）
            status_counts = {}
            for row in data_rows:
                if len(row) > 2 and row[2]:
                    status = row[2]
                    status_counts[status] = status_counts.get(status, 0) + 1
            stats["status_distribution"] = status_counts
            
            # Phase別統計（A列から推定）
            phase_counts = {}
            for row in data_rows:
                if len(row) > 0 and row[0]:
                    tracker_id = row[0]
                    if tracker_id.startswith('PH'):
                        phase = tracker_id.split('-')[0]
                    elif tracker_id.startswith('P1'):
                        phase = 'P1'
                    elif tracker_id.startswith('T-'):
                        phase = 'T-series'
                    else:
                        phase = 'Other'
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
            stats["phase_distribution"] = phase_counts
            
            logger.info(f"統計情報計算完了: {stats['total_tasks']}件のタスク")
            return stats
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}
    
    def update_task_status(self, tracker_id: str, new_status: str) -> bool:
        """
        特定タスクのステータスを更新
        
        Args:
            tracker_id: 更新対象のトラッカーID
            new_status: 新しいステータス
            
        Returns:
            bool: 更新成功/失敗
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return False
            
        try:
            # タスク行を検索
            row_number = self.find_existing_record(tracker_id)
            if row_number is None:
                logger.error(f"タスクが見つかりません: {tracker_id}")
                return False
            
            # 現在のデータを取得
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A{row_number}:W{row_number}"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            current_row = result.get('values', [[]])[0]
            if not current_row:
                logger.error(f"行データの取得に失敗: {tracker_id}")
                return False
            
            # 23列分を確保
            while len(current_row) < 23:
                current_row.append("")
            
            # ステータス(C列:インデックス2)と更新日付(E列:インデックス4)を更新
            current_row[2] = new_status  # ステータス更新
            current_row[4] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 更新日付
            
            # データ書き戻し
            update_body = {'values': [current_row]}
            
            result = self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=update_body
            ).execute()
            
            logger.info(f"ステータス更新完了: {tracker_id} -> {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"ステータス更新エラー: {e}")
            return False
    
    def update_status(self, tracker_id: str, status: str, 
                     dataset_name: str = "", total_images: int = 0,
                     timestamp: Optional[str] = None) -> bool:
        """
        ステータスシート更新
        
        Args:
            tracker_id: トラッカーID
            status: 処理状態 (処理中/完了/エラー)
            dataset_name: データセット名
            total_images: 総画像数
            timestamp: タイムスタンプ
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return False
            
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 更新データ準備
            values = [[
                tracker_id,
                status,
                dataset_name,
                total_images,
                timestamp
            ]]
            
            # シートに追記
            body = {'values': values}
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheets['status']}!A:E",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"ステータス更新完了: {tracker_id} -> {status}")
            return True
            
        except HttpError as e:
            logger.error(f"ステータス更新エラー: {e}")
            return False
    
    def update_quality_metrics(self, tracker_id: str, quality_data: Dict[str, Any]) -> bool:
        """
        品質指標シート更新
        
        Args:
            tracker_id: トラッカーID
            quality_data: 統合品質レポートJSON
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化")
            return False
            
        try:
            timestamp = quality_data.get('timestamp', 
                                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # 指標値抽出
            metrics_row = [
                tracker_id,
                timestamp,
                quality_data.get('overall_score', 0),
                quality_data.get('total_images', 0),
                quality_data.get('passed_metrics', 0),
                quality_data.get('total_metrics', 0)
            ]
            
            # 各カテゴリの指標値追加
            for category in ['evaluation_metrics', 'mask_metrics', 'objective_metrics']:
                metrics = quality_data.get(category, [])
                for metric in metrics:
                    metrics_row.append(metric.get('value', 0))
            
            values = [metrics_row]
            body = {'values': values}
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheets['quality']}!A:Z",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"品質指標更新完了: {tracker_id}")
            return True
            
        except HttpError as e:
            logger.error(f"品質指標更新エラー: {e}")
            return False
    
    def update_processing_log(self, tracker_id: str, log_type: str, 
                            message: str, details: Optional[Dict] = None) -> bool:
        """
        処理ログシート更新
        
        Args:
            tracker_id: トラッカーID
            log_type: ログタイプ (INFO/WARNING/ERROR)
            message: ログメッセージ
            details: 詳細情報
        """
        if not self.service:
            logger.warning("Google Sheets API未初期化") 
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details_str = json.dumps(details, ensure_ascii=False) if details else ""
            
            values = [[
                tracker_id,
                timestamp,
                log_type,
                message,
                details_str
            ]]
            
            body = {'values': values}
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheets['processing']}!A:E",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"処理ログ更新完了: {tracker_id} - {log_type}")
            return True
            
        except HttpError as e:
            logger.error(f"処理ログ更新エラー: {e}")
            return False
    
    def update_improvement_suggestions(self, tracker_id: str, 
                                     improvements: List[str]) -> bool:
        """
        改善提案シート更新
        
        Args:
            tracker_id: トラッカーID
            improvements: 改善提案リスト
        """
        if not self.service or not improvements:
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 各改善提案を個別行に追加
            values = []
            for improvement in improvements:
                values.append([
                    tracker_id,
                    timestamp,
                    improvement
                ])
            
            body = {'values': values}
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheets['improvement']}!A:C",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"改善提案更新完了: {tracker_id} - {len(improvements)}件")
            return True
            
        except HttpError as e:
            logger.error(f"改善提案更新エラー: {e}")
            return False


def convert_quality_report_to_metrics(quality_data: Dict[str, Any]) -> MetricsRecord:
    """統合品質レポートをMetricsRecordに変換"""
    if not PROGRESS_TRACKER_AVAILABLE:
        logger.error("progress_tracker未利用可能")
        return None
    
    # 指標データ抽出
    eval_metrics = {m['name']: m['value'] for m in quality_data.get('evaluation_metrics', [])}
    obj_metrics = {m['name']: m['value'] for m in quality_data.get('objective_metrics', [])}
    
    # MetricsRecordにマッピング
    return MetricsRecord(
        lca=eval_metrics.get('Largest-Character Accuracy'),
        ab_evaluation_rate=eval_metrics.get('A/B評価率'), 
        fps=eval_metrics.get('FPS'),
        c_plus_rate=eval_metrics.get('C以上評価率'),
        sci=obj_metrics.get('SCI (Semantic Completeness Index)'),
        pla=obj_metrics.get('PLA (Pixel-Level Accuracy)'),
        ple=obj_metrics.get('PLE (Progressive Learning Efficiency)'),
        # 未実装指標はNone
        avg_coverage_rate=None,
        avg_compactness=None,
        avg_fill_rate=None
    )


def update_progress_tracker_record(tracker_id: str, quality_data: Dict[str, Any], 
                                 priority: Optional[PriorityLevel] = None) -> bool:
    """progress_trackerレコード更新"""
    if not PROGRESS_TRACKER_AVAILABLE or not GOOGLE_SHEETS_AVAILABLE:
        logger.warning("必要なライブラリが不足")
        return False
    
    try:
        # 更新実行
        updater = GoogleSheetsUpdater()
        if not updater.service:
            logger.warning("Google Sheets API未認証")
            return False
        
        # MetricsRecord変換
        metrics = convert_quality_report_to_metrics(quality_data)
        
        # TaskRecord作成
        task_record = TaskRecord(
            tracker_id=tracker_id,
            priority=priority or PriorityLevel.MEDIUM,  # デフォルト優先度中
            status=TaskStatus.QUALITY_CHECK,  # 品質チェック完了
            description=f"Dataset: {quality_data.get('dataset_name', 'unknown')}",
            quality_evaluation=ComponentStatus.COMPLETED,
            dashboard_generation=ComponentStatus.COMPLETED,
            metrics=metrics
        )
        task_record.updated_date = datetime.now()
        
        # 既存レコード検索
        existing_row = updater.find_existing_record(tracker_id)
        
        sheet_name = "シート1"
        values = [task_record.to_sheets_row()]
        body = {'values': values}
        
        if existing_row:
            # 既存レコード更新（上書き）
            range_name = f"{sheet_name}!A{existing_row}:W{existing_row}"
            result = updater.service.spreadsheets().values().update(
                spreadsheetId=updater.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            logger.info(f"progress_tracker既存レコード更新: {tracker_id} (行{existing_row})")
        else:
            # 新規レコード追加
            range_name = f"{sheet_name}!A:W"
            result = updater.service.spreadsheets().values().append(
                spreadsheetId=updater.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            logger.info(f"progress_tracker新規レコード追加: {tracker_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"progress_tracker更新エラー: {e}")
        return False


def update_from_quality_report(report_path: str, tracker_id: str = "PH2-002"):
    """統合品質レポートからSpreadsheet更新（progress_tracker統合版）"""
    if not Path(report_path).exists():
        logger.error(f"レポートファイルが見つかりません: {report_path}")
        return
    
    # レポート読み込み
    with open(report_path, 'r', encoding='utf-8') as f:
        quality_data = json.load(f)
    
    # progress_tracker形式で更新
    success = update_progress_tracker_record(tracker_id, quality_data)
    
    if success:
        logger.info(f"Google Spreadsheet更新完了: {tracker_id}")
        logger.info(f"Spreadsheet URL: https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit")
        # 詳細: docs/google_sheets_reference.md を参照
    else:
        logger.warning(f"Google Spreadsheet更新に失敗: {tracker_id}")


def display_sheet_data(data: List[List[str]], title: str = "Google Sheetsデータ", limit: int = None) -> None:
    """シートデータをテーブル形式で表示"""
    if not data:
        print("📝 表示するデータがありません")
        return
    
    display_data = data[:limit] if limit else data
    
    print(f"\n📊 {title}")
    print("=" * 120)
    
    # ヘッダー表示（簡略版）
    if display_data:
        header_row = display_data[0]
        if len(header_row) >= 6:
            print(f"{'ID':<12} {'優先度':<8} {'ステータス':<12} {'登録日':<12} {'概要':<60}")
        print("=" * 120)
        
        # データ行表示
        for i, row in enumerate(display_data[1:] if len(display_data) > 1 else [], 1):
            if len(row) >= 6:
                tracker_id = row[0][:11] if len(row[0]) > 11 else row[0]
                priority = row[1][:7] if len(row[1]) > 7 else row[1]
                status = row[2][:11] if len(row[2]) > 11 else row[2]
                date = row[3][:11] if len(row[3]) > 11 else row[3]
                description = row[5][:57] + "..." if len(row[5]) > 60 else row[5]
                
                print(f"{tracker_id:<12} {priority:<8} {status:<12} {date:<12} {description:<60}")
        
        if limit and len(data) > limit:
            print(f"\n... 他 {len(data) - limit} 件があります")
    
    print("=" * 120)


def display_statistics(stats: Dict[str, Any]) -> None:
    """統計情報を表示"""
    print("\n📊 Google Sheets統計情報")
    print("=" * 60)
    
    if "error" in stats:
        print(f"❌ エラー: {stats['error']}")
        return
    
    print(f"総タスク数: {stats.get('total_tasks', 0)}")
    print(f"取得日時: {stats.get('timestamp', '不明')}")
    
    # 優先度別統計
    priority_dist = stats.get('priority_distribution', {})
    if priority_dist:
        print("\n🎯 優先度別統計:")
        total = stats.get('total_tasks', 1)
        for priority in ["優先度最高", "優先度高", "優先度中", "優先度低"]:
            count = priority_dist.get(priority, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {priority:<10}: {count:>3}件 ({percentage:>5.1f}%)")
    
    # ステータス別統計
    status_dist = stats.get('status_distribution', {})
    if status_dist:
        print("\n📈 ステータス別統計:")
        for status, count in sorted(status_dist.items()):
            percentage = (count / stats.get('total_tasks', 1)) * 100
            print(f"  {status:<15}: {count:>3}件 ({percentage:>5.1f}%)")
    
    # Phase別統計
    phase_dist = stats.get('phase_distribution', {})
    if phase_dist:
        print("\n🔄 Phase別統計:")
        for phase, count in sorted(phase_dist.items()):
            percentage = (count / stats.get('total_tasks', 1)) * 100
            print(f"  {phase:<10}: {count:>3}件 ({percentage:>5.1f}%)")
    
    print("=" * 60)


def main():
    """コマンドライン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Google Sheets更新・読み取りツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # データ更新
  python google_sheets_updater.py --report quality_report.json --tracker-id PH2-002
  
  # データ読み取り
  python google_sheets_updater.py --read-all --limit 10          # 全データ表示（10件まで）
  python google_sheets_updater.py --priority 優先度最高           # 優先度フィルタ
  python google_sheets_updater.py --status 実装完了              # ステータスフィルタ
  python google_sheets_updater.py --get-task PH2-002             # 特定タスク取得
  python google_sheets_updater.py --stats                        # 統計情報
  python google_sheets_updater.py --export-json tasks.json       # JSON出力
        """
    )
    
    # 更新系オプション
    parser.add_argument("--report", "-r", help="統合品質レポートJSONパス")
    parser.add_argument("--tracker-id", "-t", default="PH2-002", help="トラッカーID")
    
    # 読み取り系オプション
    parser.add_argument("--read-all", action="store_true", help="全データ読み取り")
    parser.add_argument("--priority", help="優先度フィルタ（優先度最高/高/中/低）")
    parser.add_argument("--status", help="ステータスフィルタ")
    parser.add_argument("--get-task", help="特定トラッカーIDのタスクを取得")
    parser.add_argument("--stats", action="store_true", help="統計情報表示")
    
    # 出力オプション
    parser.add_argument("--limit", type=int, help="表示件数制限")
    parser.add_argument("--export-json", help="JSON出力ファイルパス")
    
    args = parser.parse_args()
    
    # 更新処理
    if args.report:
        update_from_quality_report(args.report, args.tracker_id)
        return
    
    # 読み取り処理
    if any([args.read_all, args.priority, args.status, args.get_task, args.stats]):
        try:
            # GoogleSheetsUpdater初期化
            updater = GoogleSheetsUpdater()
            if not updater.service:
                print("❌ Google Sheets API接続失敗")
                return
            
            print("✅ Google Sheets API接続成功")
            print(f"📊 スプレッドシートURL: https://docs.google.com/spreadsheets/d/{updater.spreadsheet_id}/edit")
            # 詳細: docs/google_sheets_reference.md を参照
            
            # 統計情報表示
            if args.stats:
                stats = updater.get_statistics()
                display_statistics(stats)
                
                # JSON出力
                if args.export_json:
                    with open(args.export_json, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, ensure_ascii=False, indent=2)
                    print(f"✅ 統計情報をJSON出力: {args.export_json}")
                return
            
            # 特定タスク取得
            if args.get_task:
                task_data = updater.get_task_by_id(args.get_task)
                if task_data:
                    print(f"\n📋 タスク詳細: {args.get_task}")
                    print("=" * 80)
                    
                    headers = ["トラッカーID", "優先度", "ステータス", "登録日付", "更新日付", "概要"]
                    for i, header in enumerate(headers):
                        value = task_data[i] if i < len(task_data) else "未設定"
                        print(f"{header:<15}: {value}")
                    
                    print("=" * 80)
                    
                    # JSON出力
                    if args.export_json:
                        task_dict = {headers[i]: (task_data[i] if i < len(task_data) else "") for i in range(len(headers))}
                        with open(args.export_json, 'w', encoding='utf-8') as f:
                            json.dump(task_dict, f, ensure_ascii=False, indent=2)
                        print(f"✅ タスクデータをJSON出力: {args.export_json}")
                else:
                    print(f"⚠️ タスクが見つかりません: {args.get_task}")
                return
            
            # データフィルタリング・表示
            data = []
            title = "Google Sheetsデータ"
            
            if args.priority:
                # 優先度フィルタ（B列：インデックス1）
                data = updater.get_filtered_data(1, args.priority)
                title = f"優先度「{args.priority}」のタスク"
            elif args.status:
                # ステータスフィルタ（C列：インデックス2）
                data = updater.get_filtered_data(2, args.status)
                title = f"ステータス「{args.status}」のタスク"
            elif args.read_all:
                # 全データ取得
                data = updater.get_all_sheet_data()
                title = "全タスクデータ"
            
            # データ表示
            if data:
                display_sheet_data(data, title, args.limit)
                
                # JSON出力
                if args.export_json:
                    # ヘッダー付きでJSON化
                    headers = data[0] if data else []
                    json_data = []
                    for row in data[1:]:  # ヘッダー行をスキップ
                        row_dict = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
                        json_data.append(row_dict)
                    
                    output_data = {
                        "metadata": {
                            "source": "Google Sheets API",
                            "spreadsheet_id": updater.spreadsheet_id,
                            "exported_at": datetime.now().isoformat(),
                            "total_records": len(json_data)
                        },
                        "data": json_data
                    }
                    
                    with open(args.export_json, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    print(f"✅ データをJSON出力: {args.export_json} ({len(json_data)}件)")
            else:
                print("📝 該当データがありません")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
    else:
        # ヘルプ表示
        parser.print_help()


if __name__ == "__main__":
    main()