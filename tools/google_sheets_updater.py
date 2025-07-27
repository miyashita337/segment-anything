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
sys.path.append(str(Path(__file__).parent))
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
            range_name = f"{sheet_name}!A{existing_row}:V{existing_row}"
            result = updater.service.spreadsheets().values().update(
                spreadsheetId=updater.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            logger.info(f"progress_tracker既存レコード更新: {tracker_id} (行{existing_row})")
        else:
            # 新規レコード追加
            range_name = f"{sheet_name}!A:V"
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
    else:
        logger.warning(f"Google Spreadsheet更新に失敗: {tracker_id}")


def main():
    """コマンドライン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Sheets更新")
    parser.add_argument("--report", "-r", help="統合品質レポートJSONパス")
    parser.add_argument("--tracker-id", "-t", default="PH2-002", help="トラッカーID")
    
    args = parser.parse_args()
    
    if args.report:
        update_from_quality_report(args.report, args.tracker_id)
    else:
        print("使用例:")
        print("python google_sheets_updater.py -r /path/to/quality_report.json -t PH2-002")


if __name__ == "__main__":
    main()