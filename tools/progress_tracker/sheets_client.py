#!/usr/bin/env python3
"""
Google Sheets APIクライアント
進捗管理用スプレッドシートの読み書きを行う
"""

import os
import json
import time
import logging
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .data_models import (
    TaskRecord, ProgressTrackerConfig, SheetsAPIError, 
    TaskStatus, ComponentStatus, ProgressColumns
)

logger = logging.getLogger(__name__)


class GoogleSheetsClient:
    """Google Sheets APIクライアント"""
    
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    RETRY_COUNT = 3
    RETRY_DELAY = 1.0
    
    def __init__(self, config: ProgressTrackerConfig):
        """初期化"""
        self.config = config
        self.service = None
        self._actual_sheet_name = None  # 実際のシート名をキャッシュ
        self._authenticate()
        self._detect_actual_sheet_name()
    
    def _authenticate(self) -> None:
        """Google Sheets API認証"""
        try:
            auth_path = Path(self.config.auth_file_path)
            if not auth_path.exists():
                raise SheetsAPIError(f"認証ファイルが見つかりません: {auth_path}")
            
            credentials = Credentials.from_service_account_file(
                str(auth_path), scopes=self.SCOPES
            )
            
            self.service = build('sheets', 'v4', credentials=credentials)
            logger.info("Google Sheets API認証成功")
            
        except Exception as e:
            raise SheetsAPIError(f"Google Sheets API認証失敗: {e}")
    
    def _detect_actual_sheet_name(self) -> None:
        """実際のシート名を自動検出"""
        try:
            # スプレッドシートのメタデータを取得
            metadata = self.service.spreadsheets().get(
                spreadsheetId=self.config.spreadsheet_id
            ).execute()
            
            sheets = metadata.get('sheets', [])
            if not sheets:
                raise SheetsAPIError("シートが見つかりません")
            
            # 設定されたシート名と一致するものを探す
            for sheet in sheets:
                sheet_title = sheet['properties']['title']
                if sheet_title == self.config.sheet_name:
                    self._actual_sheet_name = sheet_title
                    logger.info(f"設定シート名が一致: {sheet_title}")
                    return
            
            # 一致しない場合は最初のシートを使用
            first_sheet_name = sheets[0]['properties']['title']
            self._actual_sheet_name = first_sheet_name
            
            logger.warning(
                f"設定シート名 '{self.config.sheet_name}' が見つかりません。"
                f"最初のシート '{first_sheet_name}' を使用します"
            )
            
            # 利用可能なシート一覧をログ出力
            sheet_names = [s['properties']['title'] for s in sheets]
            logger.info(f"利用可能なシート: {sheet_names}")
            
        except Exception as e:
            logger.error(f"シート名検出に失敗: {e}")
            # フォールバック: 設定値をそのまま使用
            self._actual_sheet_name = self.config.sheet_name
    
    def _get_safe_range(self, range_name: str) -> str:
        """安全な範囲指定文字列を生成"""
        sheet_name = self._actual_sheet_name or self.config.sheet_name
        
        # シート名に特殊文字が含まれる場合はクォートで囲む
        if any(char in sheet_name for char in [' ', '!', "'", '"']):
            # シングルクォートで囲み、内部のシングルクォートはエスケープ
            safe_sheet_name = f"'{sheet_name.replace(chr(39), chr(39) + chr(39))}'"
        else:
            safe_sheet_name = sheet_name
        
        return f"{safe_sheet_name}!{range_name}"
    
    def _execute_with_retry(self, request) -> Any:
        """リトライ機能付きAPI実行"""
        last_error = None
        
        for attempt in range(self.RETRY_COUNT):
            try:
                return request.execute()
            except HttpError as e:
                last_error = e
                
                # HTTPエラーの詳細ログ
                error_code = e.resp.status
                error_message = str(e)
                
                logger.warning(
                    f"HTTP API実行失敗 (試行 {attempt + 1}/{self.RETRY_COUNT}): "
                    f"Code {error_code}, Message: {error_message}"
                )
                
                # 400エラー（Bad Request）の場合は特別処理
                if error_code == 400:
                    if "Unable to parse range" in error_message:
                        logger.error(
                            f"範囲指定エラー: {error_message}\n"
                            f"実際のシート名: '{self._actual_sheet_name}'\n"
                            f"設定シート名: '{self.config.sheet_name}'\n"
                            f"SpreadSheet URL: https://docs.google.com/spreadsheets/d/{self.config.spreadsheet_id}/edit"
                        )
                        
                        # 手動修正の案内
                        if attempt == 0:  # 最初の試行でのみ表示
                            self._show_manual_update_guide()
                
                if attempt == self.RETRY_COUNT - 1:
                    raise SheetsAPIError(f"API実行失敗 (最終試行): {e}")
                
                time.sleep(self.RETRY_DELAY * (2 ** attempt))  # 指数バックオフ
                
            except Exception as e:
                last_error = e
                
                if attempt == self.RETRY_COUNT - 1:
                    raise SheetsAPIError(f"API実行失敗 (最終試行): {e}")
                
                logger.warning(f"API実行失敗 (試行 {attempt + 1}/{self.RETRY_COUNT}): {e}")
                time.sleep(self.RETRY_DELAY * (2 ** attempt))
        
        # ここには到達しないはずだが、安全のため
        raise SheetsAPIError(f"予期しないエラー: {last_error}")
    
    def _show_manual_update_guide(self) -> None:
        """手動更新ガイドの表示"""
        print("\n" + "="*60)
        print("🛠️  SPREADSHEET手動更新ガイド")
        print("="*60)
        print(f"📋 SpreadSheet URL:")
        print(f"   https://docs.google.com/spreadsheets/d/{self.config.spreadsheet_id}/edit")
        print(f"\n📝 手動でPHS-006のステータスを以下に変更してください:")
        print(f"   着手中 → 実装完了")
        print(f"\n💡 問題の原因:")
        print(f"   設定シート名: '{self.config.sheet_name}'")
        print(f"   実際のシート名: '{self._actual_sheet_name}'")
        print(f"\n🔧 今後の自動化のため、シート名を統一することを推奨します")
        print("="*60)
    
    def ensure_proper_sheet_setup(self) -> bool:
        """適切なシート設定の確保"""
        try:
            # メタデータ取得
            metadata = self.service.spreadsheets().get(
                spreadsheetId=self.config.spreadsheet_id
            ).execute()
            
            sheets = metadata.get('sheets', [])
            if not sheets:
                logger.error("シートが存在しません")
                return False
            
            # 最初のシートの情報
            first_sheet = sheets[0]
            sheet_id = first_sheet['properties']['sheetId']
            current_name = first_sheet['properties']['title']
            
            # シート名が不適切な場合は修正提案
            if current_name != self.config.sheet_name:
                logger.warning(
                    f"シート名が設定と異なります: '{current_name}' != '{self.config.sheet_name}'"
                )
                
                # 自動修正を試行
                if self._try_rename_sheet(sheet_id, current_name):
                    self._actual_sheet_name = self.config.sheet_name
                    logger.info(f"シート名を '{self.config.sheet_name}' に修正しました")
                    return True
                else:
                    logger.warning("シート名の自動修正に失敗しました")
            
            return True
            
        except Exception as e:
            logger.error(f"シート設定確認エラー: {e}")
            return False
    
    def _try_rename_sheet(self, sheet_id: int, current_name: str) -> bool:
        """シート名の修正を試行"""
        try:
            # シート名変更のリクエスト
            request_body = {
                "requests": [{
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_id,
                            "title": self.config.sheet_name
                        },
                        "fields": "title"
                    }
                }]
            }
            
            request = self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.config.spreadsheet_id,
                body=request_body
            )
            
            self._execute_with_retry(request)
            logger.info(f"シート名変更成功: '{current_name}' → '{self.config.sheet_name}'")
            return True
            
        except Exception as e:
            logger.warning(f"シート名変更失敗: {e}")
            return False
    
    def get_sheet_values(self, range_name: str) -> List[List[str]]:
        """シートデータ取得"""
        try:
            safe_range = self._get_safe_range(range_name)
            logger.debug(f"安全な範囲指定: {safe_range}")
            
            request = self.service.spreadsheets().values().get(
                spreadsheetId=self.config.spreadsheet_id,
                range=safe_range
            )
            
            result = self._execute_with_retry(request)
            return result.get('values', [])
            
        except Exception as e:
            raise SheetsAPIError(f"シートデータ取得失敗: {e}")
    
    def update_sheet_values(self, range_name: str, values: List[List[str]]) -> None:
        """シートデータ更新"""
        try:
            safe_range = self._get_safe_range(range_name)
            logger.debug(f"更新範囲: {safe_range}")
            
            body = {
                'values': values
            }
            
            request = self.service.spreadsheets().values().update(
                spreadsheetId=self.config.spreadsheet_id,
                range=safe_range,
                valueInputOption='USER_ENTERED',
                body=body
            )
            
            self._execute_with_retry(request)
            logger.info(f"シートデータ更新成功: {range_name}")
            
        except Exception as e:
            raise SheetsAPIError(f"シートデータ更新失敗: {e}")
    
    def append_sheet_values(self, values: List[List[str]]) -> None:
        """シートデータ追加"""
        try:
            safe_range = self._get_safe_range("A:W")
            
            body = {
                'values': values
            }
            
            request = self.service.spreadsheets().values().append(
                spreadsheetId=self.config.spreadsheet_id,
                range=safe_range,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            )
            
            self._execute_with_retry(request)
            logger.info(f"シートデータ追加成功: {len(values)}行")
            
        except Exception as e:
            raise SheetsAPIError(f"シートデータ追加失敗: {e}")
    
    def setup_sheet_validation(self) -> None:
        """シートのデータバリデーション設定"""
        try:
            # ステータス列（B列）のドロップダウン設定
            status_validation = {
                "setDataValidation": {
                    "range": {
                        "sheetId": 0,  # 最初のシート
                        "startRowIndex": 1,  # ヘッダー行をスキップ
                        "endRowIndex": 1000,  # 1000行まで
                        "startColumnIndex": 1,  # B列
                        "endColumnIndex": 2
                    },
                    "rule": {
                        "condition": {
                            "type": "ONE_OF_LIST",
                            "values": [{"userEnteredValue": status.value} for status in TaskStatus]
                        },
                        "showCustomUi": True,
                        "strict": True
                    }
                }
            }
            
            # コンポーネントステータス列（F-K列）のドロップダウン設定
            component_validation = {
                "setDataValidation": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 1,
                        "endRowIndex": 1000,
                        "startColumnIndex": 5,  # F列
                        "endColumnIndex": 11    # K列
                    },
                    "rule": {
                        "condition": {
                            "type": "ONE_OF_LIST",
                            "values": [{"userEnteredValue": status.value} for status in ComponentStatus]
                        },
                        "showCustomUi": True,
                        "strict": True
                    }
                }
            }
            
            # 条件付き書式（ステータス別色分け）
            conditional_formatting = {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [{
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": 1000,
                            "startColumnIndex": 1,
                            "endColumnIndex": 2
                        }],
                        "booleanRule": {
                            "condition": {
                                "type": "TEXT_EQ",
                                "values": [{"userEnteredValue": TaskStatus.RELEASE.value}]
                            },
                            "format": {
                                "backgroundColor": {"red": 0.8, "green": 1.0, "blue": 0.8}  # 薄緑
                            }
                        }
                    },
                    "index": 0
                }
            }
            
            # バッチ更新実行
            requests = [
                status_validation,
                component_validation,
                conditional_formatting
            ]
            
            body = {"requests": requests}
            
            request = self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.config.spreadsheet_id,
                body=body
            )
            
            self._execute_with_retry(request)
            logger.info("シートバリデーション設定完了")
            
        except Exception as e:
            raise SheetsAPIError(f"シートバリデーション設定失敗: {e}")
    
    def get_all_tasks(self) -> List[TaskRecord]:
        """全タスク取得"""
        try:
            values = self.get_sheet_values("A2:W1000")  # ヘッダー行をスキップ（23列）
            tasks = []
            
            for row in values:
                if row and row[0]:  # トラッカーIDがある行のみ
                    try:
                        task = TaskRecord.from_sheets_row(row)
                        tasks.append(task)
                    except Exception as e:
                        logger.warning(f"タスクレコード変換エラー ({row[0] if row else 'unknown'}): {e}")
                        continue
            
            logger.info(f"タスク取得完了: {len(tasks)}件")
            return tasks
            
        except Exception as e:
            raise SheetsAPIError(f"タスク取得失敗: {e}")
    
    def find_task_row(self, tracker_id: str) -> Optional[int]:
        """トラッカーIDから行番号を検索"""
        try:
            values = self.get_sheet_values("A:A")
            
            for i, row in enumerate(values):
                if row and row[0] == tracker_id:
                    return i + 1  # 1ベースのインデックス
            
            return None
            
        except Exception as e:
            raise SheetsAPIError(f"タスク行検索失敗: {e}")
    
    def update_task(self, task: TaskRecord) -> None:
        """タスク更新"""
        try:
            row_number = self.find_task_row(task.tracker_id)
            
            if row_number is None:
                # 新規タスクとして追加
                self.append_sheet_values([task.to_sheets_row()])
                logger.info(f"新規タスク追加: {task.tracker_id}")
            else:
                # 既存タスクを更新（23列）
                range_name = f"A{row_number}:W{row_number}"
                self.update_sheet_values(range_name, [task.to_sheets_row()])
                logger.info(f"タスク更新: {task.tracker_id} (行 {row_number})")
                
        except Exception as e:
            raise SheetsAPIError(f"タスク更新失敗: {e}")
    
    def get_task(self, tracker_id: str) -> Optional[TaskRecord]:
        """特定タスク取得"""
        try:
            row_number = self.find_task_row(tracker_id)
            if row_number is None:
                return None
            
            range_name = f"A{row_number}:W{row_number}"
            values = self.get_sheet_values(range_name)
            
            if values and values[0]:
                return TaskRecord.from_sheets_row(values[0])
            
            return None
            
        except Exception as e:
            raise SheetsAPIError(f"タスク取得失敗: {e}")
    
    def initialize_sheet(self) -> None:
        """シート初期化（ヘッダー設定）"""
        try:
            # ヘッダー行設定（23列：A-W）
            headers = [
                "トラッカーID",        # A
                "ステータス",          # B
                "登録日付",            # C
                "更新日付",            # D
                "概要",                # E
                "動作確認",            # F
                "テストUNIT",          # G
                "品質評価",            # H
                "統合実行スクリプト",  # I
                "ダッシュボード生成",  # J
                "抽出パイプライン",    # K
                "LCA",                 # L
                "A/B評価率",           # M
                "FPS",                 # N
                "C以上評価率",         # O
                "平均カバレッジ率",    # P
                "平均コンパクトネス",  # Q
                "平均フィル率",        # R
                "SCI",                 # S
                "PLA",                 # T
                "PLE"                  # U
            ]
            
            # 既存のヘッダーをチェック（23列）
            existing_values = self.get_sheet_values("A1:W1")
            
            if not existing_values or existing_values[0] != headers:
                self.update_sheet_values("A1:W1", [headers])
                logger.info("ヘッダー行設定完了（23列）")
            
            # データバリデーション設定
            self.setup_sheet_validation()
            
        except Exception as e:
            raise SheetsAPIError(f"シート初期化失敗: {e}")