#!/usr/bin/env python3
"""
PlanCommandHandler - Google Sheets起票専用ハンドラー
KIRO-006 Phase 2: ワークフロー計画・起票システム

Google Sheetsへの新規トラッカー起票機能を提供します。
3つの引数（tracker_id、summary、details）を必須とし、
詳細に20,000文字制限を設けています。
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskStatus, PriorityLevel

logger = logging.getLogger(__name__)


class PlanCommandHandler:
    """Google Sheets起票専用コマンドハンドラー"""
    
    # 定数定義
    MAX_DETAILS_LENGTH = 20000  # 詳細の最大文字数
    
    def __init__(self):
        """初期化"""
        self.progress_manager = None
        self._initialize_progress_manager()
    
    def _initialize_progress_manager(self) -> None:
        """ProgressManagerを初期化"""
        try:
            from tools.progress_tracker.config import get_default_config
            config = get_default_config()
            self.progress_manager = ProgressManager(config)
            logger.info("ProgressManager初期化完了")
        except Exception as e:
            logger.error(f"ProgressManager初期化失敗: {e}")
            self.progress_manager = None
    
    def validate_inputs(self, tracker_id: str, summary: str, details: str) -> Tuple[bool, Optional[str]]:
        """
        入力検証を実行
        
        Args:
            tracker_id: トラッカーID
            summary: 概要
            details: 詳細
            
        Returns:
            Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
        """
        # 1. 必須項目チェック
        if not tracker_id or not tracker_id.strip():
            return False, "❌ トラッカーIDが指定されていません"
        
        if not summary or not summary.strip():
            return False, "❌ 概要が指定されていません"
        
        if not details or not details.strip():
            return False, "❌ 詳細が指定されていません"
        
        # 2. トラッカーID形式チェック
        tracker_id = tracker_id.strip()
        if not self._validate_tracker_id_format(tracker_id):
            return False, f"❌ トラッカーID形式が無効です: {tracker_id}\n   形式例: TRACKER-001, KIRO-006, QUAL-044"
        
        # 3. 文字数制限チェック
        details_length = len(details)
        if details_length > self.MAX_DETAILS_LENGTH:
            return False, f"❌ 詳細が文字数制限を超えています: {details_length:,}文字 (制限: {self.MAX_DETAILS_LENGTH:,}文字)"
        
        # 4. 概要の長さチェック（推奨）
        summary_length = len(summary)
        if summary_length > 200:
            logger.warning(f"概要が長すぎます: {summary_length}文字 (推奨: 200文字以内)")
        
        return True, None
    
    def _validate_tracker_id_format(self, tracker_id: str) -> bool:
        """
        トラッカーID形式を検証
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            bool: 形式が有効かどうか
        """
        import re
        
        # 基本パターン: PREFIX-NUMBER (例: TRACKER-001, KIRO-006)
        pattern = r'^[A-Z][A-Z0-9]*-[0-9]+$'
        
        return bool(re.match(pattern, tracker_id))
    
    def check_existing_tracker(self, tracker_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        既存トラッカーの存在確認
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            Tuple[bool, Optional[Dict]]: (存在するか, 既存タスク情報)
        """
        if not self.progress_manager:
            return False, None
        
        try:
            existing_task = self.progress_manager.get_task(tracker_id)
            if existing_task:
                return True, {
                    'tracker_id': existing_task.tracker_id,
                    'status': existing_task.status.value,
                    'created_date': existing_task.created_date,
                    'description': existing_task.description
                }
            return False, None
        except Exception as e:
            logger.error(f"既存トラッカー確認エラー: {e}")
            return False, None
    
    def create_google_sheets_tracker(self, tracker_id: str, summary: str, details: str, 
                                   priority: PriorityLevel = PriorityLevel.MEDIUM) -> Tuple[bool, str]:
        """
        Google Sheetsに新規トラッカーを作成（リトライ機能付き）
        
        Args:
            tracker_id: トラッカーID
            summary: 概要
            details: 詳細
            priority: 優先度
            
        Returns:
            Tuple[bool, str]: (成功, メッセージ)
        """
        if not self.progress_manager:
            return False, "❌ Google Sheets連携が利用できません。設定を確認してください。"
        
        # リトライ設定
        max_retries = 3
        retry_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                # 詳細情報を含む説明を作成
                description = f"{summary}\n\n【詳細】\n{details}"
                
                # Google Sheetsにタスク作成
                task = self.progress_manager.create_task(
                    tracker_id=tracker_id,
                    description=description
                )
                
                # 更新日付を登録日付と同じ値に設定
                task.updated_date = task.created_date
                self.progress_manager.client.update_task(task)
                
                # 優先度設定（デフォルトは中）
                if priority != PriorityLevel.MEDIUM:
                    try:
                        # 優先度設定メソッドが存在するか確認
                        if hasattr(self.progress_manager, 'update_task_priority'):
                            self.progress_manager.update_task_priority(tracker_id, priority)
                        else:
                            logger.warning("優先度設定メソッドが利用できません")
                    except Exception as e:
                        logger.warning(f"優先度設定に失敗: {e}")
                
                # 作成確認
                created_task = self.progress_manager.get_task(tracker_id)
                if not created_task:
                    raise Exception("タスク作成後の確認に失敗しました")
                
                success_message = f"""✅ Google Sheetsにトラッカーを起票しました

📋 トラッカー情報:
   ID: {task.tracker_id}
   概要: {summary}
   詳細文字数: {len(details):,}文字
   ステータス: {task.status.value}
   優先度: {priority.value}
   作成日時: {task.created_date}

📊 Google Sheets URL:
   {self._get_sheets_url()}

🔄 次のステップ:
   ワークフロー状態管理を開始するには:
   python tools/workflow/workflow_cli.py create {tracker_id}

📋 確認コマンド:
   python tools/workflow/workflow_cli.py sheets {tracker_id}
"""
                
                logger.info(f"Google Sheetsトラッカー作成成功: {tracker_id} (試行 {attempt + 1}/{max_retries})")
                return True, success_message
                
            except Exception as e:
                logger.warning(f"Google Sheets起票試行 {attempt + 1}/{max_retries} 失敗: {e}")
                
                if attempt < max_retries - 1:
                    # 最後の試行でない場合はリトライ
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                    continue
                else:
                    # 最後の試行も失敗した場合
                    error_message = f"""❌ Google Sheets起票に失敗しました（{max_retries}回試行）

エラー詳細: {str(e)}

🔧 トラブルシューティング:
1. Google Sheets設定確認:
   python tools/progress_tracker/cli.py check-config

2. 接続テスト:
   python tools/progress_tracker/test_connection.py

3. 認証情報確認:
   - credentials.json ファイルの存在確認
   - Google Sheets API有効化確認
   - 共有設定確認

4. 手動作成:
   python tools/progress_tracker/cli.py create {tracker_id} "{summary}"

5. ネットワーク確認:
   - インターネット接続確認
   - プロキシ設定確認
   - ファイアウォール設定確認
"""
                    logger.error(f"Google Sheets起票最終失敗: {tracker_id} - {e}")
                    return False, error_message
        
        # ここには到達しないはずだが、安全のため
        return False, "❌ 予期しないエラーが発生しました"
    
    def _get_sheets_url(self) -> str:
        """Google Sheets URLを取得"""
        try:
            if hasattr(self.progress_manager, 'config') and hasattr(self.progress_manager.config, 'sheet_url'):
                return self.progress_manager.config.sheet_url
            elif hasattr(self.progress_manager, 'config') and hasattr(self.progress_manager.config, 'spreadsheet_id'):
                return f"https://docs.google.com/spreadsheets/d/{self.progress_manager.config.spreadsheet_id}/edit"
            else:
                return "設定ファイルで確認してください"
        except Exception:
            return "設定ファイルで確認してください"
    
    def execute_plan_command(self, tracker_id: str, summary: str, details: str, 
                           priority: str = "medium") -> Tuple[bool, str]:
        """
        planコマンドを実行
        
        Args:
            tracker_id: トラッカーID
            summary: 概要
            details: 詳細
            priority: 優先度 ("highest", "high", "medium", "low")
            
        Returns:
            Tuple[bool, str]: (成功, メッセージ)
        """
        logger.info(f"planコマンド実行開始: {tracker_id}")
        
        # 1. 入力検証
        is_valid, error_message = self.validate_inputs(tracker_id, summary, details)
        if not is_valid:
            return False, error_message
        
        # 2. 既存トラッカー確認
        exists, existing_info = self.check_existing_tracker(tracker_id)
        if exists:
            warning_message = f"""⚠️  トラッカーが既に存在します: {tracker_id}

📋 既存情報:
   ステータス: {existing_info['status']}
   作成日時: {existing_info['created_date']}
   説明: {existing_info['description'][:100]}{'...' if len(existing_info['description']) > 100 else ''}

🔄 選択肢:
1. 既存トラッカーを更新:
   python tools/progress_tracker/cli.py update {tracker_id} "{summary}"

2. 別のトラッカーIDを使用:
   python tools/workflow/workflow_cli.py plan {tracker_id}-v2 "{summary}" "{details}"

3. 既存トラッカーでワークフロー開始:
   python tools/workflow/workflow_cli.py create {tracker_id}
"""
            return False, warning_message
        
        # 3. 優先度変換
        priority_map = {
            "highest": PriorityLevel.HIGHEST,
            "high": PriorityLevel.HIGH,
            "medium": PriorityLevel.MEDIUM,
            "low": PriorityLevel.LOW
        }
        priority_level = priority_map.get(priority.lower(), PriorityLevel.MEDIUM)
        
        # 4. Google Sheets起票実行
        success, message = self.create_google_sheets_tracker(
            tracker_id, summary, details, priority_level
        )
        
        if success:
            logger.info(f"planコマンド実行成功: {tracker_id}")
        else:
            logger.error(f"planコマンド実行失敗: {tracker_id}")
        
        return success, message
    
    def get_usage_help(self) -> str:
        """使用方法ヘルプを取得"""
        return f"""
🚀 **planコマンド使用方法**

## 基本構文
```bash
python tools/workflow/workflow_cli.py plan <TRACKER_ID> <概要> <詳細> [優先度]
```

## 引数説明
- **TRACKER_ID**: トラッカーID（必須）
  - 形式: PREFIX-NUMBER（例: TRACKER-001, KIRO-006）
  - 英数字とハイフンのみ使用可能

- **概要**: プロジェクトの概要（必須）
  - 推奨: 200文字以内
  - Google Sheetsの概要欄に表示

- **詳細**: 詳細な説明（必須）
  - 制限: {self.MAX_DETAILS_LENGTH:,}文字以内
  - 要件、仕様、実装方針などを記載

- **優先度**: タスクの優先度（オプション）
  - highest: 優先度最高
  - high: 優先度高
  - medium: 優先度中（デフォルト）
  - low: 優先度低

## 使用例
```bash
# 基本的な使用例
python tools/workflow/workflow_cli.py plan KIRO-007 "新機能実装" "ユーザー認証システムの実装..."

# 優先度指定
python tools/workflow/workflow_cli.py plan URGENT-001 "緊急修正" "セキュリティ脆弱性の修正..." highest

# 長い詳細を含む例
python tools/workflow/workflow_cli.py plan FEATURE-123 "UI改善" "$(cat requirements.txt)"
```

## ワークフロー
1. **起票**: `plan` コマンドでGoogle Sheetsに登録
2. **開始**: `create` コマンドでワークフロー状態管理開始
3. **実行**: `step` コマンドで段階的実行
4. **完了**: 全ステップ完了まで継続

## 注意事項
- トラッカーIDは一意である必要があります
- 既存のトラッカーIDは使用できません
- 詳細は{self.MAX_DETAILS_LENGTH:,}文字以内で記載してください
- Google Sheets設定が必要です
"""


def main():
    """テスト用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PlanCommandHandler テスト")
    parser.add_argument('tracker_id', help='トラッカーID')
    parser.add_argument('summary', help='概要')
    parser.add_argument('details', help='詳細')
    parser.add_argument('--priority', default='medium', help='優先度')
    
    args = parser.parse_args()
    
    handler = PlanCommandHandler()
    success, message = handler.execute_plan_command(
        args.tracker_id, args.summary, args.details, args.priority
    )
    
    print(message)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())