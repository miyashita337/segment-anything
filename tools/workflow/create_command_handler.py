#!/usr/bin/env python3
"""
CreateCommandHandler - SQLite専用ワークフロー状態管理ハンドラー
KIRO-006 Phase 2: ワークフロー計画・起票システム

SQLiteベースのワークフロー状態管理専用機能を提供します。
Google Sheets機能は削除され、ローカル状態管理のみに特化しています。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = logging.getLogger(__name__)


class CreateCommandHandler:
    """SQLite専用ワークフロー状態管理コマンドハンドラー"""

    def __init__(self):
        """初期化"""
        self.workflow_controller = None
        self._initialize_workflow_controller()

    def _initialize_workflow_controller(self) -> None:
        """ワークフローコントローラーを初期化"""
        try:
            from tools.interface.workflow_controller import get_workflow_controller

            self.workflow_controller = get_workflow_controller()
            logger.info("ワークフローコントローラー初期化完了")
        except Exception as e:
            logger.error(f"ワークフローコントローラー初期化失敗: {e}")
            self.workflow_controller = None

    def validate_tracker_id(self, tracker_id: str) -> Tuple[bool, Optional[str]]:
        """
        トラッカーID検証

        Args:
            tracker_id: トラッカーID

        Returns:
            Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
        """
        if not tracker_id or not tracker_id.strip():
            return False, "❌ トラッカーIDが指定されていません"

        tracker_id = tracker_id.strip()

        # 基本的な形式チェック
        import re

        pattern = r"^[A-Z][A-Z0-9]*-[0-9]+$"
        if not re.match(pattern, tracker_id):
            return False, f"❌ トラッカーID形式が無効です: {tracker_id}\n   形式例: TRACKER-001, KIRO-006, QUAL-044"

        return True, None

    def check_existing_workflow(self, tracker_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        既存ワークフロー状態の確認

        Args:
            tracker_id: トラッカーID

        Returns:
            Tuple[bool, Optional[Dict]]: (存在するか, 既存状態情報)
        """
        if not self.workflow_controller:
            return False, None

        try:
            status = self.workflow_controller.get_workflow_status(tracker_id)

            # "not_found"または"error"の場合は存在しないと判断
            if status.get("status") == "not_found" or "error" in status:
                return False, None

            # 有効な状態情報がある場合は存在すると判断
            return True, {
                "tracker_id": tracker_id,
                "current_phase": status.get("current_phase", "不明"),
                "current_step": status.get("current_step", "不明"),
                "can_proceed": status.get("can_proceed", False),
                "completed_steps": len(status.get("completed_steps", [])),
                "pending_approvals": len(status.get("pending_approvals", [])),
            }
        except Exception as e:
            logger.error(f"既存ワークフロー確認エラー: {e}")
            return False, None

    def create_workflow_state(self, tracker_id: str) -> Tuple[bool, str]:
        """
        SQLiteワークフロー状態を作成

        Args:
            tracker_id: トラッカーID

        Returns:
            Tuple[bool, str]: (成功, メッセージ)
        """
        if not self.workflow_controller:
            return (
                False,
                """❌ ワークフローコントローラーが利用できません

🔧 トラブルシューティング:
1. ワークフローコントローラーの初期化確認
2. SQLiteデータベースファイルの権限確認
3. 依存関係の確認

📋 手動確認コマンド:
   python tools/interface/workflow_controller.py --test
""",
            )

        try:
            # ワークフロー状態作成
            success = self.workflow_controller.create_tracker_workflow(tracker_id)

            if not success:
                return (
                    False,
                    f"""❌ ワークフロー状態作成に失敗しました: {tracker_id}

🔧 考えられる原因:
- SQLiteデータベースの書き込み権限不足
- 既存の状態データとの競合
- ディスク容量不足

📋 確認コマンド:
   python tools/workflow/workflow_cli.py status {tracker_id}
""",
                )

            # 作成後の状態確認
            status = self.workflow_controller.get_workflow_status(tracker_id)
            current_step = status.get("current_step", "不明")

            success_message = f"""✅ ワークフロー状態管理を開始しました

📋 ワークフロー情報:
   トラッカーID: {tracker_id}
   現在のフェーズ: {status.get('current_phase', '不明')}
   現在のステップ: {current_step}
   進行可能: {'✅' if status.get('can_proceed', False) else '❌'}
   作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔄 次のステップ:
   現在のステップ指示を確認:
   python tools/workflow/workflow_cli.py instructions {tracker_id}

   ステップを実行:
   python tools/workflow/workflow_cli.py step {tracker_id}

   状態確認:
   python tools/workflow/workflow_cli.py status {tracker_id}

📝 重要な注意事項:
   - このコマンドはSQLiteベースのローカル状態管理のみを作成します
   - Google Sheetsへの起票は別途planコマンドを使用してください:
     python tools/workflow/workflow_cli.py plan {tracker_id} "概要" "詳細"
   
   - 統合ワークフローの場合は以下の順序で実行:
     1. planコマンド（Google Sheets起票）
     2. createコマンド（ワークフロー状態管理開始）
"""

            logger.info(f"ワークフロー状態作成成功: {tracker_id}")
            return True, success_message

        except Exception as e:
            error_message = f"""❌ ワークフロー状態作成でエラーが発生しました

エラー詳細: {str(e)}

🔧 トラブルシューティング:
1. SQLiteデータベース確認:
   - ファイル権限の確認
   - ディスク容量の確認
   - データベースファイルの整合性確認

2. 依存関係確認:
   - 必要なPythonモジュールのインストール確認
   - パス設定の確認

3. 手動デバッグ:
   python -c "from tools.interface.workflow_controller import get_workflow_controller; print(get_workflow_controller())"

4. ログ確認:
   - アプリケーションログの確認
   - システムログの確認
"""
            logger.error(f"ワークフロー状態作成エラー: {tracker_id} - {e}")
            return False, error_message

    def execute_create_command(self, tracker_id: str) -> Tuple[bool, str]:
        """
        createコマンドを実行

        Args:
            tracker_id: トラッカーID

        Returns:
            Tuple[bool, str]: (成功, メッセージ)
        """
        logger.info(f"createコマンド実行開始: {tracker_id}")

        # 1. トラッカーID検証
        is_valid, error_message = self.validate_tracker_id(tracker_id)
        if not is_valid:
            return False, error_message

        # 2. 既存ワークフロー確認
        exists, existing_info = self.check_existing_workflow(tracker_id)
        if exists:
            warning_message = f"""⚠️  ワークフロー状態が既に存在します: {tracker_id}

📋 既存情報:
   現在のフェーズ: {existing_info['current_phase']}
   現在のステップ: {existing_info['current_step']}
   進行可能: {'✅' if existing_info['can_proceed'] else '❌'}
   完了ステップ数: {existing_info['completed_steps']}
   承認待ち数: {existing_info['pending_approvals']}

🔄 選択肢:
1. 既存ワークフローの状態確認:
   python tools/workflow/workflow_cli.py status {tracker_id}

2. 既存ワークフローの続行:
   python tools/workflow/workflow_cli.py instructions {tracker_id}
   python tools/workflow/workflow_cli.py step {tracker_id}

3. 別のトラッカーIDを使用:
   python tools/workflow/workflow_cli.py create {tracker_id}-v2

📝 注意: 既存のワークフロー状態は上書きされません
"""
            return False, warning_message

        # 3. ワークフロー状態作成実行
        success, message = self.create_workflow_state(tracker_id)

        if success:
            logger.info(f"createコマンド実行成功: {tracker_id}")
        else:
            logger.error(f"createコマンド実行失敗: {tracker_id}")

        return success, message

    def get_usage_help(self) -> str:
        """使用方法ヘルプを取得"""
        return """
🚀 **createコマンド使用方法**

## 基本構文
```bash
python tools/workflow/workflow_cli.py create <TRACKER_ID>
```

## 引数説明
- **TRACKER_ID**: トラッカーID（必須）
  - 形式: PREFIX-NUMBER（例: TRACKER-001, KIRO-006）
  - 英数字とハイフンのみ使用可能

## 機能説明
createコマンドは**SQLiteベースのローカルワークフロー状態管理**を開始します。

### 🔄 統合ワークフローの推奨手順
1. **Google Sheets起票**: 
   ```bash
   python tools/workflow/workflow_cli.py plan TRACKER-001 "概要" "詳細"
   ```

2. **ワークフロー状態管理開始**:
   ```bash
   python tools/workflow/workflow_cli.py create TRACKER-001
   ```

3. **ワークフロー実行**:
   ```bash
   python tools/workflow/workflow_cli.py step TRACKER-001
   ```

## 使用例
```bash
# 基本的な使用例
python tools/workflow/workflow_cli.py create KIRO-007

# 状態確認
python tools/workflow/workflow_cli.py status KIRO-007

# 次ステップ指示
python tools/workflow/workflow_cli.py instructions KIRO-007

# ステップ実行
python tools/workflow/workflow_cli.py step KIRO-007
```

## 重要な変更点
- **Google Sheets機能削除**: createコマンドはGoogle Sheetsに起票しません
- **SQLite専用**: ローカルワークフロー状態管理のみに特化
- **planコマンド分離**: Google Sheets起票は別途planコマンドを使用

## 注意事項
- トラッカーIDは一意である必要があります
- 既存のワークフロー状態は上書きされません
- Google Sheetsとの連携が必要な場合は事前にplanコマンドを実行してください
- SQLiteデータベースファイルの書き込み権限が必要です
"""


def main():
    """テスト用メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="CreateCommandHandler テスト")
    parser.add_argument("tracker_id", help="トラッカーID")

    args = parser.parse_args()

    handler = CreateCommandHandler()
    success, message = handler.execute_create_command(args.tracker_id)

    print(message)
    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
