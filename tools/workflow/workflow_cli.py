#!/usr/bin/env python3
"""
ワークフロー CLI - ワークフロー強制実行システムのコマンドラインインターフェース
INCI-006 解決策: ワークフロー強制実行との対話用シンプルCLI

ワークフロー強制実行の管理、状態確認、ワークフローシステムの手動制御のための
コマンドラインインターフェースを提供します。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools.workflow.subagent_command_handler import SubAgentCommandHandler

# ログファイルパス設定
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "workflow_cli.log"

# Console: WARNING以上、File: INFO以上
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[console_handler, file_handler],
    force=True,
)
logger = logging.getLogger(__name__)


def get_workflow_controller():
    """ワークフローコントローラーインスタンスを取得"""
    try:
        # Add current directory to Python path
        import os
        import sys

        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        from tools.interface.workflow_controller import get_workflow_controller

        return get_workflow_controller()
    except ImportError as e:
        logger.error(f"ワークフローコントローラーのインポートに失敗: {e}")
        return None


def plan_tracker(
    tracker_id: str, summary: str, details: str, author_name: str, priority: str = "medium"
) -> bool:
    """Google Sheetsにトラッカーを起票"""
    try:
        # ワークスペース設定を保存
        from config.workspace_config import get_workspace_config

        config = get_workspace_config()

        workspace_path = f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace/{tracker_id}"
        input_path = f"/mnt/c/AItools/lora/train/{author_name}/org/kana05"

        if not config.set_workspace_config(tracker_id, author_name, workspace_path, input_path):
            print(f"❌ ワークスペース設定の保存に失敗しました")
            return False

        print(f"✅ ワークスペース設定を保存しました:")
        print(f"   作者名: {author_name}")
        print(f"   ワークスペース: {workspace_path}")

        from tools.workflow.plan_command_handler import PlanCommandHandler

        handler = PlanCommandHandler()
        success, message = handler.execute_plan_command(tracker_id, summary, details, priority)
        print(message)
        return success
    except Exception as e:
        print(f"❌ planコマンド実行エラー: {e}")
        return False


def create_tracker(tracker_id: str) -> bool:
    """新しいトラッカーワークフローを作成（SQLite専用）"""
    try:
        # ワークスペース設定の検証
        from config.workspace_config import validate_tracker_setup

        validation = validate_tracker_setup(tracker_id)

        if not validation["is_configured"]:
            for error in validation["errors"]:
                print(error)
            return False

        from tools.workflow.create_command_handler import CreateCommandHandler

        handler = CreateCommandHandler()
        success, message = handler.execute_create_command(tracker_id)
        print(message)
        return success
    except Exception as e:
        print(f"❌ createコマンド実行エラー: {e}")
        return False


def get_status(tracker_id: str) -> bool:
    """トラッカーのワークフロー状態を取得"""
    try:
        # ワークスペース設定の検証
        from config.workspace_config import validate_tracker_setup

        validation = validate_tracker_setup(tracker_id)

        if not validation["is_configured"]:
            for error in validation["errors"]:
                print(error)
            return False

        controller = get_workflow_controller()
        if not controller:
            return False

        status = controller.get_workflow_status(tracker_id)

        if "error" in status:
            print(f"❌ エラー: {status['error']}")
            return False

        print(f"📋 {tracker_id} のワークフロー状態")
        print(f"   現在のフェーズ: {status.get('current_phase', '不明')}")
        print(f"   現在のステップ: {status.get('current_step', '不明')}")
        print(f"   進行可能: {'✅' if status.get('can_proceed', False) else '❌'}")

        # 完了したステップを表示
        completed_steps = status.get("completed_steps", [])
        if completed_steps:
            print(f"   完了したステップ ({len(completed_steps)}):")
            for step in completed_steps:
                print(f"     ✅ {step['step_id']} ({step.get('completed_at', '時刻不明')})")

        # 承認待ちを表示
        pending_approvals = status.get("pending_approvals", [])
        if pending_approvals:
            print(f"   承認待ち ({len(pending_approvals)}):")
            for approval in pending_approvals:
                print(f"     ⏳ {approval['title']} (ID: {approval['approval_id']})")

        # ブロックされたアクションを表示
        blocked_actions = status.get("blocked_actions", [])
        if blocked_actions:
            print(f"   ブロックされたアクション ({len(blocked_actions)}):")
            for action in blocked_actions:
                print(f"     🚫 {action['action']}: {action['reason']}")

        # 現在のステップ指示を表示
        if "current_step_instructions" in status:
            instructions = status["current_step_instructions"]
            print(f"\n📝 現在のステップ指示:")
            print(f"   タイトル: {instructions['title']}")
            print(f"   説明: {instructions['description']}")

            if instructions["required_actions"]:
                print(f"   必要なアクション:")
                for action in instructions["required_actions"]:
                    print(f"     • {action}")

            if instructions["validation_criteria"]:
                print(f"   検証基準:")
                for criterion in instructions["validation_criteria"]:
                    print(f"     ✓ {criterion}")

            if instructions["approval_required"]:
                print(f"   ⚠️  承認が必要: はい")

            if not instructions["can_proceed"] and instructions.get("blocking_reasons"):
                print(f"   🚫 ブロック理由:")
                for reason in instructions["blocking_reasons"]:
                    print(f"     • {reason}")

        # バックグラウンドプロセス状態を表示
        if "background_process" in status:
            process = status["background_process"]
            print(f"\n🔄 バックグラウンドプロセス:")
            print(f"   状態: {process['status']}")
            print(f"   ステップ: {process['step']}")
            if process["status"] == "running":
                print(f"   PID: {process['pid']}")
                print(f"   ログ: {process['log_file']}")

        return True
    except Exception as e:
        logger.error(f"ステータス取得エラー [{tracker_id}]: {e}")
        print(f"❌ ステータス取得エラー: {e}")
        return False


def get_instructions(tracker_id: str) -> bool:
    """現在のステップ指示を取得"""
    try:
        controller = get_workflow_controller()
        if not controller:
            return False

        instructions = controller.get_current_step_instructions(tracker_id)
        if not instructions:
            print(f"❌ トラッカーの指示が見つかりません: {tracker_id}")
            return False

        print(f"📝 {tracker_id} の現在のステップ指示")
        print(f"   ステップ: {instructions.step_id}")
        print(f"   タイトル: {instructions.title}")
        print(f"   説明: {instructions.description}")

        if instructions.required_actions:
            print(f"\n🔧 必要なアクション:")
            for i, action in enumerate(instructions.required_actions, 1):
                print(f"   {i}. {action}")

        if instructions.validation_criteria:
            print(f"\n✅ 検証基準:")
            for criterion in instructions.validation_criteria:
                print(f"   • {criterion}")

        if instructions.approval_required:
            print(f"\n⚠️  このステップには人間の承認が必要です")

        if not instructions.can_proceed:
            print(f"\n🚫 進行できません - 以下によりブロック:")
            for reason in instructions.blocking_reasons or []:
                print(f"   • {reason}")
        else:
            print(f"\n✅ 進行準備完了")

        return True
    except Exception as e:
        logger.error(f"指示取得エラー [{tracker_id}]: {e}")
        print(f"❌ 指示取得エラー: {e}")
        return False


def attempt_step(tracker_id: str) -> bool:
    """現在のステップの完了を試行"""
    try:
        controller = get_workflow_controller()
        if not controller:
            return False

        print(f"🔄 {tracker_id} の現在のステップ完了を試行中...")

        result = controller.attempt_step_completion(tracker_id)

        if result.status == "completed":
            print(f"✅ ステップが正常に完了しました!")
            print(f"   メッセージ: {result.message}")
            if result.next_step:
                print(f"   次のステップ: {result.next_step}")
        elif result.status == "pending_approval":
            print(f"⏳ ステップには承認が必要です")
            print(f"   承認ID: {result.approval_id}")
            print(f"   メッセージ: {result.message}")
            print(f"\n   承認するには、承認ディレクトリで指示を確認してください")
        elif result.status == "blocked":
            print(f"🚫 ステップがブロックされています")
            print(f"   メッセージ: {result.message}")
            print(f"\n💡 対処法を確認: /workflow-troubleshoot または instructions {tracker_id}")
        elif result.status == "failed":
            print(f"❌ ステップが失敗しました")
            print(f"   メッセージ: {result.message}")
            print(f"\n💡 対処法を確認: /workflow-troubleshoot または instructions {tracker_id}")

        return result.status in ["completed", "pending_approval"]
    except Exception as e:
        logger.error(f"ステップ実行エラー [{tracker_id}]: {e}")
        print(f"❌ ステップ実行エラー: {e}")
        return False


def list_approvals() -> bool:
    """承認待ちリストを表示"""
    try:
        controller = get_workflow_controller()
        if not controller or not controller.approval_controller:
            print("❌ 承認コントローラーが利用できません")
            return False

        pending = controller.approval_controller.list_pending_approvals()

        if not pending:
            print("✅ 承認待ちはありません")
            return True

        print(f"⏳ 承認待ち ({len(pending)}):")
        for approval in pending:
            print(f"\n📋 {approval['approval_id']}")
            print(f"   トラッカー: {approval['tracker_id']}")
            print(f"   ステップ: {approval['step_name']}")
            print(f"   優先度: {approval['priority']}")
            print(f"   要求日時: {approval['requested_at']}")
            print(f"   残り時間: {approval.get('time_remaining_hours', 0):.1f} 時間")

            if approval.get("approval_criteria"):
                print(f"   基準:")
                for criterion in approval["approval_criteria"]:
                    print(f"     • {criterion}")

        return True
    except Exception as e:
        logger.error(f"承認リスト取得エラー: {e}")
        print(f"❌ 承認リスト取得エラー: {e}")
        return False


def check_process(tracker_id: str) -> bool:
    """バックグラウンドプロセス状態を確認"""
    try:
        controller = get_workflow_controller()
        if not controller or not controller.executor:
            print("❌ エグゼキューターが利用できません")
            return False

        status = controller.executor.check_process_status(tracker_id)

        if not status:
            print(f"ℹ️  {tracker_id} のバックグラウンドプロセスは実行されていません")
            return True

        print(f"🔄 {tracker_id} のバックグラウンドプロセス:")
        print(f"   状態: {status['status']}")
        print(f"   ステップ: {status['step']}")
        print(f"   開始時刻: {status['started_at']}")

        if status["status"] == "running":
            print(f"   PID: {status['pid']}")
            print(f"   ログ: {status['log_file']}")
        elif status["status"] == "completed":
            print(f"   完了時刻: {status['completed_at']}")
            print(f"   リターンコード: {status['return_code']}")
            if "validation" in status:
                print(f"   検証: {status['validation']}")

        return True
    except Exception as e:
        logger.error(f"プロセス状態確認エラー [{tracker_id}]: {e}")
        print(f"❌ プロセス状態確認エラー: {e}")
        return False


# SubAgent関連のコマンドハンドラー関数
def subagent_extraction(tracker_id: str, input_path: str = None, max_files: int = None) -> bool:
    """SubAgent抽出処理開始"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_extraction(tracker_id, input_path, max_files)
    except Exception as e:
        logger.error(f"SubAgent抽出エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent抽出エラー: {e}")
        return False


def subagent_status(tracker_id: str) -> bool:
    """SubAgent状態確認"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_status(tracker_id)
    except Exception as e:
        logger.error(f"SubAgent状態確認エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent状態確認エラー: {e}")
        return False


def subagent_retry(tracker_id: str) -> bool:
    """SubAgent再実行"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_retry(tracker_id)
    except Exception as e:
        logger.error(f"SubAgent再実行エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent再実行エラー: {e}")
        return False


def subagent_terminate(tracker_id: str, force: bool = False) -> bool:
    """SubAgent終了"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_terminate(tracker_id, force)
    except Exception as e:
        logger.error(f"SubAgent終了エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent終了エラー: {e}")
        return False


def subagent_wait(tracker_id: str, timeout_minutes: int = 60) -> bool:
    """SubAgent完了待機"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_wait(tracker_id, timeout_minutes)
    except Exception as e:
        logger.error(f"SubAgent待機エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent待機エラー: {e}")
        return False


def subagent_cleanup(tracker_id: str) -> bool:
    """SubAgentロック強制クリーンアップ"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_cleanup(tracker_id)
    except Exception as e:
        logger.error(f"SubAgentクリーンアップエラー [{tracker_id}]: {e}")
        print(f"❌ SubAgentクリーンアップエラー: {e}")
        return False


def subagent_locks_status() -> bool:
    """全SubAgentロック状況確認"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_locks_status()
    except Exception as e:
        logger.error(f"SubAgentロック状況確認エラー: {e}")
        print(f"❌ SubAgentロック状況確認エラー: {e}")
        return False


def subagent_cleanup_all() -> bool:
    """全SubAgentロック強制クリーンアップ"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_cleanup_all()
    except Exception as e:
        logger.error(f"SubAgent全クリーンアップエラー: {e}")
        print(f"❌ SubAgent全クリーンアップエラー: {e}")
        return False


def subagent_auto_retry_check(tracker_id: str) -> bool:
    """SubAgent自動再実行条件確認"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_auto_retry_check(tracker_id)
    except Exception as e:
        logger.error(f"SubAgent自動再実行確認エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent自動再実行確認エラー: {e}")
        return False


def subagent_auto_retry(tracker_id: str) -> bool:
    """SubAgent自動再実行実行"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_auto_retry(tracker_id)
    except Exception as e:
        logger.error(f"SubAgent自動再実行エラー [{tracker_id}]: {e}")
        print(f"❌ SubAgent自動再実行エラー: {e}")
        return False


def subagent_auto_retry_all() -> bool:
    """全SubAgent自動再実行バッチ実行"""
    try:
        handler = SubAgentCommandHandler()
        return handler.handle_subagent_auto_retry_all()
    except Exception as e:
        logger.error(f"SubAgent全自動再実行エラー: {e}")
        print(f"❌ SubAgent全自動再実行エラー: {e}")
        return False


def generate_template(tracker_id: str, output_path: str = None) -> bool:
    """統合テンプレートを生成"""
    controller = get_workflow_controller()
    if not controller:
        print("❌ ワークフローコントローラーが利用できません")
        return False

    # 現在の状態を取得
    status = controller.get_workflow_status(tracker_id)
    instructions = controller.get_current_step_instructions(tracker_id)

    if output_path is None:
        output_path = f"workspace/{tracker_id}/integrated_template.md"

    # テンプレート生成
    template_content = f"""# {tracker_id} 統合ワークフローテンプレート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**トラッカーID**: {tracker_id}

## 🤖 機械的状態管理

### 現在の状態
- **フェーズ**: {status.get('current_phase', '不明')}
- **ステップ**: {status.get('current_step', '不明')}
- **進行可能**: {'✅' if status.get('can_proceed', False) else '❌'}

### CLI コマンド
```bash
# 状態確認
python tools/workflow/workflow_cli.py status {tracker_id}

# 次ステップ指示
python tools/workflow/workflow_cli.py instructions {tracker_id}

# ステップ実行
python tools/workflow/workflow_cli.py step {tracker_id}

# 承認待ち確認
python tools/workflow/workflow_cli.py approvals
```

## 📋 現在のステップ指示

"""

    if instructions:
        template_content += f"""### {instructions.title}
**説明**: {instructions.description}

#### 必要なアクション:
"""
        for i, action in enumerate(instructions.required_actions, 1):
            template_content += f"{i}. {action}\n"

        template_content += "\n#### 検証基準:\n"
        for criterion in instructions.validation_criteria:
            template_content += f"- [ ] {criterion}\n"

        if instructions.approval_required:
            template_content += "\n⚠️ **このステップには承認が必要です**\n"

        if not instructions.can_proceed:
            template_content += "\n🚫 **ブロック理由:**\n"
            for reason in instructions.blocking_reasons or []:
                template_content += f"- {reason}\n"

    template_content += """

## 📊 進捗チェックリスト

### Phase 0.5: ブランチ検証
- [ ] CLI実行: `python tools/workflow/workflow_cli.py step {tracker_id}`
- [ ] ブランチ確認: feature/{tracker_id} ブランチで作業中

### Phase 1: 計画・準備
- [ ] 要件分析完了
- [ ] SOW作成完了
- [ ] 計画承認取得

### Phase 2: 実装・テスト
- [ ] 実装完了
- [ ] テスト実行完了
- [ ] 品質確認完了

### Phase 3: 品質ワークフロー
- [ ] 品質ワークフロー実行
- [ ] ダッシュボード生成
- [ ] 統計分析実行

### Phase 4: 完了・承認
- [ ] 最終検証完了
- [ ] Pull Request作成
- [ ] 最終承認取得

## 🔄 統合ワークフロー手順

1. **状態確認**: `python tools/workflow/workflow_cli.py status {tracker_id}`
2. **指示取得**: `python tools/workflow/workflow_cli.py instructions {tracker_id}`
3. **作業実行**: 指示に従って手動作業実行
4. **ステップ進行**: `python tools/workflow/workflow_cli.py step {tracker_id}`
5. **承認処理**: 必要に応じて承認ファイル作成
6. **繰り返し**: 完了まで1-5を繰り返し

## 📚 関連ドキュメント

- [統合テンプレート](docs/workflows/templates/unified_tracker_template.md)
- [13ステップチェックリスト](docs/workflows/checklists/tracker_workflow_checklist.md)
- [CLI統合ガイド](docs/workflows/integration/cli_template_integration_guide.md)

---
*このテンプレートは機械的状態管理と人間の判断を統合したハイブリッドワークフローです*
""".format(
        tracker_id=tracker_id
    )

    try:
        # ディレクトリ作成
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ファイル書き込み
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        print(f"✅ 統合テンプレートを生成しました: {output_path}")
        print(f"📋 現在のステップ: {status.get('current_step', '不明')}")
        if instructions:
            print(f"📝 次のアクション: {instructions.title}")

        return True

    except Exception as e:
        print(f"❌ テンプレート生成に失敗: {e}")
        return False


def check_sheets_status(tracker_id: str) -> bool:
    """Google Sheets状態を確認"""
    try:
        # 既存のGoogle Sheets連携システムを使用
        import os
        import sys

        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        from tools.progress_tracker.progress_manager import ProgressManager

        # Google Sheetsマネージャー初期化
        progress_manager = ProgressManager()

        # タスク情報取得
        task = progress_manager.get_task(tracker_id)

        if not task:
            print(f"❌ Google Sheetsにトラッカーが見つかりません: {tracker_id}")
            print(f"📋 作成するには:")
            print(f'   python tools/progress_tracker/cli.py create {tracker_id} "説明"')
            return False

        print(f"📊 Google Sheets状態: {tracker_id}")
        print(f"   トラッカーID: {task.tracker_id}")
        print(f"   説明: {task.description}")
        print(f"   ステータス: {task.status.value}")
        print(f"   作成日時: {task.created_date}")
        print(f"   更新日時: {task.updated_date}")

        if hasattr(task, "progress_percentage"):
            print(f"   進捗率: {task.progress_percentage}%")

        return True

    except Exception as e:
        print(f"❌ Google Sheets確認でエラーが発生: {e}")
        print(f"📋 Google Sheets設定を確認してください:")
        print(f"   python tools/progress_tracker/cli.py check-config")
        return False


def show_guide() -> bool:
    """統合ワークフローガイドを表示"""
    print(
        """
🚀 **統合ワークフローガイド**

## 🆕 新しいワークフロー（推奨）

### 1. Google Sheetsに起票
```bash
python tools/workflow/workflow_cli.py plan TRACKER-001 "概要" "詳細説明"
```

### 2. ワークフロー状態管理を開始
```bash
python tools/workflow/workflow_cli.py create TRACKER-001
```

### 3. 現在の状況を確認
```bash
python tools/workflow/workflow_cli.py status TRACKER-001
```

### 4. 次に何をすべきか確認
```bash
python tools/workflow/workflow_cli.py instructions TRACKER-001
```

### 5. ステップを実行
```bash
python tools/workflow/workflow_cli.py step TRACKER-001
```

### 6. 承認待ちを確認
```bash
python tools/workflow/workflow_cli.py approvals
```

### 7. Google Sheets状態確認
```bash
python tools/workflow/workflow_cli.py sheets TRACKER-001
```

### 8. 統合テンプレート生成
```bash
python tools/workflow/workflow_cli.py template TRACKER-001
```

## 🔄 統合ワークフロー手順

1. **起票**: `plan` でGoogle Sheetsに起票（概要・詳細・優先度設定）
2. **開始**: `create` でワークフロー状態管理開始（SQLite専用）
3. **状態確認**: `status` で現在状況把握
4. **Google Sheets確認**: `sheets` でGoogle Sheets状態確認
5. **指示確認**: `instructions` で次アクション確認
6. **作業実行**: 指示に従って手動作業
7. **進行**: `step` でステップ完了試行
8. **承認**: 必要に応じて承認ファイル作成
9. **繰り返し**: 3-8を完了まで繰り返し

## 🚨 重要な変更点

- **planコマンド新設**: Google Sheets起票専用（3引数必須、20,000文字制限）
- **createコマンド変更**: SQLite専用に特化（Google Sheets機能削除）
- **分離設計**: 起票と状態管理を明確に分離
- **統合推奨**: plan→createの順序で実行を推奨

## 📋 コマンド詳細

### planコマンド
- **目的**: Google Sheetsへの起票
- **引数**: tracker_id, summary, details, [priority]
- **制限**: 詳細20,000文字以内
- **機能**: リトライ、エラーハンドリング、優先度設定

### createコマンド  
- **目的**: SQLiteワークフロー状態管理
- **引数**: tracker_id
- **機能**: 既存確認、状態作成、進行管理

## 🚨 重要なポイント

- **ブランチ確認**: 必ず feature/{TRACKER_ID} ブランチで作業
- **統合ワークフロー**: plan→createの順序で実行
- **承認システム**: 承認が必要な場合は指示に従ってファイル作成
- **日本語表示**: すべてのメッセージが日本語で表示
- **状態管理**: 外部データベースで状態を管理（改ざん不可）

## 📚 詳細情報

詳細な手順は以下のドキュメントを参照してください：
- docs/workflows/templates/unified_tracker_template.md
- docs/workflows/checklists/tracker_workflow_checklist.md
- docs/workflows/integration/cli_template_integration_guide.md
"""
    )
    return True


def check_virtual_environment():
    """sam-env仮想環境がアクティブかチェック"""
    import os
    import sys

    # CI環境の場合はチェックをスキップ
    if os.environ.get("GITHUB_ACTIONS"):
        print("🔧 CI環境のため仮想環境チェックをスキップします")
        return True

    # VIRTUAL_ENVがsam-envを指しているかチェック
    virtual_env = os.environ.get("VIRTUAL_ENV", "")

    if not virtual_env or "sam-env" not in virtual_env:
        print("❌ sam-env仮想環境がアクティブではありません")
        print("🔧 以下のコマンドで仮想環境を有効化してください:")
        print("   source sam-env/bin/activate  # Linux/WSL")
        print("   sam-env\\Scripts\\activate     # Windows")
        print()
        return False

    print(f"✅ 仮想環境がアクティブです: {virtual_env}")
    return True


def main():
    """メインCLIエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="ワークフロー強制実行システム CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s plan TRACKER-001 "概要" "詳細" "作者名"  # Google Sheetsにトラッカーを起票
  %(prog)s create TRACKER-001                     # 新しいトラッカーワークフローを作成（SQLite専用）
  %(prog)s status TRACKER-001                     # ワークフロー状態を取得
  %(prog)s instructions TRACKER-001               # 現在のステップ指示を取得
  %(prog)s step TRACKER-001                       # 現在のステップの完了を試行
  %(prog)s approvals                              # 承認待ちリストを表示
  %(prog)s process TRACKER-001                    # バックグラウンドプロセス状態を確認
  %(prog)s sheets TRACKER-001                     # Google Sheets状態を確認
  %(prog)s template TRACKER-001                   # 統合テンプレートを生成
  %(prog)s guide                                  # 統合ワークフローガイドを表示

統合ワークフロー推奨手順:
  1. %(prog)s plan TRACKER-001 "概要" "詳細" "yado"  # Google Sheets起票（作者名必須）
  2. %(prog)s create TRACKER-001                     # ワークフロー状態管理開始
  3. %(prog)s step TRACKER-001                       # ステップ実行

注意:
- planコマンドでは作者名が必須です（例: yado, kiri, zundamon）
- ワークスペースパスは /mnt/c/AItools/lora/train/{作者名}/tracker-workspace/{トラッカーID} となります
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # Google Sheets起票
    plan_parser = subparsers.add_parser("plan", help="Google Sheetsにトラッカーを起票")
    plan_parser.add_argument("tracker_id", help="トラッカーID (例: TRACKER-001)")
    plan_parser.add_argument("summary", help="概要（200文字以内推奨）")
    plan_parser.add_argument("details", help="詳細（20,000文字以内）")
    plan_parser.add_argument("author_name", help="作者名 (例: yado, kiri, zundamon)")
    plan_parser.add_argument(
        "--priority",
        choices=["highest", "high", "medium", "low"],
        default="medium",
        help="優先度 (デフォルト: medium)",
    )

    # トラッカー作成
    create_parser = subparsers.add_parser("create", help="新しいトラッカーワークフローを作成（SQLite専用）")
    create_parser.add_argument("tracker_id", help="トラッカーID (例: TRACKER-001)")

    # 状態取得
    status_parser = subparsers.add_parser("status", help="ワークフロー状態を取得")
    status_parser.add_argument("tracker_id", help="トラッカーID")

    # 指示取得
    instructions_parser = subparsers.add_parser("instructions", help="現在のステップ指示を取得")
    instructions_parser.add_argument("tracker_id", help="トラッカーID")

    # ステップ試行
    step_parser = subparsers.add_parser("step", help="現在のステップの完了を試行")
    step_parser.add_argument("tracker_id", help="トラッカーID")

    # 承認リスト
    subparsers.add_parser("approvals", help="承認待ちリストを表示")

    # プロセス確認
    process_parser = subparsers.add_parser("process", help="バックグラウンドプロセス状態を確認")
    process_parser.add_argument("tracker_id", help="トラッカーID")

    # Google Sheets確認
    sheets_parser = subparsers.add_parser("sheets", help="Google Sheets状態を確認")
    sheets_parser.add_argument("tracker_id", help="トラッカーID")

    # テンプレート生成
    template_parser = subparsers.add_parser("template", help="統合テンプレートを生成")
    template_parser.add_argument("tracker_id", help="トラッカーID")
    template_parser.add_argument("--output", "-o", help="出力ファイルパス", default=None)

    # ガイド表示
    subparsers.add_parser("guide", help="統合ワークフローガイドを表示")

    # SubAgentコマンド
    subagent_extraction_parser = subparsers.add_parser("subagent-extraction", help="SubAgent抽出処理開始")
    subagent_extraction_parser.add_argument("tracker_id", help="トラッカーID")
    subagent_extraction_parser.add_argument(
        "input_path", nargs="?", help="入力ディレクトリパス（省略時はワークスペース設定から自動取得）"
    )
    subagent_extraction_parser.add_argument(
        "--max-files", type=int, help="処理する最大ファイル数（省略時は全ファイル処理）"
    )

    subagent_status_parser = subparsers.add_parser("subagent-status", help="SubAgent状態確認")
    subagent_status_parser.add_argument("tracker_id", help="トラッカーID")

    subagent_retry_parser = subparsers.add_parser("subagent-retry", help="SubAgent再実行")
    subagent_retry_parser.add_argument("tracker_id", help="トラッカーID")

    subagent_terminate_parser = subparsers.add_parser("subagent-terminate", help="SubAgent終了")
    subagent_terminate_parser.add_argument("tracker_id", help="トラッカーID")
    subagent_terminate_parser.add_argument("--force", action="store_true", help="強制終了")

    subagent_wait_parser = subparsers.add_parser("subagent-wait", help="SubAgent完了待機")
    subagent_wait_parser.add_argument("tracker_id", help="トラッカーID")
    subagent_wait_parser.add_argument("--timeout", type=int, default=60, help="タイムアウト（分）")

    subagent_cleanup_parser = subparsers.add_parser("subagent-cleanup", help="SubAgentロック強制クリーンアップ")
    subagent_cleanup_parser.add_argument("tracker_id", help="トラッカーID")

    subparsers.add_parser("subagent-locks-status", help="全SubAgentロック状況確認")

    subparsers.add_parser("subagent-cleanup-all", help="全SubAgentロック強制クリーンアップ")

    subagent_auto_retry_check_parser = subparsers.add_parser(
        "subagent-auto-retry-check", help="SubAgent自動再実行条件確認"
    )
    subagent_auto_retry_check_parser.add_argument("tracker_id", help="トラッカーID")

    subagent_auto_retry_parser = subparsers.add_parser(
        "subagent-auto-retry", help="SubAgent自動再実行実行"
    )
    subagent_auto_retry_parser.add_argument("tracker_id", help="トラッカーID")

    subparsers.add_parser("subagent-auto-retry-all", help="全SubAgent自動再実行バッチ実行")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 仮想環境チェック（最優先実行）- ヘルプ表示後に実行
    if not check_virtual_environment():
        print("⚠️  仮想環境設定後に再実行してください")
        return 1

    try:
        if args.command == "plan":
            success = plan_tracker(
                args.tracker_id, args.summary, args.details, args.author_name, args.priority
            )
        elif args.command == "create":
            success = create_tracker(args.tracker_id)
        elif args.command == "status":
            success = get_status(args.tracker_id)
        elif args.command == "instructions":
            success = get_instructions(args.tracker_id)
        elif args.command == "step":
            success = attempt_step(args.tracker_id)
        elif args.command == "approvals":
            success = list_approvals()
        elif args.command == "process":
            success = check_process(args.tracker_id)
        elif args.command == "template":
            success = generate_template(args.tracker_id, args.output)
        elif args.command == "sheets":
            success = check_sheets_status(args.tracker_id)
        elif args.command == "guide":
            success = show_guide()
        # SubAgentコマンド処理
        elif args.command == "subagent-extraction":
            success = subagent_extraction(
                args.tracker_id, getattr(args, "input_path", None), getattr(args, "max_files", None)
            )
        elif args.command == "subagent-status":
            success = subagent_status(args.tracker_id)
        elif args.command == "subagent-retry":
            success = subagent_retry(args.tracker_id)
        elif args.command == "subagent-terminate":
            success = subagent_terminate(args.tracker_id, args.force)
        elif args.command == "subagent-wait":
            success = subagent_wait(args.tracker_id, args.timeout)
        elif args.command == "subagent-cleanup":
            success = subagent_cleanup(args.tracker_id)
        elif args.command == "subagent-locks-status":
            success = subagent_locks_status()
        elif args.command == "subagent-cleanup-all":
            success = subagent_cleanup_all()
        elif args.command == "subagent-auto-retry-check":
            success = subagent_auto_retry_check(args.tracker_id)
        elif args.command == "subagent-auto-retry":
            success = subagent_auto_retry(args.tracker_id)
        elif args.command == "subagent-auto-retry-all":
            success = subagent_auto_retry_all()
        else:
            print(f"❌ 不明なコマンド: {args.command}")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによって操作がキャンセルされました")
        return 1
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
