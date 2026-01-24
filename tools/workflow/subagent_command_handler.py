#!/usr/bin/env python3
"""
SubAgent連携コマンドハンドラー - KIRO-011実装
ワークフローシステムとSubAgent監視システムの統合
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# プロジェクトパスを追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tools.workflow.state_manager import get_state_manager, StepStatus
from tools.workflow.subagent_monitor import get_subagent_monitor, SubAgentStatus
from tools.workflow.subagent_lock_manager import get_lock_manager, create_execution_context
from config.workspace_config import get_workspace_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubAgentCommandHandler:
    """SubAgent連携コマンド処理クラス"""

    def __init__(self):
        self.state_manager = get_state_manager()
        self.subagent_monitor = get_subagent_monitor()
        self.lock_manager = get_lock_manager()
        self.workspace_config = get_workspace_config()

    def handle_subagent_extraction(self, tracker_id: str, input_path: str = None, max_files: int = None) -> bool:
        """SubAgent抽出処理の開始"""
        logger.info(f"SubAgent抽出処理開始: {tracker_id}")

        try:
            # 二重起動防止チェック
            if self.lock_manager.is_duplicate_execution_risk(tracker_id, "extraction"):
                existing_lock = self.lock_manager.get_lock_owner_info(tracker_id, "extraction")
                if existing_lock:
                    print(f"❌ エラー: 抽出処理が既に実行中です")
                    print(f"📋 実行中の詳細:")
                    print(f"  • PID: {existing_lock.get('pid')}")
                    print(f"  • 開始時刻: {existing_lock.get('created_at')}")
                    print(f"  • ホスト: {existing_lock.get('hostname')}")
                    print("🔧 対処方法:")
                    print(f"  1. python tools/workflow/workflow_cli.py subagent-status {tracker_id}")
                    print(f"  2. python tools/workflow/workflow_cli.py subagent-terminate {tracker_id}")
                    return False
                else:
                    print(f"❌ エラー: SubAgent抽出処理の二重実行リスクが検出されました")
                    print("🔧 強制クリーンアップ:")
                    print(f"python tools/workflow/workflow_cli.py subagent-cleanup {tracker_id}")
                    return False

            # ワークスペース設定取得
            config = self.workspace_config.get_workspace_config(tracker_id)
            if not config:
                logger.error(f"ワークスペース設定が見つかりません: {tracker_id}")
                print(f"❌ エラー: トラッカー {tracker_id} のワークスペース設定が見つかりません")
                print("📋 以下のコマンドで事前設定が必要です:")
                print(f"python tools/workflow/workflow_cli.py plan {tracker_id} <概要> <詳細> <作者名>")
                return False

            workspace_path = config['workspace_path']

            # 入力パスの決定
            if input_path:
                # カスタムパスが指定された場合
                input_directory = input_path
                logger.info(f"カスタム入力パス使用: {input_path}")
                print(f"📁 カスタム入力パス: {input_path}")
            else:
                # 従来通りワークスペース設定から取得
                input_directory = self.workspace_config.get_input_directory(tracker_id)
                if not input_directory:
                    logger.error(f"入力ディレクトリが見つかりません: {tracker_id}")
                    print(f"❌ エラー: トラッカー {tracker_id} の入力ディレクトリが見つかりません")
                    return False

            # SubAgent抽出コマンド構築
            extraction_command = self._build_extraction_command(
                input_directory, workspace_path, tracker_id, max_files
            )

            # SubAgent監視システムに登録
            success = self.subagent_monitor.register_subagent(
                tracker_id=tracker_id,
                process_type="extraction",
                command=extraction_command,
                workspace_path=workspace_path,
                expected_duration=3600  # 1時間
            )

            if not success:
                print(f"❌ SubAgent登録失敗: {tracker_id}")
                return False

            # SubAgent開始
            success = self.subagent_monitor.start_subagent(tracker_id, "extraction")

            if success:
                print(f"🚀 SubAgent抽出処理開始: {tracker_id}")
                print(f"📁 入力ディレクトリ: {input_directory}")
                print(f"📁 出力ワークスペース: {workspace_path}")
                print("⏳ 処理状況は以下コマンドで確認できます:")
                print(f"python tools/workflow/workflow_cli.py subagent-status {tracker_id}")
                return True
            else:
                print(f"❌ SubAgent開始失敗: {tracker_id}")
                return False

        except Exception as e:
            logger.error(f"SubAgent抽出処理エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_status(self, tracker_id: str) -> bool:
        """SubAgent状態確認"""
        try:
            # 抽出処理状態確認
            extraction_status = self.subagent_monitor.check_subagent_status(tracker_id, "extraction")

            print(f"📊 SubAgent状態確認: {tracker_id}")
            print("="*50)
            print(f"🔄 抽出処理状態: {extraction_status.value}")

            # アクティブプロセス一覧取得
            active_processes = self.subagent_monitor.get_all_active_processes()
            tracker_processes = [p for p in active_processes if p['tracker_id'] == tracker_id]

            if tracker_processes:
                print("\n📋 プロセス詳細:")
                for process in tracker_processes:
                    elapsed_min = int(process['elapsed_seconds'] / 60)
                    expected_min = int(process['expected_duration'] / 60)

                    print(f"  • プロセスタイプ: {process['process_type']}")
                    print(f"  • 状態: {process['status']}")
                    print(f"  • PID: {process['pid'] or 'N/A'}")
                    print(f"  • 経過時間: {elapsed_min}分 / {expected_min}分")
                    print(f"  • リトライ回数: {process['retry_count']}")

            # 完了・失敗時の詳細情報
            if extraction_status == SubAgentStatus.COMPLETED:
                print("\n✅ 抽出処理完了")
                self._show_completion_details(tracker_id)
            elif extraction_status == SubAgentStatus.FAILED:
                print("\n❌ 抽出処理失敗")
                self._show_failure_details(tracker_id)
            elif extraction_status == SubAgentStatus.TIMEOUT:
                print("\n⏰ 抽出処理タイムアウト")
                print("🔧 以下コマンドで再実行できます:")
                print(f"python tools/workflow/workflow_cli.py subagent-retry {tracker_id}")
            elif extraction_status == SubAgentStatus.STALLED:
                print("\n⚠️ 抽出処理停滞中")
                print("🔧 以下コマンドで再起動できます:")
                print(f"python tools/workflow/workflow_cli.py subagent-restart {tracker_id}")

            return True

        except Exception as e:
            logger.error(f"SubAgent状態確認エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_retry(self, tracker_id: str) -> bool:
        """SubAgent再実行"""
        try:
            print(f"🔄 SubAgent再実行: {tracker_id}")

            # 現在のプロセス終了
            self.subagent_monitor.terminate_subagent(tracker_id, "extraction", force=True)
            time.sleep(2)

            # 再実行
            success = self.subagent_monitor.start_subagent(tracker_id, "extraction")

            if success:
                print("✅ SubAgent再実行開始")
                return True
            else:
                print("❌ SubAgent再実行失敗")
                return False

        except Exception as e:
            logger.error(f"SubAgent再実行エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_terminate(self, tracker_id: str, force: bool = False) -> bool:
        """SubAgent終了"""
        try:
            print(f"🛑 SubAgent終了: {tracker_id} (強制: {force})")

            success = self.subagent_monitor.terminate_subagent(tracker_id, "extraction", force=force)

            if success:
                print("✅ SubAgent終了完了")
                return True
            else:
                print("❌ SubAgent終了失敗")
                return False

        except Exception as e:
            logger.error(f"SubAgent終了エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_cleanup(self, tracker_id: str) -> bool:
        """SubAgentロックファイル強制クリーンアップ"""
        try:
            print(f"🧹 SubAgentロックファイル強制クリーンアップ: {tracker_id}")

            # 現在のロック状況確認
            locks_info = self.lock_manager.check_existing_locks()
            print("📋 クリーンアップ前の状況:")
            print(f"  • アクティブロック数: {locks_info['active_count']}")
            print(f"  • 無効ロック数: {len(locks_info['stale_locks'])}")

            # 強制クリーンアップ実行
            cleaned_count = self.lock_manager.force_cleanup_locks(tracker_id)

            if cleaned_count > 0:
                print(f"✅ クリーンアップ完了: {cleaned_count}ファイル削除")
                print("🔄 再実行可能になりました:")
                print(f"python tools/workflow/workflow_cli.py subagent-extraction {tracker_id}")
                return True
            else:
                print("ℹ️ クリーンアップ対象なし")
                return True

        except Exception as e:
            logger.error(f"SubAgentクリーンアップエラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_locks_status(self) -> bool:
        """全SubAgentロック状況確認"""
        try:
            print("📊 SubAgentロック状況確認")
            print("="*50)

            locks_info = self.lock_manager.check_existing_locks()

            # グローバルロック
            if locks_info['global_lock']:
                lock = locks_info['global_lock']
                status_icon = "✅" if lock['is_valid'] else "❌"
                print(f"\n🌐 グローバルロック {status_icon}")
                print(f"  • トラッカー: {lock.get('tracker_id')}")
                print(f"  • PID: {lock.get('pid')}")
                print(f"  • 開始時刻: {lock.get('created_at')}")
                print(f"  • ホスト: {lock.get('hostname')}")

            # 個別ロック
            if locks_info['specific_locks']:
                print(f"\n📋 アクティブな個別ロック ({len(locks_info['specific_locks'])}件)")
                for i, lock in enumerate(locks_info['specific_locks'], 1):
                    print(f"  {i}. {lock.get('tracker_id')} / {lock.get('process_type')}")
                    print(f"     PID: {lock.get('pid')}, 開始: {lock.get('created_at')}")

            # 無効ロック
            if locks_info['stale_locks']:
                print(f"\n⚠️ 無効ロック ({len(locks_info['stale_locks'])}件)")
                for i, lock in enumerate(locks_info['stale_locks'], 1):
                    print(f"  {i}. {lock.get('tracker_id')} / {lock.get('process_type')}")
                    print(f"     PID: {lock.get('pid')} (プロセス停止済み)")

            # 総括
            if locks_info['active_count'] == 0:
                print("\n✅ アクティブなSubAgentプロセスなし")
            else:
                print(f"\n⚠️ {locks_info['active_count']}件のアクティブプロセス")

            if locks_info['stale_locks']:
                print("\n🔧 推奨対応:")
                print("python tools/workflow/workflow_cli.py subagent-cleanup-all")

            return True

        except Exception as e:
            logger.error(f"ロック状況確認エラー: {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_auto_retry_check(self, tracker_id: str) -> bool:
        """SubAgent自動再実行条件確認"""
        try:
            print(f"🔍 SubAgent自動再実行条件確認: {tracker_id}")

            # 自動再実行判定
            should_retry = self.subagent_monitor.should_auto_retry(tracker_id, "extraction")

            if should_retry:
                print("✅ 自動再実行条件を満たしています")
                print("📋 条件詳細:")
                print("  • 抽出結果0件または失敗状態")
                print("  • リトライ回数が上限内")
                print("  • 最小実行時間経過済み")
                print("\n🔧 実行オプション:")
                print(f"  • 手動実行: python tools/workflow/workflow_cli.py subagent-retry {tracker_id}")
                print(f"  • 自動実行: python tools/workflow/workflow_cli.py subagent-auto-retry {tracker_id}")
            else:
                print("❌ 自動再実行条件を満たしていません")

                # 詳細理由確認
                current_status = self.subagent_monitor.check_subagent_status(tracker_id, "extraction")
                print(f"  • 現在の状態: {current_status.value}")

                # プロセス情報確認
                active_processes = self.subagent_monitor.get_all_active_processes()
                tracker_process = next((p for p in active_processes if p['tracker_id'] == tracker_id), None)

                if tracker_process:
                    print(f"  • リトライ回数: {tracker_process['retry_count']}/3")
                    print(f"  • 経過時間: {int(tracker_process['elapsed_seconds']/60)}分")

            return True

        except Exception as e:
            logger.error(f"自動再実行条件確認エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_auto_retry(self, tracker_id: str) -> bool:
        """SubAgent自動再実行実行"""
        try:
            print(f"🔄 SubAgent自動再実行実行: {tracker_id}")

            # 条件確認
            should_retry = self.subagent_monitor.should_auto_retry(tracker_id, "extraction")

            if not should_retry:
                print("❌ 自動再実行条件を満たしていません")
                print("🔍 詳細確認:")
                print(f"python tools/workflow/workflow_cli.py subagent-auto-retry-check {tracker_id}")
                return False

            # 自動再実行実行
            success = self.subagent_monitor.auto_retry_subagent(tracker_id, "extraction")

            if success:
                print("✅ SubAgent自動再実行開始")
                print("⏳ 実行状況確認:")
                print(f"python tools/workflow/workflow_cli.py subagent-status {tracker_id}")
                return True
            else:
                print("❌ SubAgent自動再実行失敗")
                return False

        except Exception as e:
            logger.error(f"SubAgent自動再実行エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_auto_retry_all(self) -> bool:
        """全SubAgent自動再実行バッチ実行"""
        try:
            print("🔄 全SubAgent自動再実行バッチ実行")

            # バッチ実行
            retry_count = self.subagent_monitor.check_and_auto_retry_all()

            if retry_count > 0:
                print(f"✅ 自動再実行バッチ完了: {retry_count}プロセス")
                print("📋 実行されたプロセス:")

                # 実行後のアクティブプロセス確認
                active_processes = self.subagent_monitor.get_all_active_processes()
                for process in active_processes:
                    if process['status'] == 'running':
                        print(f"  • {process['tracker_id']} (PID: {process['pid']}, リトライ: {process['retry_count']})")

            else:
                print("ℹ️ 自動再実行対象プロセスなし")

            return True

        except Exception as e:
            logger.error(f"全自動再実行バッチエラー: {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_cleanup_all(self) -> bool:
        """全SubAgentロック強制クリーンアップ"""
        try:
            print("🧹 全SubAgentロック強制クリーンアップ")

            # 確認プロンプト
            response = input("❗ 全てのSubAgentロックを削除します。続行しますか？ (y/N): ")
            if response.lower() != 'y':
                print("❌ キャンセルされました")
                return False

            # 強制クリーンアップ実行
            cleaned_count = self.lock_manager.force_cleanup_locks()

            if cleaned_count > 0:
                print(f"✅ 全クリーンアップ完了: {cleaned_count}ファイル削除")
                return True
            else:
                print("ℹ️ クリーンアップ対象なし")
                return True

        except Exception as e:
            logger.error(f"全クリーンアップエラー: {e}")
            print(f"❌ エラー: {e}")
            return False

    def handle_subagent_wait(self, tracker_id: str, timeout_minutes: int = 60) -> bool:
        """SubAgent完了待機"""
        try:
            print(f"⏳ SubAgent完了待機: {tracker_id} (最大{timeout_minutes}分)")

            start_time = time.time()
            timeout_seconds = timeout_minutes * 60

            while time.time() - start_time < timeout_seconds:
                status = self.subagent_monitor.check_subagent_status(tracker_id, "extraction")

                if status == SubAgentStatus.COMPLETED:
                    print("✅ SubAgent完了")
                    return True
                elif status in [SubAgentStatus.FAILED, SubAgentStatus.TIMEOUT]:
                    print(f"❌ SubAgent異常終了: {status.value}")
                    return False

                # 進捗表示
                elapsed_min = int((time.time() - start_time) / 60)
                print(f"⏱️ 待機中... {elapsed_min}/{timeout_minutes}分 (状態: {status.value})")

                time.sleep(30)  # 30秒間隔で確認

            print("⏰ 待機タイムアウト")
            return False

        except KeyboardInterrupt:
            print("\n🛑 待機中断")
            return False
        except Exception as e:
            logger.error(f"SubAgent待機エラー: {tracker_id} - {e}")
            print(f"❌ エラー: {e}")
            return False

    def _build_extraction_command(self, input_directory: str, workspace_path: str, tracker_id: str, max_files: int = None) -> str:
        """SubAgent抽出コマンド構築"""
        output_directory = os.path.join(workspace_path, "extraction")

        # 抽出コマンド構築（SUBAGENT_EXECUTION=true で実行）
        # KIRO-011修正: 仮想環境絶対パス指定でsubprocess環境問題を解決
        project_root = "/mnt/c/AItools/segment-anything"
        python_path = f"{project_root}/sam-env/bin/python"

        # max_filesが指定された場合のみ--max-filesオプションを追加
        max_files_option = f"--max-files {max_files}" if max_files else ""

        command = (
            f"SUBAGENT_EXECUTION=true {python_path} "
            f"features/extraction/commands/extract_character.py "
            f"'{input_directory}' -o '{output_directory}' "
            f"--batch {max_files_option} --resume --verbose".strip()
        )

        return command

    def _show_completion_details(self, tracker_id: str):
        """完了詳細情報表示"""
        try:
            config = self.workspace_config.get_workspace_config(tracker_id)
            if not config:
                return

            extraction_dir = os.path.join(config['workspace_path'], "extraction")

            # 結果ファイル確認
            dashboard_file = os.path.join(extraction_dir, "dashboard.html")
            index_file = os.path.join(extraction_dir, "index.html")
            progress_file = os.path.join(extraction_dir, "progress.json")

            print("📋 生成ファイル:")
            if os.path.exists(dashboard_file):
                print(f"  ✅ {dashboard_file}")
            if os.path.exists(index_file):
                print(f"  ✅ {index_file}")
            if os.path.exists(progress_file):
                print(f"  ✅ {progress_file}")

            print("\n🔄 次のステップ:")
            print(f"python tools/workflow/workflow_cli.py step {tracker_id}")

        except Exception as e:
            logger.error(f"完了詳細表示エラー: {e}")

    def _show_failure_details(self, tracker_id: str):
        """失敗詳細情報表示"""
        try:
            config = self.workspace_config.get_workspace_config(tracker_id)
            if not config:
                return

            log_dir = os.path.join(config['workspace_path'], "logs")
            log_file = os.path.join(log_dir, "extraction.log")

            print("🔍 トラブルシューティング:")
            print(f"  📄 ログファイル: {log_file}")

            if os.path.exists(log_file):
                # ログファイルの最後の10行を表示
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("  📋 最新のログ (最後の10行):")
                        for line in lines[-10:]:
                            print(f"    {line.strip()}")

            print("\n🔧 推奨対応:")
            print(f"  1. python tools/workflow/workflow_cli.py subagent-retry {tracker_id}")
            print("  2. 入力ディレクトリのファイル形式確認")
            print("  3. ディスク容量・メモリ使用量確認")

        except Exception as e:
            logger.error(f"失敗詳細表示エラー: {e}")


def main():
    """テスト実行"""
    if len(sys.argv) < 3:
        print("使用方法: python subagent_command_handler.py <command> <tracker_id>")
        print("コマンド: extraction, status, retry, terminate, wait")
        return

    command = sys.argv[1]
    tracker_id = sys.argv[2]

    handler = SubAgentCommandHandler()

    if command == "extraction":
        handler.handle_subagent_extraction(tracker_id)
    elif command == "status":
        handler.handle_subagent_status(tracker_id)
    elif command == "retry":
        handler.handle_subagent_retry(tracker_id)
    elif command == "terminate":
        force = len(sys.argv) > 3 and sys.argv[3] == "--force"
        handler.handle_subagent_terminate(tracker_id, force)
    elif command == "wait":
        timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        handler.handle_subagent_wait(tracker_id, timeout)
    else:
        print(f"未知のコマンド: {command}")


if __name__ == "__main__":
    main()