"""
SubAgentCommandHandler ユニットテスト
SubAgent連携コマンドのテスト
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tests.workflow.fixtures.workflow_fixtures import WorkflowTestBase, CLITestHelper
from tests.workflow.fixtures.mock_data import MockData, SAMPLE_TRACKER_ID


class TestSubAgentCommandHandler(WorkflowTestBase):
    """SubAgentCommandHandlerのテストクラス"""

    def setUp(self):
        """テスト前設定"""
        super().setUp()

        # SubAgentCommandHandlerの依存関係のモック設定
        self.mock_state_manager = Mock()
        self.mock_subagent_monitor = Mock()
        self.mock_lock_manager = Mock()
        self.mock_workspace_config = self.create_mock_workspace_config()

        # モックの設定
        self.add_mock('tools.workflow.subagent_command_handler.get_state_manager',
                     return_value=self.mock_state_manager)
        self.add_mock('tools.workflow.subagent_command_handler.get_subagent_monitor',
                     return_value=self.mock_subagent_monitor)
        self.add_mock('tools.workflow.subagent_command_handler.get_lock_manager',
                     return_value=self.mock_lock_manager)
        self.add_mock('tools.workflow.subagent_command_handler.get_workspace_config',
                     return_value=self.mock_workspace_config)

    def test_handle_subagent_extraction_success(self):
        """subagent-extractionコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 二重起動チェックをパス
        self.mock_lock_manager.is_duplicate_execution_risk.return_value = False

        # ワークスペース設定が存在
        self.mock_workspace_config.get_workspace_config.return_value = MockData.MOCK_WORKSPACE_CONFIG[SAMPLE_TRACKER_ID]

        # SubAgent登録成功
        self.mock_subagent_monitor.register_subagent.return_value = True

        handler = SubAgentCommandHandler()
        result = handler.handle_subagent_extraction(SAMPLE_TRACKER_ID)

        self.assertTrue(result)

        # メソッドが呼ばれたことを確認
        self.mock_lock_manager.is_duplicate_execution_risk.assert_called_once_with(SAMPLE_TRACKER_ID, "extraction")
        self.mock_workspace_config.get_workspace_config.assert_called_once_with(SAMPLE_TRACKER_ID)
        self.mock_subagent_monitor.register_subagent.assert_called_once()

    def test_handle_subagent_extraction_duplicate_execution(self):
        """subagent-extractionコマンド - 二重実行防止テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 二重起動リスクあり
        self.mock_lock_manager.is_duplicate_execution_risk.return_value = True
        self.mock_lock_manager.get_lock_owner_info.return_value = {
            'pid': 12345,
            'created_at': '2025-09-27 10:00:00',
            'hostname': 'localhost'
        }

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_extraction, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("抽出処理が既に実行中です", stdout)
        self.assertIn("PID: 12345", stdout)

    def test_handle_subagent_extraction_workspace_not_found(self):
        """subagent-extractionコマンド - ワークスペース未発見テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 二重起動チェックをパス
        self.mock_lock_manager.is_duplicate_execution_risk.return_value = False

        # ワークスペース設定が存在しない
        self.mock_workspace_config.get_workspace_config.return_value = None

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_extraction, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("ワークスペース設定が見つかりません", stdout)

    def test_handle_subagent_extraction_custom_input_path(self):
        """subagent-extractionコマンド - カスタム入力パステスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 二重起動チェックをパス
        self.mock_lock_manager.is_duplicate_execution_risk.return_value = False

        # ワークスペース設定が存在
        self.mock_workspace_config.get_workspace_config.return_value = MockData.MOCK_WORKSPACE_CONFIG[SAMPLE_TRACKER_ID]

        # SubAgent登録成功
        self.mock_subagent_monitor.register_subagent.return_value = True

        custom_input_path = "/custom/input/path"
        max_files = 50

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(
            handler.handle_subagent_extraction, SAMPLE_TRACKER_ID, custom_input_path, max_files
        )

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("📁 カスタム入力パス", stdout)
        self.assertIn(custom_input_path, stdout)

    def test_handle_subagent_status_success(self):
        """subagent-statusコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # SubAgent状態を設定
        status_data = MockData.MOCK_SUBAGENT_STATUS[SAMPLE_TRACKER_ID]
        self.mock_subagent_monitor.get_subagent_status.return_value = status_data

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_status, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("📊 SubAgent状態", stdout)
        self.assertIn(SAMPLE_TRACKER_ID, stdout)
        self.assertIn("状態: running", stdout)
        self.assertIn("PID: 12345", stdout)

    def test_handle_subagent_status_not_found(self):
        """subagent-statusコマンド - 状態未発見テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # SubAgent状態が見つからない
        self.mock_subagent_monitor.get_subagent_status.return_value = None

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_status, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("ℹ️", stdout)
        self.assertIn("SubAgentは実行されていません", stdout)

    def test_handle_subagent_wait_success(self):
        """subagent-waitコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 完了状態を返すように設定
        self.mock_subagent_monitor.wait_for_completion.return_value = {
            'success': True,
            'status': 'completed',
            'message': '処理が正常に完了しました'
        }

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_wait, SAMPLE_TRACKER_ID, 30)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("⏳", stdout)
        self.assertIn("SubAgent完了待機", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("正常に完了しました", stdout)

    def test_handle_subagent_wait_timeout(self):
        """subagent-waitコマンド - タイムアウトテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # タイムアウト状態を返すように設定
        self.mock_subagent_monitor.wait_for_completion.return_value = {
            'success': False,
            'status': 'timeout',
            'message': 'タイムアウトしました'
        }

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_wait, SAMPLE_TRACKER_ID, 1)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("⏰", stdout)
        self.assertIn("タイムアウト", stdout)

    def test_handle_subagent_retry_success(self):
        """subagent-retryコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 再実行成功
        self.mock_subagent_monitor.retry_subagent.return_value = True

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_retry, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🔄", stdout)
        self.assertIn("SubAgent再実行", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("再実行を開始しました", stdout)

    def test_handle_subagent_retry_failure(self):
        """subagent-retryコマンド - 再実行失敗テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 再実行失敗
        self.mock_subagent_monitor.retry_subagent.return_value = False

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_retry, SAMPLE_TRACKER_ID)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("再実行に失敗しました", stdout)

    def test_handle_subagent_terminate_success(self):
        """subagent-terminateコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 終了成功
        self.mock_subagent_monitor.terminate_subagent.return_value = True

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_terminate, SAMPLE_TRACKER_ID, False)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🛑", stdout)
        self.assertIn("SubAgent終了", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("正常に終了しました", stdout)

        # terminateメソッドが呼ばれたことを確認
        self.mock_subagent_monitor.terminate_subagent.assert_called_once_with(SAMPLE_TRACKER_ID, force=False)

    def test_handle_subagent_terminate_force(self):
        """subagent-terminateコマンド - 強制終了テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 強制終了成功
        self.mock_subagent_monitor.terminate_subagent.return_value = True

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_terminate, SAMPLE_TRACKER_ID, True)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🛑", stdout)
        self.assertIn("強制終了", stdout)

        # 強制終了フラグが渡されたことを確認
        self.mock_subagent_monitor.terminate_subagent.assert_called_once_with(SAMPLE_TRACKER_ID, force=True)

    def test_handle_subagent_terminate_failure(self):
        """subagent-terminateコマンド - 終了失敗テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 終了失敗
        self.mock_subagent_monitor.terminate_subagent.return_value = False

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_terminate, SAMPLE_TRACKER_ID, False)

        self.assertFalse(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("終了に失敗しました", stdout)

    def test_handle_subagent_cleanup_success(self):
        """subagent-cleanupコマンド正常実行テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # クリーンアップ成功
        self.mock_lock_manager.force_cleanup_locks.return_value = True

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_cleanup, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🧹", stdout)
        self.assertIn("SubAgentロック強制クリーンアップ", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("クリーンアップが完了しました", stdout)

    def test_handle_subagent_locks_status(self):
        """subagent-locks-statusコマンドテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # ロック状況を設定
        lock_info = {
            'active_locks': [
                {
                    'tracker_id': 'TRACKER-001',
                    'process_type': 'extraction',
                    'pid': 12345,
                    'created_at': '2025-09-27 10:00:00'
                },
                {
                    'tracker_id': 'KIRO-006',
                    'process_type': 'quality_check',
                    'pid': 12346,
                    'created_at': '2025-09-27 11:00:00'
                }
            ],
            'total_count': 2
        }

        self.mock_lock_manager.get_all_locks_status.return_value = lock_info

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_locks_status)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🔒", stdout)
        self.assertIn("全SubAgentロック状況", stdout)
        self.assertIn("アクティブロック数: 2", stdout)
        self.assertIn("TRACKER-001", stdout)
        self.assertIn("KIRO-006", stdout)

    def test_handle_subagent_locks_status_empty(self):
        """subagent-locks-statusコマンド - ロックなしテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # ロックなし
        lock_info = {
            'active_locks': [],
            'total_count': 0
        }

        self.mock_lock_manager.get_all_locks_status.return_value = lock_info

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_locks_status)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("✅", stdout)
        self.assertIn("アクティブなロックはありません", stdout)

    def test_handle_subagent_cleanup_all(self):
        """subagent-cleanup-allコマンドテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 全クリーンアップ成功
        cleanup_result = {
            'success': True,
            'cleaned_count': 3,
            'details': ['TRACKER-001', 'KIRO-006', 'QUAL-044']
        }

        self.mock_lock_manager.cleanup_all_locks.return_value = cleanup_result

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_cleanup_all)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🧹", stdout)
        self.assertIn("全SubAgentロック強制クリーンアップ", stdout)
        self.assertIn("クリーンアップ数: 3", stdout)

    def test_handle_subagent_auto_retry_check(self):
        """subagent-auto-retry-checkコマンドテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 自動再実行条件チェック結果
        check_result = {
            'eligible': True,
            'reason': '前回の実行が失敗したため再実行可能',
            'last_failure_time': '2025-09-27 10:00:00',
            'retry_count': 1,
            'max_retries': 3
        }

        self.mock_subagent_monitor.check_auto_retry_eligibility.return_value = check_result

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_auto_retry_check, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🔍", stdout)
        self.assertIn("SubAgent自動再実行条件確認", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("自動再実行可能", stdout)

    def test_handle_subagent_auto_retry_check_not_eligible(self):
        """subagent-auto-retry-checkコマンド - 再実行不可テスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 自動再実行不可
        check_result = {
            'eligible': False,
            'reason': '最大再実行回数に達しています',
            'retry_count': 3,
            'max_retries': 3
        }

        self.mock_subagent_monitor.check_auto_retry_eligibility.return_value = check_result

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_auto_retry_check, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("❌", stdout)
        self.assertIn("自動再実行不可", stdout)
        self.assertIn("最大再実行回数に達しています", stdout)

    def test_handle_subagent_auto_retry(self):
        """subagent-auto-retryコマンドテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 自動再実行成功
        retry_result = {
            'success': True,
            'message': '自動再実行を開始しました',
            'new_process_id': 'SUB-001'
        }

        self.mock_subagent_monitor.execute_auto_retry.return_value = retry_result

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_auto_retry, SAMPLE_TRACKER_ID)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🤖", stdout)
        self.assertIn("SubAgent自動再実行", stdout)
        self.assertIn("✅", stdout)
        self.assertIn("自動再実行を開始しました", stdout)

    def test_handle_subagent_auto_retry_all(self):
        """subagent-auto-retry-allコマンドテスト"""
        from tools.workflow.subagent_command_handler import SubAgentCommandHandler

        # 全自動再実行成功
        batch_result = {
            'success': True,
            'processed_count': 5,
            'successful_retries': 3,
            'failed_retries': 2,
            'details': [
                {'tracker_id': 'TRACKER-001', 'status': 'success'},
                {'tracker_id': 'KIRO-006', 'status': 'success'},
                {'tracker_id': 'QUAL-044', 'status': 'success'},
                {'tracker_id': 'TEST-001', 'status': 'failed'},
                {'tracker_id': 'TEST-002', 'status': 'failed'}
            ]
        }

        self.mock_subagent_monitor.execute_auto_retry_all.return_value = batch_result

        handler = SubAgentCommandHandler()

        output = CLITestHelper.capture_output(handler.handle_subagent_auto_retry_all)

        self.assertTrue(output['result'])
        stdout = output['stdout']

        self.assertIn("🤖", stdout)
        self.assertIn("全SubAgent自動再実行バッチ", stdout)
        self.assertIn("処理数: 5", stdout)
        self.assertIn("成功: 3", stdout)
        self.assertIn("失敗: 2", stdout)


if __name__ == '__main__':
    unittest.main()