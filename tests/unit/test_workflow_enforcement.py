"""
KIRO-007: 強制ワークフロー実行システムのユニットテスト

このテストモジュールは、SQLiteベースの状態管理、承認ゲートシステム、
フェーズ遷移の正確な動作を検証します。
"""

import unittest
import tempfile
import sqlite3
import os
from unittest.mock import patch, MagicMock, mock_open
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.workflow.lib.workflow_controller import WorkflowController
from tools.workflow.lib.workflow_state_manager import WorkflowStateManager
from tools.workflow.lib.approval_gate_controller import ApprovalGateController


class TestWorkflowEnforcement(unittest.TestCase):
    """強制ワークフロー実行システムのテストスイート"""

    def setUp(self):
        """テスト環境のセットアップ"""
        self.test_tracker_id = "TEST-001"
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # モックの作成
        self.mock_workspace = "/tmp/test_workspace"

    def tearDown(self):
        """テスト環境のクリーンアップ"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_sqlite_state_initialization(self):
        """SQLite状態管理の初期化テスト"""
        with patch('tools.workflow.lib.workflow_state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)

            # データベーステーブルの存在確認
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='workflow_states'
            """)
            result = cursor.fetchone()
            conn.close()

            self.assertIsNotNone(result, "workflow_statesテーブルが作成されていません")

    def test_approval_gate_blocking(self):
        """承認ゲートシステムのブロッキングテスト"""
        with patch('tools.workflow.lib.approval_gate_controller.os.path.exists', return_value=True):
            controller = ApprovalGateController(self.mock_workspace)

            # 承認が必要なステップの定義
            step_with_approval = {
                'needs_approval': True,
                'title': 'Implementation Work',
                'description': 'Requires human approval'
            }

            # 承認ゲートのチェック
            result = controller.check_approval_required(step_with_approval)
            self.assertTrue(result, "承認が必要なステップが正しく識別されていません")

    def test_phase_transition_validation(self):
        """フェーズ遷移の検証テスト"""
        with patch('tools.workflow.lib.workflow_state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)

            # 初期状態の設定
            manager.create_tracker(self.test_tracker_id, "phase_0", "planning")

            # フェーズ遷移のテスト
            # phase_0 → phase_0_5への遷移
            success = manager.transition_phase(self.test_tracker_id, "phase_0_5", "implementation")
            self.assertTrue(success, "正当なフェーズ遷移が失敗しました")

            # 現在の状態確認
            state = manager.get_state(self.test_tracker_id)
            self.assertEqual(state['current_phase'], "phase_0_5")
            self.assertEqual(state['current_step'], "implementation")

    def test_non_idempotent_control(self):
        """非冪等的動作制御のテスト"""
        with patch('tools.workflow.lib.workflow_controller.os.path.exists', return_value=True):
            controller = WorkflowController(self.db_path, self.mock_workspace)

            # 同じステップの複数実行防止テスト
            controller.execute_step(self.test_tracker_id, "planning")

            # 2回目の実行は防止されるべき
            with patch('builtins.print') as mock_print:
                result = controller.execute_step(self.test_tracker_id, "planning")
                mock_print.assert_called_with("⚠️ このステップは既に実行されています")

    def test_verification_based_control(self):
        """検証ベース制御のテスト"""
        with patch('tools.workflow.lib.workflow_controller.os.path.exists', return_value=True):
            controller = WorkflowController(self.db_path, self.mock_workspace)

            # 検証条件の定義
            verification_criteria = {
                'git_commits_exist': True,
                'files_modified': True,
                'tests_pass': False
            }

            # 検証失敗時のステップ進行防止
            with patch.object(controller, 'verify_step_completion', return_value=False):
                can_proceed = controller.can_proceed_to_next_step(
                    self.test_tracker_id,
                    verification_criteria
                )
                self.assertFalse(can_proceed, "検証失敗時にステップが進行してしまいました")

    def test_workflow_state_persistence(self):
        """ワークフロー状態の永続化テスト"""
        with patch('tools.workflow.lib.workflow_state_manager.os.path.exists', return_value=True):
            # 最初のマネージャーインスタンス
            manager1 = WorkflowStateManager(self.db_path)
            manager1.create_tracker(self.test_tracker_id, "phase_1", "analysis")

            # 別のマネージャーインスタンスで状態を読み込み
            manager2 = WorkflowStateManager(self.db_path)
            state = manager2.get_state(self.test_tracker_id)

            self.assertIsNotNone(state, "永続化された状態が読み込めません")
            self.assertEqual(state['current_phase'], "phase_1")
            self.assertEqual(state['current_step'], "analysis")

    def test_error_recovery_mechanism(self):
        """エラーリカバリーメカニズムのテスト"""
        with patch('tools.workflow.lib.workflow_controller.os.path.exists', return_value=True):
            controller = WorkflowController(self.db_path, self.mock_workspace)

            # エラー状態からの復旧テスト
            with patch.object(controller, 'execute_step', side_effect=Exception("Test error")):
                try:
                    controller.execute_step(self.test_tracker_id, "error_step")
                except Exception:
                    pass

            # エラー後も状態が保持されることを確認
            state = controller.get_current_state(self.test_tracker_id)
            self.assertIsNotNone(state, "エラー後の状態が失われています")

    def test_concurrent_access_protection(self):
        """並行アクセス保護のテスト"""
        with patch('tools.workflow.lib.workflow_state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)

            # SQLiteのトランザクション分離レベルのテスト
            conn1 = sqlite3.connect(self.db_path)
            conn2 = sqlite3.connect(self.db_path)

            try:
                # トランザクション開始
                conn1.execute("BEGIN EXCLUSIVE")

                # 別の接続からの書き込み試行（ブロックされるべき）
                with self.assertRaises(sqlite3.OperationalError):
                    conn2.execute("BEGIN EXCLUSIVE")

            finally:
                conn1.close()
                conn2.close()


class TestWorkflowCommandIntegration(unittest.TestCase):
    """ワークフローCLIコマンドの統合テスト"""

    def test_plan_command_execution(self):
        """planコマンドの実行テスト"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Success")

            from tools.workflow.workflow_cli import plan_tracker
            result = plan_tracker("KIRO-007", "概要", "詳細", "作者")

            self.assertTrue(result, "planコマンドの実行に失敗しました")

    def test_status_command_output(self):
        """statusコマンドの出力テスト"""
        expected_output = """
📋 KIRO-007 のワークフロー状態
   現在のフェーズ: phase_0_5
   現在のステップ: implementation
   進行可能: ✅
        """

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=expected_output
            )

            from tools.workflow.workflow_cli import get_status
            result = get_status("KIRO-007")

            self.assertIn("phase_0_5", result)
            self.assertIn("implementation", result)


if __name__ == '__main__':
    unittest.main()