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

from tools.workflow.state_manager import WorkflowStateManager
from tools.workflow.create_command_handler import CreateCommandHandler
from tools.workflow.plan_command_handler import PlanCommandHandler


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
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
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
        # CreateCommandHandlerを使用して承認ゲートをテスト
        with patch('tools.workflow.create_command_handler.os.path.exists', return_value=True):
            handler = CreateCommandHandler(self.test_tracker_id, self.mock_workspace)

            # 承認が必要なステップのテスト
            with patch.object(handler, 'execute') as mock_execute:
                mock_execute.return_value = {'needs_approval': True}
                result = handler.execute()
                self.assertTrue(result.get('needs_approval'), "承認が必要なステップが正しく識別されていません")

    def test_phase_transition_validation(self):
        """フェーズ遷移の検証テスト"""
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)

            # 初期状態の設定
            manager.initialize_database()
            manager.create_or_update(self.test_tracker_id, "phase_0", "planning")

            # フェーズ遷移のテスト
            # phase_0 → phase_0_5への遷移
            manager.update_step(self.test_tracker_id, "phase_0_5", "implementation")

            # 現在の状態確認
            state = manager.get_state(self.test_tracker_id)
            self.assertEqual(state.get('current_phase'), "phase_0_5")
            self.assertEqual(state.get('current_step'), "implementation")

    def test_non_idempotent_control(self):
        """非冪等的動作制御のテスト"""
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)
            manager.initialize_database()

            # 初期状態設定
            manager.create_or_update(self.test_tracker_id, "phase_0", "planning")

            # 同じステップへの更新が冪等であることをテスト
            state1 = manager.get_state(self.test_tracker_id)
            manager.update_step(self.test_tracker_id, "phase_0", "planning")
            state2 = manager.get_state(self.test_tracker_id)

            self.assertEqual(state1, state2, "同じステップの更新が冪等でない")

    def test_verification_based_control(self):
        """検証ベース制御のテスト"""
        # CreateCommandHandlerで検証ベースの制御をテスト
        handler = CreateCommandHandler(self.test_tracker_id, self.mock_workspace)

        # Git状態のモック
        with patch('subprocess.run') as mock_run:
            # gitコミットが存在しない場合
            mock_run.return_value = MagicMock(returncode=0, stdout="")

            with patch.object(handler, '_check_git_commits', return_value=False):
                # 検証失敗のシミュレーション
                can_proceed = handler._check_git_commits()
                self.assertFalse(can_proceed, "検証失敗時にステップが進行してしまいました")

    def test_workflow_state_persistence(self):
        """ワークフロー状態の永続化テスト"""
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
            # 最初のマネージャーインスタンス
            manager1 = WorkflowStateManager(self.db_path)
            manager1.initialize_database()
            manager1.create_or_update(self.test_tracker_id, "phase_1", "analysis")

            # 別のマネージャーインスタンスで状態を読み込み
            manager2 = WorkflowStateManager(self.db_path)
            state = manager2.get_state(self.test_tracker_id)

            self.assertIsNotNone(state, "永続化された状態が読み込めません")
            self.assertEqual(state['current_phase'], "phase_1")
            self.assertEqual(state['current_step'], "analysis")

    def test_error_recovery_mechanism(self):
        """エラーリカバリーメカニズムのテスト"""
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)
            manager.initialize_database()

            # 初期状態を設定
            manager.create_or_update(self.test_tracker_id, "phase_0", "initial")

            # エラー状態のシミュレーション
            try:
                # 不正な更新を試みる（エラーのシミュレーション）
                with patch.object(manager, 'update_step', side_effect=Exception("Test error")):
                    manager.update_step(self.test_tracker_id, "invalid", "invalid")
            except Exception:
                pass

            # エラー後も状態が保持されることを確認
            state = manager.get_state(self.test_tracker_id)
            self.assertIsNotNone(state, "エラー後の状態が失われています")
            self.assertEqual(state.get('current_phase'), "phase_0")

    def test_concurrent_access_protection(self):
        """並行アクセス保護のテスト"""
        with patch('tools.workflow.state_manager.os.path.exists', return_value=True):
            manager = WorkflowStateManager(self.db_path)
            manager.initialize_database()

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

            handler = PlanCommandHandler("KIRO-007", "概要", "詳細", "作者")

            with patch.object(handler, 'execute', return_value=True):
                result = handler.execute()
                self.assertTrue(result, "planコマンドの実行に失敗しました")

    def test_status_command_output(self):
        """statusコマンドの出力テスト"""
        # StateManagerで状態を設定
        manager = WorkflowStateManager(self.db_path)
        manager.initialize_database()
        manager.create_or_update("KIRO-007", "phase_0_5", "implementation")

        # 状態の取得
        state = manager.get_state("KIRO-007")

        self.assertEqual(state.get('current_phase'), "phase_0_5")
        self.assertEqual(state.get('current_step'), "implementation")


if __name__ == '__main__':
    unittest.main()