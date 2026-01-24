"""
ワークフローテスト用フィクスチャ
テストで共通使用するモック設定とヘルパー関数を提供
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .mock_data import MockData


class WorkflowTestBase(unittest.TestCase):
    """ワークフローテスト用ベースクラス"""

    def setUp(self):
        """テスト前の共通設定"""
        self.mock_data = MockData()
        self.patches = []
        self.mocks = {}

        # 共通モックの設定
        self.setup_common_mocks()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 全てのパッチを停止
        for patcher in self.patches:
            patcher.stop()
        self.patches.clear()
        self.mocks.clear()

    def setup_common_mocks(self):
        """共通で使用するモックの設定"""
        # 仮想環境チェックをスキップ
        self.add_mock('tools.workflow.workflow_cli.check_virtual_environment',
                     return_value=True)

        # ログ出力を無効化
        self.add_mock('logging.getLogger')

        # os.environ.get をモック（CI環境として扱う）
        self.add_mock('os.environ.get', side_effect=self._mock_environ_get)

    def add_mock(self, target: str, **kwargs) -> Mock:
        """モックを追加し、パッチリストに登録"""
        patcher = patch(target, **kwargs)
        mock_obj = patcher.start()
        self.patches.append(patcher)

        # target名から最後の部分を取得してmocksに保存
        mock_name = target.split('.')[-1]
        self.mocks[mock_name] = mock_obj

        return mock_obj

    def _mock_environ_get(self, key: str, default: Any = None) -> Any:
        """環境変数のモック関数"""
        # CI環境として扱う（仮想環境チェックをスキップ）
        if key == 'GITHUB_ACTIONS':
            return 'true'
        return default

    def create_mock_progress_manager(self) -> Mock:
        """ProgressManagerのモックを作成"""
        mock_manager = Mock()

        # get_task メソッドのモック（既存チェック＆作成後確認）
        def mock_get_task(tracker_id: str):
            task_data = self.mock_data.get_mock_task(tracker_id)
            if task_data:
                return task_data
            # 作成後確認のために、新しいタスクを作成して返す
            if not hasattr(mock_get_task, 'call_count'):
                mock_get_task.call_count = 0
            mock_get_task.call_count += 1

            # 2回目以降の呼び出し（作成後確認）では作成されたタスクを返す
            if mock_get_task.call_count > 1:
                from .mock_data import MockTask
                return MockTask(
                    tracker_id=tracker_id,
                    description="テスト用タスク",
                    status="planning",
                    created_date="2025-09-27 12:00:00",
                    updated_date="2025-09-27 12:00:00"
                )
            return None

        mock_manager.get_task.side_effect = mock_get_task

        # create_task メソッドのモック
        def mock_create_task(tracker_id: str, description: str):
            from .mock_data import MockTask
            return MockTask(
                tracker_id=tracker_id,
                description=description,
                status="planning",
                created_date="2025-09-27 12:00:00",
                updated_date="2025-09-27 12:00:00"
            )

        mock_manager.create_task.side_effect = mock_create_task

        # config属性のモック
        mock_config = Mock()
        mock_config.sheet_url = "https://docs.google.com/spreadsheets/d/test/edit"
        mock_manager.config = mock_config

        return mock_manager

    def create_mock_workflow_controller(self) -> Mock:
        """WorkflowControllerのモックを作成"""
        mock_controller = Mock()

        # get_workflow_status メソッドのモック
        def mock_get_workflow_status(tracker_id: str):
            return self.mock_data.get_mock_workflow_status(tracker_id)

        mock_controller.get_workflow_status.side_effect = mock_get_workflow_status

        # get_current_step_instructions メソッドのモック
        def mock_get_current_step_instructions(tracker_id: str):
            status = self.mock_data.get_mock_workflow_status(tracker_id)
            if "current_step_instructions" in status:
                instructions = Mock()
                inst_data = status["current_step_instructions"]
                instructions.step_id = status["current_step"]
                instructions.title = inst_data["title"]
                instructions.description = inst_data["description"]
                instructions.required_actions = inst_data["required_actions"]
                instructions.validation_criteria = inst_data["validation_criteria"]
                instructions.approval_required = inst_data["approval_required"]
                instructions.can_proceed = inst_data["can_proceed"]
                instructions.blocking_reasons = inst_data.get("blocking_reasons", [])
                return instructions
            return None

        mock_controller.get_current_step_instructions.side_effect = mock_get_current_step_instructions

        # attempt_step_completion メソッドのモック
        def mock_attempt_step_completion(tracker_id: str):
            result = Mock()
            status = self.mock_data.get_mock_workflow_status(tracker_id)

            if status.get("can_proceed", False):
                result.status = "completed"
                result.message = "ステップが正常に完了しました"
                result.next_step = "next_step_id"
                result.approval_id = None
            else:
                result.status = "pending_approval"
                result.message = "承認が必要です"
                result.approval_id = "APP-001"
                result.next_step = None

            return result

        mock_controller.attempt_step_completion.side_effect = mock_attempt_step_completion

        # approval_controller のモック
        mock_approval_controller = Mock()
        mock_approval_controller.list_pending_approvals.return_value = [
            {
                "approval_id": "APP-001",
                "tracker_id": "KIRO-006",
                "step_name": "implementation",
                "priority": "high",
                "requested_at": "2025-09-27 10:00:00",
                "time_remaining_hours": 23.5,
                "approval_criteria": ["設計レビュー完了", "技術仕様確認"]
            }
        ]
        mock_controller.approval_controller = mock_approval_controller

        # executor のモック
        mock_executor = Mock()
        mock_executor.check_process_status.return_value = {
            "status": "running",
            "step": "extraction",
            "started_at": "2025-09-27 10:30:00",
            "pid": 12345,
            "log_file": "/tmp/test.log"
        }
        mock_controller.executor = mock_executor

        return mock_controller

    def create_mock_workspace_config(self) -> Mock:
        """WorkspaceConfigのモックを作成"""
        mock_config = Mock()

        # get_workspace_config メソッドのモック
        def mock_get_workspace_config(tracker_id: str):
            return self.mock_data.get_mock_workspace_config(tracker_id)

        mock_config.get_workspace_config.side_effect = mock_get_workspace_config

        # set_workspace_config メソッドのモック
        mock_config.set_workspace_config.return_value = True

        # get_input_directory メソッドのモック
        def mock_get_input_directory(tracker_id: str):
            config = self.mock_data.get_mock_workspace_config(tracker_id)
            return config.get("input_path") if config else None

        mock_config.get_input_directory.side_effect = mock_get_input_directory

        return mock_config

    def create_mock_subagent_handler(self) -> Mock:
        """SubAgentCommandHandlerのモックを作成"""
        mock_handler = Mock()

        # 各メソッドが成功を返すように設定
        mock_handler.handle_subagent_extraction.return_value = True
        mock_handler.handle_subagent_status.return_value = True
        mock_handler.handle_subagent_wait.return_value = True
        mock_handler.handle_subagent_retry.return_value = True
        mock_handler.handle_subagent_terminate.return_value = True
        mock_handler.handle_subagent_cleanup.return_value = True
        mock_handler.handle_subagent_locks_status.return_value = True
        mock_handler.handle_subagent_cleanup_all.return_value = True
        mock_handler.handle_subagent_auto_retry_check.return_value = True
        mock_handler.handle_subagent_auto_retry.return_value = True
        mock_handler.handle_subagent_auto_retry_all.return_value = True

        return mock_handler

    def setup_plan_command_mocks(self):
        """planコマンド用のモック設定"""
        # PlanCommandHandlerクラスのモック
        mock_handler_class = self.add_mock(
            'tools.workflow.workflow_cli.PlanCommandHandler'
        )

        mock_handler = Mock()
        mock_handler.execute_plan_command.return_value = (True, "✅ 起票成功")
        mock_handler_class.return_value = mock_handler

        return mock_handler

    def setup_create_command_mocks(self):
        """createコマンド用のモック設定"""
        # CreateCommandHandlerクラスのモック
        mock_handler_class = self.add_mock(
            'tools.workflow.workflow_cli.CreateCommandHandler'
        )

        mock_handler = Mock()
        mock_handler.execute_create_command.return_value = (True, "✅ 作成成功")
        mock_handler_class.return_value = mock_handler

        return mock_handler

    def setup_workspace_validation_mocks(self, is_configured: bool = True):
        """ワークスペース検証用のモック設定"""
        mock_validation = {
            'is_configured': is_configured,
            'errors': [] if is_configured else ["❌ ワークスペース設定エラー"]
        }

        self.add_mock(
            'config.workspace_config.validate_tracker_setup',
            return_value=mock_validation
        )

        return mock_validation


class CLITestHelper:
    """CLI テスト用ヘルパークラス"""

    @staticmethod
    def capture_output(func, *args, **kwargs):
        """関数の出力をキャプチャ"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = func(*args, **kwargs)

            return {
                'result': result,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            }
        except Exception as e:
            return {
                'result': None,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'exception': e
            }

    @staticmethod
    def create_test_args(command: str, *args):
        """テスト用引数オブジェクトを作成"""
        import argparse

        # ArgumentParserのように振舞うMockオブジェクト
        mock_args = Mock()
        mock_args.command = command

        # コマンドに応じて引数を設定
        if command == 'plan':
            mock_args.tracker_id = args[0] if len(args) > 0 else "TEST-001"
            mock_args.summary = args[1] if len(args) > 1 else "テスト概要"
            mock_args.details = args[2] if len(args) > 2 else "テスト詳細"
            mock_args.author_name = args[3] if len(args) > 3 else "yado"
            mock_args.priority = args[4] if len(args) > 4 else "medium"
        elif command in ['create', 'status', 'instructions', 'step', 'process', 'sheets']:
            mock_args.tracker_id = args[0] if len(args) > 0 else "TEST-001"
        elif command == 'template':
            mock_args.tracker_id = args[0] if len(args) > 0 else "TEST-001"
            mock_args.output = args[1] if len(args) > 1 else None
        elif command.startswith('subagent-'):
            mock_args.tracker_id = args[0] if len(args) > 0 else "TEST-001"

            # subagent-extraction 固有の引数
            if command == 'subagent-extraction':
                mock_args.input_path = args[1] if len(args) > 1 else None
                mock_args.max_files = args[2] if len(args) > 2 else None

            # subagent-wait 固有の引数
            elif command == 'subagent-wait':
                mock_args.timeout = args[1] if len(args) > 1 else 60

            # subagent-terminate 固有の引数
            elif command == 'subagent-terminate':
                mock_args.force = args[1] if len(args) > 1 else False

        return mock_args