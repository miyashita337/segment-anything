#!/usr/bin/env python3
"""
SubAgent統合ワークフローテスト
QUAL-044: TaskOrchestratorとワークフロー統合のテスト
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.scripts.run_workflow_with_subagent import IntegratedWorkflowRunner
from tools.queue.task_integration import TaskOrchestrator
from tools.queue.subagent_monitor import SubAgentMonitor, SubAgentIntegration


class TestSubAgentWorkflowIntegration(unittest.TestCase):
    """SubAgent統合ワークフローテストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # テスト用トラッカーID
        self.tracker_id = "TEST-QUAL-044"
        
        # テスト用入力ディレクトリ作成
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # ダミー画像ファイル作成
        for i in range(3):
            dummy_image = self.input_dir / f"test_{i:04d}.jpg"
            dummy_image.write_text("dummy image data")
    
    def tearDown(self):
        """テストクリーンアップ"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_integrated_workflow_runner_initialization(self):
        """IntegratedWorkflowRunner初期化テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            
            self.assertEqual(runner.tracker_id, self.tracker_id)
            self.assertIsNotNone(runner.orchestrator)
            self.assertIsNotNone(runner.subagent)
            self.assertEqual(runner.workspace_base, str(self.workspace))
    
    def test_phase1_extraction_validation(self):
        """Phase 1: 抽出パイプライン検証テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            
            # 存在しないディレクトリのテスト
            non_existent_dir = "/path/that/does/not/exist"
            result = runner.run_phase1_extraction(non_existent_dir)
            
            self.assertEqual(result['status'], 'error')
            self.assertIn('存在しません', result['error'])
            
            # 画像がないディレクトリのテスト
            empty_dir = Path(self.temp_dir) / "empty"
            empty_dir.mkdir(parents=True, exist_ok=True)
            result = runner.run_phase1_extraction(str(empty_dir))
            
            self.assertEqual(result['status'], 'error')
            self.assertIn('画像ファイルが見つかりません', result['error'])
    
    @patch('tools.scripts.run_workflow_with_subagent.TaskOrchestrator')
    def test_phase1_extraction_success(self, mock_orchestrator_class):
        """Phase 1: 抽出成功シミュレーション"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # 成功結果シミュレーション
            mock_orchestrator.run_extraction_with_monitoring.return_value = (
                "task_001",
                {
                    'final_status': 'completed',
                    'duration': 10.5,
                    'images_processed': 3,
                    'success_rate': 100.0
                }
            )
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            result = runner.run_phase1_extraction(str(self.input_dir), max_files=3)
            
            self.assertEqual(result['final_status'], 'completed')
            self.assertEqual(result['images_processed'], 3)
            
            # 結果ファイル確認
            result_file = Path(self.workspace) / "phase1_result.json"
            self.assertTrue(result_file.exists())
    
    @patch('tools.scripts.run_workflow_with_subagent.TaskOrchestrator')
    def test_phase2_quality_check(self, mock_orchestrator_class):
        """Phase 2: 品質評価テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # 抽出結果ディレクトリ作成
            extraction_dir = Path(self.workspace) / "extraction"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_orchestrator.run_quality_check_with_monitoring.return_value = (
                "task_002",
                {
                    'final_status': 'completed',
                    'duration': 5.2,
                    'average_quality_score': 0.85,
                    'quality_distribution': {'A': 2, 'B': 1}
                }
            )
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            result = runner.run_phase2_quality_check()
            
            self.assertEqual(result['final_status'], 'completed')
            self.assertEqual(result['average_quality_score'], 0.85)
    
    @patch('tools.scripts.run_workflow_with_subagent.TaskOrchestrator')
    def test_phase3_dashboard_generation(self, mock_orchestrator_class):
        """Phase 3: ダッシュボード生成テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_orchestrator.run_dashboard_generation.return_value = (
                "task_003",
                {
                    'final_status': 'completed',
                    'duration': 3.1,
                    'dashboard_url': f'http://localhost:8088/{self.tracker_id}',
                    'formats_generated': ['html', 'markdown']
                }
            )
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            result = runner.run_phase3_dashboard()
            
            self.assertEqual(result['final_status'], 'completed')
            self.assertIn('dashboard_url', result)
    
    @patch('tools.scripts.run_workflow_with_subagent.TaskOrchestrator')
    def test_phase4_integration_report(self, mock_orchestrator_class):
        """Phase 4: 統合レポート生成テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # 各フェーズの結果ファイル作成
            for phase in ['phase1', 'phase2', 'phase3']:
                result_file = Path(self.workspace) / f"{phase}_result.json"
                result_data = {
                    'final_status': 'completed',
                    'duration': 10.0
                }
                with open(result_file, 'w') as f:
                    json.dump(result_data, f)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.generate_final_report.return_value = "# Integration Report\n..."
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            result = runner.run_phase4_integration_report()
            
            self.assertEqual(result['status'], 'completed')
            self.assertIn('report_path', result)
            
            # レポートファイル確認
            report_file = Path(self.workspace) / "integration_report.md"
            self.assertTrue(report_file.exists())
    
    @patch('tools.scripts.run_workflow_with_subagent.TaskOrchestrator')
    def test_full_workflow_execution(self, mock_orchestrator_class):
        """完全ワークフロー実行テスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # 各フェーズの成功結果設定
            mock_orchestrator.run_extraction_with_monitoring.return_value = (
                "task_001",
                {'final_status': 'completed', 'duration': 10.0}
            )
            mock_orchestrator.run_quality_check_with_monitoring.return_value = (
                "task_002",
                {'final_status': 'completed', 'duration': 5.0}
            )
            mock_orchestrator.run_dashboard_generation.return_value = (
                "task_003",
                {'final_status': 'completed', 'duration': 3.0}
            )
            mock_orchestrator.generate_final_report.return_value = "# Report"
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            result = runner.run_full_workflow(str(self.input_dir), max_files=3)
            
            self.assertEqual(result['status'], 'completed')
            self.assertIn('phases', result)
            self.assertEqual(len(result['phases']), 4)
            self.assertIn('duration', result)
    
    def test_task_orchestrator_extensions(self):
        """TaskOrchestrator拡張メソッドテスト"""
        with patch('tools.queue.task_integration.TaskIntegration') as mock_integration:
            orchestrator = TaskOrchestrator(self.tracker_id)
            
            # 拡張メソッドの存在確認
            self.assertTrue(hasattr(orchestrator, 'run_quality_check_with_monitoring'))
            self.assertTrue(hasattr(orchestrator, 'run_dashboard_generation'))
            self.assertTrue(hasattr(orchestrator, 'generate_final_report'))
    
    def test_subagent_context_inheritance(self):
        """SubAgentコンテキスト継承テスト"""
        integration = SubAgentIntegration()
        
        context = {
            'tracker_id': self.tracker_id,
            'workflow': 'test_workflow',
            'session_id': 'test_session_001'
        }
        
        integration.set_context(context)
        
        self.assertEqual(integration.current_context, context)
        self.assertEqual(integration.current_context['tracker_id'], self.tracker_id)
    
    def test_error_handling_and_recovery(self):
        """エラーハンドリングとリカバリーテスト"""
        with patch('tools.scripts.run_workflow_with_subagent.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            runner = IntegratedWorkflowRunner(self.tracker_id)
            
            # Phase 1でエラーが発生した場合
            with patch.object(runner, 'run_phase1_extraction') as mock_phase1:
                mock_phase1.return_value = {
                    'final_status': 'failed',
                    'error': 'Extraction failed'
                }
                
                result = runner.run_full_workflow(str(self.input_dir))
                
                self.assertEqual(result['status'], 'partial')
                self.assertIn('phase1', result['phases'])
                self.assertNotIn('phase2', result['phases'])  # Phase 2はスキップされる



class TestAsyncStageManager(unittest.TestCase):
    """AsyncStageManager多段階実行テストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / "test_workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # テスト用トラッカーID
        self.tracker_id = "TEST-ASYNC-001"
        
        # テスト用入力ディレクトリ作成
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # ダミー画像ファイル作成
        for i in range(3):
            dummy_image = self.input_dir / f"test_{i:04d}.jpg"
            dummy_image.write_text("dummy image data")
    
    def tearDown(self):
        """テストクリーンアップ"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_async_stage_manager_initialization(self):
        """AsyncStageManager初期化テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            self.assertEqual(manager.tracker_id, self.tracker_id)
            self.assertEqual(str(manager.workspace), str(self.workspace))
            self.assertTrue(manager.stage_file.name == "async_stage_status.json")
    
    def test_stage_status_persistence(self):
        """段階状態永続化テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            # テストデータ保存
            test_data = {
                'stage': 'register',
                'task_id': 'test_task_001',
                'status': 'registered'
            }
            
            manager.save_stage_status(test_data)
            
            # データ読み込み確認
            loaded_data = manager.load_stage_status()
            
            self.assertEqual(loaded_data['stage'], 'register')
            self.assertEqual(loaded_data['task_id'], 'test_task_001')
            self.assertEqual(loaded_data['status'], 'registered')
            self.assertIn('last_updated', loaded_data)
    
    @patch('tools.queue.async_stage_manager.TaskOrchestrator')
    def test_stage_register_success(self, mock_orchestrator_class):
        """段階1: 登録成功テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_orchestrator.register_async_task.return_value = {
                'task_id': 'async_task_001',
                'status': 'registered',
                'background_running': True
            }
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            result = manager.stage_register(
                input_dir=str(self.input_dir),
                task_type='extraction'
            )
            
            self.assertEqual(result['stage'], 'register')
            self.assertEqual(result['status'], 'registered')
            self.assertEqual(result['task_id'], 'async_task_001')
            self.assertTrue(result['background_running'])
    
    @patch('tools.queue.async_stage_manager.TaskOrchestrator')
    def test_stage_monitor_success(self, mock_orchestrator_class):
        """段階2: 監視成功テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_orchestrator.check_task_status.return_value = {
                'status': 'running',
                'message': 'Task is running',
                'current_task': 'async_task_001'
            }
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            # 事前に登録状態を保存
            manager.save_stage_status({
                'stage': 'register',
                'task_id': 'async_task_001',
                'status': 'registered'
            })
            
            result = manager.stage_monitor()
            
            self.assertEqual(result['stage'], 'monitor')
            self.assertEqual(result['status'], 'running')
            self.assertEqual(result['current_task'], 'async_task_001')
    
    @patch('tools.queue.async_stage_manager.TaskOrchestrator')
    def test_stage_collect_success(self, mock_orchestrator_class):
        """段階3: 収集成功テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            # モックオーケストレーター設定
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_orchestrator.collect_task_results.return_value = {
                'status': 'completed',
                'report_path': '/path/to/report.md',
                'result_files': 5,
                'extraction_dir': '/path/to/extraction'
            }
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            # 事前に完了状態を保存
            manager.save_stage_status({
                'stage': 'monitor',
                'task_id': 'async_task_001',
                'status': 'completed'
            })
            
            result = manager.stage_collect()
            
            self.assertEqual(result['stage'], 'collect')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['result_files'], 5)
    
    def test_next_action_recommendation(self):
        """次アクション推奨テスト"""
        with patch('tools.queue.async_stage_manager.WorkspaceConfig') as mock_config:
            mock_config.return_value.get_tracker_workspace.return_value = str(self.workspace)
            
            from tools.queue.async_stage_manager import AsyncStageManager
            manager = AsyncStageManager(self.tracker_id)
            
            # 各段階での推奨アクションテスト
            test_cases = [
                ({}, "register: 最初にタスクを登録してください"),
                ({'stage': 'register', 'status': 'registered'}, "monitor: タスクの状態を監視してください"),
                ({'stage': 'monitor', 'status': 'running'}, "monitor: まだ実行中です。再度監視してください"),
                ({'stage': 'monitor', 'status': 'completed'}, "collect: タスクが完了しました。結果を収集してください"),
                ({'stage': 'collect', 'status': 'completed'}, "complete: 全ての段階が完了しました"),
                ({'stage': 'monitor', 'status': 'failed'}, "review: タスクが失敗しました。エラー内容を確認してください"),
            ]
            
            for stage_status, expected_action in test_cases:
                with self.subTest(stage_status=stage_status):
                    action = manager._recommend_next_action(stage_status)
                    self.assertEqual(action, expected_action)


class TestTaskOrchestratorAsyncExtensions(unittest.TestCase):
    """TaskOrchestrator非同期拡張テストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        self.tracker_id = "TEST-ORCHESTRATOR-ASYNC"
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # ダミー画像ファイル作成
        dummy_image = self.input_dir / "test_001.jpg"
        dummy_image.write_text("dummy image data")
    
    def tearDown(self):
        """テストクリーンアップ"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('tools.queue.task_integration.TaskIntegration')
    def test_register_async_task_extraction(self, mock_integration):
        """非同期タスク登録（抽出）テスト"""
        from tools.queue.task_integration import TaskOrchestrator
        
        # モック設定
        mock_task_integration = MagicMock()
        mock_integration.return_value = mock_task_integration
        mock_task_integration.execute_extract_character.return_value = "async_extract_001"
        mock_task_integration.workspace = Path(self.temp_dir)
        mock_task_integration.queue.start_background_processing = MagicMock()
        
        orchestrator = TaskOrchestrator(self.tracker_id)
        
        result = orchestrator.register_async_task(
            input_dir=str(self.input_dir),
            task_type="extraction",
            max_files=5
        )
        
        self.assertEqual(result['stage'], 'register')
        self.assertEqual(result['task_id'], 'async_extract_001')
        self.assertEqual(result['status'], 'registered')
        self.assertTrue(result['background_running'])
    
    @patch('tools.queue.task_integration.TaskIntegration')
    def test_register_async_task_pytest(self, mock_integration):
        """非同期タスク登録（pytest）テスト"""
        from tools.queue.task_integration import TaskOrchestrator
        
        # モック設定
        mock_task_integration = MagicMock()
        mock_integration.return_value = mock_task_integration
        mock_task_integration.execute_pytest.return_value = "async_pytest_001"
        mock_task_integration.workspace = Path(self.temp_dir)
        mock_task_integration.queue.start_background_processing = MagicMock()
        
        orchestrator = TaskOrchestrator(self.tracker_id)
        
        result = orchestrator.register_async_task(
            input_dir="tests/",
            task_type="pytest",
            coverage=True
        )
        
        self.assertEqual(result['stage'], 'register')
        self.assertEqual(result['task_id'], 'async_pytest_001')
        self.assertEqual(result['status'], 'registered')
    
    @patch('tools.queue.task_integration.SubAgentMonitor')
    @patch('tools.queue.task_integration.TaskIntegration')
    def test_check_task_status(self, mock_integration, mock_monitor):
        """非同期タスク状態確認テスト"""
        from tools.queue.task_integration import TaskOrchestrator
        
        # モック設定
        mock_task_integration = MagicMock()
        mock_integration.return_value = mock_task_integration
        mock_task_integration.workspace = Path(self.temp_dir)
        mock_task_integration.queue.get_queue_status.return_value = {'pending_tasks': []}
        
        # 状態ファイル作成
        status_file = Path(self.temp_dir) / "queue" / "queue_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump({
                'status': 'task_running',
                'task_id': 'async_test_001'
            }, f)
        
        orchestrator = TaskOrchestrator(self.tracker_id)
        
        result = orchestrator.check_task_status()
        
        self.assertEqual(result['stage'], 'monitor')
        self.assertEqual(result['status'], 'running')
        self.assertIn('current_task', result)

class TestWorkflowConfiguration(unittest.TestCase):
    """ワークフロー設定テスト"""
    
    def test_subagent_workflow_yaml_loading(self):
        """subagent_workflow.yaml読み込みテスト"""
        import yaml
        
        config_path = Path(__file__).parent.parent.parent / "config" / "subagent_workflow.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 必須セクションの確認
            self.assertIn('workflow', config)
            self.assertIn('orchestrator', config)
            self.assertIn('phases', config)
            self.assertIn('queue', config)
            
            # フェーズ設定の確認
            phases = config['phases']
            self.assertIn('extraction', phases)
            self.assertIn('quality_check', phases)
            self.assertIn('dashboard', phases)
            self.assertIn('integration_report', phases)
    
    def test_bash_script_subagent_option(self):
        """run_quality_workflow.shのSubAgentオプションテスト"""
        script_path = Path(__file__).parent.parent.parent / "tools" / "scripts" / "run_quality_workflow.sh"
        
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
            
            # SubAgentモード関連の確認
            self.assertIn('USE_SUBAGENT', content)
            self.assertIn('--use-subagent', content)
            self.assertIn('run_workflow_with_subagent.py', content)


if __name__ == '__main__':
    unittest.main()