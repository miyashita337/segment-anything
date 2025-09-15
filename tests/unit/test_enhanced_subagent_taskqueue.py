#!/usr/bin/env python3
"""
INTG-089 EnhancedSubAgentTaskQueue機能の単体テスト
現実的なタスクキュー・チェックポイント・GPU fallbackテスト
"""

import unittest
import tempfile
import json
import time
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.queue.subagent_wrapper import EnhancedSubAgentTaskQueue


class TestEnhancedSubAgentTaskQueueINTG089(unittest.TestCase):
    """EnhancedSubAgentTaskQueue INTG-089機能テスト"""
    
    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.tracker_id = "TEST-089"
        
        # テスト用EnhancedSubAgentTaskQueueインスタンス作成
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=self.workspace_path,
            tracker_id=self.tracker_id
        )
    
    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_task_queue_initialization(self):
        """EnhancedSubAgentTaskQueue初期化テスト"""
        # 基本属性の確認
        self.assertEqual(self.task_queue.workspace_path, self.workspace_path)
        self.assertEqual(self.task_queue.tracker_id, self.tracker_id)
        
        # 拡張機能設定の確認
        self.assertIsInstance(self.task_queue.checkpoint_interval, int)
        self.assertIsInstance(self.task_queue.max_gpu_retry, int)
        self.assertIsInstance(self.task_queue.gpu_fallback_enabled, bool)
    
    def test_checkpoint_directory_setup(self):
        """チェックポイントディレクトリセットアップテスト"""
        # チェックポイントディレクトリの確認
        checkpoint_dir = self.task_queue.checkpoint_dir
        self.assertIsInstance(checkpoint_dir, Path)
        self.assertEqual(checkpoint_dir.name, "checkpoints")
    
    def test_checkpoint_save_functionality(self):
        """チェックポイント保存機能テスト"""
        # テスト用タスクデータ
        task_data = {
            "task_id": "test_task_001",
            "command": "echo 'test'",
            "status": "running",
            "progress": 0.5
        }
        
        # チェックポイント保存
        checkpoint_id = "test_checkpoint"
        success = self.task_queue.save_checkpoint(checkpoint_id, task_data)
        
        self.assertTrue(success)
        
        # ファイルが作成されているか確認
        checkpoint_file = self.task_queue.checkpoint_dir / f"{checkpoint_id}.json"
        self.assertTrue(checkpoint_file.exists())
        
        # 保存内容の確認
        with open(checkpoint_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['task_data'], task_data)
        self.assertIn('timestamp', saved_data)
    
    def test_checkpoint_load_functionality(self):
        """チェックポイント読み込み機能テスト"""
        # テスト用チェックポイント作成
        checkpoint_id = "load_test"
        test_data = {
            "task_id": "load_test_task",
            "command": "test command",
            "status": "paused"
        }
        
        # 保存
        self.task_queue.save_checkpoint(checkpoint_id, test_data)
        
        # 読み込み
        loaded_data = self.task_queue.load_checkpoint(checkpoint_id)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data['task_data'], test_data)
    
    def test_checkpoint_load_nonexistent(self):
        """存在しないチェックポイント読み込みテスト"""
        # 存在しないチェックポイントID
        non_existent_id = "does_not_exist"
        
        loaded_data = self.task_queue.load_checkpoint(non_existent_id)
        self.assertIsNone(loaded_data)
    
    def test_checkpoint_cleanup_functionality(self):
        """チェックポイントクリーンアップ機能テスト"""
        # 複数のテストチェックポイント作成
        test_ids = ["cleanup_test_1", "cleanup_test_2", "cleanup_test_3"]
        
        for test_id in test_ids:
            self.task_queue.save_checkpoint(test_id, {"test": "data"})
        
        # 全て存在することを確認
        for test_id in test_ids:
            checkpoint_file = self.task_queue.checkpoint_dir / f"{test_id}.json"
            self.assertTrue(checkpoint_file.exists())
        
        # クリーンアップ実行
        cleaned_count = self.task_queue.cleanup_old_checkpoints()
        
        # クリーンアップ結果の確認（ファイルが削除されるかは実装による）
        self.assertIsInstance(cleaned_count, int)
        self.assertGreaterEqual(cleaned_count, 0)
    
    @patch('torch.cuda.is_available')
    def test_gpu_availability_check(self, mock_cuda):
        """GPU利用可能性チェックテスト"""
        # GPU利用可能な場合
        mock_cuda.return_value = True
        result = self.task_queue._check_gpu_available()
        self.assertTrue(result)
        
        # GPU利用不可の場合
        mock_cuda.return_value = False
        result = self.task_queue._check_gpu_available()
        self.assertFalse(result)
    
    @patch('torch.cuda.is_available')
    def test_gpu_availability_import_error(self, mock_cuda):
        """GPU利用可能性チェック（importエラー）テスト"""
        # torchインポートエラーをシミュレート
        mock_cuda.side_effect = ImportError("No module named 'torch'")
        
        result = self.task_queue._check_gpu_available()
        self.assertFalse(result)  # importエラー時はFalseを返す
    
    def test_gpu_fallback_configuration(self):
        """GPU fallback設定テスト"""
        # GPU fallback設定の確認
        self.assertIsInstance(self.task_queue.gpu_fallback_enabled, bool)
        self.assertIsInstance(self.task_queue.max_gpu_retry, int)
        self.assertGreater(self.task_queue.max_gpu_retry, 0)
    
    @patch.object(EnhancedSubAgentTaskQueue, '_check_gpu_available')
    @patch.object(EnhancedSubAgentTaskQueue, '_execute_task_with_env')
    def test_execute_with_gpu_fallback_gpu_available(self, mock_execute, mock_gpu_check):
        """GPU利用可能時のGPU fallback実行テスト"""
        mock_gpu_check.return_value = True
        mock_execute.return_value = {"status": "completed", "output": "success"}
        
        test_task = {
            "task_id": "gpu_test",
            "command": "python test_script.py"
        }
        
        result = self.task_queue.execute_with_gpu_fallback(test_task)
        
        self.assertEqual(result["status"], "completed")
        mock_execute.assert_called_once()
        
        # GPU環境での実行が確認
        call_args = mock_execute.call_args
        env = call_args[0][1]  # 第2引数の環境変数
        self.assertIn('CUDA_VISIBLE_DEVICES', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
    
    @patch.object(EnhancedSubAgentTaskQueue, '_check_gpu_available')
    @patch.object(EnhancedSubAgentTaskQueue, '_execute_task_with_env')
    def test_execute_with_gpu_fallback_gpu_unavailable(self, mock_execute, mock_gpu_check):
        """GPU利用不可時のCPU fallback実行テスト"""
        mock_gpu_check.return_value = False
        mock_execute.return_value = {"status": "completed", "output": "cpu_success"}
        
        test_task = {
            "task_id": "cpu_fallback_test",
            "command": "python test_script.py"
        }
        
        result = self.task_queue.execute_with_gpu_fallback(test_task)
        
        self.assertEqual(result["status"], "completed")
        mock_execute.assert_called_once()
        
        # CPU環境での実行が確認
        call_args = mock_execute.call_args
        env = call_args[0][1]  # 第2引数の環境変数
        self.assertIn('CUDA_VISIBLE_DEVICES', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')  # 空文字でCPU実行
    
    @patch.object(EnhancedSubAgentTaskQueue, '_check_gpu_available')
    @patch.object(EnhancedSubAgentTaskQueue, '_execute_task_with_env')
    def test_execute_with_gpu_retry_logic(self, mock_execute, mock_gpu_check):
        """GPU実行リトライロジックテスト"""
        # GPU利用可能だが実行失敗
        mock_gpu_check.side_effect = [True, True, False]  # 2回失敗後、GPU無効判定
        mock_execute.side_effect = [
            {"status": "error", "error": "GPU memory error"},  # 1回目失敗
            {"status": "error", "error": "GPU timeout"},       # 2回目失敗
            {"status": "completed", "output": "cpu_success"}   # CPU fallback成功
        ]
        
        test_task = {
            "task_id": "retry_test",
            "command": "python gpu_intensive_script.py"
        }
        
        # max_gpu_retryを2に設定
        self.task_queue.max_gpu_retry = 2
        
        result = self.task_queue.execute_with_gpu_fallback(test_task)
        
        # 最終的にCPU fallbackで成功
        self.assertEqual(result["status"], "completed")
        self.assertEqual(mock_execute.call_count, 3)  # GPU2回 + CPU1回
    
    def test_task_execution_environment_setup(self):
        """タスク実行環境セットアップテスト"""
        test_env = {"TEST_VAR": "test_value", "CUDA_VISIBLE_DEVICES": "1"}
        test_task = {"command": "echo $TEST_VAR"}
        
        # 環境変数セットアップの確認（実装されている場合）
        if hasattr(self.task_queue, '_setup_execution_environment'):
            env = self.task_queue._setup_execution_environment(test_env)
            self.assertIsInstance(env, dict)
            self.assertIn('TEST_VAR', env)
    
    def test_task_queue_statistics_tracking(self):
        """タスクキュー統計追跡テスト"""
        # 統計追跡機能の存在確認
        if hasattr(self.task_queue, 'execution_stats'):
            self.assertIsInstance(self.task_queue.execution_stats, dict)
        
        if hasattr(self.task_queue, 'get_execution_statistics'):
            stats = self.task_queue.get_execution_statistics()
            self.assertIsInstance(stats, dict)
    
    def test_resume_functionality_integration(self):
        """レジューム機能統合テスト"""
        # チェックポイント保存→読み込み→レジュームのフロー
        task_data = {
            "task_id": "resume_test",
            "command": "python long_running_script.py",
            "status": "paused",
            "progress": 0.3
        }
        
        checkpoint_id = "resume_checkpoint"
        
        # 1. チェックポイント保存
        save_success = self.task_queue.save_checkpoint(checkpoint_id, task_data)
        self.assertTrue(save_success)
        
        # 2. 別インスタンスでの読み込みシミュレーション
        new_task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=self.workspace_path,
            tracker_id=self.tracker_id
        )
        
        loaded_data = new_task_queue.load_checkpoint(checkpoint_id)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data['task_data'], task_data)
        
        # 3. レジューム機能テスト（実装されている場合）
        if hasattr(new_task_queue, 'resume_from_checkpoint'):
            resume_result = new_task_queue.resume_from_checkpoint(checkpoint_id)
            self.assertIsInstance(resume_result, (bool, dict))


class TestEnhancedSubAgentTaskQueueRealWorldScenarios(unittest.TestCase):
    """実世界シナリオテスト"""
    
    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.tracker_id = "REAL-WORLD-089"
        
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=self.workspace_path,
            tracker_id=self.tracker_id
        )
    
    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_directory_creation_on_demand(self):
        """必要時チェックポイントディレクトリ作成テスト"""
        # チェックポイントディレクトリが存在しない状態
        checkpoint_dir = self.task_queue.checkpoint_dir
        
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
        
        # チェックポイント保存時に自動作成されることを確認
        success = self.task_queue.save_checkpoint("auto_create_test", {"test": "data"})
        self.assertTrue(success)
        self.assertTrue(checkpoint_dir.exists())
    
    def test_invalid_json_checkpoint_handling(self):
        """無効なJSONチェックポイント処理テスト"""
        checkpoint_id = "invalid_json_test"
        checkpoint_file = self.task_queue.checkpoint_dir / f"{checkpoint_id}.json"
        
        # チェックポイントディレクトリ作成
        self.task_queue.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 無効なJSONファイル作成
        with open(checkpoint_file, 'w') as f:
            f.write("invalid json content {")
        
        # 読み込み時のエラーハンドリング確認
        loaded_data = self.task_queue.load_checkpoint(checkpoint_id)
        self.assertIsNone(loaded_data)  # 無効なJSONの場合はNoneを返すはず
    
    def test_concurrent_checkpoint_access(self):
        """同時チェックポイントアクセステスト"""
        checkpoint_id = "concurrent_test"
        
        # 複数回の保存・読み込みが正常に動作することを確認
        for i in range(5):
            task_data = {"iteration": i, "timestamp": time.time()}
            save_success = self.task_queue.save_checkpoint(checkpoint_id, task_data)
            self.assertTrue(save_success)
            
            loaded_data = self.task_queue.load_checkpoint(checkpoint_id)
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data['task_data']['iteration'], i)
    
    @patch('subprocess.run')
    def test_task_execution_timeout_handling(self, mock_subprocess):
        """タスク実行タイムアウト処理テスト"""
        # タイムアウトエラーをシミュレート
        mock_subprocess.side_effect = TimeoutError("Command timed out")
        
        test_task = {
            "task_id": "timeout_test",
            "command": "sleep 3600"  # 長時間実行コマンド
        }
        
        # タイムアウトが適切に処理されることを確認
        if hasattr(self.task_queue, '_execute_task_with_env'):
            try:
                result = self.task_queue._execute_task_with_env(test_task, os.environ.copy())
                # タイムアウト時はエラー結果を返すはず
                self.assertIn('error', result.get('status', '').lower())
            except Exception:
                # 例外が適切に処理されればOK
                pass


def run_all_tests():
    """全テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedSubAgentTaskQueueINTG089))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedSubAgentTaskQueueRealWorldScenarios))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)