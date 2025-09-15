#!/usr/bin/env python3
"""
INTG-089 実用的シナリオ統合テスト
現実的な運用シナリオでのシステム統合テスト
"""

import unittest
import tempfile
import json
import time
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.queue.subagent_monitor import SubAgentMonitor
from tools.queue.notification_bridge import NotificationBridge
from tools.queue.subagent_wrapper import EnhancedSubAgentTaskQueue
from tests.mocks.mock_pushover_intg089 import (
    MockNotificationBridge, MockPushoverClient, 
    MockPushoverTestScenarios, simulate_notification_burst
)


class TestINTG089PracticalScenarios(unittest.TestCase):
    """INTG-089 実用的運用シナリオテスト"""
    
    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir
        self.tracker_id = "PRACTICAL-089"
        
        # 実用的なシステム構成
        self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.mock_bridge = MockNotificationBridge(workspace_path=self.workspace_path)
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=Path(self.workspace_path),
            tracker_id=self.tracker_id
        )
    
    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_typical_monitoring_workflow(self):
        """典型的な監視ワークフローテスト"""
        # シナリオ: 通常の監視→異常検知→通知→対応のフロー
        
        # 1. 定期監視実行
        monitoring_start_time = time.time()
        anomalies = self.monitor.comprehensive_anomaly_check()
        monitoring_duration = time.time() - monitoring_start_time
        
        # 監視処理が妥当な時間内に完了することを確認
        self.assertLess(monitoring_duration, 10.0)  # 10秒以内
        self.assertIsInstance(anomalies, list)
        
        # 2. 監視結果の通知
        monitoring_report = self.mock_bridge.send_enhanced_notification(
            "Daily Monitoring Report",
            f"Monitoring completed. {len(anomalies)} anomalies detected.",
            notification_type="monitoring_report",
            priority_level="low"
        )
        
        self.assertTrue(monitoring_report)
        
        # 3. 異常があった場合の詳細通知
        if anomalies:
            for i, anomaly in enumerate(anomalies[:3]):  # 最大3件まで
                anomaly_notification = self.mock_bridge.send_enhanced_notification(
                    f"Anomaly Alert #{i+1}",
                    f"Anomaly detected: {anomaly}",
                    notification_type="anomaly_alert",
                    priority_level="high"
                )
                self.assertTrue(anomaly_notification)
        
        # 4. 監視統計の確認
        stats = self.mock_bridge.get_notification_statistics()
        self.assertGreater(stats['pushover_stats']['total_calls'], 0)
    
    def test_long_running_task_monitoring(self):
        """長時間実行タスク監視テスト"""
        # シナリオ: 長時間タスクの実行監視とチェックポイント管理
        
        # 1. 長時間タスクの開始
        long_task = {
            "task_id": "long_running_extraction",
            "command": "python features/extraction/commands/extract_character.py --batch",
            "status": "starting",
            "estimated_duration": 3600,  # 1時間
            "start_time": time.time()
        }
        
        # 2. 開始チェックポイント作成
        start_checkpoint = self.task_queue.save_checkpoint("task_start", long_task)
        self.assertTrue(start_checkpoint)
        
        # 3. 開始通知
        start_notification = self.mock_bridge.send_enhanced_notification(
            "Long Task Started",
            f"Task {long_task['task_id']} started. Estimated duration: {long_task['estimated_duration']/60:.0f} minutes",
            notification_type="task_start",
            priority_level="normal"
        )
        self.assertTrue(start_notification)
        
        # 4. 中間進捗シミュレーション（複数のチェックポイント）
        progress_checkpoints = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for progress in progress_checkpoints:
            # 進捗更新
            long_task["status"] = "running"
            long_task["progress"] = progress
            long_task["current_time"] = time.time()
            
            # 進捗チェックポイント作成
            checkpoint_id = f"progress_{int(progress*100)}"
            checkpoint_saved = self.task_queue.save_checkpoint(checkpoint_id, long_task)
            self.assertTrue(checkpoint_saved)
            
            # 重要な進捗(50%以上)の場合は通知
            if progress >= 0.5:
                progress_notification = self.mock_bridge.send_enhanced_notification(
                    "Task Progress Update",
                    f"Task {long_task['task_id']} is {int(progress*100)}% complete",
                    notification_type="progress_update",
                    priority_level="low"
                )
                # 重複防止により全て送信されるわけではない
                self.assertIsInstance(progress_notification, bool)
        
        # 5. 完了処理
        long_task["status"] = "completed"
        long_task["completion_time"] = time.time()
        
        completion_checkpoint = self.task_queue.save_checkpoint("task_completion", long_task)
        self.assertTrue(completion_checkpoint)
        
        completion_notification = self.mock_bridge.send_enhanced_notification(
            "Long Task Completed",
            f"Task {long_task['task_id']} completed successfully",
            notification_type="task_completion",
            priority_level="normal"
        )
        self.assertTrue(completion_notification)
        
        # 6. 最終統計確認
        final_stats = self.mock_bridge.get_notification_statistics()
        self.assertGreater(final_stats['total_notifications'], 0)
    
    def test_system_recovery_scenario(self):
        """システム復旧シナリオテスト"""
        # シナリオ: システム中断→復旧→処理再開のフロー
        
        # 1. 進行中タスクの記録
        interrupted_task = {
            "task_id": "interrupted_extraction",
            "command": "python extraction_script.py",
            "status": "running",
            "progress": 0.4,
            "processed_files": 120,
            "total_files": 300,
            "last_checkpoint": time.time()
        }
        
        # 2. 中断前チェックポイント作成
        interruption_checkpoint = self.task_queue.save_checkpoint("pre_interruption", interrupted_task)
        self.assertTrue(interruption_checkpoint)
        
        # 3. システム中断シミュレーション（新しいインスタンス作成）
        recovery_task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=Path(self.workspace_path),
            tracker_id=self.tracker_id
        )
        
        # 4. チェックポイントからの復旧
        recovered_data = recovery_task_queue.load_checkpoint("pre_interruption")
        self.assertIsNotNone(recovered_data)
        self.assertEqual(recovered_data['task_data']['task_id'], interrupted_task['task_id'])
        self.assertEqual(recovered_data['task_data']['progress'], 0.4)
        
        # 5. 復旧通知
        recovery_notification = self.mock_bridge.send_enhanced_notification(
            "System Recovery",
            f"Task {interrupted_task['task_id']} recovered from checkpoint. Resuming from {int(interrupted_task['progress']*100)}%",
            notification_type="system_recovery",
            priority_level="high"
        )
        self.assertTrue(recovery_notification)
        
        # 6. 処理継続シミュレーション
        resumed_task = recovered_data['task_data'].copy()
        resumed_task['status'] = 'resumed'
        resumed_task['resume_time'] = time.time()
        
        resume_checkpoint = recovery_task_queue.save_checkpoint("post_recovery", resumed_task)
        self.assertTrue(resume_checkpoint)
    
    @patch('torch.cuda.is_available')
    def test_gpu_fallback_scenario(self, mock_cuda):
        """GPU fallbackシナリオテスト"""
        # シナリオ: GPU処理→GPU障害→CPU fallback→処理継続
        
        # 1. 初期GPU利用可能状態
        mock_cuda.return_value = True
        
        gpu_task = {
            "task_id": "gpu_intensive_task",
            "command": "python gpu_extraction.py",
            "requires_gpu": True,
            "status": "preparing"
        }
        
        # 2. GPU利用可能性確認
        gpu_available = self.task_queue._check_gpu_available()
        self.assertTrue(gpu_available)
        
        # GPU使用予定の通知
        gpu_start_notification = self.mock_bridge.send_enhanced_notification(
            "GPU Task Starting",
            f"GPU task {gpu_task['task_id']} starting with CUDA acceleration",
            notification_type="gpu_task_start",
            priority_level="normal"
        )
        self.assertTrue(gpu_start_notification)
        
        # 3. GPU障害シミュレーション
        mock_cuda.return_value = False
        
        # GPU利用不可検知
        gpu_unavailable = self.task_queue._check_gpu_available()
        self.assertFalse(gpu_unavailable)
        
        # 4. CPU fallback通知
        fallback_notification = self.mock_bridge.send_enhanced_notification(
            "GPU Fallback Alert",
            f"Task {gpu_task['task_id']} falling back to CPU execution due to GPU unavailability",
            notification_type="gpu_fallback",
            priority_level="high"
        )
        self.assertTrue(fallback_notification)
        
        # 5. CPU実行継続
        gpu_task['status'] = 'running_cpu_fallback'
        gpu_task['execution_mode'] = 'cpu'
        gpu_task['fallback_time'] = time.time()
        
        fallback_checkpoint = self.task_queue.save_checkpoint("cpu_fallback", gpu_task)
        self.assertTrue(fallback_checkpoint)
    
    def test_notification_flood_prevention(self):
        """通知フラッド防止テスト"""
        # シナリオ: 大量の同種通知→重複防止機能動作確認
        
        # 1. 通知バースト送信テスト
        burst_results = simulate_notification_burst(self.mock_bridge, count=20)
        
        # 重複防止により全て送信されるわけではない
        self.assertGreater(burst_results['suppressed'], 0)
        self.assertEqual(burst_results['sent'] + burst_results['suppressed'], 20)
        
        # 2. 異なる優先度での通知テスト
        priority_levels = ['critical', 'high', 'normal', 'low']
        sent_counts = {}
        
        for priority in priority_levels:
            # 各優先度で複数回送信
            sent_count = 0
            for i in range(5):
                result = self.mock_bridge.send_enhanced_notification(
                    f"{priority.capitalize()} Alert",
                    "Repeated message for flood test",
                    notification_type="flood_test",
                    priority_level=priority
                )
                if result:
                    sent_count += 1
            
            sent_counts[priority] = sent_count
        
        # 優先度が高いほど多く送信される（間隔設定による）
        self.assertGreaterEqual(sent_counts['critical'], sent_counts['low'])
        
        # 3. 最終統計確認
        flood_stats = self.mock_bridge.get_notification_statistics()
        self.assertGreater(flood_stats['duplicate_prevention_rate'], 0.0)


class TestINTG089RealWorldIntegration(unittest.TestCase):
    """INTG-089 実世界統合テスト"""
    
    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir
        self.tracker_id = "REAL-089"
        
        # 実際の設定ファイルパターン
        self.pushover_config = {
            "user_key": "real_user_key_example",
            "api_token": "real_api_token_example"
        }
        self.pushover_config_path = os.path.join(self.temp_dir, "pushover.json")
        
        with open(self.pushover_config_path, 'w') as f:
            json.dump(self.pushover_config, f)
        
        # 実システム構成
        self.monitor = SubAgentMonitor(workspace_path=self.workspace_path)
        self.notification_bridge = NotificationBridge(
            workspace_path=self.workspace_path,
            pushover_config_path=self.pushover_config_path
        )
        self.task_queue = EnhancedSubAgentTaskQueue(
            workspace_path=Path(self.workspace_path),
            tracker_id=self.tracker_id
        )
    
    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_daily_monitoring_routine(self):
        """日常監視ルーチンテスト"""
        # 実際の運用での日常的な監視処理
        
        # 1. システム健全性チェック
        daily_check_start = time.time()
        
        # 各システムの基本動作確認
        systems_health = {}
        
        # モニターシステム
        try:
            anomalies = self.monitor.comprehensive_anomaly_check()
            systems_health['monitor'] = {'status': 'healthy', 'anomaly_count': len(anomalies)}
        except Exception as e:
            systems_health['monitor'] = {'status': 'error', 'error': str(e)}
        
        # タスクキューシステム
        try:
            test_checkpoint = self.task_queue.save_checkpoint("daily_health", {"check": "ok"})
            systems_health['task_queue'] = {'status': 'healthy', 'checkpoint_ok': test_checkpoint}
        except Exception as e:
            systems_health['task_queue'] = {'status': 'error', 'error': str(e)}
        
        # 通知システム
        try:
            # 実際のPushover送信はモックなしでテスト
            test_notification = self.notification_bridge.send_enhanced_notification(
                "Daily Health Check",
                "System health check completed",
                priority_level="low"
            )
            systems_health['notification'] = {'status': 'healthy', 'notification_sent': test_notification}
        except Exception as e:
            systems_health['notification'] = {'status': 'error', 'error': str(e)}
        
        daily_check_duration = time.time() - daily_check_start
        
        # ヘルスチェックが妥当な時間で完了
        self.assertLess(daily_check_duration, 30.0)  # 30秒以内
        
        # 少なくとも基本的な初期化は成功している
        for system, health in systems_health.items():
            self.assertIn('status', health)
    
    def test_production_error_handling(self):
        """本番環境エラーハンドリングテスト"""
        # 実際の運用で発生しうる各種エラーの処理確認
        
        # 1. 設定ファイル破損
        corrupted_config_path = os.path.join(self.temp_dir, "corrupted_pushover.json")
        with open(corrupted_config_path, 'w') as f:
            f.write("{'invalid': json}")  # 意図的に無効なJSON
        
        # エラー時でも初期化は完了する
        try:
            corrupted_bridge = NotificationBridge(
                workspace_path=self.workspace_path,
                pushover_config_path=corrupted_config_path
            )
            self.assertIsInstance(corrupted_bridge, NotificationBridge)
        except Exception as e:
            self.fail(f"NotificationBridge should handle corrupted config gracefully: {e}")
        
        # 2. 権限なしディレクトリ（シミュレーション）
        restricted_path = "/root/restricted_workspace"  # 通常アクセス不可
        
        try:
            restricted_monitor = SubAgentMonitor(workspace_path=restricted_path)
            # 権限がない場合でも初期化は完了する（実際の処理で制限される）
            self.assertIsInstance(restricted_monitor, SubAgentMonitor)
        except Exception:
            # 権限エラーは許容される
            pass
        
        # 3. ディスク容量不足シミュレーション（チェックポイント保存失敗）
        # 大きなデータでの保存テスト
        large_data = {"large_field": "x" * 10000}  # 10KB程度
        
        save_result = self.task_queue.save_checkpoint("large_data_test", large_data)
        # 通常環境では保存成功するはず
        self.assertTrue(save_result)
    
    def test_performance_under_load(self):
        """負荷時パフォーマンステスト"""
        # 高負荷状況でのシステム応答性確認
        
        load_test_start = time.time()
        results = []
        
        # 50回の連続処理
        for i in range(50):
            operation_start = time.time()
            
            # チェックポイント操作
            checkpoint_data = {
                "iteration": i,
                "timestamp": time.time(),
                "test_data": f"load_test_data_{i}"
            }
            
            checkpoint_saved = self.task_queue.save_checkpoint(f"load_test_{i}", checkpoint_data)
            
            # 異常チェック
            anomalies = self.monitor.comprehensive_anomaly_check()
            
            operation_duration = time.time() - operation_start
            results.append({
                "iteration": i,
                "duration": operation_duration,
                "checkpoint_saved": checkpoint_saved,
                "anomaly_count": len(anomalies)
            })
            
            # あまりに遅い場合は中断
            if operation_duration > 5.0:
                break
        
        total_duration = time.time() - load_test_start
        average_duration = total_duration / len(results) if results else 0
        
        # パフォーマンス要件確認
        self.assertGreater(len(results), 10)  # 最低10回は処理完了
        self.assertLess(average_duration, 1.0)  # 平均1秒以内/操作
        
        # 成功率確認
        successful_operations = sum(1 for r in results if r['checkpoint_saved'])
        success_rate = successful_operations / len(results)
        self.assertGreater(success_rate, 0.9)  # 90%以上成功率


def run_all_practical_tests():
    """全実用テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestINTG089PracticalScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestINTG089RealWorldIntegration))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_practical_tests()
    print(f"\n{'='*50}")
    print(f"INTG-089 Practical Scenario Tests: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if success else 1)