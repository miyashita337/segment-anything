#!/usr/bin/env python3
"""
PH2-004-RESOURCE: リソース管理最適化システム - 単体テスト

【テスト対象】
- ResourceMonitor クラスの基本機能
- リソース状態取得機能
- アラート検出機能
- 自動最適化機能
- レポート生成機能
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import sys

# プロジェクトルート追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.common.resource_monitor import ResourceMonitor, ResourceStatus

class TestResourceMonitor(unittest.TestCase):
    """ResourceMonitor クラスのテスト"""
    
    def setUp(self):
        """テスト前準備"""
        self.monitor = ResourceMonitor(
            monitoring_interval=0.1,  # 高速テスト用
            history_size=10
        )
    
    def tearDown(self):
        """テスト後クリーンアップ"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()

    def test_initialization(self):
        """初期化テスト"""
        self.assertFalse(self.monitor.is_monitoring)
        self.assertEqual(len(self.monitor.resource_history), 0)
        self.assertEqual(self.monitor.optimization_count, 0)
        self.assertEqual(self.monitor.alert_count, 0)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    @patch('psutil.process_iter')
    @patch('os.getloadavg')
    def test_get_current_status(self, mock_loadavg, mock_process_iter, 
                               mock_pids, mock_net, mock_disk, mock_memory, mock_cpu):
        """現在状況取得テスト"""
        # モックデータ設定
        mock_cpu.return_value = 50.0
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 60.0
        mock_memory_obj.available = 8 * 1024**3  # 8GB
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = MagicMock()
        mock_disk_obj.used = 500 * 1024**3  # 500GB
        mock_disk_obj.total = 1000 * 1024**3  # 1TB
        mock_disk_obj.free = 500 * 1024**3  # 500GB
        mock_disk.return_value = mock_disk_obj
        
        mock_net_obj = MagicMock()
        mock_net_obj.bytes_sent = 1000000
        mock_net_obj.bytes_recv = 2000000
        mock_net.return_value = mock_net_obj
        
        mock_pids.return_value = [1, 2, 3, 4, 5]
        
        mock_python_process = MagicMock()
        mock_python_process.info = {'name': 'python3'}
        mock_process_iter.return_value = [mock_python_process, mock_python_process]
        
        mock_loadavg.return_value = (1.0, 1.5, 2.0)
        
        # GPU監視を無効化（テスト環境）
        self.monitor.gpu_available = False
        
        # テスト実行
        status = self.monitor.get_current_status()
        
        # 検証
        self.assertEqual(status.cpu_percent, 50.0)
        self.assertEqual(status.memory_percent, 60.0)
        self.assertEqual(status.memory_available_gb, 8.0)
        self.assertEqual(status.disk_percent, 50.0)
        self.assertEqual(status.disk_free_gb, 500.0)
        self.assertEqual(status.network_bytes_sent, 1000000)
        self.assertEqual(status.network_bytes_recv, 2000000)
        self.assertEqual(status.active_processes, 5)
        self.assertEqual(status.python_processes, 2)
        self.assertEqual(status.load_average, [1.0, 1.5, 2.0])
        self.assertEqual(status.gpu_count, 0)

    def test_check_alerts_cpu_high(self):
        """CPU高使用率アラートテスト"""
        # 高CPU使用率の状況を作成
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=95.0,  # 閾値90%を超過
            memory_percent=50.0,
            memory_available_gb=4.0,
            disk_percent=50.0,
            disk_free_gb=100.0,
            gpu_count=0,
            gpu_memory_used=[],
            gpu_memory_total=[],
            gpu_temperature=[],
            gpu_utilization=[],
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_processes=50,
            python_processes=3,
            load_average=[1.0, 1.0, 1.0]
        )
        
        alerts = self.monitor.check_alerts(status)
        
        # CPU高使用率アラートが発生することを確認
        cpu_alerts = [a for a in alerts if a['type'] == 'cpu_high']
        self.assertEqual(len(cpu_alerts), 1)
        self.assertEqual(cpu_alerts[0]['severity'], 'warning')
        self.assertEqual(cpu_alerts[0]['value'], 95.0)

    def test_check_alerts_memory_low(self):
        """メモリ不足アラートテスト"""
        # メモリ不足の状況を作成
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=50.0,
            memory_percent=90.0,  # 高使用率
            memory_available_gb=1.0,  # 閾値2.0GBを下回る
            disk_percent=50.0,
            disk_free_gb=100.0,
            gpu_count=0,
            gpu_memory_used=[],
            gpu_memory_total=[],
            gpu_temperature=[],
            gpu_utilization=[],
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_processes=50,
            python_processes=3,
            load_average=[1.0, 1.0, 1.0]
        )
        
        alerts = self.monitor.check_alerts(status)
        
        # メモリ関連アラートが発生することを確認
        memory_high_alerts = [a for a in alerts if a['type'] == 'memory_high']
        memory_low_alerts = [a for a in alerts if a['type'] == 'memory_low']
        
        self.assertEqual(len(memory_high_alerts), 1)
        self.assertEqual(len(memory_low_alerts), 1)
        self.assertEqual(memory_low_alerts[0]['severity'], 'critical')

    def test_check_alerts_gpu(self):
        """GPU関連アラートテスト"""
        # GPU高温度・高メモリ使用率の状況を作成
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=50.0,
            memory_percent=50.0,
            memory_available_gb=4.0,
            disk_percent=50.0,
            disk_free_gb=100.0,
            gpu_count=2,
            gpu_memory_used=[7.0, 6.0],    # 高メモリ使用
            gpu_memory_total=[8.0, 8.0],   # 8GB GPU
            gpu_temperature=[90.0, 80.0],  # GPU0が高温度（閾値85°C超過）
            gpu_utilization=[95.0, 70.0],
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_processes=50,
            python_processes=3,
            load_average=[1.0, 1.0, 1.0]
        )
        
        alerts = self.monitor.check_alerts(status)
        
        # GPU関連アラートが発生することを確認
        gpu_memory_alerts = [a for a in alerts if a['type'] == 'gpu_memory_high']
        gpu_temp_alerts = [a for a in alerts if a['type'] == 'gpu_temperature_high']
        
        self.assertEqual(len(gpu_memory_alerts), 1)  # GPU0のメモリ使用率87.5%
        self.assertEqual(len(gpu_temp_alerts), 1)    # GPU0の高温度
        self.assertEqual(gpu_temp_alerts[0]['severity'], 'critical')
        self.assertEqual(gpu_temp_alerts[0]['gpu_id'], 0)

    @patch('gc.collect')
    def test_optimize_memory(self, mock_gc):
        """メモリ最適化テスト"""
        mock_gc.return_value = 50  # 50オブジェクト解放
        
        actions = self.monitor._optimize_memory()
        
        self.assertTrue(any('Python GC実行' in action for action in actions))
        mock_gc.assert_called_once()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_optimize_gpu_memory(self, mock_empty_cache, mock_cuda_available):
        """GPU メモリ最適化テスト"""
        mock_cuda_available.return_value = True
        
        actions = self.monitor._optimize_gpu_memory()
        
        self.assertTrue(any('PyTorch GPU キャッシュクリア' in action for action in actions))
        mock_empty_cache.assert_called_once()

    @patch('psutil.process_iter')
    def test_optimize_processes(self, mock_process_iter):
        """プロセス最適化テスト"""
        # 高CPU使用プロセスのモック
        mock_high_cpu_process = MagicMock()
        mock_high_cpu_process.info = {'pid': 123, 'name': 'python3', 'cmdline': ['python3'], 'cpu_percent': 80.0}
        
        mock_low_cpu_process = MagicMock()
        mock_low_cpu_process.info = {'pid': 124, 'name': 'python3', 'cmdline': ['python3'], 'cpu_percent': 0.1}
        
        mock_process_iter.return_value = [mock_high_cpu_process] + [mock_low_cpu_process] * 6
        
        actions = self.monitor._optimize_processes()
        
        self.assertTrue(any('高CPU使用プロセス特定' in action for action in actions))
        self.assertTrue(any('アイドルプロセス検出' in action for action in actions))

    def test_add_alert_callback(self):
        """アラートコールバック追加テスト"""
        callback_called = False
        test_alert = None
        
        def test_callback(alert):
            nonlocal callback_called, test_alert
            callback_called = True
            test_alert = alert
        
        self.monitor.add_alert_callback(test_callback)
        
        # アラート発火テスト
        alerts = [{'type': 'test', 'message': 'Test alert', 'severity': 'warning'}]
        self.monitor._trigger_alerts(alerts)
        
        self.assertTrue(callback_called)
        self.assertEqual(test_alert['type'], 'test')

    def test_monitoring_lifecycle(self):
        """監視ライフサイクルテスト"""
        # 監視開始
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        # 少し待って履歴データが蓄積されることを確認
        time.sleep(0.3)
        self.assertGreater(len(self.monitor.resource_history), 0)
        
        # 監視停止
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)

    def test_statistics_calculation(self):
        """統計計算テスト"""
        # 履歴データを手動で追加
        for i in range(5):
            status = ResourceStatus(
                timestamp=f"2025-08-15T12:0{i}:00",
                cpu_percent=50.0 + i * 10,
                memory_percent=60.0 + i * 5,
                memory_available_gb=4.0,
                disk_percent=50.0,
                disk_free_gb=100.0,
                gpu_count=0,
                gpu_memory_used=[],
                gpu_memory_total=[],
                gpu_temperature=[],
                gpu_utilization=[],
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_processes=50,
                python_processes=3,
                load_average=[1.0, 1.0, 1.0]
            )
            self.monitor.resource_history.append(status)
        
        stats = self.monitor.get_statistics()
        
        self.assertEqual(stats['data_points'], 5)
        self.assertEqual(stats['optimization_count'], 0)
        self.assertEqual(stats['alert_count'], 0)
        self.assertIn('averages', stats)
        self.assertIn('peaks', stats)

    def test_export_report(self):
        """レポート出力テスト"""
        # テスト用履歴データ追加
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available_gb=4.0,
            disk_percent=50.0,
            disk_free_gb=100.0,
            gpu_count=0,
            gpu_memory_used=[],
            gpu_memory_total=[],
            gpu_temperature=[],
            gpu_utilization=[],
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_processes=50,
            python_processes=3,
            load_average=[1.0, 1.0, 1.0]
        )
        self.monitor.resource_history.append(status)
        
        # 一時ファイルにレポート出力
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            output_path = self.monitor.export_report(tmp_path)
            
            # ファイルが作成されることを確認
            self.assertTrue(output_path.exists())
            
            # JSON形式で読み込み可能なことを確認
            with open(output_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            self.assertIn('generated_at', report_data)
            self.assertIn('statistics', report_data)
            self.assertIn('system_info', report_data)
            
        finally:
            # クリーンアップ
            if tmp_path.exists():
                tmp_path.unlink()

    def test_custom_thresholds(self):
        """カスタム閾値テスト"""
        custom_thresholds = {
            'cpu_percent': 70.0,  # デフォルトより厳しい閾値
            'memory_percent': 70.0,
            'gpu_temperature': 70.0
        }
        
        custom_monitor = ResourceMonitor(alert_thresholds=custom_thresholds)
        
        # カスタム閾値が設定されることを確認
        self.assertEqual(custom_monitor.alert_thresholds['cpu_percent'], 70.0)
        self.assertEqual(custom_monitor.alert_thresholds['memory_percent'], 70.0)
        self.assertEqual(custom_monitor.alert_thresholds['gpu_temperature'], 70.0)
        
        # デフォルト値が保持されることを確認
        self.assertEqual(custom_monitor.alert_thresholds['disk_percent'], 90.0)

    def test_no_alerts_normal_status(self):
        """正常状態でアラートが発生しないことのテスト"""
        # 正常な状況を作成
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=30.0,  # 正常範囲
            memory_percent=50.0,  # 正常範囲
            memory_available_gb=8.0,  # 十分な空き
            disk_percent=60.0,  # 正常範囲
            disk_free_gb=200.0,
            gpu_count=1,
            gpu_memory_used=[4.0],  # 50%使用
            gpu_memory_total=[8.0],
            gpu_temperature=[70.0],  # 正常温度
            gpu_utilization=[60.0],
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_processes=50,
            python_processes=3,
            load_average=[1.0, 1.0, 1.0]
        )
        
        alerts = self.monitor.check_alerts(status)
        
        # アラートが発生しないことを確認
        self.assertEqual(len(alerts), 0)


class TestResourceStatus(unittest.TestCase):
    """ResourceStatus データクラスのテスト"""
    
    def test_resource_status_creation(self):
        """ResourceStatus作成テスト"""
        status = ResourceStatus(
            timestamp="2025-08-15T12:00:00",
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available_gb=4.0,
            disk_percent=70.0,
            disk_free_gb=100.0,
            gpu_count=1,
            gpu_memory_used=[4.0],
            gpu_memory_total=[8.0],
            gpu_temperature=[75.0],
            gpu_utilization=[80.0],
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            active_processes=100,
            python_processes=5,
            load_average=[1.5, 2.0, 2.5]
        )
        
        self.assertEqual(status.cpu_percent, 50.0)
        self.assertEqual(status.memory_percent, 60.0)
        self.assertEqual(status.gpu_count, 1)
        self.assertEqual(len(status.gpu_memory_used), 1)
        self.assertEqual(status.gpu_memory_used[0], 4.0)


if __name__ == '__main__':
    unittest.main()