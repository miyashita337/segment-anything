"""
P1-B002: リアルタイムダッシュボードのテスト

MetricsCollector、DashboardServer、extraction_hooksの単体テスト
"""

import json
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from features.evaluation.realtime_dashboard.metrics_collector import (
    ExtractMetrics,
    AggregatedMetrics,
    MetricsCollector
)
from features.evaluation.realtime_dashboard.extraction_hooks import (
    initialize_realtime_dashboard,
    on_image_start,
    on_image_complete,
    get_metrics_collector,
    shutdown_dashboard
)


class TestExtractMetrics(unittest.TestCase):
    """ExtractMetricsデータクラスのテスト"""
    
    def test_extract_metrics_creation(self):
        """ExtractMetricsの作成テスト"""
        metrics = ExtractMetrics(
            timestamp=time.time(),
            image_name="test.jpg",
            status="processing"
        )
        
        self.assertEqual(metrics.image_name, "test.jpg")
        self.assertEqual(metrics.status, "processing")
        self.assertIsNone(metrics.processing_time)
        self.assertIsNone(metrics.quality_score)
    
    def test_extract_metrics_to_dict(self):
        """ExtractMetricsの辞書変換テスト"""
        metrics = ExtractMetrics(
            timestamp=1234567890.0,
            image_name="test.jpg",
            status="success",
            processing_time=5.5,
            quality_score=0.85
        )
        
        result = metrics.to_dict()
        expected = {
            'timestamp': 1234567890.0,
            'image_name': 'test.jpg',
            'status': 'success',
            'processing_time': 5.5,
            'quality_score': 0.85,
            'memory_usage': None,
            'error_message': None
        }
        
        self.assertEqual(result, expected)


class TestAggregatedMetrics(unittest.TestCase):
    """AggregatedMetricsデータクラスのテスト"""
    
    def test_aggregated_metrics_defaults(self):
        """AggregatedMetricsのデフォルト値テスト"""
        metrics = AggregatedMetrics()
        
        self.assertEqual(metrics.total_images, 0)
        self.assertEqual(metrics.processed_images, 0)
        self.assertEqual(metrics.success_count, 0)
        self.assertEqual(metrics.failed_count, 0)
        self.assertEqual(metrics.average_processing_time, 0.0)
        self.assertEqual(metrics.average_quality_score, 0.0)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.current_fps, 0.0)
        self.assertEqual(metrics.memory_stats, {})
    
    def test_aggregated_metrics_to_dict(self):
        """AggregatedMetricsの辞書変換テスト"""
        metrics = AggregatedMetrics(
            total_images=10,
            processed_images=8,
            success_count=6,
            failed_count=2,
            success_rate=0.75
        )
        
        result = metrics.to_dict()
        self.assertEqual(result['total_images'], 10)
        self.assertEqual(result['success_rate'], 0.75)


class TestMetricsCollector(unittest.TestCase):
    """MetricsCollectorクラスのテスト"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.collector = MetricsCollector(max_history=100)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(len(self.collector._metrics_history), 0)
        self.assertEqual(len(self.collector._current_metrics), 0)
        self.assertIsInstance(self.collector._start_time, float)
    
    def test_start_processing(self):
        """処理開始の記録テスト"""
        self.collector.start_processing("test1.jpg")
        
        status = self.collector.get_current_status()
        self.assertEqual(len(status['processing_images']), 1)
        self.assertIn("test1.jpg", status['processing_images'])
        self.assertEqual(status['processing_count'], 1)
    
    def test_complete_processing_success(self):
        """処理完了（成功）の記録テスト"""
        # 処理開始
        self.collector.start_processing("test1.jpg")
        time.sleep(0.1)  # 処理時間をシミュレート
        
        # 処理完了
        self.collector.complete_processing(
            image_name="test1.jpg",
            success=True,
            quality_score=0.85
        )
        
        # 現在の状況を確認
        status = self.collector.get_current_status()
        self.assertEqual(len(status['processing_images']), 0)
        
        # 履歴を確認
        self.assertEqual(len(self.collector._metrics_history), 1)
        history = list(self.collector._metrics_history)
        self.assertEqual(history[0].image_name, "test1.jpg")
        self.assertEqual(history[0].status, "success")
        self.assertEqual(history[0].quality_score, 0.85)
        self.assertIsNotNone(history[0].processing_time)
        self.assertGreater(history[0].processing_time, 0)
    
    def test_complete_processing_failure(self):
        """処理完了（失敗）の記録テスト"""
        # 処理開始
        self.collector.start_processing("test2.jpg")
        
        # 処理失敗
        self.collector.complete_processing(
            image_name="test2.jpg",
            success=False,
            error_message="Processing failed"
        )
        
        # 履歴を確認
        history = list(self.collector._metrics_history)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].status, "failed")
        self.assertEqual(history[0].error_message, "Processing failed")
    
    def test_aggregated_metrics_calculation(self):
        """集計メトリクスの計算テスト"""
        # 複数の処理を実行
        images = ["test1.jpg", "test2.jpg", "test3.jpg"]
        results = [
            (True, 0.8),
            (False, None),
            (True, 0.9)
        ]
        
        for i, (image_name, (success, quality)) in enumerate(zip(images, results)):
            self.collector.start_processing(image_name)
            time.sleep(0.01)  # わずかな処理時間
            self.collector.complete_processing(
                image_name=image_name,
                success=success,
                quality_score=quality
            )
        
        # 集計メトリクスを取得
        aggregated = self.collector.get_aggregated_metrics()
        
        self.assertEqual(aggregated.total_images, 3)
        self.assertEqual(aggregated.processed_images, 3)
        self.assertEqual(aggregated.success_count, 2)
        self.assertEqual(aggregated.failed_count, 1)
        self.assertAlmostEqual(aggregated.success_rate, 2/3, places=2)
        self.assertAlmostEqual(aggregated.average_quality_score, 0.85, places=1)
    
    def test_recent_history(self):
        """最近の履歴取得テスト"""
        # 5件の処理を実行
        for i in range(5):
            image_name = f"test{i}.jpg"
            self.collector.start_processing(image_name)
            self.collector.complete_processing(
                image_name=image_name,
                success=True,
                quality_score=0.8
            )
        
        # 最近3件を取得
        recent = self.collector.get_recent_history(count=3)
        self.assertEqual(len(recent), 3)
        
        # 最新のものから順に確認
        expected_names = ["test4.jpg", "test3.jpg", "test2.jpg"]
        for i, entry in enumerate(reversed(recent)):
            self.assertEqual(entry['image_name'], expected_names[i])
    
    def test_export_metrics(self, tmp_path=None):
        """メトリクスエクスポートテスト"""
        if tmp_path is None:
            tmp_path = Path("/tmp/test_metrics_export")
            tmp_path.mkdir(exist_ok=True)
        
        # テストデータを作成
        self.collector.start_processing("export_test.jpg")
        self.collector.complete_processing(
            image_name="export_test.jpg",
            success=True,
            quality_score=0.75
        )
        
        # エクスポート
        export_path = tmp_path / "metrics.json"
        self.collector.export_metrics(export_path)
        
        # ファイルが作成されたか確認
        self.assertTrue(export_path.exists())
        
        # JSONファイルの内容を確認
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        self.assertIn('start_time', exported_data)
        self.assertIn('export_time', exported_data)
        self.assertIn('aggregated_metrics', exported_data)
        self.assertIn('history', exported_data)
        
        self.assertEqual(len(exported_data['history']), 1)
        self.assertEqual(exported_data['history'][0]['image_name'], "export_test.jpg")


class TestExtractionHooks(unittest.TestCase):
    """extraction_hooksモジュールのテスト"""
    
    def setUp(self):
        """テスト前のクリーンアップ"""
        # グローバル変数をクリーンアップ
        from features.evaluation.realtime_dashboard import extraction_hooks
        extraction_hooks._metrics_collector = None
        extraction_hooks._dashboard_server = None
    
    def test_initialize_realtime_dashboard_without_server(self):
        """ダッシュボードなしでの初期化テスト"""
        collector = initialize_realtime_dashboard(enable_dashboard=False)
        
        self.assertIsNotNone(collector)
        self.assertIsInstance(collector, MetricsCollector)
    
    @patch('features.evaluation.realtime_dashboard.extraction_hooks.DashboardServer')
    def test_initialize_realtime_dashboard_with_server(self, mock_dashboard_server):
        """ダッシュボード付きでの初期化テスト"""
        # DashboardServerのモックを設定
        mock_server_instance = Mock()
        mock_dashboard_server.return_value = mock_server_instance
        
        collector = initialize_realtime_dashboard(enable_dashboard=True, port=8081)
        
        self.assertIsNotNone(collector)
        mock_dashboard_server.assert_called_once()
        mock_server_instance.run_in_thread.assert_called_once()
    
    def test_image_processing_hooks(self):
        """画像処理フックのテスト"""
        # 初期化
        collector = initialize_realtime_dashboard(enable_dashboard=False)
        
        # 処理開始フック
        on_image_start("hook_test.jpg")
        
        # 状況確認
        status = collector.get_current_status()
        self.assertIn("hook_test.jpg", status['processing_images'])
        
        # 処理完了フック
        on_image_complete(
            image_name="hook_test.jpg",
            success=True,
            quality_score=0.9
        )
        
        # 状況確認
        status = collector.get_current_status()
        self.assertNotIn("hook_test.jpg", status['processing_images'])
        
        # 履歴確認
        history = collector.get_recent_history(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['image_name'], "hook_test.jpg")
        self.assertEqual(history[0]['status'], "success")
        self.assertEqual(history[0]['quality_score'], 0.9)
    
    def test_get_metrics_collector(self):
        """メトリクス収集器取得テスト"""
        # 初期化前は None
        self.assertIsNone(get_metrics_collector())
        
        # 初期化後は CollectorInstance
        collector = initialize_realtime_dashboard(enable_dashboard=False)
        retrieved_collector = get_metrics_collector()
        
        self.assertIs(collector, retrieved_collector)
    
    def test_hooks_without_initialization(self):
        """初期化なしでのフック呼び出しテスト（エラーなく無視される）"""
        # 初期化せずにフックを呼び出し（エラーが出ないことを確認）
        try:
            on_image_start("no_init_test.jpg")
            on_image_complete("no_init_test.jpg", success=True)
            shutdown_dashboard()
        except Exception as e:
            self.fail(f"Hooks should not fail without initialization: {e}")


if __name__ == '__main__':
    unittest.main()