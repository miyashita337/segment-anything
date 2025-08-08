#!/usr/bin/env python3
"""
PH2-006 監視システム統合テスト
リアルタイム監視・メトリクス収集・アラート・Webダッシュボードの動作確認
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests
import torch

# プロジェクトルート追加
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.evaluation.realtime_dashboard.monitoring_system import (
    PH2006MonitoringSystem,
    add_processing_metrics
)
from features.evaluation.realtime_dashboard.web_dashboard import create_web_dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PH2006MonitoringTester:
    """PH2-006 監視システムテスター"""
    
    def __init__(self):
        """初期化"""
        self.monitoring_system = PH2006MonitoringSystem(collection_interval=1.0)
        self.web_dashboard = create_web_dashboard(self.monitoring_system, port=5001)
        
        # 出力ディレクトリ
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-006")
        self.tests_dir = self.output_dir / "tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        
        logger.info("PH2-006 監視システムテスター初期化完了")
    
    def test_metrics_collection(self) -> Dict:
        """メトリクス収集テスト"""
        logger.info("🔧 メトリクス収集テスト開始")
        
        # 監視開始
        self.monitoring_system.start_monitoring()
        
        # 10秒間監視
        time.sleep(10)
        
        # メトリクス確認
        latest_metrics = self.monitoring_system.metrics_collector.get_latest_system_metrics()
        metrics_count = len(self.monitoring_system.metrics_collector.system_metrics_history)
        
        # 監視停止
        self.monitoring_system.stop_monitoring()
        
        test_result = {
            "test_type": "metrics_collection",
            "duration_seconds": 10,
            "metrics_collected": metrics_count,
            "collection_rate": metrics_count / 10,
            "latest_metrics_available": latest_metrics is not None,
            "system_info": {
                "cpu_percent": latest_metrics.cpu_percent if latest_metrics else None,
                "memory_percent": latest_metrics.memory_percent if latest_metrics else None,
                "gpu_available": latest_metrics.gpu_available if latest_metrics else False,
                "process_count": latest_metrics.process_count if latest_metrics else 0
            },
            "success": metrics_count > 5 and latest_metrics is not None
        }
        
        logger.info(f"メトリクス収集テスト完了: {metrics_count}件収集, 成功: {test_result['success']}")
        return test_result
    
    def test_alert_system(self) -> Dict:
        """アラートシステムテスト"""
        logger.info("🔧 アラートシステムテスト開始")
        
        # テスト用の低い閾値アラートルールを追加
        from features.evaluation.realtime_dashboard.monitoring_system import AlertRule
        
        test_rule = AlertRule(
            metric_name="cpu_percent",
            threshold=1.0,  # 非常に低い閾値（必ずアラートが発生）
            condition=">",
            severity="medium",
            message_template="テストアラート: CPU使用率 {current_value:.1f}%"
        )
        
        self.monitoring_system.alert_manager.add_alert_rule(test_rule)
        
        # 監視開始
        self.monitoring_system.start_monitoring()
        
        # 5秒待機（アラート発生待ち）
        time.sleep(5)
        
        # アラート確認
        active_alerts = self.monitoring_system.alert_manager.get_active_alerts()
        alert_history_count = len(self.monitoring_system.alert_manager.alert_history)
        
        # 監視停止
        self.monitoring_system.stop_monitoring()
        
        test_result = {
            "test_type": "alert_system",
            "test_rule_added": True,
            "active_alerts_count": len(active_alerts),
            "alert_history_count": alert_history_count,
            "alert_triggered": len(active_alerts) > 0 or alert_history_count > 0,
            "sample_alerts": [
                {
                    "message": alert.message,
                    "severity": alert.rule.severity,
                    "current_value": alert.current_value
                }
                for alert in active_alerts[:3]  # 最大3件
            ],
            "success": len(active_alerts) > 0 or alert_history_count > 0
        }
        
        logger.info(f"アラートシステムテスト完了: {len(active_alerts)}個アクティブ, 履歴{alert_history_count}件")
        return test_result
    
    def test_processing_metrics_integration(self) -> Dict:
        """処理メトリクス統合テスト"""
        logger.info("🔧 処理メトリクス統合テスト開始")
        
        # 監視開始
        self.monitoring_system.start_monitoring()
        
        # 模擬処理メトリクス追加
        test_metrics = [
            {"task_id": "test_001", "engine_type": "thread_pool", "duration": 0.5, "success": True, "throughput": 100.0},
            {"task_id": "test_002", "engine_type": "process_pool", "duration": 1.2, "success": True, "throughput": 83.3},
            {"task_id": "test_003", "engine_type": "async_io", "duration": 2.1, "success": False, "throughput": 0.0, "error_message": "Test error"},
            {"task_id": "test_004", "engine_type": "gpu_parallel", "duration": 0.8, "success": True, "throughput": 125.0}
        ]
        
        for metrics in test_metrics:
            add_processing_metrics(**metrics)
            time.sleep(0.5)
        
        # 少し待機
        time.sleep(2)
        
        # 処理メトリクス確認
        processing_metrics_count = len(self.monitoring_system.metrics_collector.processing_metrics_history)
        
        # 監視停止
        self.monitoring_system.stop_monitoring()
        
        test_result = {
            "test_type": "processing_metrics_integration",
            "test_metrics_added": len(test_metrics),
            "processing_metrics_collected": processing_metrics_count,
            "integration_success": processing_metrics_count >= len(test_metrics),
            "sample_metrics": test_metrics,
            "success": processing_metrics_count >= len(test_metrics)
        }
        
        logger.info(f"処理メトリクス統合テスト完了: {processing_metrics_count}件収集")
        return test_result
    
    def test_report_generation(self) -> Dict:
        """レポート生成テスト"""
        logger.info("🔧 レポート生成テスト開始")
        
        try:
            # 監視開始（データ蓄積のため）
            self.monitoring_system.start_monitoring()
            
            # データ蓄積のため少し待機
            time.sleep(3)
            
            # レポート生成
            report = self.monitoring_system.generate_report(duration_hours=1)
            
            # レポート保存
            report_path = self.monitoring_system.save_report(report, "test_monitoring_report.json")
            
            # 監視停止
            self.monitoring_system.stop_monitoring()
            
            # レポート内容確認
            report_has_system_stats = 'system_statistics' in report
            report_has_monitoring_health = 'monitoring_health' in report
            report_file_exists = Path(report_path).exists()
            
            test_result = {
                "test_type": "report_generation",
                "report_generated": True,
                "report_path": report_path,
                "report_file_exists": report_file_exists,
                "report_structure": {
                    "has_system_statistics": report_has_system_stats,
                    "has_monitoring_health": report_has_monitoring_health,
                    "has_report_period": 'report_period' in report
                },
                "report_summary": {
                    "period_hours": report.get('report_period', {}).get('duration_hours'),
                    "system_performance": report.get('monitoring_health', {}).get('system_performance')
                },
                "success": report_has_system_stats and report_has_monitoring_health and report_file_exists
            }
            
            logger.info(f"レポート生成テスト完了: {report_path}")
            return test_result
            
        except Exception as e:
            logger.error(f"レポート生成テストエラー: {e}")
            return {
                "test_type": "report_generation",
                "success": False,
                "error": str(e)
            }
    
    def test_web_dashboard_api(self) -> Dict:
        """WebダッシュボードAPIテスト"""
        logger.info("🔧 WebダッシュボードAPIテスト開始")
        
        try:
            # Webサーバー開始
            self.web_dashboard.start_server()
            time.sleep(2)  # サーバー起動待ち
            
            # 監視開始
            self.monitoring_system.start_monitoring()
            time.sleep(3)  # データ蓄積
            
            api_results = {}
            base_url = "http://localhost:5001"
            
            # 各APIエンドポイントテスト
            endpoints = [
                ("/api/status", "status"),
                ("/api/report", "report"),
                ("/api/alerts", "alerts"),
                ("/api/metrics/history", "metrics_history")
            ]
            
            for endpoint, name in endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    api_results[name] = {
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response_size": len(response.text),
                        "has_data": len(response.text) > 100,
                        "success": response.status_code == 200
                    }
                    
                    # データ内容簡易チェック
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            api_results[name]["data_structure"] = list(data.keys())[:5]
                        except:
                            pass
                            
                except Exception as e:
                    api_results[name] = {
                        "endpoint": endpoint,
                        "success": False,
                        "error": str(e)
                    }
            
            # 監視・サーバー停止
            self.monitoring_system.stop_monitoring()
            self.web_dashboard.stop_server()
            
            # 結果集計
            successful_apis = sum(1 for result in api_results.values() if result.get("success", False))
            total_apis = len(api_results)
            
            test_result = {
                "test_type": "web_dashboard_api",
                "base_url": base_url,
                "total_endpoints": total_apis,
                "successful_endpoints": successful_apis,
                "success_rate": successful_apis / total_apis if total_apis > 0 else 0,
                "api_results": api_results,
                "success": successful_apis >= total_apis * 0.75  # 75%以上成功で合格
            }
            
            logger.info(f"WebダッシュボードAPIテスト完了: {successful_apis}/{total_apis} 成功")
            return test_result
            
        except Exception as e:
            logger.error(f"WebダッシュボードAPIテストエラー: {e}")
            # クリーンアップ
            try:
                self.monitoring_system.stop_monitoring()
                self.web_dashboard.stop_server()
            except:
                pass
            
            return {
                "test_type": "web_dashboard_api",
                "success": False,
                "error": str(e)
            }
    
    def test_stress_monitoring(self) -> Dict:
        """ストレス監視テスト"""
        logger.info("🔧 ストレス監視テスト開始")
        
        try:
            # 監視開始
            self.monitoring_system.start_monitoring()
            
            # CPUストレス生成（数学計算）
            def cpu_stress():
                result = 0
                for i in range(1000000):
                    result += i ** 0.5
                return result
            
            # GPUストレス生成（利用可能な場合）
            def gpu_stress():
                if torch.cuda.is_available():
                    tensor = torch.randn(1000, 1000, device='cuda')
                    for _ in range(100):
                        tensor = torch.matmul(tensor, tensor.T)
                    return tensor.cpu()
                return None
            
            # ストレス処理前のメトリクス
            time.sleep(2)
            before_stress = self.monitoring_system.metrics_collector.get_latest_system_metrics()
            
            # ストレス処理実行
            start_time = time.time()
            cpu_result = cpu_stress()
            gpu_result = gpu_stress()
            stress_duration = time.time() - start_time
            
            # ストレス処理後のメトリクス
            time.sleep(2)
            after_stress = self.monitoring_system.metrics_collector.get_latest_system_metrics()
            
            # アラート確認
            active_alerts = self.monitoring_system.alert_manager.get_active_alerts()
            
            # 監視停止
            self.monitoring_system.stop_monitoring()
            
            # 結果分析
            cpu_increase = (after_stress.cpu_percent - before_stress.cpu_percent) if before_stress and after_stress else 0
            memory_increase = (after_stress.memory_percent - before_stress.memory_percent) if before_stress and after_stress else 0
            
            test_result = {
                "test_type": "stress_monitoring",
                "stress_duration": stress_duration,
                "cpu_stress_applied": cpu_result is not None,
                "gpu_stress_applied": gpu_result is not None,
                "metrics_before_stress": {
                    "cpu_percent": before_stress.cpu_percent if before_stress else None,
                    "memory_percent": before_stress.memory_percent if before_stress else None
                },
                "metrics_after_stress": {
                    "cpu_percent": after_stress.cpu_percent if after_stress else None,
                    "memory_percent": after_stress.memory_percent if after_stress else None
                },
                "resource_impact": {
                    "cpu_increase": cpu_increase,
                    "memory_increase": memory_increase
                },
                "alerts_triggered": len(active_alerts),
                "monitoring_responsive": before_stress is not None and after_stress is not None,
                "success": before_stress is not None and after_stress is not None
            }
            
            logger.info(f"ストレス監視テスト完了: CPU増加{cpu_increase:.1f}%, アラート{len(active_alerts)}件")
            return test_result
            
        except Exception as e:
            logger.error(f"ストレス監視テストエラー: {e}")
            try:
                self.monitoring_system.stop_monitoring()
            except:
                pass
            
            return {
                "test_type": "stress_monitoring",
                "success": False,
                "error": str(e)
            }
    
    async def run_all_tests(self):
        """全テスト実行"""
        logger.info("🚀 PH2-006 監視システム全テスト開始")
        
        self.test_results = {
            "timestamp": time.time(),
            "test_summary": {
                "total_tests": 6,
                "completed_tests": 0,
                "failed_tests": 0
            },
            "test_results": {}
        }
        
        tests = [
            ("metrics_collection", self.test_metrics_collection),
            ("alert_system", self.test_alert_system),
            ("processing_metrics_integration", self.test_processing_metrics_integration),
            ("report_generation", self.test_report_generation),
            ("web_dashboard_api", self.test_web_dashboard_api),
            ("stress_monitoring", self.test_stress_monitoring)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"📝 テスト実行: {test_name}")
                result = test_func()
                self.test_results["test_results"][test_name] = result
                
                if result.get("success", False):
                    self.test_results["test_summary"]["completed_tests"] += 1
                else:
                    self.test_results["test_summary"]["failed_tests"] += 1
                    
                # テスト間の待機
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"テスト実行エラー {test_name}: {e}")
                self.test_results["test_results"][test_name] = {
                    "test_type": test_name,
                    "success": False,
                    "error": str(e)
                }
                self.test_results["test_summary"]["failed_tests"] += 1
        
        # 結果保存
        self.save_test_results()
        
        # サマリーログ
        self.log_test_summary()
    
    def save_test_results(self):
        """テスト結果保存"""
        results_path = self.tests_dir / "ph2_006_monitoring_test_results.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📊 テスト結果保存: {results_path}")
    
    def log_test_summary(self):
        """テストサマリーログ出力"""
        logger.info("=" * 60)
        logger.info("🎯 PH2-006 監視システムテスト サマリー")
        logger.info("=" * 60)
        
        summary = self.test_results["test_summary"]
        logger.info(f"総テスト数: {summary['total_tests']}")
        logger.info(f"完了テスト: {summary['completed_tests']}")
        logger.info(f"失敗テスト: {summary['failed_tests']}")
        
        success_rate = summary['completed_tests'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        logger.info(f"成功率: {success_rate:.1%}")
        
        # 個別テスト結果
        logger.info("\n🔧 個別テスト結果:")
        for test_name, result in self.test_results["test_results"].items():
            status = "✅ 成功" if result.get("success", False) else "❌ 失敗"
            logger.info(f"  {test_name}: {status}")
        
        logger.info("=" * 60)


async def main():
    """メイン実行"""
    tester = PH2006MonitoringTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())