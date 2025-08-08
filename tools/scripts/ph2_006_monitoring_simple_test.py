#!/usr/bin/env python3
"""
PH2-006 監視システム簡易テスト
基本機能の動作確認
"""

import json
import logging
import time
from pathlib import Path

# プロジェクトルート追加
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.evaluation.realtime_dashboard.monitoring_system import PH2006MonitoringSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_monitoring():
    """基本監視機能テスト"""
    logger.info("🚀 PH2-006 基本監視機能テスト開始")
    
    # 監視システム初期化
    monitoring_system = PH2006MonitoringSystem(collection_interval=1.0)
    
    try:
        # 監視開始
        logger.info("監視開始...")
        monitoring_system.start_monitoring()
        
        # 5秒間監視
        logger.info("5秒間データ収集...")
        time.sleep(5)
        
        # 状態確認
        status = monitoring_system.get_monitoring_status()
        logger.info(f"監視状態: アクティブ={status['monitoring_active']}")
        logger.info(f"収集メトリクス数: {status['metrics_collected']}")
        logger.info(f"アクティブアラート: {status['active_alerts_count']}")
        
        # 最新メトリクス表示
        latest = status.get('latest_system_metrics')
        if latest:
            logger.info(f"最新メトリクス:")
            logger.info(f"  CPU: {latest['cpu_percent']:.1f}%")
            logger.info(f"  メモリ: {latest['memory_percent']:.1f}%")
            logger.info(f"  GPU利用可能: {latest['gpu_available']}")
            if latest['gpu_available']:
                logger.info(f"  GPU使用率: {latest['gpu_utilization']:.1f}%")
        
        # レポート生成テスト
        logger.info("レポート生成...")
        report = monitoring_system.generate_report(duration_hours=1)
        logger.info(f"レポート生成完了: {len(report)}項目")
        
        # 結果保存
        output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-006/tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        test_results = {
            "test_type": "basic_monitoring",
            "timestamp": time.time(),
            "monitoring_status": status,
            "report_summary": {
                "monitoring_health": report.get("monitoring_health", {}),
                "system_statistics": report.get("system_statistics", {})
            },
            "success": status['monitoring_active'] and status['metrics_collected'] > 0
        }
        
        results_path = output_dir / "ph2_006_basic_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📊 テスト結果保存: {results_path}")
        
        # 監視停止
        logger.info("監視停止...")
        monitoring_system.stop_monitoring()
        
        # 結果判定
        success = test_results["success"]
        logger.info(f"✅ 基本監視機能テスト: {'成功' if success else '失敗'}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        try:
            monitoring_system.stop_monitoring()
        except:
            pass
        
        return {
            "test_type": "basic_monitoring",
            "success": False,
            "error": str(e)
        }


def main():
    """メイン実行"""
    results = test_basic_monitoring()
    
    logger.info("=" * 60)
    logger.info("🎯 PH2-006 基本監視テスト サマリー")
    logger.info("=" * 60)
    logger.info(f"テスト結果: {'✅ 成功' if results.get('success', False) else '❌ 失敗'}")
    
    if results.get('monitoring_status'):
        status = results['monitoring_status']
        logger.info(f"メトリクス収集数: {status.get('metrics_collected', 0)}")
        logger.info(f"アクティブアラート: {status.get('active_alerts_count', 0)}")
        logger.info(f"監視稼働時間: {status.get('uptime_seconds', 0):.1f}秒")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()