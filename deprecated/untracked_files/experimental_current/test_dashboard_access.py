#!/usr/bin/env python3
"""
INTG-057: ダッシュボードアクセステスト

WSL環境でのダッシュボードアクセス確認用テストスクリプト
"""

import asyncio
import logging
from features.evaluation.realtime_dashboard.dashboard_server import DashboardServer
from features.evaluation.realtime_dashboard.metrics_collector import MetricsCollector

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_dashboard_server():
    """ダッシュボードサーバーのテスト起動"""

    # メトリクス収集器を作成
    collector = MetricsCollector()

    # テストデータを追加
    collector.start_processing("test1.jpg")
    collector.complete_processing("test1.jpg", success=True, quality_score=0.85)

    collector.start_processing("test2.jpg")
    collector.complete_processing("test2.jpg", success=False, error_message="Test error")

    # ダッシュボードサーバーを起動
    server = DashboardServer(collector, host="0.0.0.0", port=8083)

    try:
        await server.start()
        logger.info("🎯 ダッシュボードサーバー起動完了")
        logger.info("📍 アクセスURL:")
        logger.info("   WSL内から: http://localhost:8083")
        logger.info("   Windowsから: http://172.29.132.130:8083")
        logger.info("📱 Ctrl+C でサーバー停止")

        # サーバーを実行し続ける
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 サーバー停止中...")
        await server.stop()
        logger.info("✅ サーバー停止完了")


if __name__ == "__main__":
    asyncio.run(test_dashboard_server())
