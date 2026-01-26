#!/usr/bin/env python3
"""
P1-B002: Windows アクセス対応ダッシュボードテスト

WSL環境でWindowsからアクセス可能なダッシュボードサーバーを起動
"""

import asyncio
import logging
import socket
from features.evaluation.realtime_dashboard.dashboard_server import DashboardServer
from features.evaluation.realtime_dashboard.metrics_collector import MetricsCollector

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_wsl_ip():
    """WSLのIPアドレスを取得"""
    try:
        # WSLのeth0インターフェースのIPアドレスを取得
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "172.29.132.130"  # フォールバック


async def test_windows_accessible_dashboard():
    """Windows からアクセス可能なダッシュボードサーバーのテスト起動"""

    wsl_ip = get_wsl_ip()
    port = 8084

    # メトリクス収集器を作成
    collector = MetricsCollector()

    # テストデータを追加（リアルタイム品質ダッシュボードのデモ用）
    test_images = [
        ("anime_character_1.jpg", True, 0.92, "High quality extraction"),
        ("manga_page_1.jpg", True, 0.87, "Good character detection"),
        ("complex_scene.jpg", False, 0.45, "Multiple characters detected"),
        ("low_resolution.jpg", False, None, "Resolution too low"),
        ("perfect_character.jpg", True, 0.98, "Perfect extraction"),
    ]

    logger.info("🎯 テストデータ生成中...")
    for i, (image_name, success, quality, note) in enumerate(test_images):
        collector.start_processing(image_name)
        await asyncio.sleep(0.1)  # リアルタイム感をシミュレート

        if success:
            collector.complete_processing(
                image_name=image_name,
                success=True,
                quality_score=quality,
                memory_usage={"ram_mb": 1500 + i * 100, "gpu_mb": 2800 + i * 50},
            )
            logger.info(f"✅ {image_name}: 品質スコア {quality}")
        else:
            collector.complete_processing(image_name=image_name, success=False, error_message=note)
            logger.info(f"❌ {image_name}: {note}")

    # ダッシュボードサーバーを起動
    server = DashboardServer(collector, host="0.0.0.0", port=port)

    try:
        await server.start()
        logger.info("🎯 P1-B002 リアルタイム品質ダッシュボード起動完了")
        logger.info("=" * 60)
        logger.info("📍 アクセス方法:")
        logger.info(f"   🖥️  WSL内から: http://localhost:{port}")
        logger.info(f"   🪟 Windowsから: http://{wsl_ip}:{port}")
        logger.info("=" * 60)
        logger.info("📊 ダッシュボード機能:")
        logger.info("   • リアルタイム処理状況表示")
        logger.info("   • 品質スコア時系列グラフ")
        logger.info("   • 成功/失敗統計チャート")
        logger.info("   • 処理中画像リスト")
        logger.info("   • WebSocket自動更新（1秒間隔）")
        logger.info("=" * 60)
        logger.info("📱 Ctrl+C でサーバー停止")

        # サーバーを実行し続ける
        while True:
            await asyncio.sleep(5)
            # 定期的に新しいテストデータを追加（デモ用）
            collector.start_processing(f"demo_{asyncio.get_event_loop().time():.0f}.jpg")
            await asyncio.sleep(2)
            collector.complete_processing(
                f"demo_{asyncio.get_event_loop().time():.0f}.jpg",
                success=True,
                quality_score=0.75 + (asyncio.get_event_loop().time() % 0.25),
            )

    except KeyboardInterrupt:
        logger.info("🛑 ダッシュボードサーバー停止中...")
        await server.stop()
        logger.info("✅ P1-B002 ダッシュボードサーバー停止完了")


if __name__ == "__main__":
    try:
        asyncio.run(test_windows_accessible_dashboard())
    except KeyboardInterrupt:
        print("\n✅ サーバー終了")
