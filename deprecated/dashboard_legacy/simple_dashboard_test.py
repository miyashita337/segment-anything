#!/usr/bin/env python3
"""
P1-B002: シンプルダッシュボードテスト (ネットワーク問題解決版)
"""

import aiohttp_cors
import asyncio
import json
import logging
from aiohttp import WSMsgType, web
from features.evaluation.realtime_dashboard.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDashboardServer:
    def __init__(self, port=8085):
        self.port = port
        self.collector = MetricsCollector()
        self.app = web.Application()
        self.websockets = set()
        self._setup_routes()
        self._setup_cors()

        # テストデータ追加
        self._add_test_data()

    def _add_test_data(self):
        """テストデータを追加"""
        test_data = [
            ("test1.jpg", True, 0.92),
            ("test2.jpg", True, 0.87),
            ("test3.jpg", False, None),
            ("test4.jpg", True, 0.95),
        ]

        for name, success, quality in test_data:
            self.collector.start_processing(name)
            self.collector.complete_processing(
                name,
                success=success,
                quality_score=quality,
                error_message=None if success else "Test error",
            )

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/ws", self.handle_websocket)
        self.app.router.add_get("/api/metrics", self.handle_metrics)

    def _setup_cors(self):
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            },
        )
        for route in list(self.app.router.routes()):
            cors.add(route)

    async def handle_index(self, request):
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>P1-B002 リアルタイム品質ダッシュボード</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #27ae60; }}
        .metric-label {{ color: #7f8c8d; margin-top: 10px; }}
        .status {{ padding: 20px; background: #d5dbdb; border-radius: 8px; }}
        .success {{ color: #27ae60; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 P1-B002: リアルタイム品質ダッシュボード</h1>
            <p>WSL環境テスト版 - ポート {self.port}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="processed">0</div>
                <div class="metric-label">処理済み画像</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="success-rate">0%</div>
                <div class="metric-label">成功率</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="avg-quality">0.00</div>
                <div class="metric-label">平均品質スコア</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="fps">0.00</div>
                <div class="metric-label">処理速度 (FPS)</div>
            </div>
        </div>
        
        <div class="status">
            <h3>🔄 システム状態</h3>
            <p id="connection-status">接続中...</p>
            <p id="last-update">最終更新: 未取得</p>
        </div>
        
        <div class="status">
            <h3>📊 接続テスト</h3>
            <p>✅ サーバー応答: 正常</p>
            <p>✅ ポート {self.port}: アクセス可能</p>
            <p>✅ CORS設定: 有効</p>
            <p>✅ テストデータ: 4件</p>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://' + window.location.host + '/ws');
        
        ws.onopen = function() {{
            document.getElementById('connection-status').innerHTML = '<span class="success">✅ WebSocket接続成功</span>';
        }};
        
        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            if (data.aggregated) {{
                document.getElementById('processed').textContent = data.aggregated.processed_images;
                document.getElementById('success-rate').textContent = (data.aggregated.success_rate * 100).toFixed(1) + '%';
                document.getElementById('avg-quality').textContent = data.aggregated.average_quality_score.toFixed(2);
                document.getElementById('fps').textContent = data.aggregated.current_fps.toFixed(2);
                document.getElementById('last-update').textContent = '最終更新: ' + new Date().toLocaleTimeString();
            }}
        }};
        
        ws.onerror = function(error) {{
            document.getElementById('connection-status').innerHTML = '<span class="error">❌ WebSocket接続エラー</span>';
        }};
        
        ws.onclose = function() {{
            document.getElementById('connection-status').innerHTML = '<span class="error">❌ WebSocket接続切断</span>';
        }};
        
        // 初期データ取得
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {{
                document.getElementById('processed').textContent = data.processed_images;
                document.getElementById('success-rate').textContent = (data.success_rate * 100).toFixed(1) + '%';
                document.getElementById('avg-quality').textContent = data.average_quality_score.toFixed(2);
                document.getElementById('fps').textContent = data.current_fps.toFixed(2);
            }});
    </script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)

        try:
            # 初期データ送信
            await self._send_metrics(ws)

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    pass  # メッセージ処理
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {{ws.exception()}}")
        finally:
            self.websockets.remove(ws)

        return ws

    async def handle_metrics(self, request):
        metrics = self.collector.get_aggregated_metrics()
        return web.json_response(metrics.to_dict())

    async def _send_metrics(self, ws):
        try:
            data = {
                {
                    "type": "metrics_update",
                    "aggregated": self.collector.get_aggregated_metrics().to_dict(),
                    "timestamp": asyncio.get_event_loop().time(),
                }
            }
            await ws.send_str(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending metrics: {{e}}")

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)  # すべてのインターフェースでバインド
        await site.start()

        logger.info(f"🎯 P1-B002 ダッシュボードサーバー起動完了")
        logger.info(f"📍 アクセスURL:")
        logger.info(f"   • http://localhost:{self.port}")
        logger.info(f"   • http://127.0.0.1:{self.port}")
        logger.info(f"   • http://172.29.132.130:{self.port}")
        logger.info(f"🔧 ネットワーク: 0.0.0.0:{self.port} (全インターフェース)")


async def main():
    server = SimpleDashboardServer(port=8085)
    await server.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 サーバー停止")


if __name__ == "__main__":
    asyncio.run(main())
