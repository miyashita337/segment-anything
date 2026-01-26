#!/usr/bin/env python3
"""
P1-B002: 確実に動作するダッシュボードサーバー

WSL2ネットワーク問題を完全に回避した設計
"""

import aiohttp_cors
import asyncio
import json
import logging
import socket
from aiohttp import WSMsgType, web
from features.evaluation.realtime_dashboard.metrics_collector import MetricsCollector
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReliableDashboardServer:
    def __init__(self, port=8088):
        self.port = port
        self.collector = MetricsCollector()
        self.app = web.Application()
        self.websockets = set()
        self._add_test_data()
        self._setup_routes()
        self._setup_cors()

    def _add_test_data(self):
        """リアルなテストデータを追加"""
        test_scenarios = [
            ("anime_girl_01.jpg", True, 0.94, "Perfect extraction"),
            ("manga_page_complex.jpg", False, 0.35, "Multiple characters detected"),
            ("sketch_character.jpg", True, 0.78, "Sketch processed successfully"),
            ("low_quality.jpg", False, None, "Image resolution too low"),
            ("perfect_pose.jpg", True, 0.97, "Excellent quality"),
            ("background_noise.jpg", True, 0.82, "Background cleaned"),
            ("partial_character.jpg", False, 0.44, "Character partially cropped"),
            ("high_contrast.jpg", True, 0.89, "High contrast handled well"),
        ]

        logger.info("🎯 P1-B002 テストデータ生成中...")
        for name, success, quality, note in test_scenarios:
            self.collector.start_processing(name)
            self.collector.complete_processing(
                name,
                success=success,
                quality_score=quality,
                memory_usage={"ram_mb": 1650, "gpu_mb": 2900} if success else None,
                error_message=note if not success else None,
            )

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_dashboard)
        self.app.router.add_get("/ws", self.handle_websocket)
        self.app.router.add_get("/api/metrics", self.handle_api_metrics)
        self.app.router.add_get("/api/status", self.handle_api_status)
        self.app.router.add_get("/test", self.handle_connection_test)

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

    async def handle_connection_test(self, request):
        """接続テストページ"""
        html = f"""<!DOCTYPE html>
<html><head><title>P1-B002 接続テスト</title></head>
<body style="font-family: Arial; padding: 40px; background: #f5f5f5;">
    <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
        <h1 style="color: #27ae60;">✅ P1-B002 接続成功!</h1>
        <p><strong>サーバー:</strong> 正常動作中</p>
        <p><strong>ポート:</strong> {self.port}</p>
        <p><strong>時刻:</strong> <span id="time"></span></p>
        <p><strong>IP:</strong> {request.remote}</p>
        <hr>
        <p>🎯 <a href="/">メインダッシュボードはこちら</a></p>
        <p>📊 <a href="/api/metrics">APIメトリクス</a></p>
    </div>
    <script>
        document.getElementById('time').textContent = new Date().toLocaleString();
        setInterval(() => {{
            document.getElementById('time').textContent = new Date().toLocaleString();
        }}, 1000);
    </script>
</body></html>"""
        return web.Response(text=html, content_type="text/html")

    async def handle_dashboard(self, request):
        """メインダッシュボード"""
        metrics = self.collector.get_aggregated_metrics()

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>P1-B002: リアルタイム品質ダッシュボード</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{ 
            max-width: 1200px; margin: 0 auto; 
            background: white; border-radius: 15px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
            color: white; padding: 30px; text-align: center; 
        }}
        .header h1 {{ margin: 0; font-size: 2.2em; }}
        .content {{ padding: 30px; }}
        .metrics {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; margin-bottom: 30px; 
        }}
        .metric {{ 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            padding: 20px; border-radius: 10px; text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{ 
            font-size: 2.2em; font-weight: bold; margin: 10px 0; 
        }}
        .metric-label {{ 
            color: #666; font-size: 0.9em; text-transform: uppercase; 
        }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .info {{ color: #3498db; }}
        .chart-container {{ 
            background: white; padding: 20px; border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; 
        }}
        .status-indicator {{ 
            display: inline-block; width: 10px; height: 10px; 
            border-radius: 50%; margin-right: 10px; 
        }}
        .online {{ background: #27ae60; }}
        .connection-info {{ 
            background: #e8f6f3; padding: 15px; border-radius: 8px; 
            border-left: 4px solid #27ae60; margin-bottom: 20px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 P1-B002: リアルタイム品質ダッシュボード</h1>
            <p>WSL2 対応版 - ポート {self.port}</p>
        </div>
        
        <div class="content">
            <div class="connection-info">
                <span class="status-indicator online"></span>
                <strong>✅ 接続成功!</strong> 
                P1-B002 ダッシュボードが正常に動作しています
                <span style="float: right;">🕒 <span id="current-time"></span></span>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">処理済み画像</div>
                    <div class="metric-value info">{metrics.processed_images}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">成功数</div>
                    <div class="metric-value success">{metrics.success_count}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">失敗数</div>
                    <div class="metric-value warning">{metrics.failed_count}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">成功率</div>
                    <div class="metric-value success">{metrics.success_rate:.1%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">平均品質スコア</div>
                    <div class="metric-value info">{metrics.average_quality_score:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">処理速度</div>
                    <div class="metric-value info">{metrics.current_fps:.2f} FPS</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📊 成功/失敗統計</h3>
                <canvas id="statusChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>🔧 WebSocket リアルタイム更新テスト</h3>
                <p>接続状態: <span id="ws-status">接続中...</span></p>
                <p>受信メッセージ数: <span id="msg-count">0</span></p>
                <p>最終更新: <span id="last-update">未受信</span></p>
            </div>
        </div>
    </div>
    
    <script>
        // 現在時刻表示
        function updateTime() {{
            document.getElementById('current-time').textContent = new Date().toLocaleTimeString();
        }}
        updateTime();
        setInterval(updateTime, 1000);
        
        // 成功/失敗チャート
        const ctx = document.getElementById('statusChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['成功', '失敗'],
                datasets: [{{
                    data: [{metrics.success_count}, {metrics.failed_count}],
                    backgroundColor: ['#27ae60', '#e74c3c']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
        
        // WebSocket接続テスト
        const ws = new WebSocket(`ws://${{window.location.host}}/ws`);
        let msgCount = 0;
        
        ws.onopen = function() {{
            document.getElementById('ws-status').innerHTML = '<span style="color: #27ae60;">✅ 接続成功</span>';
        }};
        
        ws.onmessage = function(event) {{
            msgCount++;
            document.getElementById('msg-count').textContent = msgCount;
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }};
        
        ws.onerror = function() {{
            document.getElementById('ws-status').innerHTML = '<span style="color: #e74c3c;">❌ 接続エラー</span>';
        }};
        
        ws.onclose = function() {{
            document.getElementById('ws-status').innerHTML = '<span style="color: #f39c12;">⚠️ 接続切断</span>';
        }};
    </script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)

        try:
            # 定期的にテストメッセージを送信
            async def send_updates():
                while True:
                    try:
                        data = {
                            {
                                "type": "update",
                                "timestamp": asyncio.get_event_loop().time(),
                                "metrics": self.collector.get_aggregated_metrics().to_dict(),
                            }
                        }
                        await ws.send_str(json.dumps(data))
                        await asyncio.sleep(2)
                    except:
                        break

            update_task = asyncio.create_task(send_updates())

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {{ws.exception()}}")
                    break

        finally:
            self.websockets.remove(ws)
            if "update_task" in locals():
                update_task.cancel()

        return ws

    async def handle_api_metrics(self, request):
        metrics = self.collector.get_aggregated_metrics()
        return web.json_response(metrics.to_dict())

    async def handle_api_status(self, request):
        return web.json_response(
            {
                {
                    "status": "online",
                    "port": self.port,
                    "websocket_connections": len(self.websockets),
                    "uptime": asyncio.get_event_loop().time(),
                }
            }
        )

    def get_local_ip(self):
        """ローカルIPアドレス取得"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "172.29.132.130"

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        local_ip = self.get_local_ip()

        logger.info("🎉 P1-B002 確実動作ダッシュボードサーバー起動完了!")
        logger.info("=" * 60)
        logger.info(f"📍 アクセス方法 (複数の選択肢):")
        logger.info(f"   🎯 VS Code Port Forward: 推奨方法")
        logger.info(f"   🌐 WSL内部: http://localhost:{self.port}")
        logger.info(f"   🪟 Windows直接: http://{local_ip}:{self.port}")
        logger.info(f"   🔧 接続テスト: http://localhost:{self.port}/test")
        logger.info("=" * 60)
        logger.info("📊 機能:")
        logger.info("   ✅ リアルタイムメトリクス表示")
        logger.info("   ✅ WebSocket通信テスト")
        logger.info("   ✅ Chart.js グラフ表示")
        logger.info("   ✅ 接続診断機能")
        logger.info("=" * 60)


async def main():
    server = ReliableDashboardServer(port=8088)
    await server.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 サーバー停止")


if __name__ == "__main__":
    asyncio.run(main())
