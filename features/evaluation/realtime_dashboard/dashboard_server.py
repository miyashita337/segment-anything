"""
リアルタイムダッシュボードサーバー

WebSocketを使用してリアルタイムでメトリクスを配信
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Set

try:
    import aiohttp_cors
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available. Install with: pip install aiohttp aiohttp-cors")

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class DashboardServer:
    """リアルタイムダッシュボードサーバー"""
    
    def __init__(self, metrics_collector: MetricsCollector, host: str = "0.0.0.0", port: int = 8080):
        """
        初期化
        
        Args:
            metrics_collector: メトリクス収集インスタンス
            host: ホスト名
            port: ポート番号
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for dashboard server. Install with: pip install aiohttp aiohttp-cors")
            
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets: Set[web.WebSocketResponse] = set()
        self._setup_routes()
        self._setup_cors()
        self._runner = None
        self._site = None
        self._update_task = None
        
    def _setup_routes(self):
        """ルート設定"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_get('/api/metrics', self.handle_metrics_api)
        self.app.router.add_get('/api/history', self.handle_history_api)
        
    def _setup_cors(self):
        """CORS設定"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def handle_index(self, request):
        """インデックスページハンドラ"""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    async def handle_websocket(self, request):
        """WebSocketハンドラ"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # 初期データ送信
            await self._send_metrics_to_client(ws)
            
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # クライアントからのメッセージ処理（必要に応じて）
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        finally:
            self.websockets.remove(ws)
            
        return ws
    
    async def handle_metrics_api(self, request):
        """メトリクスAPI"""
        metrics = self.metrics_collector.get_aggregated_metrics()
        return web.json_response(metrics.to_dict())
    
    async def handle_history_api(self, request):
        """履歴API"""
        count = int(request.query.get('count', 50))
        history = self.metrics_collector.get_recent_history(count)
        return web.json_response(history)
    
    async def _send_metrics_to_client(self, ws: web.WebSocketResponse):
        """クライアントにメトリクスを送信"""
        try:
            data = {
                "type": "metrics_update",
                "aggregated": self.metrics_collector.get_aggregated_metrics().to_dict(),
                "current_status": self.metrics_collector.get_current_status(),
                "recent_history": self.metrics_collector.get_recent_history(10)
            }
            await ws.send_str(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending metrics: {e}")
    
    async def _broadcast_metrics(self):
        """全クライアントにメトリクスをブロードキャスト"""
        if self.websockets:
            await asyncio.gather(
                *[self._send_metrics_to_client(ws) for ws in self.websockets],
                return_exceptions=True
            )
    
    async def _update_loop(self):
        """定期的な更新ループ"""
        while True:
            try:
                await self._broadcast_metrics()
                await asyncio.sleep(1)  # 1秒ごとに更新
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(5)
    
    def _generate_dashboard_html(self) -> str:
        """ダッシュボードHTMLを生成"""
        return '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>リアルタイム品質ダッシュボード - P1-B002</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2ecc71;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            height: 400px;
        }
        .status-processing {
            color: #f39c12;
        }
        .status-failed {
            color: #e74c3c;
        }
        .processing-list {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .processing-item {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 リアルタイム品質ダッシュボード</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">処理済み画像</div>
                <div class="metric-value" id="processed-count">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">成功率</div>
                <div class="metric-value" id="success-rate">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">平均品質スコア</div>
                <div class="metric-value" id="avg-quality">0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">処理速度 (FPS)</div>
                <div class="metric-value" id="current-fps">0.00</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="quality-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="processing-chart"></canvas>
        </div>
        
        <div class="processing-list">
            <h3>処理中の画像</h3>
            <div id="processing-images"></div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.hostname}:${window.location.port}/ws`);
        
        // チャート初期化
        const qualityCtx = document.getElementById('quality-chart').getContext('2d');
        const qualityChart = new Chart(qualityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '品質スコア',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        const processingCtx = document.getElementById('processing-chart').getContext('2d');
        const processingChart = new Chart(processingCtx, {
            type: 'bar',
            data: {
                labels: ['成功', '失敗', '処理中'],
                datasets: [{
                    label: '画像数',
                    data: [0, 0, 0],
                    backgroundColor: ['#2ecc71', '#e74c3c', '#f39c12']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // メトリクス更新
            if (data.aggregated) {
                document.getElementById('processed-count').textContent = data.aggregated.processed_images;
                document.getElementById('success-rate').textContent = (data.aggregated.success_rate * 100).toFixed(1) + '%';
                document.getElementById('avg-quality').textContent = data.aggregated.average_quality_score.toFixed(3);
                document.getElementById('current-fps').textContent = data.aggregated.current_fps.toFixed(2);
                
                // 処理状況チャート更新
                processingChart.data.datasets[0].data = [
                    data.aggregated.success_count,
                    data.aggregated.failed_count,
                    data.current_status.processing_count
                ];
                processingChart.update();
            }
            
            // 品質履歴更新
            if (data.recent_history) {
                const history = data.recent_history;
                qualityChart.data.labels = history.map(h => new Date(h.timestamp * 1000).toLocaleTimeString());
                qualityChart.data.datasets[0].data = history.map(h => h.quality_score || 0);
                qualityChart.update();
            }
            
            // 処理中画像リスト更新
            if (data.current_status) {
                const processingDiv = document.getElementById('processing-images');
                if (data.current_status.processing_images.length > 0) {
                    processingDiv.innerHTML = data.current_status.processing_images
                        .map(img => `<div class="processing-item">📝 ${img}</div>`)
                        .join('');
                } else {
                    processingDiv.innerHTML = '<div class="processing-item">処理中の画像はありません</div>';
                }
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = function() {
            console.log('WebSocket connection closed');
        };
    </script>
</body>
</html>'''
    
    async def start(self):
        """サーバー開始"""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        # 更新ループ開始
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
        
    async def stop(self):
        """サーバー停止"""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
                
        if self._runner:
            await self._runner.cleanup()
            
        logger.info("Dashboard server stopped")
    
    def run_in_thread(self):
        """別スレッドでサーバーを実行"""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
            try:
                loop.run_forever()
            except KeyboardInterrupt:
                pass
            finally:
                loop.run_until_complete(self.stop())
                loop.close()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread