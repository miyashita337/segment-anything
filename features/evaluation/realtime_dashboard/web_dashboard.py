#!/usr/bin/env python3
"""
PH2-006: リアルタイムWebダッシュボード
監視システムのWeb UIフロントエンド
"""

import asyncio
import json
import logging

# プロジェクトルート追加
import sys
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from features.evaluation.realtime_dashboard.monitoring_system import PH2006MonitoringSystem


class PH2006WebDashboard:
    """PH2-006 リアルタイムWebダッシュボード"""

    def __init__(self, monitoring_system: PH2006MonitoringSystem, port: int = 5000):
        """
        初期化

        Args:
            monitoring_system: 監視システムインスタンス
            port: Webサーバーポート
        """
        self.monitoring_system = monitoring_system
        self.port = port
        self.logger = logging.getLogger(__name__)

        # Flask アプリ初期化
        self.app = Flask(__name__)
        self.app.config["JSON_AS_ASCII"] = False

        # ルート設定
        self._setup_routes()

        # サーバースレッド
        self.server_thread: threading.Thread = None
        self.server_running = False

        self.logger.info(f"Webダッシュボード初期化完了: ポート {port}")

    def _setup_routes(self):
        """ルート設定"""

        @self.app.route("/")
        def dashboard():
            """メインダッシュボード"""
            return render_template_string(self._get_dashboard_template())

        @self.app.route("/api/status")
        def api_status():
            """監視状態API"""
            try:
                status = self.monitoring_system.get_monitoring_status()
                return jsonify(status)
            except Exception as e:
                self.logger.error(f"Status API エラー: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/report")
        def api_report():
            """監視レポートAPI"""
            try:
                duration_hours = request.args.get("hours", 1, type=int)
                report = self.monitoring_system.generate_report(duration_hours)
                return jsonify(report)
            except Exception as e:
                self.logger.error(f"Report API エラー: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/alerts")
        def api_alerts():
            """アラートAPI"""
            try:
                active_alerts = self.monitoring_system.alert_manager.get_active_alerts()
                alerts_data = [
                    {
                        "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                        "message": alert.message,
                        "severity": alert.rule.severity,
                        "metric_name": alert.rule.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.rule.threshold,
                    }
                    for alert in active_alerts
                ]
                return jsonify({"alerts": alerts_data})
            except Exception as e:
                self.logger.error(f"Alerts API エラー: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/metrics/history")
        def api_metrics_history():
            """メトリクス履歴API"""
            try:
                minutes = request.args.get("minutes", 10, type=int)
                end_time = datetime.now().timestamp()
                start_time = end_time - (minutes * 60)

                (
                    system_metrics,
                    processing_metrics,
                ) = self.monitoring_system.metrics_collector.get_metrics_in_range(
                    start_time, end_time
                )

                # データポイント数制限（最大100ポイント）
                max_points = 100
                if len(system_metrics) > max_points:
                    step = len(system_metrics) // max_points
                    system_metrics = system_metrics[::step]

                metrics_data = [
                    {
                        "timestamp": datetime.fromtimestamp(m.timestamp).isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "gpu_utilization": m.gpu_utilization,
                        "gpu_memory_used_mb": m.gpu_memory_used_mb,
                    }
                    for m in system_metrics
                ]

                return jsonify({"metrics": metrics_data})
            except Exception as e:
                self.logger.error(f"Metrics History API エラー: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/control/start")
        def api_start_monitoring():
            """監視開始API"""
            try:
                self.monitoring_system.start_monitoring()
                return jsonify({"status": "monitoring started"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/control/stop")
        def api_stop_monitoring():
            """監視停止API"""
            try:
                self.monitoring_system.stop_monitoring()
                return jsonify({"status": "monitoring stopped"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def start_server(self):
        """Webサーバー開始"""
        if self.server_running:
            self.logger.warning("Webサーバーは既に実行中です")
            return

        self.server_running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        self.logger.info(f"🌐 Webダッシュボード開始: http://localhost:{self.port}")

    def stop_server(self):
        """Webサーバー停止"""
        self.server_running = False
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)

        self.logger.info("Webダッシュボード停止")

    def _run_server(self):
        """サーバー実行"""
        try:
            self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=True)
        except Exception as e:
            self.logger.error(f"Webサーバーエラー: {e}")
        finally:
            self.server_running = False

    def _get_dashboard_template(self) -> str:
        """ダッシュボードHTMLテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PH2-006 リアルタイム監視ダッシュボード</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #27ae60;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.inactive {
            background: #e74c3c;
            animation: none;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            max-width: 1400px;
            margin: 20px auto;
            padding: 0 20px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-name {
            font-weight: 500;
        }
        
        .metric-value {
            font-weight: 700;
            font-size: 1.1em;
        }
        
        .metric-value.normal { color: #27ae60; }
        .metric-value.warning { color: #f39c12; }
        .metric-value.critical { color: #e74c3c; }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        .alerts-container {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert-item {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert-critical {
            background: #fdf2f2;
            border-left-color: #e74c3c;
        }
        
        .alert-high {
            background: #fefcf3;
            border-left-color: #f39c12;
        }
        
        .alert-medium {
            background: #f0f9ff;
            border-left-color: #3498db;
        }
        
        .alert-time {
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .alert-message {
            font-weight: 500;
        }
        
        .controls {
            text-align: center;
            margin: 30px 0;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin: 0 10px;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn.stop {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .status-bar {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 PH2-006 リアルタイム監視ダッシュボード</h1>
        <div class="status-bar">
            <div class="status-item">
                <div class="status-indicator" id="monitoring-indicator"></div>
                <span id="monitoring-status">監視状態: 確認中...</span>
            </div>
            <div class="status-item">
                <span id="uptime">稼働時間: --</span>
            </div>
            <div class="status-item">
                <span id="alerts-count">アクティブアラート: --</span>
            </div>
            <div class="status-item">
                <span id="last-update">最終更新: --</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="controls">
            <button class="btn" onclick="startMonitoring()">🚀 監視開始</button>
            <button class="btn stop" onclick="stopMonitoring()">⏹️ 監視停止</button>
        </div>
        
        <div class="grid">
            <!-- システムメトリクス -->
            <div class="card">
                <h3>💻 システムメトリクス</h3>
                <div id="system-metrics" class="loading">データ読み込み中...</div>
            </div>
            
            <!-- GPUメトリクス -->
            <div class="card">
                <h3>🎮 GPU メトリクス</h3>
                <div id="gpu-metrics" class="loading">データ読み込み中...</div>
            </div>
            
            <!-- アクティブアラート -->
            <div class="card">
                <h3>🚨 アクティブアラート</h3>
                <div id="active-alerts" class="alerts-container loading">データ読み込み中...</div>
            </div>
            
            <!-- パフォーマンス統計 -->
            <div class="card">
                <h3>📊 パフォーマンス統計</h3>
                <div id="performance-stats" class="loading">データ読み込み中...</div>
            </div>
        </div>
        
        <!-- リアルタイムチャート -->
        <div class="card">
            <h3>📈 リアルタイムメトリクス</h3>
            <div class="chart-container">
                <canvas id="metrics-chart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // グローバル変数
        let metricsChart;
        let updateInterval;
        
        // 初期化
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            startUpdates();
        });
        
        // チャート初期化
        function initializeChart() {
            const ctx = document.getElementById('metrics-chart').getContext('2d');
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU使用率 (%)',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'メモリ使用率 (%)',
                            data: [],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'GPU使用率 (%)',
                            data: [],
                            borderColor: '#9b59b6',
                            backgroundColor: 'rgba(155, 89, 182, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        }
        
        // 定期更新開始
        function startUpdates() {
            updateDashboard();
            updateInterval = setInterval(updateDashboard, 3000); // 3秒間隔
        }
        
        // ダッシュボード更新
        async function updateDashboard() {
            try {
                await Promise.all([
                    updateStatus(),
                    updateSystemMetrics(),
                    updateAlerts(),
                    updateChart()
                ]);
                
                document.getElementById('last-update').textContent = 
                    `最終更新: ${new Date().toLocaleTimeString()}`;
                    
            } catch (error) {
                console.error('Dashboard update error:', error);
            }
        }
        
        // 監視状態更新
        async function updateStatus() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            const indicator = document.getElementById('monitoring-indicator');
            const status = document.getElementById('monitoring-status');
            const uptime = document.getElementById('uptime');
            const alertsCount = document.getElementById('alerts-count');
            
            if (data.monitoring_active) {
                indicator.classList.remove('inactive');
                status.textContent = '監視状態: アクティブ';
            } else {
                indicator.classList.add('inactive');
                status.textContent = '監視状態: 非アクティブ';
            }
            
            const uptimeSeconds = data.uptime_seconds || 0;
            const hours = Math.floor(uptimeSeconds / 3600);
            const minutes = Math.floor((uptimeSeconds % 3600) / 60);
            uptime.textContent = `稼働時間: ${hours}時間${minutes}分`;
            
            alertsCount.textContent = `アクティブアラート: ${data.active_alerts_count || 0}件`;
        }
        
        // システムメトリクス更新
        async function updateSystemMetrics() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            const systemMetrics = data.latest_system_metrics;
            if (!systemMetrics) {
                document.getElementById('system-metrics').innerHTML = 
                    '<div class="loading">データなし</div>';
                return;
            }
            
            const html = `
                <div class="metric">
                    <span class="metric-name">CPU使用率</span>
                    <span class="metric-value ${getMetricClass(systemMetrics.cpu_percent, 70, 85)}">
                        ${systemMetrics.cpu_percent.toFixed(1)}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-name">メモリ使用率</span>
                    <span class="metric-value ${getMetricClass(systemMetrics.memory_percent, 75, 90)}">
                        ${systemMetrics.memory_percent.toFixed(1)}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-name">メモリ使用量</span>
                    <span class="metric-value normal">
                        ${systemMetrics.memory_used_gb.toFixed(1)}GB / ${systemMetrics.memory_total_gb.toFixed(1)}GB
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-name">ディスク使用率</span>
                    <span class="metric-value ${getMetricClass(systemMetrics.disk_percent, 80, 90)}">
                        ${systemMetrics.disk_percent.toFixed(1)}%
                    </span>
                </div>
            `;
            
            document.getElementById('system-metrics').innerHTML = html;
            
            // GPU メトリクス
            if (systemMetrics.gpu_available) {
                const gpuHtml = `
                    <div class="metric">
                        <span class="metric-name">GPU使用率</span>
                        <span class="metric-value ${getMetricClass(systemMetrics.gpu_utilization, 70, 90)}">
                            ${systemMetrics.gpu_utilization.toFixed(1)}%
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-name">GPU メモリ</span>
                        <span class="metric-value normal">
                            ${(systemMetrics.gpu_memory_used_mb/1024).toFixed(1)}GB / ${(systemMetrics.gpu_memory_total_mb/1024).toFixed(1)}GB
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-name">GPU メモリ使用率</span>
                        <span class="metric-value ${getMetricClass((systemMetrics.gpu_memory_used_mb/systemMetrics.gpu_memory_total_mb)*100, 75, 90)}">
                            ${((systemMetrics.gpu_memory_used_mb/systemMetrics.gpu_memory_total_mb)*100).toFixed(1)}%
                        </span>
                    </div>
                `;
                document.getElementById('gpu-metrics').innerHTML = gpuHtml;
            } else {
                document.getElementById('gpu-metrics').innerHTML = 
                    '<div class="loading">GPU利用不可</div>';
            }
        }
        
        // アラート更新
        async function updateAlerts() {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            
            const alertsContainer = document.getElementById('active-alerts');
            
            if (!data.alerts || data.alerts.length === 0) {
                alertsContainer.innerHTML = '<div class="loading">アクティブアラートなし</div>';
                return;
            }
            
            const html = data.alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                    <div class="alert-message">${alert.message}</div>
                </div>
            `).join('');
            
            alertsContainer.innerHTML = html;
        }
        
        // チャート更新
        async function updateChart() {
            const response = await fetch('/api/metrics/history?minutes=10');
            const data = await response.json();
            
            if (!data.metrics || data.metrics.length === 0) return;
            
            const labels = data.metrics.map(m => new Date(m.timestamp).toLocaleTimeString());
            const cpuData = data.metrics.map(m => m.cpu_percent);
            const memoryData = data.metrics.map(m => m.memory_percent);
            const gpuData = data.metrics.map(m => m.gpu_utilization);
            
            metricsChart.data.labels = labels;
            metricsChart.data.datasets[0].data = cpuData;
            metricsChart.data.datasets[1].data = memoryData;
            metricsChart.data.datasets[2].data = gpuData;
            
            metricsChart.update('none');
        }
        
        // メトリクス値に応じたCSSクラス取得
        function getMetricClass(value, warningThreshold, criticalThreshold) {
            if (value >= criticalThreshold) return 'critical';
            if (value >= warningThreshold) return 'warning';
            return 'normal';
        }
        
        // 監視開始
        async function startMonitoring() {
            try {
                await fetch('/api/control/start');
                setTimeout(updateDashboard, 1000);
            } catch (error) {
                alert('監視開始エラー: ' + error.message);
            }
        }
        
        // 監視停止
        async function stopMonitoring() {
            try {
                await fetch('/api/control/stop');
                setTimeout(updateDashboard, 1000);
            } catch (error) {
                alert('監視停止エラー: ' + error.message);
            }
        }
        
        // ページ終了時のクリーンアップ
        window.addEventListener('beforeunload', function() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
    </script>
</body>
</html>
        """


def create_web_dashboard(
    monitoring_system: PH2006MonitoringSystem, port: int = 5000
) -> PH2006WebDashboard:
    """Webダッシュボード作成"""
    return PH2006WebDashboard(monitoring_system, port)
