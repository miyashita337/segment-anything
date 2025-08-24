#!/usr/bin/env python3
"""
INTEGRATE-3-6 ダッシュボード外部Web公開サーバー
"""

import json
import logging
import os
import socketserver
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def setup_logging():
    """ログ設定"""
    log_file = Path("dashboard_server.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DashboardHandler(SimpleHTTPRequestHandler):
    """カスタムHTTPハンドラー"""
    
    def __init__(self, *args, **kwargs):
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """GETリクエスト処理"""
        path = self.path.strip('/')
        
        # ルートアクセス時はインデックス画面表示
        if path == '' or path == 'index.html':
            self.serve_index()
            return
        
        # トラッカー別ダッシュボード
        if path.startswith('INTEGRATE-3-6-'):
            tracker_id = path
            dashboard_file = self.workspace_base / tracker_id / "dashboard" / "dashboard.html"
            
            if dashboard_file.exists():
                self.serve_dashboard(dashboard_file)
            else:
                self.send_error(404, f"Dashboard not found: {tracker_id}")
            return
        
        # 静的ファイル
        super().do_GET()
    
    def serve_index(self):
        """インデックス画面表示"""
        # 各トラッカーの状態確認
        trackers = ["INTEGRATE-3-6-01", "INTEGRATE-3-6-02", "INTEGRATE-3-6-03", "INTEGRATE-3-6-04"]
        tracker_status = []
        
        for tracker_id in trackers:
            dashboard_file = self.workspace_base / tracker_id / "dashboard" / "dashboard.html"
            extraction_dir = self.workspace_base / tracker_id / "extraction"
            
            # 画像数カウント
            image_count = len(list(extraction_dir.glob("kana08_*.jpg"))) if extraction_dir.exists() else 0
            
            status = {
                'id': tracker_id,
                'name': {
                    'INTEGRATE-3-6-01': 'Phase 3-6統合初期版',
                    'INTEGRATE-3-6-02': 'Phase 3-6改良版',
                    'INTEGRATE-3-6-03': 'YOLO汎用版検証',
                    'INTEGRATE-3-6-04': 'アニメ特化版検証'
                }.get(tracker_id, tracker_id),
                'available': dashboard_file.exists(),
                'image_count': image_count,
                'file_size': dashboard_file.stat().st_size / 1024 if dashboard_file.exists() else 0
            }
            tracker_status.append(status)
        
        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>INTEGRATE-3-6 Series Dashboard Portal</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 3em; margin-bottom: 15px; }}
        .header .subtitle {{ font-size: 1.3em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .tracker-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; }}
        .tracker-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; padding: 25px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .tracker-card:hover {{ transform: translateY(-5px); }}
        .tracker-card.available {{ border-left: 5px solid #27ae60; }}
        .tracker-card.unavailable {{ border-left: 5px solid #e74c3c; }}
        .tracker-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
        .tracker-id {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .tracker-stats {{ margin-bottom: 20px; }}
        .stat-item {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
        .stat-label {{ color: #666; }}
        .stat-value {{ font-weight: bold; }}
        .tracker-link {{ display: inline-block; background: #3498db; color: white; padding: 10px 20px; border-radius: 25px; text-decoration: none; transition: background 0.3s ease; }}
        .tracker-link:hover {{ background: #2980b9; }}
        .tracker-link.unavailable {{ background: #95a5a6; cursor: not-allowed; }}
        .info-section {{ background: #f8f9fa; padding: 25px; border-radius: 12px; margin-bottom: 30px; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 INTEGRATE-3-6 Series</h1>
            <div class="subtitle">キャラクター抽出結果ダッシュボード ポータル</div>
        </div>
        <div class="content">
            <div class="info-section">
                <h3>📊 ダッシュボード概要</h3>
                <p>各INTEGRATE-3-6トラッカーの抽出結果を視覚的に確認できるダッシュボードです。</p>
                <p>品質スコア付きで全画像を表示し、モデル比較・性能評価が可能です。</p>
                <p><strong>生成時刻:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="tracker-grid">"""
        
        for tracker in tracker_status:
            status_class = "available" if tracker['available'] else "unavailable"
            link_class = "" if tracker['available'] else "unavailable"
            link_text = "ダッシュボードを開く" if tracker['available'] else "未生成"
            
            html_content += f"""
                <div class="tracker-card {status_class}">
                    <div class="tracker-title">{tracker['name']}</div>
                    <div class="tracker-id">{tracker['id']}</div>
                    <div class="tracker-stats">
                        <div class="stat-item">
                            <span class="stat-label">画像数:</span>
                            <span class="stat-value">{tracker['image_count']}枚</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">ダッシュボードサイズ:</span>
                            <span class="stat-value">{tracker['file_size']:.1f}KB</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">状態:</span>
                            <span class="stat-value">{'✅ 利用可能' if tracker['available'] else '❌ 未生成'}</span>
                        </div>
                    </div>
                    {"<a href='/" + tracker['id'] + "' class='tracker-link " + link_class + "'>" + link_text + "</a>" if tracker['available'] else "<span class='tracker-link unavailable'>" + link_text + "</span>"}
                </div>"""
        
        html_content += f"""
            </div>
        </div>
        <div class="footer">
            <p>🔬 INTEGRATE-3-6 Series Dashboard Portal | Server: http://localhost:8088 | Generated with Claude Code</p>
        </div>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_dashboard(self, dashboard_file):
        """個別ダッシュボード提供"""
        try:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        
        except Exception as e:
            self.send_error(500, f"Error serving dashboard: {str(e)}")


def start_dashboard_server(port=8088):
    """ダッシュボードサーバー起動"""
    logger = setup_logging()
    
    logger.info(f"🚀 INTEGRATE-3-6 ダッシュボードサーバー起動開始")
    logger.info(f"📡 ポート: {port}")
    
    # サーバー設定
    server_address = ('', port)
    
    try:
        with HTTPServer(server_address, DashboardHandler) as httpd:
            logger.info(f"✅ サーバー起動完了")
            logger.info(f"🌐 アクセスURL: http://localhost:{port}")
            logger.info(f"📊 利用可能なダッシュボード:")
            
            workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
            for tracker_id in ["INTEGRATE-3-6-01", "INTEGRATE-3-6-02", "INTEGRATE-3-6-03", "INTEGRATE-3-6-04"]:
                dashboard_file = workspace_base / tracker_id / "dashboard" / "dashboard.html"
                if dashboard_file.exists():
                    logger.info(f"  ✅ http://localhost:{port}/{tracker_id}")
                else:
                    logger.info(f"  ❌ {tracker_id} (未生成)")
            
            logger.info("🔄 サーバー実行中... (Ctrl+C で停止)")
            
            # PIDファイル作成
            with open("dashboard_server.pid", 'w') as f:
                f.write(str(os.getpid()))
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        logger.info("⏹️ サーバー停止")
    except Exception as e:
        logger.error(f"❌ サーバーエラー: {e}")
    finally:
        # PIDファイル削除
        if Path("dashboard_server.pid").exists():
            Path("dashboard_server.pid").unlink()


if __name__ == "__main__":
    start_dashboard_server()