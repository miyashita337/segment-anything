#!/usr/bin/env python3
"""
統合ダッシュボードサーバー
全てのトラッカーダッシュボードを統合管理

http://localhost:8088/ でアクセス可能
"""

import aiohttp_cors
import asyncio
import base64
import json
import logging
import mimetypes
import re
from datetime import datetime
from aiohttp import web
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRACKER_WORKSPACE = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")

# Basic認証設定
BASIC_AUTH_USERNAME = "admin"
BASIC_AUTH_PASSWORD = "dashboard2025!"

class IntegratedDashboardServer:
    def __init__(self, port=8088):
        self.port = port
        self.app = web.Application()  # ミドルウェアを後で追加
        self.tracker_workspace = TRACKER_WORKSPACE
        self._setup_routes()
        self._setup_cors()
        self._scan_dashboards()
        
        # ミドルウェアを最後に追加
        self.app.middlewares.append(self.auth_middleware)
    
    def _scan_dashboards(self):
        """利用可能なダッシュボードをスキャン"""
        self.dashboards = {}
        
        # メインダッシュボード
        main_dashboard = self.tracker_workspace / "main_dashboard.html"
        if main_dashboard.exists():
            self.dashboards['main'] = main_dashboard
        
        # 各トラッカーのダッシュボード
        for tracker_dir in self.tracker_workspace.iterdir():
            if tracker_dir.is_dir():
                dashboard_dir = tracker_dir / "dashboard"
                if dashboard_dir.exists():
                    for html_file in dashboard_dir.glob("*.html"):
                        tracker_id = tracker_dir.name
                        dashboard_key = f"{tracker_id}/{html_file.stem}"
                        self.dashboards[dashboard_key] = html_file
        
        logger.info(f"🎯 {len(self.dashboards)}個のダッシュボードを検出完了")
    
    @web.middleware
    async def auth_middleware(self, request, handler):
        """Basic認証ミドルウェア"""
        # 詳細アクセスログ記録
        self._log_access(request)
        
        # 認証チェック
        if not self._check_basic_auth(request):
            return web.Response(
                text='Unauthorized', 
                status=401,
                headers={'WWW-Authenticate': 'Basic realm="Dashboard"'}
            )
        
        return await handler(request)
    
    def _check_basic_auth(self, request):
        """Basic認証チェック"""
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Basic '):
            return False
        
        try:
            encoded_credentials = auth_header[6:]  # "Basic " を除去
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            username, password = decoded_credentials.split(':', 1)
            
            return username == BASIC_AUTH_USERNAME and password == BASIC_AUTH_PASSWORD
        except Exception:
            return False
    
    def _log_access(self, request):
        """詳細アクセスログ記録"""
        try:
            # リモートIPアドレスの取得（複数の方法を試行）
            client_ip = getattr(request, 'remote', None) or \
                       getattr(request, 'transport', {}).get('peername', ['Unknown'])[0] if hasattr(request, 'transport') else \
                       request.headers.get('X-Forwarded-For', 'Unknown')
            
            user_agent = request.headers.get('User-Agent', 'Unknown')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"🌐 [{timestamp}] {client_ip} - {request.method} {request.path} - {user_agent}")
        except Exception as e:
            logger.error(f"❌ アクセスログ記録エラー: {e}")
    
    def _setup_routes(self):
        """ルーティング設定"""
        self.app.router.add_get('/', self.handle_main)
        self.app.router.add_get('/tracker/{tracker_id}', self.handle_tracker)
        self.app.router.add_get('/tracker/{tracker_id}/{dashboard_name}', self.handle_tracker_dashboard)
        self.app.router.add_get('/api/dashboards', self.handle_api_dashboards)
        self.app.router.add_get('/refresh', self.handle_refresh)
        # 静的ファイルサーバー機能
        self.app.router.add_get('/{path:.*}', self.handle_static)
    
    def _setup_cors(self):
        """CORS設定"""
        try:
            cors = aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            })
            for route in list(self.app.router.routes()):
                cors.add(route)
            logger.info("✅ CORS設定完了")
        except Exception as e:
            logger.warning(f"⚠️ CORS設定エラー（スキップ）: {e}")
    
    async def handle_main(self, request):
        """メインページ - ナビゲーション付き統合ダッシュボード"""
        # カスタムナビゲーション付きラッパー
        nav_html = self._generate_navigation_wrapper("メインダッシュボード", "main")
        return web.Response(text=nav_html, content_type='text/html')
    
    async def handle_tracker(self, request):
        """トラッカーのデフォルトダッシュボード"""
        tracker_id = request.match_info['tracker_id']
        dashboard_key = f"{tracker_id}/dashboard"
        
        if dashboard_key not in self.dashboards:
            # dashboard.htmlが存在しない場合、他のHTMLファイルを探す
            alternatives = [k for k in self.dashboards.keys() if k.startswith(f"{tracker_id}/")]
            if alternatives:
                dashboard_key = alternatives[0]
            else:
                return web.Response(text=f"❌ ダッシュボードが見つかりません: {tracker_id}", status=404)
        
        nav_html = self._generate_navigation_wrapper(tracker_id, dashboard_key)
        return web.Response(text=nav_html, content_type='text/html')
    
    async def handle_tracker_dashboard(self, request):
        """特定のダッシュボードHTML"""
        tracker_id = request.match_info['tracker_id']
        dashboard_name = request.match_info['dashboard_name']
        dashboard_key = f"{tracker_id}/{dashboard_name}"
        
        if dashboard_key not in self.dashboards:
            return web.Response(text=f"❌ ダッシュボードが見つかりません: {dashboard_key}", status=404)
        
        nav_html = self._generate_navigation_wrapper(f"{tracker_id} - {dashboard_name}", dashboard_key)
        return web.Response(text=nav_html, content_type='text/html')
    
    async def handle_static(self, request):
        """静的ファイル配信（HTML、画像、CSS、JS等）"""
        path = request.match_info['path']
        
        # セキュリティチェック
        if '..' in path:
            return web.Response(text="Forbidden", status=403)
        
        # 抽出画像ファイルへの直接アクセスを拒否（グラフ画像は除外）
        if self._is_extracted_image(path):
            logger.warning(f"🚫 抽出画像への直接アクセスを拒否: {path}")
            return web.Response(
                text="🚫 Access Denied: Extracted images are protected for security reasons", 
                status=403
            )
        
        # ワークスペース内のファイルを探す
        file_path = self.tracker_workspace / path
        
        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # HTMLファイルの場合、ファイル名マスキングを適用
            if file_path.suffix.lower() == '.html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # ファイル名マスキング適用
                content = self._apply_filename_masking(content)
                return web.Response(text=content, content_type='text/html')
            else:
                # バイナリファイル（画像等）
                with open(file_path, 'rb') as f:
                    content = f.read()
                return web.Response(body=content, content_type=mime_type or 'application/octet-stream')
        
        return web.Response(text="Not Found", status=404)
    
    async def handle_api_dashboards(self, request):
        """利用可能なダッシュボード一覧API"""
        dashboard_list = []
        for key, path in self.dashboards.items():
            dashboard_list.append({
                'key': key,
                'path': str(path.relative_to(self.tracker_workspace)),
                'tracker': key.split('/')[0] if '/' in key else 'main',
                'name': path.stem
            })
        return web.json_response({
            'total': len(dashboard_list),
            'dashboards': sorted(dashboard_list, key=lambda x: x['key'])
        })
    
    async def handle_refresh(self, request):
        """ダッシュボード再スキャン"""
        self._scan_dashboards()
        return web.json_response({
            'status': 'success',
            'message': f'{len(self.dashboards)} dashboards rescanned'
        })
    
    def _is_extracted_image(self, path):
        """抽出画像かどうかの判定"""
        path_lower = path.lower()
        
        # 画像ファイル拡張子チェック
        if not any(path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            return False
        
        # 抽出ディレクトリまたは抽出関連キーワードチェック
        extracted_indicators = [
            'extraction/',  # 抽出ディレクトリ
            '_extracted',   # 抽出済みファイル
            '_cropped',     # クロップ済みファイル
            '_segment',     # セグメント済みファイル
            'adaptive_cropped',  # アダプティブクロップ
            'character_extracted'  # キャラクター抽出
        ]
        
        # 除外対象: 分析チャート・グラフは許可
        allowed_patterns = [
            'radar_chart',
            'bar_chart', 
            'dashboard',  # quality_dashboard.pngを含む
            'metrics',    # metrics_analysis.pngを含む
            'quality',    # quality_dashboard.pngを含む
            'analysis',   # metrics_analysis.pngを含む
            'comparison',
            'distribution',
            'priority',
            'chart'       # 一般的なチャート画像
        ]
        
        # 許可パターンに該当する場合は許可（ダッシュボードディレクトリ内の画像も含む）
        if (any(pattern in path_lower for pattern in allowed_patterns) or 
            '/dashboard/' in path_lower):
            return False
        
        # 抽出関連パターンに該当する場合は拒否
        return any(indicator in path_lower for indicator in extracted_indicators)
    
    def _mask_filename(self, filename):
        """ファイル名部分マスキング"""
        # 抽出ファイルのマスキング
        if any(keyword in filename.lower() for keyword in ['extracted', 'cropped', 'segment']):
            # kana08_0001_extracted.jpg → kana08_***_extracted.jpg
            pattern = r'(_\d{4,}_)'
            masked = re.sub(pattern, '_***_', filename)
            return masked
        return filename
    
    def _apply_filename_masking(self, html_content):
        """HTMLコンテンツ全体にファイル名マスキングを適用"""
        # 抽出ファイル名パターンを検索してマスキング
        patterns = [
            r'(kana\d+_\d{4,}_(?:extracted|cropped|segment)[^"\s<>]*)',  # kana08_0001_extracted.jpg等
            r'(p1_b\d+_kana\d+_\d{4,}_[^"\s<>]*)',  # p1_b004_kana08_0001_adaptive_cropped.png等
            r'([a-zA-Z0-9_]+_\d{4,}_(?:extracted|cropped|segment|adaptive)[^"\s<>]*)',  # その他の抽出ファイル
        ]
        
        for pattern in patterns:
            def mask_match(match):
                filename = match.group(1)
                return self._mask_filename(filename)
            
            html_content = re.sub(pattern, mask_match, html_content)
        
        return html_content
    
    def _generate_navigation_wrapper(self, title, current_key):
        """ナビゲーション付きHTMLラッパー生成"""
        # 現在のダッシュボードのパスを取得
        dashboard_path = ""
        if current_key in self.dashboards:
            # ワークスペース相対パスを取得
            dashboard_file = self.dashboards[current_key]
            dashboard_path = str(dashboard_file.relative_to(self.tracker_workspace))
        
        # ナビゲーションメニュー生成
        nav_items = []
        
        # メインダッシュボード
        if 'main' in self.dashboards:
            active = 'active' if current_key == 'main' else ''
            nav_items.append(f'<a href="/" class="nav-item {active}">🏠 メインダッシュボード</a>')
        
        # トラッカー別にグループ化
        trackers = {}
        for key in sorted(self.dashboards.keys()):
            if key == 'main':
                continue
            parts = key.split('/')
            tracker = parts[0]
            if tracker not in trackers:
                trackers[tracker] = []
            trackers[tracker].append(key)
        
        # Phase別に整理
        phase1 = sorted([t for t in trackers.keys() if t.startswith('P1-')])
        phase2 = sorted([t for t in trackers.keys() if t.startswith('PH2-')])
        phase3 = sorted([t for t in trackers.keys() if t.startswith('PH3-')])
        integrate = sorted([t for t in trackers.keys() if t.startswith('INTEGRATE-')])
        others = sorted([t for t in trackers.keys() if not t.startswith(('P1-', 'PH2-', 'PH3-', 'INTEGRATE-'))])
        
        # ナビゲーションアイテム生成
        # INTEGRATE-3-6シリーズを最上位に配置（新規追加、重要度高）
        if integrate:
            nav_items.append('<div class="nav-category">🔬 統合パイプライン</div>')
            for tracker in integrate:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                # INTEGRATE-3-6シリーズの説明を追加
                tracker_desc = {
                    'INTEGRATE-3-6-01': 'Phase 3-6統合初期版',
                    'INTEGRATE-3-6-02': 'Phase 3-6改良版',
                    'INTEGRATE-3-6-03': 'YOLO汎用版検証',
                    'INTEGRATE-3-6-04': 'アニメ特化版検証'
                }.get(tracker, tracker)
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}" title="{tracker_desc}">{tracker}</a>')
        
        if phase1:
            nav_items.append('<div class="nav-category">📁 Phase 1</div>')
            for tracker in phase1:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        if phase2:
            nav_items.append('<div class="nav-category">📁 Phase 2</div>')
            for tracker in phase2:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        if phase3:
            nav_items.append('<div class="nav-category">📁 Phase 3</div>')
            for tracker in phase3:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        if others:
            nav_items.append('<div class="nav-category">📁 その他</div>')
            for tracker in others:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        # HTMLテンプレート
        html_template = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - 統合ダッシュボードサーバー</title>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        .sidebar {{
            width: 250px;
            background: #2c3e50;
            color: white;
            overflow-y: auto;
            flex-shrink: 0;
        }}
        .sidebar-header {{
            padding: 20px;
            background: #34495e;
            border-bottom: 1px solid #1a252f;
        }}
        .sidebar-header h2 {{
            margin: 0;
            font-size: 1.2em;
            font-weight: 300;
        }}
        .nav-category {{
            padding: 15px 20px 5px 20px;
            font-size: 0.85em;
            color: #95a5a6;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}
        .nav-item {{
            display: block;
            padding: 12px 20px;
            color: #ecf0f1;
            text-decoration: none;
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
        }}
        .nav-item:hover {{
            background: #34495e;
            border-left-color: #3498db;
        }}
        .nav-item.active {{
            background: #3498db;
            border-left-color: #2980b9;
        }}
        .main-content {{
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        .content-header {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 50px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .content-title {{
            font-size: 1.3em;
            color: #2c3e50;
            margin: 0;
        }}
        .refresh-btn {{
            margin-left: auto;
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .refresh-btn:hover {{
            background: #2980b9;
        }}
        .dashboard-frame {{
            position: absolute;
            top: 50px;
            left: 0;
            right: 0;
            bottom: 0;
            border: none;
            width: 100%;
            height: calc(100% - 50px);
        }}
        .server-info {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h2>🎯 ダッシュボード統合サーバー</h2>
            <div style="font-size: 0.85em; opacity: 0.8; margin-top: 5px;">Port: {self.port}</div>
        </div>
        <nav>
            {''.join(nav_items)}
        </nav>
    </div>
    
    <div class="main-content">
        <div class="content-header">
            <h1 class="content-title">{title}</h1>
            <button class="refresh-btn" onclick="refreshDashboards()">🔄 再スキャン</button>
        </div>
        <iframe class="dashboard-frame" src="/{dashboard_path}" sandbox="allow-scripts allow-same-origin"></iframe>
    </div>
    
    <div class="server-info">
        🌐 http://localhost:{self.port}
    </div>
    
    <script>
        async function refreshDashboards() {{
            try {{
                const response = await fetch('/refresh');
                const data = await response.json();
                if (data.status === 'success') {{
                    alert('✅ ダッシュボード再スキャン完了\\n' + data.message);
                    location.reload();
                }}
            }} catch (error) {{
                alert('❌ エラー: ' + error.message);
            }}
        }}
    </script>
</body>
</html>'''
        
        return html_template
    
    async def start(self):
        """サーバー起動"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"🚀 サーバー起動完了 http://localhost:{self.port} ({len(self.dashboards)}個のダッシュボード)")

async def main():
    """メイン実行関数"""
    server = IntegratedDashboardServer(port=8088)
    await server.start()
    
    try:
        # 永続実行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 サーバー停止")

if __name__ == "__main__":
    asyncio.run(main())