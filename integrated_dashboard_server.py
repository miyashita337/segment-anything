#!/usr/bin/env python3
"""
統合ダッシュボードサーバー
全てのトラッカーダッシュボードを統合管理

http://localhost:8088/ でアクセス可能

QUAL-036: 作者別振り分け対応 - ハードコーディング除去・動的作者検出システム統合
"""

import aiohttp_cors
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
from aiohttp import web
from datetime import datetime
from pathlib import Path
from typing import Optional

# WorkspaceConfig統合 - 動的ワークスペース管理
from config.workspace_config import WorkspaceConfig

# QUAL-036: 作者別パラメータ適応システム統合
try:
    from features.adaptation.author_parameter_adapter import AuthorParameterAdapter
    AUTHOR_ADAPTER_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ AuthorParameterAdapter not available - using default workspace")
    AUTHOR_ADAPTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# QUAL-036: ハードコーディング除去 - 動的ワークスペース生成
def get_tracker_workspace(input_path: Optional[str] = None) -> Path:
    """
    作者を検出してワークスペースパスを動的生成
    
    Args:
        input_path: 入力パス（作者検出用）
        
    Returns:
        作者別ワークスペースパス
    """
    try:
        # 環境変数からの取得を優先
        if 'TRACKER_WORKSPACE_BASE' in os.environ:
            return Path(os.environ['TRACKER_WORKSPACE_BASE'])
        
        # 入力パスから作者検出
        if input_path and AUTHOR_ADAPTER_AVAILABLE:
            detected_author = AuthorParameterAdapter.detect_author_from_path(input_path)
            if detected_author:
                workspace = Path(f"/mnt/c/AItools/lora/train/{detected_author}/tracker-workspace")
                logger.info(f"🔍 作者検出成功: {detected_author} → {workspace}")
                return workspace
        
        # フォールバック: デフォルト作者（yado）
        default_workspace = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        logger.info(f"📂 デフォルトワークスペース使用: {default_workspace}")
        return default_workspace
        
    except Exception as e:
        logger.error(f"❌ ワークスペース検出エラー: {e}")
        return Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")

# Basic認証設定
BASIC_AUTH_USERNAME = "admin"
BASIC_AUTH_PASSWORD = "secure_track_2025_q3_8f9a"

class IntegratedDashboardServer:
    def __init__(self, port=8088, input_path=None):
        """
        統合ダッシュボードサーバー初期化
        
        Args:
            port: サーバーポート
            input_path: 入力パス（作者検出用）
        """
        self.port = port
        self.input_path = input_path
        self.app = web.Application()  # ミドルウェアを後で追加
        
        # QUAL-036: 動的ワークスペース設定
        self.tracker_workspace = get_tracker_workspace(input_path)
        self.author_name = self._detect_author_name()
        
        self._setup_routes()
        self._setup_cors()
        self._scan_dashboards()
        
        # ミドルウェアを最後に追加
        self.app.middlewares.append(self.auth_middleware)
        
        # QUAL-036: 作者情報ログ出力
        logger.info(f"🎯 作者別ダッシュボード初期化完了")
        logger.info(f"   作者: {self.author_name}")
        logger.info(f"   ワークスペース: {self.tracker_workspace}")
        logger.info(f"   入力パス: {self.input_path}")
    
    def _detect_author_name(self) -> str:
        """現在の作者名を取得"""
        if self.input_path and AUTHOR_ADAPTER_AVAILABLE:
            detected = AuthorParameterAdapter.detect_author_from_path(self.input_path)
            if detected:
                return detected
        
        # ワークスペースパスから作者名を逆算
        workspace_parts = self.tracker_workspace.parts
        if 'train' in workspace_parts:
            train_idx = workspace_parts.index('train')
            if train_idx + 1 < len(workspace_parts):
                return workspace_parts[train_idx + 1]
        
        return "yado"  # デフォルト
    
    def _scan_dashboards(self):
        """利用可能なダッシュボードを動的にスキャン（WorkspaceConfig活用）"""
        self.dashboards = {}
        
        # WorkspaceConfigを使用して動的にワークスペースを検出
        base_path = Path(WorkspaceConfig.BASE_TRAIN_PATH)
        
        # 存在する全作者ディレクトリを動的に検出
        if base_path.exists():
            for author_dir in base_path.iterdir():
                if author_dir.is_dir():
                    # WorkspaceConfigの関数を使ってワークスペースパス生成
                    workspace = Path(WorkspaceConfig.get_workspace_base_for_author(author_dir.name))
                    
                    if not workspace.exists():
                        continue
                    
                    logger.info(f"📂 スキャン中: {author_dir.name}のワークスペース - {workspace}")
                    
                    # メインダッシュボード
                    main_dashboard = workspace / "main_dashboard.html"
                    if main_dashboard.exists():
                        self.dashboards[f'{author_dir.name}_main'] = main_dashboard
                    
                    # 各トラッカーのダッシュボード
                    for tracker_dir in workspace.iterdir():
                        if tracker_dir.is_dir():
                            # qualityディレクトリ内も確認（KIRO-002用）
                            quality_dashboard_dir = tracker_dir / "quality" / "dashboard"
                            if quality_dashboard_dir.exists():
                                for html_file in quality_dashboard_dir.glob("*.html"):
                                    tracker_id = tracker_dir.name
                                    dashboard_key = f"{tracker_id}/{html_file.stem}"
                                    self.dashboards[dashboard_key] = html_file
                            
                            # 通常のdashboardディレクトリ
                            dashboard_dir = tracker_dir / "dashboard"
                            if dashboard_dir.exists():
                                for html_file in dashboard_dir.glob("*.html"):
                                    tracker_id = tracker_dir.name
                                    dashboard_key = f"{tracker_id}/{html_file.stem}"
                                    self.dashboards[dashboard_key] = html_file
        
        # 環境変数から追加のワークスペースも確認
        if 'TRACKER_WORKSPACE_BASE' in os.environ:
            additional_workspace = Path(os.environ['TRACKER_WORKSPACE_BASE'])
            if additional_workspace.exists() and additional_workspace not in [
                Path(WorkspaceConfig.get_workspace_base_for_author(d.name)) 
                for d in base_path.iterdir() if d.is_dir()
            ]:
                logger.info(f"📂 環境変数から追加ワークスペース: {additional_workspace}")
                # 同様のスキャン処理を追加ワークスペースに対しても実行
                for tracker_dir in additional_workspace.iterdir():
                    if tracker_dir.is_dir():
                        for pattern in ["dashboard"]:
                            dashboard_dir = tracker_dir / pattern
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
        self.app.router.add_get('/tracker/{tracker_id}/', self.handle_tracker)  # 末尾スラッシュ対応
        self.app.router.add_get('/tracker/{tracker_id}/{dashboard_name}', self.handle_tracker_dashboard)
        self.app.router.add_get('/api/dashboards', self.handle_api_dashboards)
        self.app.router.add_get('/dashboard-list', self.handle_dashboard_list_html)  # ページ一覧HTML表示用
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
        """静的ファイル配信（複数ワークスペース対応）"""
        path = request.match_info['path']
        
        # セキュリティチェック
        if '..' in path:
            return web.Response(text="Forbidden", status=403)
        
        # WorkspaceConfigを使用して複数のワークスペースから探す
        base_path = Path(WorkspaceConfig.BASE_TRAIN_PATH)
        
        # まず動的に全作者のワークスペースから探す
        for author_dir in base_path.iterdir():
            if author_dir.is_dir():
                workspace = Path(WorkspaceConfig.get_workspace_base_for_author(author_dir.name))
                file_path = workspace / path
                
                if file_path.exists() and file_path.is_file():
                    # ファイルが見つかった場合は配信処理へ
                    break
        else:
            # 環境変数のワークスペースも確認
            if 'TRACKER_WORKSPACE_BASE' in os.environ:
                file_path = Path(os.environ['TRACKER_WORKSPACE_BASE']) / path
                if not (file_path.exists() and file_path.is_file()):
                    # デフォルトワークスペースも確認
                    file_path = self.tracker_workspace / path
            else:
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
            # ファイルの実際のワークスペースを特定（動的作者検出）
            actual_workspace = self._get_workspace_for_dashboard(path)
            dashboard_list.append({
                'key': key,
                'path': str(path.relative_to(actual_workspace)),
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
    
    async def handle_dashboard_list_html(self, request):
        """ダッシュボード一覧HTML表示（メインダッシュボード右ペイン用）"""
        dashboard_list = []
        for key, path in self.dashboards.items():
            # ファイルの実際のワークスペースを特定（動的作者検出）
            actual_workspace = self._get_workspace_for_dashboard(path)
            dashboard_list.append({
                'key': key,
                'path': str(path.relative_to(actual_workspace)),
                'tracker': key.split('/')[0] if '/' in key else 'main',
                'name': path.stem
            })
        
        # HTML生成
        html = f'''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ダッシュボード一覧</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .stats {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .dashboard-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
        }}
        .dashboard-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .tracker-name {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        .dashboard-name {{
            color: #7f8c8d;
            margin-bottom: 12px;
        }}
        .dashboard-path {{
            font-size: 0.85em;
            color: #95a5a6;
            font-family: monospace;
            background: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .category-header {{
            grid-column: 1 / -1;
            font-size: 1.3em;
            font-weight: bold;
            color: #34495e;
            margin: 20px 0 10px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 統合ダッシュボード</h1>
        <div class="stats">全 {len(dashboard_list)} 個のダッシュボード</div>
    </div>
    
    <div class="dashboard-grid">'''
        
        # トラッカー別にグループ化して表示
        trackers = {}
        for item in sorted(dashboard_list, key=lambda x: x['key']):
            tracker = item['tracker']
            if tracker not in trackers:
                trackers[tracker] = []
            trackers[tracker].append(item)
        
        for category, items in trackers.items():
            # カテゴリーアイコン
            category_icon = {
                'QUAL': '🔍',
                'INTG': '🔬', 
                'TEST': '🧪',
                'OPTM': '⚡',
                'DEMO': '📁',
                'INCI': '📁',
                'KIRO': '📁'
            }.get(category.split('-')[0], '📁')
            
            html += f'<div class="category-header">{category_icon} {category}</div>'
            
            for item in items:
                onclick = f"parent.location.href='/tracker/{item['key'].replace('/dashboard', '')}'"
                html += f'''
        <div class="dashboard-card" onclick="{onclick}">
            <div class="tracker-name">{item['key']}</div>
            <div class="dashboard-name">{item['name']}</div>
            <div class="dashboard-path">{item['path']}</div>
        </div>'''
        
        html += '''
    </div>
</body>
</html>'''
        
        return web.Response(text=html, content_type='text/html')
    
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
        
        # INTG-087系トラッカーの場合はファイル名マスキングを無効化
        if 'INTG-087' in html_content:
            return html_content
            
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
    
    def _get_workspace_for_dashboard(self, dashboard_path):
        """ダッシュボードファイルの実際のワークスペースを特定"""
        dashboard_path = Path(dashboard_path)
        
        # kiriワークスペース判定
        kiri_workspace = Path(WorkspaceConfig.get_workspace_base_for_author('kiri'))
        if str(dashboard_path).startswith(str(kiri_workspace)):
            return kiri_workspace
        
        # yadoワークスペース判定
        yado_workspace = Path(WorkspaceConfig.get_workspace_base_for_author('yado'))
        if str(dashboard_path).startswith(str(yado_workspace)):
            return yado_workspace
        
        # その他の作者のワークスペースを動的検出
        for path_part in dashboard_path.parts:
            if 'tracker-workspace' in str(path_part):
                # tracker-workspaceの親ディレクトリから作者名を抽出
                train_path_idx = next((i for i, p in enumerate(dashboard_path.parts) if p == 'train'), None)
                if train_path_idx is not None and train_path_idx + 1 < len(dashboard_path.parts):
                    author_name = dashboard_path.parts[train_path_idx + 1]
                    return Path(WorkspaceConfig.get_workspace_base_for_author(author_name))
        
        # フォールバック: デフォルトワークスペース
        return Path(self.tracker_workspace)
    
    def _generate_navigation_wrapper(self, title, current_key):
        """ナビゲーション付きHTMLラッパー生成"""
        # 現在のダッシュボードのパスを取得
        dashboard_path = ""
        
        # メインダッシュボードの場合：ページ一覧表示（元の仕様復元）
        if current_key == "main":
            dashboard_path = "dashboard-list"  # ページ一覧HTML表示エンドポイント
        elif current_key in self.dashboards:
            # ワークスペース相対パスを取得（動的作者検出）
            dashboard_file = self.dashboards[current_key]
            
            # ファイルの実際のワークスペースを特定
            actual_workspace = self._get_workspace_for_dashboard(dashboard_file)
            dashboard_path = str(dashboard_file.relative_to(actual_workspace))
        
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
        
        # 新ID体系別に整理
        qual_trackers = sorted([t for t in trackers.keys() if t.startswith('QUAL-')])  # 品質管理
        optm_trackers = sorted([t for t in trackers.keys() if t.startswith('OPTM-')])  # 最適化
        test_trackers = sorted([t for t in trackers.keys() if t.startswith('TEST-')])  # テスト
        intg_trackers = sorted([t for t in trackers.keys() if t.startswith('INTG-')])  # 統合
        others = sorted([t for t in trackers.keys() if not t.startswith(('QUAL-', 'OPTM-', 'TEST-', 'INTG-'))])
        
        # ナビゲーションアイテム生成（新ID体系）
        # 品質管理を最上位に配置
        if qual_trackers:
            nav_items.append('<div class="nav-category">🔍 品質管理 (QUAL)</div>')
            for tracker in qual_trackers:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        # 統合パイプラインを次に配置
        if intg_trackers:
            nav_items.append('<div class="nav-category">🔬 統合パイプライン (INTG)</div>')
            for tracker in intg_trackers:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        # テスト関連
        if test_trackers:
            nav_items.append('<div class="nav-category">🧪 テスト (TEST)</div>')
            for tracker in test_trackers:
                active = 'active' if current_key.startswith(f"{tracker}/") else ''
                nav_items.append(f'<a href="/tracker/{tracker}" class="nav-item {active}">{tracker}</a>')
        
        # 最適化関連
        if optm_trackers:
            nav_items.append('<div class="nav-category">⚡ 最適化 (OPTM)</div>')
            for tracker in optm_trackers:
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
    """
    メイン実行関数
    
    QUAL-036: 入力パス引数対応
    Usage: python integrated_dashboard_server.py [input_path]
    """
    import sys
    
    # コマンドライン引数から入力パスを取得
    input_path = None
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        logger.info(f"📂 入力パス指定: {input_path}")
    
    # 環境変数からも取得可能
    if not input_path and 'INPUT_PATH' in os.environ:
        input_path = os.environ['INPUT_PATH']
        logger.info(f"📂 環境変数から入力パス取得: {input_path}")
    
    server = IntegratedDashboardServer(port=8088, input_path=input_path)
    await server.start()
    
    try:
        # 永続実行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 サーバー停止")

if __name__ == "__main__":
    asyncio.run(main())