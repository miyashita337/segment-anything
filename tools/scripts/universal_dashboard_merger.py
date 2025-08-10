#!/usr/bin/env python3
"""
汎用的な統合ダッシュボード生成システム
複数作者・複数ワークスペースの結果を統合表示
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml

class UniversalDashboardMerger:
    """汎用統合ダッシュボード生成クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（未指定時はデフォルト設定使用）
        """
        self.config = self._load_config(config_path)
        self.base_workspace_path = Path(self.config['base_workspace_path'])
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # デフォルト設定
        return {
            'base_workspace_path': '/mnt/c/AItools/lora/train',
            'authors': ['yado', 'kiri', 'zundamon'],
            'quality_thresholds': {
                'high_size_kb': 100,
                'medium_size_kb': 50
            },
            'dashboard_server': {
                'host': '100.123.241.106',
                'port': 8088,
                'auth_required': True
            },
            'file_extensions': ['.jpg', '.jpeg', '.png', '.webp'],
            'integration': {
                'primary_author': 'yado',  # 統合先の基準作者
                'copy_to_primary': True,   # 他作者の画像を基準作者ワークスペースにコピー
            }
        }
    
    def merge_tracker_dashboard(self, tracker_id: str) -> str:
        """
        指定されたトラッカーIDの統合ダッシュボード生成
        
        Args:
            tracker_id: トラッカーID（例: QCA-001）
            
        Returns:
            生成されたダッシュボードのパス
        """
        print(f"🔄 {tracker_id} 統合ダッシュボード生成開始...")
        
        # 各作者のワークスペース検索
        author_workspaces = self._find_author_workspaces(tracker_id)
        primary_workspace = author_workspaces.get(self.config['integration']['primary_author'])
        
        if not primary_workspace:
            raise ValueError(f"Primary author '{self.config['integration']['primary_author']}' workspace not found for {tracker_id}")
        
        # 画像統合処理
        all_images = self._collect_and_merge_images(tracker_id, author_workspaces, primary_workspace)
        
        # ダッシュボード生成
        dashboard_path = self._generate_dashboard(tracker_id, all_images, primary_workspace)
        
        print(f"✅ 統合ダッシュボード生成完了:")
        print(f"   - パス: {dashboard_path}")
        print(f"   - サイズ: {dashboard_path.stat().st_size:,} bytes")
        print(f"   - URL: {self._get_dashboard_url(tracker_id)}")
        
        return str(dashboard_path)
    
    def _find_author_workspaces(self, tracker_id: str) -> Dict[str, Path]:
        """各作者のワークスペースを検索"""
        workspaces = {}
        
        for author in self.config['authors']:
            workspace_path = self.base_workspace_path / author / "tracker-workspace" / tracker_id
            if workspace_path.exists():
                workspaces[author] = workspace_path
                print(f"✅ {author}作者ワークスペース発見: {workspace_path}")
        
        return workspaces
    
    def _collect_and_merge_images(self, tracker_id: str, author_workspaces: Dict[str, Path], 
                                 primary_workspace: Path) -> List[Dict[str, Any]]:
        """画像収集・統合処理"""
        all_images = []
        primary_extraction_dir = primary_workspace / "extraction"
        primary_extraction_dir.mkdir(exist_ok=True)
        
        for author, workspace in author_workspaces.items():
            extraction_dir = workspace / "extraction"
            if not extraction_dir.exists():
                continue
                
            for ext in self.config['file_extensions']:
                for img_file in extraction_dir.glob(f"*{ext}"):
                    if author == self.config['integration']['primary_author']:
                        # 基準作者はそのまま使用
                        final_path = img_file
                    elif self.config['integration']['copy_to_primary']:
                        # 他作者の画像を基準作者ワークスペースにコピー
                        dest_name = f"{author}_{img_file.name}"
                        final_path = primary_extraction_dir / dest_name
                        
                        # ファイルが存在しない場合のみコピー
                        if not final_path.exists():
                            shutil.copy2(img_file, final_path)
                            print(f"📋 {author}画像コピー: {img_file.name} → {dest_name}")
                    else:
                        # コピーしない場合はそのまま参照
                        final_path = img_file
                    
                    all_images.append({
                        'path': str(final_path),
                        'name': final_path.name,
                        'size': final_path.stat().st_size,
                        'author': author,
                        'quality': self._determine_quality(final_path.stat().st_size),
                        'original_name': img_file.name
                    })
        
        print(f"✅ 統合対象画像: {len(all_images)}枚")
        for author in self.config['authors']:
            count = len([img for img in all_images if img['author'] == author])
            if count > 0:
                print(f"   - {author}作者: {count}枚")
        
        return all_images
    
    def _determine_quality(self, size_bytes: int) -> str:
        """画像品質判定"""
        size_kb = size_bytes / 1024
        
        if size_kb >= self.config['quality_thresholds']['high_size_kb']:
            return 'high'
        elif size_kb >= self.config['quality_thresholds']['medium_size_kb']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_dashboard(self, tracker_id: str, all_images: List[Dict[str, Any]], 
                           primary_workspace: Path) -> Path:
        """ダッシュボードHTML生成"""
        
        # 品質分析
        quality_counts = {'high': 0, 'medium': 0, 'low': 0}
        for img in all_images:
            quality_counts[img['quality']] += 1
        
        quality_score = (quality_counts['high'] + quality_counts['medium'] * 0.5) / len(all_images) * 100 if all_images else 0
        
        # 作者別統計
        author_stats = {}
        for author in self.config['authors']:
            author_images = [img for img in all_images if img['author'] == author]
            if author_images:
                author_stats[author] = {
                    'count': len(author_images),
                    'profile': self._get_author_profile(author)
                }
        
        # HTMLコンテンツ生成
        dashboard_content = self._generate_dashboard_html(
            tracker_id, all_images, quality_counts, quality_score, author_stats
        )
        
        # ダッシュボード保存
        dashboard_path = primary_workspace / "dashboard" / "dashboard.html"
        dashboard_path.parent.mkdir(exist_ok=True)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_content)
        
        return dashboard_path
    
    def _get_author_profile(self, author: str) -> str:
        """作者プロファイル説明取得"""
        profiles = {
            'yado': 'バランス型・キャラクター重視',
            'kiri': '細密描写特化・高品質重視（元aichi）',
            'zundamon': 'シンプルスタイル・効率重視'
        }
        return profiles.get(author, '未定義プロファイル')
    
    def _generate_dashboard_html(self, tracker_id: str, all_images: List[Dict[str, Any]],
                                quality_counts: Dict[str, int], quality_score: float,
                                author_stats: Dict[str, Dict]) -> str:
        """ダッシュボードHTML生成"""
        
        # 作者別ギャラリー生成
        gallery_html = ""
        
        for author, stats in author_stats.items():
            author_images = [img for img in all_images if img['author'] == author]
            if not author_images:
                continue
                
            author_icon = {'yado': '👤', 'kiri': '🎨', 'zundamon': '⚡'}.get(author, '🔧')
            
            gallery_html += f'<div class="author-section"><h3>{author_icon} {author}作者（{stats["profile"]}）</h3><div class="images-grid">'
            
            for img in author_images:
                quality_class = img['quality']
                quality_label = {'high': '高品質', 'medium': '中品質', 'low': '低品質'}[quality_class]
                size_kb = img['size'] // 1024
                
                # 統合ダッシュボード用相対パス生成
                relative_path = self._get_relative_path(img['path'], tracker_id)
                
                gallery_html += f'''
        <div class="image-card">
            <div class="image-container">
                <img src="{relative_path}" alt="{img['name']}" loading="lazy">
                <div class="quality-badge {quality_class}">{quality_label}</div>
            </div>
            <div class="image-info">
                <div class="image-name">{img['name']}</div>
                <div class="image-details">
                    <span>{size_kb} KB</span>
                    <span>{quality_label}</span>
                </div>
            </div>
        </div>'''
            
            gallery_html += '</div></div>'
        
        # 作者統計HTML
        author_stats_html = ""
        for author, stats in author_stats.items():
            author_icon = {'yado': '👤', 'kiri': '🎨', 'zundamon': '⚡'}.get(author, '🔧')
            author_stats_html += f'''
            <div class="author-stat">
                <div class="author-stat-value">{stats["count"]}</div>
                <div class="author-stat-label">{author_icon} {author}作者</div>
            </div>'''
        
        # HTMLテンプレート
        return f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} 統合 - 作者別パラメータ適応システム検証</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.15); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 3em; font-weight: 300; }}
        .header .subtitle {{ font-size: 1.3em; opacity: 0.9; margin-top: 15px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 40px; background: #f8f9fa; }}
        .stat-card {{ background: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
        .stat-label {{ color: #666; font-size: 1.1em; }}
        .success-rate {{ color: #27ae60; }}
        .quality-summary {{ padding: 40px; background: white; }}
        .quality-summary h2 {{ text-align: center; margin-bottom: 30px; color: #2c3e50; font-size: 2em; }}
        .quality-chart {{ display: flex; gap: 20px; margin-bottom: 40px; }}
        .quality-stat {{ flex: 1; text-align: center; padding: 20px; border-radius: 15px; color: white; }}
        .quality-stat.high {{ background: linear-gradient(135deg, #27ae60, #2ecc71); }}
        .quality-stat.medium {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
        .quality-stat.low {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
        .quality-count {{ font-size: 2.5em; font-weight: bold; }}
        .quality-label {{ font-size: 1.2em; margin-top: 10px; }}
        .gallery {{ padding: 40px; background: #f8f9fa; }}
        .gallery h2 {{ text-align: center; margin-bottom: 30px; color: #2c3e50; font-size: 2em; }}
        .author-section {{ margin-bottom: 40px; }}
        .author-section h3 {{ color: #2c3e50; font-size: 1.5em; margin-bottom: 20px; padding-left: 10px; border-left: 4px solid #3498db; }}
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 18px; margin-bottom: 30px; }}
        .image-card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .image-container {{ position: relative; min-height: 200px; overflow: visible; }}
        .image-container img {{ width: 50%; height: 50%; object-fit: contain; background: #f8f9fa; max-width: 50%; max-height: 50%; display: block; margin: 15px auto; }}
        .quality-badge {{ position: absolute; top: 10px; right: 10px; padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.8em; }}
        .quality-badge.high {{ background: #27ae60; }}
        .quality-badge.medium {{ background: #f39c12; }}
        .quality-badge.low {{ background: #e74c3c; }}
        .image-info {{ padding: 15px; }}
        .image-name {{ font-weight: bold; margin-bottom: 5px; color: #2c3e50; }}
        .image-details {{ display: flex; justify-content: space-between; color: #666; font-size: 0.9em; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 30px; }}
        .generation-info {{ font-size: 0.9em; opacity: 0.8; margin-top: 10px; }}
        .author-stats {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
        .author-stat {{ background: white; padding: 15px 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .author-stat-value {{ font-size: 1.8em; font-weight: bold; color: #3498db; }}
        .author-stat-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 {tracker_id} 統合ダッシュボード</h1>
            <div class="subtitle">作者別パラメータ適応システム検証結果</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(all_images)}</div>
                <div class="stat-label">総画像数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success-rate">{quality_score:.1f}%</div>
                <div class="stat-label">品質スコア</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{quality_counts["high"] + quality_counts["medium"]}</div>
                <div class="stat-label">高・中品質</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{quality_counts["low"]}</div>
                <div class="stat-label">要改善</div>
            </div>
        </div>
        
        <div class="author-stats">{author_stats_html}
        </div>

        <div class="quality-summary">
            <h2>📊 品質分析</h2>
            <div class="quality-chart">
                <div class="quality-stat high">
                    <div class="quality-count">{quality_counts["high"]}</div>
                    <div class="quality-label">高品質</div>
                </div>
                <div class="quality-stat medium">
                    <div class="quality-count">{quality_counts["medium"]}</div>
                    <div class="quality-label">中品質</div>
                </div>
                <div class="quality-stat low">
                    <div class="quality-count">{quality_counts["low"]}</div>
                    <div class="quality-label">低品質</div>
                </div>
            </div>
        </div>

        <div class="gallery">
            <h2>🎨 作者別抽出画像ギャラリー</h2>
            {gallery_html}
        </div>

        <div class="footer">
            <p>🤖 {tracker_id}: 作者別パラメータ適応システム統合結果</p>
            <div class="generation-info">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                URL: <a href="{self._get_dashboard_url(tracker_id)}" style="color: #3498db;">{self._get_dashboard_url(tracker_id)}</a>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    def _get_relative_path(self, image_path: str, tracker_id: str) -> str:
        """統合ダッシュボード用相対パス生成"""
        # primary author のワークスペースパス基準で相対パス生成
        primary_workspace_path = self.base_workspace_path / self.config['integration']['primary_author'] / "tracker-workspace"
        relative_path = Path(image_path).relative_to(primary_workspace_path)
        return f"/{relative_path.as_posix()}"
    
    def _get_dashboard_url(self, tracker_id: str) -> str:
        """ダッシュボードURL生成"""
        host = self.config['dashboard_server']['host']
        port = self.config['dashboard_server']['port']
        return f"http://{host}:{port}/tracker/{tracker_id}"

def main():
    """メイン実行関数"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python universal_dashboard_merger.py <TRACKER_ID> [CONFIG_PATH]")
        sys.exit(1)
    
    tracker_id = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    merger = UniversalDashboardMerger(config_path)
    dashboard_path = merger.merge_tracker_dashboard(tracker_id)
    
    print(f"🎯 統合ダッシュボード生成完了: {dashboard_path}")

if __name__ == "__main__":
    main()