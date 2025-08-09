#!/usr/bin/env python3
"""
標準ダッシュボード生成システム
統一されたBase64画像表示 + 品質評価バッジ機能

Created for: ダッシュボード生成ワークフロー標準化
Author: Claude Code Integration System
"""

import os
import sys
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandardDashboardGenerator:
    """標準ダッシュボード生成クラス"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.logger = logger
    
    @staticmethod
    def get_image_quality(file_size: int) -> Tuple[str, str]:
        """ファイルサイズに基づいて品質評価を返す"""
        if file_size > 100000:  # 100KB以上
            return "high", "高品質"
        elif file_size > 50000:  # 50KB以上
            return "medium", "中品質"
        else:
            return "low", "低品質"
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """画像をBase64エンコードして返す"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64エンコードエラー {image_path}: {e}")
            return ""
    
    def collect_images(self, extraction_dir: str) -> List[str]:
        """抽出ディレクトリから画像ファイルを収集"""
        if not os.path.exists(extraction_dir):
            self.logger.error(f"抽出ディレクトリが見つかりません: {extraction_dir}")
            return []
        
        images = []
        for file in os.listdir(extraction_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append(os.path.join(extraction_dir, file))
        
        images.sort()
        self.logger.info(f"📷 {len(images)}個の画像を検出: {self.tracker_id}")
        return images
    
    def calculate_quality_stats(self, images: List[str]) -> Dict[str, int]:
        """品質統計を計算"""
        stats = {"high": 0, "medium": 0, "low": 0}
        
        for image_path in images:
            try:
                file_size = os.path.getsize(image_path)
                quality, _ = self.get_image_quality(file_size)
                stats[quality] += 1
            except Exception:
                stats["low"] += 1  # エラー時は低品質として扱う
        
        return stats
    
    def generate_image_cards_html(self, images: List[str]) -> str:
        """画像カードHTMLを生成（直接画像パス使用・トリミングなし）"""
        if not images:
            return '<div class="no-images">抽出された画像が見つかりませんでした</div>'
        
        image_cards = ""
        for image_path in images:
            try:
                filename = os.path.basename(image_path)
                file_size = os.path.getsize(image_path)
                quality, quality_label = self.get_image_quality(file_size)
                
                # ワークスペース相対パスを生成（統合サーバーでアクセス可能）
                workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
                if workspace_base in image_path:
                    relative_path = image_path.replace(workspace_base + "/", "")
                else:
                    relative_path = image_path
                
                self.logger.debug(f"  🖼️  {filename}: 直接画像パス使用")
                
                image_cards += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="/{relative_path}" alt="{filename}" loading="lazy">
                <div class="quality-badge {quality}">{quality_label}</div>
            </div>
            <div class="image-info">
                <div class="image-name">{filename}</div>
                <div class="image-details">
                    <span>{file_size // 1024} KB</span>
                    <span>{quality_label}</span>
                </div>
            </div>
        </div>"""
            except Exception as e:
                self.logger.error(f"画像処理エラー {image_path}: {e}")
                continue
        
        return image_cards
    
    def get_dashboard_template(self) -> str:
        """統一ダッシュボードHTMLテンプレートを返す"""
        return """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} - 品質評価ダッシュボード</title>
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
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 18px; }}
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
        .no-images {{ text-align: center; padding: 60px; color: #666; font-size: 1.2em; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 30px; }}
        .generation-info {{ font-size: 0.9em; opacity: 0.8; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 {tracker_id}</h1>
            <div class="subtitle">品質評価ダッシュボード</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_images}</div>
                <div class="stat-label">総画像数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success-rate">{success_rate:.1f}%</div>
                <div class="stat-label">品質スコア</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_count}</div>
                <div class="stat-label">成功画像</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{low_count}</div>
                <div class="stat-label">要改善</div>
            </div>
        </div>

        <div class="quality-summary">
            <h2>📊 品質分析</h2>
            <div class="quality-chart">
                <div class="quality-stat high">
                    <div class="quality-count">{high_count}</div>
                    <div class="quality-label">高品質</div>
                </div>
                <div class="quality-stat medium">
                    <div class="quality-count">{medium_count}</div>
                    <div class="quality-label">中品質</div>
                </div>
                <div class="quality-stat low">
                    <div class="quality-count">{low_count}</div>
                    <div class="quality-label">低品質</div>
                </div>
            </div>
        </div>

        <div class="gallery">
            <h2>🎨 抽出画像ギャラリー</h2>
            <div class="images-grid">{image_cards}
            </div>
        </div>

        <div class="footer">
            <p>🤖 Generated by SAM+YOLO Character Extraction Pipeline</p>
            <div class="generation-info">
                Generated: {generation_time} | 
                URL: <a href="http://100.123.241.106:8088/tracker/{tracker_id}" style="color: #3498db;">http://100.123.241.106:8088/tracker/{tracker_id}</a>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    def generate_dashboard(self, extraction_dir: str, output_path: str) -> bool:
        """統一ダッシュボードを生成"""
        self.logger.info(f"🔄 {self.tracker_id}: 標準ダッシュボード生成開始")
        
        # 画像収集
        images = self.collect_images(extraction_dir)
        if not images:
            self.logger.warning(f"⚠️ {self.tracker_id}: 画像が見つかりません")
            # 空のダッシュボードも生成
        
        # 統計計算
        quality_stats = self.calculate_quality_stats(images)
        total_images = len(images)
        success_count = quality_stats["high"] + quality_stats["medium"]
        success_rate = (success_count / total_images * 100) if total_images > 0 else 0
        
        # 画像カードHTML生成
        image_cards = self.generate_image_cards_html(images)
        
        # HTMLコンテンツ生成
        html_content = self.get_dashboard_template().format(
            tracker_id=self.tracker_id,
            total_images=total_images,
            success_rate=success_rate,
            success_count=success_count,
            high_count=quality_stats["high"],
            medium_count=quality_stats["medium"],
            low_count=quality_stats["low"],
            image_cards=image_cards,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # ダッシュボードディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # HTMLファイル書き込み
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # ファイルサイズ確認
            file_size = os.path.getsize(output_path)
            self.logger.info(f"✅ {self.tracker_id}: ダッシュボード生成完了")
            self.logger.info(f"  📄 ファイルサイズ: {file_size:,} バイト ({file_size/1024/1024:.1f}MB)")
            self.logger.info(f"  🌐 URL: http://100.123.241.106:8088/tracker/{self.tracker_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.tracker_id}: ダッシュボード生成エラー: {e}")
            return False


def create_standard_dashboard(tracker_id: str, extraction_dir: str, output_dir: str) -> bool:
    """標準ダッシュボード生成のエントリーポイント"""
    generator = StandardDashboardGenerator(tracker_id)
    output_path = os.path.join(output_dir, "dashboard", "dashboard.html")
    return generator.generate_dashboard(extraction_dir, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="標準ダッシュボード生成")
    parser.add_argument("tracker_id", help="トラッカーID")
    parser.add_argument("extraction_dir", help="抽出ディレクトリパス")
    parser.add_argument("output_dir", help="出力ディレクトリパス")
    
    args = parser.parse_args()
    
    success = create_standard_dashboard(args.tracker_id, args.extraction_dir, args.output_dir)
    sys.exit(0 if success else 1)