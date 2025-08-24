from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
QI-003とQI-004のダッシュボードを完全なBase64画像データで更新
"""

import base64
import os
from typing import List, Tuple


def get_image_quality(file_size: int) -> Tuple[str, str]:
    """ファイルサイズに基づいて品質評価を返す"""
    if file_size > 100000:  # 100KB以上
        return "high", "高品質"
    elif file_size > 50000:  # 50KB以上
        return "medium", "中品質"
    else:
        return "low", "低品質"

def image_to_base64(image_path: str) -> str:
    """画像をBase64エンコードして返す"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return ""

def create_dashboard_html(tracker_id: str, images: List[str]) -> str:
    """ダッシュボードHTMLを生成"""
    
    # 品質統計計算
    high_count = medium_count = low_count = 0
    
    for image_path in images:
        try:
            file_size = os.path.getsize(image_path)
            quality, _ = get_image_quality(file_size)
            
            if quality == "high":
                high_count += 1
            elif quality == "medium":
                medium_count += 1
            else:
                low_count += 1
        except:
            low_count += 1  # エラー時は低品質として扱う

    total_images = len(images)
    success_rate = (high_count + medium_count) / total_images * 100 if total_images > 0 else 0

    # 画像カードHTML生成
    image_cards = ""
    for image_path in images:
        try:
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path)
            quality, quality_label = get_image_quality(file_size)
            
            base64_data = image_to_base64(image_path)
            if not base64_data:
                continue

            image_cards += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="data:image/jpeg;base64,{base64_data}" alt="{filename}">
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
            print(f"Error processing {image_path}: {e}")
            continue

    if not image_cards:
        image_cards = '<div class="no-images">抽出された画像が見つかりませんでした</div>'

    return f"""<!DOCTYPE html>
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
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .image-container {{ position: relative; height: 200px; overflow: hidden; }}
        .image-container img {{ width: 100%; height: 100%; object-fit: contain; background: #f8f9fa; }}
        .quality-badge {{ position: absolute; top: 10px; right: 10px; padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.8em; }}
        .quality-badge.high {{ background: #27ae60; }}
        .quality-badge.medium {{ background: #f39c12; }}
        .quality-badge.low {{ background: #e74c3c; }}
        .image-info {{ padding: 15px; }}
        .image-name {{ font-weight: bold; margin-bottom: 5px; color: #2c3e50; }}
        .image-details {{ display: flex; justify-content: space-between; color: #666; font-size: 0.9em; }}
        .no-images {{ text-align: center; padding: 60px; color: #666; font-size: 1.2em; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 {tracker_id}</h1>
            <div class="subtitle">品質改善・画像評価システム</div>
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
                <div class="stat-value">{high_count + medium_count}</div>
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
            <p>🤖 Generated by SAM+YOLO Character Extraction Pipeline | Claude Code Integration</p>
        </div>
    </div>
</body>
</html>"""

def main():
    """メイン処理"""
    trackers = ['QI-003', 'QI-004']
    
    for tracker_id in trackers:
        extraction_dir = fget_path("output", "{tracker_id}/extraction")
        dashboard_path = fget_path("output", "{tracker_id}/dashboard/dashboard.html")
        
        print(f"Processing {tracker_id}...")
        
        if not os.path.exists(extraction_dir):
            print(f"❌ {tracker_id}: 抽出ディレクトリが見つかりません: {extraction_dir}")
            continue
        
        # 画像ファイル収集
        images = []
        for file in os.listdir(extraction_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(extraction_dir, file))
        
        images.sort()
        print(f"  🖼️  {tracker_id}: {len(images)}個の画像を検出")
        
        if not images:
            print(f"❌ {tracker_id}: 画像が見つかりません")
            continue
        
        # ダッシュボード生成
        html_content = create_dashboard_html(tracker_id, images)
        
        # ダッシュボードディレクトリ作成
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        
        # HTMLファイル書き込み
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ {tracker_id}: ダッシュボード更新完了: {dashboard_path}")

if __name__ == "__main__":
    main()