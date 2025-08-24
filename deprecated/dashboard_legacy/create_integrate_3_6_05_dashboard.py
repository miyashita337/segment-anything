#!/usr/bin/env python3
"""
INTEGRATE-3-6-05 ダッシュボード生成スクリプト
バックグラウンド抽出中にダッシュボード準備
"""

import base64
import json
import logging
from datetime import datetime
from pathlib import Path


def create_dashboard():
    """INTEGRATE-3-6-05用ダッシュボード作成"""
    
    # ディレクトリ設定
    workspace_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-05")
    dashboard_dir = workspace_dir / "dashboard"
    extraction_dir = workspace_dir / "extraction"
    
    # ディレクトリ作成
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    # 抽出結果スキャン（処理中の場合は空でも対応）
    image_files = []
    if extraction_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(extraction_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    # 統計計算
    total_files = len(image_files)
    # 処理中の場合は暫定値
    if total_files == 0:
        success_rate = "処理中"
        status_message = "🔄 バックグラウンド抽出処理中..."
    else:
        # 簡単な品質推定（ファイルサイズベース）
        high_quality = 0
        medium_quality = 0 
        low_quality = 0
        
        for img_file in image_files:
            size_kb = img_file.stat().st_size / 1024
            if size_kb > 50:  # 50KB以上
                high_quality += 1
            elif size_kb > 10:  # 10-50KB
                medium_quality += 1
            else:  # 10KB未満
                low_quality += 1
        
        success_rate = f"{(total_files / 26 * 100):.1f}%" if total_files > 0 else "0%"
        status_message = f"✅ 抽出完了: {total_files}枚"
    
    # HTMLダッシュボード生成
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>INTEGRATE-3-6-05 - 抽出結果ダッシュボード</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 30px; background: #f8f9fa; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); text-align: center; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
        .stat-label {{ color: #666; font-size: 1.1em; }}
        .quality-high {{ color: #27ae60; }}
        .quality-medium {{ color: #f39c12; }}
        .quality-low {{ color: #e74c3c; }}
        .success-rate {{ color: #3498db; }}
        .total-files {{ color: #8e44ad; }}
        .processing {{ color: #e67e22; }}
        .status-message {{ background: #e8f4fd; padding: 20px; margin: 20px; border-radius: 10px; text-align: center; font-size: 1.2em; }}
        .gallery {{ padding: 30px; background: #f8f9fa; }}
        .gallery h2 {{ text-align: center; margin-bottom: 30px; color: #2c3e50; font-size: 2em; }}
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .image-container {{ position: relative; height: 200px; overflow: hidden; }}
        .image-container img {{ width: 100%; height: 100%; object-fit: contain; background: #f8f9fa; }}
        .image-info {{ padding: 15px; }}
        .image-name {{ font-weight: bold; margin-bottom: 5px; color: #2c3e50; }}
        .image-details {{ display: flex; justify-content: space-between; color: #666; font-size: 0.9em; }}
        .no-images {{ text-align: center; padding: 50px; color: #666; font-size: 1.2em; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 20px; }}
        .refresh-note {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 INTEGRATE-3-6-05</h1>
            <div class="subtitle">kana08データセット - 最終評価版 抽出結果ダッシュボード</div>
            <div class="status-message">{status_message}</div>
        </div>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value success-rate">{success_rate if total_files > 0 else "処理中"}</div><div class="stat-label">成功率</div></div>
            <div class="stat-card"><div class="stat-value total-files">{total_files}</div><div class="stat-label">現在の画像数</div></div>
            <div class="stat-card"><div class="stat-value quality-high">{high_quality if 'high_quality' in locals() else "計算中"}</div><div class="stat-label">高品質画像</div></div>
            <div class="stat-card"><div class="stat-value quality-medium">{medium_quality if 'medium_quality' in locals() else "計算中"}</div><div class="stat-label">中品質画像</div></div>
            <div class="stat-card"><div class="stat-value quality-low">{low_quality if 'low_quality' in locals() else "計算中"}</div><div class="stat-label">低品質画像</div></div>
        </div>'''
    
    if total_files > 0:
        html_content += f'''
        <div class="gallery">
            <h2>📸 抽出結果ギャラリー</h2>
            <div class="images-grid">'''
        
        # 画像をBase64エンコードして埋め込み
        for img_file in image_files[:20]:  # 最初の20枚まで表示
            try:
                with open(img_file, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                file_size_kb = len(img_data) / 1024
                quality_level = "高品質" if file_size_kb > 50 else "中品質" if file_size_kb > 10 else "低品質"
                quality_class = "high" if file_size_kb > 50 else "medium" if file_size_kb > 10 else "low"
                
                html_content += f'''
                <div class="image-card">
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,{img_base64}" alt="{img_file.name}">
                        <div class="quality-badge {quality_class}">{quality_level}</div>
                    </div>
                    <div class="image-info">
                        <div class="image-name">{img_file.name}</div>
                        <div class="image-details">
                            <span>サイズ: {file_size_kb:.1f}KB</span>
                            <span>品質: {quality_level}</span>
                        </div>
                    </div>
                </div>'''
            except Exception as e:
                print(f"画像処理エラー: {img_file.name} - {e}")
                continue
        
        html_content += '''
            </div>
        </div>'''
    else:
        html_content += '''
        <div class="gallery">
            <div class="no-images">
                🔄 バックグラウンド抽出処理中...<br>
                <small>処理完了後、このページを更新してください</small>
            </div>
            <div class="refresh-note">
                <strong>📝 処理状況確認方法:</strong><br>
                • ターミナル: <code>tail -f logs/INTEGRATE-3-6-05_extraction.log</code><br>
                • プロセス確認: <code>ps aux | grep sam_yolo</code><br>
                • 結果確認: <code>ls /mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-05/extraction/</code>
            </div>
        </div>'''
    
    html_content += f'''
        <div class="footer">
            <p>🔬 INTEGRATE-3-6-05 抽出結果ダッシュボード | 生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Generated with Claude Code</p>
        </div>
    </div>
</body>
</html>'''
    
    # ダッシュボードファイル保存
    dashboard_file = dashboard_dir / "dashboard.html"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ INTEGRATE-3-6-05ダッシュボード生成完了: {dashboard_file}")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/tracker/INTEGRATE-3-6-05")
    print(f"📊 現在の画像数: {total_files}枚")
    
    return str(dashboard_file)

if __name__ == "__main__":
    create_dashboard()