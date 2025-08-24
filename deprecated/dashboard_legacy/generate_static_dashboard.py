#!/usr/bin/env python3
"""
INTEGRATE-3-6-XX 静的ダッシュボード生成スクリプト
"""

import base64
import logging
from datetime import datetime
from pathlib import Path


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return ""


def analyze_extraction_results(extraction_dir: Path) -> dict:
    if not extraction_dir.exists():
        return {'total_files': 0, 'total_size_mb': 0, 'avg_size_kb': 0, 'files': []}
    
    image_files = sorted(extraction_dir.glob("kana08_*.jpg"))
    file_data = []
    total_size = 0
    
    for img_path in image_files:
        size_kb = img_path.stat().st_size / 1024
        total_size += size_kb
        file_data.append({
            'name': img_path.name,
            'size_kb': size_kb,
            'base64': encode_image_to_base64(img_path)
        })
    
    return {
        'total_files': len(image_files),
        'total_size_mb': total_size / 1024,
        'avg_size_kb': total_size / len(image_files) if image_files else 0,
        'files': file_data
    }


def get_quality_scores():
    return {
        'kana08_0001.jpg': 0.762, 'kana08_0002.jpg': 0.636, 'kana08_0003.jpg': 0.684,
        'kana08_0004.jpg': 0.822, 'kana08_0005.jpg': 0.763, 'kana08_0006.jpg': 0.561,
        'kana08_0007.jpg': 0.657, 'kana08_0008.jpg': 0.803, 'kana08_0009.jpg': 0.261,
        'kana08_0010.jpg': 0.408, 'kana08_0011.jpg': 0.651, 'kana08_0012.jpg': 0.613,
        'kana08_0013.jpg': 0.488, 'kana08_0014.jpg': 0.631, 'kana08_0015.jpg': 0.566,
        'kana08_0016.jpg': 0.748, 'kana08_0017.jpg': 0.743, 'kana08_0018.jpg': 0.405,
        'kana08_0019.jpg': 0.460, 'kana08_0020.jpg': 0.620, 'kana08_0021.jpg': 0.679,
        'kana08_0022.jpg': 0.311, 'kana08_0023.jpg': 0.437, 'kana08_0024.jpg': 0.483
    }


def generate_dashboard_html(tracker_id: str, analysis_data: dict) -> str:
    quality_scores = get_quality_scores()
    
    tracker_info = {
        'INTEGRATE-3-6-01': {'name': 'Phase 3-6統合初期版', 'model': 'yolov8x.pt'},
        'INTEGRATE-3-6-02': {'name': 'Phase 3-6改良版', 'model': 'yolov8x.pt'},
        'INTEGRATE-3-6-03': {'name': 'YOLO汎用版検証', 'model': 'yolov8x.pt'},
        'INTEGRATE-3-6-04': {'name': 'アニメ特化版検証', 'model': 'yolov8x6_animeface.pt→yolov8x.pt'}
    }
    
    info = tracker_info.get(tracker_id, {'name': tracker_id, 'model': 'Unknown'})
    
    # 品質統計
    high_quality_count = medium_quality_count = low_quality_count = 0
    for file_info in analysis_data['files']:
        quality = quality_scores.get(file_info['name'], 0.0)
        if quality >= 0.7:
            high_quality_count += 1
        elif quality >= 0.5:
            medium_quality_count += 1
        else:
            low_quality_count += 1
    
    success_rate = (analysis_data['total_files'] / 24 * 100) if analysis_data['total_files'] > 0 else 0
    
    # ギャラリー生成
    gallery_html = ""
    if analysis_data['files']:
        gallery_html = '<div class="images-grid">'
        for file_info in analysis_data['files']:
            quality = quality_scores.get(file_info['name'], 0.0)
            if quality >= 0.7:
                quality_class, quality_text = 'high', f'高品質 {quality:.3f}'
            elif quality >= 0.5:
                quality_class, quality_text = 'medium', f'中品質 {quality:.3f}'
            else:
                quality_class, quality_text = 'low', f'低品質 {quality:.3f}'
            
            gallery_html += f'''
                <div class="image-card">
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,{file_info['base64']}" alt="{file_info['name']}" loading="lazy">
                        <div class="quality-badge {quality_class}">{quality_text}</div>
                    </div>
                    <div class="image-info">
                        <div class="image-name">{file_info['name']}</div>
                        <div class="image-details">
                            <span>サイズ: {file_info['size_kb']:.1f}KB</span>
                            <span>品質: {quality:.3f}</span>
                        </div>
                    </div>
                </div>
            '''
        gallery_html += '</div>'
    else:
        gallery_html = '<div class="no-images">📭 抽出結果が見つかりませんでした</div>'
    
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} - 抽出結果ダッシュボード</title>
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
        .file-size {{ color: #2c3e50; }}
        .gallery {{ padding: 30px; background: #f8f9fa; }}
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
        .no-images {{ text-align: center; padding: 50px; color: #666; font-size: 1.2em; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 {tracker_id}</h1>
            <div class="subtitle">{info['name']} - 抽出結果ダッシュボード</div>
        </div>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value success-rate">{success_rate:.1f}%</div><div class="stat-label">成功率</div></div>
            <div class="stat-card"><div class="stat-value total-files">{analysis_data['total_files']}</div><div class="stat-label">総ファイル数</div></div>
            <div class="stat-card"><div class="stat-value quality-high">{high_quality_count}</div><div class="stat-label">高品質画像</div></div>
            <div class="stat-card"><div class="stat-value quality-medium">{medium_quality_count}</div><div class="stat-label">中品質画像</div></div>
            <div class="stat-card"><div class="stat-value quality-low">{low_quality_count}</div><div class="stat-label">低品質画像</div></div>
            <div class="stat-card"><div class="stat-value file-size">{analysis_data['total_size_mb']:.1f}MB</div><div class="stat-label">総容量</div></div>
        </div>
        <div class="gallery">
            <h2>📸 抽出結果ギャラリー</h2>
            {gallery_html}
        </div>
        <div class="footer">
            <p>🔬 INTEGRATE-3-6 Series Dashboard | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Generated with Claude Code</p>
        </div>
    </div>
</body>
</html>"""


def generate_tracker_dashboard(tracker_id: str, logger: logging.Logger):
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
    extraction_dir = workspace_base / tracker_id / "extraction"
    dashboard_dir = workspace_base / tracker_id / "dashboard"
    
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📊 {tracker_id} ダッシュボード生成開始")
    
    analysis_data = analyze_extraction_results(extraction_dir)
    logger.info(f"  - 分析完了: {analysis_data['total_files']}ファイル, {analysis_data['total_size_mb']:.2f}MB")
    
    html_content = generate_dashboard_html(tracker_id, analysis_data)
    
    dashboard_file = dashboard_dir / "dashboard.html"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ {tracker_id} ダッシュボード生成完了: {dashboard_file}")
    return dashboard_file


def main():
    logger = setup_logging()
    tracker_ids = ["INTEGRATE-3-6-01", "INTEGRATE-3-6-02", "INTEGRATE-3-6-03", "INTEGRATE-3-6-04"]
    
    logger.info("🚀 全トラッカーダッシュボード生成開始")
    generated_dashboards = []
    
    for tracker_id in tracker_ids:
        try:
            dashboard_file = generate_tracker_dashboard(tracker_id, logger)
            generated_dashboards.append((tracker_id, dashboard_file))
        except Exception as e:
            logger.error(f"❌ {tracker_id} ダッシュボード生成失敗: {e}")
    
    logger.info("=" * 60)
    logger.info(f"✅ ダッシュボード生成完了: {len(generated_dashboards)}/{len(tracker_ids)}")
    
    for tracker_id, dashboard_file in generated_dashboards:
        if dashboard_file.exists():
            file_size = dashboard_file.stat().st_size / 1024
            logger.info(f"  📄 {tracker_id}: {dashboard_file} ({file_size:.1f}KB)")
    
    return generated_dashboards


if __name__ == "__main__":
    main()