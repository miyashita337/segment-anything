#!/usr/bin/env python3
"""
QCA-001統合ダッシュボード生成スクリプト
yado作者とkiri作者の結果を統合表示
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# 統合ダッシュボード生成は直接HTML生成するため、インポート不要


def merge_qca001_dashboards():
    """QCA-001の複数作者結果を統合ダッシュボードに統合"""

    # WorkspaceConfigManagerを使って動的パス解決
    from config.workspace_config import WorkspaceConfig

    workspace_config = WorkspaceConfig()
    config = workspace_config.get_workspace_config("QCA-001")

    if config:
        # 動的パス生成
        primary_workspace = Path(config["workspace_path"])
        # セカンダリパス（マルチ作者対応）
        if config["author_name"] == "yado":
            secondary_workspace = Path("/mnt/c/AItools/lora/train/kiri/tracker-workspace/QCA-001")
        else:
            secondary_workspace = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QCA-001")
        yado_workspace, kiri_workspace = primary_workspace, secondary_workspace
    else:
        # フォールバック: 従来のハードコード
        yado_workspace = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QCA-001")
        kiri_workspace = Path("/mnt/c/AItools/lora/train/kiri/tracker-workspace/QCA-001")

    print("🔄 QCA-001統合ダッシュボード生成開始...")

    # 抽出画像の統合
    all_images = []

    # yado作者の画像
    yado_extraction_dir = yado_workspace / "extraction"
    if yado_extraction_dir.exists():
        for img_file in yado_extraction_dir.glob("*.jpg"):
            all_images.append(
                {
                    "path": str(img_file),
                    "name": img_file.name,
                    "size": img_file.stat().st_size,
                    "author": "yado",
                    "quality": "high" if img_file.stat().st_size > 100000 else "low",
                }
            )

    # kiri作者の画像をyado ワークスペースにコピーして統合
    kiri_extraction_dir = kiri_workspace / "extraction"
    yado_extraction_dir = yado_workspace / "extraction"

    if kiri_extraction_dir.exists():
        for img_file in kiri_extraction_dir.glob("*.jpg"):
            # kiri作者の画像をyado ワークスペースにコピー（プレフィックス付き）
            dest_name = f"kiri_{img_file.name}"
            dest_path = yado_extraction_dir / dest_name
            shutil.copy2(img_file, dest_path)

            all_images.append(
                {
                    "path": str(dest_path),
                    "name": dest_name,
                    "size": dest_path.stat().st_size,
                    "author": "kiri",
                    "quality": "high" if dest_path.stat().st_size > 50000 else "low",
                }
            )
            print(f"📋 kiri画像コピー: {img_file.name} → {dest_name}")

    print(f"✅ 統合対象画像: {len(all_images)}枚")
    print(f"   - yado作者: {len([img for img in all_images if img['author'] == 'yado'])}枚")
    print(f"   - kiri作者: {len([img for img in all_images if img['author'] == 'kiri'])}枚")

    # 統合ダッシュボード生成は直接HTML生成

    # yado作者のワークスペースに統合ダッシュボードを作成（既存URL維持）
    merged_dashboard_path = yado_workspace / "dashboard" / "dashboard.html"

    dashboard_content = generate_merged_dashboard_html(all_images)

    # ダッシュボード保存
    merged_dashboard_path.parent.mkdir(exist_ok=True)
    with open(merged_dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_content)

    print(f"✅ 統合ダッシュボード生成完了:")
    print(f"   - パス: {merged_dashboard_path}")
    print(f"   - サイズ: {merged_dashboard_path.stat().st_size:,} bytes")
    print(f"   - URL: http://100.123.241.106:8088/tracker/QCA-001")

    return str(merged_dashboard_path)


def generate_merged_dashboard_html(all_images):
    """統合ダッシュボードHTML生成"""

    # 品質分析
    high_quality = len([img for img in all_images if img["quality"] == "high"])
    low_quality = len(all_images) - high_quality
    quality_score = (high_quality / len(all_images) * 100) if all_images else 0

    # 作者別統計
    yado_images = [img for img in all_images if img["author"] == "yado"]
    kiri_images = [img for img in all_images if img["author"] == "kiri"]

    # 画像ギャラリー生成
    gallery_html = ""

    # yado作者セクション
    if yado_images:
        gallery_html += (
            '<div class="author-section"><h3>👤 yado作者（バランス型・キャラクター重視）</h3><div class="images-grid">'
        )
        for img in yado_images:
            quality_class = img["quality"]
            quality_label = "高品質" if quality_class == "high" else "低品質"
            size_kb = img["size"] // 1024

            # QCA-001統合ダッシュボード用パス変換
            # /mnt/c/AItools/lora/train/yado/tracker-workspace/QCA-001/extraction/xxx.jpg
            # → QCA-001/extraction/xxx.jpg
            relative_path = img["path"].replace(
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/", ""
            )

            gallery_html += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="/{relative_path}" alt="{img['name']}" loading="lazy">
                <div class="quality-badge {quality_class}">{quality_label}</div>
            </div>
            <div class="image-info">
                <div class="image-name">{img['name']}</div>
                <div class="image-details">
                    <span>{size_kb} KB</span>
                    <span>{quality_label}</span>
                </div>
            </div>
        </div>"""

        gallery_html += "</div></div>"

    # kiri作者セクション
    if kiri_images:
        gallery_html += (
            '<div class="author-section"><h3>🎨 kiri作者（細密描写・高品質重視）</h3><div class="images-grid">'
        )
        for img in kiri_images:
            quality_class = img["quality"]
            quality_label = "高品質" if quality_class == "high" else "低品質"
            size_kb = img["size"] // 1024

            # QCA-001統合ダッシュボード用パス変換（yado ワークスペースに統合済み）
            relative_path = img["path"].replace(
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/", ""
            )

            gallery_html += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="/{relative_path}" alt="{img['name']}" loading="lazy">
                <div class="quality-badge {quality_class}">{quality_label}</div>
            </div>
            <div class="image-info">
                <div class="image-name">{img['name']}</div>
                <div class="image-details">
                    <span>{size_kb} KB</span>
                    <span>{quality_label}</span>
                </div>
            </div>
        </div>"""

        gallery_html += "</div></div>"

    # HTMLテンプレート
    html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QCA-001 統合 - 作者別パラメータ適応システム検証</title>
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
        .author-stats {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; }}
        .author-stat {{ background: white; padding: 15px 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .author-stat-value {{ font-size: 1.8em; font-weight: bold; color: #3498db; }}
        .author-stat-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 QCA-001 統合ダッシュボード</h1>
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
                <div class="stat-value">{high_quality}</div>
                <div class="stat-label">高品質画像</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{low_quality}</div>
                <div class="stat-label">要改善画像</div>
            </div>
        </div>
        
        <div class="author-stats">
            <div class="author-stat">
                <div class="author-stat-value">{len(yado_images)}</div>
                <div class="author-stat-label">👤 yado作者</div>
            </div>
            <div class="author-stat">
                <div class="author-stat-value">{len(kiri_images)}</div>
                <div class="author-stat-label">🎨 kiri作者</div>
            </div>
        </div>

        <div class="quality-summary">
            <h2>📊 品質分析</h2>
            <div class="quality-chart">
                <div class="quality-stat high">
                    <div class="quality-count">{high_quality}</div>
                    <div class="quality-label">高品質</div>
                </div>
                <div class="quality-stat medium">
                    <div class="quality-count">0</div>
                    <div class="quality-label">中品質</div>
                </div>
                <div class="quality-stat low">
                    <div class="quality-count">{low_quality}</div>
                    <div class="quality-label">低品質</div>
                </div>
            </div>
        </div>

        <div class="gallery">
            <h2>🎨 作者別抽出画像ギャラリー</h2>
            {gallery_html}
        </div>

        <div class="footer">
            <p>🤖 QCA-001: 作者別パラメータ適応システム統合結果</p>
            <div class="generation-info">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                URL: <a href="http://100.123.241.106:8088/tracker/QCA-001" style="color: #3498db;">http://100.123.241.106:8088/tracker/QCA-001</a>
            </div>
        </div>
    </div>
</body>
</html>"""

    return html_template


if __name__ == "__main__":
    dashboard_path = merge_qca001_dashboards()
    print(f"🎯 統合ダッシュボード生成完了: {dashboard_path}")
