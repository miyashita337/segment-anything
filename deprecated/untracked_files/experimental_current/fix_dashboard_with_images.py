from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
QUAL-006-EXTENDED 画像付きダッシュボード生成
425/424矛盾修正 + 画像ギャラリー表示機能
"""

import base64
import json
import os
from datetime import datetime
from pathlib import Path


def get_image_base64(image_path):
    """画像をBase64エンコードして返す"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        print(f"⚠️ 画像読み込みエラー: {image_path} - {e}")
        return None

def get_file_size_mb(file_path):
    """ファイルサイズをMBで取得"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def get_image_quality_badge(file_size_bytes):
    """ファイルサイズに基づいて品質バッジを返す"""
    size_kb = file_size_bytes / 1024
    if size_kb > 100:
        return ("high", "高品質")
    elif size_kb > 50:
        return ("medium", "中品質")
    else:
        return ("low", "低品質")

def generate_complete_dashboard():
    """
    画像付き完全ダッシュボードを生成
    """
    tracker_id = "QUAL-006-EXTENDED"
    workspace_path = fget_path("output", "{tracker_id}")
    extraction_dir = os.path.join(workspace_path, "extraction")
    
    # 修正統計データ読み込み
    stats_file = os.path.join(workspace_path, "quality", "fixed_unified_statistics.json")
    with open(stats_file, 'r') as f:
        unified_stats = json.load(f)
    
    # 抽出ファイルリスト取得（最初の20枚のみ表示）
    extracted_files = sorted([f for f in os.listdir(extraction_dir) if f.endswith('.jpg')])[:20]
    
    print(f"🖼️ 画像ギャラリー生成開始...")
    print(f"  総画像数: {len(os.listdir(extraction_dir))}枚")
    print(f"  表示画像数: {len(extracted_files)}枚")
    
    # 画像カード生成
    image_cards_html = []
    high_count = medium_count = low_count = 0
    
    for i, filename in enumerate(extracted_files):
        image_path = os.path.join(extraction_dir, filename)
        
        # Base64エンコード
        base64_data = get_image_base64(image_path)
        if not base64_data:
            continue
            
        # ファイル情報取得
        file_size = os.path.getsize(image_path)
        quality_class, quality_label = get_image_quality_badge(file_size)
        
        # 品質カウント
        if quality_class == "high":
            high_count += 1
        elif quality_class == "medium":
            medium_count += 1
        else:
            low_count += 1
        
        # 画像カードHTML生成
        card_html = f'''        <div class="image-card">
            <div class="image-container">
                <img src="{base64_data}" alt="{filename}" loading="lazy">
                <div class="quality-badge {quality_class}">{quality_label}</div>
            </div>
            <div class="image-info">
                <div class="image-name">{filename}</div>
                <div class="image-details">
                    <span>{file_size // 1024} KB</span>
                    <span>{quality_label}</span>
                </div>
            </div>
        </div>'''
        
        image_cards_html.append(card_html)
        
        if (i + 1) % 5 == 0:
            print(f"  処理完了: {i + 1}/{len(extracted_files)}枚")
    
    # HTML生成
    success_count = high_count + medium_count
    success_rate = (success_count / len(extracted_files)) * 100 if extracted_files else 0
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} - 修正品質評価ダッシュボード</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.15); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 3em; font-weight: 300; }}
        .header .subtitle {{ font-size: 1.3em; opacity: 0.9; margin-top: 15px; }}
        .header .correction-badge {{ background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; margin-top: 15px; display: inline-block; }}
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
        .correction-info {{ padding: 40px; background: #e8f5e8; border-left: 5px solid #27ae60; }}
        .correction-info h3 {{ color: #27ae60; margin-top: 0; }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .comparison-table th, .comparison-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .comparison-table th {{ background: #f8f9fa; font-weight: bold; }}
        .old-value {{ color: #e74c3c; text-decoration: line-through; }}
        .new-value {{ color: #27ae60; font-weight: bold; }}
        .wilson-info {{ padding: 20px; background: #f0f8ff; border-radius: 10px; margin-top: 20px; }}
        .gallery {{ padding: 40px; background: #f8f9fa; }}
        .gallery h2 {{ text-align: center; margin-bottom: 30px; color: #2c3e50; font-size: 2em; }}
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 18px; }}
        .image-card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .image-container {{ position: relative; min-height: 200px; overflow: visible; }}
        .image-container img {{ width: 100%; height: 200px; object-fit: contain; background: #f8f9fa; display: block; border-radius: 8px 8px 0 0; }}
        .quality-badge {{ position: absolute; top: 10px; right: 10px; padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.8em; }}
        .quality-badge.high {{ background: #27ae60; }}
        .quality-badge.medium {{ background: #f39c12; }}
        .quality-badge.low {{ background: #e74c3c; }}
        .image-info {{ padding: 15px; }}
        .image-name {{ font-weight: bold; margin-bottom: 5px; color: #2c3e50; }}
        .image-details {{ display: flex; justify-content: space-between; color: #666; font-size: 0.9em; }}
        .gallery-note {{ text-align: center; padding: 20px; color: #666; font-style: italic; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 30px; }}
        .generation-info {{ font-size: 0.9em; opacity: 0.8; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 {tracker_id}</h1>
            <div class="subtitle">修正品質評価ダッシュボード</div>
            <div class="correction-badge">✅ QUAL-005 統一統計システム適用済み</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{unified_stats['total_input_images']}</div>
                <div class="stat-label">総入力画像数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success-rate">{unified_stats['success_rate']:.1%}</div>
                <div class="stat-label">数学的修正成功率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{unified_stats['success_count']}</div>
                <div class="stat-label">制約適用成功数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{unified_stats['total_extracted_files']}</div>
                <div class="stat-label">実抽出ファイル数</div>
            </div>
        </div>

        <div class="quality-summary">
            <h2>📊 画像品質分析（表示サンプル: {len(extracted_files)}枚）</h2>
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

        <div class="correction-info">
            <h3>🔧 425/424矛盾修正詳細</h3>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>項目</th>
                        <th>修正前（矛盾あり）</th>
                        <th>修正後（QCC-FIX-001適用）</th>
                        <th>修正内容</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>入力画像数</strong></td>
                        <td class="old-value">424枚</td>
                        <td class="new-value">{unified_stats['total_input_images']}枚</td>
                        <td>正確な入力数確定</td>
                    </tr>
                    <tr>
                        <td><strong>抽出ファイル数</strong></td>
                        <td class="old-value">425枚</td>
                        <td class="new-value">{unified_stats['total_extracted_files']}枚</td>
                        <td>実ファイル数カウント</td>
                    </tr>
                    <tr>
                        <td><strong>成功数</strong></td>
                        <td class="old-value">425枚（数学的矛盾）</td>
                        <td class="new-value">{unified_stats['success_count']}枚</td>
                        <td>数学的制約適用（成功数≤入力数）</td>
                    </tr>
                    <tr>
                        <td><strong>成功率</strong></td>
                        <td class="old-value">425/424 = 100.2%（不可能）</td>
                        <td class="new-value">{unified_stats['success_rate']:.1%}</td>
                        <td>Wilson信頼区間適用</td>
                    </tr>
                    <tr>
                        <td><strong>数学的整合性</strong></td>
                        <td class="old-value">❌ 矛盾（入力<抽出）</td>
                        <td class="new-value">✅ 整合性確保</td>
                        <td>統一統計システム適用</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="wilson-info">
                <h4>📊 Wilson信頼区間（95%信頼度）</h4>
                <p><strong>信頼区間:</strong> [{unified_stats['wilson_confidence_interval']['lower']:.3f}, {unified_stats['wilson_confidence_interval']['upper']:.3f}]</p>
                <p><strong>統計的意味:</strong> 95%の確率で真の成功率がこの範囲内に存在</p>
                <p><strong>品質保証:</strong> 統計的に妥当な成功率計算により、小サンプルでも信頼性のある評価</p>
            </div>
        </div>

        <div class="gallery">
            <h2>🎨 抽出画像ギャラリー（先頭{len(extracted_files)}枚表示）</h2>
            <div class="images-grid">
{chr(10).join(image_cards_html)}
            </div>
            <div class="gallery-note">
                ⚠️ パフォーマンス考慮により最初の{len(extracted_files)}枚のみ表示しています。<br>
                総抽出ファイル数: {unified_stats['total_extracted_files']}枚
            </div>
        </div>

        <div class="footer">
            <p>🤖 Generated by QUAL-005 Unified Statistics System</p>
            <div class="generation-info">
                修正完了日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
                URL: <a href="http://100.123.241.106:8088/tracker/{tracker_id}" style="color: #3498db;">http://100.123.241.106:8088/tracker/{tracker_id}</a><br>
                統計システム: {unified_stats['system']} | 
                Wilson信頼区間適用 | 
                数学的制約適用済み | 
                画像Base64埋め込み表示
            </div>
        </div>
    </div>
</body>
</html>'''

    # HTMLファイル保存
    dashboard_dir = os.path.join(workspace_path, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    dashboard_path = os.path.join(dashboard_dir, "dashboard.html")
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    file_size_mb = get_file_size_mb(dashboard_path)
    
    print(f"\n✅ 画像付きダッシュボード生成完了!")
    print(f"📁 ファイル: {dashboard_path}")
    print(f"📏 サイズ: {file_size_mb:.1f}MB")
    print(f"🖼️ 画像表示: {len(extracted_files)}枚（Base64埋め込み）")
    print(f"📊 品質分布: 高品質{high_count}枚・中品質{medium_count}枚・低品質{low_count}枚")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/tracker/{tracker_id}")
    
    return dashboard_path

if __name__ == "__main__":
    generate_complete_dashboard()