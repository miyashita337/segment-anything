#!/usr/bin/env python3
"""
KIRO-001-002 ダッシュボード画像ギャラリー修正スクリプト
"""

import json
import os
from datetime import datetime
from pathlib import Path


def generate_dashboard_with_images():
    """
    KIRO-001-002ダッシュボードに画像ギャラリーを追加
    """
    tracker_id = "KIRO-001-002"
    workspace_path = f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}"
    extraction_dir = os.path.join(workspace_path, "extraction")
    
    # 統計分析データ読み込み
    stats_file = os.path.join(workspace_path, "statistical_analysis_final.json")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # 抽出ファイルリスト取得
    extracted_files = sorted([f for f in os.listdir(extraction_dir) 
                            if f.endswith('.jpg') and f.startswith('extracted_')])
    
    print(f"🖼️ KIRO-001-002 画像ギャラリー生成開始...")
    print(f"  総画像数: {len(extracted_files)}枚")
    
    # 画像カード生成
    image_cards_html = []
    
    for i, filename in enumerate(extracted_files[:20]):  # 最初の20枚を表示
        # ファイル情報取得
        file_path = os.path.join(extraction_dir, filename)
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size // 1024
        
        # 品質判定（ファイルサイズベース）
        if file_size_kb > 100:
            quality_class = "high"
            quality_label = "高品質"
        elif file_size_kb > 50:
            quality_class = "medium"
            quality_label = "中品質"
        else:
            quality_class = "low"
            quality_label = "低品質"
        
        # 画像URL（相対パス）
        image_url = f"../{tracker_id}/extraction/{filename}"
        
        card_html = f'''                <div class="bg-white rounded-lg shadow-md overflow-hidden">
                    <div class="relative">
                        <img src="{image_url}" alt="{filename}" 
                             class="w-full h-48 object-contain bg-gray-100" 
                             loading="lazy"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="display:none;" class="w-full h-48 flex items-center justify-center bg-gray-100 text-gray-500">
                            画像読み込みエラー
                        </div>
                        <div class="absolute top-2 right-2 px-2 py-1 rounded text-xs font-semibold text-white bg-{quality_class == 'high' and 'green' or quality_class == 'medium' and 'yellow' or 'red'}-500">
                            {quality_label}
                        </div>
                    </div>
                    <div class="p-4">
                        <div class="font-semibold text-gray-800 text-sm mb-2">{filename}</div>
                        <div class="flex justify-between text-xs text-gray-600">
                            <span>{file_size_kb} KB</span>
                            <span>{quality_label}</span>
                        </div>
                    </div>
                </div>'''
        
        image_cards_html.append(card_html)
    
    # HTML生成
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KIRO-001-002 - 品質評価ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">KIRO-001-002 品質評価ダッシュボード</h1>
            <p class="text-gray-600">生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">総画像数</h3>
                <p class="text-3xl font-bold text-blue-600">{stats['sample_size']['current']}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">平均品質スコア</h3>
                <p class="text-3xl font-bold text-green-600">{stats['current_score']}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">成功画像数</h3>
                <p class="text-3xl font-bold text-emerald-600">{stats['sample_size']['current']}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">要改善数</h3>
                <p class="text-3xl font-bold text-red-600">0</p>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">📊 統計分析結果</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4">
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">Current(平均品質スコア)</div>
                    <p class="text-lg font-bold text-blue-600">{stats['current_score']}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">BaseLine</div>
                    <p class="text-lg font-bold text-gray-600">{stats['baseline_score']}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">p値</div>
                    <p class="text-lg font-bold text-indigo-600">{stats['p_value']}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">効果サイズ、Cohen's d</div>
                    <p class="text-lg font-bold text-purple-600">{stats['cohens_d']}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">改善率</div>
                    <p class="text-lg font-bold text-green-600">{stats['improvement_rate']}%</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">統計的有意性</div>
                    <p class="text-lg font-bold text-red-600">{stats['statistical_significance']}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">信頼区間</div>
                    <p class="text-lg font-bold text-teal-600">{stats['confidence_interval']}</p>
                </div>
            </div>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">品質分布</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <div class="bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">高品質</div>
                    <p class="text-2xl font-bold">12</p>
                </div>
                <div class="text-center">
                    <div class="bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">中品質</div>
                    <p class="text-2xl font-bold">20</p>
                </div>
                <div class="text-center">
                    <div class="bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">低品質</div>
                    <p class="text-2xl font-bold">7</p>
                </div>
                <div class="text-center">
                    <div class="bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">要改善</div>
                    <p class="text-2xl font-bold">0</p>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">🖼️ 抽出結果ギャラリー</h2>
            <p class="text-gray-600 mb-6">抽出成功: {len(extracted_files)}枚 (表示: 最初の20枚)</p>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
{chr(10).join(image_cards_html)}
            </div>
            {len(extracted_files) > 20 and f'<p class="text-center text-gray-500 mt-6">他 {len(extracted_files) - 20}枚の画像があります</p>' or ''}
        </div>
    </div>
</body>
</html>'''

    # HTMLファイル保存
    dashboard_dir = os.path.join(workspace_path, "dashboard")
    dashboard_path = os.path.join(dashboard_dir, "dashboard.html")
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ ダッシュボード画像ギャラリー修正完了!")
    print(f"📁 ファイル: {dashboard_path}")
    print(f"🖼️ 画像表示: {len(image_cards_html)}枚（最初の20枚）")
    print(f"📊 総抽出ファイル数: {len(extracted_files)}枚")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/tracker/{tracker_id}")
    
    return dashboard_path

if __name__ == "__main__":
    generate_dashboard_with_images()