#!/usr/bin/env python3
"""
QI-006: 抽出画像付きダッシュボード生成
既存のQI-006抽出画像を統合ダッシュボードでWeb表示可能にする
"""

import sys
import json
import glob
import shutil
from pathlib import Path

def copy_extraction_images_to_workspace():
    """既存の抽出画像をワークスペース内にコピー"""
    
    # ソースディレクトリ（既存のQI-006抽出結果）
    source_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/extraction")
    
    # ターゲットディレクトリ（ダッシュボードアクセス用）
    target_dir = Path("workspace/QI-006/dashboard/images")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # kana08_*.jpg ファイルを探してコピー
    copied_files = []
    
    if source_dir.exists():
        for image_file in source_dir.glob("kana08_*.jpg"):
            if "_multi_char_detection" not in image_file.name:  # 検出結果画像は除外
                target_file = target_dir / image_file.name
                shutil.copy2(image_file, target_file)
                copied_files.append(image_file.name)
                print(f"📁 コピー完了: {image_file.name}")
    
    print(f"✅ 画像コピー完了: {len(copied_files)}枚")
    return copied_files


def generate_qi006_dashboard_with_images(image_files):
    """抽出画像付きQI-006ダッシュボード生成"""
    
    # GPT-5評価結果読み込み
    gpt5_file = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/quality/gpt5_lora_quality_evaluation.json")
    gpt5_data = {}
    
    if gpt5_file.exists():
        with open(gpt5_file, 'r', encoding='utf-8') as f:
            gpt5_data = json.load(f)
    
    # 評価データから画像別結果を取得
    detailed_results = gpt5_data.get('detailed_results', [])
    grade_distribution = gpt5_data.get('grade_distribution', {}).get('grade_distribution', {})
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QI-006: 抽出画像確認ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .grade-a {{ @apply bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-b {{ @apply bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-c {{ @apply bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-d {{ @apply bg-orange-100 text-orange-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-f {{ @apply bg-red-100 text-red-800 px-2 py-1 rounded text-sm font-bold; }}
        .image-container {{ max-height: 300px; overflow: hidden; }}
        .image-container img {{ width: 100%; height: auto; object-fit: contain; }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- ヘッダー -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">📸 QI-006: 抽出画像確認ダッシュボード</h1>
            <p class="text-gray-600">修正したアルゴリズム（最大面積選択）による抽出結果とGPT-5品質評価</p>
            <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-blue-50 p-3 rounded-lg text-center">
                    <h3 class="text-sm font-medium text-blue-600">抽出画像数</h3>
                    <p class="text-xl font-bold text-blue-800">{len(image_files)}枚</p>
                </div>
                <div class="bg-green-50 p-3 rounded-lg text-center">
                    <h3 class="text-sm font-medium text-green-600">A/B高品質</h3>
                    <p class="text-xl font-bold text-green-800">{grade_distribution.get('A', 0) + grade_distribution.get('B', 0)}枚</p>
                </div>
                <div class="bg-yellow-50 p-3 rounded-lg text-center">
                    <h3 class="text-sm font-medium text-yellow-600">C注意必要</h3>
                    <p class="text-xl font-bold text-yellow-800">{grade_distribution.get('C', 0)}枚</p>
                </div>
                <div class="bg-red-50 p-3 rounded-lg text-center">
                    <h3 class="text-sm font-medium text-red-600">D/F低品質</h3>
                    <p class="text-xl font-bold text-red-800">{grade_distribution.get('D', 0) + grade_distribution.get('F', 0)}枚</p>
                </div>
            </div>
        </div>

        <!-- 抽出画像一覧 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🖼️ 抽出結果画像一覧</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">"""
    
    # 画像ファイルソート
    sorted_images = sorted(image_files)
    
    # 評価結果を辞書化
    evaluation_dict = {}
    for result in detailed_results:
        if result.get('status') == 'success':
            evaluation_dict[result.get('image_name', '')] = result
    
    # 各画像のカードを生成
    for image_name in sorted_images:
        evaluation = evaluation_dict.get(image_name, {})
        grade = evaluation.get('grade', 'N/A')
        lora_suitability = evaluation.get('lora_suitability', '不明')
        reason = evaluation.get('detailed_reason', '評価なし')
        
        # 短縮版理由
        short_reason = reason[:50] + "..." if len(reason) > 50 else reason
        
        html += f"""
                <div class="border rounded-lg p-3 bg-gray-50">
                    <h4 class="text-sm font-medium mb-2 truncate">{image_name}</h4>
                    <div class="image-container mb-2">
                        <img src="images/{image_name}" 
                             alt="{image_name}" 
                             class="border rounded">
                    </div>
                    <div class="space-y-1">
                        <div class="flex justify-between items-center">
                            <span class="grade-{grade.lower()}">{grade}</span>
                            <span class="text-xs text-gray-500">{lora_suitability}</span>
                        </div>
                        <p class="text-xs text-gray-600" title="{reason}">{short_reason}</p>
                    </div>
                </div>"""
    
    html += """
            </div>
        </div>

        <!-- 修正アルゴリズム情報 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mt-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">⚙️ 適用されたアルゴリズム修正</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="border-l-4 border-blue-500 pl-4">
                    <h3 class="font-medium text-gray-800">最大面積選択</h3>
                    <p class="text-sm text-gray-600">複数キャラクター検出時、最大面積のマスクを選択</p>
                </div>
                <div class="border-l-4 border-green-500 pl-4">
                    <h3 class="font-medium text-gray-800">fullbody_priority</h3>
                    <p class="text-sm text-gray-600">全身検出を優先する品質評価方式</p>
                </div>
                <div class="border-l-4 border-purple-500 pl-4">
                    <h3 class="font-medium text-gray-800">YOLO→SAM Hybrid</h3>
                    <p class="text-sm text-gray-600">YOLOのbboxをSAMプロンプトに使用</p>
                </div>
            </div>
        </div>

        <!-- フッター -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>QI-006: 複数キャラクター検出問題対応 - 抽出画像確認ダッシュボード</p>
            <p>画像数: {len(image_files)}枚 | アルゴリズム: 最大面積選択 + fullbody_priority</p>
        </div>
    </div>
</body>
</html>"""
    
    return html


def main():
    """メイン実行関数"""
    print("📊 QI-006: 抽出画像付きダッシュボード生成開始")
    print("=" * 60)
    
    # 1. 抽出画像をワークスペースにコピー
    print("1. 抽出画像コピー中...")
    image_files = copy_extraction_images_to_workspace()
    
    if not image_files:
        print("❌ コピーする画像が見つかりませんでした")
        return False
    
    # 2. ダッシュボードHTML生成
    print("2. ダッシュボードHTML生成中...")
    html_content = generate_qi006_dashboard_with_images(image_files)
    
    # 3. ダッシュボード保存
    dashboard_file = Path("workspace/QI-006/dashboard/dashboard.html")
    dashboard_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    file_size = dashboard_file.stat().st_size
    
    print(f"✅ ダッシュボード生成完了")
    print(f"   ファイル: {dashboard_file}")
    print(f"   サイズ: {file_size / 1024:.1f}KB")
    print(f"   画像数: {len(image_files)}枚")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/workspace/QI-006/dashboard/dashboard.html")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)