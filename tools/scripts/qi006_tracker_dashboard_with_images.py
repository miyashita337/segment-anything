#!/usr/bin/env python3
"""
QI-006: トラッカーダッシュボード画像統合更新
http://100.123.241.106:8088/tracker/QI-006 に抽出画像を表示
"""

import json
import shutil
import sys
from pathlib import Path


def copy_images_to_tracker_workspace():
    """抽出画像をトラッカーワークスペースにコピー"""

    # ソース: workspace内の画像
    source_dir = Path("workspace/QI-006/dashboard/images")

    # ターゲット: tracker-workspace（統合ダッシュボードサーバーがアクセス可能）
    target_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/dashboard/images")
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    if source_dir.exists():
        for image_file in source_dir.glob("*.jpg"):
            target_file = target_dir / image_file.name
            shutil.copy2(image_file, target_file)
            copied_files.append(image_file.name)
            print(f"📁 トラッカー用コピー: {image_file.name}")

    print(f"✅ トラッカー画像コピー完了: {len(copied_files)}枚")
    return copied_files


def generate_tracker_dashboard_with_images(image_files):
    """抽出画像付きトラッカーダッシュボード生成"""

    # GPT-5評価結果読み込み
    gpt5_file = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/quality/gpt5_lora_quality_evaluation.json"
    )
    gpt5_data = {}

    if gpt5_file.exists():
        with open(gpt5_file, "r", encoding="utf-8") as f:
            gpt5_data = json.load(f)

    # 複数キャラクター検出結果読み込み
    detection_file = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/quality/qi006_detection_results.json"
    )
    detection_data = {}

    if detection_file.exists():
        with open(detection_file, "r", encoding="utf-8") as f:
            detection_data = json.load(f)

    # 評価データ
    detailed_results = gpt5_data.get("detailed_results", [])
    grade_distribution = gpt5_data.get("grade_distribution", {}).get("grade_distribution", {})
    detection_stats = detection_data.get("statistics", {})

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QI-006: 複数キャラクター検出問題 - 統合ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .grade-a {{ @apply bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-b {{ @apply bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-c {{ @apply bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-d {{ @apply bg-orange-100 text-orange-800 px-2 py-1 rounded text-sm font-bold; }}
        .grade-f {{ @apply bg-red-100 text-red-800 px-2 py-1 rounded text-sm font-bold; }}
        .image-card {{ max-height: 350px; }}
        .image-card img {{ width: 100%; height: 200px; object-fit: contain; border-radius: 0.5rem; }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- ヘッダー -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">📊 QI-006: 複数キャラクター検出問題</h1>
            <p class="text-gray-600">GPT-5品質評価 + 抽出アルゴリズム修正（最大面積選択）統合結果</p>
            
            <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-purple-50 p-3 rounded-lg text-center">
                    <h3 class="text-sm font-medium text-purple-600">抽出画像数</h3>
                    <p class="text-xl font-bold text-purple-800">{len(image_files)}枚</p>
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

        <!-- アルゴリズム修正情報 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🔧 適用されたアルゴリズム修正</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="border-l-4 border-blue-500 pl-4">
                    <h3 class="font-medium text-gray-800 mb-1">1. 最大面積選択</h3>
                    <p class="text-sm text-gray-600">複数キャラクター時→最大面積マスク選択</p>
                    <p class="text-xs text-blue-600">信頼度ベース→面積ベースに変更</p>
                </div>
                <div class="border-l-4 border-green-500 pl-4">
                    <h3 class="font-medium text-gray-800 mb-1">2. fullbody_priority</h3>
                    <p class="text-sm text-gray-600">全身検出を優先する品質評価</p>
                    <p class="text-xs text-green-600">balanced→fullbody_priorityに変更</p>
                </div>
                <div class="border-l-4 border-purple-500 pl-4">
                    <h3 class="font-medium text-gray-800 mb-1">3. YOLO→SAM Hybrid</h3>
                    <p class="text-sm text-gray-600">YOLOのbboxをSAMプロンプトに使用</p>
                    <p class="text-xs text-purple-600">grid単体→hybrid方式に拡張</p>
                </div>
            </div>
        </div>

        <!-- 抽出結果画像一覧 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🖼️ 抽出結果画像 ({len(image_files)}枚)</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">"""

    # 評価結果を辞書化
    evaluation_dict = {}
    for result in detailed_results:
        if result.get("status") == "success":
            evaluation_dict[result.get("image_name", "")] = result

    # 画像カード生成
    sorted_images = sorted(image_files)
    for image_name in sorted_images:
        evaluation = evaluation_dict.get(image_name, {})
        grade = evaluation.get("grade", "N/A")
        lora_suitability = evaluation.get("lora_suitability", "不明")
        reason = evaluation.get("detailed_reason", "評価なし")
        person_count = evaluation.get("person_count", "不明")
        extraction_quality = evaluation.get("extraction_quality", "不明")

        # 短縮版理由
        short_reason = reason[:40] + "..." if len(reason) > 40 else reason

        html += f"""
                <div class="image-card border rounded-lg p-3 bg-gray-50 hover:shadow-md transition-shadow">
                    <div class="mb-2">
                        <img src="images/{image_name}" 
                             alt="{image_name}" 
                             class="border rounded">
                    </div>
                    <div class="space-y-2">
                        <h4 class="text-xs font-medium truncate">{image_name}</h4>
                        <div class="flex justify-between items-center">
                            <span class="grade-{grade.lower()}">{grade}</span>
                            <span class="text-xs text-gray-500">{lora_suitability}</span>
                        </div>
                        <div class="text-xs text-gray-600 space-y-1">
                            <div>👤 人物: {person_count}</div>
                            <div>✂️ 品質: {extraction_quality}</div>
                            <div class="text-gray-500" title="{reason}">{short_reason}</div>
                        </div>
                    </div>
                </div>"""

    html += f"""
            </div>
        </div>

        <!-- GPT-5品質分布 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🏆 GPT-5品質評価分布</h2>
            <div class="grid grid-cols-5 gap-4">"""

    grade_descriptions = {
        "A": ("🏆", "LoRA学習に最適"),
        "B": ("✅", "適している"),
        "C": ("⚠️", "注意が必要"),
        "D": ("❌", "問題あり"),
        "F": ("🚫", "使用不可"),
    }

    for grade in ["A", "B", "C", "D", "F"]:
        count = grade_distribution.get(grade, 0)
        emoji, desc = grade_descriptions[grade]
        html += f"""
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl mb-1">{emoji}</div>
                    <div class="grade-{grade.lower()} mb-1">{grade}</div>
                    <div class="text-xl font-bold text-gray-800">{count}枚</div>
                    <div class="text-xs text-gray-600">{desc}</div>
                </div>"""

    html += f"""
            </div>
        </div>

        <!-- 複数キャラクター検出統計 -->"""

    if detection_stats:
        html += f"""
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🔍 複数キャラクター検出統計</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center p-4 bg-green-50 rounded-lg">
                    <div class="text-2xl font-bold text-green-600">{detection_stats.get('single_character', 0)}</div>
                    <div class="text-sm text-green-700">単一キャラクター</div>
                </div>
                <div class="text-center p-4 bg-red-50 rounded-lg">
                    <div class="text-2xl font-bold text-red-600">{detection_stats.get('multiple_character', 0)}</div>
                    <div class="text-sm text-red-700">複数キャラクター残存</div>
                </div>
                <div class="text-center p-4 bg-blue-50 rounded-lg">
                    <div class="text-2xl font-bold text-blue-600">{detection_stats.get('success_rate', 0):.1f}%</div>
                    <div class="text-sm text-blue-700">検出成功率</div>
                </div>
            </div>
        </div>"""

    html += f"""
        <!-- 改善効果サマリー -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">📈 改善効果</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-red-50 p-4 rounded-lg">
                    <h3 class="font-medium text-red-800 mb-2">❌ 修正前の問題</h3>
                    <ul class="text-sm text-red-700 space-y-1">
                        <li>• 信頼度ベースで小さいキャラクターが選ばれる</li>
                        <li>• 複数キャラクター時の主要キャラ取得失敗</li>
                        <li>• 下半身のみ抽出等の部分抽出</li>
                        <li>• LoRA学習に不適切な画像生成</li>
                    </ul>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <h3 class="font-medium text-green-800 mb-2">✅ 修正後の改善</h3>
                    <ul class="text-sm text-green-700 space-y-1">
                        <li>• 最大面積選択で確実に主要キャラクター取得</li>
                        <li>• fullbody_priorityで全身検出重視</li>
                        <li>• YOLO→SAM hybridでより精密な抽出</li>
                        <li>• GPT-5による客観的品質評価実現</li>
                    </ul>
                </div>
            </div>
            
            <div class="mt-6 p-4 bg-blue-50 rounded-lg">
                <h3 class="font-medium text-blue-800 mb-2">🎯 今後の課題（QI-007）</h3>
                <p class="text-sm text-blue-700">
                    人物認定精度向上・上半身+下半身統合抽出システム構築により、
                    より完全なキャラクター抽出を実現。下半身のみ抽出される問題の根本解決を目指す。
                </p>
            </div>
        </div>

        <!-- フッター -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>QI-006: 複数キャラクター検出問題 - GPT-5品質評価 + 抽出アルゴリズム修正統合結果</p>
            <p>処理画像: {len(image_files)}枚 | 高品質(A/B): {grade_distribution.get('A', 0) + grade_distribution.get('B', 0)}枚 | 要改善(C/D/F): {grade_distribution.get('C', 0) + grade_distribution.get('D', 0) + grade_distribution.get('F', 0)}枚</p>
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    """メイン実行関数"""
    print("📊 QI-006: トラッカーダッシュボード画像統合更新開始")
    print("=" * 60)

    # 1. 画像をトラッカーワークスペースにコピー
    print("1. トラッカー用画像コピー中...")
    image_files = copy_images_to_tracker_workspace()

    if not image_files:
        print("❌ コピーする画像が見つかりませんでした")
        return False

    # 2. トラッカーダッシュボードHTML生成
    print("2. トラッカーダッシュボードHTML生成中...")
    html_content = generate_tracker_dashboard_with_images(image_files)

    # 3. トラッカーダッシュボード保存
    dashboard_file = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/dashboard/dashboard.html"
    )
    dashboard_file.parent.mkdir(parents=True, exist_ok=True)

    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_size = dashboard_file.stat().st_size

    print(f"✅ トラッカーダッシュボード更新完了")
    print(f"   ファイル: {dashboard_file}")
    print(f"   サイズ: {file_size / 1024:.1f}KB")
    print(f"   画像数: {len(image_files)}枚")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/tracker/QI-006")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
