#!/usr/bin/env python3
"""
QI-006: 複数キャラクター検出システム - ダッシュボード生成
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """QI-006 ダッシュボード生成"""
    print("📊 QI-006: ダッシュボード生成開始")

    # パス設定
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
    qi006_workspace = workspace_base / "QI-006"

    extraction_dir = qi006_workspace / "extraction"
    quality_dir = qi006_workspace / "quality"
    dashboard_dir = qi006_workspace / "dashboard"

    # 結果データ読み込み
    results_file = quality_dir / "qi006_detection_results.json"
    if not results_file.exists():
        print(f"❌ 結果データが見つかりません: {results_file}")
        return False

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data["statistics"]
    results = data["detailed_results"]

    print(f"📊 データ読み込み完了: {len(results)}件")

    # ダッシュボードHTML生成
    html_content = generate_dashboard_html(stats, results, extraction_dir)

    # ダッシュボード保存
    dashboard_file = dashboard_dir / "dashboard.html"
    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ ダッシュボード生成完了: {dashboard_file}")
    print(f"📏 ファイルサイズ: {dashboard_file.stat().st_size / 1024 / 1024:.1f}MB")

    return True


def generate_dashboard_html(stats, results, extraction_dir):
    """QI-006専用ダッシュボードHTML生成"""

    # 画像ファイル一覧を収集（直接参照用）
    available_images = {}
    if extraction_dir.exists():
        for img_file in extraction_dir.glob("*_multi_char_detection.jpg"):
            # 統合サーバー経由でアクセス可能なパスを生成
            relative_path = f"QI-006/extraction/{img_file.name}"
            available_images[img_file.name] = relative_path
            print(f"✅ 画像発見: {img_file.name}")

        # 抽出後画像も含める
        for img_file in extraction_dir.glob("*.jpg"):
            if "_multi_char_detection" not in img_file.name:
                relative_path = f"QI-006/extraction/{img_file.name}"
                available_images[img_file.name] = relative_path
                print(f"✅ 抽出画像発見: {img_file.name}")

    # 統計データ整理
    multiple_char_results = [r for r in results if r.get("is_multiple", False)]
    high_penalty_results = [r for r in results if r.get("penalty_score", 0) > 0.7]

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QI-006: 抽出後複数キャラ残存検出システム - ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .badge-high {{ @apply bg-red-100 text-red-800 px-2 py-1 rounded-full text-sm font-medium; }}
        .badge-medium {{ @apply bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-sm font-medium; }}
        .badge-low {{ @apply bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm font-medium; }}
        .detection-image {{ max-height: 400px; object-fit: contain; }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- ヘッダー -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">QI-006: 抽出後複数キャラ残存検出システム</h1>
            <p class="text-gray-600">抽出処理後も複数キャラクターが残存している問題画像の検出・品質評価システム実行結果</p>
            <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-blue-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-blue-600">処理画像数</h3>
                    <p class="text-2xl font-bold text-blue-800">{stats['total_images']}枚</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-green-600">成功率</h3>
                    <p class="text-2xl font-bold text-green-800">{stats['success_rate']:.1f}%</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-red-600">抽出後問題率</h3>
                    <p class="text-2xl font-bold text-red-800">{stats['multiple_character_rate']:.1f}%</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-purple-600">処理時間</h3>
                    <p class="text-2xl font-bold text-purple-800">{stats.get('average_processing_time', 0):.2f}s/枚</p>
                </div>
            </div>
        </div>

        <!-- ペナルティレベル分析 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🎯 ペナルティレベル分析</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-red-600">{stats['high_penalty']}</div>
                    <div class="text-sm text-gray-600">重大な抽出問題 (>0.7)</div>
                    <div class="text-xs text-red-600">抽出処理改善必須</div>
                </div>
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-yellow-600">{stats['medium_penalty']}</div>
                    <div class="text-sm text-gray-600">軽微な抽出問題 (0.3-0.7)</div>
                    <div class="text-xs text-yellow-600">処理見直し検討</div>
                </div>
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-green-600">{stats['low_penalty']}</div>
                    <div class="text-sm text-gray-600">抽出品質良好 (≤0.3)</div>
                    <div class="text-xs text-green-600">使用推奨</div>
                </div>
            </div>
        </div>

        <!-- 検出タイプ分布 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🏷️ 検出タイプ分布</h2>
            <div class="space-y-2">
"""

    # 検出タイプ分布表示
    for det_type, count in stats["detection_types"].items():
        percentage = count / stats["total_images"] * 100
        html += f"""
                <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <span class="font-medium">{det_type}</span>
                    <span class="text-sm text-gray-600">{count}枚 ({percentage:.1f}%)</span>
                </div>
"""

    html += """
            </div>
        </div>

        <!-- 高ペナルティ画像一覧 -->
"""

    if high_penalty_results:
        html += f"""
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🚨 重大な抽出問題画像 ({len(high_penalty_results)}枚)</h2>
            <p class="text-sm text-red-600 mb-4">抽出処理後も複数キャラが残存している問題画像（抽出アルゴリズム改善必須）</p>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
"""

        for result in high_penalty_results:
            image_name = result["image_name"]
            vis_name = image_name.replace(".jpg", "_multi_char_detection.jpg")

            html += f"""
                <div class="border rounded-lg p-4">
                    <h4 class="font-medium mb-2">{image_name}</h4>
"""

            # 可視化画像と抽出後画像の両方を表示
            vis_url = None
            extract_url = None

            if vis_name in available_images:
                vis_url = f"/{available_images[vis_name]}"
            if image_name in available_images:
                extract_url = f"/{available_images[image_name]}"

            # 可視化画像がある場合は表示
            if vis_url:
                html += f"""
                    <div class="mb-3">
                        <div class="text-xs text-blue-600 font-medium mb-1">🔍 複数キャラ検出結果</div>
                        <img src="{vis_url}" 
                             alt="検出結果: {vis_name}" class="detection-image w-full mb-2 rounded border">
                    </div>
"""

            # 抽出後画像がある場合は表示
            if extract_url:
                html += f"""
                    <div class="mb-3">
                        <div class="text-xs text-green-600 font-medium mb-1">📷 実際の抽出画像</div>
                        <img src="{extract_url}" 
                             alt="抽出画像: {image_name}" class="detection-image w-full mb-2 rounded border">
                    </div>
"""

            html += f"""
                    <div class="space-y-1 text-sm">
                        <div>キャラクター数: <span class="font-medium">{result.get('character_count', 0)}体</span></div>
                        <div>検出タイプ: <span class="font-medium">{result.get('detection_type', '')}</span></div>
                        <div>ペナルティ: <span class="badge-high">{result.get('penalty_score', 0):.3f}</span></div>
                        <div>処理時間: <span class="text-gray-600">{result.get('processing_time', 0):.2f}s</span></div>
                    </div>
"""

            # 改善提案表示
            suggestions = result.get("improvement_suggestions", [])
            if suggestions:
                html += """
                    <div class="mt-2">
                        <div class="text-xs text-gray-600 mb-1">改善提案:</div>
                        <ul class="text-xs text-gray-600 space-y-1">
"""
                for suggestion in suggestions[:3]:  # 最大3件表示
                    html += f"<li>• {suggestion}</li>"

                html += """
                        </ul>
                    </div>
"""

            html += """
                </div>
"""

        html += """
            </div>
        </div>
"""

    # 単一キャラクター推奨画像一覧
    single_char_results = [
        r for r in results if not r.get("is_multiple", False) or r.get("penalty_score", 0) <= 0.3
    ]

    if single_char_results:
        html += f"""
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">✅ LoRA学習推奨画像 ({len(single_char_results)}枚)</h2>
            <p class="text-sm text-green-600 mb-4">抽出後も単一キャラクターまたは軽微な問題のみの良好品質画像</p>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
"""

        for result in single_char_results[:16]:  # 最大16件表示
            image_name = result["image_name"]
            penalty = result.get("penalty_score", 0)
            badge_class = "badge-low" if penalty <= 0.3 else "badge-medium"

            # 可視化画像と抽出後画像の両方を探す
            vis_name = image_name.replace(".jpg", "_multi_char_detection.jpg")
            vis_url = None
            extract_url = None

            if vis_name in available_images:
                vis_url = f"/{available_images[vis_name]}"
            if image_name in available_images:
                extract_url = f"/{available_images[image_name]}"

            html += f"""
                <div class="border rounded-lg p-3">
                    <h4 class="text-sm font-medium mb-2">{image_name}</h4>
"""

            # 可視化画像がある場合は表示
            if vis_url:
                html += f"""
                    <div class="mb-2">
                        <div class="text-xs text-gray-500 mb-1">🔍 検出結果</div>
                        <img src="{vis_url}" 
                             alt="検出結果: {vis_name}" class="mb-2 border rounded">
                    </div>
"""

            # 抽出後画像がある場合は表示
            if extract_url:
                html += f"""
                    <div class="mb-2">
                        <div class="text-xs text-gray-500 mb-1">📷 抽出画像</div>
                        <img src="{extract_url}" 
                             alt="抽出画像: {image_name}" class="mb-2 border rounded">
                    </div>
"""

            html += f"""
                    <div class="text-center">
                        <div class="{badge_class} mb-1">{penalty:.3f}</div>
                        <div class="text-xs text-gray-500">{result.get('character_count', 0)}体検出</div>
                    </div>
                </div>
"""

        if len(single_char_results) > 16:
            html += f"""
                <div class="text-center p-2 border rounded bg-gray-50">
                    <div class="text-xs text-gray-600">他{len(single_char_results) - 16}枚</div>
                </div>
"""

        html += """
            </div>
        </div>
"""

    # フッター
    html += f"""
        <!-- システム効果サマリー -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🎉 システム効果</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-medium text-gray-800 mb-2">フィルタリング効果</h3>
                    <ul class="text-sm text-gray-600 space-y-1">
                        <li>• 抽出後問題画像{stats['multiple_character']}枚を自動識別</li>
                        <li>• 重大問題{stats['high_penalty']}枚の抽出処理改善推奨</li>
                        <li>• 抽出品質向上によりLoRA学習効率改善</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-medium text-gray-800 mb-2">処理効率</h3>
                    <ul class="text-sm text-gray-600 space-y-1">
                        <li>• {stats.get('average_processing_time', 0):.2f}秒/枚の高速処理</li>
                        <li>• 手動確認不要の自動品質判定</li>
                        <li>• 可視化による結果確認サポート</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- フッター -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>QI-006: 抽出後複数キャラ残存検出システム実行結果</p>
            <p>生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

    return html


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
