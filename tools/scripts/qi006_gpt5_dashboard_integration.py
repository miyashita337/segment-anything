#!/usr/bin/env python3
"""
QI-006: GPT-5評価結果のダッシュボード統合
GPT-5によるLoRA品質評価結果をQI-006ダッシュボードに追加表示
"""

import json
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """GPT-5評価結果のダッシュボード統合"""
    print("🔗 QI-006: GPT-5評価結果ダッシュボード統合開始")
    print("=" * 60)

    # パス設定
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
    qi006_workspace = workspace_base / "QI-006"

    quality_dir = qi006_workspace / "quality"
    dashboard_dir = qi006_workspace / "dashboard"

    # GPT-5評価結果読み込み
    gpt5_results_file = quality_dir / "gpt5_lora_quality_evaluation.json"
    if not gpt5_results_file.exists():
        print(f"❌ GPT-5評価結果が見つかりません: {gpt5_results_file}")
        return False

    with open(gpt5_results_file, "r", encoding="utf-8") as f:
        gpt5_data = json.load(f)

    print(f"📊 GPT-5評価データ読み込み完了")

    # 既存の複数キャラクター検出結果も読み込み
    detection_results_file = quality_dir / "qi006_detection_results.json"
    detection_data = None

    if detection_results_file.exists():
        with open(detection_results_file, "r", encoding="utf-8") as f:
            detection_data = json.load(f)
        print(f"📊 複数キャラクター検出データ読み込み完了")

    # 統合ダッシュボードHTML生成
    html_content = generate_integrated_dashboard(gpt5_data, detection_data)

    # ダッシュボード保存
    dashboard_file = dashboard_dir / "dashboard.html"
    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 統合ダッシュボード生成完了: {dashboard_file}")
    print(f"📏 ファイルサイズ: {dashboard_file.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"🌐 アクセスURL: http://100.123.241.106:8088/tracker/QI-006")

    return True


def generate_integrated_dashboard(gpt5_data, detection_data=None):
    """統合ダッシュボードHTML生成"""

    # GPT-5評価統計
    gpt5_summary = gpt5_data["evaluation_summary"]
    gpt5_grades = gpt5_data["grade_distribution"]["grade_distribution"]
    gpt5_results = gpt5_data["detailed_results"]

    # 複数キャラクター検出統計（利用可能な場合）
    detection_stats = None
    if detection_data:
        detection_stats = detection_data["statistics"]

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QI-006: GPT-5品質評価 + 複数キャラクター検出統合ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .grade-a {{ @apply bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-bold; }}
        .grade-b {{ @apply bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-bold; }}
        .grade-c {{ @apply bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-bold; }}
        .grade-d {{ @apply bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm font-bold; }}
        .grade-f {{ @apply bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-bold; }}
        .detection-image {{ max-height: 300px; object-fit: contain; }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- ヘッダー -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">QI-006: GPT-5品質評価統合ダッシュボード</h1>
            <p class="text-gray-600">GPT-5によるLoRA学習画像品質評価 + 複数キャラクター検出システム統合結果</p>
            <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-purple-600">総評価画像数</h3>
                    <p class="text-2xl font-bold text-purple-800">{gpt5_summary['total_images']}枚</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-green-600">GPT-5評価成功率</h3>
                    <p class="text-2xl font-bold text-green-800">{gpt5_summary['successful_evaluations']}/{gpt5_summary['total_images']} (100%)</p>
                </div>
                <div class="bg-blue-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-blue-600">A/B高品質画像</h3>
                    <p class="text-2xl font-bold text-blue-800">{gpt5_grades.get('A', 0) + gpt5_grades.get('B', 0)}枚</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-red-600">D/F低品質画像</h3>
                    <p class="text-2xl font-bold text-red-800">{gpt5_grades.get('D', 0) + gpt5_grades.get('F', 0)}枚</p>
                </div>
            </div>
        </div>

        <!-- GPT-5品質グレード分布 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🏆 GPT-5品質グレード分布</h2>
            <div class="grid grid-cols-5 gap-4 text-center">"""

    # グレード分布表示
    grade_descriptions = {
        "A": "LoRA学習に最適",
        "B": "LoRA学習に適している",
        "C": "注意が必要",
        "D": "問題あり",
        "F": "使用不可",
    }

    for grade in ["A", "B", "C", "D", "F"]:
        count = gpt5_grades.get(grade, 0)
        html += f"""
                <div class="p-4 border rounded-lg">
                    <div class="text-3xl font-bold grade-{grade.lower()}">{grade}</div>
                    <div class="text-xl font-bold text-gray-800">{count}枚</div>
                    <div class="text-xs text-gray-600 mt-1">{grade_descriptions[grade]}</div>
                </div>"""

    html += """
            </div>
        </div>

        <!-- 詳細評価結果：複数キャラクター検出結果 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🔍 複数キャラクター検出詳細結果</h2>
            {generate_detection_results_section(detection_data)}
        </div>

        <!-- 詳細評価結果：GPT-5品質評価 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">📋 GPT-5詳細評価結果</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full table-auto">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">画像名</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">グレード</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">人物数</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">抽出品質</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">明度</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">LoRA適合性</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">評価理由</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">"""

    # 複数キャラクター検出結果（利用可能な場合）
    if detection_stats:
        html += f"""
        <!-- 複数キャラクター検出統計 -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">🔍 複数キャラクター検出結果</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-green-600">{detection_stats.get('single_character', 0)}</div>
                    <div class="text-sm text-gray-600">単一キャラクター</div>
                </div>
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-red-600">{detection_stats.get('multiple_character', 0)}</div>
                    <div class="text-sm text-gray-600">複数キャラクター残存</div>
                </div>
                <div class="text-center p-4 border rounded-lg">
                    <div class="text-3xl font-bold text-blue-600">{detection_stats.get('success_rate', 0):.1f}%</div>
                    <div class="text-sm text-gray-600">検出成功率</div>
                </div>
            </div>
        </div>"""

    # 改善推奨事項
    html += f"""
        <!-- 改善推奨事項 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-xl font-bold text-gray-800 mb-4">💡 GPT-5による改善推奨事項</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-medium text-gray-800 mb-2">✅ LoRA学習推奨画像 ({gpt5_grades.get('A', 0) + gpt5_grades.get('B', 0)}枚)</h3>
                    <ul class="text-sm text-gray-600 space-y-1">
                        <li>• A評価: {gpt5_grades.get('A', 0)}枚 - そのまま使用可能</li>
                        <li>• B評価: {gpt5_grades.get('B', 0)}枚 - 軽微な調整で最適化可能</li>
                        <li>• 高品質画像活用により学習効率向上期待</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-medium text-gray-800 mb-2">⚠️ 要改善画像 ({gpt5_grades.get('C', 0) + gpt5_grades.get('D', 0) + gpt5_grades.get('F', 0)}枚)</h3>
                    <ul class="text-sm text-gray-600 space-y-1">
                        <li>• C評価: {gpt5_grades.get('C', 0)}枚 - 品質向上の余地あり</li>
                        <li>• D評価: {gpt5_grades.get('D', 0)}枚 - 使用非推奨</li>
                        <li>• F評価: {gpt5_grades.get('F', 0)}枚 - 抽出失敗・要再処理</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- フッター -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>QI-006: GPT-5品質評価 + 複数キャラクター検出統合システム実行結果</p>
            <p>GPT-5評価実行日時: {gpt5_summary.get('evaluation_timestamp', 'N/A')}</p>
            <p>統合ダッシュボード生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

    return html


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
