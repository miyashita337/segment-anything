"""
シンプルなダッシュボード生成システム

仕様書に従った直接的なHTML生成（約100行）
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StandardDashboardGenerator:
    """標準ダッシュボード生成クラス - シンプル版"""

    def create_dashboard(self, tracker_id: str, workspace_dir: str,
                         extraction_result_path: Optional[str] = None) -> str:
        """ダッシュボード作成（シンプル）

        Args:
            tracker_id: トラッカーID
            workspace_dir: ワークスペースディレクトリ
            extraction_result_path: 抽出結果JSONパス（オプション）

        Returns:
            生成されたダッシュボードファイルのパス
        """
        workspace_path = Path(workspace_dir)

        # 抽出結果JSONパス決定
        if extraction_result_path is None:
            extraction_result_path = workspace_path / "extraction_result.json"

        # 出力パス決定
        dashboard_dir = workspace_path / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        dashboard_file = dashboard_dir / "dashboard.html"

        # データ読み込み
        with open(extraction_result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # HTML生成
        html_content = self._generate_html(tracker_id, data)

        # ダッシュボードディレクトリにextraction_result.jsonもコピー
        dashboard_extraction_result = dashboard_dir / "extraction_result.json"
        if not dashboard_extraction_result.exists():
            import shutil
            shutil.copy2(extraction_result_path, dashboard_extraction_result)
            print(f"📋 dashboard/extraction_result.json作成: {dashboard_extraction_result}")

        # ファイル出力
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(dashboard_file)

    def _generate_html(self, tracker_id: str, data: Dict[str, Any]) -> str:
        """シンプルなHTML生成（仕様書準拠修正版）"""
        
        # 基本統計（正しいキー名で取得）
        total = data.get('total_images', 0)
        successful = data.get('successful_extractions', 0)  # 修正: success_count → successful_extractions
        avg_quality = data.get('average_quality_score', 0.0)  # 修正: summary → 直接取得
        
        # 画像リスト（extraction_resultsとresults両方に対応）
        results = data.get('extraction_results', data.get('results', []))
        
        # 品質分布計算（正しいキー名で取得）
        quality_dist = {'高品質': 0, '中品質': 0, '低品質': 0, '要改善': 0}
        for r in results:
            if r.get('success'):
                score = r.get('quality_score', 0.0)  # 修正: quality_metrics.overall_score → quality_score
                if score >= 0.8:
                    quality_dist['高品質'] += 1
                elif score >= 0.6:
                    quality_dist['中品質'] += 1
                elif score >= 0.4:
                    quality_dist['低品質'] += 1
                else:
                    quality_dist['要改善'] += 1

        # 現在時刻（自然な表示）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 統計分析結果セクション追加
        statistical_section = ""
        if 'statistical_analysis' in data:
            stats = data['statistical_analysis']
            statistical_section = f'''
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">📊 統計分析結果</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4">
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">Current(平均品質スコア)</div>
                    <p class="text-lg font-bold text-blue-600">{stats.get('current_score', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">BaseLine</div>
                    <p class="text-lg font-bold text-gray-600">{stats.get('baseline_score', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">p値</div>
                    <p class="text-lg font-bold text-indigo-600">{stats.get('p_value', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">効果サイズ、Cohen's d</div>
                    <p class="text-lg font-bold text-purple-600">{stats.get('effect_size', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">改善率</div>
                    <p class="text-lg font-bold text-green-600">{stats.get('improvement_rate', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">統計的有意性</div>
                    <p class="text-lg font-bold text-red-600">{stats.get('significance', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">信頼区間</div>
                    <p class="text-lg font-bold text-teal-600">{stats.get('confidence_interval', 'N/A')}</p>
                </div>
            </div>
        </div>'''

        # HTML生成
        html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} - 品質評価ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">{tracker_id} 品質評価ダッシュボード</h1>
            <p class="text-gray-600">生成日時: {current_time}</p>
        </header>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">総画像数</h3>
                <p class="text-3xl font-bold text-blue-600">{total}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">平均品質スコア</h3>
                <p class="text-3xl font-bold text-green-600">{avg_quality:.3f}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">成功画像数</h3>
                <p class="text-3xl font-bold text-emerald-600">{successful}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">要改善数</h3>
                <p class="text-3xl font-bold text-red-600">{quality_dist["要改善"]}</p>
            </div>
        </div>
        {statistical_section}
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">品質分布</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <div class="bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">高品質</div>
                    <p class="text-2xl font-bold">{quality_dist['高品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">中品質</div>
                    <p class="text-2xl font-bold">{quality_dist['中品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">低品質</div>
                    <p class="text-2xl font-bold">{quality_dist['低品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">要改善</div>
                    <p class="text-2xl font-bold">{quality_dist['要改善']}</p>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">🖼️ 抽出結果ギャラリー</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">'''

        # 画像カード生成（修正版）
        for r in results:
            if r.get('success'):
                # image_nameから実際のファイル名を取得（修正）
                filename = r.get('image_name', 'unknown.jpg')  # 修正: image_path → image_name
                score = r.get('quality_score', 0.0)  # 修正: quality_metrics.overall_score → quality_score
                quality_label = self._get_quality_label(score)
                quality_class = self._get_quality_class(score)
                
                html += f'''
                <div class="border rounded-lg p-3 bg-gray-50">
                    <img src="/{tracker_id}/extraction/{filename}" 
                         alt="{filename}" 
                         class="w-full object-contain rounded" 
                         onerror="this.parentElement.innerHTML='<div class=\\"p-4 text-center text-gray-500\\">画像読み込みエラー<br>{filename}</div>'">
                    <div class="mt-2 text-center">
                        <span class="{quality_class}">{quality_label}</span>
                        <p class="text-sm text-gray-600 mt-1">{filename}</p>
                        <p class="text-xs text-gray-500 mt-1">スコア: {score:.3f}</p>
                    </div>
                </div>'''

        html += '''
            </div>
        </div>
    </div>
</body>
</html>'''
        
        return html

    def _get_quality_label(self, score: float) -> str:
        """品質ラベル取得"""
        if score >= 0.8:
            return '高品質'
        elif score >= 0.6:
            return '中品質'
        elif score >= 0.4:
            return '低品質'
        else:
            return '要改善'
    
    def _get_quality_class(self, score: float) -> str:
        """品質クラス取得"""
        if score >= 0.8:
            return 'bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold'
        elif score >= 0.6:
            return 'bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold'
        elif score >= 0.4:
            return 'bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold'
        else:
            return 'bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold'