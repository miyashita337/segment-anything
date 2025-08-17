#!/usr/bin/env python3
"""
汎用統計分析ダッシュボード・レポート生成システム

BASELINE-RECALC-001から始まる汎用統計分析システム用の
ダッシュボード・レポート自動生成システム

機能:
    - 統計結果ダッシュボード生成（HTML）
    - 改善率推移グラフ（Chart.js）
    - 画像ギャラリー（パスベース表示）
    - 詳細レポート（Markdown）
    - トラッカーワークスペース統合

Created for: BASELINE-RECALC-001 汎用統計分析システム実装
Author: Claude Code Integration System
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

# from tools.progress_tracker.universal_statistical_analyzer import UniversalStatisticalResult
# 循環インポート回避のため、実行時インポートを使用

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""
    use_path_based_images: bool = True
    max_images_per_gallery: int = 20
    server_base_url: str = "http://100.123.241.106:8088"
    chart_color_scheme: List[str] = None
    
    def __post_init__(self):
        if self.chart_color_scheme is None:
            self.chart_color_scheme = [
                "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#fd79a8",
                "#fdcb6e", "#6c5ce7", "#a29bfe", "#fd79a8", "#e17055"
            ]


class UniversalDashboardGenerator:
    """汎用統計分析ダッシュボード生成器"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """初期化"""
        self.config = config or DashboardConfig()
        logger.info("🎨 汎用ダッシュボード生成器初期化完了")
    
    def generate_html_dashboard(
        self, 
        result: Any,  # UniversalStatisticalResult
        workspace_dir: Path,
        image_paths: List[str] = None
    ) -> str:
        """
        HTML統計ダッシュボード生成
        
        Args:
            result: 統計分析結果
            workspace_dir: ワークスペースディレクトリ
            image_paths: 画像パスリスト
            
        Returns:
            str: HTML内容
        """
        try:
            # 画像パス取得
            if image_paths is None:
                image_paths = self._collect_image_paths(workspace_dir)
            
            # 改善率グラフデータ準備
            chart_data = self._prepare_chart_data(result, workspace_dir)
            
            # 統計サマリーセクション
            stats_section = self._generate_stats_section(result)
            
            # 画像ギャラリーセクション
            gallery_section = self._generate_image_gallery(result.current_tracker, image_paths)
            
            # Chart.jsグラフセクション
            chart_section = self._generate_chart_section(chart_data)
            
            # HTML統合
            html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>統計分析ダッシュボード - {result.current_tracker}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .stat-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .improvement-positive {{ color: #10b981; }}
        .improvement-negative {{ color: #ef4444; }}
        .significance-significant {{ background-color: #10b981; }}
        .significance-non-significant {{ background-color: #6b7280; }}
        .effect-size-large {{ border-left: 4px solid #10b981; }}
        .effect-size-medium {{ border-left: 4px solid #f59e0b; }}
        .effect-size-small {{ border-left: 4px solid #ef4444; }}
        
        /* QCA-001準拠の画像ギャラリースタイル */
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
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- ヘッダー -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">🔬 汎用統計分析ダッシュボード</h1>
            <p class="text-gray-600">トラッカー: <span class="font-semibold text-blue-600">{result.current_tracker}</span> vs <span class="font-semibold text-green-600">{result.baseline_tracker}</span></p>
            <p class="text-sm text-gray-500">生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 作者: {result.author_name}</p>
        </div>

        {stats_section}

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {chart_section}
            
            <!-- 詳細統計 -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-xl font-bold text-gray-800 mb-4">📊 詳細統計</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">サンプルサイズ（現在）:</span>
                        <span class="font-semibold">{result.current_sample_size}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">サンプルサイズ（ベースライン）:</span>
                        <span class="font-semibold">{result.baseline_sample_size}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">95%信頼区間:</span>
                        <span class="font-semibold">[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">統計的有意性:</span>
                        <span class="px-2 py-1 rounded text-white text-sm significance-{'significant' if result.is_significant else 'non-significant'}">
                            {'有意' if result.is_significant else '非有意'}
                        </span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">実用的意義:</span>
                        <span class="font-semibold">{result.practical_significance}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">サンプル妥当性:</span>
                        <span class="font-semibold">{result.sample_adequacy}</span>
                    </div>
                </div>
            </div>
        </div>

        {gallery_section}
        
        <!-- フッター -->
        <div class="bg-white rounded-lg shadow-lg p-4 text-center text-gray-500 text-sm">
            🤖 Generated with <a href="https://claude.ai/code" class="text-blue-600 hover:underline">Claude Code</a> | 
            Universal Statistical Analyzer v1.0 | 
            <a href="{self.config.server_base_url}/tracker/{result.current_tracker}" class="text-blue-600 hover:underline">Live Dashboard</a>
        </div>
    </div>
</body>
</html>"""
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ HTMLダッシュボード生成エラー: {e}")
            raise
    
    def _generate_stats_section(self, result: Any) -> str:
        """統計サマリーセクション生成"""
        improvement_class = "improvement-positive" if result.improvement_rate >= 0 else "improvement-negative"
        improvement_icon = "📈" if result.improvement_rate >= 0 else "📉"
        
        effect_size_class = ""
        if abs(result.cohens_d) >= 0.8:
            effect_size_class = "effect-size-large"
        elif abs(result.cohens_d) >= 0.5:
            effect_size_class = "effect-size-medium"
        else:
            effect_size_class = "effect-size-small"
        
        return f"""
        <!-- 統計サマリー -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="stat-card rounded-lg p-6 text-center">
                <h3 class="text-lg font-semibold mb-2">Current Score</h3>
                <p class="text-3xl font-bold">{result.current_score:.3f}</p>
                <p class="text-sm opacity-80">現在の品質スコア</p>
            </div>
            
            <div class="stat-card rounded-lg p-6 text-center">
                <h3 class="text-lg font-semibold mb-2">Baseline Score</h3>
                <p class="text-3xl font-bold">{result.baseline_score:.3f}</p>
                <p class="text-sm opacity-80">ベースライン品質スコア</p>
            </div>
            
            <div class="bg-white rounded-lg p-6 text-center {effect_size_class}">
                <h3 class="text-lg font-semibold mb-2 text-gray-800">{improvement_icon} 改善率</h3>
                <p class="text-3xl font-bold {improvement_class}">{result.improvement_rate:+.1f}%</p>
                <p class="text-sm text-gray-600">{result.interpretation}</p>
            </div>
            
            <div class="bg-white rounded-lg p-6 text-center">
                <h3 class="text-lg font-semibold mb-2 text-gray-800">Cohen's d</h3>
                <p class="text-3xl font-bold text-purple-600">{result.cohens_d:.3f}</p>
                <p class="text-sm text-gray-600">効果サイズ (p={result.p_value:.4f})</p>
            </div>
        </div>
        """
    
    def _generate_image_gallery(self, tracker_id: str, image_paths: List[str]) -> str:
        """画像ギャラリーセクション生成"""
        if not image_paths:
            return """
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h3 class="text-xl font-bold text-gray-800 mb-4">🖼️ 抽出画像ギャラリー</h3>
                <p class="text-gray-500">画像が見つかりませんでした。</p>
            </div>
            """
        
        # 最大表示数制限
        displayed_paths = image_paths[:self.config.max_images_per_gallery]
        
        gallery_items = ""
        for i, img_path in enumerate(displayed_paths):
            # QCA-001準拠の相対パス形式（/トラッカーID/extraction/ファイル名）
            filename = Path(img_path).name
            relative_url = f"/{tracker_id}/extraction/{filename}"
            
            # 簡易品質判定（ファイルサイズベース）
            try:
                file_size = Path(img_path).stat().st_size
                size_kb = file_size // 1024
                if size_kb > 150:
                    quality_badge = '<div class="quality-badge high">高品質</div>'
                    quality_text = "高品質"
                elif size_kb > 80:
                    quality_badge = '<div class="quality-badge medium">中品質</div>'
                    quality_text = "中品質"
                else:
                    quality_badge = '<div class="quality-badge low">低品質</div>'
                    quality_text = "低品質"
            except:
                quality_badge = '<div class="quality-badge medium">-</div>'
                quality_text = "-"
                size_kb = 0
            
            gallery_items += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="{relative_url}" alt="{filename}" loading="lazy">
                {quality_badge}
            </div>
            <div class="image-info">
                <div class="image-name">{filename}</div>
                <div class="image-details">
                    <span>{size_kb} KB</span>
                    <span>{quality_text}</span>
                </div>
            </div>
        </div>"""
        
        return f"""
        <div class="gallery">
            <h2>🖼️ 抽出画像ギャラリー</h2>
            <div class="author-section">
                <h3>📊 統計分析結果画像（{tracker_id}）</h3>
                <div class="images-grid">
                    {gallery_items}
                </div>
            </div>
        </div>
        """
    
    def _generate_chart_section(self, chart_data: Dict[str, Any]) -> str:
        """Chart.jsグラフセクション生成"""
        return f"""
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h3 class="text-xl font-bold text-gray-800 mb-4">📈 改善率推移</h3>
            <canvas id="improvementChart" width="400" height="200"></canvas>
        </div>
        
        <script>
            const ctx = document.getElementById('improvementChart').getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'line',
                data: {json.dumps(chart_data, ensure_ascii=False)},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: '品質スコア推移'
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            title: {{
                                display: true,
                                text: '品質スコア'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'トラッカー'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """
    
    def _prepare_chart_data(self, result: Any, workspace_dir: Path) -> Dict[str, Any]:
        """チャート用データ準備"""
        try:
            # 基本的な2点データ（現在と過去の実装）
            labels = [result.baseline_tracker or "Baseline", result.current_tracker]
            scores = [result.baseline_score, result.current_score]
            
            # 追加のトラッカーデータ取得試行
            additional_data = self._collect_additional_tracker_data(workspace_dir.parent)
            
            if additional_data:
                labels.extend(additional_data['labels'])
                scores.extend(additional_data['scores'])
            
            chart_data = {
                "labels": labels,
                "datasets": [{
                    "label": "品質スコア",
                    "data": scores,
                    "borderColor": self.config.chart_color_scheme[0],
                    "backgroundColor": f"{self.config.chart_color_scheme[0]}20",
                    "tension": 0.1,
                    "pointRadius": 6,
                    "pointHoverRadius": 8
                }]
            }
            
            return chart_data
            
        except Exception as e:
            logger.warning(f"⚠️ チャートデータ準備エラー: {e}")
            # フォールバック: 基本データのみ
            return {
                "labels": [result.baseline_tracker or "Baseline", result.current_tracker],
                "datasets": [{
                    "label": "品質スコア",
                    "data": [result.baseline_score, result.current_score],
                    "borderColor": "#4ecdc4",
                    "backgroundColor": "#4ecdc420",
                    "tension": 0.1
                }]
            }
    
    def _collect_additional_tracker_data(self, workspace_base: Path) -> Optional[Dict[str, List]]:
        """追加のトラッカーデータ収集"""
        try:
            additional_trackers = []
            
            # ワークスペース内の他のトラッカーを検索
            for tracker_dir in workspace_base.iterdir():
                if tracker_dir.is_dir() and tracker_dir.name.startswith(('QCC-', 'QI-', 'P1-')):
                    result_file = tracker_dir / "extraction_result.json"
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'results' in data and data['results']:
                                scores = [r.get('quality_score', 0) for r in data['results'] if r.get('quality_score')]
                                if scores:
                                    avg_score = sum(scores) / len(scores)
                                    additional_trackers.append({
                                        'tracker_id': tracker_dir.name,
                                        'avg_score': avg_score
                                    })
            
            if additional_trackers:
                # トラッカーID順でソート
                additional_trackers.sort(key=lambda x: x['tracker_id'])
                
                return {
                    'labels': [t['tracker_id'] for t in additional_trackers[-5:]],  # 最新5件
                    'scores': [t['avg_score'] for t in additional_trackers[-5:]]
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 追加トラッカーデータ収集エラー: {e}")
            return None
    
    def _collect_image_paths(self, workspace_dir: Path) -> List[str]:
        """ワークスペースから画像パス収集"""
        try:
            image_paths = []
            extraction_dir = workspace_dir / "extraction"
            
            if extraction_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend([str(p) for p in extraction_dir.glob(ext)])
            
            return sorted(image_paths)
            
        except Exception as e:
            logger.warning(f"⚠️ 画像パス収集エラー: {e}")
            return []
    
    def generate_markdown_report(self, result: Any) -> str:
        """
        Markdown詳細レポート生成
        
        Args:
            result: 統計分析結果
            
        Returns:
            str: Markdownレポート内容
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            report_content = f"""# 🔬 汎用統計分析レポート

## 📋 基本情報

- **現在のトラッカー**: {result.current_tracker}
- **ベースライントラッカー**: {result.baseline_tracker}
- **作者**: {result.author_name}
- **分析日時**: {timestamp}
- **分析成功**: {'✅ 成功' if result.success else '❌ 失敗'}

## 📊 統計分析結果

### 主要指標

| 項目 | 値 |
|------|------------|
| **Current Score** | {result.current_score:.3f} |
| **Baseline Score** | {result.baseline_score:.3f} |
| **改善率** | {result.improvement_rate:+.1f}% |
| **p値** | {result.p_value:.4f} |
| **Cohen's d** | {result.cohens_d:.3f} |
| **統計的有意性** | {'有意' if result.is_significant else '非有意'} |

### 詳細統計

- **95%信頼区間**: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]
- **実用的意義**: {result.practical_significance}
- **解釈**: {result.interpretation}
- **サンプル妥当性**: {result.sample_adequacy}

### サンプルサイズ

- **現在のサンプル数**: {result.current_sample_size}
- **ベースラインサンプル数**: {result.baseline_sample_size}
- **総サンプル数**: {result.current_sample_size + result.baseline_sample_size}

## 🎯 解釈と推奨事項

### 統計的解釈

{self._generate_statistical_interpretation(result)}

### 推奨事項

{self._generate_recommendations(result)}

## 📈 データ品質評価

{self._generate_data_quality_assessment(result)}

## 🔗 関連リソース

- **ダッシュボード**: [Live Dashboard]({self.config.server_base_url}/tracker/{result.current_tracker})
- **Google Sheets**: 統計データ N-S列に自動更新済み
- **ワークスペース**: `/mnt/c/AItools/lora/train/{result.author_name}/tracker-workspace/{result.current_tracker}/`

---

*🤖 Generated with [Claude Code](https://claude.ai/code) | Universal Statistical Analyzer v1.0*
"""
            
            return report_content
            
        except Exception as e:
            logger.error(f"❌ Markdownレポート生成エラー: {e}")
            raise
    
    def _generate_statistical_interpretation(self, result: Any) -> str:
        """統計的解釈セクション生成"""
        interpretation_parts = []
        
        # 効果サイズ解釈
        abs_d = abs(result.cohens_d)
        if abs_d >= 0.8:
            interpretation_parts.append("**効果サイズ**: 大きな効果が観測されました。")
        elif abs_d >= 0.5:
            interpretation_parts.append("**効果サイズ**: 中程度の効果が観測されました。")
        elif abs_d >= 0.2:
            interpretation_parts.append("**効果サイズ**: 小さな効果が観測されました。")
        else:
            interpretation_parts.append("**効果サイズ**: 効果は観測されませんでした。")
        
        # 統計的有意性解釈
        if result.is_significant:
            interpretation_parts.append("**統計的有意性**: 統計的に有意な差が確認されました（p < 0.05）。")
        else:
            interpretation_parts.append("**統計的有意性**: 統計的に有意な差は確認されませんでした（p ≥ 0.05）。")
        
        # 実用性解釈
        interpretation_parts.append(f"**実用的意義**: {result.practical_significance}")
        
        return "\n".join(interpretation_parts)
    
    def _generate_recommendations(self, result: Any) -> str:
        """推奨事項セクション生成"""
        recommendations = []
        
        # サンプルサイズベース推奨
        if "不足" in result.sample_adequacy:
            recommendations.append("- **サンプルサイズ増加**: より信頼性の高い結果のため、サンプルサイズの増加を推奨します。")
        
        # 改善率ベース推奨
        if result.improvement_rate > 10:
            recommendations.append("- **手法の継続**: 大幅な改善が確認されているため、現在の手法を継続してください。")
        elif result.improvement_rate < -5:
            recommendations.append("- **手法の見直し**: 品質低下が確認されているため、手法の見直しを推奨します。")
        else:
            recommendations.append("- **継続観察**: 改善効果は限定的です。さらなるデータ収集で傾向を確認してください。")
        
        # 統計的有意性ベース推奨
        if not result.is_significant and abs(result.cohens_d) > 0.5:
            recommendations.append("- **追加検証**: 効果サイズは大きいが統計的有意性が確認されていません。追加データでの検証を推奨します。")
        
        return "\n".join(recommendations) if recommendations else "- 現在の結果に基づく特別な推奨事項はありません。"
    
    def _generate_data_quality_assessment(self, result: Any) -> str:
        """データ品質評価セクション生成"""
        quality_points = []
        
        # サンプルサイズ評価
        total_samples = result.current_sample_size + result.baseline_sample_size
        if total_samples >= 50:
            quality_points.append("✅ **サンプルサイズ**: 十分な量のデータが確保されています。")
        elif total_samples >= 20:
            quality_points.append("⚠️ **サンプルサイズ**: 中程度のデータ量です。より多くのデータがあれば信頼性が向上します。")
        else:
            quality_points.append("❌ **サンプルサイズ**: データ量が不足しています。結果の解釈には注意が必要です。")
        
        # バランス評価
        balance_ratio = min(result.current_sample_size, result.baseline_sample_size) / max(result.current_sample_size, result.baseline_sample_size)
        if balance_ratio >= 0.8:
            quality_points.append("✅ **データバランス**: 現在とベースラインのサンプル数がバランスよく配分されています。")
        elif balance_ratio >= 0.5:
            quality_points.append("⚠️ **データバランス**: やや不均衡ですが、分析には問題ありません。")
        else:
            quality_points.append("❌ **データバランス**: サンプル数に大きな偏りがあります。結果の解釈に注意してください。")
        
        return "\n".join(quality_points)
    
    def generate_complete_dashboard(
        self, 
        result: Any,  # UniversalStatisticalResult
        workspace_dir: Path,
        save_html: bool = True,
        save_markdown: bool = True
    ) -> Dict[str, Path]:
        """
        完全なダッシュボード・レポート生成
        
        Args:
            result: 統計分析結果
            workspace_dir: ワークスペースディレクトリ
            save_html: HTML保存フラグ
            save_markdown: Markdown保存フラグ
            
        Returns:
            Dict[str, Path]: 生成ファイルパス辞書
        """
        try:
            generated_files = {}
            
            # ダッシュボードディレクトリ作成
            dashboard_dir = workspace_dir / "dashboard"
            dashboard_dir.mkdir(exist_ok=True)
            
            # HTML ダッシュボード生成
            if save_html:
                html_content = self.generate_html_dashboard(result, workspace_dir)
                html_file = dashboard_dir / "dashboard.html"
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                generated_files['html'] = html_file
                logger.info(f"✅ HTMLダッシュボード生成: {html_file}")
            
            # Markdown レポート生成
            if save_markdown:
                markdown_content = self.generate_markdown_report(result)
                markdown_file = dashboard_dir / f"{result.current_tracker}_statistical_report.md"
                
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                generated_files['markdown'] = markdown_file
                logger.info(f"✅ Markdownレポート生成: {markdown_file}")
            
            # メタデータ保存
            metadata = {
                'tracker_id': result.current_tracker,
                'baseline_tracker': result.baseline_tracker,
                'author_name': result.author_name,
                'generation_timestamp': datetime.now().isoformat(),
                'analysis_timestamp': result.analysis_timestamp,
                'generated_files': {k: str(v) for k, v in generated_files.items()},
                'dashboard_url': f"{self.config.server_base_url}/tracker/{result.current_tracker}",
                'summary': {
                    'improvement_rate': float(result.improvement_rate),
                    'cohens_d': float(result.cohens_d),
                    'is_significant': bool(result.is_significant),
                    'practical_significance': str(result.practical_significance)
                }
            }
            
            metadata_file = dashboard_dir / "dashboard_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            generated_files['metadata'] = metadata_file
            
            logger.info(f"✅ 完全ダッシュボード生成完了: {len(generated_files)}ファイル")
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ 完全ダッシュボード生成エラー: {e}")
            raise


def main():
    """テスト実行"""
    logging.basicConfig(level=logging.INFO)
    
    # テストデータ作成（実行時インポート）
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # テスト用のダミーオブジェクト作成
    class TestResult:
        def __init__(self):
            self.success = True
            self.current_tracker = "TEST-001"
            self.baseline_tracker = "QCC-022"
            self.author_name = "yado"
            self.current_score = 0.825
            self.baseline_score = 0.756
            self.p_value = 0.0341
            self.cohens_d = 1.245
            self.improvement_rate = 9.1
            self.is_significant = True
            self.confidence_interval = (0.892, 1.598)
            self.practical_significance = "大きな実用的意義"
            self.interpretation = "大きな改善"
            self.sample_adequacy = "サンプルサイズやや不足"
            self.current_sample_size = 34
            self.baseline_sample_size = 28
            self.analysis_timestamp = datetime.now().isoformat()
    
    test_result = TestResult()
    
    # ダッシュボード生成器初期化
    generator = UniversalDashboardGenerator()
    
    # テスト用ワークスペース
    test_workspace = Path("/tmp/test_dashboard")
    test_workspace.mkdir(exist_ok=True)
    
    try:
        # 完全ダッシュボード生成
        generated_files = generator.generate_complete_dashboard(
            test_result, 
            test_workspace,
            save_html=True,
            save_markdown=True
        )
        
        print(f"✅ テスト完了: {len(generated_files)}ファイル生成")
        for file_type, file_path in generated_files.items():
            print(f"   {file_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ テスト失敗: {e}")


if __name__ == "__main__":
    main()