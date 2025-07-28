#!/usr/bin/env python3
"""
P1-A002: 品質基準統一システム - ダッシュボード生成スクリプト

統一品質評価結果の可視化ダッシュボード生成
PROGRESS_TRACKER.md準拠のワークフロー実装
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tools.core.unified_quality_standard import UnifiedQualityStandardSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


class P1A002DashboardGenerator:
    """P1-A002 ダッシュボード生成システム"""
    
    def __init__(self):
        """初期化"""
        self.project_root = project_root
        
        # PROGRESS_TRACKER.md準拠のワークスペース
        self.workspace_root = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
        self.workspace_dir = self.workspace_root / "P1-A002"
        self.dashboard_dir = self.workspace_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 統一品質基準システム
        self.quality_system = UnifiedQualityStandardSystem()
        
        print(f"🎯 P1-A002: ダッシュボード生成システム初期化完了")
        print(f"ダッシュボード出力: {self.dashboard_dir}")
    
    def load_integration_summary(self) -> Optional[Dict[str, Any]]:
        """統合サマリー読み込み"""
        logger.info("統合サマリー読み込み開始")
        
        # 最新の統合サマリーファイル検索
        summary_files = list(self.workspace_dir.glob("P1A002_integration_summary_*.json"))
        if not summary_files:
            logger.error("統合サマリーファイルが見つかりません")
            return None
        
        latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"最新統合サマリー: {latest_summary}")
        
        try:
            with open(latest_summary, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            logger.info("✅ 統合サマリー読み込み完了")
            return summary_data
            
        except Exception as e:
            logger.error(f"統合サマリー読み込みエラー: {e}")
            return None
    
    def load_quality_results(self) -> List[Dict[str, Any]]:
        """品質評価結果読み込み"""
        logger.info("品質評価結果読み込み開始")
        
        quality_dir = self.workspace_dir / "quality"
        result_files = list(quality_dir.glob("unified_quality_*.json"))
        
        if not result_files:
            logger.warning("品質評価結果ファイルが見つかりません")
            return []
        
        results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    # unified_quality_report以外のファイルのみ処理
                    if "unified_quality_report" not in result_file.name:
                        # 必要なキーが存在するか確認
                        required_keys = ["dataset_name", "unified_score", "quality_level"]
                        if all(key in result_data for key in required_keys):
                            results.append(result_data)
                        else:
                            logger.warning(f"必要なキーが不足: {result_file}")
            except Exception as e:
                logger.warning(f"結果ファイル読み込みエラー {result_file}: {e}")
        
        logger.info(f"品質評価結果読み込み完了: {len(results)}件")
        return results
    
    def generate_unified_score_chart(self, results: List[Dict[str, Any]]) -> Path:
        """統一スコアチャート生成"""
        logger.info("統一スコアチャート生成開始")
        
        # データ抽出
        datasets = [r["dataset_name"] for r in results]
        scores = [r["unified_score"] for r in results]
        quality_levels = [r["quality_level"] for r in results]
        
        # 色マッピング
        color_map = {
            "EXCELLENT": "#2E8B57",  # Sea Green
            "GOOD": "#4169E1",       # Royal Blue
            "ACCEPTABLE": "#FF8C00", # Dark Orange
            "POOR": "#DC143C"        # Crimson
        }
        colors = [color_map.get(level, "#808080") for level in quality_levels]
        
        # チャート作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(datasets, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # チャート装飾
        ax.set_title("P1-A002: Dataset Unified Quality Scores", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Dataset Name", fontsize=12, fontweight='bold')
        ax.set_ylabel("Unified Score", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # グリッド追加
        ax.grid(True, alpha=0.3, axis='y')
        
        # 統一スコア値をバーの上に表示
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 品質レベル凡例作成
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                          for level, color in color_map.items()]
        ax.legend(legend_elements, color_map.keys(), 
                 title="Quality Levels", loc='upper right', framealpha=0.9)
        
        # 閾値線追加
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='EXCELLENT Threshold')
        ax.axhline(y=0.7, color='blue', linestyle='--', alpha=0.7, label='GOOD Threshold')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='ACCEPTABLE Threshold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存
        chart_file = self.dashboard_dir / f"unified_score_chart_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"統一スコアチャート保存: {chart_file}")
        return chart_file
    
    def generate_metrics_radar_chart(self, results: List[Dict[str, Any]]) -> Path:
        """メトリクスレーダーチャート生成"""
        logger.info("メトリクスレーダーチャート生成開始")
        
        # データ準備
        metrics = ['AB Rate', 'SCI Score', 'PLA Score', 'PLE Score', 'Success Rate', 'Fill Ratio']
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 角度設定
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 円を閉じる
        
        # データセットごとにプロット
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            values = [
                result["ab_evaluation_rate"],
                result["sci_score"],
                result["pla_score"],
                result["ple_score"],
                result["success_rate"],
                result["avg_fill_ratio"]
            ]
            values += values[:1]  # 円を閉じる
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=result["dataset_name"], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # チャート装飾
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("P1-A002: Multi-Dataset Quality Metrics Radar", 
                    fontsize=14, fontweight='bold', pad=30)
        
        # グリッド線
        ax.grid(True, alpha=0.3)
        
        # 凡例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # 保存
        radar_file = self.dashboard_dir / f"metrics_radar_chart_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"メトリクスレーダーチャート保存: {radar_file}")
        return radar_file
    
    def generate_quality_distribution_pie(self, results: List[Dict[str, Any]]) -> Path:
        """品質分布円グラフ生成"""
        logger.info("品質分布円グラフ生成開始")
        
        # 品質レベル集計
        quality_counts = {}
        for result in results:
            level = result["quality_level"]
            quality_counts[level] = quality_counts.get(level, 0) + 1
        
        # データ準備
        labels = list(quality_counts.keys())
        sizes = list(quality_counts.values())
        colors = ['#2E8B57', '#4169E1', '#FF8C00', '#DC143C'][:len(labels)]
        
        # 円グラフ作成
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90,
                                         explode=[0.05] * len(labels))
        
        # テキスト装飾
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title("P1-A002: Quality Level Distribution", 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        pie_file = self.dashboard_dir / f"quality_distribution_pie_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(pie_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"品質分布円グラフ保存: {pie_file}")
        return pie_file
    
    def generate_comparison_heatmap(self, results: List[Dict[str, Any]]) -> Path:
        """比較ヒートマップ生成"""
        logger.info("比較ヒートマップ生成開始")
        
        # データ準備
        datasets = [r["dataset_name"] for r in results]
        metrics = ['Unified Score', 'AB Rate', 'SCI', 'PLA', 'PLE', 'Success Rate']
        
        data_matrix = []
        for result in results:
            row = [
                result["unified_score"],
                result["ab_evaluation_rate"],
                result["sci_score"],
                result["pla_score"],
                result["ple_score"],
                result["success_rate"]
            ]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 軸設定
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(datasets)
        
        # 値表示
        for i in range(len(datasets)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                             color="black" if value > 0.5 else "white", fontweight='bold')
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
        
        ax.set_title("P1-A002: Dataset Quality Metrics Heatmap", 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        heatmap_file = self.dashboard_dir / f"comparison_heatmap_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比較ヒートマップ保存: {heatmap_file}")
        return heatmap_file
    
    def generate_dashboard_html(self, summary: Dict[str, Any], 
                              chart_files: Dict[str, Path]) -> Path:
        """HTMLダッシュボード生成"""
        logger.info("HTMLダッシュボード生成開始")
        
        # HTML生成
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P1-A002: Quality Standards Unification Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .findings {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .findings ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .findings li {{
            margin-bottom: 10px;
        }}
        .actions {{
            background: #e8f8f5;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
        }}
        .actions ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .actions li {{
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 P1-A002: Quality Standards Unification Dashboard</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Datasets</h3>
                <div class="value">{summary['summary']['total_datasets_evaluated']}</div>
            </div>
            <div class="summary-card">
                <h3>Overall Quality</h3>
                <div class="value">{summary['summary']['overall_quality_level']}</div>
            </div>
            <div class="summary-card">
                <h3>Average Score</h3>
                <div class="value">{summary['summary']['average_unified_score']:.3f}</div>
            </div>
            <div class="summary-card">
                <h3>Standard Version</h3>
                <div class="value">{summary['summary']['unified_standard_version']}</div>
            </div>
        </div>
        
        <h2>📊 Quality Metrics Visualization</h2>
        <div class="chart-grid">
            <div class="chart-container">
                <h3>Unified Score Comparison</h3>
                <img src="{chart_files['unified_score'].name}" alt="Unified Score Chart">
            </div>
            <div class="chart-container">
                <h3>Quality Level Distribution</h3>
                <img src="{chart_files['quality_pie'].name}" alt="Quality Distribution Pie">
            </div>
            <div class="chart-container">
                <h3>Multi-Metrics Radar</h3>
                <img src="{chart_files['metrics_radar'].name}" alt="Metrics Radar Chart">
            </div>
            <div class="chart-container">
                <h3>Comparison Heatmap</h3>
                <img src="{chart_files['comparison_heatmap'].name}" alt="Comparison Heatmap">
            </div>
        </div>
        
        <h2>🔍 Key Findings</h2>
        <div class="findings">
            <ul>
"""
        
        for finding in summary['key_findings']:
            html_content += f"                <li>{finding}</li>\n"
        
        html_content += f"""
            </ul>
        </div>
        
        <h2>🎯 Recommended Actions</h2>
        <div class="actions">
            <ul>
"""
        
        for action in summary['next_actions']:
            html_content += f"                <li>{action}</li>\n"
        
        html_content += f"""
            </ul>
        </div>
        
        <div class="timestamp">
            Generated at: {summary['generated_at']}<br>
            Integration ID: {summary['integration_id']}
        </div>
    </div>
</body>
</html>
"""
        
        # HTML保存
        html_file = self.dashboard_dir / f"P1A002_dashboard_{datetime.now():%Y%m%d_%H%M%S}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTMLダッシュボード保存: {html_file}")
        return html_file
    
    def generate_full_dashboard(self) -> Dict[str, Any]:
        """フルダッシュボード生成"""
        logger.info("🎨 P1-A002 フルダッシュボード生成開始")
        start_time = datetime.now()
        
        try:
            # 1. データ読み込み
            summary = self.load_integration_summary()
            if summary is None:
                return {"success": False, "error": "統合サマリーが見つかりません"}
            
            results = self.load_quality_results()
            if not results:
                return {"success": False, "error": "品質評価結果が見つかりません"}
            
            # 2. チャート生成
            chart_files = {
                "unified_score": self.generate_unified_score_chart(results),
                "metrics_radar": self.generate_metrics_radar_chart(results),
                "quality_pie": self.generate_quality_distribution_pie(results),
                "comparison_heatmap": self.generate_comparison_heatmap(results)
            }
            
            # 3. HTMLダッシュボード生成
            html_file = self.generate_dashboard_html(summary, chart_files)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "processing_time": processing_time,
                "html_dashboard": str(html_file),
                "chart_files": {k: str(v) for k, v in chart_files.items()},
                "datasets_analyzed": len(results)
            }
            
            logger.info(f"✅ P1-A002 フルダッシュボード生成完了 (処理時間: {processing_time:.2f}秒)")
            return result
            
        except Exception as e:
            logger.error(f"ダッシュボード生成エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-A002: ダッシュボード生成")
    parser.add_argument("--full", action="store_true", help="フルダッシュボード生成")
    parser.add_argument("--charts-only", action="store_true", help="チャートのみ生成")
    
    args = parser.parse_args()
    
    generator = P1A002DashboardGenerator()
    
    if args.full:
        # フルダッシュボード生成
        result = generator.generate_full_dashboard()
        
        if result["success"]:
            print(f"🎨 P1-A002ダッシュボード生成完了")
            print(f"   分析データセット: {result['datasets_analyzed']}件")
            print(f"   処理時間: {result['processing_time']:.2f}秒")
            print(f"   HTMLダッシュボード: {result['html_dashboard']}")
            print(f"   チャートファイル: {len(result['chart_files'])}件")
            return 0
        else:
            print(f"❌ ダッシュボード生成失敗: {result['error']}")
            return 1
    
    elif args.charts_only:
        # チャートのみ生成
        results = generator.load_quality_results()
        if not results:
            print("❌ 品質評価結果が見つかりません")
            return 1
        
        chart_files = {
            "unified_score": generator.generate_unified_score_chart(results),
            "metrics_radar": generator.generate_metrics_radar_chart(results),
            "quality_pie": generator.generate_quality_distribution_pie(results),
            "comparison_heatmap": generator.generate_comparison_heatmap(results)
        }
        
        print(f"✅ チャート生成完了: {len(chart_files)}件")
        for name, path in chart_files.items():
            print(f"   {name}: {path}")
        
        return 0
    
    else:
        print("🎨 P1-A002: 品質基準統一システム - ダッシュボード生成")
        print("使用例:")
        print("  python p1a002_dashboard_generator.py --full")
        print("  python p1a002_dashboard_generator.py --charts-only")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())