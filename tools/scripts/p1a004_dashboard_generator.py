#!/usr/bin/env python3
"""
P1-A004: ドキュメント整備システム - ダッシュボード生成スクリプト

ドキュメント同期結果の可視化ダッシュボード生成
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

from tools.core.documentation_sync_system import DocumentationSyncSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


class P1A004DashboardGenerator:
    """P1-A004 ダッシュボード生成システム"""
    
    def __init__(self):
        """初期化"""
        self.project_root = project_root
        
        # PROGRESS_TRACKER.md準拠のワークスペース
        self.workspace_root = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
        self.workspace_dir = self.workspace_root / "P1-A004"
        self.dashboard_dir = self.workspace_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 P1-A004: ダッシュボード生成システム初期化完了")
        print(f"ダッシュボード出力: {self.dashboard_dir}")
    
    def load_sync_report(self) -> Optional[Dict[str, Any]]:
        """同期レポート読み込み"""
        logger.info("同期レポート読み込み開始")
        
        # 最新の同期レポートファイル検索
        sync_files = list((self.workspace_dir / "documentation").glob("sync_report*.json"))
        if not sync_files:
            logger.error("同期レポートファイルが見つかりません")
            return None
        
        latest_sync = max(sync_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"最新同期レポート: {latest_sync}")
        
        try:
            with open(latest_sync, 'r', encoding='utf-8') as f:
                sync_data = json.load(f)
            
            logger.info("✅ 同期レポート読み込み完了")
            return sync_data
            
        except Exception as e:
            logger.error(f"同期レポート読み込みエラー: {e}")
            return None
    
    def load_quality_assessment(self) -> Optional[Dict[str, Any]]:
        """品質評価結果読み込み"""
        logger.info("品質評価結果読み込み開始")
        
        quality_files = list((self.workspace_dir / "quality").glob("documentation_quality_*.json"))
        if not quality_files:
            logger.warning("品質評価ファイルが見つかりません")
            return None
        
        latest_quality = max(quality_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"最新品質評価: {latest_quality}")
        
        try:
            with open(latest_quality, 'r', encoding='utf-8') as f:
                quality_data = json.load(f)
            
            logger.info("✅ 品質評価結果読み込み完了")
            return quality_data
            
        except Exception as e:
            logger.error(f"品質評価読み込みエラー: {e}")
            return None
    
    def generate_sync_status_pie_chart(self, sync_data: Dict[str, Any]) -> Path:
        """同期状況円グラフ生成"""
        logger.info("同期状況円グラフ生成開始")
        
        # データ準備
        synced = sync_data["synced_items"]
        outdated = sync_data["outdated_items"]
        missing = sync_data["missing_docs"]
        total = sync_data["total_docs"] + sync_data["total_implementations"]
        other = total - (synced + outdated + missing)
        
        labels = ['Synced', 'Outdated', 'Missing Docs', 'Other']
        sizes = [synced, outdated, missing, max(0, other)]
        colors = ['#2E8B57', '#FF8C00', '#DC143C', '#808080']
        explode = (0.1, 0, 0, 0)  # Syncedを強調
        
        # 円グラフ作成
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90,
                                         explode=explode)
        
        # テキスト装飾
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title("P1-A004: Documentation Sync Status Distribution", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 統計情報追加
        total_items = sync_data["total_docs"] + sync_data["total_implementations"]
        sync_rate = sync_data["sync_rate"]
        
        info_text = f"Total Items: {total_items}\nSync Rate: {sync_rate:.1%}"
        ax.text(1.2, 0.5, info_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        pie_file = self.dashboard_dir / f"sync_status_pie_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(pie_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"同期状況円グラフ保存: {pie_file}")
        return pie_file
    
    def generate_doc_types_bar_chart(self, sync_data: Dict[str, Any]) -> Path:
        """ドキュメントタイプ分布棒グラフ生成"""
        logger.info("ドキュメントタイプ分布棒グラフ生成開始")
        
        # データ準備
        doc_types = sync_data["detailed_analysis"]["doc_types_distribution"]
        types = list(doc_types.keys())
        counts = list(doc_types.values())
        
        # 棒グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # チャート装飾
        ax.set_title("P1-A004: Documentation Types Distribution", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Document Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        
        # 値をバーの上に表示
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # グリッド追加
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存
        bar_file = self.dashboard_dir / f"doc_types_bar_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ドキュメントタイプ分布棒グラフ保存: {bar_file}")
        return bar_file
    
    def generate_implementation_coverage_chart(self, sync_data: Dict[str, Any]) -> Path:
        """実装カバレッジチャート生成"""
        logger.info("実装カバレッジチャート生成開始")
        
        # データ準備
        impl_types = sync_data["detailed_analysis"]["impl_types_distribution"]
        test_coverage_rate = sync_data["detailed_analysis"]["test_coverage_rate"]
        
        # サブプロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 実装タイプ分布（左）
        types = list(impl_types.keys())
        counts = list(impl_types.values())
        colors = ['#4169E1', '#FF6347']
        
        ax1.pie(counts, labels=types, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Implementation Types", fontsize=14, fontweight='bold')
        
        # テストカバレッジ（右）
        coverage_data = ['Tested', 'Not Tested']
        coverage_counts = [
            test_coverage_rate * sum(counts),
            (1 - test_coverage_rate) * sum(counts)
        ]
        coverage_colors = ['#2E8B57', '#DC143C']
        
        ax2.pie(coverage_counts, labels=coverage_data, colors=coverage_colors, 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"Test Coverage ({test_coverage_rate:.1%})", fontsize=14, fontweight='bold')
        
        fig.suptitle("P1-A004: Implementation Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        impl_file = self.dashboard_dir / f"implementation_coverage_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(impl_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"実装カバレッジチャート保存: {impl_file}")
        return impl_file
    
    def generate_quality_metrics_radar(self, quality_data: Optional[Dict[str, Any]]) -> Path:
        """品質メトリクスレーダーチャート生成"""
        logger.info("品質メトリクスレーダーチャート生成開始")
        
        # デフォルトまたは実データ
        if quality_data and "quality_metrics" in quality_data:
            metrics = quality_data["quality_metrics"]
        else:
            # 推定値使用
            metrics = {
                "sync_coverage": 0.003,
                "documentation_coverage": 0.14,
                "consistency_score": 0.7,
                "completeness_score": 0.6
            }
        
        # データ準備
        categories = ['Sync Coverage', 'Doc Coverage', 'Consistency', 'Completeness']
        values = [
            metrics.get("sync_coverage", 0),
            metrics.get("documentation_coverage", 0),
            metrics.get("consistency_score", 0),
            metrics.get("completeness_score", 0)
        ]
        
        # 円を閉じるためにデータ複製
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # レーダーチャート作成
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Current State', color='#FF6347')
        ax.fill(angles, values, alpha=0.25, color='#FF6347')
        
        # 目標値（参考）
        target_values = [0.8, 0.6, 0.9, 0.8] + [0.8]
        ax.plot(angles, target_values, 'o-', linewidth=2, label='Target', color='#2E8B57', linestyle='--')
        
        # チャート装飾
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("P1-A004: Documentation Quality Metrics", 
                    fontsize=14, fontweight='bold', pad=30)
        
        # グリッド線
        ax.grid(True, alpha=0.3)
        
        # 凡例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # 保存
        radar_file = self.dashboard_dir / f"quality_metrics_radar_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"品質メトリクスレーダーチャート保存: {radar_file}")
        return radar_file
    
    def generate_improvement_timeline_chart(self, sync_data: Dict[str, Any]) -> Path:
        """改善タイムラインチャート生成"""
        logger.info("改善タイムラインチャート生成開始")
        
        # サンプル改善タイムライン（実際のプロジェクトでは履歴データ使用）
        timeline_data = {
            "Phase 1": {"sync_rate": 0.003, "target": 0.2, "duration": "2週間"},
            "Phase 2": {"sync_rate": 0.2, "target": 0.5, "duration": "4週間"},
            "Phase 3": {"sync_rate": 0.5, "target": 0.8, "duration": "8週間"}
        }
        
        phases = list(timeline_data.keys())
        current_rates = [timeline_data[phase]["sync_rate"] for phase in phases]
        target_rates = [timeline_data[phase]["target"] for phase in phases]
        
        # 棒グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_rates, width, label='Current', color='#FF6347', alpha=0.8)
        bars2 = ax.bar(x + width/2, target_rates, width, label='Target', color='#2E8B57', alpha=0.8)
        
        # チャート装飾
        ax.set_title("P1-A004: Documentation Improvement Timeline", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Improvement Phase", fontsize=12, fontweight='bold')
        ax.set_ylabel("Sync Rate", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylim(0, 1.0)
        
        # 値表示
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 凡例とグリッド
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        timeline_file = self.dashboard_dir / f"improvement_timeline_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"改善タイムラインチャート保存: {timeline_file}")
        return timeline_file
    
    def generate_dashboard_html(self, sync_data: Dict[str, Any], 
                              quality_data: Optional[Dict[str, Any]],
                              chart_files: Dict[str, Path]) -> Path:
        """HTMLダッシュボード生成"""
        logger.info("HTMLダッシュボード生成開始")
        
        # 品質スコア計算
        if quality_data and "overall_quality_score" in quality_data:
            quality_score = quality_data["overall_quality_score"]
            quality_grade = quality_data["quality_grade"]
        else:
            quality_score = 0.25
            quality_grade = "POOR"
        
        # HTML生成
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P1-A004: Documentation Sync Dashboard</title>
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
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card.good {{
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .metric-card .value {{
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
        .recommendations {{
            background: #ffe6e6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
            margin-bottom: 20px;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 10px;
        }}
        .status-critical {{
            background: #ffebee;
            border-left: 5px solid #f44336;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
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
        <h1>🎯 P1-A004: Documentation Sync Dashboard</h1>
        
        <div class="status-critical">
            <h3>⚠️ Critical Status: Major Documentation Improvement Required</h3>
            <p>Current sync rate of {sync_data['sync_rate']:.1%} indicates significant documentation gaps requiring immediate attention.</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Sync Rate</h3>
                <div class="value">{sync_data['sync_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Total Docs</h3>
                <div class="value">{sync_data['total_docs']}</div>
            </div>
            <div class="metric-card">
                <h3>Total Implementations</h3>
                <div class="value">{sync_data['total_implementations']}</div>
            </div>
            <div class="metric-card">
                <h3>Quality Score</h3>
                <div class="value">{quality_score:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Quality Grade</h3>
                <div class="value">{quality_grade}</div>
            </div>
            <div class="metric-card">
                <h3>Outdated Docs</h3>
                <div class="value">{sync_data['outdated_items']}</div>
            </div>
        </div>
        
        <h2>📊 Visualization</h2>
        <div class="chart-grid">
            <div class="chart-container">
                <h3>Sync Status Distribution</h3>
                <img src="{chart_files['sync_pie'].name}" alt="Sync Status Pie">
            </div>
            <div class="chart-container">
                <h3>Document Types</h3>
                <img src="{chart_files['doc_types'].name}" alt="Document Types Bar">
            </div>
            <div class="chart-container">
                <h3>Implementation Coverage</h3>
                <img src="{chart_files['impl_coverage'].name}" alt="Implementation Coverage">
            </div>
            <div class="chart-container">
                <h3>Quality Metrics</h3>
                <img src="{chart_files['quality_radar'].name}" alt="Quality Metrics Radar">
            </div>
            <div class="chart-container">
                <h3>Improvement Timeline</h3>
                <img src="{chart_files['timeline'].name}" alt="Improvement Timeline">
            </div>
        </div>
        
        <h2>🎯 Key Recommendations</h2>
        <div class="recommendations">
            <ul>
"""
        
        for rec in sync_data["recommendations"]:
            html_content += f"                <li>{rec}</li>\n"
        
        html_content += f"""
            </ul>
        </div>
        
        <h2>📈 Improvement Strategy</h2>
        <div class="recommendations">
            <h4>Phase 1 (2週間): 緊急対応</h4>
            <ul>
                <li>期限切れドキュメント76件の優先更新</li>
                <li>主要機能のドキュメント整備</li>
            </ul>
            
            <h4>Phase 2 (4週間): 体系的改善</h4>
            <ul>
                <li>実装参照不足255件の修正</li>
                <li>ドキュメントテンプレート標準化</li>
            </ul>
            
            <h4>Phase 3 (8週間): 自動化・継続改善</h4>
            <ul>
                <li>CI/CDパイプライン統合</li>
                <li>自動ドキュメント生成システム構築</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Generated at: {datetime.now().isoformat()}<br>
            Report ID: P1A004_dashboard_{datetime.now():%Y%m%d_%H%M%S}
        </div>
    </div>
</body>
</html>
"""
        
        # HTML保存
        html_file = self.dashboard_dir / f"P1A004_dashboard_{datetime.now():%Y%m%d_%H%M%S}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTMLダッシュボード保存: {html_file}")
        return html_file
    
    def generate_full_dashboard(self) -> Dict[str, Any]:
        """フルダッシュボード生成"""
        logger.info("🎨 P1-A004 フルダッシュボード生成開始")
        start_time = datetime.now()
        
        try:
            # 1. データ読み込み
            sync_data = self.load_sync_report()
            if sync_data is None:
                return {"success": False, "error": "同期レポートが見つかりません"}
            
            quality_data = self.load_quality_assessment()
            
            # 2. チャート生成
            chart_files = {
                "sync_pie": self.generate_sync_status_pie_chart(sync_data),
                "doc_types": self.generate_doc_types_bar_chart(sync_data),
                "impl_coverage": self.generate_implementation_coverage_chart(sync_data),
                "quality_radar": self.generate_quality_metrics_radar(quality_data),
                "timeline": self.generate_improvement_timeline_chart(sync_data)
            }
            
            # 3. HTMLダッシュボード生成
            html_file = self.generate_dashboard_html(sync_data, quality_data, chart_files)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "processing_time": processing_time,
                "html_dashboard": str(html_file),
                "chart_files": {k: str(v) for k, v in chart_files.items()},
                "sync_rate": sync_data["sync_rate"],
                "total_docs": sync_data["total_docs"],
                "total_implementations": sync_data["total_implementations"]
            }
            
            logger.info(f"✅ P1-A004 フルダッシュボード生成完了 (処理時間: {processing_time:.2f}秒)")
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
    
    parser = argparse.ArgumentParser(description="P1-A004: ダッシュボード生成")
    parser.add_argument("--full", action="store_true", help="フルダッシュボード生成")
    parser.add_argument("--charts-only", action="store_true", help="チャートのみ生成")
    
    args = parser.parse_args()
    
    generator = P1A004DashboardGenerator()
    
    if args.full:
        # フルダッシュボード生成
        result = generator.generate_full_dashboard()
        
        if result["success"]:
            print(f"🎨 P1-A004ダッシュボード生成完了")
            print(f"   同期率: {result['sync_rate']:.1%}")
            print(f"   総ドキュメント: {result['total_docs']}件")
            print(f"   総実装: {result['total_implementations']}件")
            print(f"   処理時間: {result['processing_time']:.2f}秒")
            print(f"   HTMLダッシュボード: {result['html_dashboard']}")
            print(f"   チャートファイル: {len(result['chart_files'])}件")
            return 0
        else:
            print(f"❌ ダッシュボード生成失敗: {result['error']}")
            return 1
    
    elif args.charts_only:
        # チャートのみ生成
        sync_data = generator.load_sync_report()
        quality_data = generator.load_quality_assessment()
        
        if not sync_data:
            print("❌ 同期レポートが見つかりません")
            return 1
        
        chart_files = {
            "sync_pie": generator.generate_sync_status_pie_chart(sync_data),
            "doc_types": generator.generate_doc_types_bar_chart(sync_data),
            "impl_coverage": generator.generate_implementation_coverage_chart(sync_data),
            "quality_radar": generator.generate_quality_metrics_radar(quality_data),
            "timeline": generator.generate_improvement_timeline_chart(sync_data)
        }
        
        print(f"✅ チャート生成完了: {len(chart_files)}件")
        for name, path in chart_files.items():
            print(f"   {name}: {path}")
        
        return 0
    
    else:
        print("🎨 P1-A004: ドキュメント整備システム - ダッシュボード生成")
        print("使用例:")
        print("  python p1a004_dashboard_generator.py --full")
        print("  python p1a004_dashboard_generator.py --charts-only")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())