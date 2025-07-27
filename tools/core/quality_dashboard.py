#!/usr/bin/env python3
"""
統合品質ダッシュボード
現在の10指標品質チェックシステムの可視化ダッシュボード

機能:
- 統合品質レポートの可視化
- トレンド分析
- 改善提案の優先度表示
- インタラクティブなグラフ表示
- Google Spreadsheet自動更新

技術仕様: 
- 出力パス標準: ../../spec/OUTPUT_PATH_STANDARDS.md
- Google Sheets統合: ../../spec/GOOGLE_SHEETS_INTEGRATION.md
"""

import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
import matplotlib.patches as mpatches
plt.rcParams['figure.figsize'] = (15, 10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityDashboard:
    """統合品質ダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.colors = {
            'passed': '#2ecc71',
            'failed': '#e74c3c',
            'insufficient_data': '#f39c12',
            'not_implemented': '#95a5a6',
            'error': '#c0392b',
            'no_data': '#ecf0f1',
            'baseline_created': '#3498db'
        }
        
        self.category_colors = {
            'evaluation': '#3498db',
            'mask': '#9b59b6',
            'objective': '#e67e22'
        }
    
    def create_dashboard(self, report_path: str, output_dir: str = None) -> str:
        """
        ダッシュボード作成
        
        Args:
            report_path: 統合品質レポートJSONファイルパス
            output_dir: 出力ディレクトリ（省略時は自動生成）
        
        Returns:
            str: 作成されたダッシュボードHTMLファイルパス
        """
        try:
            # レポート読み込み
            report_data = self._load_report(report_path)
            
            # 出力ディレクトリ設定
            if output_dir is None:
                output_dir = Path(report_path).parent / "dashboard"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # グラフ作成
            graphs = self._create_all_graphs(report_data, output_dir)
            
            # HTMLダッシュボード作成
            html_path = self._create_html_dashboard(report_data, graphs, output_dir)
            
            logger.info(f"ダッシュボード作成完了: {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
            raise
    
    def _load_report(self, report_path: str) -> Dict[str, Any]:
        """レポート読み込み"""
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_all_graphs(self, report_data: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """全グラフ作成"""
        graphs = {}
        
        # 1. 総合スコアレーダーチャート
        graphs['radar'] = self._create_radar_chart(report_data, output_dir)
        
        # 2. カテゴリ別合格率
        graphs['category_bar'] = self._create_category_bar_chart(report_data, output_dir)
        
        # 3. 指標詳細比較
        graphs['metrics_comparison'] = self._create_metrics_comparison(report_data, output_dir)
        
        # 4. 改善提案優先度
        graphs['improvement_priority'] = self._create_improvement_priority(report_data, output_dir)
        
        # 5. ステータス分布
        graphs['status_distribution'] = self._create_status_distribution(report_data, output_dir)
        
        return graphs
    
    def _create_radar_chart(self, report_data: Dict[str, Any], output_dir: Path) -> str:
        """レーダーチャート作成"""
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 全指標を取得
            all_metrics = []
            all_metrics.extend(report_data.get('evaluation_metrics', []))
            all_metrics.extend(report_data.get('mask_metrics', []))
            all_metrics.extend(report_data.get('objective_metrics', []))
            
            # 実装済み指標のみ
            implemented_metrics = [m for m in all_metrics 
                                 if m.get('status') not in ['not_implemented', 'error', 'no_data']]
            
            if not implemented_metrics:
                # データなしの場合のプレースホルダー
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=20)
                output_path = output_dir / 'radar_chart.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(output_path.name)
            
            # 角度設定
            metrics_names = [m['name'] for m in implemented_metrics]
            metrics_values = []
            metrics_thresholds = []
            
            for m in implemented_metrics:
                value = m.get('value', 0.0)
                threshold = m.get('threshold', 1.0)
                
                # 閾値で正規化（0-1範囲）
                if threshold and threshold > 0:
                    normalized_value = min(value / threshold, 1.0)
                else:
                    normalized_value = value
                
                metrics_values.append(normalized_value)
                metrics_thresholds.append(1.0)  # 閾値は常に1.0（正規化後）
            
            # 短縮ラベル作成
            short_names = []
            for name in metrics_names:
                if len(name) > 20:
                    short_names.append(name[:17] + '...')
                else:
                    short_names.append(name)
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
            
            # データを閉じる
            metrics_values += metrics_values[:1]
            metrics_thresholds += metrics_thresholds[:1]
            angles += angles[:1]
            
            # プロット
            ax.plot(angles, metrics_values, 'o-', linewidth=2, label='現在値', color='#3498db')
            ax.fill(angles, metrics_values, alpha=0.25, color='#3498db')
            ax.plot(angles, metrics_thresholds, '--', linewidth=1, label='目標値', color='#e74c3c')
            
            # カスタマイズ
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(short_names, fontsize=10)
            ax.set_ylim(0, 1.2)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.set_title('Quality Metrics Radar Chart', fontsize=16, pad=20)
            
            output_path = output_dir / 'radar_chart.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path.name)
            
        except Exception as e:
            logger.error(f"レーダーチャート作成エラー: {e}")
            return ""
    
    def _create_category_bar_chart(self, report_data: Dict[str, Any], output_dir: Path) -> str:
        """カテゴリ別合格率バーチャート作成"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            categories = ['評価指標', 'マスク品質', '客観指標']
            metric_groups = [
                report_data.get('evaluation_metrics', []),
                report_data.get('mask_metrics', []),
                report_data.get('objective_metrics', [])
            ]
            
            pass_rates = []
            total_counts = []
            
            for metrics in metric_groups:
                implemented = [m for m in metrics 
                             if m.get('status') not in ['not_implemented', 'error', 'no_data']]
                passed = [m for m in implemented if m.get('status') == 'passed']
                
                if implemented:
                    pass_rate = len(passed) / len(implemented)
                    pass_rates.append(pass_rate)
                    total_counts.append(len(implemented))
                else:
                    pass_rates.append(0.0)
                    total_counts.append(0)
            
            # バーチャート
            bars = ax.bar(categories, pass_rates, 
                         color=[self.category_colors['evaluation'], 
                               self.category_colors['mask'], 
                               self.category_colors['objective']])
            
            # 数値表示
            for i, (bar, rate, count) in enumerate(zip(bars, pass_rates, total_counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.1%}\n({count}指標)', 
                       ha='center', va='bottom', fontsize=12)
            
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('合格率', fontsize=12)
            ax.set_title('カテゴリ別品質指標合格率', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            # 目標ライン
            ax.axhline(y=0.8, color='#e74c3c', linestyle='--', alpha=0.7, label='目標合格率 (80%)')
            ax.legend()
            
            output_path = output_dir / 'category_bar_chart.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path.name)
            
        except Exception as e:
            logger.error(f"カテゴリ別バーチャート作成エラー: {e}")
            return ""
    
    def _create_metrics_comparison(self, report_data: Dict[str, Any], output_dir: Path) -> str:
        """指標詳細比較チャート作成"""
        try:
            # 全指標を取得
            all_metrics = []
            all_metrics.extend(report_data.get('evaluation_metrics', []))
            all_metrics.extend(report_data.get('mask_metrics', []))
            all_metrics.extend(report_data.get('objective_metrics', []))
            
            # 実装済み指標のみ
            implemented_metrics = [m for m in all_metrics 
                                 if m.get('status') not in ['not_implemented', 'error', 'no_data']]
            
            if not implemented_metrics:
                return ""
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            y_pos = np.arange(len(implemented_metrics))
            values = [m.get('value', 0.0) for m in implemented_metrics]
            thresholds = [m.get('threshold', 1.0) for m in implemented_metrics]
            names = [m['name'] for m in implemented_metrics]
            statuses = [m.get('status', 'unknown') for m in implemented_metrics]
            
            # 短縮名作成
            short_names = []
            for name in names:
                if len(name) > 30:
                    short_names.append(name[:27] + '...')
                else:
                    short_names.append(name)
            
            # 色設定
            colors = [self.colors.get(status, '#95a5a6') for status in statuses]
            
            # 横棒グラフ
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
            
            # 閾値ライン
            for i, threshold in enumerate(thresholds):
                if threshold and threshold > 0:
                    ax.axvline(x=threshold, color='#e74c3c', linestyle='--', alpha=0.7)
            
            # 数値表示
            for i, (bar, value, threshold) in enumerate(zip(bars, values, thresholds)):
                width = bar.get_width()
                ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}',
                       ha='left', va='center', fontsize=10)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_names, fontsize=10)
            ax.set_xlabel('指標値', fontsize=12)
            ax.set_title('品質指標詳細比較', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            
            # 凡例
            legend_elements = []
            for status, color in self.colors.items():
                if status in statuses:
                    label = {'passed': '合格', 'failed': '不合格', 'insufficient_data': 'データ不足'}.get(status, status)
                    legend_elements.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='lower right')
            
            # 閾値線の説明
            ax.text(0.98, 0.02, '破線: 合格閾値', transform=ax.transAxes, 
                   ha='right', va='bottom', fontsize=10, alpha=0.7)
            
            plt.tight_layout()
            
            output_path = output_dir / 'metrics_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path.name)
            
        except Exception as e:
            logger.error(f"指標比較チャート作成エラー: {e}")
            return ""
    
    def _create_improvement_priority(self, report_data: Dict[str, Any], output_dir: Path) -> str:
        """改善提案優先度チャート作成"""
        try:
            improvements = report_data.get('priority_improvements', [])
            
            if not improvements:
                return ""
            
            # 頻度カウント
            improvement_counts = {}
            for improvement in improvements:
                improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1
            
            # 上位5項目
            sorted_improvements = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if not sorted_improvements:
                return ""
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            items = [item[0] for item in sorted_improvements]
            counts = [item[1] for item in sorted_improvements]
            
            # 短縮ラベル
            short_items = []
            for item in items:
                if len(item) > 20:
                    short_items.append(item[:17] + '...')
                else:
                    short_items.append(item)
            
            # バーチャート
            bars = ax.bar(range(len(short_items)), counts, color='#e74c3c', alpha=0.8)
            
            # 数値表示
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{count}',
                       ha='center', va='bottom', fontsize=12)
            
            ax.set_xticks(range(len(short_items)))
            ax.set_xticklabels(short_items, rotation=45, ha='right')
            ax.set_ylabel('言及回数', fontsize=12)
            ax.set_title('改善提案優先度ランキング', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            output_path = output_dir / 'improvement_priority.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path.name)
            
        except Exception as e:
            logger.error(f"改善優先度チャート作成エラー: {e}")
            return ""
    
    def _create_status_distribution(self, report_data: Dict[str, Any], output_dir: Path) -> str:
        """ステータス分布円グラフ作成"""
        try:
            # 全指標のステータス収集
            all_metrics = []
            all_metrics.extend(report_data.get('evaluation_metrics', []))
            all_metrics.extend(report_data.get('mask_metrics', []))
            all_metrics.extend(report_data.get('objective_metrics', []))
            
            status_counts = {}
            for metric in all_metrics:
                status = metric.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if not status_counts:
                return ""
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # ラベル日本語化
            status_labels = {
                'passed': '合格',
                'failed': '不合格',
                'insufficient_data': 'データ不足',
                'not_implemented': '未実装',
                'error': 'エラー',
                'no_data': 'データなし',
                'baseline_created': 'ベースライン作成済み'
            }
            
            labels = [status_labels.get(status, status) for status in status_counts.keys()]
            sizes = list(status_counts.values())
            colors = [self.colors.get(status, '#95a5a6') for status in status_counts.keys()]
            
            # 円グラフ
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            
            # テキストスタイル
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('品質指標ステータス分布', fontsize=14)
            
            output_path = output_dir / 'status_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path.name)
            
        except Exception as e:
            logger.error(f"ステータス分布チャート作成エラー: {e}")
            return ""
    
    def _create_html_dashboard(self, report_data: Dict[str, Any], graphs: Dict[str, str], 
                              output_dir: Path) -> str:
        """HTMLダッシュボード作成"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>品質ダッシュボード - {report_data.get('dataset_name', 'Unknown')}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .status-pass {{ color: #2ecc71; }}
        .status-partial {{ color: #f39c12; }}
        .status-fail {{ color: #e74c3c; }}
        .graphs {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .graph-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .graph-card h3 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .graph-card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .details {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .details h3 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .metric-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .metric-item {{
            padding: 15px;
            border-left: 4px solid #ddd;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-item.passed {{ border-left-color: #2ecc71; }}
        .metric-item.failed {{ border-left-color: #e74c3c; }}
        .metric-item.insufficient_data {{ border-left-color: #f39c12; }}
        .metric-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.2em;
            margin-bottom: 5px;
        }}
        .metric-notes {{
            font-size: 0.9em;
            color: #666;
        }}
        .improvements {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .improvements h3 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .improvement-list {{
            list-style: none;
            padding: 0;
        }}
        .improvement-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .graphs {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 統合品質ダッシュボード</h1>
        <p>Dataset: {report_data.get('dataset_name', 'Unknown')} | 
           Generated: {report_data.get('timestamp', 'Unknown')}</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>総合スコア</h3>
            <div class="value status-{report_data.get('status', 'fail').lower()}">{report_data.get('overall_score', 0)*100:.1f}%</div>
        </div>
        <div class="summary-card">
            <h3>合格指標</h3>
            <div class="value">{report_data.get('passed_metrics', 0)}/{report_data.get('total_metrics', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>総画像数</h3>
            <div class="value">{report_data.get('total_images', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>総合判定</h3>
            <div class="value status-{report_data.get('status', 'fail').lower()}">{report_data.get('status', 'UNKNOWN')}</div>
        </div>
    </div>
    
    <div class="graphs">
        {"" if not graphs.get('radar') else f'<div class="graph-card"><h3>📊 総合指標レーダーチャート</h3><img src="{graphs["radar"]}" alt="レーダーチャート"></div>'}
        {"" if not graphs.get('category_bar') else f'<div class="graph-card"><h3>📈 カテゴリ別合格率</h3><img src="{graphs["category_bar"]}" alt="カテゴリ別合格率"></div>'}
        {"" if not graphs.get('metrics_comparison') else f'<div class="graph-card full-width"><h3>📋 指標詳細比較</h3><img src="{graphs["metrics_comparison"]}" alt="指標詳細比較"></div>'}
        {"" if not graphs.get('status_distribution') else f'<div class="graph-card"><h3>🎯 ステータス分布</h3><img src="{graphs["status_distribution"]}" alt="ステータス分布"></div>'}
        {"" if not graphs.get('improvement_priority') else f'<div class="graph-card"><h3>🚀 改善優先度</h3><img src="{graphs["improvement_priority"]}" alt="改善優先度"></div>'}
    </div>
    
    <div class="details">
        <h3>📋 詳細指標一覧</h3>
        <div class="metric-list">
            {self._generate_metrics_html(report_data)}
        </div>
    </div>
    
    <div class="improvements">
        <h3>🚀 改善提案</h3>
        <h4>優先改善項目:</h4>
        <ul class="improvement-list">
            {self._generate_improvements_html(report_data.get('priority_improvements', []))}
        </ul>
        <h4>技術的推奨事項:</h4>
        <ul class="improvement-list">
            {self._generate_improvements_html(report_data.get('technical_recommendations', []))}
        </ul>
    </div>
    
    <div class="footer">
        <p>Generated by 統合品質チェックシステム | © 2025</p>
    </div>
</body>
</html>
"""
            
            html_path = output_dir / 'dashboard.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_path)
            
        except Exception as e:
            logger.error(f"HTMLダッシュボード作成エラー: {e}")
            raise
    
    def _generate_metrics_html(self, report_data: Dict[str, Any]) -> str:
        """指標一覧HTML生成"""
        html_parts = []
        
        categories = [
            ('評価指標', report_data.get('evaluation_metrics', [])),
            ('マスク品質', report_data.get('mask_metrics', [])),
            ('客観指標', report_data.get('objective_metrics', []))
        ]
        
        for category_name, metrics in categories:
            for metric in metrics:
                status = metric.get('status', 'unknown')
                name = metric.get('name', 'Unknown')
                value = metric.get('value', 0.0)
                threshold = metric.get('threshold')
                notes = metric.get('notes', '')
                
                threshold_text = f" (閾値: {threshold:.3f})" if threshold else ""
                
                html_parts.append(f"""
            <div class="metric-item {status}">
                <div class="metric-name">{name}</div>
                <div class="metric-value">{value:.3f}{threshold_text}</div>
                <div class="metric-notes">{notes}</div>
            </div>
                """)
        
        return ''.join(html_parts)
    
    def _generate_improvements_html(self, improvements: List[str]) -> str:
        """改善提案HTML生成"""
        if not improvements:
            return "<li>改善提案はありません</li>"
        
        html_parts = []
        for improvement in improvements[:10]:  # 上位10件
            html_parts.append(f"<li>{improvement}</li>")
        
        return ''.join(html_parts)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="統合品質ダッシュボード作成")
    parser.add_argument("--report", "-r", required=True, help="統合品質レポートJSONファイルパス")
    parser.add_argument("--output", "-o", help="出力ディレクトリ（省略時は自動生成）")
    
    args = parser.parse_args()
    
    try:
        dashboard = QualityDashboard()
        html_path = dashboard.create_dashboard(args.report, args.output)
        
        print(f"\n🎉 品質ダッシュボード作成完了!")
        print(f"📄 HTML: {html_path}")
        print(f"🌐 ブラウザで開く: file://{Path(html_path).absolute()}")
        
        # Google Spreadsheet自動更新
        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from google_sheets_updater import update_from_quality_report
            update_from_quality_report(args.report)
            print("📊 Google Spreadsheet更新完了")
        except ImportError:
            logger.warning("Google Sheets更新スキップ: google_sheets_updaterが見つかりません")
        except Exception as sheet_error:
            logger.warning(f"Google Sheets更新エラー: {sheet_error}")
        
    except Exception as e:
        logger.error(f"ダッシュボード作成失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()