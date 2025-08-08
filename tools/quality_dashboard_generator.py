#!/usr/bin/env python3
"""
品質分析ダッシュボード生成ツール
quality_statistics.jsonlから品質トレンドを可視化

QI-001: 品質指標「良いとこ取り」戦略 Phase 1実装
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# 日本語フォント設定
import matplotlib
# インストールした日本語フォントを優先順位順に設定
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'TakaoGothic', 'IPAGothic', 'IPAPGothic', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'TakaoGothic', 'IPAGothic', 'IPAPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策


class QualityDashboardGenerator:
    """品質分析ダッシュボード生成クラス"""
    
    def __init__(self, stats_file: Path, output_dir: Path):
        """
        Args:
            stats_file: quality_statistics.jsonlファイルパス
            output_dir: ダッシュボード出力ディレクトリ
        """
        self.stats_file = stats_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
        
    def load_statistics(self) -> bool:
        """統計データ読み込み"""
        if not self.stats_file.exists():
            print(f"⚠️ 統計ファイルが存在しません: {self.stats_file}")
            return False
            
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            
            print(f"✅ {len(self.data)}件の品質データを読み込みました")
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return False
    
    def analyze_quality_distribution(self) -> Dict:
        """品質スコア分布分析"""
        if not self.data:
            return {}
        
        scores = [d['quality_score'] for d in self.data]
        
        # 品質レベル分類（現在の閾値に基づく）
        critical = sum(1 for s in scores if s < 0.05)  # 5%未満
        monitoring = sum(1 for s in scores if 0.05 <= s < 0.10)  # 5-10%
        improvement = sum(1 for s in scores if 0.10 <= s < 0.15)  # 10-15%
        acceptable = sum(1 for s in scores if 0.15 <= s < 0.30)  # 15-30%
        good = sum(1 for s in scores if 0.30 <= s < 0.50)  # 30-50%
        excellent = sum(1 for s in scores if s >= 0.50)  # 50%以上
        
        distribution = {
            'total': len(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores),
            'levels': {
                'critical': critical,
                'monitoring': monitoring,
                'improvement': improvement,
                'acceptable': acceptable,
                'good': good,
                'excellent': excellent
            },
            'percentages': {
                'critical': critical / len(scores) * 100,
                'monitoring': monitoring / len(scores) * 100,
                'improvement': improvement / len(scores) * 100,
                'acceptable': acceptable / len(scores) * 100,
                'good': good / len(scores) * 100,
                'excellent': excellent / len(scores) * 100
            }
        }
        
        return distribution
    
    def analyze_metrics_correlation(self) -> Dict:
        """メトリクス相関分析"""
        if not self.data:
            return {}
        
        # 各メトリクス抽出
        compactness = []
        fill_ratio = []
        coverage_ratio = []
        quality_scores = []
        
        for d in self.data:
            if 'metrics' in d:
                metrics = d['metrics']
                compactness.append(metrics.get('compactness', 0))
                fill_ratio.append(metrics.get('fill_ratio', 0))
                coverage_ratio.append(metrics.get('coverage_ratio', 0))
                quality_scores.append(d['quality_score'])
        
        if not compactness:
            return {}
        
        # 相関係数計算
        from scipy.stats import pearsonr
        
        correlations = {}
        if len(compactness) > 1:
            correlations['compactness_quality'] = pearsonr(compactness, quality_scores)[0]
            correlations['fill_ratio_quality'] = pearsonr(fill_ratio, quality_scores)[0]
            correlations['coverage_ratio_quality'] = pearsonr(coverage_ratio, quality_scores)[0]
        
        return {
            'metrics_stats': {
                'compactness': {
                    'mean': np.mean(compactness),
                    'std': np.std(compactness),
                    'min': min(compactness),
                    'max': max(compactness)
                },
                'fill_ratio': {
                    'mean': np.mean(fill_ratio),
                    'std': np.std(fill_ratio),
                    'min': min(fill_ratio),
                    'max': max(fill_ratio)
                },
                'coverage_ratio': {
                    'mean': np.mean(coverage_ratio),
                    'std': np.std(coverage_ratio),
                    'min': min(coverage_ratio),
                    'max': max(coverage_ratio)
                }
            },
            'correlations': correlations
        }
    
    def generate_distribution_chart(self, distribution: Dict):
        """品質分布チャート生成"""
        if not distribution:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('品質分析ダッシュボード - QI-001 品質指標「良いとこ取り」戦略', fontsize=16)
        
        # 1. ヒストグラム
        ax1 = axes[0, 0]
        scores = [d['quality_score'] for d in self.data]
        ax1.hist(scores, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(distribution['mean'], color='red', linestyle='--', label=f'平均: {distribution["mean"]:.3f}')
        ax1.axvline(distribution['median'], color='green', linestyle='--', label=f'中央値: {distribution["median"]:.3f}')
        ax1.set_xlabel('品質スコア')
        ax1.set_ylabel('頻度')
        ax1.set_title('品質スコア分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 品質レベル円グラフ
        ax2 = axes[0, 1]
        levels = distribution['levels']
        labels = ['Critical (<5%)', 'Monitoring (5-10%)', 'Improvement (10-15%)', 
                 'Acceptable (15-30%)', 'Good (30-50%)', 'Excellent (≥50%)']
        sizes = [levels['critical'], levels['monitoring'], levels['improvement'],
                levels['acceptable'], levels['good'], levels['excellent']]
        colors = ['#ff4444', '#ff8844', '#ffbb44', '#88dd44', '#44dd88', '#44aaff']
        
        # 0でない要素のみ表示
        non_zero_indices = [i for i, s in enumerate(sizes) if s > 0]
        if non_zero_indices:
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            ax2.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                   autopct='%1.1f%%', startangle=90)
        ax2.set_title('品質レベル分布')
        
        # 3. 時系列トレンド
        ax3 = axes[1, 0]
        timestamps = []
        time_scores = []
        for d in self.data:
            if 'timestamp' in d:
                timestamps.append(datetime.fromisoformat(d['timestamp']))
                time_scores.append(d['quality_score'])
        
        if timestamps:
            ax3.plot(timestamps, time_scores, marker='o', markersize=3, alpha=0.5)
            ax3.set_xlabel('時間')
            ax3.set_ylabel('品質スコー')
            ax3.set_title('品質スコアの時系列変化')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. 統計サマリーテーブル
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [
            ['指標', '値'],
            ['総サンプル数', f'{distribution["total"]}'],
            ['平均スコア', f'{distribution["mean"]:.3f}'],
            ['中央値', f'{distribution["median"]:.3f}'],
            ['標準偏差', f'{distribution["std"]:.3f}'],
            ['最小値', f'{distribution["min"]:.3f}'],
            ['最大値', f'{distribution["max"]:.3f}'],
            ['', ''],
            ['Critical率 (<5%)', f'{distribution["percentages"]["critical"]:.1f}%'],
            ['Monitoring率 (5-10%)', f'{distribution["percentages"]["monitoring"]:.1f}%'],
            ['Improvement率 (10-15%)', f'{distribution["percentages"]["improvement"]:.1f}%'],
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'quality_dashboard.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ ダッシュボード保存: {output_path}")
        plt.close()
    
    def generate_metrics_analysis_chart(self, metrics_analysis: Dict):
        """メトリクス分析チャート生成"""
        if not metrics_analysis or 'metrics_stats' not in metrics_analysis:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('メトリクス詳細分析 - アニメキャラクター特性との相関', fontsize=14)
        
        metrics_stats = metrics_analysis['metrics_stats']
        metrics_names = ['compactness', 'fill_ratio', 'coverage_ratio']
        titles = ['Compactness (円形度)', 'Fill Ratio (充填率)', 'Coverage Ratio (カバレッジ)']
        
        for i, (metric_name, title) in enumerate(zip(metrics_names, titles)):
            ax = axes[i]
            
            # メトリクス値のヒストグラム
            values = []
            for d in self.data:
                if 'metrics' in d and metric_name in d['metrics']:
                    values.append(d['metrics'][metric_name])
            
            if values:
                ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
                stats = metrics_stats[metric_name]
                ax.axvline(stats['mean'], color='red', linestyle='--', 
                          label=f'平均: {stats["mean"]:.3f}')
                ax.set_xlabel(title)
                ax.set_ylabel('頻度')
                ax.set_title(f'{title}分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'metrics_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ メトリクス分析保存: {output_path}")
        plt.close()
    
    def generate_html_report(self, distribution: Dict, metrics_analysis: Dict):
        """HTML形式のレポート生成"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>品質分析ダッシュボード - QI-001</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .quality-level {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .level-critical {{ background-color: #ffebee; color: #c62828; }}
        .level-monitoring {{ background-color: #fff3e0; color: #e65100; }}
        .level-improvement {{ background-color: #fffde7; color: #f57f17; }}
        .level-acceptable {{ background-color: #f1f8e9; color: #558b2f; }}
        .level-good {{ background-color: #e8f5e9; color: #2e7d32; }}
        .level-excellent {{ background-color: #e3f2fd; color: #1565c0; }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .insights {{
            background-color: #f0f7ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
        .timestamp {{
            text-align: right;
            color: #999;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 品質分析ダッシュボード</h1>
        <p><strong>QI-001</strong>: 品質指標「良いとこ取り」戦略 - Phase 1 実装</p>
        
        <h2>📊 基本統計</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">総サンプル数</div>
                <div class="stat-value">{distribution.get('total', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均品質スコア</div>
                <div class="stat-value">{distribution.get('mean', 0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">中央値</div>
                <div class="stat-value">{distribution.get('median', 0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">標準偏差</div>
                <div class="stat-value">{distribution.get('std', 0):.3f}</div>
            </div>
        </div>
        
        <h2>🎨 品質レベル分布</h2>
        <div class="quality-levels">
"""
        
        # 品質レベル詳細
        if distribution and 'levels' in distribution:
            levels = distribution['levels']
            percentages = distribution['percentages']
            
            level_info = [
                ('critical', 'Critical (<5%)', f"{levels['critical']}件 ({percentages['critical']:.1f}%)"),
                ('monitoring', 'Monitoring (5-10%)', f"{levels['monitoring']}件 ({percentages['monitoring']:.1f}%)"),
                ('improvement', 'Improvement (10-15%)', f"{levels['improvement']}件 ({percentages['improvement']:.1f}%)"),
                ('acceptable', 'Acceptable (15-30%)', f"{levels['acceptable']}件 ({percentages['acceptable']:.1f}%)"),
                ('good', 'Good (30-50%)', f"{levels['good']}件 ({percentages['good']:.1f}%)"),
                ('excellent', 'Excellent (≥50%)', f"{levels['excellent']}件 ({percentages['excellent']:.1f}%)")
            ]
            
            for level_key, level_name, level_stat in level_info:
                html_content += f"""
            <div class="quality-level level-{level_key}">
                <strong>{level_name}</strong>: {level_stat}
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>📈 品質分布チャート</h2>
        <img src="quality_dashboard.png" alt="品質分析ダッシュボード">
        
        <h2>🔬 メトリクス詳細分析</h2>
        <img src="metrics_analysis.png" alt="メトリクス分析">
        
        <div class="insights">
            <h3>💡 主要な洞察</h3>
            <ul>
"""
        
        # インサイト生成
        if distribution:
            mean_score = distribution.get('mean', 0)
            if mean_score < 0.15:
                html_content += "<li>⚠️ 平均品質スコアが低め（<15%）です。アニメキャラクター特化指標の開発が推奨されます。</li>"
            elif mean_score < 0.30:
                html_content += "<li>📊 平均品質スコアは中程度（15-30%）です。継続的な監視と改善が有効です。</li>"
            else:
                html_content += "<li>✅ 平均品質スコアは良好（≥30%）です。安定した品質を維持しています。</li>"
            
            if 'percentages' in distribution:
                critical_pct = distribution['percentages']['critical']
                if critical_pct > 10:
                    html_content += f"<li>🚨 Critical品質（<5%）の割合が{critical_pct:.1f}%と高めです。要因分析が必要です。</li>"
        
        if metrics_analysis and 'correlations' in metrics_analysis:
            correlations = metrics_analysis['correlations']
            if 'compactness_quality' in correlations:
                corr = correlations['compactness_quality']
                if abs(corr) < 0.3:
                    html_content += f"<li>🔍 Compactness（円形度）と品質スコアの相関が低い（{corr:.3f}）です。アニメ特化指標の必要性を示唆しています。</li>"
        
        html_content += f"""
            </ul>
        </div>
        
        <div class="insights">
            <h3>🎯 推奨アクション（Phase 1）</h3>
            <ol>
                <li>品質統計データの継続的な蓄積（現在実施中）</li>
                <li>アニメキャラクター特化品質指標の研究開発</li>
                <li>低品質サンプルの詳細分析による改善点特定</li>
                <li>品質トレンド監視による早期問題検出</li>
            </ol>
        </div>
        
        <div class="timestamp">
            生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # HTML保存
        html_path = self.output_dir / 'quality_dashboard.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTMLレポート保存: {html_path}")
    
    def generate(self):
        """ダッシュボード生成メイン処理"""
        print("🚀 品質分析ダッシュボード生成開始...")
        
        # データ読み込み
        if not self.load_statistics():
            print("⚠️ 統計データが見つかりません。抽出処理を実行してください。")
            return False
        
        # 分析実行
        distribution = self.analyze_quality_distribution()
        metrics_analysis = self.analyze_metrics_correlation()
        
        # チャート生成
        self.generate_distribution_chart(distribution)
        self.generate_metrics_analysis_chart(metrics_analysis)
        
        # HTMLレポート生成
        self.generate_html_report(distribution, metrics_analysis)
        
        print(f"\n✅ ダッシュボード生成完了: {self.output_dir}")
        print("📊 生成されたファイル:")
        print(f"  - quality_dashboard.html (メインレポート)")
        print(f"  - quality_dashboard.png (品質分布チャート)")
        print(f"  - metrics_analysis.png (メトリクス分析チャート)")
        
        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='品質分析ダッシュボード生成')
    parser.add_argument('--stats-file', type=str, 
                       help='quality_statistics.jsonlファイルパス')
    parser.add_argument('--output-dir', type=str,
                       default='quality_dashboard',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # デフォルトパス設定
    if args.stats_file:
        stats_file = Path(args.stats_file)
    else:
        # 最新のtracker-workspaceから自動検索
        workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        stats_files = list(workspace_base.glob("*/quality_statistics.jsonl"))
        
        if not stats_files:
            print("⚠️ quality_statistics.jsonlが見つかりません")
            print("使用方法:")
            print("  python quality_dashboard_generator.py --stats-file <path>")
            return
        
        # 最新のファイルを使用
        stats_file = max(stats_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 使用する統計ファイル: {stats_file}")
    
    output_dir = Path(args.output_dir)
    
    # ダッシュボード生成
    generator = QualityDashboardGenerator(stats_file, output_dir)
    generator.generate()


if __name__ == "__main__":
    main()