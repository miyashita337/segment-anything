"""
統計検定結果のレポート生成・可視化システム
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import seaborn as sns

from .statistical_validator import TTestResult, StatisticalValidator


class StatisticalReporter:
    """
    統計検定結果のレポート生成・可視化クラス
    
    主要機能:
    - 検定結果のJSON/HTML出力
    - 可視化（箱ひげ図、信頼区間プロット）
    - 複数比較結果のサマリー生成
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            output_dir: 出力ディレクトリ（デフォルト: カレント）
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Matplotlibの設定
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
        sns.set_style("whitegrid")
    
    def generate_json_report(
        self, 
        result: TTestResult,
        test_name: str = "t-test",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        JSON形式のレポートを生成
        
        Args:
            result: t検定結果
            test_name: テスト名
            metadata: 追加メタデータ
            
        Returns:
            Dict: JSONレポート
        """
        report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "results": {
                "statistical_test": {
                    "test_type": "Welch's t-test",
                    "statistic": float(result.statistic),
                    "p_value": float(result.p_value),
                    "degrees_of_freedom": float(result.degrees_of_freedom),
                    "is_significant": bool(result.is_significant),
                    "alpha": 0.05
                },
                "descriptive_statistics": {
                    "group_a": {
                        "mean": float(result.mean_a),
                        "std": float(result.std_a),
                        "sample_size": result.sample_size_a
                    },
                    "group_b": {
                        "mean": float(result.mean_b),
                        "std": float(result.std_b),
                        "sample_size": result.sample_size_b
                    },
                    "mean_difference": float(result.mean_a - result.mean_b)
                },
                "confidence_interval": {
                    "level": 0.95,
                    "lower": float(result.confidence_interval[0]),
                    "upper": float(result.confidence_interval[1])
                },
                "effect_size": {
                    "cohens_d": float(result.effect_size),
                    "interpretation": StatisticalValidator().interpret_effect_size(
                        result.effect_size
                    )
                },
                "interpretation": result.interpretation
            }
        }
        
        return report
    
    def save_json_report(
        self, 
        result: TTestResult,
        filename: str = "statistical_report.json",
        test_name: str = "t-test",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        JSON形式のレポートを保存
        
        Args:
            result: t検定結果
            filename: ファイル名
            test_name: テスト名
            metadata: 追加メタデータ
            
        Returns:
            Path: 保存先パス
        """
        report = self.generate_json_report(result, test_name, metadata)
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_html_report(
        self, 
        result: TTestResult,
        test_name: str = "統計的有意性検定",
        group_a_name: str = "グループA",
        group_b_name: str = "グループB"
    ) -> str:
        """
        HTML形式のレポートを生成
        
        Args:
            result: t検定結果
            test_name: テスト名
            group_a_name: グループAの名前
            group_b_name: グループBの名前
            
        Returns:
            str: HTMLコンテンツ
        """
        # p値の表示形式
        p_display = f"{result.p_value:.4f}" if result.p_value >= 0.0001 else "< 0.0001"
        
        # 有意性の色分け
        sig_color = "#27ae60" if result.is_significant else "#e74c3c"
        sig_text = "有意差あり" if result.is_significant else "有意差なし"
        
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{test_name} - 統計検定レポート</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 10px;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        .significance {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            background-color: {sig_color};
        }}
        .interpretation {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .ci-display {{
            font-family: monospace;
            font-size: 1.2em;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{test_name}</h1>
            <p>ウェルチのt検定による統計的有意性検定</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>検定結果サマリー</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{p_display}</div>
                        <div class="stat-label">p値</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.statistic:.3f}</div>
                        <div class="stat-label">t統計量</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.degrees_of_freedom:.1f}</div>
                        <div class="stat-label">自由度</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.effect_size:.3f}</div>
                        <div class="stat-label">効果サイズ (Cohen's d)</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 20px 0;">
                    <span class="significance">{sig_text}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>記述統計量</h2>
                <table>
                    <thead>
                        <tr>
                            <th>グループ</th>
                            <th>平均値</th>
                            <th>標準偏差</th>
                            <th>サンプルサイズ</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{group_a_name}</td>
                            <td>{result.mean_a:.4f}</td>
                            <td>{result.std_a:.4f}</td>
                            <td>{result.sample_size_a}</td>
                        </tr>
                        <tr>
                            <td>{group_b_name}</td>
                            <td>{result.mean_b:.4f}</td>
                            <td>{result.std_b:.4f}</td>
                            <td>{result.sample_size_b}</td>
                        </tr>
                        <tr style="font-weight: bold;">
                            <td>差</td>
                            <td>{result.mean_a - result.mean_b:.4f}</td>
                            <td>-</td>
                            <td>-</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>95%信頼区間</h2>
                <div class="ci-display">
                    [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]
                </div>
                <p>平均値の差の95%信頼区間が0を含まない場合、統計的に有意な差があると判断されます。</p>
            </div>
            
            <div class="section">
                <h2>解釈</h2>
                <div class="interpretation">
                    <p>{result.interpretation}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>技術的詳細</h2>
                <ul>
                    <li>検定手法: ウェルチのt検定（不等分散を仮定）</li>
                    <li>有意水準: α = 0.05</li>
                    <li>対立仮説: 両側検定（二つのグループの平均に差がある）</li>
                    <li>効果サイズ: Cohen's d（プールされた標準偏差を使用）</li>
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #7f8c8d; border-top: 1px solid #ecf0f1;">
            生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def save_html_report(
        self, 
        result: TTestResult,
        filename: str = "statistical_report.html",
        test_name: str = "統計的有意性検定",
        group_a_name: str = "グループA",
        group_b_name: str = "グループB"
    ) -> Path:
        """
        HTML形式のレポートを保存
        
        Args:
            result: t検定結果
            filename: ファイル名
            test_name: テスト名
            group_a_name: グループAの名前
            group_b_name: グループBの名前
            
        Returns:
            Path: 保存先パス
        """
        html = self.generate_html_report(result, test_name, group_a_name, group_b_name)
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return filepath
    
    def plot_comparison(
        self,
        group_a: Union[List[float], np.ndarray],
        group_b: Union[List[float], np.ndarray],
        result: TTestResult,
        group_a_name: str = "グループA",
        group_b_name: str = "グループB",
        title: str = "グループ間比較"
    ) -> plt.Figure:
        """
        グループ比較の可視化（箱ひげ図と信頼区間）
        
        Args:
            group_a: グループAのデータ
            group_b: グループBのデータ
            result: t検定結果
            group_a_name: グループAの名前
            group_b_name: グループBの名前
            title: グラフタイトル
            
        Returns:
            plt.Figure: 生成した図
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 箱ひげ図
        data = [group_a, group_b]
        positions = [1, 2]
        bp = ax1.boxplot(data, positions=positions, widths=0.6, 
                         patch_artist=True, labels=[group_a_name, group_b_name])
        
        # 色分け
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # 平均値をプロット
        ax1.scatter([1], [result.mean_a], color='black', s=100, zorder=3, marker='D')
        ax1.scatter([2], [result.mean_b], color='black', s=100, zorder=3, marker='D')
        
        ax1.set_ylabel('値')
        ax1.set_title('グループ間比較（箱ひげ図）')
        ax1.grid(True, alpha=0.3)
        
        # 凡例
        black_diamond = mpatches.Patch(color='black', label='平均値')
        ax1.legend(handles=[black_diamond], loc='upper right')
        
        # 効果サイズと信頼区間の可視化
        mean_diff = result.mean_a - result.mean_b
        ci_lower, ci_upper = result.confidence_interval
        
        # 信頼区間プロット
        ax2.errorbar([0], [mean_diff], 
                    yerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]],
                    fmt='o', markersize=10, capsize=10, capthick=2,
                    color='#2ecc71' if result.is_significant else '#95a5a6')
        
        # ゼロライン
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='差なし')
        
        # 効果サイズの表示
        effect_interpretation = StatisticalValidator().interpret_effect_size(result.effect_size)
        
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['平均値の差'])
        ax2.set_ylabel('差')
        ax2.set_title(f'95%信頼区間\n効果サイズ (d={result.effect_size:.3f}): {effect_interpretation}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # p値とサンプルサイズの表示
        fig.suptitle(
            f'{title}\np値 = {result.p_value:.4f}, '
            f'n₁ = {result.sample_size_a}, n₂ = {result.sample_size_b}',
            fontsize=14
        )
        
        plt.tight_layout()
        return fig
    
    def save_plot(
        self,
        group_a: Union[List[float], np.ndarray],
        group_b: Union[List[float], np.ndarray],
        result: TTestResult,
        filename: str = "comparison_plot.png",
        group_a_name: str = "グループA",
        group_b_name: str = "グループB",
        title: str = "グループ間比較"
    ) -> Path:
        """
        比較プロットを保存
        
        Args:
            group_a: グループAのデータ
            group_b: グループBのデータ
            result: t検定結果
            filename: ファイル名
            group_a_name: グループAの名前
            group_b_name: グループBの名前
            title: グラフタイトル
            
        Returns:
            Path: 保存先パス
        """
        fig = self.plot_comparison(
            group_a, group_b, result, 
            group_a_name, group_b_name, title
        )
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def generate_multiple_comparison_report(
        self,
        results: Dict[str, TTestResult],
        correction_method: Optional[str] = 'bonferroni'
    ) -> Dict:
        """
        複数比較結果のレポート生成
        
        Args:
            results: 比較名をキーとする検定結果の辞書
            correction_method: 多重比較補正方法
            
        Returns:
            Dict: 複数比較レポート
        """
        report = {
            "n_comparisons": len(results),
            "correction_method": correction_method,
            "comparisons": {}
        }
        
        # p値の抽出
        p_values = [r.p_value for r in results.values()]
        
        # 多重比較補正
        if correction_method and len(p_values) > 1:
            validator = StatisticalValidator()
            corrected_p = validator.perform_multiple_comparison_correction(
                p_values, correction_method
            )
        else:
            corrected_p = p_values
        
        # 各比較結果を整理
        for (name, result), corrected in zip(results.items(), corrected_p):
            report["comparisons"][name] = {
                "original_p_value": float(result.p_value),
                "corrected_p_value": float(corrected),
                "effect_size": float(result.effect_size),
                "is_significant_original": result.is_significant,
                "is_significant_corrected": corrected < 0.05,
                "mean_difference": float(result.mean_a - result.mean_b),
                "confidence_interval": [
                    float(result.confidence_interval[0]),
                    float(result.confidence_interval[1])
                ]
            }
        
        return report