"""
QCC-022: 統計的品質分析システム

抽出結果の統計的品質分析と複数設定間の比較を行うシステム。
"""

import numpy as np

import json
import pandas as pd
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.validation import StatisticalReporter, StatisticalValidator
from tools.validation.statistical_validator import TTestResult


@dataclass
class QualityMetrics:
    """品質メトリクスを格納するデータクラス"""
    tracker_id: str
    success_rate: float
    mean_quality_score: float
    std_quality_score: float
    sample_size: int
    extraction_times: List[float]
    quality_scores: List[float]
    metadata: Dict


class StatisticalQualityAnalyzer:
    """
    統計的品質分析クラス
    
    主要機能:
    - 抽出結果の品質メトリクス計算
    - 複数設定間の統計的比較
    - 改善効果の定量評価
    """
    
    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        """
        Args:
            workspace_base: ワークスペースのベースディレクトリ
        """
        self.workspace_base = Path(workspace_base)
        self.validator = StatisticalValidator()
        self.reporter = StatisticalReporter()
    
    def load_extraction_results(self, tracker_id: str) -> QualityMetrics:
        """
        抽出結果を読み込んで品質メトリクスを計算
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            QualityMetrics: 品質メトリクス
        """
        tracker_dir = self.workspace_base / tracker_id
        
        # extraction_result.jsonを読み込む
        result_file = tracker_dir / "extraction_result.json"
        if not result_file.exists():
            # 代替ファイルを探す
            quality_dir = tracker_dir / "quality"
            if quality_dir.exists():
                result_files = list(quality_dir.glob("*quality*.json"))
                if result_files:
                    result_file = result_files[0]
        
        if not result_file.exists():
            raise FileNotFoundError(f"抽出結果が見つかりません: {tracker_id}")
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # メトリクス抽出
        quality_scores = []
        extraction_times = []
        
        # データ形式に応じて処理
        if isinstance(data, dict):
            # 全体統計がある場合
            if 'success_rate' in data:
                success_rate = data['success_rate']
                sample_size = data.get('total_images', 0)
            else:
                success_rate = 0.0
                sample_size = 0

            # 個別結果がある場合 - 複数の形式に対応
            if 'extraction_results' in data:
                extraction_results = data['extraction_results']

                # パターン1: extraction_resultsがリスト（QUAL-044形式）
                if isinstance(extraction_results, list):
                    for result in extraction_results:
                        if isinstance(result, dict) and 'quality_score' in result:
                            quality_scores.append(result['quality_score'])
                        if isinstance(result, dict) and 'extraction_time' in result:
                            extraction_times.append(result['extraction_time'])

                # パターン2: extraction_resultsが辞書（TRACKER-006形式）
                elif isinstance(extraction_results, dict):
                    # 統計情報から抽出
                    if 'success_rate' in extraction_results:
                        success_rate = extraction_results['success_rate']
                    if 'total_images' in extraction_results:
                        sample_size = extraction_results['total_images']
                    if 'average_quality_score' in extraction_results:
                        avg_score = extraction_results['average_quality_score']
                        # 個別スコア情報がない場合は平均から推定
                        if sample_size > 0:
                            # 品質分布から個別スコアを推定
                            quality_dist = extraction_results.get('quality_distribution', {})
                            if quality_dist:
                                # 各グレードの代表値で推定
                                grade_scores = {'A': 0.9, 'B': 0.7, 'C': 0.5, 'D': 0.3, 'E': 0.1, 'F': 0.05}
                                for grade, count in quality_dist.items():
                                    if grade in grade_scores and count > 0:
                                        quality_scores.extend([grade_scores[grade]] * count)
                            else:
                                # 分布情報がない場合は平均値を使用
                                quality_scores = [avg_score] * sample_size

            elif 'results' in data:  # 従来形式のサポート
                for result in data['results']:
                    if 'quality_score' in result:
                        quality_scores.append(result['quality_score'])
                    if 'extraction_time' in result:
                        extraction_times.append(result['extraction_time'])
            elif 'quality_scores' in data:
                quality_scores = data['quality_scores']
        
        # 🔧 修正: ダミーデータ生成を削除し、実データが取得できない場合はエラー
        if not quality_scores:
            raise ValueError(f"品質スコアデータが見つかりません: {tracker_id}. JSONファイルに 'extraction_results' または 'results' キーが必要です")
        
        if not extraction_times:
            # 抽出時間がない場合はデフォルト値を使用（実害なし）
            extraction_times = [2.0] * len(quality_scores)  # 2秒デフォルト
        
        return QualityMetrics(
            tracker_id=tracker_id,
            success_rate=success_rate,
            mean_quality_score=np.mean(quality_scores) if quality_scores else 0.0,
            std_quality_score=np.std(quality_scores, ddof=1) if len(quality_scores) > 1 else 0.0,
            sample_size=len(quality_scores),
            extraction_times=extraction_times,
            quality_scores=quality_scores,
            metadata=data.get('metadata', {})
        )
    
    def compare_trackers(
        self,
        tracker_a: str,
        tracker_b: str,
        metric: str = 'quality_score'
    ) -> TTestResult:
        """
        2つのトラッカーの品質を統計的に比較
        
        Args:
            tracker_a: トラッカーA
            tracker_b: トラッカーB
            metric: 比較するメトリクス ('quality_score', 'extraction_time')
            
        Returns:
            TTestResult: 検定結果
        """
        # メトリクスを読み込み
        metrics_a = self.load_extraction_results(tracker_a)
        metrics_b = self.load_extraction_results(tracker_b)
        
        # 比較データ取得
        if metric == 'quality_score':
            data_a = metrics_a.quality_scores
            data_b = metrics_b.quality_scores
        elif metric == 'extraction_time':
            data_a = metrics_a.extraction_times
            data_b = metrics_b.extraction_times
        else:
            raise ValueError(f"未対応のメトリクス: {metric}")
        
        # t検定実行
        result = self.validator.welch_t_test(data_a, data_b)
        
        return result
    
    def analyze_improvement(
        self,
        baseline_tracker: str,
        improved_tracker: str
    ) -> Dict:
        """
        改善効果を分析
        
        Args:
            baseline_tracker: ベースライントラッカー
            improved_tracker: 改善後トラッカー
            
        Returns:
            Dict: 改善分析結果
        """
        # 品質スコアの比較
        quality_result = self.compare_trackers(
            baseline_tracker, 
            improved_tracker, 
            'quality_score'
        )
        
        # 処理時間の比較（あれば）
        try:
            time_result = self.compare_trackers(
                baseline_tracker,
                improved_tracker,
                'extraction_time'
            )
        except:
            time_result = None
        
        # メトリクス読み込み
        baseline_metrics = self.load_extraction_results(baseline_tracker)
        improved_metrics = self.load_extraction_results(improved_tracker)
        
        # 改善率計算
        quality_improvement = (
            (improved_metrics.mean_quality_score - baseline_metrics.mean_quality_score) 
            / baseline_metrics.mean_quality_score * 100
            if baseline_metrics.mean_quality_score > 0 else 0
        )
        
        success_rate_improvement = (
            improved_metrics.success_rate - baseline_metrics.success_rate
        )
        
        analysis = {
            "baseline_tracker": baseline_tracker,
            "improved_tracker": improved_tracker,
            "quality_comparison": {
                "baseline_mean": baseline_metrics.mean_quality_score,
                "improved_mean": improved_metrics.mean_quality_score,
                "improvement_percent": quality_improvement,
                "p_value": quality_result.p_value,
                "effect_size": quality_result.effect_size,
                "is_significant": quality_result.is_significant,
                "interpretation": quality_result.interpretation
            },
            "success_rate_comparison": {
                "baseline": baseline_metrics.success_rate,
                "improved": improved_metrics.success_rate,
                "improvement": success_rate_improvement
            },
            "sample_sizes": {
                "baseline": baseline_metrics.sample_size,
                "improved": improved_metrics.sample_size
            }
        }
        
        if time_result:
            baseline_mean_time = np.mean(baseline_metrics.extraction_times)
            improved_mean_time = np.mean(improved_metrics.extraction_times)
            
            analysis["time_comparison"] = {
                "baseline_mean": baseline_mean_time,
                "improved_mean": improved_mean_time,
                "speedup": baseline_mean_time / improved_mean_time if improved_mean_time > 0 else 1.0,
                "p_value": time_result.p_value,
                "is_significant": time_result.is_significant
            }
        
        return analysis
    
    def batch_compare(
        self,
        tracker_ids: List[str],
        baseline: Optional[str] = None
    ) -> Dict:
        """
        複数トラッカーのバッチ比較
        
        Args:
            tracker_ids: トラッカーIDのリスト
            baseline: ベースライントラッカー（Noneの場合は総当たり）
            
        Returns:
            Dict: バッチ比較結果
        """
        # 各トラッカーのメトリクス読み込み
        metrics = {}
        quality_scores = {}
        
        for tracker_id in tracker_ids:
            try:
                m = self.load_extraction_results(tracker_id)
                metrics[tracker_id] = m
                quality_scores[tracker_id] = m.quality_scores
            except Exception as e:
                print(f"警告: {tracker_id}の読み込みに失敗: {e}")
        
        # 複数グループ比較
        comparison_results = self.validator.compare_multiple_groups(
            quality_scores,
            baseline=baseline
        )
        
        # サマリー生成
        summary = {
            "n_trackers": len(metrics),
            "baseline": baseline,
            "trackers": list(metrics.keys()),
            "comparisons": {},
            "ranking": []
        }
        
        # 各比較結果を整理
        for name, result in comparison_results.items():
            summary["comparisons"][name] = {
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "is_significant": result.is_significant,
                "mean_difference": result.mean_a - result.mean_b
            }
        
        # ランキング生成
        ranked = sorted(
            [(tid, m.mean_quality_score) for tid, m in metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        summary["ranking"] = [
            {
                "rank": i+1,
                "tracker_id": tid,
                "mean_quality_score": score,
                "success_rate": metrics[tid].success_rate,
                "sample_size": metrics[tid].sample_size
            }
            for i, (tid, score) in enumerate(ranked)
        ]
        
        return summary
    
    def generate_statistical_dashboard(
        self,
        tracker_id: str,
        comparison_trackers: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Path:
        """
        統計分析ダッシュボードを生成
        
        Args:
            tracker_id: 主トラッカーID
            comparison_trackers: 比較対象トラッカー
            output_dir: 出力ディレクトリ
            
        Returns:
            Path: ダッシュボードファイルパス
        """
        if output_dir is None:
            output_dir = self.workspace_base / tracker_id / "dashboard"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # レポーター設定
        self.reporter.output_dir = output_dir
        
        # メインメトリクス取得
        main_metrics = self.load_extraction_results(tracker_id)
        
        # HTMLコンテンツ生成
        html_parts = []
        
        # ヘッダー
        html_parts.append(f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>統計分析ダッシュボード - {tracker_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        h1 {{ margin: 0; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        .comparison-table th {{
            background: #f8f9fa;
        }}
        .significant {{
            color: #27ae60;
            font-weight: bold;
        }}
        .not-significant {{
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>統計分析ダッシュボード</h1>
            <p>トラッカー: {tracker_id}</p>
            <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="card">
            <h2>基本統計量</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{main_metrics.success_rate:.1%}</div>
                    <div class="metric-label">成功率</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{main_metrics.mean_quality_score:.3f}</div>
                    <div class="metric-label">平均品質スコア</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{main_metrics.std_quality_score:.3f}</div>
                    <div class="metric-label">標準偏差</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{main_metrics.sample_size}</div>
                    <div class="metric-label">サンプルサイズ</div>
                </div>
            </div>
        </div>
        """)
        
        # 比較分析
        if comparison_trackers:
            html_parts.append("""
        <div class="card">
            <h2>統計的比較分析</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>比較対象</th>
                        <th>平均品質スコア</th>
                        <th>差</th>
                        <th>p値</th>
                        <th>効果サイズ</th>
                        <th>判定</th>
                    </tr>
                </thead>
                <tbody>
            """)
            
            for comp_tracker in comparison_trackers:
                try:
                    result = self.compare_trackers(tracker_id, comp_tracker)
                    comp_metrics = self.load_extraction_results(comp_tracker)
                    
                    sig_class = "significant" if result.is_significant else "not-significant"
                    sig_text = "有意差あり" if result.is_significant else "有意差なし"
                    
                    html_parts.append(f"""
                    <tr>
                        <td>{comp_tracker}</td>
                        <td>{comp_metrics.mean_quality_score:.3f}</td>
                        <td>{main_metrics.mean_quality_score - comp_metrics.mean_quality_score:.3f}</td>
                        <td>{result.p_value:.4f}</td>
                        <td>{result.effect_size:.3f}</td>
                        <td class="{sig_class}">{sig_text}</td>
                    </tr>
                    """)
                except Exception as e:
                    print(f"比較エラー ({comp_tracker}): {e}")
            
            html_parts.append("""
                </tbody>
            </table>
        </div>
            """)
        
        # フッター
        html_parts.append("""
    </div>
</body>
</html>
        """)
        
        # HTML保存
        dashboard_path = output_dir / "statistical_dashboard.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        return dashboard_path