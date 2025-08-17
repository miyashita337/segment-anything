#!/usr/bin/env python3
"""
T-004: 品質トレンド分析ダッシュボード生成システム
HTMLダッシュボードで時系列品質変化を可視化
"""

import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class QualityTrendDashboard:
    """品質トレンドダッシュボード生成クラス"""
    
    def __init__(self, trend_report_path: str = "/tmp/t004_quality_trend_report.json"):
        """初期化"""
        self.trend_report_path = Path(trend_report_path)
        self.report_data = None
        self.charts = {}
        
    def load_report(self) -> bool:
        """トレンド分析レポート読み込み"""
        try:
            with open(self.trend_report_path, 'r', encoding='utf-8') as f:
                self.report_data = json.load(f)
            logger.info(f"レポート読み込み成功: {self.trend_report_path}")
            return True
        except Exception as e:
            logger.error(f"レポート読み込み失敗: {e}")
            return False
    
    def generate_charts(self) -> Dict[str, str]:
        """チャート生成（Base64エンコード）"""
        if not self.report_data:
            self.load_report()
        
        charts = {}
        
        # 1. 時系列品質トレンドチャート
        charts['time_series_chart'] = self._create_time_series_chart()
        
        # 2. トラッカー別品質比較チャート
        charts['tracker_comparison'] = self._create_tracker_comparison_chart()
        
        # 3. 品質分布ヒストグラム
        charts['quality_distribution'] = self._create_distribution_chart()
        
        # 4. トレンド予測チャート
        charts['trend_prediction'] = self._create_prediction_chart()
        
        # 5. 異常検知チャート
        charts['anomaly_detection'] = self._create_anomaly_chart()
        
        self.charts = charts
        return charts
    
    def _create_time_series_chart(self) -> str:
        """時系列品質トレンドチャート生成"""
        try:
            viz_data = self.report_data.get('detailed_analysis', {}).get('visualization_data', {})
            time_series = viz_data.get('time_series_data', [])
            
            if not time_series:
                return self._create_placeholder_chart("No time series data available")
            
            # データフレーム作成
            df = pd.DataFrame(time_series)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # プロット作成
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # トラッカー別に色分け
            for tracker in df['tracker_id'].unique():
                tracker_data = df[df['tracker_id'] == tracker]
                ax.plot(tracker_data['timestamp'], tracker_data['quality_score'], 
                       marker='o', label=tracker, alpha=0.7)
            
            # トレンドライン追加
            if len(df) >= 2:
                z = np.polyfit(df.index, df['quality_score'], 1)
                p = np.poly1d(z)
                ax.plot(df['timestamp'], p(df.index), "--", alpha=0.5, color='red', 
                       label=f'Trend (slope: {z[0]:.4f})')
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Quality Score')
            ax.set_title('Quality Trend Over Time')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"時系列チャート生成エラー: {e}")
            return self._create_placeholder_chart("Error generating time series chart")
    
    def _create_tracker_comparison_chart(self) -> str:
        """トラッカー別品質比較チャート"""
        try:
            tracker_data = self.report_data.get('detailed_analysis', {}).get('tracker_analysis', {}).get('tracker_statistics', {})
            
            if not tracker_data:
                return self._create_placeholder_chart("No tracker data available")
            
            # データ準備
            trackers = list(tracker_data.keys())
            avg_qualities = [data['avg_quality'] for data in tracker_data.values()]
            improvements = [data.get('improvement', 0) for data in tracker_data.values()]
            
            # サブプロット作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 平均品質比較
            colors = ['green' if q >= 0.7 else 'orange' if q >= 0.5 else 'red' for q in avg_qualities]
            ax1.bar(trackers, avg_qualities, color=colors, alpha=0.7)
            ax1.set_xlabel('Tracker ID')
            ax1.set_ylabel('Average Quality Score')
            ax1.set_title('Average Quality by Tracker')
            ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good threshold')
            ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Fair threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 改善度比較
            colors2 = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(trackers, improvements, color=colors2, alpha=0.7)
            ax2.set_xlabel('Tracker ID')
            ax2.set_ylabel('Quality Improvement')
            ax2.set_title('Quality Improvement by Tracker')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"トラッカー比較チャート生成エラー: {e}")
            return self._create_placeholder_chart("Error generating tracker comparison")
    
    def _create_distribution_chart(self) -> str:
        """品質分布ヒストグラム"""
        try:
            viz_data = self.report_data.get('detailed_analysis', {}).get('visualization_data', {})
            time_series = viz_data.get('time_series_data', [])
            
            if not time_series:
                return self._create_placeholder_chart("No distribution data available")
            
            # 品質スコア抽出
            quality_scores = [item['quality_score'] for item in time_series]
            
            # ヒストグラム作成
            fig, ax = plt.subplots(figsize=(10, 6))
            
            n, bins, patches = ax.hist(quality_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            
            # 統計値追加
            mean_val = np.mean(quality_scores)
            median_val = np.median(quality_scores)
            std_val = np.std(quality_scores)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # 色分け
            for i, patch in enumerate(patches):
                if bins[i] < 0.5:
                    patch.set_facecolor('red')
                elif bins[i] < 0.7:
                    patch.set_facecolor('orange')
                else:
                    patch.set_facecolor('green')
            
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Quality Score Distribution (σ={std_val:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"分布チャート生成エラー: {e}")
            return self._create_placeholder_chart("Error generating distribution chart")
    
    def _create_prediction_chart(self) -> str:
        """トレンド予測チャート"""
        try:
            predictions = self.report_data.get('detailed_analysis', {}).get('predictions', {})
            
            if 'error' in predictions or not predictions.get('predictions'):
                return self._create_placeholder_chart("Insufficient data for predictions")
            
            # 予測データ準備
            pred_data = predictions['predictions']
            dates = [pd.to_datetime(p['date']) for p in pred_data]
            pred_values = [p['predicted_quality'] for p in pred_data]
            confidence = [p['confidence'] for p in pred_data]
            
            # 既存データも含める
            viz_data = self.report_data.get('detailed_analysis', {}).get('visualization_data', {})
            time_series = viz_data.get('time_series_data', [])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 既存データプロット
            if time_series:
                df_existing = pd.DataFrame(time_series)
                df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                ax.plot(df_existing['timestamp'], df_existing['quality_score'], 
                       'o-', color='blue', label='Actual', alpha=0.7)
            
            # 予測データプロット
            ax.plot(dates, pred_values, 's--', color='red', label='Predicted', alpha=0.7)
            
            # 信頼区間
            upper_bound = [v + (1-c)*0.1 for v, c in zip(pred_values, confidence)]
            lower_bound = [v - (1-c)*0.1 for v, c in zip(pred_values, confidence)]
            ax.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color='red', label='Confidence interval')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Quality Score')
            ax.set_title(f'Quality Trend Prediction (Model Accuracy: {predictions.get("model_accuracy", 0):.2%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"予測チャート生成エラー: {e}")
            return self._create_placeholder_chart("Error generating prediction chart")
    
    def _create_anomaly_chart(self) -> str:
        """異常検知チャート"""
        try:
            anomalies = self.report_data.get('detailed_analysis', {}).get('anomalies', [])
            viz_data = self.report_data.get('detailed_analysis', {}).get('visualization_data', {})
            time_series = viz_data.get('time_series_data', [])
            
            if not time_series:
                return self._create_placeholder_chart("No data for anomaly detection")
            
            # データフレーム作成
            df = pd.DataFrame(time_series)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 通常データプロット
            ax.scatter(df['timestamp'], df['quality_score'], alpha=0.5, color='blue', label='Normal')
            
            # 異常値プロット
            if anomalies:
                anomaly_times = [pd.to_datetime(a['timestamp']) for a in anomalies]
                anomaly_scores = [a['quality_score'] for a in anomalies]
                anomaly_severities = [a['severity'] for a in anomalies]
                
                colors = ['red' if s == 'high' else 'orange' for s in anomaly_severities]
                sizes = [100 if s == 'high' else 50 for s in anomaly_severities]
                
                ax.scatter(anomaly_times, anomaly_scores, c=colors, s=sizes, 
                          marker='^', edgecolors='black', linewidths=2, 
                          label='Anomaly', alpha=0.8)
            
            # 平均線と標準偏差バンド
            mean_val = df['quality_score'].mean()
            std_val = df['quality_score'].std()
            
            ax.axhline(mean_val, color='green', linestyle='-', alpha=0.3, label=f'Mean: {mean_val:.3f}')
            ax.axhline(mean_val + 2*std_val, color='red', linestyle='--', alpha=0.3, label='±2σ')
            ax.axhline(mean_val - 2*std_val, color='red', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Quality Score')
            ax.set_title(f'Anomaly Detection ({len(anomalies)} anomalies detected)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"異常検知チャート生成エラー: {e}")
            return self._create_placeholder_chart("Error generating anomaly chart")
    
    def _create_placeholder_chart(self, message: str) -> str:
        """プレースホルダーチャート生成"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """MatplotlibフィギュアをBase64エンコード"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def generate_dashboard_html(self, output_path: str = "/tmp/t004_quality_trend_dashboard.html") -> str:
        """HTMLダッシュボード生成"""
        logger.info("品質トレンドダッシュボード生成開始")
        
        if not self.report_data:
            self.load_report()
        
        # チャート生成
        if not self.charts:
            self.generate_charts()
        
        # HTML生成
        html_content = self._generate_html_template()
        
        # ファイル保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"ダッシュボード生成完了: {output_path}")
        return output_path
    
    def _generate_html_template(self) -> str:
        """HTMLテンプレート生成"""
        summary = self.report_data.get('executive_summary', {})
        basic_stats = self.report_data.get('detailed_analysis', {}).get('basic_statistics', {})
        recommendations = self.report_data.get('recommendations', [])
        
        # ステータスバッジ色決定
        status = summary.get('overall_status', 'unknown')
        status_color = 'success' if status == 'healthy' else 'warning' if status == 'needs_attention' else 'secondary'
        
        # トレンド方向アイコン
        trend_direction = summary.get('trend_summary', {}).get('direction', 'unknown')
        trend_icon = '📈' if trend_direction == 'improving' else '📉' if trend_direction == 'declining' else '➡️'
        
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>T-004: 品質トレンド分析ダッシュボード</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .dashboard-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin: 30px auto;
            max-width: 1400px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .recommendation-card {{
            background: white;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .priority-high {{
            border-left-color: #dc3545;
        }}
        .priority-medium {{
            border-left-color: #ffc107;
        }}
        .priority-low {{
            border-left-color: #28a745;
        }}
        .header-gradient {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px 15px 0 0;
            margin: -30px -30px 30px -30px;
        }}
        .finding-item {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header-gradient">
            <h1 class="display-4 mb-3">🎯 T-004: 品質トレンド分析ダッシュボード</h1>
            <p class="lead mb-0">時系列品質変化の追跡・分析・予測システム</p>
            <p class="mt-2">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <!-- エグゼクティブサマリー -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h3 class="mb-3">📊 エグゼクティブサマリー</h3>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="text-center">
                                <div class="metric-label">Overall Status</div>
                                <span class="badge bg-{status_color} fs-5 mt-2">{status.upper()}</span>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <div class="metric-label">Total Records</div>
                                <div class="metric-value">{summary.get('data_coverage', {}).get('total_records', 0)}</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <div class="metric-label">Avg Quality</div>
                                <div class="metric-value">{summary.get('quality_overview', {}).get('current_average', 0):.3f}</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <div class="metric-label">Trend</div>
                                <div class="metric-value">{trend_icon} {trend_direction}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 主要な発見 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h3 class="mb-3">🔍 主要な発見</h3>
                    {''.join([f'<div class="finding-item">{finding}</div>' for finding in summary.get('key_findings', [])])}
                </div>
            </div>
        </div>
        
        <!-- チャート表示 -->
        <div class="row">
            <div class="col-12">
                <div class="chart-container">
                    <div class="chart-title">📈 時系列品質トレンド</div>
                    <img src="{self.charts.get('time_series_chart', '')}" class="img-fluid" alt="Time Series Chart">
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <div class="chart-title">📊 トラッカー別品質比較</div>
                    <img src="{self.charts.get('tracker_comparison', '')}" class="img-fluid" alt="Tracker Comparison">
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <div class="chart-title">📉 品質分布</div>
                    <img src="{self.charts.get('quality_distribution', '')}" class="img-fluid" alt="Quality Distribution">
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <div class="chart-title">🔮 トレンド予測</div>
                    <img src="{self.charts.get('trend_prediction', '')}" class="img-fluid" alt="Trend Prediction">
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <div class="chart-title">⚠️ 異常検知</div>
                    <img src="{self.charts.get('anomaly_detection', '')}" class="img-fluid" alt="Anomaly Detection">
                </div>
            </div>
        </div>
        
        <!-- 推奨事項 -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="metric-card">
                    <h3 class="mb-3">💡 推奨事項</h3>
                    {''.join([self._format_recommendation(rec) for rec in recommendations[:5]])}
                </div>
            </div>
        </div>
        
        <!-- 統計詳細 -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="metric-card">
                    <h3 class="mb-3">📊 統計詳細</h3>
                    <table class="table table-hover">
                        <tbody>
                            <tr>
                                <td><strong>データ期間</strong></td>
                                <td>{basic_stats.get('time_range', {}).get('start', 'N/A')} ～ {basic_stats.get('time_range', {}).get('end', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td><strong>期間（日数）</strong></td>
                                <td>{basic_stats.get('time_range', {}).get('duration_days', 0)}日</td>
                            </tr>
                            <tr>
                                <td><strong>トラッカー数</strong></td>
                                <td>{basic_stats.get('unique_trackers', 0)}</td>
                            </tr>
                            <tr>
                                <td><strong>品質スコア範囲</strong></td>
                                <td>{basic_stats.get('quality_stats', {}).get('min', 0):.3f} ～ {basic_stats.get('quality_stats', {}).get('max', 0):.3f}</td>
                            </tr>
                            <tr>
                                <td><strong>標準偏差</strong></td>
                                <td>{basic_stats.get('quality_stats', {}).get('std', 0):.3f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="text-center mt-4 text-muted">
            <small>Generated by T-004 Quality Trend Analysis System v1.0.0</small>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        return html
    
    def _format_recommendation(self, rec: Dict[str, str]) -> str:
        """推奨事項フォーマット"""
        priority_class = f"priority-{rec.get('priority', 'low')}"
        return f"""
        <div class="recommendation-card {priority_class}">
            <h5>{rec.get('title', 'Recommendation')}</h5>
            <p class="mb-1">{rec.get('description', '')}</p>
            <small class="text-muted"><strong>Action:</strong> {rec.get('action', '')}</small>
        </div>
        """


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='T-004: 品質トレンドダッシュボード生成')
    parser.add_argument('--report', default="/tmp/t004_quality_trend_report.json",
                       help='トレンド分析レポートファイル')
    parser.add_argument('--output', default="/tmp/t004_quality_trend_dashboard.html",
                       help='出力HTMLファイル')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # ダッシュボード生成
    dashboard = QualityTrendDashboard(args.report)
    dashboard_path = dashboard.generate_dashboard_html(args.output)
    
    print(f"✅ T-004品質トレンドダッシュボード生成完了")
    print(f"📊 ダッシュボード: {dashboard_path}")
    print(f"🌐 ブラウザで開く: file://{dashboard_path}")


if __name__ == "__main__":
    main()