#!/usr/bin/env python3
"""
統合品質チェッカー結果の時系列トレンド分析

複数の品質レポートJSONファイルを読み込んで、
品質指標の時系列変化を分析し、改善/悪化傾向を判定する
"""

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class QualitySnapshot:
    """品質スナップショット"""
    timestamp: datetime
    filename: str
    overall_score: float
    passed_metrics: int
    total_metrics: int
    
    # 個別指標
    largest_char_accuracy: Optional[float] = None
    ab_evaluation_rate: Optional[float] = None
    fps: Optional[float] = None
    c_above_rate: Optional[float] = None
    
    # マスク品質
    coverage_ratio: Optional[float] = None
    compactness: Optional[float] = None
    fill_rate: Optional[float] = None
    
    # 客観指標
    sci: Optional[float] = None
    pla: Optional[float] = None
    ple: Optional[float] = None


class QualityTrendAnalyzer:
    """品質トレンド分析器"""
    
    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.snapshots: List[QualitySnapshot] = []
        
    def load_reports(self, pattern: str = "unified_quality_report*.json") -> None:
        """レポートファイルを読み込み"""
        report_files = sorted(self.report_dir.glob(pattern))
        
        print(f"📁 {len(report_files)}個のレポートファイルを発見")
        
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # タイムスタンプをパース
                timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                # スナップショット作成
                snapshot = QualitySnapshot(
                    timestamp=timestamp,
                    filename=report_file.name,
                    overall_score=data.get('overall_score', 0),
                    passed_metrics=data.get('passed_metrics', 0),
                    total_metrics=data.get('total_metrics', 0)
                )
                
                # 個別指標を抽出
                for metric in data.get('evaluation_metrics', []):
                    if metric['name'] == 'Largest-Character Accuracy':
                        snapshot.largest_char_accuracy = metric['value']
                    elif metric['name'] == 'A/B評価率':
                        snapshot.ab_evaluation_rate = metric['value']
                    elif metric['name'] == 'FPS':
                        snapshot.fps = metric['value']
                    elif metric['name'] == 'C以上評価率':
                        snapshot.c_above_rate = metric['value']
                
                # マスク品質を抽出
                for metric in data.get('mask_metrics', []):
                    if metric['name'] == '平均カバレッジ率':
                        snapshot.coverage_ratio = metric['value']
                    elif metric['name'] == '平均コンパクトネス':
                        snapshot.compactness = metric['value']
                    elif metric['name'] == '平均フィル率':
                        snapshot.fill_rate = metric['value']
                
                # 客観指標を抽出
                for metric in data.get('objective_metrics', []):
                    if 'SCI' in metric['name']:
                        snapshot.sci = metric['value']
                    elif 'PLA' in metric['name']:
                        snapshot.pla = metric['value']
                    elif 'PLE' in metric['name']:
                        snapshot.ple = metric['value']
                
                self.snapshots.append(snapshot)
                print(f"  ✅ {report_file.name} - スコア: {snapshot.overall_score:.1%}")
                
            except Exception as e:
                print(f"  ❌ {report_file.name} - エラー: {e}")
        
        # タイムスタンプでソート
        self.snapshots.sort(key=lambda s: s.timestamp)
        
    def calculate_trends(self) -> Dict[str, Dict[str, float]]:
        """各指標のトレンドを計算"""
        if len(self.snapshots) < 2:
            print("⚠️ トレンド分析には2つ以上のデータポイントが必要です")
            return {}
        
        trends = {}
        
        # 分析する指標リスト
        metrics = [
            ('overall_score', '総合スコア'),
            ('largest_char_accuracy', 'キャラクター検出精度'),
            ('ab_evaluation_rate', 'A/B評価率'),
            ('fps', 'FPS'),
            ('c_above_rate', 'C以上評価率'),
            ('coverage_ratio', 'カバレッジ率'),
            ('compactness', 'コンパクトネス'),
            ('fill_rate', 'フィル率'),
            ('sci', 'SCI (意味的完全性)'),
            ('pla', 'PLA (ピクセル精度)'),
            ('ple', 'PLE (学習効率)')
        ]
        
        for metric_key, metric_name in metrics:
            values = []
            timestamps = []
            
            for snapshot in self.snapshots:
                value = getattr(snapshot, metric_key, None)
                if value is not None:
                    values.append(value)
                    timestamps.append(snapshot.timestamp)
            
            if len(values) >= 2:
                # 線形回帰で傾向を計算
                x = np.array([(t - timestamps[0]).total_seconds() / 3600 for t in timestamps])  # 時間単位
                y = np.array(values)
                
                # 線形フィット
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]  # 傾き（変化率/時間）
                
                # 統計情報
                mean_value = np.mean(y)
                std_value = np.std(y)
                latest_value = y[-1]
                initial_value = y[0]
                total_change = latest_value - initial_value
                percent_change = (total_change / initial_value * 100) if initial_value != 0 else 0
                
                trends[metric_key] = {
                    'name': metric_name,
                    'slope': slope,
                    'mean': mean_value,
                    'std': std_value,
                    'initial': initial_value,
                    'latest': latest_value,
                    'total_change': total_change,
                    'percent_change': percent_change,
                    'data_points': len(values),
                    'trend': '↗️' if slope > 0.001 else '↘️' if slope < -0.001 else '→'
                }
        
        return trends
    
    def generate_report(self) -> None:
        """トレンド分析レポートを生成"""
        print("\n" + "="*60)
        print("📊 品質トレンド分析レポート")
        print("="*60)
        
        if not self.snapshots:
            print("❌ レポートデータがありません")
            return
        
        # 基本情報
        print(f"\n📅 分析期間: {self.snapshots[0].timestamp} ～ {self.snapshots[-1].timestamp}")
        print(f"📁 データポイント数: {len(self.snapshots)}")
        
        # 時間スパン
        time_span = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 3600
        print(f"⏱️ 時間スパン: {time_span:.1f}時間")
        
        # トレンド分析
        trends = self.calculate_trends()
        
        if not trends:
            print("\n⚠️ トレンド分析に十分なデータがありません")
            return
        
        # 重要指標のサマリー
        print("\n🎯 主要指標のトレンド:")
        print("-" * 60)
        
        key_metrics = ['overall_score', 'largest_char_accuracy', 'ab_evaluation_rate', 
                      'sci', 'pla', 'ple']
        
        for metric in key_metrics:
            if metric in trends:
                t = trends[metric]
                print(f"\n{t['name']}:")
                print(f"  現在値: {t['latest']:.3f} {t['trend']}")
                print(f"  初期値: {t['initial']:.3f}")
                print(f"  変化量: {t['total_change']:+.3f} ({t['percent_change']:+.1f}%)")
                print(f"  傾向: {t['slope']:+.6f}/時間")
                
                # 判定
                if t['percent_change'] > 5:
                    status = "✅ 改善傾向"
                elif t['percent_change'] < -5:
                    status = "⚠️ 悪化傾向"
                else:
                    status = "➡️ 横ばい"
                print(f"  判定: {status}")
        
        # 総合判定
        print("\n" + "="*60)
        print("🏁 総合判定:")
        
        overall_trend = trends.get('overall_score', {})
        if overall_trend:
            if overall_trend['percent_change'] > 0:
                print(f"✅ 品質は改善傾向にあります (+{overall_trend['percent_change']:.1f}%)")
            elif overall_trend['percent_change'] < 0:
                print(f"⚠️ 品質は悪化傾向にあります ({overall_trend['percent_change']:.1f}%)")
            else:
                print("➡️ 品質は横ばいです")
        
        # 推奨事項
        print("\n💡 推奨事項:")
        
        # A/B評価率が低い場合
        ab_trend = trends.get('ab_evaluation_rate', {})
        if ab_trend and ab_trend['latest'] < 0.3:
            print("- A/B評価率が低い（{:.1%}）ため、品質基準の見直しが必要".format(ab_trend['latest']))
        
        # 検出精度が下がっている場合
        accuracy_trend = trends.get('largest_char_accuracy', {})
        if accuracy_trend and accuracy_trend['slope'] < -0.01:
            print("- キャラクター検出精度が低下傾向のため、モデルパラメータの調整が必要")
        
        # PLEが負の場合
        ple_trend = trends.get('ple', {})
        if ple_trend and ple_trend['latest'] < 0:
            print("- 学習効率（PLE）が負のため、手法の見直しが必要")
        
        print("\n" + "="*60)
    
    def plot_trends(self, output_path: Optional[str] = None) -> None:
        """トレンドグラフを生成"""
        if len(self.snapshots) < 2:
            print("⚠️ グラフ生成には2つ以上のデータポイントが必要です")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('品質指標トレンド分析', fontsize=16)
        
        # プロットする指標
        plot_configs = [
            (axes[0, 0], 'overall_score', '総合スコア', 'b-'),
            (axes[0, 1], 'largest_char_accuracy', 'キャラクター検出精度', 'g-'),
            (axes[1, 0], 'ab_evaluation_rate', 'A/B評価率', 'r-'),
            (axes[1, 1], 'sci', 'SCI (意味的完全性)', 'm-')
        ]
        
        for ax, metric_key, title, style in plot_configs:
            timestamps = []
            values = []
            
            for snapshot in self.snapshots:
                value = getattr(snapshot, metric_key, None)
                if value is not None:
                    timestamps.append(snapshot.timestamp)
                    values.append(value)
            
            if timestamps:
                ax.plot(timestamps, values, style, marker='o')
                ax.set_title(title)
                ax.set_xlabel('時刻')
                ax.set_ylabel('値')
                ax.grid(True, alpha=0.3)
                
                # X軸の日時フォーマット
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 グラフを保存しました: {output_path}")
        else:
            plt.show()


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="品質トレンド分析")
    parser.add_argument("--dir", "-d", 
                       default="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge",
                       help="レポートディレクトリ")
    parser.add_argument("--plot", "-p", action="store_true", help="グラフ生成")
    parser.add_argument("--output", "-o", help="グラフ出力パス")
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = QualityTrendAnalyzer(args.dir)
    analyzer.load_reports()
    analyzer.generate_report()
    
    if args.plot:
        analyzer.plot_trends(args.output)


if __name__ == "__main__":
    main()