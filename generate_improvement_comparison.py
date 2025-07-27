#!/usr/bin/env python3
"""
改善効果測定システム
ベースラインと現在の実装結果を比較し、定量的改善効果を測定
"""

import json
import argparse
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np

class ImprovementComparisonSystem:
    """改善効果比較システム"""
    
    def __init__(self):
        self.comparison_metrics = [
            'Largest-Character Accuracy',
            'A/B評価率', 
            'SCI (Semantic Completeness Index)',
            'PLA (Pixel-Level Accuracy)',
            'PLE (Progressive Learning Efficiency)',
            'FPS',
            'C以上評価率'
        ]
    
    def load_quality_report(self, report_path: str) -> Optional[Dict]:
        """品質レポートの読み込み"""
        if not os.path.exists(report_path):
            print(f"⚠️  品質レポートが見つかりません: {report_path}")
            return None
            
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 品質レポート読み込みエラー: {e}")
            return None
    
    def extract_metrics(self, report: Dict) -> Dict[str, float]:
        """レポートからメトリクスを抽出"""
        metrics = {}
        
        # 評価メトリクス
        for metric_group in ['evaluation_metrics', 'objective_metrics', 'mask_metrics']:
            if metric_group in report:
                for metric in report[metric_group]:
                    if isinstance(metric, dict) and 'name' in metric and 'value' in metric:
                        metrics[metric['name']] = metric['value']
        
        return metrics
    
    def calculate_improvement(self, baseline_value: float, current_value: float) -> Dict[str, Any]:
        """改善率計算"""
        if baseline_value == 0:
            if current_value == 0:
                return {
                    'absolute_change': 0,
                    'percent_change': 0,
                    'improvement_type': 'no_change'
                }
            else:
                return {
                    'absolute_change': current_value,
                    'percent_change': 100.0,  # 0から値があることは100%改善
                    'improvement_type': 'improvement'
                }
        
        absolute_change = current_value - baseline_value
        percent_change = (absolute_change / baseline_value) * 100
        
        improvement_type = 'improvement' if percent_change > 0 else 'degradation' if percent_change < 0 else 'no_change'
        
        return {
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'improvement_type': improvement_type
        }
    
    def find_quality_reports(self, directory: str) -> List[str]:
        """ディレクトリから品質レポートを検索"""
        report_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('unified_quality') and file.endswith('.json'):
                    report_files.append(os.path.join(root, file))
        
        # 最新のファイルを優先
        report_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return report_files
    
    def generate_comparison_report(self, baseline_dir: str, current_dir: str) -> Dict[str, Any]:
        """比較レポート生成"""
        
        # ベースライン品質レポート検索
        baseline_reports = self.find_quality_reports(baseline_dir)
        if not baseline_reports:
            print(f"❌ ベースライン品質レポートが見つかりません: {baseline_dir}")
            return None
        
        baseline_report = self.load_quality_report(baseline_reports[0])
        if not baseline_report:
            return None
        
        # 現在の品質レポート検索
        current_reports = self.find_quality_reports(current_dir)
        if not current_reports:
            print(f"❌ 現在の品質レポートが見つかりません: {current_dir}")
            return None
        
        current_report = self.load_quality_report(current_reports[0])
        if not current_report:
            return None
        
        # メトリクス抽出
        baseline_metrics = self.extract_metrics(baseline_report)
        current_metrics = self.extract_metrics(current_report)
        
        # 比較分析
        comparison_results = {}
        
        for metric_name in self.comparison_metrics:
            if metric_name in baseline_metrics and metric_name in current_metrics:
                baseline_value = baseline_metrics[metric_name]
                current_value = current_metrics[metric_name]
                
                improvement = self.calculate_improvement(baseline_value, current_value)
                
                comparison_results[metric_name] = {
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'absolute_change': improvement['absolute_change'],
                    'percent_change': improvement['percent_change'],
                    'improvement_type': improvement['improvement_type']
                }
        
        # 総合改善評価
        improvement_count = sum(1 for result in comparison_results.values() 
                              if result['improvement_type'] == 'improvement')
        degradation_count = sum(1 for result in comparison_results.values() 
                               if result['improvement_type'] == 'degradation')
        
        overall_status = 'improvement' if improvement_count > degradation_count else \
                        'degradation' if degradation_count > improvement_count else 'mixed'
        
        # レポート作成
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'baseline_info': {
                'directory': baseline_dir,
                'report_file': baseline_reports[0],
                'dataset': baseline_report.get('dataset_name', 'unknown'),
                'total_images': baseline_report.get('total_images', 0)
            },
            'current_info': {
                'directory': current_dir,
                'report_file': current_reports[0],
                'dataset': current_report.get('dataset_name', 'unknown'),
                'total_images': current_report.get('total_images', 0)
            },
            'comparison_results': comparison_results,
            'summary': {
                'total_metrics': len(comparison_results),
                'improvements': improvement_count,
                'degradations': degradation_count,
                'no_changes': len(comparison_results) - improvement_count - degradation_count,
                'overall_status': overall_status
            }
        }
        
        return report
    
    def create_comparison_chart(self, comparison_report: Dict, output_path: str):
        """比較チャート作成"""
        results = comparison_report['comparison_results']
        
        if not results:
            print("⚠️  比較データがありません。チャートをスキップします。")
            return
        
        # データ準備
        metric_names = list(results.keys())
        baseline_values = [results[name]['baseline_value'] for name in metric_names]
        current_values = [results[name]['current_value'] for name in metric_names]
        percent_changes = [results[name]['percent_change'] for name in metric_names]
        
        # チャート作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 値比較バーチャート
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_values, width, 
                       label='ベースライン', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, current_values, width, 
                       label='現在', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('メトリクス')
        ax1.set_ylabel('値')
        ax1.set_title('ベースライン vs 現在の値比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace(' ', '\n') for name in metric_names], 
                           rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                    f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                    f'{height2:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 改善率チャート
        colors = ['green' if rate > 0 else 'red' if rate < 0 else 'gray' 
                 for rate in percent_changes]
        bars = ax2.bar(range(len(metric_names)), percent_changes, 
                      color=colors, alpha=0.7)
        
        ax2.set_xlabel('メトリクス')
        ax2.set_ylabel('改善率 (%)')
        ax2.set_title('改善率（正：改善、負：劣化）')
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in metric_names], 
                           rotation=45, ha='right', fontsize=8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 改善率を表示
        for bar, rate in zip(bars, percent_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 比較チャートを保存: {output_path}")
    
    def generate_summary_text(self, comparison_report: Dict) -> str:
        """サマリーテキスト生成"""
        summary = comparison_report['summary']
        results = comparison_report['comparison_results']
        
        text = f"""
📊 改善効果測定結果サマリー
============================

🔍 比較対象
- ベースライン: {comparison_report['baseline_info']['dataset']} ({comparison_report['baseline_info']['total_images']}枚)
- 現在: {comparison_report['current_info']['dataset']} ({comparison_report['current_info']['total_images']}枚)

📈 総合結果
- 総評価指標数: {summary['total_metrics']}
- 改善指標数: {summary['improvements']} ✅
- 劣化指標数: {summary['degradations']} ❌
- 変化なし: {summary['no_changes']} ➖
- 総合判定: {self._get_status_emoji(summary['overall_status'])} {summary['overall_status'].upper()}

🔍 主要指標の変化:
"""
        
        # 重要指標を優先表示
        priority_metrics = ['Largest-Character Accuracy', 'A/B評価率', 'SCI (Semantic Completeness Index)']
        
        for metric in priority_metrics:
            if metric in results:
                result = results[metric]
                emoji = self._get_improvement_emoji(result['improvement_type'])
                text += f"  {emoji} {metric}: {result['baseline_value']:.3f} → {result['current_value']:.3f} "
                text += f"({result['percent_change']:+.1f}%)\n"
        
        # その他の指標
        other_metrics = [m for m in results.keys() if m not in priority_metrics]
        if other_metrics:
            text += "\n📋 その他の指標:\n"
            for metric in other_metrics:
                result = results[metric]
                emoji = self._get_improvement_emoji(result['improvement_type'])
                text += f"  {emoji} {metric}: {result['baseline_value']:.3f} → {result['current_value']:.3f} "
                text += f"({result['percent_change']:+.1f}%)\n"
        
        return text
    
    def _get_status_emoji(self, status: str) -> str:
        """ステータス絵文字取得"""
        emoji_map = {
            'improvement': '🚀',
            'degradation': '⚠️',
            'mixed': '📊'
        }
        return emoji_map.get(status, '❓')
    
    def _get_improvement_emoji(self, improvement_type: str) -> str:
        """改善タイプ絵文字取得"""
        emoji_map = {
            'improvement': '✅',
            'degradation': '❌',
            'no_change': '➖'
        }
        return emoji_map.get(improvement_type, '❓')

def main():
    parser = argparse.ArgumentParser(description='改善効果測定システム')
    parser.add_argument('--baseline', required=True, 
                       help='ベースライン結果ディレクトリ')
    parser.add_argument('--current', required=True,
                       help='現在の結果ディレクトリ')
    parser.add_argument('--output', required=True,
                       help='改善レポート出力ファイル (JSON)')
    parser.add_argument('--chart', 
                       help='比較チャート出力パス (PNG)')
    parser.add_argument('--summary', 
                       help='サマリーテキスト出力パス')
    
    args = parser.parse_args()
    
    # 改善効果測定実行
    system = ImprovementComparisonSystem()
    
    print("🔍 改善効果測定開始...")
    comparison_report = system.generate_comparison_report(args.baseline, args.current)
    
    if not comparison_report:
        print("❌ 改善効果測定に失敗しました")
        return 1
    
    # 結果保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)
    print(f"📊 改善レポートを保存: {args.output}")
    
    # チャート生成
    if args.chart:
        system.create_comparison_chart(comparison_report, args.chart)
    
    # サマリーテキスト生成
    summary_text = system.generate_summary_text(comparison_report)
    print(summary_text)
    
    if args.summary:
        with open(args.summary, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"📋 サマリーテキストを保存: {args.summary}")
    
    print("✅ 改善効果測定完了")
    return 0

if __name__ == "__main__":
    exit(main())