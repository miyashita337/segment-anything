#!/usr/bin/env python3
"""
QI-006 vs P1-021 統計分析スクリプト
Welch's t-test & Cohen's d効果サイズ計算
"""
import sys
import json
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_cohens_d(group1, group2):
    """Cohen's d効果サイズ計算"""
    n1, n2 = len(group1), len(group2)
    
    # サンプルサイズ不足チェック
    if n1 < 2 or n2 < 2:
        return 0.0
    
    # 平均値
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # 標準偏差（不偏）
    std1 = np.std(group1, ddof=1) if n1 > 1 else 0.0
    std2 = np.std(group2, ddof=1) if n2 > 1 else 0.0
    
    # プールされた標準偏差（Welch's t-testとは異なるが効果サイズの標準計算）
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    # Cohen's d
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d

def perform_statistical_analysis(current_data, baseline_data):
    """統計分析実行"""
    
    # 品質スコア抽出
    current_scores = [r['quality_score'] for r in current_data['results']]
    baseline_scores = [r['quality_score'] for r in baseline_data['results']]
    
    # 基本統計
    current_mean = np.mean(current_scores)
    baseline_mean = np.mean(baseline_scores)
    current_std = np.std(current_scores, ddof=1)
    baseline_std = np.std(baseline_scores, ddof=1)
    
    # Welch's t-test（等分散を仮定しない）
    t_stat, p_value = stats.ttest_ind(current_scores, baseline_scores, equal_var=False)
    
    # Cohen's d効果サイズ
    effect_size = calculate_cohens_d(current_scores, baseline_scores)
    
    # 改善率計算
    improvement_rate = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0.0
    
    # 統計的有意性判定（α=0.05）
    is_significant = p_value < 0.05
    
    return {
        'current_mean': current_mean,
        'baseline_mean': baseline_mean,
        'current_std': current_std,
        'baseline_std': baseline_std,
        'current_sample_size': len(current_scores),
        'baseline_sample_size': len(baseline_scores),
        'p_value': p_value,
        'effect_size': effect_size,
        'improvement_rate': improvement_rate,
        'is_significant': is_significant,
        't_statistic': t_stat
    }

def main():
    """メイン処理"""
    
    print("🔍 QI-006 vs P1-021 統計分析開始")
    
    # データ読み込み
    with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/extraction_result.json', 'r') as f:
        qi006_data = json.load(f)
    
    with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-021/extraction_result.json', 'r') as f:
        p1021_data = json.load(f)
    
    print(f"📈 Current(QI-006): {qi006_data['total_images']}枚、平均品質スコア: {qi006_data['mean_quality_score']:.6f}")
    print(f"📈 Baseline(P1-021): {p1021_data['total_images']}枚、平均品質スコア: {p1021_data['mean_quality_score']:.6f}")
    
    # 統計分析実行
    stats_result = perform_statistical_analysis(qi006_data, p1021_data)
    
    print()
    print("🔍 統計分析結果:")
    print(f"   p値: {stats_result['p_value']:.6f}")
    print(f"   効果サイズ(Cohen's d): {stats_result['effect_size']:.6f}")
    print(f"   改善率: {stats_result['improvement_rate']:.2f}%")
    print(f"   統計的有意性: {'有意' if stats_result['is_significant'] else '非有意'} (α=0.05)")
    
    # 効果サイズ解釈
    effect_abs = abs(stats_result['effect_size'])
    if effect_abs > 0.8:
        effect_interpretation = "大効果"
    elif effect_abs > 0.5:
        effect_interpretation = "中効果"
    elif effect_abs > 0.2:
        effect_interpretation = "小効果"
    else:
        effect_interpretation = "効果なし"
    
    print(f"   効果サイズ解釈: {effect_interpretation} (|d| = {effect_abs:.3f})")
    
    # 詳細結果作成
    analysis_result = {
        'analysis_timestamp': datetime.now().isoformat(),
        'current_tracker': 'QI-006',
        'baseline_tracker': 'P1-021',
        'current_metrics': {
            'mean_quality_score': stats_result['current_mean'],
            'sample_size': stats_result['current_sample_size'],
            'std_quality_score': stats_result['current_std']
        },
        'baseline_metrics': {
            'mean_quality_score': stats_result['baseline_mean'],
            'sample_size': stats_result['baseline_sample_size'],
            'std_quality_score': stats_result['baseline_std']
        },
        'statistical_results': {
            'p_value': stats_result['p_value'],
            'effect_size': stats_result['effect_size'],
            'improvement_rate': stats_result['improvement_rate'],
            'is_significant': stats_result['is_significant'],
            'alpha_level': 0.05,
            't_statistic': stats_result['t_statistic'],
            'interpretation': f"統計的有意性: {'有意' if stats_result['is_significant'] else '非有意'} (p {'<' if stats_result['p_value'] < 0.05 else '>='} 0.05)、効果サイズ: {effect_interpretation} (|d| = {effect_abs:.3f})"
        }
    }
    
    # JSON保存
    output_file = '/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/statistical_analysis_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📄 分析結果保存: {output_file}")
    print("✅ QI-006 vs P1-021 統計分析完了")
    
    return analysis_result

if __name__ == "__main__":
    result = main()