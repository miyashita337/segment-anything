#!/usr/bin/env python3
"""
P1-006 階層的品質評価のデモスクリプト
複数レベルでの品質評価と統合判定
"""

import numpy as np
import cv2

import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.evaluation.hierarchical_quality import (
    HierarchicalQualityEvaluator,
    QualityLevel,
    QualityMetric,
    evaluate_hierarchical_quality,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """サンプルデータ作成"""
    samples = []
    
    # 高品質サンプル
    print("📸 高品質サンプル作成中...")
    good_image = np.random.randint(120, 180, (200, 200, 3), dtype=np.uint8)
    good_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(good_mask, (100, 100), 60, 1, -1)
    samples.append({
        'id': 'high_quality_sample',
        'image': good_image,
        'mask': good_mask,
        'bbox': (40, 40, 120, 120),
        'expected_grade': 'A-B'
    })
    
    # 中品質サンプル
    print("📸 中品質サンプル作成中...")
    medium_image = np.random.randint(80, 150, (200, 200, 3), dtype=np.uint8)
    medium_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(medium_mask, (100, 100), (40, 30), 0, 0, 360, 1, -1)
    samples.append({
        'id': 'medium_quality_sample',
        'image': medium_image,
        'mask': medium_mask,
        'bbox': (60, 70, 80, 60),
        'expected_grade': 'B-C'
    })
    
    # 低品質サンプル
    print("📸 低品質サンプル作成中...")
    poor_image = np.random.randint(20, 80, (200, 200, 3), dtype=np.uint8)
    poor_mask = np.zeros((200, 200), dtype=np.uint8)
    poor_mask[180:190, 180:190] = 1  # 小さな角のマスク
    samples.append({
        'id': 'poor_quality_sample',
        'image': poor_image,
        'mask': poor_mask,
        'bbox': (175, 175, 15, 15),
        'expected_grade': 'D-F'
    })
    
    # 不完全サンプル
    print("📸 不完全サンプル作成中...")
    incomplete_image = np.random.randint(60, 120, (200, 200, 3), dtype=np.uint8)
    incomplete_mask = np.zeros((200, 200), dtype=np.uint8)
    # 複数の分離されたピクセル
    incomplete_mask[50:55, 50:55] = 1
    incomplete_mask[150:155, 150:155] = 1
    incomplete_mask[100:105, 100:105] = 1
    samples.append({
        'id': 'incomplete_sample',
        'image': incomplete_image,
        'mask': incomplete_mask,
        'bbox': (45, 45, 110, 110),
        'expected_grade': 'E-F'
    })
    
    return samples


def demo_single_evaluation():
    """単一評価のデモ"""
    print("\n=== 単一評価のデモ ===")
    
    # サンプル画像作成
    image = np.random.randint(100, 200, (150, 150, 3), dtype=np.uint8)
    mask = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(mask, (75, 75), 40, 1, -1)
    
    # 評価実行
    result = evaluate_hierarchical_quality(
        image, mask, 
        item_id="demo_single",
        bbox=(35, 35, 80, 80)
    )
    
    print(f"🎯 アイテムID: {result.item_id}")
    print(f"📊 総合スコア: {result.overall_score:.3f}")
    print(f"🏆 総合グレード: {result.overall_grade}")
    
    # 詳細スコア表示
    print("\n📋 詳細スコア:")
    for score in result.scores:
        print(f"  {score.level.value:>8} {score.metric.value:>15}: {score.score:.3f} (信頼度: {score.confidence:.2f})")
    
    # レベル別サマリー
    print("\n📈 レベル別サマリー:")
    for level, summary in result.level_summaries.items():
        print(f"  {level:>8}: 平均スコア {summary['average_score']:.3f}, メトリクス数 {summary['metric_count']}")
    
    # 改善提案
    if result.recommendations:
        print("\n💡 改善提案:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    return result


def demo_batch_evaluation():
    """バッチ評価のデモ"""
    print("\n=== バッチ評価のデモ ===")
    
    evaluator = HierarchicalQualityEvaluator()
    samples = create_sample_data()
    
    # バッチ評価実行
    results = evaluator.evaluate_batch(samples)
    
    # 結果表示
    print(f"\n📊 バッチ評価結果 ({len(results)}件):")
    print("-" * 80)
    print(f"{'ID':>20} {'スコア':>8} {'グレード':>8} {'期待値':>10} {'推奨事項数':>10}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        expected = samples[i]['expected_grade']
        rec_count = len(result.recommendations)
        print(f"{result.item_id:>20} {result.overall_score:>8.3f} {result.overall_grade:>8} {expected:>10} {rec_count:>10}")
    
    return evaluator, results


def demo_dataset_analysis(evaluator):
    """データセット分析のデモ"""
    print("\n=== データセット分析のデモ ===")
    
    summary = evaluator.get_dataset_summary()
    
    if not summary:
        print("📭 評価履歴がありません")
        return
    
    print(f"📈 データセット統計:")
    print(f"  総評価数: {summary['total_evaluations']}")
    print(f"  平均スコア: {summary['average_score']:.3f}")
    print(f"  スコア標準偏差: {summary['score_std']:.3f}")
    print(f"  成功率 (A+B): {summary['success_rate']}")
    
    print(f"\n🏆 グレード分布:")
    for grade, count in summary['grade_distribution'].items():
        percentage = count / summary['total_evaluations'] * 100
        bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        print(f"  {grade}: {count:>2}件 ({percentage:>5.1f}%) {bar}")


def demo_level_detail_analysis(results):
    """レベル詳細分析のデモ"""
    print("\n=== レベル詳細分析のデモ ===")
    
    # レベル別統計計算
    level_stats = {}
    for level in QualityLevel:
        level_scores = []
        for result in results:
            for score in result.scores:
                if score.level == level:
                    level_scores.append(score.score)
        
        if level_scores:
            level_stats[level.value] = {
                'count': len(level_scores),
                'average': np.mean(level_scores),
                'std': np.std(level_scores),
                'min': np.min(level_scores),
                'max': np.max(level_scores)
            }
    
    # 結果表示
    print("📊 レベル別詳細統計:")
    print("-" * 70)
    print(f"{'レベル':>10} {'件数':>6} {'平均':>8} {'標準偏差':>8} {'最小':>8} {'最大':>8}")
    print("-" * 70)
    
    for level_name, stats in level_stats.items():
        print(f"{level_name:>10} {stats['count']:>6} {stats['average']:>8.3f} {stats['std']:>8.3f} {stats['min']:>8.3f} {stats['max']:>8.3f}")


def demo_metric_correlation(results):
    """メトリクス相関分析のデモ"""
    print("\n=== メトリクス相関分析のデモ ===")
    
    # メトリクス別スコア収集
    metric_scores = {}
    for metric in QualityMetric:
        metric_scores[metric.value] = []
    
    for result in results:
        for score in result.scores:
            metric_scores[score.metric.value].append(score.score)
    
    # 相関計算（簡易版）
    print("📈 メトリクス別統計:")
    print("-" * 50)
    print(f"{'メトリクス':>15} {'件数':>6} {'平均':>8} {'標準偏差':>8}")
    print("-" * 50)
    
    for metric_name, scores in metric_scores.items():
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"{metric_name:>15} {len(scores):>6} {avg:>8.3f} {std:>8.3f}")


def demo_recommendation_analysis(results):
    """推奨事項分析のデモ"""
    print("\n=== 推奨事項分析のデモ ===")
    
    # 推奨事項の集計
    recommendation_freq = {}
    for result in results:
        for rec in result.recommendations:
            recommendation_freq[rec] = recommendation_freq.get(rec, 0) + 1
    
    if recommendation_freq:
        print("💡 推奨事項頻度分析:")
        sorted_recs = sorted(recommendation_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (rec, freq) in enumerate(sorted_recs[:5], 1):  # トップ5のみ表示
            print(f"  {i}. ({freq}回) {rec}")
    else:
        print("💡 推奨事項はありません（全て高品質）")


def main():
    """メイン実行関数"""
    print("🔍 P1-006 階層的品質評価デモ")
    
    # 各種デモを実行
    demo_single_evaluation()
    evaluator, results = demo_batch_evaluation()
    demo_dataset_analysis(evaluator)
    demo_level_detail_analysis(results)
    demo_metric_correlation(results)
    demo_recommendation_analysis(results)
    
    # 結果保存デモ
    output_path = Path("demo_output/hierarchical_quality")
    if output_path.exists():
        evaluator.save_results(output_path)
        print(f"\n💾 デモ結果を保存しました: {output_path}")
    
    print("\n✅ デモ完了！")


if __name__ == "__main__":
    main()