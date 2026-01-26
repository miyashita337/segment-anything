#!/usr/bin/env python3
"""
P1-007 評価説明可能性のデモスクリプト
品質評価の根拠と改善提案の詳細説明
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

from features.evaluation.explainable_quality import (
    ExplainableQualityEvaluator,
    ExplanationResult,
    QualityFactor,
    explain_quality_evaluation,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """サンプルデータ作成"""
    samples = []

    # 高品質サンプル
    print("📸 高品質サンプル作成中...")
    good_image = np.random.randint(120, 200, (300, 300, 3), dtype=np.uint8)
    good_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(good_mask, (150, 150), 80, 1, -1)

    good_quality_scores = {
        "coverage": 0.92,
        "edge_accuracy": 0.88,
        "clarity": 0.85,
        "size_relevance": 0.91,
        "position_relevance": 0.95,
    }

    samples.append(
        {
            "id": "high_quality_anime_character",
            "image": good_image,
            "mask": good_mask,
            "quality_scores": good_quality_scores,
            "bbox": (70, 70, 160, 160),
            "expected_explanation": "全体的に優秀な品質で、改善提案は少ない",
        }
    )

    # 中品質サンプル
    print("📸 中品質サンプル作成中...")
    medium_image = np.random.randint(80, 160, (300, 300, 3), dtype=np.uint8)
    medium_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.ellipse(medium_mask, (150, 150), (60, 40), 0, 0, 360, 1, -1)

    medium_quality_scores = {
        "coverage": 0.68,
        "edge_accuracy": 0.55,
        "clarity": 0.72,
        "size_relevance": 0.64,
        "position_relevance": 0.78,
    }

    samples.append(
        {
            "id": "medium_quality_partial_extraction",
            "image": medium_image,
            "mask": medium_mask,
            "quality_scores": medium_quality_scores,
            "bbox": (90, 110, 120, 80),
            "expected_explanation": "部分的な改善が必要、特にエッジ精度向上",
        }
    )

    # 低品質サンプル
    print("📸 低品質サンプル作成中...")
    poor_image = np.random.randint(20, 100, (300, 300, 3), dtype=np.uint8)
    poor_mask = np.zeros((300, 300), dtype=np.uint8)
    poor_mask[280:295, 280:295] = 1  # 小さな角のマスク

    poor_quality_scores = {
        "coverage": 0.15,
        "edge_accuracy": 0.22,
        "clarity": 0.38,
        "size_relevance": 0.08,
        "position_relevance": 0.12,
    }

    samples.append(
        {
            "id": "poor_quality_failed_extraction",
            "image": poor_image,
            "mask": poor_mask,
            "quality_scores": poor_quality_scores,
            "bbox": (275, 275, 20, 20),
            "expected_explanation": "大幅な改善が必要、ほぼ全要因で問題",
        }
    )

    # 特殊ケース（エッジのみ良好）
    print("📸 特殊ケースサンプル作成中...")
    special_image = np.random.randint(60, 140, (300, 300, 3), dtype=np.uint8)
    special_mask = np.zeros((300, 300), dtype=np.uint8)

    # 複雑な形状マスク（エッジは良いがカバレッジは悪い）
    points = np.array([[150, 50], [200, 100], [180, 150], [120, 150], [100, 100]], np.int32)
    cv2.fillPoly(special_mask, [points], 1)

    special_quality_scores = {
        "coverage": 0.35,
        "edge_accuracy": 0.89,  # 高いエッジ精度
        "clarity": 0.45,
        "size_relevance": 0.42,
        "position_relevance": 0.55,
    }

    samples.append(
        {
            "id": "special_case_good_edges_poor_coverage",
            "image": special_image,
            "mask": special_mask,
            "quality_scores": special_quality_scores,
            "bbox": (100, 50, 100, 100),
            "expected_explanation": "エッジ精度は優秀だがカバレッジに課題",
        }
    )

    return samples


def demo_single_explanation():
    """単一説明のデモ"""
    print("\n=== 単一説明のデモ ===")

    # サンプル画像作成
    image = np.random.randint(100, 180, (200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 1, -1)

    quality_scores = {
        "coverage": 0.75,
        "edge_accuracy": 0.62,
        "clarity": 0.68,
        "size_relevance": 0.83,
        "position_relevance": 0.71,
    }

    # 説明付き評価実行
    result = explain_quality_evaluation(
        image,
        mask,
        quality_scores,
        item_id="demo_single",
        output_dir=Path("demo_output/explainable_quality"),
    )

    print(f"🎯 アイテムID: {result.item_id}")
    print(f"📊 総合スコア: {result.overall_score:.3f}")
    print(f"🏆 総合グレード: {result.overall_grade}")
    print(f"🔍 評価信頼性: {result.confidence:.3f}")

    # 品質要因詳細
    print("\n📋 品質要因詳細:")
    for factor in result.factors:
        impact_level = "高" if factor.importance > 0.7 else "中" if factor.importance > 0.4 else "低"
        quality_level = "高" if factor.value > 0.7 else "中" if factor.value > 0.4 else "低"
        print(f"  {factor.name}:")
        print(f"    スコア: {factor.value:.3f} ({quality_level}品質)")
        print(f"    重要度: {factor.importance:.3f} ({impact_level}影響)")
        print(f"    説明: {factor.description}")
        if factor.improvement_suggestion:
            print(f"    改善案: {factor.improvement_suggestion}")
        print()

    # テキスト説明
    print("📝 詳細説明:")
    for key, explanation in result.explanations.items():
        print(f"\n【{key}】")
        print(explanation)

    # 改善提案
    if result.recommendations:
        print("\n💡 改善提案:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

    # ビジュアル出力確認
    if result.visual_paths:
        print(f"\n🖼️ ビジュアル説明:")
        for name, path in result.visual_paths.items():
            print(f"  {name}: {path}")

    return result


def demo_batch_explanation():
    """バッチ説明のデモ"""
    print("\n=== バッチ説明のデモ ===")

    evaluator = ExplainableQualityEvaluator(Path("demo_output/explainable_quality"))
    samples = create_sample_data()

    results = []

    # 各サンプルを評価
    for sample in samples:
        print(f"\n🔍 評価中: {sample['id']}")

        result = evaluator.evaluate_with_explanation(
            sample["id"],
            sample["image"],
            sample["mask"],
            sample["quality_scores"],
            sample.get("bbox"),
        )

        results.append(result)

        # 期待される説明との比較
        print(f"期待: {sample['expected_explanation']}")
        print(f"実際: {result.explanations.get('overall', '説明なし')[:100]}...")

    return evaluator, results


def demo_explanation_analysis(results):
    """説明分析のデモ"""
    print("\n=== 説明分析のデモ ===")

    # 結果サマリー
    print(f"📊 評価結果サマリー ({len(results)}件):")
    print("-" * 80)
    print(f"{'ID':>30} {'スコア':>8} {'グレード':>8} {'信頼性':>8} {'要因数':>8} {'推奨数':>8}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.item_id:>30} {result.overall_score:>8.3f} {result.overall_grade:>8} {result.confidence:>8.3f} {len(result.factors):>8} {len(result.recommendations):>8}"
        )

    # 品質要因統計
    print(f"\n📈 品質要因統計:")
    factor_stats = {}

    for result in results:
        for factor in result.factors:
            if factor.name not in factor_stats:
                factor_stats[factor.name] = []
            factor_stats[factor.name].append(factor.value)

    print("-" * 60)
    print(f"{'要因名':>20} {'件数':>8} {'平均':>8} {'最小':>8} {'最大':>8}")
    print("-" * 60)

    for name, values in factor_stats.items():
        avg_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"{name:>20} {len(values):>8} {avg_val:>8.3f} {min_val:>8.3f} {max_val:>8.3f}")

    # 信頼性分析
    print(f"\n🔍 信頼性分析:")
    confidences = [r.confidence for r in results]
    print(f"  平均信頼性: {np.mean(confidences):.3f}")
    print(f"  信頼性範囲: {np.min(confidences):.3f} - {np.max(confidences):.3f}")

    high_conf_count = sum(1 for c in confidences if c > 0.8)
    medium_conf_count = sum(1 for c in confidences if 0.5 <= c <= 0.8)
    low_conf_count = sum(1 for c in confidences if c < 0.5)

    print(f"  高信頼性 (>0.8): {high_conf_count}件")
    print(f"  中信頼性 (0.5-0.8): {medium_conf_count}件")
    print(f"  低信頼性 (<0.5): {low_conf_count}件")


def demo_improvement_suggestions(results):
    """改善提案分析のデモ"""
    print("\n=== 改善提案分析のデモ ===")

    # 全提案の収集
    all_recommendations = []
    for result in results:
        all_recommendations.extend(result.recommendations)

    if not all_recommendations:
        print("💡 改善提案はありません（全て高品質）")
        return

    print(f"📋 改善提案統計:")
    print(f"  総提案数: {len(all_recommendations)}")
    print(f"  平均提案数/アイテム: {len(all_recommendations)/len(results):.1f}")

    # 提案の頻度分析
    from collections import Counter

    recommendation_freq = Counter(all_recommendations)

    print(f"\n🔍 頻出改善提案 (トップ5):")
    for i, (rec, freq) in enumerate(recommendation_freq.most_common(5), 1):
        percentage = freq / len(all_recommendations) * 100
        print(f"  {i}. ({freq}回, {percentage:.1f}%) {rec}")

    # グレード別改善提案
    print(f"\n📊 グレード別改善提案:")
    grade_recommendations = {}
    for result in results:
        grade = result.overall_grade
        if grade not in grade_recommendations:
            grade_recommendations[grade] = []
        grade_recommendations[grade].extend(result.recommendations)

    for grade in sorted(grade_recommendations.keys()):
        count = len(grade_recommendations[grade])
        print(f"  グレード{grade}: {count}件の提案")


def demo_explanation_export(evaluator):
    """説明結果エクスポートのデモ"""
    print("\n=== 説明結果エクスポート ===")

    output_path = Path("demo_output/explainable_quality/results")
    evaluator.save_explanations(output_path)

    # 出力ファイル確認
    if output_path.exists():
        files = list(output_path.glob("*.json"))
        images = list(output_path.parent.glob("visuals/*.png"))

        print(f"📄 生成されたファイル:")
        print(f"  JSON説明ファイル: {len(files)}件")
        print(f"  ビジュアル説明: {len(images)}件")

        # サンプルファイル内容表示
        if files:
            sample_file = files[0]
            print(f"\n📋 サンプル説明ファイル ({sample_file.name}):")
            with open(sample_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  アイテムID: {data.get('item_id', 'N/A')}")
                print(f"  総合スコア: {data.get('overall_score', 0):.3f}")
                print(f"  要因数: {len(data.get('factors', []))}")
                print(f"  説明数: {len(data.get('explanations', {}))}")

        print(f"\n💾 全結果を保存しました: {output_path}")
    else:
        print("❌ エクスポートに失敗しました")


def main():
    """メイン実行関数"""
    print("🔍 P1-007 評価説明可能性デモ")
    print("=" * 60)

    try:
        # 各種デモを実行
        demo_single_explanation()
        evaluator, results = demo_batch_explanation()
        demo_explanation_analysis(results)
        demo_improvement_suggestions(results)
        demo_explanation_export(evaluator)

        print("\n✅ デモ完了！")
        print(f"📁 出力ディレクトリ: demo_output/explainable_quality/")

    except Exception as e:
        print(f"\n❌ デモ実行エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
