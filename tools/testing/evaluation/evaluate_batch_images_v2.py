#!/usr/bin/env python3
"""
v0.3.5成功画像バッチ評価スクリプト v2
人間評価フォーマット（A-F判定）に統一したGPT-4O評価システム

27枚の成功画像を新しい評価基準で自動評価
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from features.evaluation.image_evaluation_mcp import ImageEvaluationMCP


def main():
    """メイン実行関数"""
    print("🚀 v0.3.5成功画像バッチ評価開始（新評価フォーマット）")

    # 設定
    results_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_0_3_5")
    output_file = Path("batch_evaluation_results_v2.json")

    # batch_results_v035.jsonから成功画像リストを取得
    batch_results_path = results_dir / "batch_results_v035.json"

    if not batch_results_path.exists():
        print(f"❌ バッチ結果ファイルが見つかりません: {batch_results_path}")
        return

    # 成功画像リストを抽出
    with open(batch_results_path, "r", encoding="utf-8") as f:
        batch_results = json.load(f)

    success_images = []
    for result in batch_results["results"]:
        if result["success"]:
            image_path = results_dir / result["filename"]
            if image_path.exists():
                success_images.append(str(image_path))

    print(f"📊 評価対象: {len(success_images)}枚の成功画像")
    print(f"🎯 評価フォーマット: A-F判定（人間評価準拠）")

    if not success_images:
        print("❌ 評価対象画像が見つかりません")
        return

    # 評価システム初期化
    evaluator = ImageEvaluationMCP()

    try:
        # バッチ評価実行
        summary = evaluator.batch_evaluate(success_images, str(output_file))

        # 追加分析（新フォーマット対応）
        print(f"\n📈 評価分析:")

        successful_evaluations = [
            r for r in summary["results"] if r["success"] and not r.get("parse_error")
        ]

        if successful_evaluations:
            # 評価グレード分布
            grade_distribution = {}
            issues_count = {}

            for result in successful_evaluations:
                grade = result.get("grade", "Unknown")
                grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

                # 問題分類の集計
                issues = result.get("issues", [])
                for issue in issues:
                    issues_count[issue] = issues_count.get(issue, 0) + 1

            print(f"\n📊 評価グレード分布:")
            for grade, count in sorted(grade_distribution.items()):
                percentage = (count / len(successful_evaluations)) * 100
                print(f"  {grade}評価: {count}枚 ({percentage:.1f}%)")

            if issues_count:
                print(f"\n⚠️ 問題分類頻度:")
                for issue, count in sorted(issues_count.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {issue}: {count}枚")

            # API使用統計
            api_usage = {}
            for result in successful_evaluations:
                api_used = result.get("api_used", "Unknown")
                api_usage[api_used] = api_usage.get(api_used, 0) + 1

            print(f"\n🤖 API使用統計:")
            for api, count in api_usage.items():
                print(f"  {api}: {count}枚")

        print(f"\n✅ 評価完了! 新フォーマット結果: {output_file}")

    except Exception as e:
        print(f"❌ 評価エラー: {str(e)}")
        return


if __name__ == "__main__":
    main()
