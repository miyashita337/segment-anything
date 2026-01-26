#!/usr/bin/env python3
"""
v0.3.5成功画像バッチ評価スクリプト
GPT-4O + Gemini フォールバック評価システム使用

27枚の成功画像を自動評価し、品質分析を行う
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
    print("🚀 v0.3.5成功画像バッチ評価開始")

    # 設定
    results_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_0_3_5")
    output_file = Path("batch_evaluation_results.json")

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

    if not success_images:
        print("❌ 評価対象画像が見つかりません")
        return

    # 評価システム初期化
    evaluator = ImageEvaluationMCP()

    try:
        # バッチ評価実行
        summary = evaluator.batch_evaluate(success_images, str(output_file))

        # 追加分析
        print(f"\n📈 品質分析:")

        successful_evaluations = [
            r for r in summary["results"] if r["success"] and not r.get("parse_error")
        ]

        if successful_evaluations:
            # 平均スコア計算
            completeness_scores = [r.get("completeness", 0) for r in successful_evaluations]
            boundary_scores = [r.get("boundary_quality", 0) for r in successful_evaluations]
            background_scores = [r.get("background_removal", 0) for r in successful_evaluations]
            overall_scores = [r.get("overall_quality", 0) for r in successful_evaluations]

            if completeness_scores:
                print(f"  完全性平均: {sum(completeness_scores) / len(completeness_scores):.2f}")
            if boundary_scores:
                print(f"  境界品質平均: {sum(boundary_scores) / len(boundary_scores):.2f}")
            if background_scores:
                print(f"  背景除去平均: {sum(background_scores) / len(background_scores):.2f}")
            if overall_scores:
                print(f"  総合評価平均: {sum(overall_scores) / len(overall_scores):.2f}")

            # API使用統計
            api_usage = {}
            for result in successful_evaluations:
                api_used = result.get("api_used", "Unknown")
                api_usage[api_used] = api_usage.get(api_used, 0) + 1

            print(f"\n📊 API使用統計:")
            for api, count in api_usage.items():
                print(f"  {api}: {count}枚")

        print(f"\n✅ 評価完了! 結果: {output_file}")

    except Exception as e:
        print(f"❌ 評価エラー: {str(e)}")
        return


if __name__ == "__main__":
    main()
