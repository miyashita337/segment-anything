#!/usr/bin/env python3
"""
抽出失敗原因分析スクリプト
v0.3.5バッチ処理の失敗パターンを詳細分析

目的: 31件の失敗原因を特定し、具体的改善策を提示
"""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze_batch_failures():
    """バッチ処理失敗の詳細分析"""

    # バッチ結果ファイルの読み込み
    results_path = Path(
        "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_0_3_5/batch_results_v035.json"
    )

    if not results_path.exists():
        print(f"❌ バッチ結果ファイルが見つかりません: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("📊 抽出失敗原因分析レポート")
    print("=" * 60)

    # 基本統計
    total_images = data["total_images"]
    success_count = data["success_count"]
    error_count = data["error_count"]
    success_rate = data["success_rate"]

    print(f"\n【事実】基本統計")
    print(f"- 入力画像総数: {total_images}件")
    print(f"- 抽出成功: {success_count}件 ({success_rate:.1f}%)")
    print(f"- 抽出失敗: {error_count}件 ({100-success_rate:.1f}%)")
    print(f"- 平均処理時間: {data['average_time_per_image']:.1f}秒/枚")

    # 失敗画像の詳細分析
    failed_results = []
    success_results = []

    for result in data["results"]:
        if result["success"]:
            success_results.append(result)
        else:
            failed_results.append(result)

    print(f"\n【失敗分析】エラー詳細")
    print(f"失敗件数: {len(failed_results)}件")

    # エラーメッセージの分類
    error_messages = [result.get("error", "Unknown error") for result in failed_results]
    error_counter = Counter(error_messages)

    print(f"\n主要失敗パターン:")
    for error_msg, count in error_counter.most_common():
        percentage = (count / len(failed_results)) * 100
        print(f"  • {error_msg}: {count}件 ({percentage:.1f}%)")

    # 失敗した画像ファイル名
    failed_files = [result["filename"] for result in failed_results]

    print(f"\n失敗画像ファイル名:")
    for i, filename in enumerate(sorted(failed_files), 1):
        print(f"  {i:2d}. {filename}")

    # 成功画像の品質分析
    print(f"\n【成功画像分析】")
    quality_scores = [
        result.get("quality_score", 0)
        for result in success_results
        if result.get("quality_score") is not None
    ]

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)

        print(f"- 平均品質スコア: {avg_quality:.3f}")
        print(f"- 最低品質スコア: {min_quality:.3f}")
        print(f"- 最高品質スコア: {max_quality:.3f}")
    else:
        print("- 品質スコア情報なし")

    # 処理時間分析
    failed_times = [
        result.get("processing_time", 0)
        for result in failed_results
        if result.get("processing_time")
    ]
    success_times = [
        result.get("processing_time", 0)
        for result in success_results
        if result.get("processing_time")
    ]

    if failed_times and success_times:
        avg_failed_time = sum(failed_times) / len(failed_times)
        avg_success_time = sum(success_times) / len(success_times)

        print(f"\n【処理時間分析】")
        print(f"- 失敗画像平均処理時間: {avg_failed_time:.1f}秒")
        print(f"- 成功画像平均処理時間: {avg_success_time:.1f}秒")
        print(f"- 時間差: {avg_success_time - avg_failed_time:.1f}秒")

    # 失敗パターン分析
    print(f"\n【技術的原因分析】")

    # YOLOスコア関連のエラー
    yolo_errors = [err for err in error_counter.keys() if "YOLO" in err or "score" in err]
    if yolo_errors:
        yolo_count = sum(error_counter[err] for err in yolo_errors)
        print(f"- YOLO検出失敗: {yolo_count}件")
        print(f"  → キャラクター認識できない画像")
        print(f"  → 閾値調整 (現在0.01) の検討必要")

    # マスク関連のエラー
    mask_errors = [err for err in error_counter.keys() if "mask" in err.lower()]
    if mask_errors:
        mask_count = sum(error_counter[err] for err in mask_errors)
        print(f"- マスク生成失敗: {mask_count}件")
        print(f"  → SAMセグメンテーション問題")

    # その他のエラー
    other_errors = [
        err
        for err in error_counter.keys()
        if not any(keyword in err.lower() for keyword in ["yolo", "score", "mask"])
    ]
    if other_errors:
        other_count = sum(error_counter[err] for err in other_errors)
        print(f"- その他のエラー: {other_count}件")
        for err in other_errors:
            print(f"  → {err}: {error_counter[err]}件")

    # 改善提案
    print(f"\n【改善必要事項】")
    print(f"1. YOLO検出率向上:")
    print(f"   - 閾値を0.01から0.005に下げる")
    print(f"   - 前処理でコントラスト調整")
    print(f"   - より大きなYOLOモデル使用検討")

    print(f"2. 画像前処理強化:")
    print(f"   - 低コントラスト画像の補正")
    print(f"   - ノイズ除去処理追加")
    print(f"   - リサイズ処理の最適化")

    print(f"3. エラーハンドリング改善:")
    print(f"   - 段階的閾値降下の実装")
    print(f"   - 複数モデルでのリトライ")
    print(f"   - より詳細なエラーログ")

    print(f"\n【現実的予測】")
    print(f"- 改善後予想成功率: 55-65% (現在46.6%)")
    print(f"- 根本的課題: アニメ画像のキャラクター検出限界")
    print(f"- 完全解決は困難、段階的改善が必要")


if __name__ == "__main__":
    analyze_batch_failures()
