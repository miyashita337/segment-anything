#!/usr/bin/env python3
"""
QI-006: 複数キャラクター検出システム - 全データ実行スクリプト
kana08データセット全26枚での複数キャラ検出・結果出力
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.utils.multiple_character_detector import (
    MultipleCharacterDetector,
    detect_multiple_characters_from_image,
)
from features.extraction.models.yolo_wrapper import YOLOModelWrapper


def main():
    """QI-006 複数キャラクター検出実行"""
    print("🚀 QI-006: 複数キャラクター検出システム - 全データ実行")
    print("=" * 60)

    # パス設定（修正: 抽出後画像での複数キャラ検出）
    input_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/extraction")
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
    qi006_workspace = workspace_base / "QI-006"

    extraction_dir = qi006_workspace / "extraction"
    quality_dir = qi006_workspace / "quality"
    tests_dir = qi006_workspace / "tests"

    print(f"📁 入力: {input_dir}")
    print(f"📁 出力: {qi006_workspace}")

    # 入力ディレクトリ存在確認
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False

    # 抽出後画像ファイル収集（修正: 抽出済み画像のパターンマッチ）
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []

    # 抽出後画像の拡張パターン検索
    extraction_patterns = [
        "*extracted*",  # 基本抽出パターン
        "*cropped*",  # クロップ済み
        "*segment*",  # セグメント済み
        "*adaptive*",  # アダプティブ処理済み
    ]

    for pattern in extraction_patterns:
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"{pattern}{ext}"))
            image_files.extend(input_dir.glob(f"{pattern}{ext.upper()}"))

    # パターンマッチしない場合は全ての画像ファイルを対象
    if not image_files:
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # 可視化ファイル除外（前回のテスト結果）
    image_files = [f for f in image_files if "_multi_char_detection" not in f.name]
    image_files = sorted(image_files)

    print(f"🖼️ 対象画像: {len(image_files)}枚（抽出後画像）")

    if len(image_files) == 0:
        print(f"❌ 抽出後画像が見つかりません。抽出パイプライン実行後に再実行してください。")
        return False

    print(f"📋 検出対象: 抽出後画像での複数キャラ残存問題")

    # YOLO Wrapper初期化
    yolo_wrapper = YOLOModelWrapper()
    if not yolo_wrapper.load_model():
        print("❌ YOLO model loading failed")
        return False

    try:
        # 実行統計
        stats = {
            "total_images": len(image_files),
            "successful_detections": 0,
            "detection_errors": 0,
            "single_character": 0,
            "multiple_character": 0,
            "high_penalty": 0,  # ペナルティ > 0.7
            "medium_penalty": 0,  # 0.3 < ペナルティ <= 0.7
            "low_penalty": 0,  # ペナルティ <= 0.3
            "detection_types": {},
            "character_counts": {},
            "penalty_scores": [],
            "processing_times": [],
            "start_time": time.time(),
        }

        results = []

        print(f"\n🔍 複数キャラクター検出開始: {len(image_files)}枚")

        for i, image_path in enumerate(image_files):
            try:
                print(f"\n📸 処理 {i+1}/{len(image_files)}: {image_path.name}")

                # 処理時間計測
                img_start_time = time.time()

                # 複数キャラクター検出実行（可視化付き）
                result = detect_multiple_characters_from_image(
                    image_path, yolo_wrapper, save_visualization=True
                )

                processing_time = time.time() - img_start_time

                # 統計更新
                stats["successful_detections"] += 1
                stats["processing_times"].append(processing_time)
                stats["penalty_scores"].append(result.penalty_score)

                # 検出タイプ統計
                detection_type = result.detection_type.value
                stats["detection_types"][detection_type] = (
                    stats["detection_types"].get(detection_type, 0) + 1
                )

                # キャラクター数統計
                char_count = result.character_count
                stats["character_counts"][char_count] = (
                    stats["character_counts"].get(char_count, 0) + 1
                )

                # ペナルティレベル分類
                if result.is_multiple:
                    stats["multiple_character"] += 1
                    if result.penalty_score > 0.7:
                        stats["high_penalty"] += 1
                    elif result.penalty_score > 0.3:
                        stats["medium_penalty"] += 1
                    else:
                        stats["low_penalty"] += 1
                else:
                    stats["single_character"] += 1
                    stats["low_penalty"] += 1

                # 詳細結果記録
                result_data = {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "character_count": char_count,
                    "is_multiple": result.is_multiple,
                    "detection_type": detection_type,
                    "penalty_score": result.penalty_score,
                    "confidence_score": result.confidence_score,
                    "primary_character_index": result.primary_character_index,
                    "improvement_suggestions": result.improvement_suggestions,
                    "processing_time": processing_time,
                    "technical_details": result.technical_details,
                }
                results.append(result_data)

                # 結果表示
                print(f"   ✅ 検出完了: {char_count}体 (タイプ: {detection_type})")
                print(f"   📊 ペナルティ: {result.penalty_score:.3f}")
                print(f"   ⏱️ 処理時間: {processing_time:.2f}s")

                if result.is_multiple:
                    print(f"   🎯 メインキャラ: #{result.primary_character_index + 1}")
                    print(f"   💡 改善提案: {len(result.improvement_suggestions)}件")

                # 可視化ファイルをextractionディレクトリにコピー
                vis_file = image_path.parent / f"{image_path.stem}_multi_char_detection.jpg"
                if vis_file.exists():
                    dest_vis = extraction_dir / vis_file.name
                    shutil.copy2(vis_file, dest_vis)
                    print(f"   📊 可視化保存: {dest_vis.name}")

            except Exception as e:
                stats["detection_errors"] += 1
                print(f"   ❌ エラー: {str(e)}")

                results.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "error": str(e),
                        "processing_time": 0,
                    }
                )

        # 統計計算
        total_time = time.time() - stats["start_time"]
        stats["total_processing_time"] = total_time

        if stats["penalty_scores"]:
            stats["average_penalty"] = sum(stats["penalty_scores"]) / len(stats["penalty_scores"])
            stats["max_penalty"] = max(stats["penalty_scores"])
            stats["min_penalty"] = min(stats["penalty_scores"])

        if stats["processing_times"]:
            stats["average_processing_time"] = sum(stats["processing_times"]) / len(
                stats["processing_times"]
            )

        stats["success_rate"] = stats["successful_detections"] / stats["total_images"] * 100
        stats["multiple_character_rate"] = stats["multiple_character"] / stats["total_images"] * 100

        # 結果保存
        print(f"\n💾 結果保存中...")

        # 詳細結果JSON保存
        results_file = quality_dir / "qi006_detection_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tracker_id": "QI-006",
                    "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "statistics": stats,
                    "detailed_results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"✅ 詳細結果保存: {results_file}")

        # サマリーレポート生成
        summary_file = quality_dir / "qi006_summary_report.md"
        generate_summary_report(stats, results, summary_file)
        print(f"📄 サマリー保存: {summary_file}")

        # 統計表示
        print(f"\n📊 QI-006実行完了 - 複数キャラクター検出システム")
        print(
            f"   成功率: {stats['success_rate']:.1f}% ({stats['successful_detections']}/{stats['total_images']})"
        )
        print(
            f"   複数キャラ率: {stats['multiple_character_rate']:.1f}% ({stats['multiple_character']}/{stats['total_images']})"
        )
        print(f"   平均ペナルティ: {stats.get('average_penalty', 0):.3f}")
        print(f"   平均処理時間: {stats.get('average_processing_time', 0):.2f}s/枚")
        print(f"   総処理時間: {total_time:.1f}s")

        print(f"\n🎯 ペナルティレベル分布:")
        print(f"   高ペナルティ(>0.7): {stats['high_penalty']}枚")
        print(f"   中ペナルティ(0.3-0.7): {stats['medium_penalty']}枚")
        print(f"   低ペナルティ(≤0.3): {stats['low_penalty']}枚")

        print(f"\n📋 検出タイプ分布:")
        for det_type, count in stats["detection_types"].items():
            print(f"   {det_type}: {count}枚")

        return True

    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # リソース解放
        yolo_wrapper.unload_model()
        print("🧹 リソース解放完了")


def generate_summary_report(stats, results, output_file):
    """サマリーレポート生成"""
    lines = [
        "# QI-006: 複数キャラクター検出システム - 実行サマリー",
        "",
        f"**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**対象データセット**: kana08 ({stats['total_images']}枚)",
        "",
        "## 📊 実行結果サマリー",
        "",
        f"- **総合成功率**: {stats['success_rate']:.1f}% ({stats['successful_detections']}/{stats['total_images']})",
        f"- **複数キャラクター率**: {stats['multiple_character_rate']:.1f}% ({stats['multiple_character']}/{stats['total_images']})",
        f"- **平均ペナルティスコア**: {stats.get('average_penalty', 0):.3f}",
        f"- **平均処理時間**: {stats.get('average_processing_time', 0):.2f}秒/枚",
        f"- **総処理時間**: {stats['total_processing_time']:.1f}秒",
        "",
        "## 🎯 ペナルティレベル分析",
        "",
        f"- **高ペナルティ (>0.7)**: {stats['high_penalty']}枚 - LoRA学習不適切",
        f"- **中ペナルティ (0.3-0.7)**: {stats['medium_penalty']}枚 - 注意が必要",
        f"- **低ペナルティ (≤0.3)**: {stats['low_penalty']}枚 - 使用推奨",
        "",
        "## 🏷️ 検出タイプ分布",
        "",
    ]

    for det_type, count in stats["detection_types"].items():
        percentage = count / stats["total_images"] * 100
        lines.append(f"- **{det_type}**: {count}枚 ({percentage:.1f}%)")

    lines.extend(
        [
            "",
            "## 👥 キャラクター数分布",
            "",
        ]
    )

    for char_count, count in stats["character_counts"].items():
        percentage = count / stats["total_images"] * 100
        lines.append(f"- **{char_count}体**: {count}枚 ({percentage:.1f}%)")

    lines.extend(
        [
            "",
            "## 🎉 システム効果",
            "",
            f"- **フィルタリング効果**: 複数キャラ画像{stats['multiple_character']}枚を自動識別",
            f"- **品質向上**: 高ペナルティ{stats['high_penalty']}枚をLoRA学習から除外推奨",
            f"- **処理効率**: {stats.get('average_processing_time', 0):.2f}秒/枚の高速処理",
            f"- **自動化**: 手動確認不要の自動品質判定システム",
            "",
            "---",
            f"*QI-006実行結果 - {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        ]
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
