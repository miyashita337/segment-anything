#!/usr/bin/env python3
"""
Phase A完全実装テスト：GPT-4O推奨ボックス拡張 + fullbody基準
全身キャラクター抽出の決定版テスト

目的: 顔検出→ボックス拡張→SAM全身抽出
GPT-4O推奨: 水平2.75倍 × 垂直4.0倍の拡張適用
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Character extraction import
from features.extraction.commands.extract_character import CharacterExtractor


def test_fullbody_with_expansion():
    """GPT-4O推奨完全実装での全身抽出テスト"""

    print("🎌 Phase A完全実装: ボックス拡張 + 全身抽出テスト")
    print("📋 GPT-4O推奨機能: 顔検出→2.75倍×4.0倍拡張→SAM全身抽出")
    print("=" * 70)

    # テスト画像: v0.3.5失敗画像5枚
    test_images = [
        "kaname09_001.jpg",
        "kaname09_006.jpg",
        "kaname09_013.jpg",
        "kaname09_017.jpg",
        "kaname09_022.jpg",
    ]

    # 標準パス構成
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kaname09")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_gpt4o_fullbody")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 入力ディレクトリ: {input_dir}")
    print(f"📂 出力ディレクトリ: {output_dir}")
    print(f"📋 テスト画像数: {len(test_images)}枚")

    for i, image in enumerate(test_images, 1):
        print(f"  {i}. {image}")

    # CharacterExtractor初期化
    try:
        extractor = CharacterExtractor()
        print("✅ CharacterExtractor初期化完了")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return

    # Phase A完全実装テスト結果記録
    test_results = {
        "test_info": {
            "phase": "Phase A完全実装 - GPT-4O推奨ボックス拡張",
            "approach": "アニメYOLO + ボックス拡張 + fullbody_priority_enhanced",
            "model": "yolov8x6_animeface.pt",
            "expansion_strategy": "balanced (2.75x水平 × 4.0x垂直)",
            "selection_criteria": "fullbody_priority_enhanced",
            "gpt4o_feature": "Box Expansion (水平2.75倍 × 垂直4.0倍)",
            "total_images": len(test_images),
            "timestamp": datetime.now().isoformat(),
            "expected_improvement": "顔のみ→全身キャラクター抽出",
        },
        "results": [],
    }

    success_count = 0
    total_time = 0

    print(f"\n🎯 GPT-4O推奨完全実装テスト開始")
    print("📋 実装機能:")
    print("  - アニメYOLO: yolov8x6_animeface.pt")
    print("  - ボックス拡張: 水平2.75倍 × 垂直4.0倍")
    print("  - 選択基準: fullbody_priority_enhanced")
    print("  - 境界制限: 画像境界内に自動調整")
    print("=" * 50)

    for i, image_name in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] 処理中: {image_name}")

        input_path = input_dir / image_name
        output_path = output_dir / image_name

        start_time = time.time()

        try:
            # Phase A完全実装: ボックス拡張 + 全身基準での抽出実行
            result = extractor.extract(
                str(input_path),
                str(output_path),
                save_mask=True,
                save_transparent=True,
                verbose=True,
                high_quality=True,
                min_yolo_score=0.01,
                multi_character_criteria="fullbody_priority_enhanced",  # 全身基準
                use_box_expansion=True,  # 🎯 GPT-4O推奨ボックス拡張
                expansion_strategy="balanced",  # 水平2.75倍×垂直4.0倍
            )

            processing_time = time.time() - start_time
            total_time += processing_time

            if result.get("success", False):
                success_count += 1
                print(f"  ✅ 成功 (処理時間: {processing_time:.1f}秒)")
                if result.get("quality_score"):
                    print(f"     品質スコア: {result.get('quality_score'):.3f}")
                if result.get("expansion_applied"):
                    print(f"     🔍 ボックス拡張: 適用済み")
                status = "success"
            else:
                print(f"  ❌ 失敗 (処理時間: {processing_time:.1f}秒)")
                print(f"     エラー: {result.get('error', 'Unknown error')}")
                status = "failed"

            # 結果記録
            test_results["results"].append(
                {
                    "filename": image_name,
                    "success": result.get("success", False),
                    "processing_time": processing_time,
                    "error": result.get("error"),
                    "quality_score": result.get("quality_score"),
                    "expansion_applied": result.get("expansion_applied", False),
                    "box_expansion_details": result.get("box_expansion_details"),
                    "status": status,
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            total_time += processing_time

            print(f"  ❌ 例外エラー (処理時間: {processing_time:.1f}秒)")
            print(f"     例外: {str(e)}")

            test_results["results"].append(
                {
                    "filename": image_name,
                    "success": False,
                    "processing_time": processing_time,
                    "error": str(e),
                    "expansion_applied": False,
                    "status": "exception",
                }
            )

    # 統計計算
    success_rate = (success_count / len(test_images)) * 100
    avg_time = total_time / len(test_images)

    test_results["test_info"].update(
        {
            "success_count": success_count,
            "success_rate": success_rate,
            "total_time": total_time,
            "average_time": avg_time,
        }
    )

    # 結果サマリー表示
    print(f"\n📈 Phase A完全実装テスト結果")
    print("=" * 70)
    print(f"実装: GPT-4O推奨ボックス拡張 + 全身基準")
    print(f"機能: 顔検出→2.75倍×4.0倍拡張→SAM全身抽出")
    print(f"成功: {success_count}枚")
    print(f"失敗: {len(test_images) - success_count}枚")
    print(f"成功率: {success_rate:.1f}%")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"平均処理時間: {avg_time:.1f}秒/枚")

    # 前回（顔のみ＋ボックス拡張なし）との比較
    prev_success_rate = 100.0  # 顔のみでの成功率
    prev_avg_time = 25.1  # 前回の平均処理時間

    print(f"\n📊 前回（顔のみ、拡張なし）との比較:")
    print(
        f"  成功率: {prev_success_rate:.1f}% → {success_rate:.1f}% ({success_rate-prev_success_rate:+.1f}%)"
    )
    print(
        f"  処理時間: {prev_avg_time:.1f}秒 → {avg_time:.1f}秒 ({(avg_time/prev_avg_time-1)*100:+.1f}%)"
    )
    print(f"  抽出範囲: 顔のみ → 全身キャラクター（期待）")

    # GPT-4O予測との比較
    print(f"\n🤖 GPT-4O予測との比較:")
    print(f"  予測成功率: 80-90%")
    print(f"  実際成功率: {success_rate:.1f}%")
    print(f"  予測効果: 顔のみ→全身抽出")

    if success_rate >= 80:
        print(f"  → ✅ GPT-4O予測範囲内の優秀な結果")
        print(f"  → 🎯 全身抽出効果の検証が必要")
    elif success_rate >= 60:
        print(f"  → 📈 予測よりやや低いが改善傾向")
    else:
        print(f"  → 🔍 追加調整が必要")

    # ボックス拡張効果の統計
    expansion_applied_count = sum(
        1 for r in test_results["results"] if r.get("expansion_applied", False)
    )
    print(f"\n🔍 ボックス拡張統計:")
    print(f"  拡張適用: {expansion_applied_count}/{len(test_images)}枚")
    print(f"  拡張率: {(expansion_applied_count/len(test_images))*100:.1f}%")

    # 詳細結果
    print(f"\n📋 詳細結果:")
    for i, result in enumerate(test_results["results"], 1):
        status_emoji = "✅" if result["success"] else "❌"
        expansion_emoji = "🔍" if result.get("expansion_applied", False) else "🔸"
        print(
            f"  {i}. {result['filename']}: {status_emoji} {result['status']} "
            f"{expansion_emoji} ({result['processing_time']:.1f}秒)"
        )

    # JSON保存
    results_file = output_dir / "gpt4o_fullbody_expansion_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 結果保存: {results_file}")

    # Phase A完全実装評価
    print(f"\n🎯 Phase A完全実装評価:")
    print(f"  ✅ GPT-4O推奨ボックス拡張機能: 実装完了")
    print(f"  ✅ アニメYOLO + 全身基準: 統合完了")
    print(f"  📊 成功率: {success_rate:.1f}%")

    if success_rate >= 80:
        print(f"  🎉 優秀！GPT-4O戦略の有効性確認")
        print(f"     → 全身抽出の品質評価が次ステップ")
    elif success_rate >= 60:
        print(f"  📈 良好！さらなる改善余地あり")
        print(f"     → パラメータ微調整で向上可能")
    else:
        print(f"  🔍 要改善。Phase B検討が必要")
        print(f"     → negative prompt等の追加対策")

    print(f"\n📁 出力ファイル確認:")
    print(f"   ls {output_dir}/*.jpg")
    print(f"\n🔍 視覚確認:")
    print(f"   顔のみ vs 全身抽出の比較が重要")

    return test_results


if __name__ == "__main__":
    import torch

    test_fullbody_with_expansion()
