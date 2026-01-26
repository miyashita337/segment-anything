#!/usr/bin/env python3
"""
YOLO閾値0.005テストスクリプト
Phase A: 失敗画像5枚での小規模テスト

目的: 処理時間・品質・偽陽性の測定
対象: v0.3.5で失敗した画像から5枚選定
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


def select_test_images():
    """失敗画像から5枚を選定"""

    # v0.3.5バッチ結果から失敗画像を取得
    results_path = Path(
        "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_0_3_5/batch_results_v035.json"
    )

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    failed_files = []
    for result in data["results"]:
        if not result["success"]:
            failed_files.append(result["filename"])

    # 最初の1枚をテスト用に選定
    test_files = failed_files[:1]

    print(f"📋 テスト対象画像:")
    for i, filename in enumerate(test_files, 1):
        print(f"  {i}. {filename}")

    return test_files


def test_threshold_005():
    """閾値0.005でのテスト実行"""

    print("🚀 YOLO閾値0.005テスト開始")
    print("=" * 60)

    # テスト画像選定
    test_files = select_test_images()

    # 入力・出力パス設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kaname09")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/test_threshold_005")
    output_dir.mkdir(exist_ok=True)

    # キャラクター抽出システム初期化
    try:
        extractor = CharacterExtractor()
        print("✅ キャラクター抽出システム初期化完了")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return

    # テスト結果格納
    test_results = {
        "test_info": {
            "threshold": 0.001,
            "test_files": test_files,
            "total_files": len(test_files),
            "timestamp": datetime.now().isoformat(),
        },
        "results": [],
    }

    success_count = 0
    total_time = 0

    print(f"\n📊 テスト実行開始（閾値: 0.001）")

    for i, filename in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] 処理中: {filename}")

        input_path = input_dir / filename
        output_path = output_dir / filename

        start_time = time.time()

        try:
            # キャラクター抽出実行
            result = extractor.extract(
                str(input_path),
                str(output_path),
                save_mask=True,
                save_transparent=True,
                verbose=True,
                high_quality=True,
                min_yolo_score=0.001,  # 極端な閾値で緊急テスト
            )

            processing_time = time.time() - start_time
            total_time += processing_time

            if result.get("success", False):
                success_count += 1
                print(f"  ✅ 成功 (処理時間: {processing_time:.1f}秒)")
                status = "success"
            else:
                print(f"  ❌ 失敗 (処理時間: {processing_time:.1f}秒)")
                print(f"     エラー: {result.get('error', 'Unknown error')}")
                status = "failed"

            # 結果記録
            test_results["results"].append(
                {
                    "filename": filename,
                    "success": result.get("success", False),
                    "processing_time": processing_time,
                    "error": result.get("error"),
                    "quality_score": result.get("quality_score"),
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
                    "filename": filename,
                    "success": False,
                    "processing_time": processing_time,
                    "error": str(e),
                    "status": "exception",
                }
            )

    # 統計情報の計算
    success_rate = (success_count / len(test_files)) * 100
    avg_time = total_time / len(test_files)

    test_results["test_info"].update(
        {
            "success_count": success_count,
            "success_rate": success_rate,
            "total_time": total_time,
            "average_time": avg_time,
        }
    )

    # 結果表示
    print(f"\n📈 テスト結果サマリー")
    print(f"=" * 60)
    print(f"テスト画像数: {len(test_files)}枚")
    print(f"成功: {success_count}枚")
    print(f"失敗: {len(test_files) - success_count}枚")
    print(f"成功率: {success_rate:.1f}%")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"平均処理時間: {avg_time:.1f}秒/枚")

    # 比較データ（v0.3.5での平均処理時間）
    v035_avg_time = 17.3  # 秒
    time_increase = (avg_time / v035_avg_time - 1) * 100
    print(f"\nv0.3.5との比較:")
    print(f"  v0.3.5平均処理時間: {v035_avg_time:.1f}秒/枚")
    print(f"  今回平均処理時間: {avg_time:.1f}秒/枚")
    print(f"  処理時間増加率: {time_increase:.1f}%")

    # 結果をJSONファイルに保存
    results_file = output_dir / "test_results_threshold_005.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 結果保存: {results_file}")

    return test_results


if __name__ == "__main__":
    import torch

    test_threshold_005()
