#!/usr/bin/env python3
"""
Phase 2機能の簡単テスト
失敗していた2画像に対してエフェクト線除去・マルチコマ分割を試行
"""

import argparse
import os
import sys

sys.path.append(".")


def test_phase2_on_failed_images(
    input_dir=None, output_dir=None, score_threshold=0.07, test_solid_fill=False
):
    """Phase 2機能をバッチ処理"""

    # モデル初期化
    print("Initializing models...")
    from features.common.hooks.start import start

    start()
    print("Model initialization completed\n")

    from features.extraction.commands.extract_character import extract_character_from_path
    from pathlib import Path

    # 画像リスト作成
    if input_dir:
        input_path = Path(input_dir)
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        failed_images = [
            {"path": str(img), "name": img.name, "description": f"Image {i+1}/{len(image_files)}"}
            for i, img in enumerate(image_files)
        ]
    else:
        # デフォルトのテスト画像
        failed_images = [
            {
                "path": "/mnt/c/AItools/lora/train/yado/org/kaname03/21_kaname03_0020.jpg",
                "name": "21_kaname03_0020.jpg",
                "description": "ダイナミックなポーズ + エフェクト線",
            },
            {
                "path": "/mnt/c/AItools/lora/train/yado/org/kaname03/16_kaname03_0015.jpg",
                "name": "16_kaname03_0015.jpg",
                "description": "マルチコマ構成",
            },
        ]

    # バッチ処理用設定（最適設定のみ使用）
    if input_dir and output_dir:
        test_configs = [
            {
                "name": "balanced",
                "params": {"auto_retry": True, "low_threshold": True, "manga_mode": True},
            }
        ]
    else:
        # デフォルトのテスト設定
        test_configs = [
            {"name": "Phase 1: 自動リトライ（参考）", "params": {"auto_retry": True}},
            {
                "name": "Phase 2: エフェクト線除去",
                "params": {"manga_mode": True, "effect_removal": True, "low_threshold": True},
            },
            {
                "name": "Phase 2: マルチコマ分割",
                "params": {"manga_mode": True, "panel_split": True, "low_threshold": True},
            },
            {
                "name": "Phase 2: 全機能",
                "params": {
                    "manga_mode": True,
                    "effect_removal": True,
                    "panel_split": True,
                    "low_threshold": True,
                },
            },
        ]

    # ソリッドフィル検出テストを追加
    if test_solid_fill:
        test_configs.extend(
            [
                {
                    "name": "Phase 2: ソリッドフィル検出",
                    "params": {
                        "manga_mode": True,
                        "solid_fill_detection": True,
                        "low_threshold": True,
                    },
                },
                {
                    "name": "Phase 2: 全機能 + ソリッドフィル",
                    "params": {
                        "manga_mode": True,
                        "effect_removal": True,
                        "panel_split": True,
                        "solid_fill_detection": True,
                        "low_threshold": True,
                    },
                },
            ]
        )

    results = []

    print("Starting Phase 2 batch processing")
    print("=" * 60)

    for i, image in enumerate(failed_images, 1):
        print(f"\nProcessing {i}/{len(failed_images)}: {image['name']}")
        print(f"   説明: {image['description']}")
        print("-" * 40)

        image_results = {"image": image["name"], "configs": {}}

        # 各設定でテスト
        for config in test_configs:
            print(f"\nTesting with {config['name']}...")

            try:
                if output_dir:
                    output_path = os.path.join(output_dir, image["name"])
                else:
                    output_path = f"/tmp/phase2_test_{image['name'].replace('.jpg', '')}_{config['name'].replace(' ', '_').replace(':', '')}"

                # 閾値パラメータを動的に設定
                params = config["params"].copy()
                if "yolo_params" not in params:
                    params["yolo_params"] = {}
                params["yolo_params"]["conf"] = score_threshold

                result = extract_character_from_path(
                    image["path"], output_path=output_path, verbose=False, **params
                )

                success = result.get("success", False)
                processing_time = result.get("processing_time", 0)
                error = result.get("error", "")

                if success:
                    print(f"   SUCCESS! ({processing_time:.1f}s)")
                    if "retry_stage" in result:
                        print(f"      リトライ段階: {result['retry_stage']}")
                    if "complexity_info" in result:
                        complexity = result["complexity_info"].get("complexity", "unknown")
                        print(f"      複雑度: {complexity}")
                else:
                    print(f"   FAILED: {error}")

                image_results["configs"][config["name"]] = {
                    "success": success,
                    "time": processing_time,
                    "error": error,
                }

            except Exception as e:
                print(f"   EXCEPTION: {e}")
                image_results["configs"][config["name"]] = {
                    "success": False,
                    "time": 0,
                    "error": f"Exception: {e}",
                }

        results.append(image_results)

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 Phase 2テスト結果サマリー")
    print("=" * 60)

    for image_result in results:
        print(f"\n📸 {image_result['image']}:")

        success_count = 0
        for config_name, config_result in image_result["configs"].items():
            status = "✅" if config_result["success"] else "❌"
            time_str = f"{config_result['time']:.1f}s" if config_result["success"] else ""

            print(f"   {status} {config_name:<25} {time_str}")

            if config_result["success"]:
                success_count += 1

        success_rate = success_count / len(test_configs) * 100
        print(f"   📈 成功率: {success_count}/{len(test_configs)} ({success_rate:.0f}%)")

    # 全体統計
    total_tests = len(failed_images) * len(test_configs)
    total_successes = sum(
        sum(1 for config_result in image_result["configs"].values() if config_result["success"])
        for image_result in results
    )

    overall_success_rate = total_successes / total_tests * 100

    print(f"\n🎯 Phase 2改善効果:")
    print(f"   総テスト数: {total_tests}")
    print(f"   成功数: {total_successes}")
    print(f"   全体成功率: {overall_success_rate:.1f}%")

    # Phase 1からの改善を評価
    print(f"\n📈 Phase 1からの改善:")
    print(f"   Phase 1時: 0% (2画像とも全失敗)")
    print(f"   Phase 2時: {overall_success_rate:.1f}%")

    if overall_success_rate > 50:
        print("🎉 Phase 2で大幅改善！失敗画像の抽出に成功しています。")
    elif overall_success_rate > 25:
        print("✅ Phase 2で改善効果あり。更なる調整で向上が期待できます。")
    else:
        print("⚠️ Phase 2でも改善が限定的。Phase 3機能が必要です。")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2機能テスト with YOLO閾値調整")
    parser.add_argument(
        "--score_threshold", type=float, default=0.07, help="YOLO人物検出スコア閾値 (default: 0.07)"
    )
    parser.add_argument("--test_solid_fill", action="store_true", help="ソリッドフィル検出機能をテスト")
    parser.add_argument("--input_dir", type=str, help="Input directory for batch processing")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch processing")
    args = parser.parse_args()

    if args.input_dir:
        print(f"Input directory: {args.input_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print(f"YOLO threshold: {args.score_threshold}")
    if args.test_solid_fill:
        print("Including solid fill detection")
    test_phase2_on_failed_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        test_solid_fill=args.test_solid_fill,
    )
