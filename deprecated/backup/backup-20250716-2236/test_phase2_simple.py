#!/usr/bin/env python3
"""
Phase 2機能の簡単テスト
失敗していた2画像に対してエフェクト線除去・マルチコマ分割を試行
"""

import os
import sys

sys.path.append(".")


def test_phase2_on_failed_images():
    """失敗画像2枚でPhase 2機能をテスト"""

    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start

    start()
    print("✅ モデル初期化完了\n")

    from commands.extract_character import extract_character_from_path

    # 失敗していた画像2枚
    failed_images = [
        {
            "path": "/mnt/c/AItools/lora/train/yado/org/kana03/21_kana03_0020.jpg",
            "name": "21_kana03_0020.jpg",
            "description": "ダイナミックなポーズ + エフェクト線",
        },
        {
            "path": "/mnt/c/AItools/lora/train/yado/org/kana03/16_kana03_0015.jpg",
            "name": "16_kana03_0015.jpg",
            "description": "マルチコマ構成",
        },
    ]

    # Phase 2テスト設定
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

    results = []

    print("🧪 Phase 2機能テスト開始")
    print("=" * 60)

    for i, image in enumerate(failed_images, 1):
        print(f"\n📸 失敗画像 {i}/2: {image['name']}")
        print(f"   説明: {image['description']}")
        print("-" * 40)

        image_results = {"image": image["name"], "configs": {}}

        # 各設定でテスト
        for config in test_configs:
            print(f"\n🔧 {config['name']} でテスト中...")

            try:
                output_path = f"/tmp/phase2_test_{image['name'].replace('.jpg', '')}_{config['name'].replace(' ', '_').replace(':', '')}"

                result = extract_character_from_path(
                    image["path"], output_path=output_path, verbose=False, **config["params"]
                )

                success = result.get("success", False)
                processing_time = result.get("processing_time", 0)
                error = result.get("error", "")

                if success:
                    print(f"   ✅ 成功! ({processing_time:.1f}秒)")
                    if "retry_stage" in result:
                        print(f"      リトライ段階: {result['retry_stage']}")
                    if "complexity_info" in result:
                        complexity = result["complexity_info"].get("complexity", "unknown")
                        print(f"      複雑度: {complexity}")
                else:
                    print(f"   ❌ 失敗: {error}")

                image_results["configs"][config["name"]] = {
                    "success": success,
                    "time": processing_time,
                    "error": error,
                }

            except Exception as e:
                print(f"   💥 例外発生: {e}")
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
    test_phase2_on_failed_images()
