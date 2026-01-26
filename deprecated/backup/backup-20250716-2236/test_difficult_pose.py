#!/usr/bin/env python3
"""
複雑ポーズ処理機能のテストスクリプト
"""

import os
import sys

sys.path.append(".")


def test_difficult_poses():
    """失敗画像での複雑ポーズ処理テスト"""

    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start

    start()
    print("✅ モデル初期化完了\n")

    from commands.extract_character import extract_character_from_path

    # テスト対象画像（失敗していた4枚）
    test_images = [
        {
            "path": "/mnt/c/AItools/lora/train/yado/org/kana03/25_kana03_0024.jpg",
            "name": "25_kana03_0024.jpg",
            "description": "複雑な座りポーズ + 宣伝画像",
        },
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
        {
            "path": "/mnt/c/AItools/lora/train/yado/org/kana03/20_kana03_0019.jpg",
            "name": "20_kana03_0019.jpg",
            "description": "複雑な絡み合い + エフェクト",
        },
    ]

    # テスト設定 (Phase 2対応版)
    test_configs = [
        {"name": "通常設定（参考）", "params": {}},
        {"name": "低閾値モード", "params": {"low_threshold": True}},
        {"name": "複雑ポーズモード", "params": {"difficult_pose": True}},
        {"name": "高品質モード", "params": {"high_quality": True, "low_threshold": True}},
        {"name": "自動リトライモード", "params": {"auto_retry": True}},
        {"name": "Phase 2: エフェクト線除去", "params": {"manga_mode": True, "effect_removal": True}},
        {"name": "Phase 2: マルチコマ分割", "params": {"manga_mode": True, "panel_split": True}},
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

    print("🧪 複雑ポーズ処理テスト開始\n")
    print("=" * 80)

    for i, image in enumerate(test_images, 1):
        print(f"\n📸 画像 {i}/4: {image['name']}")
        print(f"   説明: {image['description']}")
        print(f"   パス: {image['path']}")
        print("-" * 60)

        image_results = {"image": image["name"], "configs": {}}

        # 各設定でテスト
        for config in test_configs:
            print(f"\n🔧 {config['name']} でテスト中...")

            try:
                # 出力パス設定
                output_path = f"/tmp/test_{image['name'].replace('.jpg', '')}_{config['name'].replace(' ', '_')}"

                # 抽出実行
                result = extract_character_from_path(
                    image["path"],
                    output_path=output_path,
                    verbose=False,  # テスト中は簡潔に
                    **config["params"],
                )

                success = result.get("success", False)
                processing_time = result.get("processing_time", 0)
                error = result.get("error", "")

                if success:
                    print(f"   ✅ 成功! ({processing_time:.1f}秒)")
                    if "retry_stage" in result:
                        print(f"      リトライ段階: {result['retry_stage']}")
                    if "config_used" in result:
                        print(f"      使用設定: {result['config_used']}")
                else:
                    print(f"   ❌ 失敗: {error}")

                image_results["configs"][config["name"]] = {
                    "success": success,
                    "time": processing_time,
                    "error": error,
                    "retry_stage": result.get("retry_stage", 0),
                    "config_used": result.get("config_used", ""),
                }

            except Exception as e:
                print(f"   💥 例外発生: {e}")
                image_results["configs"][config["name"]] = {
                    "success": False,
                    "time": 0,
                    "error": f"Exception: {e}",
                    "retry_stage": 0,
                    "config_used": "",
                }

        results.append(image_results)

    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)

    for image_result in results:
        print(f"\n📸 {image_result['image']}:")

        success_count = 0
        for config_name, config_result in image_result["configs"].items():
            status = "✅" if config_result["success"] else "❌"
            time_str = f"{config_result['time']:.1f}s" if config_result["success"] else ""
            retry_str = (
                f"(Stage {config_result['retry_stage']})"
                if config_result["retry_stage"] > 1
                else ""
            )

            print(f"   {status} {config_name:<20} {time_str:<8} {retry_str}")

            if config_result["success"]:
                success_count += 1

        success_rate = success_count / len(test_configs) * 100
        print(f"   📈 成功率: {success_count}/{len(test_configs)} ({success_rate:.0f}%)")

    # 全体統計
    total_tests = len(test_images) * len(test_configs)
    total_successes = sum(
        sum(1 for config_result in image_result["configs"].values() if config_result["success"])
        for image_result in results
    )

    overall_success_rate = total_successes / total_tests * 100

    print(f"\n🎯 全体成績:")
    print(f"   総テスト数: {total_tests}")
    print(f"   成功数: {total_successes}")
    print(f"   全体成功率: {overall_success_rate:.1f}%")

    if overall_success_rate > 75:
        print("🎉 Phase 2実装成功！大幅な改善が確認されました。")
    elif overall_success_rate > 60:
        print("✅ Phase 2で顕著な改善。実用レベルに達しています。")
    elif overall_success_rate > 40:
        print("🔧 Phase 2で部分的改善。個別調整で更なる向上が可能です。")
    else:
        print("⚠️ Phase 3機能の実装が必要です。")

    return results


if __name__ == "__main__":
    test_difficult_poses()
