#!/usr/bin/env python3
"""
Phase A: GPT-4O推奨ボックス拡張機能のテスト
顔検出ボックスを2.5-3倍水平、4倍垂直に拡張してからSAM処理
"""

import numpy as np
import cv2

import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))


def test_box_expansion_utilities():
    """ボックス拡張ユーティリティのテスト"""
    print("🧪 Phase A: ボックス拡張ユーティリティテスト開始")

    try:
        from features.extraction.utils.box_expansion import (
            BoxExpansionProcessor,
            apply_gpt4o_expansion_strategy,
        )

        # テスト用ダミーデータ
        test_detections = [
            {"bbox": [100, 150, 80, 120], "confidence": 0.85, "class_name": "person"},
            {"bbox": [300, 200, 60, 80], "confidence": 0.92, "class_name": "face"},
            {"bbox": [500, 100, 90, 140], "confidence": 0.78, "class_name": "person"},
        ]

        test_image_shape = (720, 1280)  # height, width

        print(f"📏 テスト画像サイズ: {test_image_shape[1]}x{test_image_shape[0]}")
        print(f"🎯 検出数: {len(test_detections)}")

        # 各戦略をテスト
        strategies = ["conservative", "balanced", "aggressive"]

        for strategy in strategies:
            print(f"\n🔬 戦略テスト: {strategy}")

            expanded = apply_gpt4o_expansion_strategy(test_detections, test_image_shape, strategy)

            for i, detection in enumerate(expanded):
                orig_bbox = detection["bbox_original"]
                exp_bbox = detection["bbox"]
                exp_info = detection["expansion_info"]

                print(f"   検出{i+1} ({detection.get('class_name', 'unknown')}):")
                print(f"      元ボックス: {orig_bbox}")
                print(f"      拡張ボックス: {exp_bbox}")
                print(
                    f"      拡張倍率: H{exp_info['horizontal_factor']:.2f}x V{exp_info['vertical_factor']:.2f}x"
                )
                print(f"      境界制限: {'あり' if exp_info['clipped_to_bounds'] else 'なし'}")

        print("\n✅ ボックス拡張ユーティリティテスト完了")
        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


def test_yolo_wrapper_integration():
    """YOLOラッパーとの統合テスト"""
    print("\n🧪 YOLOラッパー統合テスト開始")

    try:
        from features.extraction.models.yolo_wrapper import YOLOModelWrapper

        # YOLOラッパーの初期化テスト
        wrapper = YOLOModelWrapper()
        info = wrapper.get_model_info()

        print(f"📊 YOLO情報: {info}")

        # 新しいメソッドシグネチャの確認
        import inspect

        score_method = wrapper.score_masks_with_detections
        signature = inspect.signature(score_method)

        print(f"🔍 score_masks_with_detections シグネチャ:")
        for param_name, param in signature.parameters.items():
            print(f"   {param_name}: {param.annotation} = {param.default}")

        # use_expanded_boxesパラメータが存在するかチェック
        if "use_expanded_boxes" in signature.parameters:
            print("✅ use_expanded_boxes パラメータ追加済み")
        else:
            print("❌ use_expanded_boxes パラメータが見つかりません")
            return False

        if "expansion_strategy" in signature.parameters:
            print("✅ expansion_strategy パラメータ追加済み")
        else:
            print("❌ expansion_strategy パラメータが見つかりません")
            return False

        print("✅ YOLOラッパー統合テスト完了")
        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


def test_extract_character_integration():
    """extract_character.py統合テスト"""
    print("\n🧪 extract_character統合テスト開始")

    try:
        # 新しいパラメータの確認
        import inspect
        from features.extraction.commands.extract_character import extract_character_from_path

        signature = inspect.signature(extract_character_from_path)

        print(f"🔍 extract_character_from_path シグネチャ:")
        param_count = 0
        for param_name, param in signature.parameters.items():
            if param_name in ["use_box_expansion", "expansion_strategy"]:
                print(f"   ✅ {param_name}: {param.annotation} = {param.default}")
                param_count += 1
            elif param_name in ["kwargs"]:
                continue  # **kwargsはスキップ
            else:
                # 他のパラメータは省略して表示
                if param_count < 5:  # 最初の5つだけ表示
                    print(f"   {param_name}: {param.annotation} = {param.default}")

        # ボックス拡張パラメータが存在するかチェック
        if (
            "use_box_expansion" in signature.parameters
            and "expansion_strategy" in signature.parameters
        ):
            print("✅ ボックス拡張パラメータ追加済み")
        else:
            print("❌ ボックス拡張パラメータが見つかりません")
            return False

        print("✅ extract_character統合テスト完了")
        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


def test_command_line_interface():
    """コマンドラインインターフェースのテスト"""
    print("\n🧪 コマンドラインインターフェーステスト開始")

    try:
        import subprocess
        import sys

        # ヘルプ表示でオプションが追加されているかチェック
        result = subprocess.run(
            [sys.executable, "features/extraction/commands/extract_character.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )

        help_text = result.stdout

        # ボックス拡張オプションが含まれているかチェック
        if "--use-box-expansion" in help_text:
            print("✅ --use-box-expansion オプション追加済み")
        else:
            print("❌ --use-box-expansion オプションが見つかりません")
            return False

        if "--expansion-strategy" in help_text:
            print("✅ --expansion-strategy オプション追加済み")
        else:
            print("❌ --expansion-strategy オプションが見つかりません")
            return False

        # ヘルプテキストにPhase Aの説明があるかチェック
        if "Phase A" in help_text:
            print("✅ Phase A 説明文追加済み")
        else:
            print("❌ Phase A 説明文が見つかりません")

        # GPT-4O推奨の説明があるかチェック
        if "GPT-4O" in help_text:
            print("✅ GPT-4O推奨の説明追加済み")
        else:
            print("❌ GPT-4O推奨の説明が見つかりません")

        print("✅ コマンドラインインターフェーステスト完了")
        return True

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🚀 Phase A: GPT-4O推奨ボックス拡張機能 総合テスト開始")
    print("=" * 60)

    tests = [
        ("ボックス拡張ユーティリティ", test_box_expansion_utilities),
        ("YOLOラッパー統合", test_yolo_wrapper_integration),
        ("extract_character統合", test_extract_character_integration),
        ("コマンドラインインターフェース", test_command_line_interface),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n📋 テスト: {test_name}")
        print("-" * 40)

        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ テスト例外: {e}")
            results[test_name] = False

    # 結果サマリ
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリ:")

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 総合結果: {passed_tests}/{total_tests} テスト成功")

    if passed_tests == total_tests:
        print("🎉 Phase A: GPT-4O推奨ボックス拡張機能の実装完了!")
        print("\n📝 使用方法:")
        print(
            "   python features/extraction/commands/extract_character.py <image> --use-box-expansion"
        )
        print(
            "   python features/extraction/commands/extract_character.py <image> --use-box-expansion --expansion-strategy aggressive"
        )
        return True
    else:
        print("⚠️ 一部テストが失敗しました。実装を確認してください。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
