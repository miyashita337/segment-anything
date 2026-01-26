#!/usr/bin/env python3
"""
境界強調システムテスト - 問題画像での効果検証
"""

import subprocess
import sys
from pathlib import Path


def test_boundary_enhanced_extraction():
    """境界強調版キャラクター抽出のテスト"""

    # テスト対象画像（境界認識問題があった画像）
    test_cases = [
        "kaname08_0000_cover.jpg",  # 腕だけ抽出問題
        "kaname08_0022.jpg",  # 顔境界認識問題
        "kaname08_0001.jpg",  # A評価だったもの（比較用）
    ]

    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kaname08")
    output_dir = Path(
        "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname08_boundary_enhanced_test"
    )

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 境界強調システム テスト開始")
    print(f"📁 出力: {output_dir}")
    print("=" * 60)

    success_count = 0

    for i, filename in enumerate(test_cases, 1):
        input_path = input_dir / filename
        output_path = output_dir / filename

        print(f"📸 テスト [{i}/{len(test_cases)}]: {filename}")

        if not input_path.exists():
            print(f"❌ 入力ファイルが見つかりません: {input_path}")
            continue

        try:
            # 境界強調版抽出実行
            cmd = [
                "python3",
                "-m",
                "features.extraction.commands.extract_character",
                str(input_path),
                "-o",
                str(output_path),
                "--verbose",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if output_path.exists():
                success_count += 1
                print(f"✅ 成功: {filename}")

                # 統計情報表示
                stdout_lines = result.stdout.split("\n")
                for line in stdout_lines:
                    if "境界強調統計" in line:
                        print(f"   📊 {line.strip()}")
                    elif "Character extracted:" in line and "size:" in line:
                        size_info = line.split("size:")[1].strip().rstrip(")")
                        print(f"   📏 抽出サイズ: {size_info}")

                # 前回結果との比較
                print(f"   💡 前回の問題: ", end="")
                if filename == "kaname08_0000_cover.jpg":
                    print("腕だけ抽出 → 境界強調で改善期待")
                elif filename == "kaname08_0022.jpg":
                    print("顔境界不正確 → 境界強調で改善期待")
                elif filename == "kaname08_0001.jpg":
                    print("A評価だった画像（ベースライン）")

            else:
                print(f"❌ 失敗: {filename}")
                if result.stderr:
                    print(f"   エラー: {result.stderr.strip()[-100:]}")

        except subprocess.TimeoutExpired:
            print(f"❌ タイムアウト: {filename}")
        except Exception as e:
            print(f"❌ エラー: {filename} - {e}")

        print("-" * 40)

    print("=" * 60)
    print(f"🎯 境界強調テスト完了")
    print(f"✅ 成功: {success_count}/{len(test_cases)}")

    if success_count == len(test_cases):
        print("🎉 全テスト成功！境界強調システムが正常動作")
    elif success_count > 0:
        print(f"🔧 {success_count}件成功。一部改善が見られます")
    else:
        print("⚠️ 全テスト失敗。システム調整が必要です")

    print(f"\n📁 結果確認: {output_dir}")
    print("💡 評価システムで前回結果と比較してください")


if __name__ == "__main__":
    test_boundary_enhanced_extraction()
