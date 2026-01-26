#!/usr/bin/env python3
"""
v0.3.5最高品質キャラクター抽出バッチ実行
Phase 1品質評価システム完全統合版 - 命名規則改善版

改善点:
- ファイル命名規則の統一化
- 出力ファイル名が入力ファイル名と完全一致
- 評価システムとの完全互換性確保
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def run_batch_v035():
    """v0.3.5最高品質バッチ抽出実行（命名規則改善版）"""

    # パス設定（ユーザー指定）
    input_path = "/mnt/c/AItools/lora/train/yado/org/kaname09"
    output_path = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_0_3_5"

    print("🚀 v0.3.5最高品質キャラクター抽出バッチ実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    print(f"バージョン: v0.3.5 (命名規則改善版)")
    print("✨ 改善点: ファイル命名規則統一、評価システム完全互換")
    print("✨ Phase 1機能: 滑らかさ評価、切断検出、混入定量化、フィードバック学習、効率サンプリング")

    # 入力パス検証
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)

    # 画像ファイル取得
    image_files = sorted(
        list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    )

    if not image_files:
        print(f"❌ エラー: 入力パスに画像ファイルがありません: {input_path}")
        sys.exit(1)

    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")

    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # バッチ処理実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path

        success_count = 0
        error_count = 0
        start_time = time.time()
        results = []

        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")

            try:
                image_start = time.time()

                # 出力ファイル名設定（入力ファイル名と同じ - PROJECT_SETTINGS.md準拠）
                output_filename = image_file.name
                output_file_path = Path(output_path) / output_filename

                # v0.3.5最高品質設定での抽出実行
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file_path),
                    multi_character_criteria="balanced",
                    enhance_contrast=True,
                    save_mask=True,
                    save_transparent=True,
                    verbose=False,
                    high_quality=True,
                    min_yolo_score=0.01,
                )

                image_time = time.time() - image_start

                if result.get("success", False):
                    success_count += 1
                    quality_score = result.get("quality_score", 0.0)
                    print(f"✅ 成功: {output_filename}")
                    print(f"   品質スコア: {quality_score:.3f}")
                    print(f"   処理時間: {image_time:.2f}秒")

                    results.append(
                        {
                            "filename": output_filename,
                            "success": True,
                            "quality_score": quality_score,
                            "processing_time": image_time,
                        }
                    )
                else:
                    error_count += 1
                    error_msg = result.get("error", "不明なエラー")
                    print(f"❌ 失敗: {output_filename} - {error_msg}")

                    results.append(
                        {
                            "filename": output_filename,
                            "success": False,
                            "error": error_msg,
                            "processing_time": image_time,
                        }
                    )

            except Exception as e:
                error_count += 1
                print(f"❌ 処理エラー: {image_file.name} - {str(e)}")
                results.append(
                    {
                        "filename": image_file.name,
                        "success": False,
                        "error": str(e),
                        "processing_time": 0,
                    }
                )

        # バッチ処理完了統計
        total_time = time.time() - start_time
        success_rate = success_count / len(image_files) * 100

        # 結果をJSONファイルに保存
        result_summary = {
            "version": "v0.3.5",
            "timestamp": datetime.now().isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "total_images": len(image_files),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "total_time": total_time,
            "average_time_per_image": total_time / len(image_files),
            "results": results,
        }

        result_file = Path(output_path) / "batch_results_v035.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)

        # 結果レポート出力
        print("\n" + "=" * 80)
        print("📊 v0.3.5バッチ処理完了レポート")
        print("=" * 80)

        print(f"\n📈 処理結果:")
        print(f"  総処理数: {len(image_files)}枚")
        print(f"  成功: {success_count}枚")
        print(f"  失敗: {error_count}枚")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  総処理時間: {total_time:.1f}秒")
        print(f"  平均処理時間: {total_time / len(image_files):.1f}秒/枚")

        # 品質スコア統計
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            quality_scores = [r["quality_score"] for r in successful_results]
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)

            print(f"\n📊 品質スコア統計:")
            print(f"  平均品質スコア: {avg_quality:.3f}")
            print(f"  最低品質スコア: {min_quality:.3f}")
            print(f"  最高品質スコア: {max_quality:.3f}")

        print(f"\n💾 詳細結果保存先: {result_file}")
        print(f"\n✅ v0.3.5バッチ処理完了!")

        return success_rate >= 70.0  # 70%以上の成功率で成功判定

    except Exception as e:
        print(f"❌ バッチ処理で重大エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")
        return False


def main():
    """メイン実行関数"""
    print("🚀 v0.3.5最高品質バッチ処理開始")
    print("📋 PROJECT_SETTINGS.md準拠の命名規則で実行")

    success = run_batch_v035()

    if success:
        print("\n✅ バッチ処理成功!")
        sys.exit(0)
    else:
        print("\n❌ バッチ処理失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
