#!/usr/bin/env python3
"""
kaname07最高品質キャラクター抽出バッチ実行
Phase 0リファクタリング後の新構造対応版
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def run_kaname07_batch_extraction():
    """kaname07バッチ抽出実行"""

    # パス設定（ユーザー指定）
    input_path = "/mnt/c/AItools/lora/train/yado/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname07"

    print("🚀 kaname07最高品質キャラクター抽出バッチ実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")

    # 入力パス検証
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)

    # 画像ファイル取得
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))

    if not image_files:
        print(f"❌ エラー: 入力パスに画像ファイルがありません: {input_path}")
        sys.exit(1)

    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")

    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 新構造での抽出実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path

        success_count = 0
        error_count = 0
        start_time = time.time()

        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")

            try:
                # 出力ファイル名設定（番号付きで整理）
                output_filename = f"{i:05d}_{image_file.stem}.jpg"
                output_file_path = Path(output_path) / output_filename

                # 最高品質設定での抽出実行
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file_path),
                    multi_character_criteria="fullbody_priority_enhanced",  # 改良版全身検出
                    enhance_contrast=True,  # コントラスト強化
                    filter_text=True,  # テキストフィルタリング
                    save_mask=True,  # マスク保存
                    save_transparent=True,  # 透明背景保存
                    verbose=False,  # バッチ処理なので詳細ログは抑制
                    high_quality=True,  # 高品質処理
                    difficult_pose=True,  # 困難姿勢対応
                    adaptive_learning=True,  # 適応学習
                    manga_mode=True,  # 漫画モード
                    effect_removal=True,  # エフェクト除去
                    min_yolo_score=0.05,  # YOLO閾値を緩めに設定
                )

                if result.get("success", False):
                    success_count += 1
                    print(f"✅ 成功: {output_filename}")

                    # 品質情報表示
                    if "quality_score" in result:
                        print(f"   品質スコア: {result['quality_score']:.3f}")
                    if "extraction_analysis" in result:
                        analysis = result["extraction_analysis"]
                        print(f"   完全性: {analysis.get('completeness_score', 'N/A'):.3f}")
                else:
                    error_count += 1
                    print(f"❌ 失敗: {image_file.name}")
                    if "error" in result:
                        print(f"   エラー: {result['error']}")

            except Exception as e:
                error_count += 1
                print(f"❌ 例外: {image_file.name} - {str(e)}")

            # 進捗表示
            elapsed = time.time() - start_time
            remaining = len(image_files) - i
            if i > 0:
                avg_time = elapsed / i
                eta = avg_time * remaining
                print(
                    f"   進捗: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%), "
                    f"経過: {elapsed/60:.1f}分, 残り推定: {eta/60:.1f}分"
                )

        # 結果サマリー
        total_time = time.time() - start_time
        print(f"\n🎉 バッチ処理完了!")
        print(f"📊 結果サマリー:")
        print(f"   総処理数: {len(image_files)}")
        print(f"   成功: {success_count}")
        print(f"   失敗: {error_count}")
        print(f"   成功率: {success_count/len(image_files)*100:.1f}%")
        print(f"   総処理時間: {total_time/60:.1f}分")
        print(f"   平均処理時間: {total_time/len(image_files):.1f}秒/画像")
        print(f"📁 出力ディレクトリ: {output_path}")

        if error_count > 0:
            print(f"⚠️  {error_count}個のファイルでエラーが発生しました")

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("extract_character モジュールが見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_kaname07_batch_extraction()
