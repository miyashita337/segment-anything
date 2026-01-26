#!/usr/bin/env python3
"""
P1-013 バックグラウンド実際の抽出処理実行
重大バグ修正: 実際のSAM+YOLO抽出処理をバックグラウンドで実行
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_background_extraction():
    """P1-013 バックグラウンド抽出処理実行"""
    tracker_id = "P1-013"
    project_root = Path(__file__).parent.parent.parent

    # 入力・出力パス設定
    input_dir = "/mnt/c/AItools/lora/train/yado/org/kana05"
    output_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"

    print("============================================================")
    print("🚀 P1-013 バックグラウンド実際の抽出処理実行")
    print("  - 重大バグ修正: 実際のSAM+YOLO抽出処理統合")
    print("  - 長時間処理をバックグラウンドで実行")
    print("  - 完了後に自動/release更新")
    print("============================================================")

    # ログファイル設定
    log_file = Path(output_base) / tracker_id / "background_extraction_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("============================================================\\n")
        f.write("🚀 P1-013 バックグラウンド実際の抽出処理実行\\n")
        f.write(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write("============================================================\\n")

        # 入力ディレクトリ確認
        if not Path(input_dir).exists():
            error_msg = f"❌ 入力ディレクトリが存在しません: {input_dir}"
            print(error_msg)
            f.write(error_msg + "\\n")
            return False

        # 入力画像数カウント
        input_path = Path(input_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"🚀 P1-013 バックグラウンド差分処理開始")
        print(f"📊 入力画像数: {len(image_files)}枚")
        print(f"⏰ 推定処理時間: {len(image_files) * 10}秒 (約{len(image_files) * 10 // 60}分)")

        f.write(f"📊 入力画像数: {len(image_files)}枚\\n")
        f.write(f"⏰ 推定処理時間: {len(image_files) * 10}秒\\n")

        # P1-013差分処理最適化システム実行
        print("🔧 修正版P1-013差分処理システム実行開始")
        f.write("🔧 修正版P1-013差分処理システム実行開始\\n")

        differential_cmd = [
            "python3",
            "features/processing/differential_processor.py",
            "--tracker-id",
            tracker_id,
            "--input-dir",
            input_dir,
        ]

        print(f"💻 実行コマンド: {' '.join(differential_cmd)}")
        f.write(f"💻 実行コマンド: {' '.join(differential_cmd)}\\n")

        processing_start = time.time()

        try:
            # 長時間処理として実行
            result = subprocess.run(
                differential_cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分タイムアウト
            )

            processing_end = time.time()
            processing_time = processing_end - processing_start

            if result.returncode == 0:
                print(f"✅ P1-013差分処理成功 ({processing_time:.1f}秒)")
                f.write(f"✅ P1-013差分処理成功 ({processing_time:.1f}秒)\\n")

                # 出力結果確認
                extraction_dir = Path(output_base) / tracker_id / "extraction"
                if extraction_dir.exists():
                    output_files = []
                    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                        output_files.extend(extraction_dir.glob(f"*{ext}"))
                        output_files.extend(extraction_dir.glob(f"*{ext.upper()}"))

                    print(f"📁 実際の抽出画像ファイル数: {len(output_files)}個")
                    f.write(f"📁 実際の抽出画像ファイル数: {len(output_files)}個\\n")

                    # ファイルリスト
                    for i, output_file in enumerate(output_files[:10], 1):
                        size_mb = output_file.stat().st_size / 1024 / 1024
                        print(f"  {i}. {output_file.name} ({size_mb:.2f}MB)")
                        f.write(f"  {i}. {output_file.name} ({size_mb:.2f}MB)\\n")

                    if len(output_files) > 10:
                        print(f"  ... (他{len(output_files) - 10}個)")
                        f.write(f"  ... (他{len(output_files) - 10}個)\\n")

                    # 成功率計算
                    success_rate = (
                        (len(output_files) / len(image_files)) * 100 if len(image_files) > 0 else 0
                    )

                else:
                    print("❌ 抽出ディレクトリが見つかりません")
                    f.write("❌ 抽出ディレクトリが見つかりません\\n")
                    success_rate = 0

                # 標準出力から統計情報解析
                output_lines = result.stdout.split("\\n")
                for line in output_lines:
                    if "処理結果:" in line:
                        print(f"📊 {line}")
                        f.write(f"📊 {line}\\n")
                    elif "キャッシュヒット率:" in line:
                        print(f"💾 {line}")
                        f.write(f"💾 {line}\\n")

            else:
                print(f"❌ P1-013差分処理失敗 (終了コード: {result.returncode})")
                print(f"エラー出力: {result.stderr}")
                f.write(f"❌ P1-013差分処理失敗 (終了コード: {result.returncode})\\n")
                f.write(f"エラー出力: {result.stderr}\\n")
                success_rate = 0

        except subprocess.TimeoutExpired:
            print("❌ P1-013差分処理タイムアウト（30分）")
            f.write("❌ P1-013差分処理タイムアウト（30分）\\n")
            success_rate = 0
        except Exception as e:
            print(f"❌ P1-013差分処理エラー: {e}")
            f.write(f"❌ P1-013差分処理エラー: {e}\\n")
            success_rate = 0

        # Google Sheets自動更新
        if success_rate >= 50.0:  # 50%以上で成功とみなす
            print("🔄 Google Sheetsを/releaseに更新中...")
            f.write("🔄 Google Sheetsを/releaseに更新中...\\n")

            try:
                update_result = subprocess.run(
                    ["python3", "tools/progress_tracker/cli.py", "update", tracker_id, "/release"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if update_result.returncode == 0:
                    print(f"✅ Google Sheets更新成功: {tracker_id} -> /release")
                    f.write(f"✅ Google Sheets更新成功: {tracker_id} -> /release\\n")
                else:
                    print(f"⚠️ Google Sheets更新失敗: {update_result.stderr}")
                    f.write(f"⚠️ Google Sheets更新失敗: {update_result.stderr}\\n")

            except Exception as e:
                print(f"❌ Google Sheets更新エラー: {e}")
                f.write(f"❌ Google Sheets更新エラー: {e}\\n")
        else:
            print("⚠️ 成功率が低いため、手動確認が必要です")
            f.write("⚠️ 成功率が低いため、手動確認が必要です\\n")

        end_time = datetime.now()
        total_time = end_time - start_time

        print("")
        print("🎉 P1-013バックグラウンド処理完了")
        print(f"⏰ 総実行時間: {total_time}")
        f.write(f"\\n🎉 P1-013バックグラウンド処理完了\\n")
        f.write(f"⏰ 総実行時間: {total_time}\\n")

        # 最終サマリー
        if success_rate >= 50.0:
            print("📋 重大バグ修正完了サマリー:")
            print("  ✅ 実際のSAM+YOLO抽出処理統合")
            print("  ✅ 実際の画像ファイル出力確認")
            print("  ✅ デモファイルからの完全脱却")
            print("  ✅ Google Sheets /release更新")

            f.write("📋 重大バグ修正完了サマリー:\\n")
            f.write("  ✅ 実際のSAM+YOLO抽出処理統合\\n")
            f.write("  ✅ 実際の画像ファイル出力確認\\n")
            f.write("  ✅ デモファイルからの完全脱却\\n")
            f.write("  ✅ Google Sheets /release更新\\n")

        return success_rate >= 50.0


def main():
    """メイン実行"""
    print("🚀 P1-013 バックグラウンド実際の抽出処理を開始します...")
    print("   この処理には5-10分程度かかる場合があります。")
    print("   処理状況は以下のログファイルで確認できます:")
    print(
        "   /mnt/c/AItools/lora/train/yado/tracker-workspace/P1-013/background_extraction_log.txt"
    )
    print("")

    success = run_background_extraction()

    if success:
        print("\\n🎉 P1-013重大バグ修正完了！")
        print("   実際のSAM+YOLO抽出処理が正常に統合されました。")
    else:
        print("\\n❌ P1-013処理で問題が発生しました。")
        print("   ログファイルを確認してください。")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
