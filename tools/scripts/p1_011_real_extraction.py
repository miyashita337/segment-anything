#!/usr/bin/env python3
"""
P1-011実際のSAM+YOLO抽出パイプライン実行
完了後に自動的に/releaseステータスに更新
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_real_extraction(tracker_id: str = "P1-011"):
    """実際の抽出パイプライン実行"""
    print(f"🚀 {tracker_id} 実際の抽出パイプライン開始")

    # 入力・出力ディレクトリ設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana05")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")

    # 入力ディレクトリ確認
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False

    # 画像数確認
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"📊 入力画像数: {len(image_files)}枚")

    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)

    # 既存のタスクディレクトリクリーンアップ
    for task_dir in output_dir.glob("task_*"):
        import shutil

        if task_dir.is_dir():
            shutil.rmtree(task_dir)
            print(f"🧹 既存タスクディレクトリ削除: {task_dir.name}")

    try:
        # 実際のSAM+YOLO抽出実行
        print("🔧 SAM+YOLO抽出コマンド実行開始")
        start_time = time.time()

        command = [
            "python3",
            "tools/core/sam_yolo_character_segment.py",
            "--mode",
            "reproduce-auto",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--score_threshold",
            "0.07",
        ]

        print(f"💻 実行コマンド: {' '.join(command)}")

        # 実行
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分タイムアウト
        )

        end_time = time.time()
        processing_time = end_time - start_time

        if result.returncode == 0:
            # 成功
            print(f"✅ 抽出処理成功 ({processing_time:.1f}秒)")

            # 出力ファイル確認
            output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            print(f"📁 出力ファイル数: {len(output_files)}個")

            # 成功率計算
            success_rate = (len(output_files) / len(image_files)) * 100 if image_files else 0
            print(f"📈 抽出成功率: {success_rate:.1f}%")

            # 自動的に/releaseに更新
            if success_rate >= 50.0:  # 50%以上で成功とみなす
                print("🔄 Google Sheetsを/releaseに自動更新中...")
                try:
                    update_result = subprocess.run(
                        [
                            "python3",
                            "tools/progress_tracker/cli.py",
                            "update",
                            tracker_id,
                            "/release",
                        ],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if update_result.returncode == 0:
                        print(f"✅ Google Sheets自動更新成功: {tracker_id} -> /release")
                    else:
                        print(f"⚠️ Google Sheets更新失敗: {update_result.stderr}")

                except Exception as e:
                    print(f"❌ Google Sheets更新エラー: {e}")
            else:
                print(f"⚠️ 成功率{success_rate:.1f}%のため、手動確認が必要です")

            return True

        else:
            # 失敗
            print(f"❌ 抽出処理失敗")
            print(f"📋 エラー出力: {result.stderr[:500]}...")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 抽出処理タイムアウト（30分）")
        return False
    except Exception as e:
        print(f"❌ 抽出処理エラー: {e}")
        return False


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-011実際の抽出パイプライン実行")
    parser.add_argument("--tracker_id", type=str, default="P1-011", help="トラッカーID")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 P1-011実際の抽出パイプライン実行")
    print("  - 実際のSAM+YOLO抽出処理")
    print("  - 完了後に自動/release更新")
    print("=" * 60)

    try:
        success = run_real_extraction(args.tracker_id)

        if success:
            print("\n🎉 P1-011実際の抽出パイプライン完了")
            return 0
        else:
            print("\n❌ P1-011抽出パイプライン失敗")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
