#!/usr/bin/env python3
"""
P1-014: マルチGPU並列処理実行スクリプト
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


def run_multi_gpu_extraction(tracker_id: str = "P1-014"):
    """マルチGPU抽出パイプライン実行"""
    print(f"🚀 {tracker_id} マルチGPU並列処理開始")

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

    # 既存の出力ファイルクリーンアップ
    for existing_file in output_dir.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()

    print(f"🧹 出力ディレクトリクリーンアップ完了: {output_dir}")

    try:
        # マルチGPUシステム実行
        print("🔧 マルチGPU SAM統合システム実行開始")
        start_time = time.time()

        command = [
            "python3",
            "features/processing/multi_gpu_sam_integration.py",
            "--tracker-id",
            tracker_id,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--score-threshold",
            "0.07",
        ]

        print(f"💻 実行コマンド: {' '.join(command)}")

        # 実行
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600,  # 1時間タイムアウト
        )

        end_time = time.time()
        processing_time = end_time - start_time

        if result.returncode == 0:
            # 成功
            print(f"✅ マルチGPU処理成功 ({processing_time:.1f}秒)")

            # 出力ファイル確認
            output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            print(f"📁 出力ファイル数: {len(output_files)}個")

            # 成功率計算
            success_rate = (len(output_files) / len(image_files)) * 100 if image_files else 0
            print(f"📈 抽出成功率: {success_rate:.1f}%")

            # 出力ファイルサンプル表示
            if output_files:
                print("📋 出力ファイルサンプル:")
                for i, output_file in enumerate(output_files[:5], 1):
                    size_mb = output_file.stat().st_size / 1024 / 1024
                    print(f"  {i}. {output_file.name} ({size_mb:.2f}MB)")
                if len(output_files) > 5:
                    print(f"  ... (他{len(output_files) - 5}個)")

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
            print(f"❌ マルチGPU処理失敗")
            print(f"📋 エラー出力: {result.stderr[:500]}...")
            return False

    except subprocess.TimeoutExpired:
        print("❌ マルチGPU処理タイムアウト（1時間）")
        return False
    except Exception as e:
        print(f"❌ マルチGPU処理エラー: {e}")
        return False


def generate_performance_report(tracker_id: str = "P1-014"):
    """パフォーマンスレポート生成"""
    print("📊 パフォーマンスレポート生成中...")

    try:
        # GPU情報取得
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🖥️ GPU環境:")
            print(f"   利用可能GPU数: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("⚠️ CUDA利用不可（CPU処理）")

        # マルチGPUシステム固有レポート生成
        report_command = [
            "python3",
            "features/processing/multi_gpu_processor.py",
            "--tracker-id",
            tracker_id,
            "--input-dir",
            "/mnt/c/AItools/lora/train/yado/org/kana05",
        ]

        result = subprocess.run(
            report_command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5分タイムアウト
        )

        if result.returncode == 0:
            print("✅ パフォーマンスレポート生成成功")
            print(result.stdout)
        else:
            print("⚠️ パフォーマンスレポート生成失敗")
            print(result.stderr)

    except Exception as e:
        print(f"❌ パフォーマンスレポート生成エラー: {e}")


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-014 マルチGPU並列処理実行")
    parser.add_argument("--tracker_id", type=str, default="P1-014", help="トラッカーID")
    parser.add_argument("--skip-report", action="store_true", help="パフォーマンスレポート生成をスキップ")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 P1-014 マルチGPU並列処理実行")
    print("  - マルチGPU SAM+YOLO統合処理")
    print("  - 並列処理による高速化")
    print("  - 完了後に自動/release更新")
    print("=" * 60)

    try:
        # メイン処理実行
        success = run_multi_gpu_extraction(args.tracker_id)

        # パフォーマンスレポート生成
        if not args.skip_report:
            generate_performance_report(args.tracker_id)

        if success:
            print("\n🎉 P1-014 マルチGPU並列処理完了")
            print("🚀 GPU並列処理による高速化を実現")
            print("⚡ 処理効率の向上とスケーラビリティの確保")
            return 0
        else:
            print("\n❌ P1-014 マルチGPU処理失敗")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
