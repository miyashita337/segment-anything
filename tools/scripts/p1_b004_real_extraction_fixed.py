#!/usr/bin/env python3
"""
P1-B004実際の抽出パイプライン実行（環境修正版）
adaptive-cropping機能を有効化して実際の画像生成
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_real_extraction_fixed(tracker_id: str = "P1-B004"):
    """P1-B004の実際の抽出パイプライン実行（修正版）"""
    print(f"🚀 {tracker_id} 実際の抽出パイプライン開始（adaptive-cropping有効）")

    # 入力・出力ディレクトリ設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")

    # 入力ディレクトリ確認
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False

    # 画像数確認
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"📊 入力画像数: {len(image_files)}枚")

    if not image_files:
        print("❌ 入力ディレクトリに画像ファイルがありません")
        return False

    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)

    # sam-env環境のPythonパス
    sam_python = project_root / "sam-env" / "Scripts" / "python.exe"
    if not sam_python.exists():
        print(f"❌ sam-env Python not found: {sam_python}")
        return False

    try:
        # P1-B004 adaptive-cropping付き抽出実行
        print("🔧 P1-B004 adaptive-cropping付き抽出実行開始")
        start_time = time.time()

        # 最初の3枚でテスト（高速化）
        command = [
            str(sam_python),
            "features/extraction/commands/extract_character.py",
            "--batch",
            str(input_dir),
            "-o",
            str(output_dir),
            "--adaptive-cropping",  # P1-B004の機能を有効化
            "--max-files",
            "3",  # テスト用に3枚限定
            "--verbose",
        ]

        print(f"💻 実行コマンド: {' '.join(command)}")
        print(f"🌏 Python環境: {sam_python}")
        print(f"📂 入力: {input_dir}")
        print(f"📁 出力: {output_dir}")

        # 実行
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10分タイムアウト
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"⏱️ 処理時間: {processing_time:.1f}秒")
        print(f"📊 実行結果: returncode={result.returncode}")

        if result.stdout:
            print("📋 標準出力:")
            print(result.stdout[-1000:])  # 最後の1000文字

        if result.stderr:
            print("⚠️ エラー出力:")
            print(result.stderr[-1000:])  # 最後の1000文字

        # 出力ファイル確認
        output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        print(f"📁 出力ファイル数: {len(output_files)}個")

        if output_files:
            print("✅ 生成ファイル:")
            for f in output_files[:5]:  # 最初の5個表示
                print(f"  - {f.name}")

        # 成功率計算
        test_count = min(3, len(image_files))
        success_rate = (len(output_files) / test_count) * 100 if test_count > 0 else 0
        print(f"📈 抽出成功率: {success_rate:.1f}% ({len(output_files)}/{test_count})")

        # extraction_report.json生成
        report = {
            "tracker_id": tracker_id,
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_count": test_count,
            "output_count": len(output_files),
            "success_rate": success_rate,
            "processing_time": processing_time,
            "adaptive_cropping": True,
            "returncode": result.returncode,
            "files": [str(f.name) for f in output_files],
            "command": " ".join(command),
            "python_env": str(sam_python),
        }

        report_path = output_dir.parent / "extraction_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 レポート生成: {report_path}")

        if len(output_files) > 0:
            print("✅ 抽出処理成功: 画像ファイルが生成されました")
            return True
        else:
            print("❌ 抽出処理失敗: 画像ファイルが生成されませんでした")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 抽出処理タイムアウト（10分）")
        return False
    except Exception as e:
        print(f"❌ 抽出処理エラー: {e}")
        return False


def main():
    """メイン実行"""
    print("=" * 60)
    print("🎯 P1-B004実際の抽出パイプライン実行（修正版）")
    print("  - sam-env環境使用")
    print("  - adaptive-cropping機能有効")
    print("  - 実際の画像ファイル生成")
    print("=" * 60)

    try:
        success = run_real_extraction_fixed("P1-B004")

        if success:
            print("\n🎉 P1-B004実際の抽出パイプライン完了")
            return 0
        else:
            print("\n❌ P1-B004抽出パイプライン失敗")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
