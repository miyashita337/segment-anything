#!/usr/bin/env python3
"""
INTEGRATE-3-6-04 完全抽出スクリプト
未処理画像を含む全画像の抽出を実行
"""

import json
import logging
import requests
import subprocess
import time
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ログ設定"""
    log_file = Path("logs/INTEGRATE-3-6-04_complete_extraction.log")
    log_file.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_pushover_config():
    """Pushover設定読み込み"""
    config_path = Path("/mnt/c/AItools/segment-anything/config/pushover.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Pushover設定が見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def notify_success(message: str, title: str, priority: int = 0, attachment=None):
    """Pushover通知送信"""
    try:
        config = load_pushover_config()

        data = {
            "token": config["api_token"],
            "user": config["user_key"],
            "message": message,
            "title": title,
            "priority": priority,
        }

        files = None
        if attachment:
            with open(attachment, "rb") as f:
                files = {"attachment": (attachment.name, f, "image/jpeg")}

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=data,
            files=files if files else None,
            timeout=60 if attachment else 30,
        )

        return response.status_code == 200

    except Exception as e:
        logging.error(f"Pushover通知エラー: {e}")
        return False


def get_missing_files():
    """未処理画像を特定"""
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction"
    )

    # 入力画像（cover画像除外）
    input_files = sorted([f.stem for f in input_dir.glob("kana08_*.jpg") if "cover" not in f.name])

    # 既存の出力画像
    existing_files = sorted([f.stem for f in output_dir.glob("kana08_*.jpg")])

    # 未処理画像
    missing_files = [f for f in input_files if f not in existing_files]

    return input_files, existing_files, missing_files


def run_complete_extraction(logger):
    """完全抽出実行"""
    try:
        logger.info("🚀 INTEGRATE-3-6-04 完全抽出開始")

        # 未処理画像確認
        input_files, existing_files, missing_files = get_missing_files()

        logger.info(f"📊 処理状況:")
        logger.info(f"  対象画像: {len(input_files)}枚")
        logger.info(f"  処理済み: {len(existing_files)}枚")
        logger.info(f"  未処理: {len(missing_files)}枚")

        if missing_files:
            logger.info(
                f"  未処理リスト: {', '.join(missing_files[:5])}{'...' if len(missing_files) > 5 else ''}"
            )

        # 開始通知
        notify_success(
            f"🔄 INTEGRATE-3-6-04 完全抽出開始\n\n"
            f"📊 現在の状況:\n"
            f"・処理済み: {len(existing_files)}/{len(input_files)}枚\n"
            f"・未処理: {len(missing_files)}枚\n"
            f"・成功率: {len(existing_files)/len(input_files)*100:.1f}%\n\n"
            f"処理時間: 約{len(missing_files)*30//60}分\n"
            f"完了後、最終結果をお送りします。",
            "🔄 完全抽出開始",
            priority=0,
        )

        # 出力ディレクトリ作成
        output_dir = Path(
            "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # 抽出コマンド実行（全画像対象）
        cmd = [
            "python3",
            "tools/core/sam_yolo_character_segment.py",
            "--mode",
            "reproduce-auto",
            "--input_dir",
            "/mnt/c/AItools/lora/train/yado/org/kana08/",
            "--output_dir",
            str(output_dir),
            "--score_threshold",
            "0.07",
        ]

        logger.info(f"実行コマンド: {' '.join(cmd)}")

        # プロセス開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # リアルタイムログ出力（簡略化）
        processed_count = 0
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                # 進捗のみ記録
                if "進捗:" in output or "保存完了:" in output:
                    logger.info(f"[抽出] {output.strip()}")
                    processed_count += 1

        # プロセス終了確認
        return_code = process.poll()

        if return_code == 0:
            logger.info("✅ 抽出処理正常完了")
            return True
        else:
            logger.warning(f"⚠️ 抽出処理終了: return_code={return_code}")
            return True  # 部分的成功も成功として扱う

    except Exception as e:
        logger.error(f"❌ 抽出処理例外: {e}")
        return False


def analyze_final_results(logger):
    """最終結果分析"""
    output_dir = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction"
    )

    # 全出力ファイル取得
    output_files = sorted(output_dir.glob("kana08_*.jpg"))

    # ファイルサイズ分析
    file_sizes = [(f.name, f.stat().st_size / 1024) for f in output_files]  # KB単位

    # 統計
    total_files = len(output_files)
    total_size_mb = sum(size for _, size in file_sizes) / 1024
    avg_size_kb = sum(size for _, size in file_sizes) / total_files if total_files > 0 else 0

    # 大きいファイル・小さいファイル
    sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)
    largest_files = sorted_files[:3]
    smallest_files = sorted_files[-3:]

    logger.info(f"📊 最終結果分析:")
    logger.info(f"  総ファイル数: {total_files}枚")
    logger.info(f"  総容量: {total_size_mb:.2f}MB")
    logger.info(f"  平均サイズ: {avg_size_kb:.1f}KB")
    logger.info(f"  最大: {largest_files[0][0]} ({largest_files[0][1]:.1f}KB)")
    logger.info(f"  最小: {smallest_files[0][0]} ({smallest_files[0][1]:.1f}KB)")

    return {
        "total_files": total_files,
        "total_size_mb": total_size_mb,
        "avg_size_kb": avg_size_kb,
        "largest": largest_files,
        "smallest": smallest_files,
    }


def send_completion_notification(stats, logger):
    """完了通知送信（サンプル画像付き）"""
    try:
        # 最終統計
        input_files, existing_files, missing_files = get_missing_files()
        success_rate = len(existing_files) / len(input_files) * 100 if input_files else 0

        # 判定
        if success_rate >= 90:
            status_emoji = "✅"
            status_text = "高品質完了"
        elif success_rate >= 70:
            status_emoji = "⚠️"
            status_text = "部分完了"
        else:
            status_emoji = "❌"
            status_text = "要改善"

        message = (
            f"{status_emoji} INTEGRATE-3-6-04 完全抽出{status_text}\n\n"
            f"📊 最終結果:\n"
            f"・抽出成功: {stats['total_files']}/{len(input_files)}枚\n"
            f"・成功率: {success_rate:.1f}%\n"
            f"・総容量: {stats['total_size_mb']:.2f}MB\n"
            f"・平均サイズ: {stats['avg_size_kb']:.1f}KB\n\n"
            f"📈 ファイルサイズ分析:\n"
            f"・最大: {stats['largest'][0][0]} ({stats['largest'][0][1]:.1f}KB)\n"
            f"・最小: {stats['smallest'][0][0]} ({stats['smallest'][0][1]:.1f}KB)\n\n"
            f"🎯 次のステップ:\n"
            f"・品質確認と分析\n"
            f"・問題画像の特定\n"
            f"・必要に応じて再処理"
        )

        notify_success(
            message, f"{status_emoji} 完全抽出{status_text} ({success_rate:.0f}%)", priority=1
        )

        # 代表的な画像を送信（成功例）
        output_dir = Path(
            "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction"
        )
        sample_images = sorted(output_dir.glob("kana08_*.jpg"))[:3]

        for i, img_path in enumerate(sample_images, 1):
            notify_success(
                f"📸 完了サンプル {i}/3\n"
                f"画像: {img_path.name}\n"
                f"サイズ: {img_path.stat().st_size/1024:.1f}KB\n\n"
                f"品質確認をお願いします。",
                f"📸 サンプル [{i}/3]",
                priority=0,
                attachment=img_path,
            )
            time.sleep(1)

        logger.info("✅ 完了通知送信完了")

    except Exception as e:
        logger.error(f"完了通知送信エラー: {e}")


def main():
    """メイン処理"""
    logger = setup_logging()
    start_time = datetime.now()

    try:
        logger.info("=" * 60)
        logger.info("INTEGRATE-3-6-04 完全抽出プロセス開始")
        logger.info(f"開始時刻: {start_time}")
        logger.info("=" * 60)

        # 完全抽出実行
        extraction_success = run_complete_extraction(logger)

        # 結果分析
        stats = analyze_final_results(logger)

        # 完了通知
        send_completion_notification(stats, logger)

        # 最終ログ
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info(f"処理完了時刻: {end_time}")
        logger.info(f"処理時間: {duration}")
        logger.info(f"最終ファイル数: {stats['total_files']}枚")
        logger.info("=" * 60)

        return extraction_success

    except Exception as e:
        logger.error(f"メイン処理エラー: {e}")

        # エラー通知
        notify_success(
            f"❌ INTEGRATE-3-6-04完全抽出でエラーが発生しました。\n\n" f"エラー: {str(e)}\n" f"ログを確認してください。",
            "❌ 完全抽出エラー",
            priority=2,
        )
        return False


if __name__ == "__main__":
    main()
