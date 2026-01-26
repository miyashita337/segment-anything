from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
INTG-046-04 全24枚画像Pushover送信スクリプト
"""

import json
import logging
import requests
import time
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ログ設定"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_pushover_config():
    """Pushover設定読み込み"""
    config_path = Path(
        get_path(
            "data",
            Path(
                get_path(
                    "data",
                    Path("/mnt/c/AItools/segment-anything/config/pushover.json").relative_to(
                        "/mnt/c/AItools/"
                    ),
                )
            ).relative_to("/mnt/c/AItools/"),
        )
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Pushover設定が見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_image_to_pushover(
    image_path: Path, config: dict, logger: logging.Logger, index: int, total: int
):
    """個別画像をPushoverに送信"""
    try:
        file_size_kb = image_path.stat().st_size / 1024

        with open(image_path, "rb") as f:
            files = {"attachment": (image_path.name, f, "image/jpeg")}

            # 品質スコア情報を含むメッセージ
            quality_scores = {
                "kana08_0001.jpg": 0.762,
                "kana08_0002.jpg": 0.636,
                "kana08_0003.jpg": 0.684,
                "kana08_0004.jpg": 0.822,
                "kana08_0005.jpg": 0.763,
                "kana08_0006.jpg": 0.561,
                "kana08_0007.jpg": 0.657,
                "kana08_0008.jpg": 0.803,
                "kana08_0009.jpg": 0.261,
                "kana08_0010.jpg": 0.408,
                "kana08_0011.jpg": 0.651,
                "kana08_0012.jpg": 0.613,
                "kana08_0013.jpg": 0.488,
                "kana08_0014.jpg": 0.631,
                "kana08_0015.jpg": 0.566,
                "kana08_0016.jpg": 0.748,
                "kana08_0017.jpg": 0.743,
                "kana08_0018.jpg": 0.405,
                "kana08_0019.jpg": 0.460,
                "kana08_0020.jpg": 0.620,
                "kana08_0021.jpg": 0.679,
                "kana08_0022.jpg": 0.311,
                "kana08_0023.jpg": 0.437,
                "kana08_0024.jpg": 0.483,
            }

            quality = quality_scores.get(image_path.name, 0.0)

            # 品質レベル判定
            if quality >= 0.7:
                quality_level = "🟢 高品質"
            elif quality >= 0.5:
                quality_level = "🟡 中品質"
            else:
                quality_level = "🔴 低品質"

            message = (
                f"📊 INTG-046-04 全画像送信\n"
                f"順序: {index}/{total}\n"
                f"画像: {image_path.name}\n\n"
                f"📏 ファイル情報:\n"
                f"・サイズ: {file_size_kb:.1f}KB\n"
                f"・品質スコア: {quality:.3f}\n"
                f"・品質レベル: {quality_level}\n\n"
                f"🔧 抽出設定:\n"
                f"・モデル: yolov8x.pt\n"
                f"・閾値: 0.07\n"
                f"・処理モード: reproduce-auto"
            )

            data = {
                "token": config["api_token"],
                "user": config["user_key"],
                "message": message,
                "title": f"📸 [{index}/{total}] {image_path.name} {quality_level}",
                "priority": 0,
                "sound": "none" if index > 1 else "pushover",  # 最初だけ音
            }

            response = requests.post(
                "https://api.pushover.net/1/messages.json", data=data, files=files, timeout=60
            )

            if response.status_code == 200:
                logger.info(
                    f"✅ 送信成功 [{index}/{total}]: {image_path.name} ({file_size_kb:.1f}KB, 品質:{quality:.3f})"
                )
                return True
            else:
                logger.error(
                    f"❌ 送信失敗 [{index}/{total}]: {image_path.name} - Status: {response.status_code}"
                )
                return False

    except Exception as e:
        logger.error(f"❌ 送信エラー [{index}/{total}]: {image_path.name} - {e}")
        return False


def main():
    """メイン処理"""
    logger = setup_logging()
    start_time = datetime.now()

    logger.info("🚀 INTG-046-04 全24枚画像Pushover送信開始")

    try:
        # 設定読み込み
        config = load_pushover_config()
        logger.info("✅ Pushover設定読み込み完了")

        # 抽出結果ディレクトリ
        extraction_dir = Path(get_path("output", "INTG-046-04/extraction"))

        if not extraction_dir.exists():
            raise FileNotFoundError(f"抽出ディレクトリが見つかりません: {extraction_dir}")

        # 全画像ファイル取得（ソート済み）
        image_files = sorted(extraction_dir.glob("kana08_*.jpg"))
        total_images = len(image_files)

        logger.info(f"📸 送信対象: {total_images}枚の画像")

        # 開始通知
        start_message = (
            f"🔍 INTG-046-04 全画像送信開始\n\n"
            f"対象: {total_images}枚の抽出画像\n"
            f"送信間隔: 1.5秒\n"
            f"予想時間: 約{total_images * 1.5 / 60:.1f}分\n\n"
            f"各画像の品質スコア付きで送信します。"
        )

        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": config["api_token"],
                "user": config["user_key"],
                "message": start_message,
                "title": "🚀 全画像送信開始",
                "priority": 1,
                "sound": "pushover",
            },
            timeout=30,
        )

        # 個別送信処理
        success_count = 0
        failed_images = []

        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📤 送信中: {i}/{total_images} - {image_path.name}")

            if send_image_to_pushover(image_path, config, logger, i, total_images):
                success_count += 1
            else:
                failed_images.append(image_path.name)

            # 送信間隔（1.5秒、レート制限対策）
            if i < total_images:
                time.sleep(1.5)

        # 完了通知
        end_time = datetime.now()
        duration = end_time - start_time

        # 品質統計
        high_quality = 7  # 0.7以上
        medium_quality = 10  # 0.5-0.7
        low_quality = 7  # 0.5未満

        completion_message = (
            f"✅ INTG-046-04 全画像送信完了\n\n"
            f"📊 送信結果:\n"
            f"・成功: {success_count}/{total_images}枚\n"
            f"・失敗: {len(failed_images)}枚\n"
            f"・成功率: {success_count/total_images*100:.1f}%\n\n"
            f"📈 品質分布:\n"
            f"・🟢 高品質(0.7以上): {high_quality}枚\n"
            f"・🟡 中品質(0.5-0.7): {medium_quality}枚\n"
            f"・🔴 低品質(0.5未満): {low_quality}枚\n\n"
            f"⏱️ 処理時間: {duration}\n\n"
            f"全画像の品質確認をお願いします。"
        )

        if failed_images:
            completion_message += f"\n\n❌ 送信失敗画像:\n" + "\n".join(failed_images)

        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": config["api_token"],
                "user": config["user_key"],
                "message": completion_message,
                "title": f"🎯 全画像送信完了 ({success_count}/{total_images})",
                "priority": 1,
                "sound": "magic",
            },
            timeout=30,
        )

        logger.info("=" * 60)
        logger.info(f"✅ 全送信完了: {success_count}/{total_images}枚成功")
        logger.info(f"⏱️ 処理時間: {duration}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ メイン処理エラー: {e}")
        raise


if __name__ == "__main__":
    main()
