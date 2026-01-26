#!/usr/bin/env python3
"""
QI-004ダッシュボード生成スクリプト

QI-004要件:
- ダッシュボード標準化システム実装
- Base64画像表示システム構築
- 画像パス参照方式の最適化
- パフォーマンス最適化とUI改善

使用方法:
    python tools/scripts/qi004_dashboard_generator.py
    python tools/scripts/qi004_dashboard_generator.py --input_dir custom_dir --output_dir custom_output
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.evaluation.qi004_dashboard_optimization_system import (
    QI004DashboardOptimizationSystem,
    create_qi004_optimized_dashboard,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="QI-004ダッシュボード生成スクリプト")

    parser.add_argument("--tracker_id", default="QI-004", help="トラッカーID（デフォルト: QI-004）")

    parser.add_argument(
        "--input_dir",
        default="/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-004/extraction",
        help="抽出画像ディレクトリ",
    )

    parser.add_argument(
        "--output_dir",
        default="/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-004",
        help="出力ディレクトリ",
    )

    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")

    parser.add_argument("--wait_for_extraction", action="store_true", help="抽出完了まで待機")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"🎯 QI-004ダッシュボード生成開始")
    logger.info(f"   トラッカーID: {args.tracker_id}")
    logger.info(f"   入力ディレクトリ: {args.input_dir}")
    logger.info(f"   出力ディレクトリ: {args.output_dir}")

    # 入力ディレクトリ存在チェック
    if not os.path.exists(args.input_dir):
        if args.wait_for_extraction:
            logger.info(f"⏳ 抽出ディレクトリ作成待機中: {args.input_dir}")
            import time

            # 最大5分間待機
            for i in range(300):
                if os.path.exists(args.input_dir):
                    logger.info(f"✅ 抽出ディレクトリ確認: {args.input_dir}")
                    break
                time.sleep(1)
                if i % 30 == 0:  # 30秒ごとにログ
                    logger.info(f"   待機中... ({i}秒経過)")
            else:
                logger.error(f"❌ タイムアウト: 抽出ディレクトリが見つかりません: {args.input_dir}")
                return False
        else:
            logger.error(f"❌ 入力ディレクトリが存在しません: {args.input_dir}")
            return False

    # 画像ファイル存在チェック
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = []

    if os.path.exists(args.input_dir):
        for file in os.listdir(args.input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

    if not image_files:
        if args.wait_for_extraction:
            logger.info(f"⏳ 抽出画像生成待機中...")
            import time

            # 最大10分間待機
            for i in range(600):
                if os.path.exists(args.input_dir):
                    current_files = [
                        f
                        for f in os.listdir(args.input_dir)
                        if any(f.lower().endswith(ext) for ext in image_extensions)
                    ]
                    if current_files:
                        image_files = current_files
                        logger.info(f"✅ 抽出画像確認: {len(image_files)}枚")
                        break

                time.sleep(2)
                if i % 30 == 0:  # 60秒ごとにログ
                    logger.info(f"   待機中... ({i*2}秒経過)")
            else:
                logger.warning(f"⚠️ タイムアウト: 抽出画像が見つかりません")
                # 空でも続行（テスト用ダッシュボード生成）
        else:
            logger.warning(f"⚠️ 抽出画像が見つかりません: {args.input_dir}")
            # 空でも続行（テスト用ダッシュボード生成）

    try:
        # QI-004最適化ダッシュボード生成
        logger.info(f"🔄 QI-004最適化プロセス開始...")

        success = create_qi004_optimized_dashboard(args.tracker_id, args.input_dir, args.output_dir)

        if success:
            logger.info(f"✅ QI-004ダッシュボード生成完了")

            # 生成されたファイル確認
            dashboard_path = os.path.join(args.output_dir, "dashboard", "dashboard.html")
            if os.path.exists(dashboard_path):
                file_size = os.path.getsize(dashboard_path) / (1024 * 1024)
                logger.info(f"   📄 ダッシュボードファイル: {dashboard_path}")
                logger.info(f"   📊 ダッシュボードサイズ: {file_size:.2f}MB")
                logger.info(f"   🌐 アクセスURL: http://100.123.241.106:8088/tracker/{args.tracker_id}")

                # ダッシュボード内容の簡易検証
                with open(dashboard_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # QI-004要件確認
                qi004_features = {
                    "Base64画像埋め込み": "data:image/jpeg;base64," in content,
                    "品質バッジシステム": "quality-badge-" in content,
                    "Tailwind CSS": "tailwindcss.com" in content,
                    "レスポンシブデザイン": "grid-cols-1 md:grid-cols-2 lg:grid-cols-3" in content,
                    "実ファイル名表示": "kana" in content or len(image_files) == 0,  # 画像がない場合はスキップ
                }

                logger.info(f"   🎯 QI-004要件検証:")
                for feature, status in qi004_features.items():
                    status_icon = "✅" if status else "❌"
                    logger.info(f"      {status_icon} {feature}")

                return True
            else:
                logger.error(f"❌ ダッシュボードファイルが生成されませんでした")
                return False
        else:
            logger.error(f"❌ QI-004ダッシュボード生成失敗")
            return False

    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
