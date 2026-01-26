#!/usr/bin/env python3
"""
PH2-008: 復旧機能強化システム実行スクリプト

目的: kana08データセットに対する復旧機能付き抽出実行
- 復旧機能システムによる自動リトライ・指数バックオフ
- 統合パイプラインによる段階的品質確認
- Pushover画像送信とダッシュボード生成
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.core.integrated_quality_pipeline import IntegratedQualityPipeline


def setup_ph2_008_logging() -> logging.Logger:
    """PH2-008専用ロギング設定"""
    logger = logging.getLogger("ph2_008_recovery")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # コンソール出力
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - PH2-008 - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ファイル出力
        log_file = Path("logs/PH2-008_recovery_extraction.log")
        log_file.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_ph2_008_prerequisites(logger: logging.Logger) -> bool:
    """PH2-008前提条件確認"""

    # 入力ディレクトリ確認（PH2-008はkana08データセット使用）
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08/")
    if not input_dir.exists():
        logger.error(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False

    # 画像ファイル確認
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if len(image_files) == 0:
        logger.error("❌ 入力ディレクトリに画像ファイルが存在しません")
        return False

    logger.info(f"✅ 入力確認完了: {len(image_files)}個の画像ファイル")

    # ワークスペースベースディレクトリ確認
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/")
    if not workspace_base.exists():
        workspace_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ ワークスペースベース作成: {workspace_base}")

    # 設定ファイル確認
    config_file = Path("config/pipeline_config.yaml")
    if not config_file.exists():
        logger.error(f"❌ 設定ファイルが存在しません: {config_file}")
        return False

    logger.info("✅ PH2-008前提条件確認完了")
    return True


def execute_ph2_008_recovery_pipeline(resume: bool = False, dry_run: bool = False) -> bool:
    """PH2-008復旧機能付きパイプライン実行"""

    logger = setup_ph2_008_logging()
    logger.info("🚀 PH2-008: 復旧機能強化システム開始")

    # 前提条件確認
    if not validate_ph2_008_prerequisites(logger):
        logger.error("❌ 前提条件確認失敗、処理を中断")
        return False

    if dry_run:
        logger.info("🔍 ドライランモード: 実際の処理は実行されません")
        return True

    try:
        # 統合パイプライン初期化
        config_path = "config/pipeline_config.yaml"
        tracker_id = "PH2-008"

        pipeline = IntegratedQualityPipeline(config_path, tracker_id)
        logger.info(f"✅ 統合パイプライン初期化完了: {tracker_id}")

        # 復旧機能付きパイプライン実行
        logger.info("🔄 復旧機能付きパイプライン実行開始")
        result = pipeline.execute_pipeline_with_recovery(resume=resume, max_retries=3)

        if result.success:
            logger.info(f"✅ PH2-008パイプライン実行成功: {result.total_duration_seconds:.1f}秒")

            # 成果確認
            workspace_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-008/")
            extraction_dir = workspace_dir / "extraction"
            dashboard_dir = workspace_dir / "dashboard"

            # 抽出結果確認
            if extraction_dir.exists():
                extracted_files = list(extraction_dir.glob("*.jpg")) + list(
                    extraction_dir.glob("*.png")
                )
                logger.info(f"📊 抽出結果: {len(extracted_files)}個のファイル")

            # ダッシュボード確認
            if dashboard_dir.exists() and (dashboard_dir / "dashboard.html").exists():
                logger.info(f"📈 ダッシュボード生成完了: {dashboard_dir}/dashboard.html")

            logger.info("🎉 PH2-008: 復旧機能強化システム完了")
            return True
        else:
            logger.error("❌ PH2-008パイプライン実行失敗")
            return False

    except Exception as e:
        logger.error(f"❌ PH2-008実行中エラー: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="PH2-008: 復旧機能強化システム実行")

    parser.add_argument("--resume", action="store_true", help="中断からの再開実行")

    parser.add_argument("--dry-run", action="store_true", help="ドライランモード（実際の処理は実行しない）")

    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # PH2-008実行
    success = execute_ph2_008_recovery_pipeline(resume=args.resume, dry_run=args.dry_run)

    if success:
        print("✅ PH2-008: 復旧機能強化システム実行完了")
        sys.exit(0)
    else:
        print("❌ PH2-008: 復旧機能強化システム実行失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
