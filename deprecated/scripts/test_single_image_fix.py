#!/usr/bin/env python3
"""
Single Image Test for Face Detection Fix
実際の画像で修正された顔検出ロジックをテスト
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

import logging
from features.extraction.robust_extractor import RobustCharacterExtractor

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_image():
    """単一画像で修正ロジックをテスト"""
    logger.info("🔧 修正されたキャラクター抽出ロジック - 単一画像テスト")

    # テスト対象画像
    input_path = Path("/mnt/c/AItools/lora/train/yado/org/kaname08/kaname08_0009.jpg")
    output_path = Path("/mnt/c/AItools/segment-anything/test_fix_result.jpg")

    if not input_path.exists():
        logger.error(f"テスト画像が見つかりません: {input_path}")
        return

    try:
        # RobustCharacterExtractorでテスト
        extractor = RobustCharacterExtractor()

        logger.info(f"📸 テスト画像: {input_path.name}")
        logger.info(f"📂 出力先: {output_path}")

        # 修正された抽出実行
        result = extractor.extract_character_robust(input_path, output_path, verbose=True)

        logger.info("🎯 テスト結果:")
        logger.info(f"   成功: {result.get('success', False)}")
        logger.info(f"   品質スコア: {result.get('quality_score', 0.0):.3f}")
        logger.info(f"   使用手法: {result.get('best_method', 'unknown')}")

        if result.get("success", False) and output_path.exists():
            logger.info(f"✅ 抽出成功 - 結果: {output_path}")
        else:
            logger.warning("⚠️ 抽出失敗")

    except Exception as e:
        logger.error(f"❌ テスト実行エラー: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    test_single_image()
