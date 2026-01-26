#!/usr/bin/env python3
"""
Face Detection Fix Test
修正した顔検出ロジックをテスト
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

import logging
from features.evaluation.utils.face_detection import FaceDetector

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_face_detection_fix():
    """修正された顔検出ロジックをテスト"""
    logger.info("🔧 修正された顔検出ロジックのテスト開始")

    detector = FaceDetector()

    # テスト1: 顔が見える正面キャラクター（仮想画像）
    logger.info("\n📸 テスト1: 正面キャラクター（顔あり）")
    front_image = np.zeros((200, 150, 3), dtype=np.uint8)
    front_image[40:80, 60:90] = [200, 180, 160]  # 顔部分（肌色）
    front_image[80:180, 50:100] = [100, 150, 200]  # 体部分（服）

    # 仮想マスク作成
    front_mask = np.zeros((200, 150), dtype=np.uint8)
    front_mask[40:180, 50:100] = 255
    front_bbox = (50, 40, 50, 140)

    validation = detector.validate_character_mask(front_image, front_mask, front_bbox)
    logger.info(f"   顔検出: {validation['has_face']}")
    logger.info(f"   キャラクター判定: {validation['is_character']}")
    logger.info(f"   信頼度: {validation['confidence']:.3f}")

    # テスト2: 後ろ向きキャラクター（顔なし）
    logger.info("\n📸 テスト2: 後ろ向きキャラクター（顔なし）")
    back_image = np.zeros((200, 150, 3), dtype=np.uint8)
    back_image[50:180, 50:100] = [80, 120, 160]  # 後ろ姿（服のみ）

    # 仮想マスク作成
    back_mask = np.zeros((200, 150), dtype=np.uint8)
    back_mask[50:180, 50:100] = 255
    back_bbox = (50, 50, 50, 130)

    validation = detector.validate_character_mask(back_image, back_mask, back_bbox)
    logger.info(f"   顔検出: {validation['has_face']}")
    logger.info(f"   キャラクター判定: {validation['is_character']}")
    logger.info(f"   信頼度: {validation['confidence']:.3f}")

    # テスト3: 横長の非キャラクター（テキストや背景）
    logger.info("\n📸 テスト3: 横長の非キャラクター")
    wide_image = np.zeros((50, 200, 3), dtype=np.uint8)
    wide_image[10:40, 20:180] = [255, 255, 255]  # 横長の白い領域

    wide_mask = np.zeros((50, 200), dtype=np.uint8)
    wide_mask[10:40, 20:180] = 255
    wide_bbox = (20, 10, 160, 30)

    validation = detector.validate_character_mask(wide_image, wide_mask, wide_bbox)
    logger.info(f"   顔検出: {validation['has_face']}")
    logger.info(f"   キャラクター判定: {validation['is_character']}")
    logger.info(f"   信頼度: {validation['confidence']:.3f}")

    # テスト4: 小さすぎる領域
    logger.info("\n📸 テスト4: 小さすぎる領域")
    small_image = np.zeros((100, 100, 3), dtype=np.uint8)
    small_image[40:50, 45:55] = [150, 150, 150]

    small_mask = np.zeros((100, 100), dtype=np.uint8)
    small_mask[40:50, 45:55] = 255
    small_bbox = (45, 40, 10, 10)

    validation = detector.validate_character_mask(small_image, small_mask, small_bbox)
    logger.info(f"   顔検出: {validation['has_face']}")
    logger.info(f"   キャラクター判定: {validation['is_character']}")
    logger.info(f"   信頼度: {validation['confidence']:.3f}")

    logger.info("\n✅ 修正された顔検出ロジックのテスト完了")

    # 結果サマリー
    logger.info("\n📊 期待される結果:")
    logger.info("   テスト1（正面）: キャラクター=True, 高信頼度")
    logger.info("   テスト2（後ろ向き）: キャラクター=True, 中程度信頼度")
    logger.info("   テスト3（横長）: キャラクター=False, 低信頼度")
    logger.info("   テスト4（小さすぎ）: キャラクター=False, 低信頼度")


if __name__ == "__main__":
    test_face_detection_fix()
