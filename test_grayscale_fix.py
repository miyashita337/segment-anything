#!/usr/bin/env python3
"""
YOLO拡張システム グレースケール対応修正テスト
"""

import numpy as np
import cv2

import logging
import os
import sys
from pathlib import Path

# プロジェクトルートを追加
sys.path.append("/mnt/c/AItools/segment-anything")

# 環境変数設定
os.environ["PYTHONPATH"] = "/mnt/c/AItools/segment-anything"

try:
    from features.extraction.yolo_detection_expansion import YOLODetectionExpander

    print("✅ モジュールインポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)


def test_grayscale_anime_scoring():
    """グレースケール画像でのアニメスコア計算テスト"""
    print("\n🧪 グレースケール対応アニメスコア計算テスト")

    # ログ設定
    logging.basicConfig(level=logging.DEBUG)

    # YOLO拡張システム初期化
    expansion_system = YOLODetectionExpander()

    # テスト用グレースケール画像作成
    test_images = [
        # 1. シンプルなグレースケール矩形
        np.full((400, 300, 3), 128, dtype=np.uint8),  # 中間グレー
        # 2. 明暗がはっきりしたアニメ風パターン
        create_anime_like_grayscale(),
        # 3. ノイジーなテクスチャ
        create_noisy_grayscale(),
    ]

    for i, test_image in enumerate(test_images):
        print(f"\n--- テスト画像 {i+1} ---")

        # モックマスク作成（中央部分）
        h, w = test_image.shape[:2]
        mock_segmentation = np.zeros((h, w), dtype=np.uint8)
        mock_segmentation[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1

        mock_mask = {
            "segmentation": mock_segmentation,
            "bbox": [w // 4, h // 4, w // 2, h // 2],
            "area": np.sum(mock_segmentation),
            "score": 0.8,
        }

        # アニメスコア計算
        try:
            anime_score = expansion_system._calculate_anime_character_score(mock_mask, test_image)
            print(f"🎨 アニメスコア: {anime_score:.3f}")

            # グレースケール判定確認
            hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            global_saturation = np.mean(hsv_image[:, :, 1]) / 255.0
            is_grayscale = global_saturation < 0.15
            print(f"🔍 グレースケール判定: {is_grayscale} (彩度: {global_saturation:.3f})")

        except Exception as e:
            print(f"❌ エラー: {e}")


def create_anime_like_grayscale():
    """アニメ風グレースケール画像作成"""
    image = np.full((400, 300, 3), 200, dtype=np.uint8)  # 明るいベース

    # 暗い影部分
    cv2.rectangle(image, (50, 50), (150, 350), (80, 80, 80), -1)
    # 中間調
    cv2.rectangle(image, (100, 100), (200, 300), (140, 140, 140), -1)
    # ハイライト
    cv2.circle(image, (125, 150), 30, (220, 220, 220), -1)

    return image


def create_noisy_grayscale():
    """ノイジーなグレースケール画像作成"""
    image = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
    # BGR全チャンネルを同じ値にしてグレースケール化
    gray_values = image[:, :, 0]
    image[:, :, 0] = gray_values
    image[:, :, 1] = gray_values
    image[:, :, 2] = gray_values
    return image


def test_anime_filtering():
    """アニメフィルタリング統合テスト"""
    print("\n🎛️ アニメフィルタリング統合テスト")

    # YOLO拡張システム初期化
    expansion_system = YOLODetectionExpander()

    # テスト用グレースケール画像
    test_image = create_anime_like_grayscale()

    # モックマスク候補作成（複数）
    masks = []
    for i in range(3):
        h, w = test_image.shape[:2]
        mock_segmentation = np.zeros((h, w), dtype=np.uint8)
        start_x = i * 80 + 50
        mock_segmentation[100:300, start_x : start_x + 60] = 1

        mock_mask = {
            "segmentation": mock_segmentation,
            "bbox": [start_x, 100, 60, 200],
            "area": np.sum(mock_segmentation),
            "score": 0.7 + i * 0.1,
            "class_id": 0,  # person
        }
        masks.append(mock_mask)

    print(f"📊 入力マスク数: {len(masks)}")

    # アニメフィルタリング実行
    try:
        filtered_masks = expansion_system._apply_anime_character_filter(masks, test_image)
        print(f"✅ フィルタ後マスク数: {len(filtered_masks)}")

        for i, mask in enumerate(filtered_masks):
            score = mask.get("anime_character_score", 0)
            print(f"  マスク{i+1}: アニメスコア {score:.3f}")

    except Exception as e:
        print(f"❌ フィルタリングエラー: {e}")


if __name__ == "__main__":
    print("🔧 YOLO拡張システム グレースケール対応修正テスト")

    test_grayscale_anime_scoring()
    test_anime_filtering()

    print("\n✅ テスト完了")
