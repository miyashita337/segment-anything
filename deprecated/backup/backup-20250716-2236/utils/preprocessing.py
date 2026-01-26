#!/usr/bin/env python3
"""
Image Preprocessing Utilities
Extracted and modularized from original sam_yolo_character_segment.py
"""

import numpy as np
import cv2

from pathlib import Path
from typing import List, Optional, Tuple


def load_and_validate_image(image_path: str) -> Optional[np.ndarray]:
    """
    画像を読み込んで検証

    Args:
        image_path: 画像ファイルパス

    Returns:
        読み込んだ画像 (BGR format) または None
    """
    try:
        if not Path(image_path).exists():
            print(f"❌ 画像ファイルが見つかりません: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 画像の読み込みに失敗: {image_path}")
            return None

        # 画像サイズチェック
        height, width = image.shape[:2]
        if height < 32 or width < 32:
            print(f"❌ 画像サイズが小さすぎます: {width}x{height}")
            return None

        print(f"✅ 画像読み込み成功: {width}x{height}, {image_path}")
        return image

    except Exception as e:
        print(f"❌ 画像読み込みエラー: {e}")
        return None


def resize_image_if_needed(
    image: np.ndarray, max_size: int = 1024, min_size: int = 512
) -> Tuple[np.ndarray, float]:
    """
    必要に応じて画像をリサイズ

    Args:
        image: 入力画像
        max_size: 最大サイズ
        min_size: 最小サイズ

    Returns:
        リサイズ後の画像とスケール比
    """
    height, width = image.shape[:2]
    original_size = max(height, width)

    # リサイズが必要かチェック
    if original_size <= max_size and min(height, width) >= min_size:
        return image, 1.0

    # スケール計算
    if original_size > max_size:
        scale = max_size / original_size
    else:
        scale = min_size / min(height, width)

    # 新しいサイズ計算
    new_width = int(width * scale)
    new_height = int(height * scale)

    # リサイズ実行
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    print(f"🔄 画像リサイズ: {width}x{height} → {new_width}x{new_height} (scale={scale:.3f})")

    return resized, scale


def normalize_image_for_sam(image: np.ndarray) -> np.ndarray:
    """
    SAM用に画像を正規化 (BGR → RGB)

    Args:
        image: 入力画像 (BGR format)

    Returns:
        正規化された画像 (RGB format)
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    else:
        return image


def is_color_image(image: np.ndarray, threshold: float = 0.01) -> bool:
    """
    画像がカラー画像かどうかを判定

    Args:
        image: 入力画像 (BGR/RGB)
        threshold: カラー判定の閾値

    Returns:
        True if カラー画像, False if グレースケール画像
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False

    # RGBチャンネル間の差分を計算
    r, g, b = cv2.split(image)

    # 各チャンネル間の標準偏差を計算
    diff_rg = np.std(r.astype(np.float32) - g.astype(np.float32))
    diff_rb = np.std(r.astype(np.float32) - b.astype(np.float32))
    diff_gb = np.std(g.astype(np.float32) - b.astype(np.float32))

    # いずれかのチャンネル間差分が閾値を超えればカラー画像
    max_diff = max(diff_rg, diff_rb, diff_gb)
    is_color = max_diff > threshold

    print(f"カラー判定: 最大チャンネル差分={max_diff:.3f}, 閾値={threshold}, 結果={'カラー' if is_color else 'グレースケール'}")

    return is_color


def enhance_image_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    画像のコントラストを強化

    Args:
        image: 入力画像
        alpha: コントラスト係数
        beta: 明度調整値

    Returns:
        コントラスト強化後の画像
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    ガウシアンブラーを適用

    Args:
        image: 入力画像
        kernel_size: カーネルサイズ（奇数）

    Returns:
        ブラー適用後の画像
    """
    # kernel_sizeが偶数の場合は+1して奇数にする
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def detect_edges(
    image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
) -> np.ndarray:
    """
    エッジ検出

    Args:
        image: 入力画像
        low_threshold: 低閾値
        high_threshold: 高閾値

    Returns:
        エッジ画像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def crop_image_to_bbox(
    image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
) -> np.ndarray:
    """
    バウンディングボックスに基づいて画像をクロップ

    Args:
        image: 入力画像
        bbox: バウンディングボックス [x, y, width, height]
        padding: パディング値

    Returns:
        クロップされた画像
    """
    x, y, w, h = bbox
    height, width = image.shape[:2]

    # パディングを考慮した座標計算
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    cropped = image[y1:y2, x1:x2]
    return cropped


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    画像の統計情報を計算

    Args:
        image: 入力画像

    Returns:
        統計情報の辞書
    """
    height, width = image.shape[:2]

    stats = {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": height / width if width > 0 else 0,
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
    }

    # グレースケール変換して統計計算
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    stats.update(
        {
            "mean_brightness": np.mean(gray),
            "std_brightness": np.std(gray),
            "min_brightness": np.min(gray),
            "max_brightness": np.max(gray),
        }
    )

    return stats


def preprocess_image_pipeline(
    image_path: str, max_size: int = 1024, enhance_contrast: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    画像前処理パイプライン

    Args:
        image_path: 画像ファイルパス
        max_size: 最大サイズ
        enhance_contrast: コントラスト強化を行うか

    Returns:
        (処理済み画像BGR, 処理済み画像RGB, スケール比) のタプル
    """
    # 画像読み込み
    image_bgr = load_and_validate_image(image_path)
    if image_bgr is None:
        return None, None, 0.0

    # リサイズ
    image_bgr, scale = resize_image_if_needed(image_bgr, max_size=max_size)

    # コントラスト強化（オプション）
    if enhance_contrast:
        image_bgr = enhance_image_contrast(image_bgr)
        print("✨ コントラスト強化適用")

    # SAM用にRGBに変換
    image_rgb = normalize_image_for_sam(image_bgr)

    # 統計情報出力
    stats = calculate_image_statistics(image_bgr)
    color_type = "カラー" if is_color_image(image_bgr) else "グレースケール"
    print(
        f"📊 画像統計: {stats['width']}x{stats['height']}, "
        f"平均輝度: {stats['mean_brightness']:.1f}, {color_type}"
    )

    return image_bgr, image_rgb, scale


if __name__ == "__main__":
    # Test preprocessing functions
    test_image_path = "../assets/masks1.png"

    if Path(test_image_path).exists():
        print("🧪 Preprocessing test starting...")

        bgr_img, rgb_img, scale = preprocess_image_pipeline(test_image_path)

        if bgr_img is not None:
            print("✅ Preprocessing pipeline test successful")
            print(f"   Scale: {scale}")
            print(f"   BGR shape: {bgr_img.shape}")
            print(f"   RGB shape: {rgb_img.shape}")
        else:
            print("❌ Preprocessing pipeline test failed")
    else:
        print(f"⚠️ Test image not found: {test_image_path}")
        print("✅ Preprocessing module loaded successfully")
