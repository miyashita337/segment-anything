#!/usr/bin/env python3
"""
Post-processing Utilities
Mask refinement and character extraction post-processing
"""

import numpy as np
import cv2

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def refine_mask_edges(mask: np.ndarray, 
                     kernel_size: int = 3,
                     iterations: int = 1) -> np.ndarray:
    """
    マスクのエッジを滑らかにする
    
    Args:
        mask: 入力マスク (0-255)
        kernel_size: モルフォロジカル処理のカーネルサイズ
        iterations: 処理の反復回数
        
    Returns:
        処理済みマスク
    """
    # カーネル作成
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # ノイズ除去（opening）
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # ホール埋め（closing）
    refined = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return refined


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    小さな連結成分を除去
    
    Args:
        mask: 入力マスク (0-255)
        min_area: 最小面積
        
    Returns:
        処理済みマスク
    """
    # 連結成分分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 新しいマスクを作成
    cleaned_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # 0はバックグラウンド
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask


def fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    """
    マスク内のホールを埋める
    
    Args:
        mask: 入力マスク (0-255)
        
    Returns:
        ホール埋め済みマスク
    """
    # 輪郭を見つける
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # すべての外側輪郭を塗りつぶし
    filled_mask = np.zeros_like(mask)
    for contour in contours:
        cv2.fillPoly(filled_mask, [contour], 255)
    
    return filled_mask


def apply_gaussian_smoothing_to_mask(mask: np.ndarray, 
                                   kernel_size: int = 5,
                                   threshold: int = 127) -> np.ndarray:
    """
    マスクにガウシアン平滑化を適用
    
    Args:
        mask: 入力マスク (0-255)
        kernel_size: ガウシアンカーネルサイズ
        threshold: 二値化閾値
        
    Returns:
        平滑化済みマスク
    """
    # kernel_sizeを奇数にする
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # ガウシアンフィルタ適用
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    # 二値化
    _, smoothed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    return smoothed


def extract_character_from_image(image: np.ndarray, 
                               mask: np.ndarray,
                               background_color: Tuple[int, int, int] = (0, 0, 0),
                               padding: int = 10) -> np.ndarray:
    """
    マスクを使用してキャラクターを抽出
    
    Args:
        image: 元画像 (BGR)
        mask: キャラクターマスク (0-255)
        background_color: 背景色 (B, G, R)
        padding: パディング
        
    Returns:
        抽出されたキャラクター画像
    """
    # マスクを3チャンネルに変換
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
    
    # マスクの正規化
    mask_normalized = mask_3ch.astype(np.float32) / 255.0
    
    # 背景色の画像を作成
    background = np.full_like(image, background_color, dtype=np.uint8)
    
    # マスクを使って合成
    result = image.astype(np.float32) * mask_normalized + background.astype(np.float32) * (1.0 - mask_normalized)
    result = result.astype(np.uint8)
    
    return result


def crop_to_content(image: np.ndarray, 
                   mask: np.ndarray,
                   padding: int = 10) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    コンテンツ領域にクロップ
    
    Args:
        image: 元画像
        mask: マスク
        padding: パディング
        
    Returns:
        (クロップ画像, クロップマスク, バウンディングボックス)
    """
    # マスクから輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, mask, (0, 0, image.shape[1], image.shape[0])
    
    # 最大の輪郭のバウンディングボックスを取得
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # パディングを追加
    height, width = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # クロップ
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    return cropped_image, cropped_mask, bbox


def create_transparent_character(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    透明背景のキャラクター画像を作成
    
    Args:
        image: 元画像 (BGR)
        mask: キャラクターマスク (0-255)
        
    Returns:
        透明背景画像 (BGRA)
    """
    # BGRAフォーマットに変換
    bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # マスクをアルファチャンネルとして使用
    bgra_image[:, :, 3] = mask
    
    return bgra_image


def enhance_character_mask(mask: np.ndarray,
                         remove_small_area: int = 100,
                         smooth_kernel: int = 3,
                         fill_holes: bool = True) -> np.ndarray:
    """
    キャラクターマスクの総合的な強化
    
    Args:
        mask: 入力マスク (0-255)
        remove_small_area: 除去する小領域の最小面積
        smooth_kernel: スムージングカーネルサイズ
        fill_holes: ホール埋めを行うか
        
    Returns:
        強化されたマスク
    """
    enhanced = mask.copy()
    
    # 小さな成分を除去
    if remove_small_area > 0:
        enhanced = remove_small_components(enhanced, min_area=remove_small_area)
    
    # ホール埋め
    if fill_holes:
        enhanced = fill_holes_in_mask(enhanced)
    
    # エッジ滑らかに
    if smooth_kernel > 0:
        enhanced = refine_mask_edges(enhanced, kernel_size=smooth_kernel)
    
    return enhanced


def calculate_mask_quality_metrics(mask: np.ndarray) -> Dict[str, float]:
    """
    マスクの品質メトリクスを計算
    
    Args:
        mask: 入力マスク (0-255)
        
    Returns:
        品質メトリクス辞書
    """
    height, width = mask.shape[:2]
    total_pixels = height * width
    
    # 基本統計
    mask_pixels = np.sum(mask > 0)
    coverage_ratio = mask_pixels / total_pixels
    
    # 輪郭分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
        
        # コンパクトネス (円形度)
        compactness = (4 * np.pi * contour_area) / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
        
        # バウンディングボックス分析
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        aspect_ratio = h / w if w > 0 else 0
        
    else:
        compactness = 0
        fill_ratio = 0
        aspect_ratio = 0
        contour_area = 0
    
    return {
        'coverage_ratio': coverage_ratio,
        'compactness': compactness,
        'fill_ratio': fill_ratio,
        'aspect_ratio': aspect_ratio,
        'contour_area': contour_area,
        'mask_pixels': mask_pixels,
        'total_pixels': total_pixels
    }


def save_character_result(image: np.ndarray,
                        mask: np.ndarray,
                        output_path: str,
                        save_mask: bool = True,
                        save_transparent: bool = True) -> bool:
    """
    キャラクター抽出結果を保存
    
    Args:
        image: 抽出されたキャラクター画像
        mask: キャラクターマスク
        output_path: 出力パス（拡張子なし）
        save_mask: マスクも保存するか
        save_transparent: 透明背景版も保存するか
        
    Returns:
        保存成功フラグ
    """
    try:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # メイン画像保存
        main_path = output_path.with_suffix('.jpg')
        cv2.imwrite(str(main_path), image)
        print(f"💾 キャラクター画像保存: {main_path}")
        
        # マスク保存
        if save_mask:
            mask_path = output_path.with_name(f"{output_path.stem}_mask.png")
            cv2.imwrite(str(mask_path), mask)
            print(f"💾 マスク保存: {mask_path}")
        
        # 透明背景版保存
        if save_transparent:
            transparent_image = create_transparent_character(image, mask)
            transparent_path = output_path.with_name(f"{output_path.stem}_transparent.png")
            cv2.imwrite(str(transparent_path), transparent_image)
            print(f"💾 透明背景版保存: {transparent_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存エラー: {e}")
        return False


if __name__ == "__main__":
    # Test post-processing functions
    print("🧪 Post-processing test starting...")
    
    # Create test mask
    test_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_mask, (100, 100), 80, 255, -1)
    
    # Add some noise
    noise = np.random.randint(0, 2, test_mask.shape, dtype=np.uint8) * 50
    noisy_mask = test_mask + noise
    noisy_mask = np.clip(noisy_mask, 0, 255).astype(np.uint8)
    
    # Test enhancement
    enhanced_mask = enhance_character_mask(noisy_mask)
    
    # Test metrics
    metrics = calculate_mask_quality_metrics(enhanced_mask)
    
    print("✅ Post-processing test completed")
    print(f"   Quality metrics: {metrics}")
    print(f"   Original mask area: {np.sum(test_mask > 0)}")
    print(f"   Enhanced mask area: {np.sum(enhanced_mask > 0)}")