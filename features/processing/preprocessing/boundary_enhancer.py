#!/usr/bin/env python3
"""
Boundary Enhancer - SAM前処理改善システム
肌色・衣装境界強調処理で境界認識精度を向上

主な機能:
1. 肌色領域検出と強調
2. 衣装領域（黒・白）の境界明確化  
3. エッジ強化による境界情報増強
4. 適応的コントラスト調整
"""

import numpy as np
import cv2

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BoundaryEnhancer:
    """境界強調処理クラス"""
    
    def __init__(self, 
                 skin_enhancement_factor: float = 1.5,
                 edge_enhancement_factor: float = 2.0,
                 contrast_enhancement: float = 1.3):
        """
        Args:
            skin_enhancement_factor: 肌色強調係数
            edge_enhancement_factor: エッジ強調係数  
            contrast_enhancement: コントラスト強調係数
        """
        self.skin_enhancement_factor = skin_enhancement_factor
        self.edge_enhancement_factor = edge_enhancement_factor
        self.contrast_enhancement = contrast_enhancement
        
        # 肌色HSV範囲（複数レンジをサポート）
        self.skin_ranges = [
            # 明るい肌色
            {"lower": np.array([0, 20, 70]), "upper": np.array([20, 255, 255])},
            # やや暗い肌色
            {"lower": np.array([0, 10, 60]), "upper": np.array([25, 180, 230])},
            # アニメ調の肌色
            {"lower": np.array([0, 15, 80]), "upper": np.array([15, 200, 255])},
        ]
        
        logger.info(f"BoundaryEnhancer初期化: skin_factor={skin_enhancement_factor}, "
                   f"edge_factor={edge_enhancement_factor}, contrast={contrast_enhancement}")

    def enhance_image_boundaries(self, image: np.ndarray) -> np.ndarray:
        """
        画像の境界を強調してSAMの認識精度を向上
        
        Args:
            image: 入力画像 (H, W, 3)
            
        Returns:
            強調された画像 (H, W, 3)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"RGB画像が必要です: {image.shape}")
            
        logger.debug(f"境界強調処理開始: {image.shape}")
        
        # 1. 肌色領域検出・強調
        enhanced_image = self._enhance_skin_regions(image.copy())
        
        # 2. 衣装境界（白-黒）強調
        enhanced_image = self._enhance_clothing_boundaries(enhanced_image)
        
        # 3. エッジ情報強化
        enhanced_image = self._enhance_edges(enhanced_image)
        
        # 4. 適応的コントラスト調整
        enhanced_image = self._adaptive_contrast_adjustment(enhanced_image)
        
        logger.debug("境界強調処理完了")
        return enhanced_image

    def _enhance_skin_regions(self, image: np.ndarray) -> np.ndarray:
        """肌色領域を検出して強調"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 複数の肌色レンジでマスク作成
        skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for skin_range in self.skin_ranges:
            range_mask = cv2.inRange(hsv, skin_range["lower"], skin_range["upper"])
            skin_mask = cv2.bitwise_or(skin_mask, range_mask)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # 肌色領域の明度とコントラストを強調
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # L（明度）チャネルの肌色部分を強調
        enhanced_l = l.copy().astype(np.float32)
        skin_pixels = skin_mask > 0
        enhanced_l[skin_pixels] = np.clip(
            enhanced_l[skin_pixels] * self.skin_enhancement_factor, 0, 255
        )
        
        # LABからRGBに変換して戻す
        enhanced_lab = cv2.merge([enhanced_l.astype(np.uint8), a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        logger.debug(f"肌色強調: {np.sum(skin_pixels)}ピクセル処理")
        return enhanced_image

    def _enhance_clothing_boundaries(self, image: np.ndarray) -> np.ndarray:
        """衣装境界（白-黒、黒-肌色）を強調 - 白色系部位検出改善版"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 黒い領域（衣装）の検出 - 閾値を調整
        black_mask = gray < 60  # 50->60で黒領域を拡大
        
        # 白い領域の多段階検出 - 白色系部位の検出精度向上
        # 純白領域
        white_high = gray > 220
        # 薄白・グレー領域  
        white_mid = (gray > 180) & (gray <= 220)
        # 肌色系白領域
        white_low = (gray > 140) & (gray <= 180)
        
        # HSV色空間での白色系検出補強
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 低彩度・高明度の白色系領域
        white_hsv = (s < 30) & (v > 150)  # 彩度低く明度高い
        
        # 白色系マスクの統合
        white_mask = white_high | white_mid | white_low | white_hsv
        
        # ノイズ除去とエッジ保護
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 黒領域の処理
        black_cleaned = cv2.morphologyEx(black_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_smooth)
        black_dilated = cv2.dilate(black_cleaned, kernel_dilate, iterations=1)
        
        # 白領域の処理 - より慎重に
        white_cleaned = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_smooth)
        white_dilated = cv2.dilate(white_cleaned, kernel_smooth, iterations=1)  # 控えめな拡張
        
        # エッジベース境界検出の追加
        edges = cv2.Canny(gray, 30, 100)  # 低閾値でエッジ検出
        edge_dilated = cv2.dilate(edges, kernel_smooth, iterations=1)
        
        # 境界領域の複合検出
        # 1. 従来の黒白境界
        boundary_traditional = cv2.bitwise_and(black_dilated, white_dilated)
        # 2. エッジベース境界
        boundary_edge = cv2.bitwise_and(edge_dilated, white_mask.astype(np.uint8))
        # 3. 統合境界
        boundary_mask = cv2.bitwise_or(boundary_traditional, boundary_edge)
        
        # 境界部分の適応的強調 - 白色系部位専用処理
        enhanced_image = image.copy().astype(np.float32)
        boundary_pixels = boundary_mask > 0
        white_pixels = white_mask
        
        if np.any(boundary_pixels):
            # LAB色空間での処理
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # L(明度)チャネルでの境界強調
            enhanced_l = l.astype(np.float32)
            
            # 境界部分の明度コントラストを強化
            boundary_mean = np.mean(enhanced_l[boundary_pixels])
            enhanced_l[boundary_pixels] = np.where(
                enhanced_l[boundary_pixels] > boundary_mean,
                np.clip(enhanced_l[boundary_pixels] * 1.15, 0, 255),
                np.clip(enhanced_l[boundary_pixels] * 0.85, 0, 255)
            )
            
            # 白色系領域の輪郭を保護
            white_boundary = cv2.bitwise_and(boundary_mask, white_pixels.astype(np.uint8))
            if np.any(white_boundary > 0):
                # 白色系境界の明度を適度に下げて境界を明確化
                enhanced_l[white_boundary > 0] = np.clip(
                    enhanced_l[white_boundary > 0] * 0.95, 0, 255
                )
            
            # LAB->RGB変換
            enhanced_lab = cv2.merge([enhanced_l.astype(np.uint8), a, b])
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        logger.debug(f"改善版衣装境界強調: 境界={np.sum(boundary_pixels)}px, 白色系={np.sum(white_pixels)}px")
        return enhanced_image.astype(np.uint8)

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """エッジ情報を強化して境界認識を改善"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 複数のエッジ検出手法を組み合わせ
        # 1. Canny エッジ
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # 2. Sobel エッジ  
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = np.clip(edges_sobel / edges_sobel.max() * 255, 0, 255).astype(np.uint8)
        
        # 3. Laplacian エッジ
        edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges_laplacian = np.clip(np.abs(edges_laplacian) / np.abs(edges_laplacian).max() * 255, 0, 255).astype(np.uint8)
        
        # エッジ情報を統合
        combined_edges = cv2.bitwise_or(edges_canny, cv2.bitwise_or(edges_sobel, edges_laplacian))
        
        # エッジ部分を画像に重ね合わせて強調
        enhanced_image = image.copy().astype(np.float32)
        edge_pixels = combined_edges > 0
        
        # エッジ部分のコントラストを強化
        for channel in range(3):
            channel_data = enhanced_image[:, :, channel]
            channel_data[edge_pixels] = np.clip(
                channel_data[edge_pixels] * self.edge_enhancement_factor, 0, 255
            )
            enhanced_image[:, :, channel] = channel_data
        
        logger.debug(f"エッジ強調: {np.sum(edge_pixels)}ピクセル処理")
        return enhanced_image.astype(np.uint8)

    def _adaptive_contrast_adjustment(self, image: np.ndarray) -> np.ndarray:
        """適応的コントラスト調整"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) を使用
        # LAB色空間でL(明度)チャネルのみ処理
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHEを適用
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # 全体的なコントラスト調整
        enhanced_l = enhanced_l.astype(np.float32)
        enhanced_l = np.clip(enhanced_l * self.contrast_enhancement, 0, 255)
        
        # LABからRGBに変換
        enhanced_lab = cv2.merge([enhanced_l.astype(np.uint8), a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        logger.debug("適応的コントラスト調整完了")
        return enhanced_image

    def get_enhancement_stats(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        """強調処理の統計情報を取得"""
        # コントラスト比較
        original_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY))
        enhanced_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY))
        
        # エッジ強度比較
        original_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)
        enhanced_edges = cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY), 50, 150)
        
        return {
            "contrast_improvement": enhanced_contrast / max(original_contrast, 1.0),
            "edge_density_original": np.sum(original_edges > 0),
            "edge_density_enhanced": np.sum(enhanced_edges > 0),
            "edge_improvement": np.sum(enhanced_edges > 0) / max(np.sum(original_edges > 0), 1)
        }


def test_boundary_enhancer():
    """境界強調システムのテスト"""
    enhancer = BoundaryEnhancer()
    
    # テスト画像の読み込み
    test_image_path = Path("/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0000_cover.jpg")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"テスト画像読み込み: {image.shape}")
        
        # 強調処理実行
        enhanced = enhancer.enhance_image_boundaries(image)
        
        # 統計情報取得
        stats = enhancer.get_enhancement_stats(image, enhanced)
        print("強調処理統計:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")
        
        # 結果保存
        output_path = Path("/tmp/boundary_enhancement_test.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        print(f"強調結果保存: {output_path}")
    else:
        print(f"テスト画像が見つかりません: {test_image_path}")


if __name__ == "__main__":
    test_boundary_enhancer()