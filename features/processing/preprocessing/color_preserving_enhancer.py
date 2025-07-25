#!/usr/bin/env python3
"""
Color Preserving Enhancer - 色調保持境界強調システム
白っぽさ問題を解決し、自然な色調を保持する境界強調処理
"""

import numpy as np
import cv2

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ColorPreservingEnhancer:
    """色調保持境界強調クラス - 白っぽさ問題解決版"""
    
    def __init__(self, 
                 preserve_luminance: bool = True,
                 preserve_saturation: bool = True,
                 adaptive_enhancement: bool = True):
        """
        Args:
            preserve_luminance: 明度保持フラグ
            preserve_saturation: 彩度保持フラグ  
            adaptive_enhancement: 適応的強調フラグ
        """
        self.preserve_luminance = preserve_luminance
        self.preserve_saturation = preserve_saturation
        self.adaptive_enhancement = adaptive_enhancement
        
        logger.info(f"ColorPreservingEnhancer初期化: luminance={preserve_luminance}, "
                   f"saturation={preserve_saturation}, adaptive={adaptive_enhancement}")

    def enhance_image_boundaries(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        色調を保持しながら境界を強調
        
        Args:
            image: 入力画像 (H, W, 3)
            
        Returns:
            強調された画像とメトリクス
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"RGB画像が必要です: {image.shape}")
            
        logger.debug(f"色調保持境界強調開始: {image.shape}")
        
        # 元画像の色調特性を保存
        original_stats = self._analyze_color_properties(image)
        
        # 1. グレースケール検出と適応的処理
        is_grayscale = self._is_grayscale_image(image)
        enhancement_factor = 0.3 if is_grayscale else 0.7
        
        # 2. HSV色空間での処理
        enhanced_image = self._hsv_aware_enhancement(image, enhancement_factor)
        
        # 3. 色調保持処理
        if self.preserve_luminance or self.preserve_saturation:
            enhanced_image = self._preserve_color_properties(
                image, enhanced_image, original_stats
            )
        
        # 4. エッジ強化（控えめ）
        enhanced_image = self._gentle_edge_enhancement(enhanced_image)
        
        # 5. 最終調整
        enhanced_image = self._final_color_adjustment(image, enhanced_image)
        
        # メトリクス計算
        metrics = self._calculate_enhancement_metrics(image, enhanced_image)
        
        logger.debug("色調保持境界強調完了")
        return enhanced_image, metrics

    def _analyze_color_properties(self, image: np.ndarray) -> Dict[str, float]:
        """元画像の色調特性を分析"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        return {
            'mean_hue': np.mean(h),
            'mean_saturation': np.mean(s),
            'mean_value': np.mean(v),
            'std_value': np.std(v),
            'brightness_range': np.max(v) - np.min(v),
            'saturation_range': np.max(s) - np.min(s)
        }

    def _is_grayscale_image(self, image: np.ndarray) -> bool:
        """グレースケール画像かどうか判定"""
        # RGBチャンネル間の最大差分を計算
        r, g, b = cv2.split(image)
        max_diff_rg = np.max(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        max_diff_rb = np.max(np.abs(r.astype(np.float32) - b.astype(np.float32)))
        max_diff_gb = np.max(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        
        max_channel_diff = max(max_diff_rg, max_diff_rb, max_diff_gb)
        
        # 閾値を下げて、ほぼグレースケールの画像も検出
        is_gray = max_channel_diff < 15.0  # 前回39.198から大幅に下げる
        
        logger.debug(f"グレースケール判定: max_diff={max_channel_diff:.3f}, "
                    f"is_grayscale={is_gray}")
        return is_gray

    def _hsv_aware_enhancement(self, image: np.ndarray, factor: float) -> np.ndarray:
        """HSV色空間での色調認識境界強調"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # V(明度)チャネルのみを慎重に強調
        enhanced_v = v.astype(np.float32)
        
        # 肌色領域の特別処理
        skin_mask = self._detect_skin_regions(hsv)
        
        # 一般領域: 控えめな強調
        enhanced_v = enhanced_v * (1.0 + factor * 0.2)
        
        # 肌色領域: さらに控えめに
        if np.any(skin_mask):
            enhanced_v[skin_mask] = v[skin_mask].astype(np.float32) * (1.0 + factor * 0.1)
        
        # 値域クリッピング
        enhanced_v = np.clip(enhanced_v, 0, 255).astype(np.uint8)
        
        # HSVを再構成してRGBに変換
        enhanced_hsv = cv2.merge([h, s, enhanced_v])
        enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced_rgb

    def _detect_skin_regions(self, hsv: np.ndarray) -> np.ndarray:
        """肌色領域を検出"""
        skin_ranges = [
            # より保守的な肌色範囲
            {"lower": np.array([0, 20, 80]), "upper": np.array([20, 180, 255])},
            {"lower": np.array([0, 10, 60]), "upper": np.array([25, 150, 230])},
        ]
        
        skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for skin_range in skin_ranges:
            range_mask = cv2.inRange(hsv, skin_range["lower"], skin_range["upper"])
            skin_mask = cv2.bitwise_or(skin_mask, range_mask)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask > 0

    def _preserve_color_properties(self, original: np.ndarray, 
                                  enhanced: np.ndarray, 
                                  original_stats: Dict[str, float]) -> np.ndarray:
        """色調特性を保持"""
        if not self.preserve_luminance and not self.preserve_saturation:
            return enhanced
        
        # HSV変換
        enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h_enh, s_enh, v_enh = cv2.split(enhanced_hsv)
        
        # 明度保持処理
        if self.preserve_luminance:
            # 元画像の明度分布に近づける
            current_mean_v = np.mean(v_enh)
            target_mean_v = original_stats['mean_value']
            
            # 明度を元画像レベルに調整
            adjustment_factor = target_mean_v / max(current_mean_v, 1.0)
            # 極端な調整を避ける
            adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)
            
            v_enh = np.clip(v_enh.astype(np.float32) * adjustment_factor, 0, 255).astype(np.uint8)
        
        # 彩度保持処理
        if self.preserve_saturation:
            # 元画像の彩度レベルを維持
            current_mean_s = np.mean(s_enh)
            target_mean_s = original_stats['mean_saturation']
            
            if current_mean_s > 0:
                saturation_factor = target_mean_s / current_mean_s
                saturation_factor = np.clip(saturation_factor, 0.9, 1.1)
                s_enh = np.clip(s_enh.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        
        # HSVを再構成
        preserved_hsv = cv2.merge([h_enh, s_enh, v_enh])
        preserved_rgb = cv2.cvtColor(preserved_hsv, cv2.COLOR_HSV2RGB)
        
        return preserved_rgb

    def _gentle_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """控えめなエッジ強調"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 控えめなエッジ検出
        edges = cv2.Canny(gray, 30, 100)  # 閾値を下げる
        
        # エッジを画像に軽く重ね合わせ
        enhanced_image = image.copy().astype(np.float32)
        edge_pixels = edges > 0
        
        # 非常に控えめな強調（10%のみ）
        for channel in range(3):
            channel_data = enhanced_image[:, :, channel]
            channel_data[edge_pixels] = np.clip(
                channel_data[edge_pixels] * 1.1, 0, 255  # 1.1倍のみ
            )
            enhanced_image[:, :, channel] = channel_data
        
        return enhanced_image.astype(np.uint8)

    def _final_color_adjustment(self, original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """最終色調調整"""
        # 元画像の色調に近づける加重平均
        blend_ratio = 0.7  # 元画像30%, 強調画像70%
        
        final_image = (original.astype(np.float32) * (1 - blend_ratio) + 
                      enhanced.astype(np.float32) * blend_ratio)
        
        return np.clip(final_image, 0, 255).astype(np.uint8)

    def _calculate_enhancement_metrics(self, original: np.ndarray, 
                                     enhanced: np.ndarray) -> Dict[str, float]:
        """強調処理のメトリクス計算"""
        # コントラスト比較
        orig_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY))
        enh_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY))
        
        # 色調変化量
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
        enh_hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        
        color_change = np.mean(np.abs(orig_hsv.astype(np.float32) - enh_hsv.astype(np.float32)))
        
        # エッジ密度
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)
        enh_edges = cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY), 50, 150)
        
        return {
            "contrast_improvement": enh_contrast / max(orig_contrast, 1.0),
            "color_preservation": max(0, 1.0 - color_change / 100.0),
            "edge_density_original": np.sum(orig_edges > 0),
            "edge_density_enhanced": np.sum(enh_edges > 0),
            "edge_improvement": np.sum(enh_edges > 0) / max(np.sum(orig_edges > 0), 1),
            "overall_quality": (enh_contrast / max(orig_contrast, 1.0) + 
                              max(0, 1.0 - color_change / 100.0)) / 2.0
        }


def test_color_preserving_enhancer():
    """色調保持強調システムのテスト"""
    enhancer = ColorPreservingEnhancer(
        preserve_luminance=True,
        preserve_saturation=True,
        adaptive_enhancement=True
    )
    
    # テスト画像の読み込み
    test_image_path = Path("/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0001.jpg")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"テスト画像読み込み: {image.shape}")
        
        # 色調保持強調実行
        enhanced, metrics = enhancer.enhance_image_boundaries(image)
        
        # 統計情報表示
        print("色調保持強調統計:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        # 結果保存
        output_path = Path("/tmp/color_preserving_test.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        print(f"結果保存: {output_path}")
    else:
        print(f"テスト画像が見つかりません: {test_image_path}")


if __name__ == "__main__":
    test_color_preserving_enhancer()