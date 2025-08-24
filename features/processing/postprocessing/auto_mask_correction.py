#!/usr/bin/env python3
"""
P1-005: 自動マスク修正機能
Advanced automatic mask correction with edge smoothing and noise removal
マスクエッジ自動スムージング・ノイズ除去機能
"""

import numpy as np
import cv2

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class MaskCorrectionParams:
    """マスク修正パラメータ"""
    # エッジスムージング
    edge_smoothing_enabled: bool = True
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    
    # ノイズ除去
    noise_removal_enabled: bool = True
    min_contour_area: int = 100
    morphology_kernel_size: int = 3
    morphology_iterations: int = 2
    
    # ホール埋め
    hole_filling_enabled: bool = True
    max_hole_area: int = 500
    
    # 適応的調整
    adaptive_parameters: bool = True
    quality_threshold: float = 0.7


class AutoMaskCorrector:
    """自動マスク修正システム"""
    
    def __init__(self, params: Optional[MaskCorrectionParams] = None):
        """
        初期化
        
        Args:
            params: 修正パラメータ
        """
        self.params = params or MaskCorrectionParams()
        logger.info("自動マスク修正システム初期化完了")
    
    def correct_mask_automatically(self, 
                                 mask: np.ndarray,
                                 original_image: Optional[np.ndarray] = None,
                                 quality_score: Optional[float] = None) -> Dict[str, Any]:
        """
        マスクを自動修正
        
        Args:
            mask: 入力マスク (0-255)
            original_image: 元画像（より精密な修正に使用）
            quality_score: 現在の品質スコア
            
        Returns:
            修正結果辞書
        """
        logger.info("自動マスク修正開始")
        
        corrected_mask = mask.copy()
        correction_log = []
        
        # 1. 品質評価と適応的パラメータ調整
        if self.params.adaptive_parameters:
            corrected_mask, adaptive_log = self._apply_adaptive_correction(
                corrected_mask, quality_score, original_image
            )
            correction_log.extend(adaptive_log)
        
        # 2. ノイズ除去
        if self.params.noise_removal_enabled:
            corrected_mask, noise_log = self._remove_noise(corrected_mask)
            correction_log.extend(noise_log)
        
        # 3. ホール埋め
        if self.params.hole_filling_enabled:
            corrected_mask, hole_log = self._fill_holes(corrected_mask)
            correction_log.extend(hole_log)
        
        # 4. エッジスムージング
        if self.params.edge_smoothing_enabled:
            corrected_mask, edge_log = self._smooth_edges(corrected_mask, original_image)
            correction_log.extend(edge_log)
        
        # 5. 品質検証
        quality_metrics = self._calculate_quality_metrics(mask, corrected_mask)
        
        result = {
            'corrected_mask': corrected_mask,
            'correction_log': correction_log,
            'quality_metrics': quality_metrics,
            'improvement_ratio': quality_metrics.get('improvement_ratio', 0.0),
            'processing_success': True
        }
        
        logger.info(f"自動マスク修正完了 - 改善率: {quality_metrics.get('improvement_ratio', 0.0):.2%}")
        return result
    
    def _apply_adaptive_correction(self, 
                                 mask: np.ndarray,
                                 quality_score: Optional[float],
                                 original_image: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """適応的修正を適用"""
        log = []
        
        if quality_score is not None and quality_score < self.params.quality_threshold:
            # 品質が低い場合はより強力な修正を適用
            self.params.morphology_iterations = min(self.params.morphology_iterations + 1, 5)
            self.params.gaussian_kernel_size = min(self.params.gaussian_kernel_size + 2, 11)
            log.append(f"低品質検出 (score: {quality_score:.3f}) - 修正パラメータ強化")
        
        # 画像サイズに基づく適応
        height, width = mask.shape[:2]
        if height * width > 1000000:  # 大画像
            self.params.min_contour_area = max(self.params.min_contour_area, 200)
            log.append("大画像検出 - ノイズ除去強化")
        elif height * width < 100000:  # 小画像
            self.params.min_contour_area = min(self.params.min_contour_area, 50)
            log.append("小画像検出 - ノイズ除去調整")
        
        return mask, log
    
    def _remove_noise(self, mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """ノイズ除去処理"""
        log = []
        
        # モルフォロジカル処理
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.params.morphology_kernel_size, self.params.morphology_kernel_size)
        )
        
        # Opening（ノイズ除去）
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, 
            iterations=self.params.morphology_iterations
        )
        log.append(f"モルフォロジカルOpening実行 - カーネル: {self.params.morphology_kernel_size}x{self.params.morphology_kernel_size}")
        
        # 小さな連結成分除去
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        final_mask = np.zeros_like(mask)
        removed_components = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.params.min_contour_area:
                final_mask[labels == i] = 255
            else:
                removed_components += 1
        
        log.append(f"小成分除去: {removed_components}個除去 (閾値: {self.params.min_contour_area}px)")
        
        return final_mask, log
    
    def _fill_holes(self, mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """ホール埋め処理"""
        log = []
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask = mask.copy()
        holes_filled = 0
        
        # ホールを検出して埋める
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 0 and area <= self.params.max_hole_area:
                # 小さなホールを埋める
                cv2.fillPoly(filled_mask, [contour], 255)
                holes_filled += 1
        
        if holes_filled > 0:
            log.append(f"ホール埋め: {holes_filled}個のホールを埋めました")
        else:
            log.append("ホール埋め: 対象ホールなし")
        
        return filled_mask, log
    
    def _smooth_edges(self, mask: np.ndarray, original_image: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """エッジスムージング処理"""
        log = []
        
        if original_image is not None and len(original_image.shape) == 3:
            # バイラテラルフィルタ（エッジ保持しながら平滑化）
            mask_float = mask.astype(np.float32)
            bilateral_smoothed = cv2.bilateralFilter(
                mask_float,
                self.params.bilateral_d,
                self.params.bilateral_sigma_color,
                self.params.bilateral_sigma_space
            )
            
            # 二値化
            _, smoothed_mask = cv2.threshold(bilateral_smoothed, 127, 255, cv2.THRESH_BINARY)
            smoothed_mask = smoothed_mask.astype(np.uint8)
            log.append(f"バイラテラルフィルタ適用 - d:{self.params.bilateral_d}")
        else:
            # ガウシアンフィルタ
            if self.params.gaussian_kernel_size % 2 == 0:
                kernel_size = self.params.gaussian_kernel_size + 1
            else:
                kernel_size = self.params.gaussian_kernel_size
            
            gaussian_smoothed = cv2.GaussianBlur(
                mask, 
                (kernel_size, kernel_size), 
                self.params.gaussian_sigma
            )
            
            # 二値化
            _, smoothed_mask = cv2.threshold(gaussian_smoothed, 127, 255, cv2.THRESH_BINARY)
            log.append(f"ガウシアンフィルタ適用 - カーネル: {kernel_size}x{kernel_size}")
        
        return smoothed_mask, log
    
    def _calculate_quality_metrics(self, original_mask: np.ndarray, corrected_mask: np.ndarray) -> Dict[str, float]:
        """品質メトリクス計算"""
        # 面積比較
        original_area = cv2.countNonZero(original_mask)
        corrected_area = cv2.countNonZero(corrected_mask)
        area_ratio = corrected_area / max(original_area, 1)
        
        # 形状複雑度（周囲長/面積比）
        def calculate_complexity(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            total_area = cv2.countNonZero(mask)
            return total_perimeter / max(total_area, 1) if total_area > 0 else 0.0
        
        original_complexity = calculate_complexity(original_mask)
        corrected_complexity = calculate_complexity(corrected_mask)
        
        # 重複度（IoU）
        intersection = cv2.countNonZero(cv2.bitwise_and(original_mask, corrected_mask))
        union = cv2.countNonZero(cv2.bitwise_or(original_mask, corrected_mask))
        iou = intersection / max(union, 1)
        
        # 改善率計算（複雑度削減を良い方向とする）
        complexity_improvement = max(0, (original_complexity - corrected_complexity) / max(original_complexity, 0.001))
        
        return {
            'area_ratio': area_ratio,
            'original_complexity': original_complexity,
            'corrected_complexity': corrected_complexity,
            'complexity_improvement': complexity_improvement,
            'iou': iou,
            'improvement_ratio': (complexity_improvement + iou) / 2.0
        }


def create_auto_mask_corrector(quality_focused: bool = True) -> AutoMaskCorrector:
    """
    自動マスク修正システムを作成
    
    Args:
        quality_focused: 品質重視設定
        
    Returns:
        設定済み自動マスク修正システム
    """
    if quality_focused:
        params = MaskCorrectionParams(
            gaussian_kernel_size=7,
            bilateral_d=11,
            min_contour_area=150,
            morphology_iterations=3,
            adaptive_parameters=True
        )
    else:
        params = MaskCorrectionParams()
    
    return AutoMaskCorrector(params)


# 使用例
if __name__ == "__main__":
    # テスト用のダミーマスク作成
    test_mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(test_mask, (250, 250), 200, 255, -1)
    
    # ノイズ追加
    noise_points = np.random.randint(0, 500, (50, 2))
    for point in noise_points:
        cv2.circle(test_mask, tuple(point), 5, 255, -1)
    
    # 自動修正実行
    corrector = create_auto_mask_corrector(quality_focused=True)
    result = corrector.correct_mask_automatically(test_mask)
    
    print("修正ログ:")
    for log_entry in result['correction_log']:
        print(f"  - {log_entry}")
    
    print(f"\n品質メトリクス:")
    for key, value in result['quality_metrics'].items():
        print(f"  {key}: {value:.4f}")