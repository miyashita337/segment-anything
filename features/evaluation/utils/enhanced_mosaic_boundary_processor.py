#!/usr/bin/env python3
"""
P1-005: 高度なモザイク境界処理システム

現在のシンプルなモザイク検出を大幅に改良し、
多スケール解析、回転不変性、精密境界処理を実装
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MosaicDetectionResult:
    """モザイク検出結果"""
    mosaic_mask: np.ndarray
    confidence: float
    mosaic_type: str  # 'grid', 'pixelated', 'blur', 'mixed'
    pattern_size: Tuple[int, int]
    pattern_angle: float
    boundary_quality: float


@dataclass
class BoundaryProcessingResult:
    """境界処理結果"""
    processed_image: np.ndarray
    boundary_mask: np.ndarray
    processing_quality: float
    applied_methods: List[str]


class MultiScaleMosaicDetector:
    """多スケールモザイク検出器"""
    
    def __init__(self):
        self.scales = [1.0, 0.75, 0.5, 0.25]  # 異なるスケール
        self.min_pattern_size = 3
        self.max_pattern_size = 50
        
    def detect_at_multiple_scales(self, image: np.ndarray) -> List[MosaicDetectionResult]:
        """複数スケールでモザイク検出"""
        results = []
        
        for scale in self.scales:
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
            else:
                scaled_image = image
                
            result = self._detect_mosaic_single_scale(scaled_image, scale)
            if result.confidence > 0.3:
                results.append(result)
                
        return results
    
    def _detect_mosaic_single_scale(self, image: np.ndarray, scale: float) -> MosaicDetectionResult:
        """単一スケールでのモザイク検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. 格子パターン検出
        grid_result = self._detect_grid_pattern(gray)
        
        # 2. ピクセル化パターン検出
        pixel_result = self._detect_pixelated_pattern(gray)
        
        # 3. ブラーパターン検出
        blur_result = self._detect_blur_pattern(gray)
        
        # 最も信頼度の高い結果を選択
        candidates = [
            ('grid', grid_result),
            ('pixelated', pixel_result),
            ('blur', blur_result)
        ]
        
        best_type, (mask, conf, props) = max(candidates, key=lambda x: x[1][1])
        
        # スケールを元に戻す
        if scale != 1.0:
            original_h, original_w = image.shape[:2] if len(image.shape) == 3 else (image.shape[0], image.shape[1])
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return MosaicDetectionResult(
            mosaic_mask=mask,
            confidence=conf,
            mosaic_type=best_type,
            pattern_size=props.get('pattern_size', (0, 0)),
            pattern_angle=props.get('angle', 0.0),
            boundary_quality=props.get('boundary_quality', 0.0)
        )
    
    def _detect_grid_pattern(self, gray: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """格子パターン検出（改良版）"""
        # エッジ検出
        edges = cv2.Canny(gray, 30, 100)
        
        # 直線検出（Hough変換）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return np.zeros_like(gray), 0.0, {}
        
        # 水平・垂直線の分類
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:  # 水平線
                horizontal_lines.append(line[0])
            elif 75 < angle < 105:  # 垂直線
                vertical_lines.append(line[0])
        
        # 格子の規則性評価
        grid_regularity = self._evaluate_grid_regularity(horizontal_lines, vertical_lines)
        
        # 格子マスク生成
        mask = self._create_grid_mask(gray.shape, horizontal_lines, vertical_lines)
        
        confidence = min(grid_regularity, 1.0)
        
        return mask, confidence, {
            'pattern_size': self._estimate_grid_size(horizontal_lines, vertical_lines),
            'angle': 0.0,  # 格子は基本的に水平垂直
            'boundary_quality': grid_regularity
        }
    
    def _detect_pixelated_pattern(self, gray: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """ピクセル化パターン検出"""
        # ダウンサンプリング→アップサンプリングでピクセル化効果を検出
        original_shape = gray.shape
        
        # 複数の解像度で試行
        best_conf = 0.0
        best_mask = np.zeros_like(gray)
        best_props = {}
        
        for factor in [2, 4, 8, 16]:
            h, w = original_shape
            small_h, small_w = h // factor, w // factor
            
            if small_h < 10 or small_w < 10:
                continue
                
            # ダウンサンプリング→アップサンプリング
            small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            reconstructed = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 元画像との類似度を計算
            diff = cv2.absdiff(gray, reconstructed)
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            # ピクセル境界の検出
            pixel_edges = self._detect_pixel_boundaries(reconstructed, factor)
            
            if similarity > 0.7 and np.sum(pixel_edges) > 0:
                conf = similarity * 0.8  # ピクセル化の信頼度
                if conf > best_conf:
                    best_conf = conf
                    best_mask = (pixel_edges > 0).astype(np.uint8) * 255
                    best_props = {
                        'pattern_size': (factor, factor),
                        'angle': 0.0,
                        'boundary_quality': similarity
                    }
        
        return best_mask, best_conf, best_props
    
    def _detect_blur_pattern(self, gray: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """ブラーパターン検出"""
        # ラプラシアンによるブラー検出
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = laplacian.var()
        
        # 局所的なブラー検出
        kernel_size = 15
        blur_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(0, gray.shape[0] - kernel_size, kernel_size // 2):
            for x in range(0, gray.shape[1] - kernel_size, kernel_size // 2):
                roi = gray[y:y + kernel_size, x:x + kernel_size]
                local_variance = cv2.Laplacian(roi, cv2.CV_64F).var()
                blur_map[y:y + kernel_size, x:x + kernel_size] = local_variance
        
        # ブラー領域の特定
        blur_threshold = np.mean(blur_map) * 0.5
        blur_mask = (blur_map < blur_threshold).astype(np.uint8) * 255
        
        # ブラー領域の連続性評価
        contours, _ = cv2.findContours(blur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(gray), 0.0, {}
        
        # 最大の連続ブラー領域
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (gray.shape[0] * gray.shape[1])
        
        confidence = min(area_ratio * 2.0, 1.0) if area_ratio > 0.1 else 0.0
        
        return blur_mask, confidence, {
            'pattern_size': (kernel_size, kernel_size),
            'angle': 0.0,
            'boundary_quality': confidence
        }
    
    def _evaluate_grid_regularity(self, h_lines: List, v_lines: List) -> float:
        """格子の規則性を評価"""
        if len(h_lines) < 2 or len(v_lines) < 2:
            return 0.0
        
        # 水平線の間隔の規則性
        h_positions = [min(y1, y2) for x1, y1, x2, y2 in h_lines]
        h_positions.sort()
        h_intervals = [h_positions[i+1] - h_positions[i] for i in range(len(h_positions)-1)]
        
        # 垂直線の間隔の規則性
        v_positions = [min(x1, x2) for x1, y1, x2, y2 in v_lines]
        v_positions.sort()
        v_intervals = [v_positions[i+1] - v_positions[i] for i in range(len(v_positions)-1)]
        
        # 間隔の変動係数（CV）で規則性を評価
        h_cv = np.std(h_intervals) / np.mean(h_intervals) if h_intervals else 1.0
        v_cv = np.std(v_intervals) / np.mean(v_intervals) if v_intervals else 1.0
        
        # CV が小さいほど規則的
        regularity = max(0.0, 1.0 - (h_cv + v_cv) / 2.0)
        return regularity
    
    def _estimate_grid_size(self, h_lines: List, v_lines: List) -> Tuple[int, int]:
        """格子サイズの推定"""
        if not h_lines or not v_lines:
            return (0, 0)
        
        # 平均間隔を計算
        h_positions = [min(y1, y2) for x1, y1, x2, y2 in h_lines]
        v_positions = [min(x1, x2) for x1, y1, x2, y2 in v_lines]
        
        h_intervals = np.diff(sorted(h_positions))
        v_intervals = np.diff(sorted(v_positions))
        
        avg_h = int(np.mean(h_intervals)) if len(h_intervals) > 0 else 0
        avg_v = int(np.mean(v_intervals)) if len(v_intervals) > 0 else 0
        
        return (avg_v, avg_h)
    
    def _create_grid_mask(self, shape: Tuple[int, int], h_lines: List, v_lines: List) -> np.ndarray:
        """格子マスクの生成"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        # 線を描画
        for x1, y1, x2, y2 in h_lines:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
        for x1, y1, x2, y2 in v_lines:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
        # 線の周辺を拡張
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _detect_pixel_boundaries(self, image: np.ndarray, factor: int) -> np.ndarray:
        """ピクセル境界の検出"""
        # ピクセル境界は factor の倍数位置にある
        edges = np.zeros_like(image)
        
        # 垂直境界
        for x in range(factor, image.shape[1], factor):
            edges[:, x] = 255
        
        # 水平境界
        for y in range(factor, image.shape[0], factor):
            edges[y, :] = 255
        
        return edges


class RotationInvariantDetector:
    """回転不変なパターン検出器"""
    
    def __init__(self):
        self.angles = np.arange(0, 180, 15)  # 15度刻みで回転をチェック
    
    def detect_rotated_patterns(self, image: np.ndarray) -> MosaicDetectionResult:
        """回転したパターンの検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        best_conf = 0.0
        best_result = None
        
        for angle in self.angles:
            # 画像を回転
            rotated = self._rotate_image(gray, angle)
            
            # 回転した画像でパターン検出
            result = self._detect_aligned_pattern(rotated)
            
            if result['confidence'] > best_conf:
                best_conf = result['confidence']
                best_result = result
                best_result['angle'] = angle
        
        # 結果のマスクを元の向きに戻す
        if best_result and best_result['angle'] != 0:
            best_result['mask'] = self._rotate_image(
                best_result['mask'], -best_result['angle']
            )
        
        return MosaicDetectionResult(
            mosaic_mask=best_result['mask'] if best_result else np.zeros_like(gray),
            confidence=best_conf,
            mosaic_type='rotated_grid',
            pattern_size=best_result.get('pattern_size', (0, 0)) if best_result else (0, 0),
            pattern_angle=best_result.get('angle', 0.0) if best_result else 0.0,
            boundary_quality=best_conf
        )
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """画像を回転"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _detect_aligned_pattern(self, image: np.ndarray) -> Dict[str, Any]:
        """回転済み画像での整列パターン検出"""
        # 水平・垂直線の検出
        edges = cv2.Canny(image, 50, 150)
        
        # 水平線カーネル
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        
        # 垂直線カーネル
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        # 格子パターンの強度
        grid_pattern = cv2.bitwise_or(h_lines, v_lines)
        grid_ratio = np.sum(grid_pattern > 0) / grid_pattern.size
        
        confidence = min(grid_ratio * 5.0, 1.0)  # 感度調整
        
        return {
            'mask': grid_pattern,
            'confidence': confidence,
            'pattern_size': self._estimate_pattern_size(h_lines, v_lines)
        }
    
    def _estimate_pattern_size(self, h_lines: np.ndarray, v_lines: np.ndarray) -> Tuple[int, int]:
        """パターンサイズの推定"""
        # 線の間隔を推定
        h_proj = np.sum(h_lines, axis=1)
        v_proj = np.sum(v_lines, axis=0)
        
        # ピークの間隔を計算
        h_peaks = np.where(h_proj > np.max(h_proj) * 0.3)[0]
        v_peaks = np.where(v_proj > np.max(v_proj) * 0.3)[0]
        
        h_interval = np.mean(np.diff(h_peaks)) if len(h_peaks) > 1 else 0
        v_interval = np.mean(np.diff(v_peaks)) if len(v_peaks) > 1 else 0
        
        return (int(v_interval), int(h_interval))


class AdaptiveBoundaryProcessor:
    """適応的境界処理器"""
    
    def __init__(self):
        self.edge_threshold = 50
        self.blur_kernel_sizes = [3, 5, 7, 9]
    
    def process_mosaic_boundaries(self, image: np.ndarray, 
                                 mosaic_result: MosaicDetectionResult) -> BoundaryProcessingResult:
        """モザイク境界の適応的処理"""
        
        if mosaic_result.confidence < 0.3:
            # モザイクが検出されない場合は元画像をそのまま返す
            return BoundaryProcessingResult(
                processed_image=image.copy(),
                boundary_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                processing_quality=1.0,
                applied_methods=[]
            )
        
        # モザイクタイプに応じた処理
        if mosaic_result.mosaic_type == 'grid':
            return self._process_grid_boundaries(image, mosaic_result)
        elif mosaic_result.mosaic_type == 'pixelated':
            return self._process_pixelated_boundaries(image, mosaic_result)
        elif mosaic_result.mosaic_type == 'blur':
            return self._process_blur_boundaries(image, mosaic_result)
        else:
            return self._process_mixed_boundaries(image, mosaic_result)
    
    def _process_grid_boundaries(self, image: np.ndarray, 
                               mosaic_result: MosaicDetectionResult) -> BoundaryProcessingResult:
        """格子パターンの境界処理"""
        # エッジ保持型フィルタを使用
        processed = cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)
        
        # 格子線の周辺のみ処理
        mask = mosaic_result.mosaic_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # マスク領域のみ処理済み画像を適用
        result = image.copy()
        
        # マスクサイズを画像に合わせる
        if expanded_mask.shape[:2] != image.shape[:2]:
            expanded_mask = cv2.resize(expanded_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask_3ch = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
        mask_norm = mask_3ch.astype(np.float32) / 255.0
        
        result = result.astype(np.float32)
        processed = processed.astype(np.float32)
        
        result = result * (1 - mask_norm) + processed * mask_norm
        result = result.astype(np.uint8)
        
        return BoundaryProcessingResult(
            processed_image=result,
            boundary_mask=expanded_mask,
            processing_quality=mosaic_result.confidence,
            applied_methods=['edge_preserving_filter', 'selective_application']
        )
    
    def _process_pixelated_boundaries(self, image: np.ndarray, 
                                    mosaic_result: MosaicDetectionResult) -> BoundaryProcessingResult:
        """ピクセル化パターンの境界処理"""
        # ピクセル境界をスムージング
        processed = cv2.bilateralFilter(image, 9, 75, 75)
        
        mask = mosaic_result.mosaic_mask
        
        # ピクセル境界の拡張
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 段階的ブレンディング
        result = self._gradual_blending(image, processed, expanded_mask)
        
        return BoundaryProcessingResult(
            processed_image=result,
            boundary_mask=expanded_mask,
            processing_quality=mosaic_result.confidence,
            applied_methods=['bilateral_filter', 'gradual_blending']
        )
    
    def _process_blur_boundaries(self, image: np.ndarray, 
                               mosaic_result: MosaicDetectionResult) -> BoundaryProcessingResult:
        """ブラーパターンの境界処理"""
        # シャープニングフィルタを適用
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # ブラー領域のマスク
        mask = mosaic_result.mosaic_mask
        
        # アンシャープマスキング
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # マスク領域に適用
        result = self._apply_with_mask(image, unsharp, mask)
        
        return BoundaryProcessingResult(
            processed_image=result,
            boundary_mask=mask,
            processing_quality=mosaic_result.confidence,
            applied_methods=['unsharp_masking', 'selective_sharpening']
        )
    
    def _process_mixed_boundaries(self, image: np.ndarray, 
                                mosaic_result: MosaicDetectionResult) -> BoundaryProcessingResult:
        """混合パターンの境界処理"""
        # 複数の手法を組み合わせ
        edge_preserved = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
        bilateral = cv2.bilateralFilter(image, 9, 80, 80)
        
        # 重み付き平均
        combined = cv2.addWeighted(edge_preserved, 0.6, bilateral, 0.4, 0)
        
        mask = mosaic_result.mosaic_mask
        result = self._apply_with_mask(image, combined, mask)
        
        return BoundaryProcessingResult(
            processed_image=result,
            boundary_mask=mask,
            processing_quality=mosaic_result.confidence,
            applied_methods=['edge_preserving_filter', 'bilateral_filter', 'weighted_combination']
        )
    
    def _gradual_blending(self, original: np.ndarray, processed: np.ndarray, 
                         mask: np.ndarray) -> np.ndarray:
        """段階的ブレンディング"""
        # マスクサイズを画像に合わせる
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # マスクをガウシアンブラーで滑らかに
        smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5.0)
        smooth_mask = smooth_mask / 255.0
        
        # 3チャンネルに拡張
        if len(original.shape) == 3:
            smooth_mask = np.stack([smooth_mask] * 3, axis=2)
        
        # ブレンディング
        result = original.astype(np.float32) * (1 - smooth_mask) + \
                processed.astype(np.float32) * smooth_mask
        
        return result.astype(np.uint8)
    
    def _apply_with_mask(self, original: np.ndarray, processed: np.ndarray, 
                        mask: np.ndarray) -> np.ndarray:
        """マスクを使用した選択的適用"""
        # マスクサイズを画像に合わせる
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(original.shape) == 3 else mask
        mask_norm = mask_3ch.astype(np.float32) / 255.0
        
        result = original.astype(np.float32) * (1 - mask_norm) + \
                processed.astype(np.float32) * mask_norm
        
        return result.astype(np.uint8)


class EnhancedMosaicBoundaryProcessor:
    """統合された高度なモザイク境界処理システム"""
    
    def __init__(self):
        self.multiscale_detector = MultiScaleMosaicDetector()
        self.rotation_detector = RotationInvariantDetector()
        self.boundary_processor = AdaptiveBoundaryProcessor()
        
        # 設定
        self.min_confidence = 0.3
        self.enable_rotation_detection = True
        self.enable_multiscale = True
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像の完全なモザイク境界処理
        
        Args:
            image: 入力画像
            
        Returns:
            処理結果の辞書
        """
        results = {
            'original_image': image.copy(),
            'mosaic_detected': False,
            'detection_results': [],
            'boundary_processing': None,
            'final_image': image.copy(),
            'processing_info': {
                'methods_used': [],
                'confidence': 0.0,
                'mosaic_type': 'none'
            }
        }
        
        # 1. 多スケールモザイク検出
        if self.enable_multiscale:
            multiscale_results = self.multiscale_detector.detect_at_multiple_scales(image)
            results['detection_results'].extend(multiscale_results)
        
        # 2. 回転不変検出
        if self.enable_rotation_detection:
            rotation_result = self.rotation_detector.detect_rotated_patterns(image)
            if rotation_result.confidence > self.min_confidence:
                results['detection_results'].append(rotation_result)
        
        # 3. 最適な検出結果を選択
        if results['detection_results']:
            best_result = max(results['detection_results'], key=lambda x: x.confidence)
            
            if best_result.confidence > self.min_confidence:
                results['mosaic_detected'] = True
                
                # 4. 境界処理実行
                boundary_result = self.boundary_processor.process_mosaic_boundaries(
                    image, best_result
                )
                
                results['boundary_processing'] = boundary_result
                results['final_image'] = boundary_result.processed_image
                
                results['processing_info'].update({
                    'methods_used': boundary_result.applied_methods,
                    'confidence': best_result.confidence,
                    'mosaic_type': best_result.mosaic_type,
                    'pattern_size': best_result.pattern_size,
                    'pattern_angle': best_result.pattern_angle,
                    'boundary_quality': boundary_result.processing_quality
                })
        
        return results
    
    def get_processing_summary(self, results: Dict[str, Any]) -> str:
        """処理結果のサマリーを生成"""
        if not results['mosaic_detected']:
            return "モザイクは検出されませんでした"
        
        info = results['processing_info']
        summary = f"""モザイク境界処理完了:
- 検出タイプ: {info['mosaic_type']}
- 信頼度: {info['confidence']:.3f}
- パターンサイズ: {info['pattern_size']}
- 回転角度: {info['pattern_angle']:.1f}°
- 適用手法: {', '.join(info['methods_used'])}
- 境界品質: {info['boundary_quality']:.3f}"""
        
        return summary


def evaluate_mosaic_boundary_enhancement(image_path: str, 
                                       save_results: bool = True) -> Dict[str, Any]:
    """
    モザイク境界処理の評価関数
    
    Args:
        image_path: 入力画像パス
        save_results: 結果保存フラグ
        
    Returns:
        評価結果
    """
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    # 処理実行
    processor = EnhancedMosaicBoundaryProcessor()
    results = processor.process_image(image)
    
    # 結果保存
    if save_results:
        output_dir = Path(image_path).parent / "mosaic_boundary_results"
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(image_path).stem
        
        # 元画像保存
        cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), results['original_image'])
        
        # 処理済み画像保存
        cv2.imwrite(str(output_dir / f"{base_name}_processed.jpg"), results['final_image'])
        
        # モザイクが検出された場合は検出結果も保存
        if results['mosaic_detected'] and results['boundary_processing']:
            boundary_mask = results['boundary_processing'].boundary_mask
            cv2.imwrite(str(output_dir / f"{base_name}_boundary_mask.jpg"), boundary_mask)
    
    # 処理サマリー
    summary = processor.get_processing_summary(results)
    results['summary'] = summary
    
    return results


if __name__ == "__main__":
    # テスト実行
    test_image = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07/test_single.jpg"
    
    try:
        results = evaluate_mosaic_boundary_enhancement(test_image, save_results=True)
        print("=== P1-005 モザイク境界処理テスト ===")
        print(results['summary'])
        
        if results['mosaic_detected']:
            print(f"\n検出された手法数: {len(results['detection_results'])}")
            for i, detection in enumerate(results['detection_results']):
                print(f"  手法{i+1}: {detection.mosaic_type} (信頼度: {detection.confidence:.3f})")
        
    except Exception as e:
        print(f"エラー: {e}")