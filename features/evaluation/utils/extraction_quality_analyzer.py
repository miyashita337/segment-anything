"""
QI-002: 抽出品質分析器 (ExtractionQualityAnalyzer)

抽出されたキャラクター画像の総合的な品質を分析・評価します。
エッジ品質、サイズ適正性、鮮明度、全体品質を統合評価。
"""

import numpy as np
import cv2

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional


@dataclass
class ExtractionQualityResult:
    """抽出品質評価結果"""
    overall_quality_score: float
    edge_quality_score: float
    completeness_score: float
    sharpness_score: float
    size_adequacy_score: float
    brightness_quality_score: float
    contrast_quality_score: float
    noise_level_score: float
    quality_issues: List[str]
    quality_recommendations: List[str]
    additional_metrics: Dict[str, float]


class ExtractionQualityAnalyzer:
    """抽出品質分析を行うクラス"""
    
    def __init__(self,
                 min_edge_density: float = 0.05,
                 max_edge_density: float = 0.3,
                 min_sharpness: float = 100.0,
                 ideal_brightness_range: tuple = (50, 200),
                 min_contrast: float = 30.0):
        """
        ExtractionQualityAnalyzer の初期化
        
        Args:
            min_edge_density: 最小エッジ密度
            max_edge_density: 最大エッジ密度
            min_sharpness: 最小鮮明度（Laplacianバリアンス）
            ideal_brightness_range: 理想的な明度範囲
            min_contrast: 最小コントラスト
        """
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density
        self.min_sharpness = min_sharpness
        self.ideal_brightness_range = ideal_brightness_range
        self.min_contrast = min_contrast
    
    def analyze_extraction_quality(self, image: np.ndarray) -> ExtractionQualityResult:
        """
        抽出画像の総合品質分析
        
        Args:
            image: 抽出されたキャラクター画像 (H, W, C) numpy配列
            
        Returns:
            ExtractionQualityResult: 品質評価結果
        """
        try:
            issues = []
            recommendations = []
            additional_metrics = {}
            
            # 基本的な画像特性の取得
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                is_color = True
            else:
                gray = image
                is_color = False
            
            # 1. エッジ品質分析
            edge_quality_score, edge_metrics = self._analyze_edge_quality(
                gray, issues, recommendations
            )
            additional_metrics.update(edge_metrics)
            
            # 2. 完全性分析（輪郭の完全性）
            completeness_score, completeness_metrics = self._analyze_completeness(
                gray, issues, recommendations
            )
            additional_metrics.update(completeness_metrics)
            
            # 3. 鮮明度分析
            sharpness_score, sharpness_metrics = self._analyze_sharpness(
                gray, issues, recommendations
            )
            additional_metrics.update(sharpness_metrics)
            
            # 4. サイズ適正性分析
            size_adequacy_score, size_metrics = self._analyze_size_adequacy(
                image, issues, recommendations
            )
            additional_metrics.update(size_metrics)
            
            # 5. 明度品質分析
            brightness_quality_score, brightness_metrics = self._analyze_brightness_quality(
                gray, issues, recommendations
            )
            additional_metrics.update(brightness_metrics)
            
            # 6. コントラスト品質分析
            contrast_quality_score, contrast_metrics = self._analyze_contrast_quality(
                gray, issues, recommendations
            )
            additional_metrics.update(contrast_metrics)
            
            # 7. ノイズレベル分析
            noise_level_score, noise_metrics = self._analyze_noise_level(
                gray, issues, recommendations
            )
            additional_metrics.update(noise_metrics)
            
            # 総合品質スコアの計算（重み付き平均）
            overall_quality_score = (
                edge_quality_score * 0.20 +
                completeness_score * 0.20 +
                sharpness_score * 0.15 +
                size_adequacy_score * 0.15 +
                brightness_quality_score * 0.10 +
                contrast_quality_score * 0.10 +
                noise_level_score * 0.10
            )
            
            return ExtractionQualityResult(
                overall_quality_score=overall_quality_score,
                edge_quality_score=edge_quality_score,
                completeness_score=completeness_score,
                sharpness_score=sharpness_score,
                size_adequacy_score=size_adequacy_score,
                brightness_quality_score=brightness_quality_score,
                contrast_quality_score=contrast_quality_score,
                noise_level_score=noise_level_score,
                quality_issues=issues,
                quality_recommendations=recommendations,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            return ExtractionQualityResult(
                overall_quality_score=0.0,
                edge_quality_score=0.0,
                completeness_score=0.0,
                sharpness_score=0.0,
                size_adequacy_score=0.0,
                brightness_quality_score=0.0,
                contrast_quality_score=0.0,
                noise_level_score=0.0,
                quality_issues=[f"Analysis error: {str(e)}"],
                quality_recommendations=["Retry with valid image format"],
                additional_metrics={'error': 1.0}
            )
    
    def _analyze_edge_quality(self, gray: np.ndarray, issues: List[str], 
                             recommendations: List[str]) -> tuple:
        """エッジ品質の分析"""
        # Canny エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # エッジ品質スコア
        if self.min_edge_density <= edge_density <= self.max_edge_density:
            edge_score = 1.0
        elif edge_density < self.min_edge_density:
            edge_score = 0.6
            issues.append(f"Low edge density: {edge_density:.3f}")
            recommendations.append("Increase image sharpness or reduce blur")
        else:
            edge_score = 0.7
            issues.append(f"High edge density: {edge_density:.3f} (may be noisy)")
            recommendations.append("Apply noise reduction")
        
        # エッジ連続性の評価
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_continuity = 0.0
        if contours:
            # 最長の輪郭の連続性
            longest_contour = max(contours, key=cv2.arcLength, default=[])
            if len(longest_contour) > 0:
                perimeter = cv2.arcLength(longest_contour, True)
                edge_continuity = min(1.0, perimeter / 1000.0)  # 1000ピクセルで正規化
        
        final_edge_score = (edge_score * 0.7 + edge_continuity * 0.3)
        
        metrics = {
            'edge_density': edge_density,
            'edge_continuity': edge_continuity,
            'edge_contour_count': len(contours)
        }
        
        return final_edge_score, metrics
    
    def _analyze_completeness(self, gray: np.ndarray, issues: List[str], 
                             recommendations: List[str]) -> tuple:
        """完全性の分析"""
        # 非ゼロピクセル領域の解析
        binary_mask = (gray > 10).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            issues.append("No valid character region detected")
            recommendations.append("Check image content and brightness")
            return 0.0, {'completeness_contours': 0, 'largest_area_ratio': 0.0}
        
        # 最大輪郭の面積比
        areas = [cv2.contourArea(contour) for contour in contours]
        largest_area = max(areas)
        total_area = gray.shape[0] * gray.shape[1]
        largest_area_ratio = largest_area / total_area
        
        # 連結成分の数による評価
        if len(contours) == 1:
            contour_score = 1.0  # 単一連結成分
        elif len(contours) <= 3:
            contour_score = 0.8  # 少数の断片
            issues.append(f"Character fragmented into {len(contours)} parts")
            recommendations.append("Improve segmentation to reduce fragmentation")
        else:
            contour_score = 0.5  # 多数の断片
            issues.append(f"Highly fragmented: {len(contours)} separate regions")
            recommendations.append("Apply morphological operations to connect regions")
        
        # 面積比による評価
        if 0.1 <= largest_area_ratio <= 0.6:
            area_score = 1.0
        elif largest_area_ratio < 0.1:
            area_score = 0.6
            issues.append(f"Character too small: {largest_area_ratio*100:.1f}% of image")
            recommendations.append("Increase character size or improve cropping")
        else:
            area_score = 0.8
        
        completeness_score = (contour_score * 0.6 + area_score * 0.4)
        
        metrics = {
            'completeness_contours': len(contours),
            'largest_area_ratio': largest_area_ratio,
            'fragmentation_level': max(0, len(contours) - 1) / 10.0  # 正規化
        }
        
        return completeness_score, metrics
    
    def _analyze_sharpness(self, gray: np.ndarray, issues: List[str], 
                          recommendations: List[str]) -> tuple:
        """鮮明度の分析"""
        # Laplacianバリアンスによる鮮明度測定
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # 鮮明度スコア
        if laplacian_variance >= self.min_sharpness * 2:
            sharpness_score = 1.0
        elif laplacian_variance >= self.min_sharpness:
            sharpness_score = 0.8
        elif laplacian_variance >= self.min_sharpness * 0.5:
            sharpness_score = 0.6
            issues.append(f"Moderate blur detected: {laplacian_variance:.1f}")
            recommendations.append("Improve image sharpness")
        else:
            sharpness_score = 0.3
            issues.append(f"Significant blur detected: {laplacian_variance:.1f}")
            recommendations.append("Apply sharpening filter or use higher resolution image")
        
        # Sobelグラディエントによる追加評価
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        gradient_score = min(1.0, gradient_mean / 50.0)  # 50で正規化
        
        final_sharpness_score = (sharpness_score * 0.7 + gradient_score * 0.3)
        
        metrics = {
            'laplacian_variance': laplacian_variance,
            'gradient_magnitude_mean': gradient_mean,
            'gradient_score': gradient_score
        }
        
        return final_sharpness_score, metrics
    
    def _analyze_size_adequacy(self, image: np.ndarray, issues: List[str], 
                              recommendations: List[str]) -> tuple:
        """サイズ適正性の分析"""
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # 解像度評価
        if total_pixels >= 250000:  # 500x500以上
            resolution_score = 1.0
        elif total_pixels >= 100000:  # 316x316以上
            resolution_score = 0.8
        elif total_pixels >= 40000:   # 200x200以上
            resolution_score = 0.6
        else:
            resolution_score = 0.4
            issues.append(f"Low resolution: {w}x{h} ({total_pixels} pixels)")
            recommendations.append("Use higher resolution image for better quality")
        
        # アスペクト比評価
        aspect_ratio = h / w
        if 1.0 <= aspect_ratio <= 3.0:  # 正方形〜縦長（キャラクターに適切）
            aspect_score = 1.0
        elif 0.5 <= aspect_ratio < 1.0:  # 横長
            aspect_score = 0.7
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f} (too wide)")
            recommendations.append("Check for horizontal cropping issues")
        elif aspect_ratio > 3.0:  # 極端な縦長
            aspect_score = 0.8
            issues.append(f"Very tall aspect ratio: {aspect_ratio:.2f}")
        else:
            aspect_score = 0.5
            issues.append(f"Extreme aspect ratio: {aspect_ratio:.2f}")
        
        size_adequacy_score = (resolution_score * 0.6 + aspect_score * 0.4)
        
        metrics = {
            'image_width': w,
            'image_height': h,
            'total_pixels': total_pixels,
            'aspect_ratio': aspect_ratio,
            'resolution_score': resolution_score
        }
        
        return size_adequacy_score, metrics
    
    def _analyze_brightness_quality(self, gray: np.ndarray, issues: List[str], 
                                   recommendations: List[str]) -> tuple:
        """明度品質の分析"""
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        min_bright, max_bright = self.ideal_brightness_range
        
        # 明度レベル評価
        if min_bright <= mean_brightness <= max_bright:
            brightness_score = 1.0
        elif mean_brightness < min_bright:
            if mean_brightness < min_bright * 0.5:
                brightness_score = 0.3
                issues.append(f"Very dark image: {mean_brightness:.1f}")
                recommendations.append("Increase brightness significantly")
            else:
                brightness_score = 0.6
                issues.append(f"Dark image: {mean_brightness:.1f}")
                recommendations.append("Increase brightness")
        else:  # mean_brightness > max_bright
            if mean_brightness > max_bright * 1.2:
                brightness_score = 0.4
                issues.append(f"Very bright image: {mean_brightness:.1f}")
                recommendations.append("Reduce brightness or avoid overexposure")
            else:
                brightness_score = 0.7
                issues.append(f"Bright image: {mean_brightness:.1f}")
        
        # 明度分布評価
        if brightness_std > 40:
            distribution_score = 1.0  # 良好な明度分布
        elif brightness_std > 20:
            distribution_score = 0.8
        else:
            distribution_score = 0.6
            issues.append(f"Low brightness variation: {brightness_std:.1f}")
            recommendations.append("Improve contrast to increase detail visibility")
        
        final_brightness_score = (brightness_score * 0.7 + distribution_score * 0.3)
        
        metrics = {
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'brightness_range': float(np.max(gray) - np.min(gray))
        }
        
        return final_brightness_score, metrics
    
    def _analyze_contrast_quality(self, gray: np.ndarray, issues: List[str], 
                                 recommendations: List[str]) -> tuple:
        """コントラスト品質の分析"""
        contrast = np.std(gray)
        
        # RMS コントラスト
        rms_contrast = contrast
        
        # Michelson コントラスト
        max_val, min_val = np.max(gray), np.min(gray)
        if (max_val + min_val) > 0:
            michelson_contrast = (max_val - min_val) / (max_val + min_val)
        else:
            michelson_contrast = 0.0
        
        # コントラストスコア
        if contrast >= self.min_contrast * 2:
            contrast_score = 1.0
        elif contrast >= self.min_contrast:
            contrast_score = 0.8
        elif contrast >= self.min_contrast * 0.5:
            contrast_score = 0.6
            issues.append(f"Low contrast: {contrast:.1f}")
            recommendations.append("Increase image contrast")
        else:
            contrast_score = 0.3
            issues.append(f"Very low contrast: {contrast:.1f}")
            recommendations.append("Significantly increase contrast for better detail")
        
        metrics = {
            'rms_contrast': rms_contrast,
            'michelson_contrast': michelson_contrast,
            'dynamic_range': float(max_val - min_val)
        }
        
        return contrast_score, metrics
    
    def _analyze_noise_level(self, gray: np.ndarray, issues: List[str], 
                            recommendations: List[str]) -> tuple:
        """ノイズレベルの分析"""
        # ガウシアンブラーとの差分でノイズを推定
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_map = cv2.absdiff(gray, blurred)
        noise_level = np.mean(noise_map)
        noise_std = np.std(noise_map)
        
        # ノイズスコア（ノイズが少ないほど高スコア）
        if noise_level <= 5.0:
            noise_score = 1.0
        elif noise_level <= 10.0:
            noise_score = 0.8
        elif noise_level <= 20.0:
            noise_score = 0.6
            issues.append(f"Moderate noise detected: {noise_level:.1f}")
            recommendations.append("Apply noise reduction")
        else:
            noise_score = 0.3
            issues.append(f"High noise level: {noise_level:.1f}")
            recommendations.append("Apply strong noise reduction or use cleaner source")
        
        # ノイズパターンの一様性
        if noise_std < noise_level * 0.3:  # 一様なノイズ
            pattern_score = 0.8
        else:  # 不均一なノイズ（ストライプなど）
            pattern_score = 0.6
            issues.append("Non-uniform noise pattern detected")
        
        final_noise_score = (noise_score * 0.8 + pattern_score * 0.2)
        
        metrics = {
            'noise_level': noise_level,
            'noise_std': noise_std,
            'noise_uniformity': noise_std / (noise_level + 1e-8)
        }
        
        return final_noise_score, metrics
    
    def calculate_completeness_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """完全性メトリクス計算（テスト用メソッド）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        non_zero_pixels = np.sum(gray > 10)
        total_pixels = h * w
        
        # アスペクト比スコア
        aspect_ratio = h / w
        if 1.5 <= aspect_ratio <= 3.0:  # 理想的な縦長比率
            aspect_ratio_score = 1.0
        elif 1.0 <= aspect_ratio < 1.5 or 3.0 < aspect_ratio <= 4.0:
            aspect_ratio_score = 0.8
        else:
            aspect_ratio_score = 0.6
        
        # 垂直カバレッジスコア
        y_coords = np.where(gray > 10)[0]
        if len(y_coords) > 0:
            vertical_coverage = (np.max(y_coords) - np.min(y_coords) + 1) / h
        else:
            vertical_coverage = 0.0
        vertical_coverage_score = min(1.0, vertical_coverage * 1.2)
        
        # 水平カバレッジスコア
        x_coords = np.where(gray > 10)[1]
        if len(x_coords) > 0:
            horizontal_coverage = (np.max(x_coords) - np.min(x_coords) + 1) / w
        else:
            horizontal_coverage = 0.0
        horizontal_coverage_score = min(1.0, horizontal_coverage * 1.5)
        
        # 全体完全性スコア
        overall_completeness = (
            aspect_ratio_score * 0.3 +
            vertical_coverage_score * 0.4 +
            horizontal_coverage_score * 0.3
        )
        
        return {
            'aspect_ratio_score': aspect_ratio_score,
            'vertical_coverage_score': vertical_coverage_score,
            'horizontal_coverage_score': horizontal_coverage_score,
            'overall_completeness': overall_completeness,
            'aspect_ratio': aspect_ratio,
            'vertical_coverage': vertical_coverage,
            'horizontal_coverage': horizontal_coverage
        }