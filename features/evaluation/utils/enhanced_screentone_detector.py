#!/usr/bin/env python3
"""
Enhanced Screentone Detection System
Phase 1 P1-004: スクリーントーン検出アルゴリズム強化

マルチスケール解析、パターン分類、機械学習ベース特徴量を統合した
高精度スクリーントーン検出システム
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("⚠️ PyWavelets not available, wavelet analysis disabled")

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not available, LBP features disabled")


class ScreentoneType(Enum):
    """スクリーントーンの種類"""
    NONE = "none"
    DOT_PATTERN = "dot_pattern"
    LINE_PATTERN = "line_pattern"
    GRADIENT_PATTERN = "gradient_pattern"
    COMPLEX_PATTERN = "complex_pattern"
    NOISE_PATTERN = "noise_pattern"


@dataclass
class ScreentoneDetectionResult:
    """スクリーントーン検出結果"""
    has_screentone: bool
    screentone_type: ScreentoneType
    confidence: float
    mask: np.ndarray
    pattern_density: float
    dominant_frequency: float
    orientation: float
    coverage_ratio: float
    quality_score: float
    reasoning: str


class ScreentoneFeatureExtractor:
    """スクリーントーン特徴量抽出器"""
    
    def __init__(self):
        """特徴量抽出器の初期化"""
        self.logger = logging.getLogger(__name__)
        
        # Gaborフィルタパラメータ
        self.gabor_params = {
            'frequencies': [0.1, 0.2, 0.3, 0.4],
            'orientations': [0, 30, 60, 90, 120, 150],
            'sigma_x': 2.0,
            'sigma_y': 2.0
        }
        
        # LBPパラメータ
        self.lbp_params = {
            'radius': 3,
            'n_points': 24,
            'method': 'uniform'
        }
    
    def extract_fft_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        FFT特徴量の抽出
        
        Args:
            image: グレースケール画像
            
        Returns:
            FFT特徴量の辞書
        """
        # FFT変換
        f_transform = np.fft.fft2(image.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 周波数特徴量
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # 各周波数帯域のパワー
        low_freq_power = self._calculate_freq_power(magnitude_spectrum, center_h, center_w, 0, h//8)
        mid_freq_power = self._calculate_freq_power(magnitude_spectrum, center_h, center_w, h//8, h//4)
        high_freq_power = self._calculate_freq_power(magnitude_spectrum, center_h, center_w, h//4, h//2)
        
        # 周波数分布の特徴
        total_power = low_freq_power + mid_freq_power + high_freq_power
        
        # 主要周波数の検出
        dominant_freq = self._find_dominant_frequency(magnitude_spectrum, center_h, center_w)
        
        # 周期性スコア
        periodicity_score = self._calculate_periodicity_score(magnitude_spectrum)
        
        return {
            'low_freq_power': low_freq_power,
            'mid_freq_power': mid_freq_power,
            'high_freq_power': high_freq_power,
            'freq_ratio_mid_low': mid_freq_power / (low_freq_power + 1e-6),
            'freq_ratio_high_mid': high_freq_power / (mid_freq_power + 1e-6),
            'dominant_frequency': dominant_freq,
            'periodicity_score': periodicity_score,
            'spectral_centroid': self._calculate_spectral_centroid(magnitude_spectrum),
            'spectral_spread': self._calculate_spectral_spread(magnitude_spectrum)
        }
    
    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Gaborフィルタ特徴量の抽出
        
        Args:
            image: グレースケール画像
            
        Returns:
            Gabor特徴量の辞書
        """
        gabor_responses = []
        orientations = []
        
        for freq in self.gabor_params['frequencies']:
            for angle in self.gabor_params['orientations']:
                # Gaborフィルタの適用
                real, _ = cv2.getGaborKernel(
                    (21, 21), 
                    self.gabor_params['sigma_x'], 
                    np.radians(angle), 
                    2 * np.pi * freq, 
                    0.5, 
                    0, 
                    ktype=cv2.CV_32F
                ), None
                
                filtered = cv2.filter2D(image, cv2.CV_8UC3, real)
                response = np.var(filtered)
                gabor_responses.append(response)
                orientations.append(angle)
        
        gabor_responses = np.array(gabor_responses)
        
        # 統計特徴量
        max_response = np.max(gabor_responses)
        mean_response = np.mean(gabor_responses)
        std_response = np.std(gabor_responses)
        
        # 主要方向の検出
        max_idx = np.argmax(gabor_responses)
        dominant_orientation = orientations[max_idx]
        
        # 方向性の一様性
        orientation_uniformity = std_response / (mean_response + 1e-6)
        
        return {
            'gabor_max_response': max_response,
            'gabor_mean_response': mean_response,
            'gabor_std_response': std_response,
            'dominant_orientation': dominant_orientation,
            'orientation_uniformity': orientation_uniformity,
            'gabor_energy': np.sum(gabor_responses ** 2),
            'gabor_entropy': self._calculate_entropy(gabor_responses)
        }
    
    def extract_lbp_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        LBP（Local Binary Pattern）特徴量の抽出
        
        Args:
            image: グレースケール画像
            
        Returns:
            LBP特徴量の辞書
        """
        if not SKIMAGE_AVAILABLE:
            # フォールバック: 簡易テクスチャ分析
            return self._fallback_texture_analysis(image)
        
        # LBP計算
        lbp = local_binary_pattern(
            image, 
            self.lbp_params['n_points'], 
            self.lbp_params['radius'], 
            method=self.lbp_params['method']
        )
        
        # LBPヒストグラム
        n_bins = self.lbp_params['n_points'] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # 統計特徴量
        lbp_mean = np.mean(hist)
        lbp_std = np.std(hist)
        lbp_entropy = self._calculate_entropy(hist)
        
        # 一様パターンの割合
        uniform_patterns = np.sum(hist[:self.lbp_params['n_points']])
        
        # テクスチャの複雑さ
        texture_complexity = lbp_entropy
        
        return {
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
            'lbp_entropy': lbp_entropy,
            'uniform_ratio': uniform_patterns,
            'texture_complexity': texture_complexity,
            'lbp_contrast': np.var(lbp)
        }
    
    def extract_wavelet_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        ウェーブレット特徴量の抽出
        
        Args:
            image: グレースケール画像
            
        Returns:
            ウェーブレット特徴量の辞書
        """
        if not PYWT_AVAILABLE:
            # フォールバック: Sobelフィルタベースの方向性分析
            return self._fallback_directional_analysis(image)
        
        # 2D離散ウェーブレット変換
        coeffs = pywt.dwt2(image, 'db4')
        cA, (cH, cV, cD) = coeffs
        
        # 各サブバンドのエネルギー
        energy_approx = np.sum(cA ** 2)
        energy_horizontal = np.sum(cH ** 2)
        energy_vertical = np.sum(cV ** 2)
        energy_diagonal = np.sum(cD ** 2)
        
        total_energy = energy_approx + energy_horizontal + energy_vertical + energy_diagonal
        
        # エネルギー比率
        ratio_h = energy_horizontal / (total_energy + 1e-6)
        ratio_v = energy_vertical / (total_energy + 1e-6)
        ratio_d = energy_diagonal / (total_energy + 1e-6)
        
        # 方向性特徴量
        directionality = max(ratio_h, ratio_v) / (ratio_d + 1e-6)
        
        return {
            'wavelet_energy_h': energy_horizontal,
            'wavelet_energy_v': energy_vertical,
            'wavelet_energy_d': energy_diagonal,
            'wavelet_ratio_h': ratio_h,
            'wavelet_ratio_v': ratio_v,
            'wavelet_ratio_d': ratio_d,
            'wavelet_directionality': directionality,
            'wavelet_total_energy': total_energy
        }
    
    def extract_spatial_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        空間的特徴量の抽出
        
        Args:
            image: グレースケール画像
            
        Returns:
            空間特徴量の辞書
        """
        # 局所分散
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # エッジ密度
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # グラデーション特徴
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 統計特徴量
        variance_mean = np.mean(local_variance)
        variance_std = np.std(local_variance)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        
        # 規則性スコア
        regularity_score = variance_std / (variance_mean + 1e-6)
        
        return {
            'local_variance_mean': variance_mean,
            'local_variance_std': variance_std,
            'edge_density': edge_density,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'regularity_score': regularity_score,
            'contrast': np.std(image),
            'homogeneity': 1.0 / (1.0 + variance_mean)
        }
    
    def _calculate_freq_power(self, magnitude_spectrum: np.ndarray, center_h: int, center_w: int, 
                            inner_radius: int, outer_radius: int) -> float:
        """周波数帯域のパワーを計算"""
        h, w = magnitude_spectrum.shape
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2 >= inner_radius**2) & \
               ((x - center_w)**2 + (y - center_h)**2 < outer_radius**2)
        return np.sum(magnitude_spectrum[mask])
    
    def _find_dominant_frequency(self, magnitude_spectrum: np.ndarray, center_h: int, center_w: int) -> float:
        """主要周波数を検出"""
        h, w = magnitude_spectrum.shape
        max_radius = min(center_h, center_w, h - center_h, w - center_w)
        
        # 極座標での周波数分析
        radial_profile = []
        for radius in range(1, max_radius):
            power = self._calculate_freq_power(magnitude_spectrum, center_h, center_w, radius-1, radius)
            radial_profile.append(power)
        
        if radial_profile:
            max_idx = np.argmax(radial_profile)
            return max_idx / max_radius  # 正規化された周波数
        return 0.0
    
    def _calculate_periodicity_score(self, magnitude_spectrum: np.ndarray) -> float:
        """周期性スコアを計算"""
        # スペクトラムのピーク検出
        mean_magnitude = np.mean(magnitude_spectrum)
        std_magnitude = np.std(magnitude_spectrum)
        threshold = mean_magnitude + 2 * std_magnitude
        
        peaks = magnitude_spectrum > threshold
        peak_count = np.sum(peaks)
        
        # 周期性は離散的なピークの存在で判断
        total_pixels = magnitude_spectrum.size
        return min(peak_count / total_pixels * 100, 1.0)
    
    def _calculate_spectral_centroid(self, magnitude_spectrum: np.ndarray) -> float:
        """スペクトラル重心を計算"""
        h, w = magnitude_spectrum.shape
        total_magnitude = np.sum(magnitude_spectrum)
        
        if total_magnitude == 0:
            return 0.0
        
        # 重心計算
        y_indices, x_indices = np.mgrid[:h, :w]
        centroid_y = np.sum(y_indices * magnitude_spectrum) / total_magnitude
        centroid_x = np.sum(x_indices * magnitude_spectrum) / total_magnitude
        
        # 中心からの距離
        center_h, center_w = h // 2, w // 2
        distance = np.sqrt((centroid_y - center_h)**2 + (centroid_x - center_w)**2)
        
        return distance / max(center_h, center_w)  # 正規化
    
    def _calculate_spectral_spread(self, magnitude_spectrum: np.ndarray) -> float:
        """スペクトラルスプレッドを計算"""
        centroid = self._calculate_spectral_centroid(magnitude_spectrum)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y_indices, x_indices = np.mgrid[:h, :w]
        distances = np.sqrt((y_indices - center_h)**2 + (x_indices - center_w)**2)
        
        total_magnitude = np.sum(magnitude_spectrum)
        if total_magnitude == 0:
            return 0.0
        
        spread = np.sum(((distances / max(center_h, center_w)) - centroid)**2 * magnitude_spectrum) / total_magnitude
        return np.sqrt(spread)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """エントロピーを計算"""
        # 正規化
        data = data + 1e-12  # ゼロ除算回避
        data = data / np.sum(data)
        
        # エントロピー計算
        entropy = -np.sum(data * np.log2(data))
        return entropy
    
    def _fallback_texture_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """LBPが利用できない場合のフォールバックテクスチャ分析"""
        # 局所分散ベースのテクスチャ分析
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # 統計特徴量
        variance_mean = np.mean(local_variance)
        variance_std = np.std(local_variance)
        
        # エントロピー近似（ヒストグラムベース）
        hist, _ = np.histogram(image, bins=32, range=(0, 256), density=True)
        hist = hist + 1e-12
        entropy_approx = -np.sum(hist * np.log2(hist))
        
        # 一様パターン近似
        uniform_ratio = 1.0 - (variance_std / (variance_mean + 1e-6))
        uniform_ratio = max(0.0, min(1.0, uniform_ratio))
        
        return {
            'lbp_mean': variance_mean / 1000.0,  # スケール調整
            'lbp_std': variance_std / 1000.0,
            'lbp_entropy': entropy_approx,
            'uniform_ratio': uniform_ratio,
            'texture_complexity': entropy_approx,
            'lbp_contrast': np.var(image) / 1000.0
        }
    
    def _fallback_directional_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """ウェーブレットが利用できない場合のフォールバック方向性分析"""
        # Sobelフィルタによる方向性分析
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # エネルギー計算
        energy_h = np.sum(grad_x ** 2)
        energy_v = np.sum(grad_y ** 2)
        energy_total = energy_h + energy_v
        
        # 比率計算
        ratio_h = energy_h / (energy_total + 1e-6)
        ratio_v = energy_v / (energy_total + 1e-6)
        ratio_d = 0.5 * (ratio_h + ratio_v)  # 対角成分は平均で近似
        
        # 方向性
        directionality = max(ratio_h, ratio_v) / (ratio_d + 1e-6)
        
        return {
            'wavelet_energy_h': energy_h,
            'wavelet_energy_v': energy_v,
            'wavelet_energy_d': energy_total * 0.5,  # 近似
            'wavelet_ratio_h': ratio_h,
            'wavelet_ratio_v': ratio_v,
            'wavelet_ratio_d': ratio_d,
            'wavelet_directionality': directionality,
            'wavelet_total_energy': energy_total
        }


class ScreentonePatternClassifier:
    """スクリーントーンパターン分類器"""
    
    def __init__(self):
        """分類器の初期化"""
        self.logger = logging.getLogger(__name__)
        
        # パターン分類の閾値（調整済み）
        self.thresholds = {
            'dot_pattern': {
                'periodicity_min': 0.15,  # 緩和: 0.3 -> 0.15
                'regularity_max': 3.0,    # 緩和: 2.0 -> 3.0
                'gabor_energy_min': 500   # 緩和: 1000 -> 500
            },
            'line_pattern': {
                'directionality_min': 1.5,    # 緩和: 2.0 -> 1.5
                'orientation_consistency': 0.5, # 緩和: 0.7 -> 0.5
                'edge_density_min': 0.08       # 緩和: 0.1 -> 0.08
            },
            'gradient_pattern': {
                'gradient_consistency': 0.6,   # 緩和: 0.8 -> 0.6
                'low_freq_dominance': 0.4,     # 緩和: 0.6 -> 0.4
                'variance_uniformity': 0.3     # 緩和: 0.5 -> 0.3
            },
            'noise_pattern': {
                'entropy_min': 3.5,      # 緩和: 4.0 -> 3.5
                'regularity_min': 3.0,   # 緩和: 5.0 -> 3.0
                'high_freq_ratio': 0.3   # 緩和: 0.4 -> 0.3
            }
        }
    
    def classify_pattern(self, features: Dict[str, float]) -> Tuple[ScreentoneType, float]:
        """
        特徴量からパターンを分類
        
        Args:
            features: 抽出された特徴量
            
        Returns:
            (パターンタイプ, 信頼度)
        """
        scores = {}
        
        # ドットパターンの評価
        scores[ScreentoneType.DOT_PATTERN] = self._evaluate_dot_pattern(features)
        
        # 線パターンの評価
        scores[ScreentoneType.LINE_PATTERN] = self._evaluate_line_pattern(features)
        
        # グラデーションパターンの評価
        scores[ScreentoneType.GRADIENT_PATTERN] = self._evaluate_gradient_pattern(features)
        
        # ノイズパターンの評価
        scores[ScreentoneType.NOISE_PATTERN] = self._evaluate_noise_pattern(features)
        
        # 最高スコアのパターンを選択
        best_pattern = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_pattern]
        
        # 信頼度が低い場合はNONEを返す
        if best_score < 0.5:
            return ScreentoneType.NONE, 0.0
        
        return best_pattern, best_score
    
    def _evaluate_dot_pattern(self, features: Dict[str, float]) -> float:
        """ドットパターンの評価"""
        score = 0.0
        
        # 周期性（重要）
        periodicity = features.get('periodicity_score', 0)
        if periodicity > self.thresholds['dot_pattern']['periodicity_min']:
            score += 0.4 * min(periodicity / 0.8, 1.0)
        
        # 規則性
        regularity = features.get('regularity_score', 0)
        if regularity < self.thresholds['dot_pattern']['regularity_max']:
            score += 0.3 * (1.0 - regularity / self.thresholds['dot_pattern']['regularity_max'])
        
        # Gabor応答（テクスチャの強度）
        gabor_energy = features.get('gabor_energy', 0)
        if gabor_energy > self.thresholds['dot_pattern']['gabor_energy_min']:
            score += 0.2 * min(gabor_energy / 5000, 1.0)
        
        # 一様パターン比率（LBP）
        uniform_ratio = features.get('uniform_ratio', 0)
        score += 0.1 * uniform_ratio
        
        return min(score, 1.0)
    
    def _evaluate_line_pattern(self, features: Dict[str, float]) -> float:
        """線パターンの評価"""
        score = 0.0
        
        # 方向性（重要）
        directionality = features.get('wavelet_directionality', 0)
        if directionality > self.thresholds['line_pattern']['directionality_min']:
            score += 0.3 * min(directionality / 3.0, 1.0)
        
        # 方向の一貫性
        orientation_uniformity = features.get('orientation_uniformity', 0)
        if orientation_uniformity < 1.0:  # 低い値ほど一貫性が高い
            score += 0.25 * (1.0 - orientation_uniformity)
        
        # エッジ密度（線パターンの強い指標）
        edge_density = features.get('edge_density', 0)
        if edge_density > self.thresholds['line_pattern']['edge_density_min']:
            score += 0.3 * min(edge_density / 0.25, 1.0)
        
        # 水平・垂直方向の強さ
        h_ratio = features.get('wavelet_ratio_h', 0)
        v_ratio = features.get('wavelet_ratio_v', 0)
        directional_strength = max(h_ratio, v_ratio)
        score += 0.15 * directional_strength
        
        return min(score, 1.0)
    
    def _evaluate_gradient_pattern(self, features: Dict[str, float]) -> float:
        """グラデーションパターンの評価"""
        score = 0.0
        
        # 低周波数の優位性
        low_freq = features.get('low_freq_power', 0)
        mid_freq = features.get('mid_freq_power', 0)
        high_freq = features.get('high_freq_power', 0)
        total_freq = low_freq + mid_freq + high_freq
        
        if total_freq > 0:
            low_dominance = low_freq / total_freq
            if low_dominance > self.thresholds['gradient_pattern']['low_freq_dominance']:
                score += 0.4 * min(low_dominance / 0.8, 1.0)
        
        # グラデーションの一貫性
        gradient_std = features.get('gradient_std', 0)
        gradient_mean = features.get('gradient_mean', 0)
        if gradient_mean > 0:
            gradient_consistency = 1.0 - (gradient_std / gradient_mean)
            if gradient_consistency > self.thresholds['gradient_pattern']['gradient_consistency']:
                score += 0.3 * gradient_consistency
        
        # 局所分散の均一性
        variance_std = features.get('local_variance_std', 0)
        variance_mean = features.get('local_variance_mean', 0)
        if variance_mean > 0:
            variance_uniformity = 1.0 - (variance_std / variance_mean)
            score += 0.2 * variance_uniformity
        
        # ホモジェニティ
        homogeneity = features.get('homogeneity', 0)
        score += 0.1 * homogeneity
        
        return min(score, 1.0)
    
    def _evaluate_noise_pattern(self, features: Dict[str, float]) -> float:
        """ノイズパターンの評価"""
        score = 0.0
        
        # 高エントロピー（不規則性）
        lbp_entropy = features.get('lbp_entropy', 0)
        if lbp_entropy > self.thresholds['noise_pattern']['entropy_min']:
            score += 0.4 * min(lbp_entropy / 6.0, 1.0)
        
        # 高い不規則性
        regularity = features.get('regularity_score', 0)
        if regularity > self.thresholds['noise_pattern']['regularity_min']:
            score += 0.3 * min(regularity / 10.0, 1.0)
        
        # 高周波数成分の優位性
        high_freq_ratio = features.get('freq_ratio_high_mid', 0)
        if high_freq_ratio > self.thresholds['noise_pattern']['high_freq_ratio']:
            score += 0.2 * min(high_freq_ratio / 1.0, 1.0)
        
        # テクスチャの複雑さ
        texture_complexity = features.get('texture_complexity', 0)
        score += 0.1 * min(texture_complexity / 5.0, 1.0)
        
        return min(score, 1.0)


class EnhancedScreentoneDetector:
    """
    改良版スクリーントーン検出器
    Phase 1 P1-004: マルチスケール解析とパターン分類を統合
    """
    
    def __init__(self):
        """検出器の初期化"""
        self.logger = logging.getLogger(__name__)
        
        self.feature_extractor = ScreentoneFeatureExtractor()
        self.pattern_classifier = ScreentonePatternClassifier()
        
        # 検出パラメータ（調整済み）
        self.detection_params = {
            'min_confidence': 0.5,  # 緩和: 0.6 -> 0.5
            'min_coverage': 0.05,   # 緩和: 0.1 -> 0.05
            'quality_threshold': 0.6, # 緩和: 0.7 -> 0.6
            'adaptive_threshold': True
        }
    
    def detect_screentone(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> ScreentoneDetectionResult:
        """
        スクリーントーン検出のメイン関数
        
        Args:
            image: 入力画像（BGR or グレースケール）
            roi_mask: 検出対象領域マスク（Noneの場合は全領域）
            
        Returns:
            スクリーントーン検出結果
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ROI適用
        if roi_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 特徴量抽出
        fft_features = self.feature_extractor.extract_fft_features(gray)
        gabor_features = self.feature_extractor.extract_gabor_features(gray)
        lbp_features = self.feature_extractor.extract_lbp_features(gray)
        wavelet_features = self.feature_extractor.extract_wavelet_features(gray)
        spatial_features = self.feature_extractor.extract_spatial_features(gray)
        
        # 特徴量統合
        combined_features = {**fft_features, **gabor_features, **lbp_features, 
                           **wavelet_features, **spatial_features}
        
        # パターン分類
        screentone_type, type_confidence = self.pattern_classifier.classify_pattern(combined_features)
        
        # スクリーントーン検出の総合判定
        has_screentone = bool(screentone_type != ScreentoneType.NONE and type_confidence > self.detection_params['min_confidence'])
        
        # マスク生成
        screentone_mask = self._generate_screentone_mask(gray, screentone_type, combined_features)
        
        # カバレッジ計算
        coverage_ratio = np.sum(screentone_mask > 0) / screentone_mask.size
        
        # 品質スコア計算
        quality_score = self._calculate_quality_score(combined_features, type_confidence, coverage_ratio)
        
        # 推論の生成
        reasoning = self._generate_reasoning(screentone_type, type_confidence, combined_features, coverage_ratio)
        
        return ScreentoneDetectionResult(
            has_screentone=has_screentone,
            screentone_type=screentone_type,
            confidence=type_confidence,
            mask=screentone_mask,
            pattern_density=combined_features.get('periodicity_score', 0.0),
            dominant_frequency=combined_features.get('dominant_frequency', 0.0),
            orientation=combined_features.get('dominant_orientation', 0.0),
            coverage_ratio=coverage_ratio,
            quality_score=quality_score,
            reasoning=reasoning
        )
    
    def _generate_screentone_mask(self, image: np.ndarray, screentone_type: ScreentoneType, 
                                features: Dict[str, float]) -> np.ndarray:
        """
        スクリーントーンマスクの生成
        
        Args:
            image: グレースケール画像
            screentone_type: 検出されたパターンタイプ
            features: 抽出された特徴量
            
        Returns:
            スクリーントーンマスク
        """
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if screentone_type == ScreentoneType.NONE:
            return mask
        
        # パターンタイプに応じたマスク生成
        if screentone_type == ScreentoneType.DOT_PATTERN:
            mask = self._generate_dot_mask(image, features)
        elif screentone_type == ScreentoneType.LINE_PATTERN:
            mask = self._generate_line_mask(image, features)
        elif screentone_type == ScreentoneType.GRADIENT_PATTERN:
            mask = self._generate_gradient_mask(image, features)
        elif screentone_type == ScreentoneType.NOISE_PATTERN:
            mask = self._generate_noise_mask(image, features)
        
        # 後処理
        mask = self._refine_mask(mask, image)
        
        return mask
    
    def _generate_dot_mask(self, image: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """ドットパターン用マスク生成"""
        # 適応的閾値処理
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # モルフォロジー処理でドットを強調
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return 255 - mask  # ドット部分を白にする
    
    def _generate_line_mask(self, image: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """線パターン用マスク生成"""
        # エッジ検出
        edges = cv2.Canny(image, 50, 150)
        
        # 線構造を強調
        if features.get('wavelet_ratio_h', 0) > features.get('wavelet_ratio_v', 0):
            # 水平線優位
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        else:
            # 垂直線優位
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        
        mask = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        
        return mask
    
    def _generate_gradient_mask(self, image: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """グラデーションパターン用マスク生成"""
        # グラデーション領域の検出
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 適応的閾値でグラデーション領域を抽出
        mean_grad = np.mean(gradient_magnitude)
        threshold = mean_grad * 0.5
        
        mask = (gradient_magnitude > threshold).astype(np.uint8) * 255
        
        # 平滑化
        mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
        
        return mask
    
    def _generate_noise_mask(self, image: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """ノイズパターン用マスク生成"""
        # 局所分散による検出
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # 高分散領域をノイズパターンとして検出
        variance_threshold = np.mean(local_variance) + np.std(local_variance)
        mask = (local_variance > variance_threshold).astype(np.uint8) * 255
        
        return mask
    
    def _refine_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """マスクの精密化"""
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 小さな穴を埋める
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # エッジ平滑化
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    
    def _calculate_quality_score(self, features: Dict[str, float], type_confidence: float, 
                               coverage_ratio: float) -> float:
        """品質スコアの計算"""
        # 基本スコア（パターン分類の信頼度）
        quality_score = type_confidence * 0.6
        
        # カバレッジによる調整
        if 0.1 <= coverage_ratio <= 0.7:  # 適切なカバレッジ範囲
            quality_score += 0.2
        else:
            quality_score += 0.1 * (1.0 - abs(coverage_ratio - 0.4) / 0.4)
        
        # 特徴量の一貫性
        consistency_score = 0.0
        
        # 周期性と規則性の一貫性
        periodicity = features.get('periodicity_score', 0)
        regularity = features.get('regularity_score', 0)
        if periodicity > 0.3 and regularity < 3.0:
            consistency_score += 0.1
        
        # 方向性の一貫性
        orientation_uniformity = features.get('orientation_uniformity', 1.0)
        if orientation_uniformity < 0.5:
            consistency_score += 0.1
        
        quality_score += consistency_score
        
        return min(quality_score, 1.0)
    
    def _generate_reasoning(self, screentone_type: ScreentoneType, confidence: float, 
                          features: Dict[str, float], coverage: float) -> str:
        """推論の生成"""
        if screentone_type == ScreentoneType.NONE:
            return "スクリーントーンなし: パターン特徴量が検出閾値未満"
        
        reasoning_parts = []
        
        # パターンタイプの説明
        type_names = {
            ScreentoneType.DOT_PATTERN: "ドットパターン",
            ScreentoneType.LINE_PATTERN: "線パターン", 
            ScreentoneType.GRADIENT_PATTERN: "グラデーションパターン",
            ScreentoneType.NOISE_PATTERN: "ノイズパターン",
            ScreentoneType.COMPLEX_PATTERN: "複合パターン"
        }
        
        reasoning_parts.append(f"{type_names.get(screentone_type, '不明')}検出")
        
        # 信頼度の評価
        if confidence > 0.8:
            reasoning_parts.append("高信頼度")
        elif confidence > 0.6:
            reasoning_parts.append("中信頼度")
        else:
            reasoning_parts.append("低信頼度")
        
        # 主要特徴量の説明
        key_features = []
        
        if features.get('periodicity_score', 0) > 0.4:
            key_features.append(f"周期性({features['periodicity_score']:.2f})")
        
        if features.get('edge_density', 0) > 0.15:
            key_features.append(f"エッジ密度({features['edge_density']:.2f})")
        
        if features.get('dominant_frequency', 0) > 0.3:
            key_features.append(f"主要周波数({features['dominant_frequency']:.2f})")
        
        if key_features:
            reasoning_parts.append(f"特徴: {', '.join(key_features)}")
        
        # カバレッジの評価
        reasoning_parts.append(f"カバレッジ: {coverage:.1%}")
        
        return "; ".join(reasoning_parts)


# 便利関数
def detect_screentone_enhanced(image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> ScreentoneDetectionResult:
    """
    改良版スクリーントーン検出の便利関数
    
    Args:
        image: 入力画像
        roi_mask: 検出対象領域マスク
        
    Returns:
        スクリーントーン検出結果
    """
    detector = EnhancedScreentoneDetector()
    return detector.detect_screentone(image, roi_mask)