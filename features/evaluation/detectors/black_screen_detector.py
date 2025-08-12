"""
QI-002: 黒画面検出器 (BlackScreenDetector)

低明度・黒画面の検出を行う機能を提供します。
QI-002で報告された「24枚中3枚（12.5%）の黒画面問題」の解決を目指します。
"""

import numpy as np
from typing import NamedTuple, Optional
from .brightness_analyzer import BrightnessAnalyzer


class BlackScreenDetectionResult(NamedTuple):
    """黒画面検出結果"""
    is_black_screen: bool
    brightness_score: float
    confidence: float
    reason: str
    additional_info: Optional[dict] = None


class BlackScreenDetector:
    """黒画面検出を行うクラス"""
    
    def __init__(self, brightness_threshold: float = 20.0):
        """
        BlackScreenDetector の初期化
        
        Args:
            brightness_threshold: 黒画面判定の明度閾値（デフォルト: 20.0）
        """
        self.brightness_threshold = brightness_threshold
        self.analyzer = BrightnessAnalyzer()
    
    def detect(self, image: np.ndarray) -> BlackScreenDetectionResult:
        """
        画像が黒画面かどうかを検出
        
        Args:
            image: 入力画像 (H, W, C) numpy配列
            
        Returns:
            BlackScreenDetectionResult: 検出結果
        """
        try:
            # 基本明度計算
            brightness = self.analyzer.calculate_brightness(image)
            
            # 明度分布解析
            distribution = self.analyzer.analyze_brightness_distribution(image)
            
            # 暗い領域の分析
            dark_regions = self.analyzer.analyze_dark_regions(image, self.brightness_threshold)
            
            # 黒画面判定ロジック
            is_black_screen, confidence, reason = self._evaluate_black_screen(
                brightness, distribution, dark_regions
            )
            
            additional_info = {
                'brightness_distribution': distribution,
                'dark_regions_analysis': dark_regions,
                'threshold_used': self.brightness_threshold
            }
            
            return BlackScreenDetectionResult(
                is_black_screen=is_black_screen,
                brightness_score=brightness,
                confidence=confidence,
                reason=reason,
                additional_info=additional_info
            )
            
        except Exception as e:
            # エラー時のフォールバック
            return BlackScreenDetectionResult(
                is_black_screen=False,
                brightness_score=0.0,
                confidence=0.0,
                reason=f"Detection error: {str(e)}"
            )
    
    def _evaluate_black_screen(self, brightness: float, distribution: dict, 
                              dark_regions: dict) -> tuple[bool, float, str]:
        """
        黒画面判定の内部ロジック
        
        Args:
            brightness: 平均明度
            distribution: 明度分布統計
            dark_regions: 暗い領域統計
            
        Returns:
            (is_black_screen, confidence, reason)
        """
        # 1. 完全黒画面の判定（明度5以下）
        if brightness <= 5.0:
            return True, 0.95, "Pure black detected"
        
        # 2. 非常に暗い画像の判定（明度20以下）
        elif brightness <= self.brightness_threshold:
            # 暗い領域が80%以上を占める場合
            if dark_regions['dark_pixel_ratio'] >= 0.8:
                confidence = 0.9 - (brightness / self.brightness_threshold) * 0.1
                return True, confidence, "Very dark image with high dark pixel ratio"
            else:
                confidence = 0.7 - (brightness / self.brightness_threshold) * 0.2
                return True, confidence, "Dark image below threshold"
        
        # 3. 境界的ケースの判定（明度20-40）
        elif brightness <= 40.0:
            # 標準偏差が非常に小さい（ほぼ均一に暗い）
            if distribution['std'] < 10.0 and dark_regions['dark_pixel_ratio'] >= 0.9:
                confidence = 0.8 - ((brightness - self.brightness_threshold) / 20.0) * 0.3
                return True, confidence, "Uniformly dark image"
            else:
                confidence = 0.95 + ((brightness - 40.0) / 60.0) * 0.05
                return False, confidence, "Borderline brightness, but not black screen"
        
        # 4. 通常の明度の画像
        elif brightness <= 120.0:
            confidence = 0.9 + min((brightness - 40.0) / 80.0 * 0.1, 0.1)
            return False, confidence, "Normal brightness detected"
        
        # 5. 明るい画像
        else:
            confidence = 0.95
            return False, confidence, "Bright image, clearly not black screen"
    
    def detect_batch(self, images: list[np.ndarray]) -> list[BlackScreenDetectionResult]:
        """
        複数画像の一括黒画面検出
        
        Args:
            images: 入力画像のリスト
            
        Returns:
            検出結果のリスト
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        
        return results
    
    def get_black_screen_statistics(self, results: list[BlackScreenDetectionResult]) -> dict:
        """
        黒画面検出結果の統計情報を取得
        
        Args:
            results: 検出結果のリスト
            
        Returns:
            統計情報の辞書
        """
        if not results:
            return {}
        
        black_screen_count = sum(1 for r in results if r.is_black_screen)
        total_count = len(results)
        
        brightness_scores = [r.brightness_score for r in results]
        confidence_scores = [r.confidence for r in results]
        
        black_brightness_scores = [r.brightness_score for r in results if r.is_black_screen]
        
        stats = {
            'total_images': total_count,
            'black_screen_count': black_screen_count,
            'black_screen_ratio': black_screen_count / total_count,
            'average_brightness': np.mean(brightness_scores),
            'average_confidence': np.mean(confidence_scores),
            'brightness_std': np.std(brightness_scores),
            'black_screen_avg_brightness': np.mean(black_brightness_scores) if black_brightness_scores else 0.0,
            'threshold_used': self.brightness_threshold
        }
        
        return stats
    
    def adjust_threshold(self, new_threshold: float):
        """
        検出閾値の動的調整
        
        Args:
            new_threshold: 新しい明度閾値
        """
        self.brightness_threshold = max(0.0, min(255.0, new_threshold))
    
    def calibrate_threshold(self, labeled_images: list[tuple[np.ndarray, bool]]) -> float:
        """
        ラベル付きデータを使用した閾値キャリブレーション
        
        Args:
            labeled_images: (画像, is_black_screen)のタプルのリスト
            
        Returns:
            最適化された閾値
        """
        if not labeled_images:
            return self.brightness_threshold
        
        # 各画像の明度を計算
        brightness_values = []
        labels = []
        
        for image, is_black in labeled_images:
            brightness = self.analyzer.calculate_brightness(image)
            brightness_values.append(brightness)
            labels.append(is_black)
        
        # 最適閾値の探索（F1スコアを最大化）
        best_threshold = self.brightness_threshold
        best_f1_score = 0.0
        
        for threshold in np.arange(5.0, 50.0, 1.0):
            tp = sum(1 for i, brightness in enumerate(brightness_values) 
                    if brightness <= threshold and labels[i])
            fp = sum(1 for i, brightness in enumerate(brightness_values) 
                    if brightness <= threshold and not labels[i])
            fn = sum(1 for i, brightness in enumerate(brightness_values) 
                    if brightness > threshold and labels[i])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_threshold = threshold
        
        self.brightness_threshold = best_threshold
        return best_threshold