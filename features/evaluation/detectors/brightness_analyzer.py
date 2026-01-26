"""
QI-002: 明度解析器 (BrightnessAnalyzer)

画像の明度を計算し、分布を解析する機能を提供します。
黒画面検出の基盤となる明度計算ロジックを実装しています。
"""

import numpy as np
import cv2

from typing import Dict, List, Tuple


class BrightnessAnalyzer:
    """画像の明度解析を行うクラス"""

    def __init__(self):
        """BrightnessAnalyzer の初期化"""
        pass

    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        画像の平均明度を計算

        Args:
            image: 入力画像 (H, W, C) numpy配列

        Returns:
            平均明度値 (0.0-255.0)
        """
        if len(image.shape) == 3:
            # カラー画像の場合、グレースケールに変換
            # OpenCVの重み付け: 0.299*R + 0.587*G + 0.114*B
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # すでにグレースケールの場合
            gray = image

        # 平均明度の計算
        mean_brightness = float(np.mean(gray))
        return mean_brightness

    def analyze_brightness_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """
        画像の明度分布を解析

        Args:
            image: 入力画像 (H, W, C) numpy配列

        Returns:
            明度統計情報の辞書
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        distribution = {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "min": float(np.min(gray)),
            "max": float(np.max(gray)),
            "median": float(np.median(gray)),
            "q25": float(np.percentile(gray, 25)),
            "q75": float(np.percentile(gray, 75)),
        }

        return distribution

    def generate_brightness_histogram(self, image: np.ndarray) -> List[int]:
        """
        明度ヒストグラムを生成

        Args:
            image: 入力画像 (H, W, C) numpy配列

        Returns:
            256要素のヒストグラム配列
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # ヒストグラム計算（0-255の256ビン）
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # numpy配列からPythonリストに変換
        histogram_list = [int(count) for count in histogram.flatten()]

        return histogram_list

    def calculate_brightness_percentiles(
        self, image: np.ndarray, percentiles: List[float]
    ) -> Dict[str, float]:
        """
        指定したパーセンタイルの明度値を計算

        Args:
            image: 入力画像 (H, W, C) numpy配列
            percentiles: 計算するパーセンタイルのリスト [0-100]

        Returns:
            パーセンタイル値の辞書
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        result = {}
        for p in percentiles:
            result[f"p{p}"] = float(np.percentile(gray, p))

        return result

    def is_low_brightness(self, image: np.ndarray, threshold: float = 20.0) -> Tuple[bool, float]:
        """
        低明度画像かどうかを判定

        Args:
            image: 入力画像 (H, W, C) numpy配列
            threshold: 明度閾値

        Returns:
            (低明度フラグ, 実際の明度値)
        """
        brightness = self.calculate_brightness(image)
        is_low = brightness <= threshold

        return is_low, brightness

    def analyze_dark_regions(
        self, image: np.ndarray, dark_threshold: float = 30.0
    ) -> Dict[str, float]:
        """
        暗い領域の分析

        Args:
            image: 入力画像 (H, W, C) numpy配列
            dark_threshold: 暗い領域とする明度閾値

        Returns:
            暗い領域の統計情報
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 暗い領域のマスク作成
        dark_mask = gray <= dark_threshold
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_pixels = np.sum(dark_mask)

        dark_region_stats = {
            "dark_pixel_count": int(dark_pixels),
            "total_pixel_count": int(total_pixels),
            "dark_pixel_ratio": float(dark_pixels / total_pixels),
            "dark_region_mean_brightness": float(np.mean(gray[dark_mask]))
            if dark_pixels > 0
            else 0.0,
            "dark_threshold_used": float(dark_threshold),
        }

        return dark_region_stats
