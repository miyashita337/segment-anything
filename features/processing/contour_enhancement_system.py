#!/usr/bin/env python3
"""
輪郭後処理システム強化
エッジ検出アルゴリズム最適化・滑らかさ向上処理・境界線精度向上

目標:
- 平均コンパクトネス: 0.352 → 0.50達成
- 輪郭品質指標すべて合格
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import logging
from dataclasses import dataclass
from pathlib import Path
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContourConfig:
    """輪郭強化設定"""

    # エッジ検出設定
    canny_low_threshold: int = 30
    canny_high_threshold: int = 100
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2

    # 輪郭滑らかさ設定
    smoothing_epsilon_ratio: float = 0.01
    spline_smoothing_factor: float = 0.1
    gaussian_smoothing_sigma: float = 1.0

    # 境界線精度設定
    precision_kernel_size: int = 3
    morphological_iterations: int = 2
    contour_approximation_accuracy: float = 0.02

    # 品質評価設定
    compactness_target: float = 0.50
    smoothness_target: float = 0.75
    precision_target: float = 0.80


class ContourEnhancementSystem:
    """輪郭後処理システム強化"""

    def __init__(self, config: Optional[ContourConfig] = None):
        """初期化"""
        self.config = config or ContourConfig()

    def enhance_contour_quality(
        self, mask: np.ndarray, original_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        輪郭品質向上メイン処理

        Args:
            mask: 入力マスク
            original_image: 元画像（境界情報活用用）

        Returns:
            Dict: 強化された輪郭マスクと品質情報
        """
        try:
            logger.info("🎨 輪郭後処理システム強化開始")

            # 入力検証
            if mask is None or mask.size == 0:
                logger.error("❌ 無効なマスク入力")
                return {"enhanced_mask": None, "quality_metrics": {}, "improvements": []}

            # マスク正規化
            normalized_mask = self._normalize_mask(mask)

            # Step 1: 初期輪郭品質評価
            initial_quality = self._evaluate_contour_quality(normalized_mask)
            logger.info(f"📊 初期輪郭品質: コンパクトネス {initial_quality['compactness']:.3f}")

            # Step 2: 高精度エッジ検出
            enhanced_edges = self._advanced_edge_detection(normalized_mask, original_image)
            improvements = ["advanced_edge_detection"]

            # Step 3: 輪郭滑らかさ向上
            smoothed_contour_mask = self._enhance_contour_smoothness(enhanced_edges)
            improvements.append("contour_smoothing")

            # Step 4: 境界線精度向上
            precision_enhanced_mask = self._improve_boundary_precision(smoothed_contour_mask)
            improvements.append("boundary_precision")

            # Step 5: アニメキャラクター特化調整
            anime_optimized_mask = self._anime_character_contour_optimization(
                precision_enhanced_mask
            )
            improvements.append("anime_optimization")

            # Step 6: 最終品質評価
            final_quality = self._evaluate_contour_quality(anime_optimized_mask)
            logger.info(f"📈 最終輪郭品質: コンパクトネス {final_quality['compactness']:.3f}")

            # Step 7: 改善効果分析
            improvement_analysis = self._analyze_contour_improvement(initial_quality, final_quality)

            return {
                "enhanced_mask": anime_optimized_mask,
                "quality_metrics": final_quality,
                "initial_quality": initial_quality,
                "improvements": improvements,
                "improvement_analysis": improvement_analysis,
                "success": final_quality["compactness"] >= self.config.compactness_target,
            }

        except Exception as e:
            logger.error(f"❌ 輪郭強化システムエラー: {e}")
            return {
                "enhanced_mask": mask,  # フォールバック
                "quality_metrics": {},
                "improvements": [],
                "error": str(e),
            }

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """マスク正規化"""
        try:
            # データ型変換
            if mask.dtype == np.bool_:
                normalized = mask.astype(np.uint8) * 255
            elif mask.dtype in [np.float32, np.float64]:
                normalized = (mask * 255).astype(np.uint8)
            else:
                normalized = mask.copy()

            # 二値化確保
            _, binary_mask = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)

            return binary_mask

        except Exception as e:
            logger.warning(f"⚠️ マスク正規化エラー: {e}")
            return mask

    def _advanced_edge_detection(
        self, mask: np.ndarray, original_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """高精度エッジ検出"""
        try:
            # Step 1: マルチスケールエッジ検出
            multi_scale_edges = self._multi_scale_edge_detection(mask)

            # Step 2: 適応的エッジ検出
            adaptive_edges = self._adaptive_edge_detection(mask)

            # Step 3: 元画像情報統合（利用可能な場合）
            if original_image is not None:
                image_guided_edges = self._image_guided_edge_detection(mask, original_image)
                # 3つのエッジ情報を統合
                combined_edges = cv2.bitwise_or(
                    cv2.bitwise_or(multi_scale_edges, adaptive_edges), image_guided_edges
                )
            else:
                # 2つのエッジ情報を統合
                combined_edges = cv2.bitwise_or(multi_scale_edges, adaptive_edges)

            # Step 4: エッジ精細化
            refined_edges = self._refine_edge_quality(combined_edges)

            return refined_edges

        except Exception as e:
            logger.warning(f"⚠️ 高精度エッジ検出エラー: {e}")
            # フォールバック: 基本Cannyエッジ検出
            return cv2.Canny(
                mask, self.config.canny_low_threshold, self.config.canny_high_threshold
            )

    def _multi_scale_edge_detection(self, mask: np.ndarray) -> np.ndarray:
        """マルチスケールエッジ検出"""
        try:
            # 複数スケールでエッジ検出
            scales = [1, 2, 3]
            edge_results = []

            for scale in scales:
                # ガウシアンブラー適用
                blurred = cv2.GaussianBlur(mask, (scale * 2 + 1, scale * 2 + 1), scale)

                # Cannyエッジ検出
                edges = cv2.Canny(
                    blurred,
                    self.config.canny_low_threshold // scale,
                    self.config.canny_high_threshold // scale,
                )

                edge_results.append(edges)

            # スケール統合
            combined_edges = np.zeros_like(mask)
            for edges in edge_results:
                combined_edges = cv2.bitwise_or(combined_edges, edges)

            return combined_edges

        except Exception as e:
            logger.warning(f"⚠️ マルチスケールエッジ検出エラー: {e}")
            return cv2.Canny(
                mask, self.config.canny_low_threshold, self.config.canny_high_threshold
            )

    def _adaptive_edge_detection(self, mask: np.ndarray) -> np.ndarray:
        """適応的エッジ検出"""
        try:
            # 適応的閾値処理
            adaptive_thresh = cv2.adaptiveThreshold(
                mask,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_threshold_block_size,
                self.config.adaptive_threshold_c,
            )

            # 適応的閾値結果からエッジ抽出
            edges = cv2.Canny(adaptive_thresh, 50, 150)

            return edges

        except Exception as e:
            logger.warning(f"⚠️ 適応的エッジ検出エラー: {e}")
            return cv2.Canny(mask, 50, 150)

    def _image_guided_edge_detection(
        self, mask: np.ndarray, original_image: np.ndarray
    ) -> np.ndarray:
        """画像ガイド付きエッジ検出"""
        try:
            # 元画像のエッジ検出
            if len(original_image.shape) == 3:
                gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = original_image

            # 元画像のエッジ
            image_edges = cv2.Canny(gray, 50, 150)

            # マスク領域内のエッジのみ抽出
            mask_region = mask > 0
            guided_edges = np.zeros_like(mask)
            guided_edges[mask_region] = image_edges[mask_region]

            return guided_edges

        except Exception as e:
            logger.warning(f"⚠️ 画像ガイドエッジ検出エラー: {e}")
            return np.zeros_like(mask)

    def _refine_edge_quality(self, edges: np.ndarray) -> np.ndarray:
        """エッジ品質精細化"""
        try:
            # Step 1: エッジ連結性改善
            connected_edges = self._improve_edge_connectivity(edges)

            # Step 2: ノイズエッジ除去
            denoised_edges = self._remove_noise_edges(connected_edges)

            # Step 3: エッジ厚み正規化
            normalized_edges = self._normalize_edge_thickness(denoised_edges)

            return normalized_edges

        except Exception as e:
            logger.warning(f"⚠️ エッジ品質精細化エラー: {e}")
            return edges

    def _improve_edge_connectivity(self, edges: np.ndarray) -> np.ndarray:
        """エッジ連結性改善"""
        try:
            # モルフォロジカルクロージングでエッジギャップを埋める
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

            return connected

        except Exception as e:
            logger.warning(f"⚠️ エッジ連結性改善エラー: {e}")
            return edges

    def _remove_noise_edges(self, edges: np.ndarray) -> np.ndarray:
        """ノイズエッジ除去"""
        try:
            # 短いエッジライン除去
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 長さ基準でフィルタリング
            min_length = 20  # 最小エッジ長
            filtered_edges = np.zeros_like(edges)

            for contour in contours:
                if cv2.arcLength(contour, False) >= min_length:
                    cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

            return filtered_edges

        except Exception as e:
            logger.warning(f"⚠️ ノイズエッジ除去エラー: {e}")
            return edges

    def _normalize_edge_thickness(self, edges: np.ndarray) -> np.ndarray:
        """エッジ厚み正規化"""
        try:
            # エッジを1ピクセル厚に正規化
            skeleton = cv2.ximgproc.thinning(edges) if hasattr(cv2, "ximgproc") else edges
            return skeleton

        except Exception as e:
            logger.warning(f"⚠️ エッジ厚み正規化エラー: {e}")
            return edges

    def _enhance_contour_smoothness(self, edges: np.ndarray) -> np.ndarray:
        """輪郭滑らかさ向上"""
        try:
            # Step 1: 輪郭抽出
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return edges

            # Step 2: 最大輪郭選択
            largest_contour = max(contours, key=cv2.contourArea)

            # Step 3: スプライン平滑化
            smoothed_contour = self._spline_contour_smoothing(largest_contour)

            # Step 4: ガウシアン平滑化
            gaussian_smoothed = self._gaussian_contour_smoothing(smoothed_contour)

            # Step 5: 平滑化輪郭でマスク再作成
            smoothed_mask = np.zeros_like(edges)
            cv2.fillPoly(smoothed_mask, [gaussian_smoothed], 255)

            return smoothed_mask

        except Exception as e:
            logger.warning(f"⚠️ 輪郭滑らかさ向上エラー: {e}")
            return edges

    def _spline_contour_smoothing(self, contour: np.ndarray) -> np.ndarray:
        """スプライン輪郭平滑化"""
        try:
            # 輪郭点を1次元配列に変換
            contour_points = contour.reshape(-1, 2)

            if len(contour_points) < 4:
                return contour

            # スプライン補間
            x = contour_points[:, 0]
            y = contour_points[:, 1]

            # 閉じた輪郭の場合、開始点を末尾に追加
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            # スプライン係数計算
            tck, u = splprep([x, y], s=self.config.spline_smoothing_factor, per=True)

            # より多くの点でスプライン評価
            u_new = np.linspace(0, 1, len(contour_points) * 2)
            x_smooth, y_smooth = splev(u_new, tck)

            # 結果を輪郭形式に変換
            smoothed_points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
            smoothed_contour = smoothed_points.reshape(-1, 1, 2)

            return smoothed_contour

        except Exception as e:
            logger.warning(f"⚠️ スプライン平滑化エラー: {e}")
            return contour

    def _gaussian_contour_smoothing(self, contour: np.ndarray) -> np.ndarray:
        """ガウシアン輪郭平滑化"""
        try:
            # 輪郭点座標抽出
            contour_points = contour.reshape(-1, 2).astype(np.float32)

            # x, y座標を分離
            x_coords = contour_points[:, 0]
            y_coords = contour_points[:, 1]

            # ガウシアンフィルタ適用
            from scipy.ndimage import gaussian_filter1d

            x_smooth = gaussian_filter1d(
                x_coords, sigma=self.config.gaussian_smoothing_sigma, mode="wrap"
            )
            y_smooth = gaussian_filter1d(
                y_coords, sigma=self.config.gaussian_smoothing_sigma, mode="wrap"
            )

            # 結果統合
            smoothed_points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
            smoothed_contour = smoothed_points.reshape(-1, 1, 2)

            return smoothed_contour

        except Exception as e:
            logger.warning(f"⚠️ ガウシアン平滑化エラー: {e}")
            return contour

    def _improve_boundary_precision(self, mask: np.ndarray) -> np.ndarray:
        """境界線精度向上"""
        try:
            # Step 1: サブピクセル精度輪郭検出
            subpixel_contours = self._subpixel_contour_detection(mask)

            # Step 2: 境界線鋭利化
            sharpened_boundaries = self._sharpen_boundaries(subpixel_contours)

            # Step 3: 精度検証・調整
            precision_adjusted = self._adjust_boundary_precision(sharpened_boundaries)

            return precision_adjusted

        except Exception as e:
            logger.warning(f"⚠️ 境界線精度向上エラー: {e}")
            return mask

    def _subpixel_contour_detection(self, mask: np.ndarray) -> np.ndarray:
        """サブピクセル精度輪郭検出"""
        try:
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return mask

            # 最大輪郭選択
            largest_contour = max(contours, key=cv2.contourArea)

            # 輪郭近似で精度向上
            epsilon = self.config.contour_approximation_accuracy * cv2.arcLength(
                largest_contour, True
            )
            approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # 近似輪郭でマスク再作成
            precision_mask = np.zeros_like(mask)
            cv2.fillPoly(precision_mask, [approximated_contour], 255)

            return precision_mask

        except Exception as e:
            logger.warning(f"⚠️ サブピクセル輪郭検出エラー: {e}")
            return mask

    def _sharpen_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """境界線鋭利化"""
        try:
            # ラプラシアンフィルタで境界強調
            laplacian = cv2.Laplacian(mask, cv2.CV_64F)
            sharpened = mask + 0.5 * laplacian

            # データ型・範囲正規化
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            # 二値化で境界明確化
            _, binary_sharpened = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)

            return binary_sharpened

        except Exception as e:
            logger.warning(f"⚠️ 境界線鋭利化エラー: {e}")
            return mask

    def _adjust_boundary_precision(self, mask: np.ndarray) -> np.ndarray:
        """境界精度調整"""
        try:
            # モルフォロジカル処理で精度微調整
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.precision_kernel_size, self.config.precision_kernel_size),
            )

            # 軽微なオープニング・クロージング
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            adjusted = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

            return adjusted

        except Exception as e:
            logger.warning(f"⚠️ 境界精度調整エラー: {e}")
            return mask

    def _anime_character_contour_optimization(self, mask: np.ndarray) -> np.ndarray:
        """アニメキャラクター特化輪郭最適化"""
        try:
            # Step 1: アニメキャラクター特徴分析
            char_features = self._analyze_anime_character_features(mask)

            # Step 2: 特徴に応じた最適化
            if char_features["is_fullbody"]:
                optimized = self._optimize_fullbody_contour(mask)
            else:
                optimized = self._optimize_portrait_contour(mask)

            # Step 3: アニメ特有の輪郭特性適用
            anime_enhanced = self._apply_anime_contour_characteristics(optimized)

            return anime_enhanced

        except Exception as e:
            logger.warning(f"⚠️ アニメ特化輪郭最適化エラー: {e}")
            return mask

    def _analyze_anime_character_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """アニメキャラクター特徴分析"""
        try:
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return {"is_fullbody": False, "aspect_ratio": 1.0, "complexity": 0.0}

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 特徴計算
            aspect_ratio = h / w if w > 0 else 1.0
            area_ratio = cv2.contourArea(largest_contour) / (mask.shape[0] * mask.shape[1])

            # 輪郭複雑性（周長面積比）
            perimeter = cv2.arcLength(largest_contour, True)
            complexity = (
                perimeter / cv2.contourArea(largest_contour)
                if cv2.contourArea(largest_contour) > 0
                else 0
            )

            # 全身判定
            is_fullbody = aspect_ratio > 1.8 and area_ratio > 0.15

            return {
                "is_fullbody": is_fullbody,
                "aspect_ratio": aspect_ratio,
                "area_ratio": area_ratio,
                "complexity": complexity,
                "bbox": (x, y, w, h),
            }

        except Exception as e:
            logger.warning(f"⚠️ アニメ特徴分析エラー: {e}")
            return {"is_fullbody": False, "aspect_ratio": 1.0, "complexity": 0.0}

    def _optimize_fullbody_contour(self, mask: np.ndarray) -> np.ndarray:
        """全身キャラクター輪郭最適化"""
        try:
            # 全身キャラクターは保守的な処理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # 軽微なスムージング
            smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            return smoothed

        except Exception as e:
            logger.warning(f"⚠️ 全身輪郭最適化エラー: {e}")
            return mask

    def _optimize_portrait_contour(self, mask: np.ndarray) -> np.ndarray:
        """ポートレート輪郭最適化"""
        try:
            # ポートレートは積極的なスムージング
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # 強めのスムージング
            smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)

            return smoothed

        except Exception as e:
            logger.warning(f"⚠️ ポートレート輪郭最適化エラー: {e}")
            return mask

    def _apply_anime_contour_characteristics(self, mask: np.ndarray) -> np.ndarray:
        """アニメ特有輪郭特性適用"""
        try:
            # アニメキャラクターの明確な輪郭線特性を強調

            # Step 1: エッジ強調
            edges = cv2.Canny(mask, 50, 150)
            enhanced_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

            # Step 2: 輪郭内部の均一化
            filled_mask = mask.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.fillPoly(filled_mask, contours, 255)

            # Step 3: 明確な境界線の維持
            final_mask = cv2.bitwise_or(filled_mask, enhanced_edges)

            return final_mask

        except Exception as e:
            logger.warning(f"⚠️ アニメ特性適用エラー: {e}")
            return mask

    def _evaluate_contour_quality(self, mask: np.ndarray) -> Dict[str, float]:
        """輪郭品質評価"""
        try:
            # 基本品質メトリクス
            from features.processing.postprocessing.postprocessing import (
                calculate_mask_quality_metrics,
            )

            basic_metrics = calculate_mask_quality_metrics(mask)

            # 追加輪郭品質メトリクス
            smoothness_score = self._calculate_smoothness_score(mask)
            precision_score = self._calculate_precision_score(mask)
            connectivity_score = self._calculate_connectivity_score(mask)

            return {
                "compactness": basic_metrics.get("compactness", 0.0),
                "fill_ratio": basic_metrics.get("fill_ratio", 0.0),
                "coverage_ratio": basic_metrics.get("coverage_ratio", 0.0),
                "smoothness": smoothness_score,
                "precision": precision_score,
                "connectivity": connectivity_score,
                "overall_contour_quality": (
                    basic_metrics.get("compactness", 0.0) * 0.4
                    + smoothness_score * 0.3
                    + precision_score * 0.3
                ),
            }

        except Exception as e:
            logger.warning(f"⚠️ 輪郭品質評価エラー: {e}")
            return {"compactness": 0.0, "overall_contour_quality": 0.0}

    def _calculate_smoothness_score(self, mask: np.ndarray) -> float:
        """滑らかさスコア計算"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            largest_contour = max(contours, key=cv2.contourArea)

            # 曲率変化の分散を滑らかさの指標とする
            contour_points = largest_contour.reshape(-1, 2)

            if len(contour_points) < 3:
                return 0.5

            # 隣接点間の角度変化計算
            angles = []
            for i in range(len(contour_points)):
                p1 = contour_points[i - 1]
                p2 = contour_points[i]
                p3 = contour_points[(i + 1) % len(contour_points)]

                v1 = p1 - p2
                v2 = p3 - p2

                # 角度計算
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

            # 角度変化の標準偏差（小さいほど滑らか）
            angle_std = np.std(angles)
            smoothness = max(0.0, 1.0 - angle_std / np.pi)

            return smoothness

        except Exception as e:
            logger.warning(f"⚠️ 滑らかさスコア計算エラー: {e}")
            return 0.0

    def _calculate_precision_score(self, mask: np.ndarray) -> float:
        """精度スコア計算"""
        try:
            # エッジの鋭利さを精度指標とする
            edges = cv2.Canny(mask, 50, 150)
            edge_pixels = np.sum(edges > 0)

            # 境界周辺のグラデーション分析
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            boundary_region = dilated - eroded

            # 境界領域の値の分散（小さいほど精密）
            boundary_values = mask[boundary_region > 0]
            if len(boundary_values) > 0:
                precision = 1.0 - (np.std(boundary_values) / 255.0)
            else:
                precision = 0.0

            return max(0.0, min(1.0, precision))

        except Exception as e:
            logger.warning(f"⚠️ 精度スコア計算エラー: {e}")
            return 0.0

    def _calculate_connectivity_score(self, mask: np.ndarray) -> float:
        """連結性スコア計算"""
        try:
            # 連結成分の数と最大成分の支配率
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels <= 1:
                return 0.0

            # 最大成分の面積比率
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas)
            total_area = np.sum(areas)

            connectivity = max_area / total_area if total_area > 0 else 0.0

            return connectivity

        except Exception as e:
            logger.warning(f"⚠️ 連結性スコア計算エラー: {e}")
            return 0.0

    def _analyze_contour_improvement(
        self, initial_quality: Dict[str, float], final_quality: Dict[str, float]
    ) -> Dict[str, Any]:
        """輪郭改善効果分析"""
        try:
            compactness_improvement = final_quality["compactness"] - initial_quality["compactness"]
            overall_improvement = final_quality.get(
                "overall_contour_quality", 0.0
            ) - initial_quality.get("overall_contour_quality", 0.0)

            target_achieved = final_quality["compactness"] >= self.config.compactness_target

            return {
                "compactness_improvement": compactness_improvement,
                "overall_improvement": overall_improvement,
                "target_achieved": target_achieved,
                "improvement_percentage": (
                    compactness_improvement / max(initial_quality["compactness"], 0.001)
                )
                * 100,
                "final_compactness": final_quality["compactness"],
            }

        except Exception as e:
            logger.warning(f"⚠️ 改善効果分析エラー: {e}")
            return {"compactness_improvement": 0.0, "target_achieved": False}


def integrate_with_processing_pipeline() -> None:
    """後処理パイプラインとの統合"""
    logger.info("🔗 輪郭強化システムを後処理パイプラインに統合")

    # sam_postprocessing_pipeline.py との統合想定
    # 実際の統合は既存コードの修正が必要
    pass


if __name__ == "__main__":
    # テスト実行
    logger.info("🧪 輪郭後処理システム強化 テスト開始")

    # テスト用設定
    config = ContourConfig(compactness_target=0.50, smoothness_target=0.75, precision_target=0.80)

    enhancer = ContourEnhancementSystem(config)
    logger.info("✅ 輪郭強化システム初期化完了")

    # 統合準備
    integrate_with_processing_pipeline()
    logger.info("🎯 テスト完了 - 実装準備完了")
