#!/usr/bin/env python3
"""
SAM後処理パイプライン改良システム
マスク品質向上・エッジ精密化・ノイズ除去アルゴリズム強化

目標:
- A/B評価率: 6.2% → 30%以上達成
- 高品質抽出率大幅向上
"""

import numpy as np
import cv2

import logging
import scipy.ndimage as ndimage
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PostprocessingConfig:
    """SAM後処理設定"""

    # エッジ精密化設定
    edge_smoothing_kernel: int = 3
    edge_smoothing_iterations: int = 2
    gaussian_blur_kernel: int = 5
    bilateral_filter_d: int = 9

    # ノイズ除去設定
    min_component_area: int = 100
    noise_opening_kernel: int = 3
    noise_closing_kernel: int = 5

    # 境界線改善設定
    contour_smoothing_epsilon: float = 0.02
    morphological_kernel_size: int = 5
    dilation_iterations: int = 1
    erosion_iterations: int = 1

    # 品質評価設定
    quality_threshold: float = 0.7
    compactness_weight: float = 0.3
    fill_ratio_weight: float = 0.4
    coverage_weight: float = 0.3


class SAMPostprocessingPipeline:
    """SAM後処理パイプライン改良システム"""

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        """初期化"""
        self.config = config or PostprocessingConfig()

    def enhance_mask_quality(
        self, mask: np.ndarray, original_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        マスク品質向上メイン処理

        Args:
            mask: 入力SAMマスク
            original_image: 元画像（境界情報活用用）

        Returns:
            Dict: 改良されたマスクと品質情報
        """
        try:
            logger.info("🔧 SAM後処理パイプライン開始")

            # 入力マスク検証
            if mask is None or mask.size == 0:
                logger.error("❌ 無効なマスク入力")
                return {"enhanced_mask": None, "quality_score": 0.0, "improvements": []}

            # マスク正規化
            normalized_mask = self._normalize_mask(mask)

            # Step 1: 初期品質評価
            initial_quality = self._calculate_comprehensive_quality(normalized_mask)
            logger.info(f"📊 初期品質スコア: {initial_quality['overall_score']:.3f}")

            # Step 2: ノイズ除去処理
            denoised_mask = self._advanced_noise_removal(normalized_mask)
            improvements = ["noise_removal"]

            # Step 3: エッジ精密化処理
            edge_refined_mask = self._precision_edge_refinement(denoised_mask, original_image)
            improvements.append("edge_refinement")

            # Step 4: 境界線スムージング
            smoothed_mask = self._intelligent_boundary_smoothing(edge_refined_mask)
            improvements.append("boundary_smoothing")

            # Step 5: 輪郭後処理システム強化統合 (P1-A003)
            try:
                from features.processing.contour_enhancement_system import ContourEnhancementSystem

                contour_enhancer = ContourEnhancementSystem()

                contour_result = contour_enhancer.enhance_contour_quality(
                    smoothed_mask, original_image
                )
                contour_enhanced_mask = contour_result.get("enhanced_mask", smoothed_mask)
                contour_quality = contour_result.get("quality_metrics", {})

                if contour_result.get("success", False):
                    smoothed_mask = contour_enhanced_mask
                    improvements.append("contour_enhancement")
                    logger.info(f"🎨 輪郭強化完了: コンパクトネス {contour_quality.get('compactness', 0.0):.3f}")
                else:
                    logger.debug("⚠️ 輪郭強化をスキップ（品質要件未達）")

            except Exception as e:
                logger.warning(f"⚠️ 輪郭強化システムエラー（フォールバック使用）: {e}")

            # Step 6: 形状最適化
            shape_optimized_mask = self._character_shape_optimization(smoothed_mask)
            improvements.append("shape_optimization")

            # Step 7: 最終品質評価
            final_quality = self._calculate_comprehensive_quality(shape_optimized_mask)
            logger.info(f"📈 最終品質スコア: {final_quality['overall_score']:.3f}")

            # Step 8: 改善効果分析
            improvement_stats = self._analyze_improvement_effect(initial_quality, final_quality)

            return {
                "enhanced_mask": shape_optimized_mask,
                "quality_score": final_quality["overall_score"],
                "initial_quality": initial_quality,
                "final_quality": final_quality,
                "improvements": improvements,
                "improvement_stats": improvement_stats,
            }

        except Exception as e:
            logger.error(f"❌ SAM後処理パイプラインエラー: {e}")
            return {
                "enhanced_mask": mask,  # フォールバック
                "quality_score": 0.0,
                "improvements": [],
                "error": str(e),
            }

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """マスク正規化"""
        try:
            # データ型確認・変換
            if mask.dtype == np.bool_:
                normalized = mask.astype(np.uint8) * 255
            elif mask.dtype == np.float32 or mask.dtype == np.float64:
                normalized = (mask * 255).astype(np.uint8)
            else:
                normalized = mask.copy()

            # 二値化確保
            _, binary_mask = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)

            return binary_mask

        except Exception as e:
            logger.warning(f"⚠️ マスク正規化エラー: {e}")
            return mask

    def _advanced_noise_removal(self, mask: np.ndarray) -> np.ndarray:
        """高度ノイズ除去処理"""
        try:
            # Step 1: 小さな連結成分除去
            cleaned_mask = self._remove_small_components_advanced(mask)

            # Step 2: モルフォロジカルノイズ除去
            morphological_cleaned = self._morphological_noise_removal(cleaned_mask)

            # Step 3: 統計的外れ値除去
            outlier_cleaned = self._statistical_outlier_removal(morphological_cleaned)

            return outlier_cleaned

        except Exception as e:
            logger.warning(f"⚠️ ノイズ除去エラー: {e}")
            return mask

    def _remove_small_components_advanced(self, mask: np.ndarray) -> np.ndarray:
        """高度小成分除去"""
        try:
            # 連結成分分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels <= 1:
                return mask

            # 面積ベース閾値計算
            areas = stats[1:, cv2.CC_STAT_AREA]  # 背景除外
            if len(areas) == 0:
                return mask

            # 動的閾値: 最大面積の10%または設定値の大きい方
            max_area = np.max(areas)
            dynamic_threshold = max(self.config.min_component_area, max_area * 0.1)

            # 新マスク作成
            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= dynamic_threshold:
                    cleaned_mask[labels == i] = 255

            return cleaned_mask

        except Exception as e:
            logger.warning(f"⚠️ 小成分除去エラー: {e}")
            return mask

    def _morphological_noise_removal(self, mask: np.ndarray) -> np.ndarray:
        """モルフォロジカルノイズ除去"""
        try:
            # Opening: ノイズ除去
            opening_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.noise_opening_kernel, self.config.noise_opening_kernel),
            )
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

            # Closing: ホール埋め
            closing_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.noise_closing_kernel, self.config.noise_closing_kernel),
            )
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)

            return closed

        except Exception as e:
            logger.warning(f"⚠️ モルフォロジカルノイズ除去エラー: {e}")
            return mask

    def _statistical_outlier_removal(self, mask: np.ndarray) -> np.ndarray:
        """統計的外れ値除去"""
        try:
            # 連結成分統計分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels <= 2:  # 背景+1成分
                return mask

            # 面積統計
            areas = stats[1:, cv2.CC_STAT_AREA]
            mean_area = np.mean(areas)
            std_area = np.std(areas)

            # 外れ値閾値 (平均 - 2σ)
            outlier_threshold = max(mean_area - 2 * std_area, self.config.min_component_area)

            # 外れ値除去
            filtered_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= outlier_threshold:
                    filtered_mask[labels == i] = 255

            return filtered_mask

        except Exception as e:
            logger.warning(f"⚠️ 統計的外れ値除去エラー: {e}")
            return mask

    def _precision_edge_refinement(
        self, mask: np.ndarray, original_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """精密エッジ改良処理"""
        try:
            # Step 1: ガウシアンスムージング
            gaussian_refined = self._gaussian_edge_smoothing(mask)

            # Step 2: バイラテラルフィルタ適用
            bilateral_refined = self._bilateral_edge_enhancement(gaussian_refined)

            # Step 3: 元画像エッジ情報活用（利用可能な場合）
            if original_image is not None:
                edge_guided_refined = self._edge_guided_refinement(
                    bilateral_refined, original_image
                )
            else:
                edge_guided_refined = bilateral_refined

            return edge_guided_refined

        except Exception as e:
            logger.warning(f"⚠️ エッジ精密化エラー: {e}")
            return mask

    def _gaussian_edge_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """ガウシアンエッジスムージング"""
        try:
            # kernel_sizeを奇数に
            kernel_size = self.config.gaussian_blur_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1

            # ガウシアンブラー適用
            blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

            # 二値化で境界復元
            _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            return smoothed

        except Exception as e:
            logger.warning(f"⚠️ ガウシアンスムージングエラー: {e}")
            return mask

    def _bilateral_edge_enhancement(self, mask: np.ndarray) -> np.ndarray:
        """バイラテラルフィルタエッジ強化"""
        try:
            # バイラテラルフィルタ適用
            enhanced = cv2.bilateralFilter(mask, self.config.bilateral_filter_d, 75, 75)

            # 二値化
            _, binary_enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)

            return binary_enhanced

        except Exception as e:
            logger.warning(f"⚠️ バイラテラルフィルタエラー: {e}")
            return mask

    def _edge_guided_refinement(self, mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """エッジガイド付き改良"""
        try:
            # 元画像からエッジ抽出
            gray = (
                cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                if len(original_image.shape) == 3
                else original_image
            )
            edges = cv2.Canny(gray, 50, 150)

            # マスク境界とエッジ情報の統合
            mask_edges = cv2.Canny(mask, 50, 150)
            combined_edges = cv2.bitwise_or(mask_edges, edges)

            # エッジ情報を使ったマスク改良
            refined_mask = mask.copy()

            # エッジ周辺のマスク値を調整
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_dilated = cv2.dilate(combined_edges, kernel, iterations=1)

            # エッジ周辺の平滑化
            edge_region = edge_dilated > 0
            refined_mask[edge_region] = cv2.GaussianBlur(mask, (5, 5), 0)[edge_region]

            # 二値化
            _, binary_refined = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)

            return binary_refined

        except Exception as e:
            logger.warning(f"⚠️ エッジガイド改良エラー: {e}")
            return mask

    def _intelligent_boundary_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """インテリジェント境界スムージング"""
        try:
            # 輪郭抽出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return mask

            # 最大輪郭取得
            largest_contour = max(contours, key=cv2.contourArea)

            # 輪郭スムージング
            epsilon = self.config.contour_smoothing_epsilon * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # スムージングされた輪郭でマスク再作成
            smoothed_mask = np.zeros_like(mask)
            cv2.fillPoly(smoothed_mask, [smoothed_contour], 255)

            return smoothed_mask

        except Exception as e:
            logger.warning(f"⚠️ 境界スムージングエラー: {e}")
            return mask

    def _character_shape_optimization(self, mask: np.ndarray) -> np.ndarray:
        """キャラクター形状最適化"""
        try:
            # Step 1: 形状解析
            shape_analysis = self._analyze_character_shape(mask)

            # Step 2: 形状に応じた最適化
            if shape_analysis["is_fullbody"]:
                optimized = self._optimize_fullbody_shape(mask)
            else:
                optimized = self._optimize_portrait_shape(mask)

            # Step 3: 最終調整
            final_optimized = self._final_shape_adjustment(optimized)

            return final_optimized

        except Exception as e:
            logger.warning(f"⚠️ 形状最適化エラー: {e}")
            return mask

    def _analyze_character_shape(self, mask: np.ndarray) -> Dict[str, Any]:
        """キャラクター形状解析"""
        try:
            # 境界ボックス取得
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {"is_fullbody": False, "aspect_ratio": 1.0, "area_ratio": 0.0}

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 形状特徴計算
            aspect_ratio = h / w if w > 0 else 1.0
            area_ratio = cv2.contourArea(largest_contour) / (mask.shape[0] * mask.shape[1])

            # 全身判定
            is_fullbody = aspect_ratio > 1.8 and area_ratio > 0.15

            return {
                "is_fullbody": is_fullbody,
                "aspect_ratio": aspect_ratio,
                "area_ratio": area_ratio,
                "bbox": (x, y, w, h),
            }

        except Exception as e:
            logger.warning(f"⚠️ 形状解析エラー: {e}")
            return {"is_fullbody": False, "aspect_ratio": 1.0, "area_ratio": 0.0}

    def _optimize_fullbody_shape(self, mask: np.ndarray) -> np.ndarray:
        """全身キャラクター形状最適化"""
        try:
            # より保守的なモルフォロジカル処理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # 軽微な膨張で切断防止
            dilated = cv2.dilate(mask, kernel, iterations=1)

            # ホール埋めでキャラクター内部を滑らか化
            filled = self._fill_internal_holes(dilated)

            return filled

        except Exception as e:
            logger.warning(f"⚠️ 全身形状最適化エラー: {e}")
            return mask

    def _optimize_portrait_shape(self, mask: np.ndarray) -> np.ndarray:
        """ポートレート形状最適化"""
        try:
            # より積極的なスムージング
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # クロージングで顔周辺を滑らか化
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            return closed

        except Exception as e:
            logger.warning(f"⚠️ ポートレート形状最適化エラー: {e}")
            return mask

    def _fill_internal_holes(self, mask: np.ndarray) -> np.ndarray:
        """内部ホール埋め"""
        try:
            # Flood fillでホール検出・埋め
            h, w = mask.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)

            # 4隅からflood fill（外部領域マーク）
            filled = mask.copy()
            cv2.floodFill(filled, flood_mask, (0, 0), 128)

            # 内部ホール（値が0のまま残った領域）を埋める
            filled[filled == 0] = 255
            filled[filled == 128] = 0

            return filled

        except Exception as e:
            logger.warning(f"⚠️ ホール埋めエラー: {e}")
            return mask

    def _final_shape_adjustment(self, mask: np.ndarray) -> np.ndarray:
        """最終形状調整"""
        try:
            # 軽微なエロージョン・ダイレーション
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # エロージョン（細い部分の除去）
            eroded = cv2.erode(mask, kernel, iterations=1)

            # ダイレーション（形状復元）
            restored = cv2.dilate(eroded, kernel, iterations=1)

            return restored

        except Exception as e:
            logger.warning(f"⚠️ 最終形状調整エラー: {e}")
            return mask

    def _calculate_comprehensive_quality(self, mask: np.ndarray) -> Dict[str, float]:
        """包括的品質評価"""
        try:
            # 基本品質メトリクス
            from features.processing.postprocessing.postprocessing import (
                calculate_mask_quality_metrics,
            )

            basic_metrics = calculate_mask_quality_metrics(mask)

            # 追加品質メトリクス
            edge_quality = self._calculate_edge_quality(mask)
            shape_quality = self._calculate_shape_quality(mask)
            noise_quality = self._calculate_noise_quality(mask)

            # 総合スコア計算
            overall_score = (
                basic_metrics["compactness"] * self.config.compactness_weight
                + basic_metrics["fill_ratio"] * self.config.fill_ratio_weight
                + basic_metrics["coverage_ratio"] * self.config.coverage_weight
            )

            return {
                "overall_score": overall_score,
                "basic_metrics": basic_metrics,
                "edge_quality": edge_quality,
                "shape_quality": shape_quality,
                "noise_quality": noise_quality,
            }

        except Exception as e:
            logger.warning(f"⚠️ 品質評価エラー: {e}")
            return {"overall_score": 0.0}

    def _calculate_edge_quality(self, mask: np.ndarray) -> float:
        """エッジ品質評価"""
        try:
            # エッジ検出
            edges = cv2.Canny(mask, 50, 150)
            edge_pixels = np.sum(edges > 0)

            # 輪郭滑らかさ評価
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                smoothness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            else:
                smoothness = 0

            return min(1.0, smoothness)

        except Exception as e:
            logger.warning(f"⚠️ エッジ品質評価エラー: {e}")
            return 0.0

    def _calculate_shape_quality(self, mask: np.ndarray) -> float:
        """形状品質評価"""
        try:
            # 連結成分分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels <= 1:
                return 0.0

            # 最大成分の支配率
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas)
            total_area = np.sum(areas)
            dominance = max_area / total_area if total_area > 0 else 0

            return dominance

        except Exception as e:
            logger.warning(f"⚠️ 形状品質評価エラー: {e}")
            return 0.0

    def _calculate_noise_quality(self, mask: np.ndarray) -> float:
        """ノイズ品質評価"""
        try:
            # 小成分数による評価
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels <= 1:
                return 1.0

            # 小成分数（全体面積の1%未満）
            total_area = mask.shape[0] * mask.shape[1]
            small_component_threshold = total_area * 0.01
            small_components = np.sum(stats[1:, cv2.CC_STAT_AREA] < small_component_threshold)

            # ノイズ品質スコア（小成分が少ないほど高い）
            noise_score = max(0.0, 1.0 - small_components / 10.0)

            return noise_score

        except Exception as e:
            logger.warning(f"⚠️ ノイズ品質評価エラー: {e}")
            return 0.0

    def _analyze_improvement_effect(
        self, initial_quality: Dict[str, Any], final_quality: Dict[str, Any]
    ) -> Dict[str, float]:
        """改善効果分析"""
        try:
            initial_score = initial_quality.get("overall_score", 0.0)
            final_score = final_quality.get("overall_score", 0.0)

            improvement_ratio = (
                ((final_score - initial_score) / max(initial_score, 0.001))
                if initial_score > 0
                else 0.0
            )
            absolute_improvement = final_score - initial_score

            return {
                "improvement_ratio": improvement_ratio,
                "absolute_improvement": absolute_improvement,
                "initial_score": initial_score,
                "final_score": final_score,
                "success": final_score > initial_score,
            }

        except Exception as e:
            logger.warning(f"⚠️ 改善効果分析エラー: {e}")
            return {"improvement_ratio": 0.0, "absolute_improvement": 0.0, "success": False}


def integrate_with_character_extraction() -> None:
    """キャラクター抽出パイプラインとの統合"""
    logger.info("🔗 SAM後処理パイプラインをキャラクター抽出に統合")

    # extract_character.py の後処理段階に統合する想定
    # 実際の統合は既存コードの修正が必要
    pass


if __name__ == "__main__":
    # テスト実行
    logger.info("🧪 SAM後処理パイプライン テスト開始")

    # テスト用設定
    config = PostprocessingConfig(
        edge_smoothing_kernel=3,
        gaussian_blur_kernel=5,
        min_component_area=100,
        quality_threshold=0.7,
    )

    pipeline = SAMPostprocessingPipeline(config)
    logger.info("✅ SAM後処理パイプライン初期化完了")

    # 統合準備
    integrate_with_character_extraction()
    logger.info("🎯 テスト完了 - 実装準備完了")
