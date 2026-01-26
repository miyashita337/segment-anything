"""
QI-002: 部分抽出検出器 (PartialExtractionDetector)

キャラクターの一部のみが抽出される問題を検出し、
抽出品質の劣化を特定する機能を提供します。
"""

import numpy as np
import cv2

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional


@dataclass
class PartialExtractionResult:
    """部分抽出検出結果"""

    is_partial_extraction: bool
    extraction_type: str  # 'complete', 'upper_body_only', 'lower_body_only', 'headless', 'torso_only', 'fragmented'
    completeness_score: float  # 0.0-1.0
    confidence: float
    extraction_quality: str  # 'complete', 'partial', 'poor', 'fragmented'
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    missing_parts: List[str]
    detected_parts: List[str]
    fragmentation_level: float
    aspect_ratio_warning: bool
    aspect_ratio_analysis: Dict[str, float]
    recommendations: List[str]
    additional_info: Optional[Dict] = None


class PartialExtractionDetector:
    """部分抽出検出を行うクラス"""

    def __init__(
        self,
        completeness_threshold: float = 0.7,
        fragmentation_threshold: float = 0.3,
        min_part_size: int = 500,
        ideal_aspect_ratio: float = 2.0,
    ):
        """
        PartialExtractionDetector の初期化

        Args:
            completeness_threshold: 完全性判定閾値
            fragmentation_threshold: 断片化判定閾値
            min_part_size: 最小部位サイズ（ピクセル）
            ideal_aspect_ratio: 理想的なアスペクト比（縦/横）
        """
        self.completeness_threshold = completeness_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.min_part_size = min_part_size
        self.ideal_aspect_ratio = ideal_aspect_ratio

    def detect_partial_extraction(self, image: np.ndarray) -> PartialExtractionResult:
        """
        画像から部分抽出を検出

        Args:
            image: 入力画像 (H, W, C) numpy配列

        Returns:
            PartialExtractionResult: 検出結果
        """
        try:
            # 基本的な画像解析
            basic_analysis = self._analyze_basic_properties(image)

            # 身体部位検出
            body_parts = self._detect_body_parts(image)

            # 完全性評価
            completeness_analysis = self._analyze_completeness(image, body_parts)

            # 断片化解析
            fragmentation_analysis = self._analyze_fragmentation(image)

            # アスペクト比解析
            aspect_ratio_analysis = self._analyze_aspect_ratio(image)

            # 統合判定
            final_result = self._integrate_analysis_results(
                basic_analysis,
                body_parts,
                completeness_analysis,
                fragmentation_analysis,
                aspect_ratio_analysis,
            )

            return final_result

        except Exception as e:
            return PartialExtractionResult(
                is_partial_extraction=False,
                extraction_type="error",
                completeness_score=0.0,
                confidence=0.0,
                extraction_quality="unknown",
                severity="none",
                missing_parts=[],
                detected_parts=[],
                fragmentation_level=0.0,
                aspect_ratio_warning=False,
                aspect_ratio_analysis={},
                recommendations=[],
                additional_info={"error": str(e)},
            )

    def _analyze_basic_properties(self, image: np.ndarray) -> Dict:
        """基本的な画像特性の解析"""
        h, w = image.shape[:2]

        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 非ゼロピクセルの解析
        non_zero_mask = gray > 10
        non_zero_pixels = np.sum(non_zero_mask)
        total_pixels = h * w

        # バウンディングボックスの取得
        y_coords, x_coords = np.where(non_zero_mask)
        if len(y_coords) > 0:
            bbox = {
                "min_y": int(np.min(y_coords)),
                "max_y": int(np.max(y_coords)),
                "min_x": int(np.min(x_coords)),
                "max_x": int(np.max(x_coords)),
            }
            bbox_width = bbox["max_x"] - bbox["min_x"] + 1
            bbox_height = bbox["max_y"] - bbox["min_y"] + 1
        else:
            bbox = {"min_y": 0, "max_y": h, "min_x": 0, "max_x": w}
            bbox_width = w
            bbox_height = h

        return {
            "image_size": (h, w),
            "non_zero_pixels": non_zero_pixels,
            "coverage_ratio": non_zero_pixels / total_pixels,
            "bbox": bbox,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "aspect_ratio": bbox_height / bbox_width if bbox_width > 0 else 1.0,
        }

    def _detect_body_parts(self, image: np.ndarray) -> Dict:
        """身体部位の検出"""
        h, w = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        non_zero_mask = gray > 10
        y_coords, x_coords = np.where(non_zero_mask)

        if len(y_coords) == 0:
            return {"head": False, "torso": False, "arms": False, "legs": False, "regions": []}

        # 垂直方向の分割による部位推定
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        height = max_y - min_y + 1

        # 画像の実際の高さに基づく部位判定
        # 理想的な全身キャラクターの高さを基準とした相対判定
        image_height = gray.shape[0]
        char_height_ratio = height / image_height

        # 高さが画像の半分以下なら上半身のみと推定
        if char_height_ratio < 0.5:  # 画像の50%未満なら部分抽出
            head_region_end = min_y + height // 2  # 上半分が頭部
            torso_region_end = max_y  # 残り全体が胴体、脚部なし
            legs_likely = False
        else:
            head_region_end = min_y + height // 4  # 上部1/4が頭部
            torso_region_end = min_y + 2 * height // 3  # 中部が胴体
            legs_likely = True

        # 各領域での存在確認
        head_pixels = np.sum((y_coords >= min_y) & (y_coords <= head_region_end))
        torso_pixels = np.sum((y_coords > head_region_end) & (y_coords <= torso_region_end))
        legs_pixels = np.sum(y_coords > torso_region_end)

        # 横方向の拡がりによる腕部の推定
        center_x = (np.min(x_coords) + np.max(x_coords)) // 2
        torso_region_mask = (y_coords > head_region_end) & (y_coords <= torso_region_end)
        if np.sum(torso_region_mask) > 0:
            torso_x_coords = x_coords[torso_region_mask]
            torso_width = np.max(torso_x_coords) - np.min(torso_x_coords) + 1
            has_arms = torso_width > height // 6  # より緩い腕部判定
        else:
            has_arms = False

        detected_parts = {
            "head": head_pixels > 100,  # 固定閾値で判定
            "torso": torso_pixels > 100,  # 固定閾値で判定
            "arms": has_arms,
            "legs": legs_likely and legs_pixels > 100,  # 脚部の可能性と閾値の両方をチェック
            "regions": {
                "head_pixels": head_pixels,
                "torso_pixels": torso_pixels,
                "legs_pixels": legs_pixels,
                "total_height": height,
            },
        }

        return detected_parts

    def _analyze_completeness(self, image: np.ndarray, body_parts: Dict) -> Dict:
        """完全性の解析"""
        # 検出された部位の数
        detected_parts_count = sum(
            [body_parts["head"], body_parts["torso"], body_parts["arms"], body_parts["legs"]]
        )

        # 完全性スコアの計算（重み付け）
        part_weights = {"head": 0.3, "torso": 0.3, "arms": 0.2, "legs": 0.2}
        completeness_score = sum(
            part_weights[part] for part in ["head", "torso", "arms", "legs"] if body_parts[part]
        )

        # 欠落部位の特定
        missing_parts = []
        detected_parts = []

        if not body_parts["head"]:
            missing_parts.append("head")
        else:
            detected_parts.append("head")

        if not body_parts["torso"]:
            missing_parts.append("torso")
        else:
            detected_parts.append("torso")

        if not body_parts["arms"]:
            missing_parts.append("arms")
        else:
            detected_parts.append("arms")

        if not body_parts["legs"]:
            missing_parts.append("legs")
        else:
            detected_parts.append("legs")

        # 抽出タイプの判定（より厳密に）
        if detected_parts_count == 4:
            extraction_type = "complete"
        elif (
            body_parts["head"]
            and body_parts["torso"]
            and body_parts["arms"]
            and not body_parts["legs"]
        ):
            extraction_type = "upper_body_only"
        elif body_parts["legs"] and body_parts["torso"] and not body_parts["head"]:
            extraction_type = "headless"
        elif body_parts["legs"] and not body_parts["head"] and not body_parts["torso"]:
            extraction_type = "lower_body_only"
        elif body_parts["torso"] and not body_parts["head"] and not body_parts["legs"]:
            extraction_type = "torso_only"
        else:
            extraction_type = "incomplete"

        return {
            "completeness_score": completeness_score,
            "extraction_type": extraction_type,
            "missing_parts": missing_parts,
            "detected_parts": detected_parts,
            "detected_parts_count": detected_parts_count,
        }

    def _analyze_fragmentation(self, image: np.ndarray) -> Dict:
        """断片化の解析"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 連結成分の解析
        binary_mask = (gray > 10).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

        # 背景ラベル（0）を除外
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []

        if len(component_areas) == 0:
            return {
                "is_fragmented": False,
                "fragment_count": 0,
                "fragmentation_level": 0.0,
                "largest_fragment_ratio": 1.0,
            }

        # 断片化レベルの計算
        total_area = np.sum(component_areas)
        largest_area = np.max(component_areas)
        largest_fragment_ratio = largest_area / total_area if total_area > 0 else 1.0

        # 有意なサイズの断片のみカウント
        significant_fragments = np.sum(component_areas > self.min_part_size // 4)

        fragmentation_level = 1.0 - largest_fragment_ratio
        is_fragmented = (significant_fragments > 3) or (
            fragmentation_level > self.fragmentation_threshold
        )

        return {
            "is_fragmented": is_fragmented,
            "fragment_count": significant_fragments,
            "fragmentation_level": fragmentation_level,
            "largest_fragment_ratio": largest_fragment_ratio,
            "component_areas": component_areas.tolist(),
        }

    def _analyze_aspect_ratio(self, image: np.ndarray) -> Dict:
        """アスペクト比の解析"""
        basic_props = self._analyze_basic_properties(image)
        aspect_ratio = basic_props["aspect_ratio"]

        # 理想的なアスペクト比からの乖離
        ratio_deviation = abs(aspect_ratio - self.ideal_aspect_ratio)
        ratio_score = max(0.0, 1.0 - ratio_deviation / self.ideal_aspect_ratio)

        # 警告条件
        aspect_ratio_warning = (
            aspect_ratio < 1.0
            or aspect_ratio > 4.0  # 横長すぎ
            or ratio_deviation > self.ideal_aspect_ratio * 0.5  # 縦長すぎ
        )

        return {
            "aspect_ratio": aspect_ratio,
            "ideal_ratio": self.ideal_aspect_ratio,
            "ratio_deviation": ratio_deviation,
            "ratio_score": ratio_score,
            "aspect_ratio_warning": aspect_ratio_warning,
            "bbox_width": basic_props["bbox_width"],
            "bbox_height": basic_props["bbox_height"],
        }

    def _integrate_analysis_results(
        self,
        basic_analysis,
        body_parts,
        completeness_analysis,
        fragmentation_analysis,
        aspect_ratio_analysis,
    ) -> PartialExtractionResult:
        """解析結果の統合"""
        # 部分抽出の判定
        is_partial = (
            completeness_analysis["completeness_score"] < self.completeness_threshold
            or fragmentation_analysis["is_fragmented"]
            or aspect_ratio_analysis["aspect_ratio_warning"]
        )

        # 深刻度の判定
        if fragmentation_analysis["is_fragmented"] and fragmentation_analysis["fragment_count"] > 5:
            severity = "critical"
        elif "head" in completeness_analysis["missing_parts"]:
            severity = "high"
        elif completeness_analysis["completeness_score"] < 0.5:
            severity = "high"
        elif completeness_analysis["completeness_score"] < 0.7:
            severity = "medium"
        elif aspect_ratio_analysis["aspect_ratio_warning"]:
            severity = "low"
        else:
            severity = "none"

        # 品質レベルの判定
        if not is_partial:
            quality = "complete"
        elif fragmentation_analysis["is_fragmented"]:
            quality = "fragmented"
        elif completeness_analysis["completeness_score"] < 0.4:
            quality = "poor"
        else:
            quality = "partial"

        # 抽出タイプの最終決定
        if fragmentation_analysis["is_fragmented"]:
            extraction_type = "fragmented"
        else:
            extraction_type = completeness_analysis["extraction_type"]

        # 信頼度の計算
        confidence = min(
            1.0,
            (
                completeness_analysis["completeness_score"] * 0.4
                + aspect_ratio_analysis["ratio_score"] * 0.3
                + (1.0 - fragmentation_analysis["fragmentation_level"]) * 0.3
            ),
        )

        # 推奨事項の生成
        recommendations = self._generate_recommendations(
            completeness_analysis, fragmentation_analysis, aspect_ratio_analysis
        )

        return PartialExtractionResult(
            is_partial_extraction=is_partial,
            extraction_type=extraction_type,
            completeness_score=completeness_analysis["completeness_score"],
            confidence=confidence,
            extraction_quality=quality,
            severity=severity,
            missing_parts=completeness_analysis["missing_parts"],
            detected_parts=completeness_analysis["detected_parts"],
            fragmentation_level=fragmentation_analysis["fragmentation_level"],
            aspect_ratio_warning=aspect_ratio_analysis["aspect_ratio_warning"],
            aspect_ratio_analysis=aspect_ratio_analysis,
            recommendations=recommendations,
            additional_info={
                "basic_analysis": basic_analysis,
                "body_parts_analysis": body_parts,
                "fragmentation_analysis": fragmentation_analysis,
            },
        )

    def _generate_recommendations(
        self, completeness_analysis, fragmentation_analysis, aspect_ratio_analysis
    ) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []

        if completeness_analysis["completeness_score"] < 0.7:
            if "head" in completeness_analysis["missing_parts"]:
                recommendations.append("Adjust detection parameters to include head region")
            if "legs" in completeness_analysis["missing_parts"]:
                recommendations.append(
                    "Increase detection area to capture full body including legs"
                )
            if "arms" in completeness_analysis["missing_parts"]:
                recommendations.append("Consider wider detection area for arm regions")

        if fragmentation_analysis["is_fragmented"]:
            recommendations.append("Apply morphological operations to connect fragmented regions")
            recommendations.append("Review segmentation threshold parameters")

        if aspect_ratio_analysis["aspect_ratio_warning"]:
            if aspect_ratio_analysis["aspect_ratio"] < 1.0:
                recommendations.append("Check for limb truncation in width direction")
            elif aspect_ratio_analysis["aspect_ratio"] > 4.0:
                recommendations.append("Verify character is not cropped vertically")

        if not recommendations:
            recommendations.append("Extraction appears to be of good quality")

        return recommendations
