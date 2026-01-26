"""
QI-002: 完全性検証器 (CompletenessValidator)

抽出されたキャラクター画像の完全性を検証し、欠損部位の特定や
断片化の検出、アスペクト比の妥当性を総合的に評価します。
"""

import numpy as np
import cv2

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional


@dataclass
class CompletenessValidationResult:
    """完全性検証結果"""

    is_complete: bool
    completeness_percentage: float
    validation_confidence: float
    missing_parts: List[str]
    detected_parts: List[str]
    fragmentation_detected: bool
    fragment_count: int
    aspect_ratio_warning: bool
    aspect_ratio_score: float
    body_parts_analysis: Dict[str, float]
    quality_indicators: Dict[str, float]
    validation_issues: List[str]
    improvement_suggestions: List[str]


class CompletenessValidator:
    """完全性検証を行うクラス"""

    def __init__(
        self,
        completeness_threshold: float = 0.8,
        fragmentation_threshold: int = 3,
        ideal_aspect_ratio: float = 2.5,
        aspect_ratio_tolerance: float = 1.0,
        min_part_size: int = 200,
    ):
        """
        CompletenessValidator の初期化

        Args:
            completeness_threshold: 完全性判定閾値（0-1）
            fragmentation_threshold: 断片化判定閾値（連結成分数）
            ideal_aspect_ratio: 理想的なアスペクト比（縦/横）
            aspect_ratio_tolerance: アスペクト比許容範囲
            min_part_size: 最小部位サイズ（ピクセル数）
        """
        self.completeness_threshold = completeness_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.ideal_aspect_ratio = ideal_aspect_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.min_part_size = min_part_size

    def validate_completeness(self, image: np.ndarray) -> CompletenessValidationResult:
        """
        キャラクター画像の完全性検証

        Args:
            image: 検証対象画像 (H, W, C) numpy配列

        Returns:
            CompletenessValidationResult: 検証結果
        """
        try:
            issues = []
            suggestions = []

            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 1. 断片化解析
            fragmentation_result = self._analyze_fragmentation(gray, issues, suggestions)

            # 2. 身体部位検出
            body_parts_result = self._detect_body_parts(gray, issues, suggestions)

            # 3. アスペクト比検証
            aspect_ratio_result = self._validate_aspect_ratio(gray, issues, suggestions)

            # 4. 完全性パーセンテージ計算
            completeness_percentage = self._calculate_completeness_percentage(
                body_parts_result, fragmentation_result, aspect_ratio_result
            )

            # 5. 検証信頼度計算
            validation_confidence = self._calculate_validation_confidence(
                body_parts_result, fragmentation_result, aspect_ratio_result
            )

            # 6. 完全性判定
            is_complete = (
                completeness_percentage >= self.completeness_threshold * 100
                and not fragmentation_result["is_fragmented"]
                and not aspect_ratio_result["has_warning"]
            )

            # 7. 品質指標の統合
            quality_indicators = {
                "overall_completeness": completeness_percentage / 100.0,
                "structural_integrity": 1.0 - fragmentation_result["fragmentation_level"],
                "aspect_ratio_quality": aspect_ratio_result["quality_score"],
                "body_coverage": body_parts_result["coverage_score"],
                "contour_quality": body_parts_result["contour_quality"],
            }

            return CompletenessValidationResult(
                is_complete=is_complete,
                completeness_percentage=completeness_percentage,
                validation_confidence=validation_confidence,
                missing_parts=body_parts_result["missing_parts"],
                detected_parts=body_parts_result["detected_parts"],
                fragmentation_detected=fragmentation_result["is_fragmented"],
                fragment_count=fragmentation_result["fragment_count"],
                aspect_ratio_warning=aspect_ratio_result["has_warning"],
                aspect_ratio_score=aspect_ratio_result["quality_score"],
                body_parts_analysis=body_parts_result["part_scores"],
                quality_indicators=quality_indicators,
                validation_issues=issues,
                improvement_suggestions=suggestions,
            )

        except Exception as e:
            return CompletenessValidationResult(
                is_complete=False,
                completeness_percentage=0.0,
                validation_confidence=0.0,
                missing_parts=["unknown"],
                detected_parts=[],
                fragmentation_detected=True,
                fragment_count=0,
                aspect_ratio_warning=True,
                aspect_ratio_score=0.0,
                body_parts_analysis={},
                quality_indicators={},
                validation_issues=[f"Validation error: {str(e)}"],
                improvement_suggestions=["Check image format and retry validation"],
            )

    def _analyze_fragmentation(
        self, gray: np.ndarray, issues: List[str], suggestions: List[str]
    ) -> Dict:
        """断片化の解析"""
        # 二値化
        binary_mask = (gray > 10).astype(np.uint8)

        # 連結成分解析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # 背景ラベル（0）を除外
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
        valid_components = [area for area in component_areas if area >= self.min_part_size]

        fragment_count = len(valid_components)
        is_fragmented = fragment_count > self.fragmentation_threshold

        # 断片化レベルの計算
        if fragment_count > 0:
            total_area = sum(valid_components)
            largest_fragment = max(valid_components)
            fragmentation_level = 1.0 - (largest_fragment / total_area)
        else:
            fragmentation_level = 1.0  # 完全に断片化

        if is_fragmented:
            issues.append(f"Character fragmented into {fragment_count} pieces")
            suggestions.append("Apply morphological closing to connect fragments")

        return {
            "is_fragmented": is_fragmented,
            "fragment_count": fragment_count,
            "fragmentation_level": fragmentation_level,
            "component_areas": valid_components,
            "largest_fragment_area": max(valid_components) if valid_components else 0,
        }

    def _detect_body_parts(
        self, gray: np.ndarray, issues: List[str], suggestions: List[str]
    ) -> Dict:
        """身体部位の検出"""
        h, w = gray.shape

        # 非ゼロピクセルの分布解析
        non_zero_mask = gray > 10
        y_coords, x_coords = np.where(non_zero_mask)

        if len(y_coords) == 0:
            issues.append("No character content detected")
            suggestions.append("Check image brightness and content")
            return {
                "missing_parts": ["head", "torso", "arms", "legs"],
                "detected_parts": [],
                "part_scores": {},
                "coverage_score": 0.0,
                "contour_quality": 0.0,
            }

        # 垂直方向の分布解析
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        char_height = max_y - min_y + 1

        # 身体部位の領域定義（3等分 + 調整）
        head_end = min_y + char_height // 4
        torso_end = min_y + 2 * char_height // 3

        # 各部位の存在確認
        head_pixels = np.sum((y_coords >= min_y) & (y_coords <= head_end))
        torso_pixels = np.sum((y_coords > head_end) & (y_coords <= torso_end))
        legs_pixels = np.sum(y_coords > torso_end)

        # 水平方向の拡がり（腕部の推定）
        center_x = (np.min(x_coords) + np.max(x_coords)) // 2
        torso_region_mask = (y_coords > head_end) & (y_coords <= torso_end)

        if np.sum(torso_region_mask) > 0:
            torso_x_coords = x_coords[torso_region_mask]
            torso_width = np.max(torso_x_coords) - np.min(torso_x_coords) + 1
            has_arms = torso_width > char_height // 5  # 身体の1/5以上の幅
        else:
            has_arms = False

        # 部位の存在判定
        detected_parts = []
        missing_parts = []
        part_scores = {}

        # 頭部
        head_threshold = self.min_part_size // 4
        if head_pixels >= head_threshold:
            detected_parts.append("head")
            part_scores["head"] = min(1.0, head_pixels / (head_threshold * 2))
        else:
            missing_parts.append("head")
            part_scores["head"] = 0.0

        # 胴体
        torso_threshold = self.min_part_size // 2
        if torso_pixels >= torso_threshold:
            detected_parts.append("torso")
            part_scores["torso"] = min(1.0, torso_pixels / (torso_threshold * 2))
        else:
            missing_parts.append("torso")
            part_scores["torso"] = 0.0

        # 腕部
        if has_arms:
            detected_parts.append("arms")
            part_scores["arms"] = 0.8  # 形状ベースの推定なので控えめ
        else:
            missing_parts.append("arms")
            part_scores["arms"] = 0.0

        # 脚部
        legs_threshold = self.min_part_size // 3
        if legs_pixels >= legs_threshold:
            detected_parts.append("legs")
            part_scores["legs"] = min(1.0, legs_pixels / (legs_threshold * 2))
        else:
            missing_parts.append("legs")
            part_scores["legs"] = 0.0

        # カバレッジスコア
        total_detected = len(detected_parts)
        coverage_score = total_detected / 4.0  # 4つの主要部位

        # 輪郭品質の評価
        contours, _ = cv2.findContours(
            non_zero_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter**2)
                # 人型キャラクターの適切な複雑さ
                contour_quality = 1.0 - abs(circularity - 0.3) / 0.7
                contour_quality = max(0.0, min(1.0, contour_quality))
            else:
                contour_quality = 0.0
        else:
            contour_quality = 0.0

        # 欠損部位に関する提案
        if "head" in missing_parts:
            issues.append("Head region missing or too small")
            suggestions.append("Check upper portion of character extraction")

        if "legs" in missing_parts:
            issues.append("Leg region missing or too small")
            suggestions.append("Ensure full-body extraction including lower limbs")

        if "torso" in missing_parts:
            issues.append("Torso region missing - critical structural issue")
            suggestions.append("Review core character region detection")

        return {
            "missing_parts": missing_parts,
            "detected_parts": detected_parts,
            "part_scores": part_scores,
            "coverage_score": coverage_score,
            "contour_quality": contour_quality,
        }

    def _validate_aspect_ratio(
        self, gray: np.ndarray, issues: List[str], suggestions: List[str]
    ) -> Dict:
        """アスペクト比の検証"""
        h, w = gray.shape
        aspect_ratio = h / w

        # 理想値からの偏差
        deviation = abs(aspect_ratio - self.ideal_aspect_ratio)
        normalized_deviation = deviation / self.ideal_aspect_ratio

        # 品質スコアの計算
        if normalized_deviation <= 0.2:  # 20%以内
            quality_score = 1.0
            has_warning = False
        elif normalized_deviation <= 0.4:  # 40%以内
            quality_score = 0.8
            has_warning = False
        elif normalized_deviation <= 0.6:  # 60%以内
            quality_score = 0.6
            has_warning = True
        else:
            quality_score = 0.3
            has_warning = True

        # 具体的な問題の特定
        if aspect_ratio < 1.0:
            issues.append(f"Unusually wide aspect ratio: {aspect_ratio:.2f}")
            suggestions.append("Check for horizontal truncation of limbs")
        elif aspect_ratio > 4.0:
            issues.append(f"Extremely tall aspect ratio: {aspect_ratio:.2f}")
            suggestions.append("Verify character is not stretched vertically")
        elif deviation > self.aspect_ratio_tolerance:
            if aspect_ratio < self.ideal_aspect_ratio:
                issues.append(f"Character appears too wide: {aspect_ratio:.2f}")
                suggestions.append("Consider tighter cropping or check for missing parts")
            else:
                issues.append(f"Character appears too tall: {aspect_ratio:.2f}")

        return {
            "aspect_ratio": aspect_ratio,
            "ideal_ratio": self.ideal_aspect_ratio,
            "deviation": deviation,
            "normalized_deviation": normalized_deviation,
            "quality_score": quality_score,
            "has_warning": has_warning,
        }

    def _calculate_completeness_percentage(
        self, body_parts_result: Dict, fragmentation_result: Dict, aspect_ratio_result: Dict
    ) -> float:
        """完全性パーセンテージの計算"""
        # 身体部位の完全性（40%の重み）
        body_completeness = body_parts_result["coverage_score"] * 40

        # 構造的完全性（30%の重み）
        structural_completeness = (1.0 - fragmentation_result["fragmentation_level"]) * 30

        # アスペクト比の適正性（20%の重み）
        aspect_completeness = aspect_ratio_result["quality_score"] * 20

        # 輪郭品質（10%の重み）
        contour_completeness = body_parts_result["contour_quality"] * 10

        total_percentage = (
            body_completeness + structural_completeness + aspect_completeness + contour_completeness
        )

        return min(100.0, max(0.0, total_percentage))

    def _calculate_validation_confidence(
        self, body_parts_result: Dict, fragmentation_result: Dict, aspect_ratio_result: Dict
    ) -> float:
        """検証信頼度の計算"""
        confidence_factors = []

        # 断片化が少ないほど高信頼
        fragmentation_confidence = 1.0 - fragmentation_result["fragmentation_level"]
        confidence_factors.append(fragmentation_confidence)

        # 輪郭品質が高いほど高信頼
        contour_confidence = body_parts_result["contour_quality"]
        confidence_factors.append(contour_confidence)

        # アスペクト比が適切ほど高信頼
        aspect_confidence = aspect_ratio_result["quality_score"]
        confidence_factors.append(aspect_confidence)

        # 検出部位数が多いほど高信頼
        parts_confidence = body_parts_result["coverage_score"]
        confidence_factors.append(parts_confidence)

        # 重み付き平均
        weights = [0.3, 0.25, 0.25, 0.2]
        confidence = sum(cf * w for cf, w in zip(confidence_factors, weights))

        return min(1.0, max(0.0, confidence))

    def detect_missing_parts(self, image: np.ndarray) -> List[str]:
        """欠損部位検出（外部用簡易メソッド）"""
        result = self.validate_completeness(image)
        return result.missing_parts

    def get_validation_summary(self, result: CompletenessValidationResult) -> Dict[str, any]:
        """検証結果のサマリー取得"""
        return {
            "overall_status": "complete" if result.is_complete else "incomplete",
            "completeness_grade": self._grade_completeness(result.completeness_percentage),
            "primary_issues": result.missing_parts[:3] if result.missing_parts else [],
            "confidence_level": "high"
            if result.validation_confidence > 0.8
            else "medium"
            if result.validation_confidence > 0.6
            else "low",
            "fragmentation_status": "fragmented" if result.fragmentation_detected else "intact",
            "aspect_ratio_status": "warning" if result.aspect_ratio_warning else "normal",
            "recommended_actions": result.improvement_suggestions[:3],
        }

    def _grade_completeness(self, percentage: float) -> str:
        """完全性パーセンテージのグレード化"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
