"""
KIRO-012: 品質評価専用モジュール

balanced方式の品質評価を専門に行うモジュール
従来のCharacterQualityAssessorから品質評価ロジックを抽出・モジュール化
"""

import time
import numpy as np
import cv2
import math
from typing import Dict, List, Optional

from .module_interfaces import (
    ConfigurableModule,
    JudgmentInput,
    JudgmentResult,
    QualityGrade
)


class QualityAssessmentModule(ConfigurableModule):
    """品質評価専用モジュール - balanced方式実装"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.module_name = "QualityAssessment"
        self.version = "1.0.0"

    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値設定"""
        return {
            'grade_a_threshold': 0.85,
            'grade_b_threshold': 0.70,
            'grade_c_threshold': 0.55,
            'grade_d_threshold': 0.40,
            'min_character_area': 1000.0,
            'ideal_character_ratio': 0.15,
            'min_contrast': 30.0,
            'min_edge_density': 0.05,
            'max_edge_density': 0.3,
            'min_solidity': 0.4,
            'max_solidity': 0.9,
            'texture_variance_threshold': 50.0
        }

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """品質評価判定の実行"""
        start_time = time.time()

        if not self.validate_input(input_data):
            return self._create_error_result("Invalid input data", start_time)

        try:
            image = input_data.image
            issues = []
            recommendations = []
            metrics = {}

            # 1. 完全性評価（輪郭の完全性）
            completeness_score, comp_issues, comp_recs = self._assess_completeness(image)
            issues.extend(comp_issues)
            recommendations.extend(comp_recs)
            metrics['completeness'] = completeness_score

            # 2. 明瞭性評価（エッジとコントラスト）
            clarity_score, clar_issues, clar_recs = self._assess_clarity(image)
            issues.extend(clar_issues)
            recommendations.extend(clar_recs)
            metrics['clarity'] = clarity_score

            # 3. サイズ適正性評価
            size_score, size_issues, size_recs = self._assess_size_adequacy(image)
            issues.extend(size_issues)
            recommendations.extend(size_recs)
            metrics['size_adequacy'] = size_score

            # 4. 形状品質評価
            shape_score, shape_issues, shape_recs = self._assess_shape_quality(image)
            issues.extend(shape_issues)
            recommendations.extend(shape_recs)
            metrics['shape_quality'] = shape_score

            # 5. 詳細保存評価
            detail_score, detail_issues, detail_recs = self._assess_detail_preservation(image)
            issues.extend(detail_issues)
            recommendations.extend(detail_recs)
            metrics['detail_preservation'] = detail_score

            # 総合スコア計算（重み付き平均）
            overall_score = (
                completeness_score * 0.25 +
                clarity_score * 0.20 +
                size_score * 0.20 +
                shape_score * 0.20 +
                detail_score * 0.15
            )

            metrics['overall_score'] = overall_score

            # グレード判定
            grade = self._calculate_grade(overall_score)

            # 信頼度計算（スコアの分散に基づく）
            score_variance = np.var([completeness_score, clarity_score, size_score,
                                   shape_score, detail_score])
            confidence = max(0.5, 1.0 - score_variance)

            processing_time = time.time() - start_time

            return self.create_result(
                grade=grade,
                confidence=confidence,
                score=overall_score,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics,
                processing_time=processing_time
            )

        except Exception as e:
            return self._create_error_result(f"Assessment error: {str(e)}", start_time)

    def _assess_completeness(self, image: np.ndarray) -> tuple[float, List[str], List[str]]:
        """完全性の評価（輪郭の完全性）"""
        issues = []
        recommendations = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 輪郭検出
        contours, _ = cv2.findContours(
            (gray > 10).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            issues.append("No character contour detected")
            recommendations.append("Check image brightness and contrast")
            return 0.0, issues, recommendations

        # 最大輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < self._current_thresholds['min_character_area']:
            issues.append(f"Character too small: {contour_area} pixels")
            recommendations.append("Increase character size in source image")

        # 輪郭の滑らかさ評価
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            compactness = 4 * math.pi * contour_area / (perimeter ** 2)
            # 人型キャラクターの適正範囲: 0.1-0.6
            if 0.1 <= compactness <= 0.6:
                completeness_score = 0.9
            elif compactness < 0.1:
                completeness_score = 0.6
                issues.append("Character contour too elongated or fragmented")
                recommendations.append("Improve segmentation quality")
            else:
                completeness_score = 0.7
                issues.append("Character contour too circular")
        else:
            completeness_score = 0.0
            issues.append("Invalid contour detected")

        return min(1.0, max(0.0, completeness_score)), issues, recommendations

    def _assess_clarity(self, image: np.ndarray) -> tuple[float, List[str], List[str]]:
        """明瞭性の評価（エッジとコントラスト）"""
        issues = []
        recommendations = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # コントラスト評価
        contrast = np.std(gray)

        # 明度評価
        mean_brightness = np.mean(gray)

        clarity_score = 0.0

        # エッジ密度評価
        min_edge = self._current_thresholds['min_edge_density']
        max_edge = self._current_thresholds['max_edge_density']

        if min_edge <= edge_density <= max_edge:
            clarity_score += 0.4
        elif edge_density < min_edge:
            clarity_score += 0.2
            issues.append("Low edge density - character may be blurry")
            recommendations.append("Improve image sharpness")
        else:
            clarity_score += 0.3
            issues.append("High edge density - may be noisy")

        # コントラスト評価
        min_contrast = self._current_thresholds['min_contrast']
        if contrast > min_contrast:
            clarity_score += 0.3
        elif contrast > min_contrast / 2:
            clarity_score += 0.2
        else:
            clarity_score += 0.1
            issues.append(f"Low contrast: {contrast:.1f}")
            recommendations.append("Increase image contrast")

        # 明度評価
        if 50 <= mean_brightness <= 200:
            clarity_score += 0.3
        else:
            clarity_score += 0.1
            if mean_brightness < 50:
                issues.append("Character too dark")
                recommendations.append("Increase brightness")
            else:
                issues.append("Character too bright")

        return min(1.0, max(0.0, clarity_score)), issues, recommendations

    def _assess_size_adequacy(self, image: np.ndarray) -> tuple[float, List[str], List[str]]:
        """サイズ適正性の評価"""
        issues = []
        recommendations = []

        total_area = image.shape[0] * image.shape[1]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        character_pixels = np.sum(gray > 10)
        character_ratio = character_pixels / total_area

        # 理想的な比率との比較
        ideal_ratio = self._current_thresholds['ideal_character_ratio']
        ratio_diff = abs(character_ratio - ideal_ratio)

        if ratio_diff < 0.05:
            size_score = 1.0
        elif ratio_diff < 0.1:
            size_score = 0.8
        elif ratio_diff < 0.2:
            size_score = 0.6
        else:
            size_score = 0.4

        if character_ratio < 0.05:
            issues.append(f"Character too small: {character_ratio*100:.1f}% of image")
            recommendations.append("Increase character size or crop tighter")
        elif character_ratio > 0.7:
            issues.append(f"Character too large: {character_ratio*100:.1f}% of image")
            recommendations.append("Add more background or crop looser")

        return size_score, issues, recommendations

    def _assess_shape_quality(self, image: np.ndarray) -> tuple[float, List[str], List[str]]:
        """形状品質の評価"""
        issues = []
        recommendations = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        contours, _ = cv2.findContours(
            (gray > 10).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0, issues, recommendations

        largest_contour = max(contours, key=cv2.contourArea)

        # 凸包との比較（凹凸の評価）
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(largest_contour)

        if hull_area > 0:
            solidity = contour_area / hull_area
            min_solidity = self._current_thresholds['min_solidity']
            max_solidity = self._current_thresholds['max_solidity']

            # 人型キャラクターは適度な凹凸を持つ
            if min_solidity <= solidity <= max_solidity:
                shape_score = 0.9
            elif 0.4 <= solidity < min_solidity:
                shape_score = 0.7
                issues.append("Character shape has many concave regions")
            else:
                shape_score = 0.5
                if solidity > max_solidity:
                    issues.append("Character shape too convex")
                else:
                    issues.append("Character shape too fragmented")
        else:
            shape_score = 0.0

        return shape_score, issues, recommendations

    def _assess_detail_preservation(self, image: np.ndarray) -> tuple[float, List[str], List[str]]:
        """詳細保存性の評価"""
        issues = []
        recommendations = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # テクスチャの評価（Laplacianバリアンス）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()

        # ヒストグラムの分散（詳細の多様性）
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum(hist * np.log(hist + 1e-8))

        detail_score = 0.0
        threshold = self._current_thresholds['texture_variance_threshold']

        # テクスチャ評価
        if texture_variance > threshold * 2:
            detail_score += 0.5
        elif texture_variance > threshold:
            detail_score += 0.3
        else:
            detail_score += 0.1
            issues.append("Low texture detail")
            recommendations.append("Improve image resolution or reduce smoothing")

        # エントロピー評価
        normalized_entropy = hist_entropy / 8.0  # log2(256)で正規化
        detail_score += min(0.5, normalized_entropy)

        return min(1.0, detail_score), issues, recommendations

    def _calculate_grade(self, score: float) -> QualityGrade:
        """スコアに基づくグレード判定"""
        thresholds = self._current_thresholds

        if score >= thresholds['grade_a_threshold']:
            return QualityGrade.A
        elif score >= thresholds['grade_b_threshold']:
            return QualityGrade.B
        elif score >= thresholds['grade_c_threshold']:
            return QualityGrade.C
        elif score >= thresholds['grade_d_threshold']:
            return QualityGrade.D
        else:
            return QualityGrade.F

    def _create_error_result(self, error_message: str, start_time: float) -> JudgmentResult:
        """エラー時の結果作成"""
        processing_time = time.time() - start_time
        return self.create_result(
            grade=QualityGrade.F,
            confidence=0.0,
            score=0.0,
            issues=[error_message],
            recommendations=["Check input data and retry"],
            metrics={},
            processing_time=processing_time
        )