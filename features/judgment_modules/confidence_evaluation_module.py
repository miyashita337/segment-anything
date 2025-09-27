"""
KIRO-012: 信頼度評価専用モジュール

confidence方式の評価を専門に行うモジュール
SAMの信頼度スコアとマスク品質の相関関係を評価
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Optional

from .module_interfaces import (
    ConfigurableModule,
    JudgmentInput,
    JudgmentResult,
    QualityGrade
)


class ConfidenceEvaluationModule(ConfigurableModule):
    """信頼度評価専用モジュール - confidence方式実装"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.module_name = "ConfidenceEvaluation"
        self.version = "1.0.0"

    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値設定"""
        return {
            'grade_a_threshold': 0.90,
            'grade_b_threshold': 0.75,
            'grade_c_threshold': 0.60,
            'grade_d_threshold': 0.45,
            'sam_confidence_weight': 0.4,
            'mask_consistency_weight': 0.3,
            'boundary_quality_weight': 0.2,
            'stability_weight': 0.1,
            'min_mask_area_ratio': 0.01,
            'max_mask_holes': 5,
            'boundary_smoothness_threshold': 0.7
        }

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """信頼度評価判定の実行"""
        start_time = time.time()

        if not self.validate_input(input_data):
            return self._create_error_result("Invalid input data", start_time)

        try:
            image = input_data.image
            mask = input_data.mask
            metadata = input_data.metadata or {}

            issues = []
            recommendations = []
            metrics = {}

            # 1. SAM信頼度スコア評価
            sam_confidence = self._evaluate_sam_confidence(metadata, issues, recommendations)
            metrics['sam_confidence'] = sam_confidence

            # 2. マスク一貫性評価
            if mask is not None:
                mask_consistency = self._evaluate_mask_consistency(mask, issues, recommendations)
                boundary_quality = self._evaluate_boundary_quality(mask, issues, recommendations)
            else:
                mask_consistency = 0.5  # マスクなしの場合は中立
                boundary_quality = 0.5
                issues.append("No mask provided for consistency evaluation")

            metrics['mask_consistency'] = mask_consistency
            metrics['boundary_quality'] = boundary_quality

            # 3. 予測安定性評価
            stability_score = self._evaluate_prediction_stability(
                image, metadata, issues, recommendations
            )
            metrics['stability'] = stability_score

            # 重み付き総合スコア計算
            weights = self._current_thresholds
            overall_score = (
                sam_confidence * weights['sam_confidence_weight'] +
                mask_consistency * weights['mask_consistency_weight'] +
                boundary_quality * weights['boundary_quality_weight'] +
                stability_score * weights['stability_weight']
            )

            metrics['overall_confidence'] = overall_score

            # グレード判定
            grade = self._calculate_grade(overall_score)

            # 信頼度計算（メタスコアとして）
            confidence_variance = np.var([sam_confidence, mask_consistency,
                                        boundary_quality, stability_score])
            meta_confidence = max(0.6, 1.0 - confidence_variance)

            processing_time = time.time() - start_time

            return self.create_result(
                grade=grade,
                confidence=meta_confidence,
                score=overall_score,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics,
                processing_time=processing_time
            )

        except Exception as e:
            return self._create_error_result(f"Confidence evaluation error: {str(e)}", start_time)

    def _evaluate_sam_confidence(self, metadata: Dict, issues: List[str],
                                recommendations: List[str]) -> float:
        """SAM信頼度スコアの評価"""
        # SAMのIoU予測スコアやStability scoreを使用
        sam_iou = metadata.get('sam_iou_prediction', 0.5)
        sam_stability = metadata.get('sam_stability_score', 0.5)

        # SAMの内部信頼度指標
        sam_confidence = (sam_iou + sam_stability) / 2.0

        if sam_confidence < 0.3:
            issues.append(f"Low SAM confidence: {sam_confidence:.3f}")
            recommendations.append("Consider re-running SAM with different parameters")
        elif sam_confidence > 0.9:
            recommendations.append("High SAM confidence - good segmentation quality")

        return sam_confidence

    def _evaluate_mask_consistency(self, mask: np.ndarray, issues: List[str],
                                  recommendations: List[str]) -> float:
        """マスク一貫性の評価"""
        if mask is None:
            return 0.0

        # バイナリマスクに変換
        binary_mask = (mask > 0.5).astype(np.uint8)

        # 1. マスク面積の妥当性
        mask_area = np.sum(binary_mask)
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = mask_area / total_area

        area_score = 0.0
        min_ratio = self._current_thresholds['min_mask_area_ratio']

        if area_ratio < min_ratio:
            issues.append(f"Mask area too small: {area_ratio*100:.1f}%")
            recommendations.append("Check segmentation parameters")
            area_score = 0.2
        elif area_ratio > 0.8:
            issues.append("Mask covers most of the image")
            recommendations.append("Improve background separation")
            area_score = 0.3
        else:
            area_score = 1.0

        # 2. 連結成分の評価
        connectivity_score = self._evaluate_connectivity(binary_mask, issues, recommendations)

        # 3. マスクの形状妥当性
        shape_score = self._evaluate_mask_shape(binary_mask, issues, recommendations)

        consistency_score = (area_score + connectivity_score + shape_score) / 3.0
        return consistency_score

    def _evaluate_connectivity(self, binary_mask: np.ndarray, issues: List[str],
                              recommendations: List[str]) -> float:
        """連結成分の評価"""
        # 連結成分ラベリング
        num_labels, labels = cv2.connectedComponents(binary_mask)

        # 背景を除いた連結成分数
        num_components = num_labels - 1

        if num_components == 0:
            issues.append("No connected components found")
            return 0.0
        elif num_components == 1:
            return 1.0  # 理想的
        elif num_components <= 3:
            issues.append(f"Multiple components: {num_components}")
            recommendations.append("Consider mask post-processing to merge components")
            return 0.7
        else:
            issues.append(f"Too many fragments: {num_components}")
            recommendations.append("Improve segmentation quality")
            return 0.3

    def _evaluate_mask_shape(self, binary_mask: np.ndarray, issues: List[str],
                           recommendations: List[str]) -> float:
        """マスク形状の妥当性評価"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)

        # 穴の数評価
        hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        if hierarchy is not None:
            holes = np.sum(hierarchy[0][:, 3] >= 0)  # 内側輪郭の数
            max_holes = self._current_thresholds['max_mask_holes']

            if holes > max_holes:
                issues.append(f"Too many holes in mask: {holes}")
                recommendations.append("Apply hole filling post-processing")
                hole_score = 0.3
            else:
                hole_score = 1.0
        else:
            hole_score = 1.0

        # 輪郭の滑らかさ評価
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        smoothness = len(approx) / len(largest_contour)

        threshold = self._current_thresholds['boundary_smoothness_threshold']
        if smoothness > threshold:
            smoothness_score = 0.5
            issues.append("Mask boundary too jagged")
            recommendations.append("Apply boundary smoothing")
        else:
            smoothness_score = 1.0

        return (hole_score + smoothness_score) / 2.0

    def _evaluate_boundary_quality(self, mask: np.ndarray, issues: List[str],
                                  recommendations: List[str]) -> float:
        """境界品質の評価"""
        if mask is None:
            return 0.0

        # グラデーションマスクの場合の境界解析
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            # ソフトマスクの境界品質
            gradient = np.gradient(mask)
            gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)

            # 境界の鮮明さ
            sharp_boundary_ratio = np.sum(gradient_magnitude > 0.1) / gradient_magnitude.size

            if sharp_boundary_ratio < 0.1:
                issues.append("Boundary too blurry")
                recommendations.append("Increase mask edge sharpness")
                return 0.4
            elif sharp_boundary_ratio > 0.5:
                issues.append("Boundary too sharp - may be artificial")
                return 0.7
            else:
                return 1.0
        else:
            # バイナリマスクの境界品質
            binary_mask = (mask > 0.5).astype(np.uint8)
            edges = cv2.Canny(binary_mask * 255, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if 0.05 <= edge_density <= 0.2:
                return 1.0
            elif edge_density < 0.05:
                issues.append("Insufficient edge definition")
                return 0.5
            else:
                issues.append("Excessive edge fragmentation")
                return 0.6

    def _evaluate_prediction_stability(self, image: np.ndarray, metadata: Dict,
                                     issues: List[str], recommendations: List[str]) -> float:
        """予測安定性の評価"""
        # マルチスケール実行結果の一貫性
        multiscale_consistency = metadata.get('multiscale_consistency', 0.7)

        # パラメータ摂動に対する安定性
        parameter_stability = metadata.get('parameter_stability', 0.7)

        # ノイズ耐性
        noise_robustness = metadata.get('noise_robustness', 0.7)

        stability_score = (multiscale_consistency + parameter_stability + noise_robustness) / 3.0

        if stability_score < 0.5:
            issues.append(f"Low prediction stability: {stability_score:.3f}")
            recommendations.append("Consider ensemble methods or parameter tuning")

        return stability_score

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