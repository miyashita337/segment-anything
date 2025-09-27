"""
KIRO-012: 全身検出専用モジュール

fullbody方式の評価を専門に行うモジュール
キャラクターの全身性（頭部から足先まで）の完全性を評価
"""

import time
import numpy as np
import cv2
import math
from typing import Dict, List, Optional, Tuple

from .module_interfaces import (
    ConfigurableModule,
    JudgmentInput,
    JudgmentResult,
    QualityGrade
)


class FullbodyDetectionModule(ConfigurableModule):
    """全身検出専用モジュール - fullbody方式実装"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.module_name = "FullbodyDetection"
        self.version = "1.0.0"

    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値設定"""
        return {
            'grade_a_threshold': 0.90,
            'grade_b_threshold': 0.75,
            'grade_c_threshold': 0.60,
            'grade_d_threshold': 0.45,
            'head_detection_weight': 0.25,
            'torso_detection_weight': 0.30,
            'limbs_detection_weight': 0.25,
            'proportions_weight': 0.20,
            'min_fullbody_ratio': 0.15,  # 最小全身面積比
            'ideal_head_ratio': 0.12,   # 理想的頭身比
            'min_aspect_ratio': 1.2,    # 全身の最小縦横比
            'max_aspect_ratio': 4.0,    # 全身の最大縦横比
            'completeness_threshold': 0.8,  # 完全性閾値
        }

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """全身検出判定の実行"""
        start_time = time.time()

        if not self.validate_input(input_data):
            return self._create_error_result("Invalid input data", start_time)

        try:
            image = input_data.image
            mask = input_data.mask
            issues = []
            recommendations = []
            metrics = {}

            # 身体部位検出分析
            body_analysis = self._analyze_body_parts(image, mask)
            metrics.update(body_analysis)

            # 1. 頭部検出評価
            head_score = self._evaluate_head_detection(
                body_analysis, issues, recommendations
            )
            metrics['head_detection_score'] = head_score

            # 2. 胴体検出評価
            torso_score = self._evaluate_torso_detection(
                body_analysis, issues, recommendations
            )
            metrics['torso_detection_score'] = torso_score

            # 3. 四肢検出評価
            limbs_score = self._evaluate_limbs_detection(
                body_analysis, issues, recommendations
            )
            metrics['limbs_detection_score'] = limbs_score

            # 4. 身体比例評価
            proportions_score = self._evaluate_body_proportions(
                body_analysis, issues, recommendations
            )
            metrics['proportions_score'] = proportions_score

            # 重み付き総合スコア計算
            weights = self._current_thresholds
            overall_score = (
                head_score * weights['head_detection_weight'] +
                torso_score * weights['torso_detection_weight'] +
                limbs_score * weights['limbs_detection_weight'] +
                proportions_score * weights['proportions_weight']
            )

            metrics['overall_fullbody_score'] = overall_score

            # グレード判定
            grade = self._calculate_grade(overall_score)

            # 信頼度計算
            score_variance = np.var([head_score, torso_score, limbs_score, proportions_score])
            confidence = max(0.6, 1.0 - score_variance)

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
            return self._create_error_result(f"Fullbody detection error: {str(e)}", start_time)

    def _analyze_body_parts(self, image: np.ndarray,
                           mask: Optional[np.ndarray]) -> Dict[str, float]:
        """身体部位の分析"""
        if mask is not None:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            # マスクがない場合はグレースケール閾値で推定
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            binary_mask = (gray > 30).astype(np.uint8)

        # 全体のバウンディングボックス取得
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'total_area': 0,
                'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 0, 'bbox_h': 0,
                'aspect_ratio': 0,
                'head_region_score': 0,
                'torso_region_score': 0,
                'limbs_region_score': 0,
                'vertical_distribution': 0
            }

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        total_area = cv2.contourArea(largest_contour)
        aspect_ratio = h / w if w > 0 else 0

        # 垂直分割による身体部位推定
        head_region, torso_region, limbs_region = self._segment_body_regions(
            binary_mask, x, y, w, h
        )

        # 各領域のスコア計算
        head_score = self._analyze_region_density(head_region)
        torso_score = self._analyze_region_density(torso_region)
        limbs_score = self._analyze_region_density(limbs_region)

        # 垂直分布の分析
        vertical_dist = self._analyze_vertical_distribution(binary_mask, y, h)

        return {
            'total_area': total_area,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'aspect_ratio': aspect_ratio,
            'head_region_score': head_score,
            'torso_region_score': torso_score,
            'limbs_region_score': limbs_score,
            'vertical_distribution': vertical_dist
        }

    def _segment_body_regions(self, binary_mask: np.ndarray, x: int, y: int,
                            w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """身体領域の分割"""
        # 頭部領域（上部20%）
        head_h = int(h * 0.2)
        head_region = binary_mask[y:y+head_h, x:x+w]

        # 胴体領域（中央50%）
        torso_start = y + head_h
        torso_h = int(h * 0.5)
        torso_region = binary_mask[torso_start:torso_start+torso_h, x:x+w]

        # 四肢領域（下部30%）
        limbs_start = torso_start + torso_h
        limbs_h = h - head_h - torso_h
        limbs_region = binary_mask[limbs_start:limbs_start+limbs_h, x:x+w]

        return head_region, torso_region, limbs_region

    def _analyze_region_density(self, region: np.ndarray) -> float:
        """領域の密度分析"""
        if region.size == 0:
            return 0.0

        total_pixels = region.size
        filled_pixels = np.sum(region > 0)
        density = filled_pixels / total_pixels

        return density

    def _analyze_vertical_distribution(self, binary_mask: np.ndarray,
                                     start_y: int, height: int) -> float:
        """垂直分布の分析"""
        if height == 0:
            return 0.0

        # 垂直プロファイル取得
        vertical_profile = np.sum(binary_mask[start_y:start_y+height, :], axis=1)

        # 分布の均等性評価
        if len(vertical_profile) == 0:
            return 0.0

        # 上中下の分布バランス
        third = len(vertical_profile) // 3
        upper_density = np.mean(vertical_profile[:third]) if third > 0 else 0
        middle_density = np.mean(vertical_profile[third:2*third]) if third > 0 else 0
        lower_density = np.mean(vertical_profile[2*third:]) if third > 0 else 0

        # 理想的には上<中>下の分布
        balance_score = 0.0
        if middle_density > upper_density and middle_density > lower_density:
            balance_score = 0.8
        elif middle_density > 0:
            balance_score = 0.6
        else:
            balance_score = 0.3

        return balance_score

    def _evaluate_head_detection(self, body_analysis: Dict, issues: List[str],
                               recommendations: List[str]) -> float:
        """頭部検出の評価"""
        head_score = body_analysis['head_region_score']
        total_area = body_analysis['total_area']

        if head_score < 0.3:
            issues.append("Head region not clearly detected")
            recommendations.append("Ensure character head is visible and unobstructed")
            return 0.2

        # 頭部サイズの妥当性
        head_area_ratio = head_score * 0.2  # 上部20%の密度
        ideal_head_ratio = self._current_thresholds['ideal_head_ratio']

        if head_area_ratio < ideal_head_ratio * 0.5:
            issues.append("Head appears too small relative to body")
            recommendations.append("Check if full head is visible")
            return 0.5
        elif head_area_ratio > ideal_head_ratio * 2:
            issues.append("Head appears disproportionately large")
            return 0.7
        else:
            return 1.0

    def _evaluate_torso_detection(self, body_analysis: Dict, issues: List[str],
                                recommendations: List[str]) -> float:
        """胴体検出の評価"""
        torso_score = body_analysis['torso_region_score']

        if torso_score < 0.4:
            issues.append("Torso region not adequately detected")
            recommendations.append("Ensure character torso is fully visible")
            return 0.3

        # 胴体は通常最も密度が高い領域
        if torso_score < 0.6:
            issues.append("Torso detection quality could be improved")
            recommendations.append("Check for occlusion or pose issues")
            return 0.6
        else:
            return 1.0

    def _evaluate_limbs_detection(self, body_analysis: Dict, issues: List[str],
                                recommendations: List[str]) -> float:
        """四肢検出の評価"""
        limbs_score = body_analysis['limbs_region_score']

        if limbs_score < 0.2:
            issues.append("Limbs (arms/legs) not detected")
            recommendations.append("Ensure full body including limbs is visible")
            return 0.1

        if limbs_score < 0.4:
            issues.append("Partial limb detection - may be cropped")
            recommendations.append("Include full character body in frame")
            return 0.5
        else:
            return 1.0

    def _evaluate_body_proportions(self, body_analysis: Dict, issues: List[str],
                                  recommendations: List[str]) -> float:
        """身体比例の評価"""
        aspect_ratio = body_analysis['aspect_ratio']
        min_aspect = self._current_thresholds['min_aspect_ratio']
        max_aspect = self._current_thresholds['max_aspect_ratio']

        proportion_score = 0.0

        # アスペクト比評価
        if aspect_ratio < min_aspect:
            issues.append(f"Character too wide for fullbody: aspect ratio {aspect_ratio:.2f}")
            recommendations.append("Character may be crouching or incomplete")
            proportion_score = 0.3
        elif aspect_ratio > max_aspect:
            issues.append(f"Character unusually tall/narrow: aspect ratio {aspect_ratio:.2f}")
            recommendations.append("Check for stretching or unusual pose")
            proportion_score = 0.5
        else:
            # 理想的な全身比例（1.5-2.5程度）
            if 1.5 <= aspect_ratio <= 2.5:
                proportion_score = 1.0
            else:
                proportion_score = 0.8

        # 垂直分布評価
        vertical_dist = body_analysis['vertical_distribution']
        proportion_score = (proportion_score + vertical_dist) / 2.0

        return proportion_score

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