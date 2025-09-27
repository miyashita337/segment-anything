"""
KIRO-012: 中央配置判定専用モジュール

central方式の評価を専門に行うモジュール
キャラクターの画像内中央配置と構図バランスを評価
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


class CentralPositioningModule(ConfigurableModule):
    """中央配置判定専用モジュール - central方式実装"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.module_name = "CentralPositioning"
        self.version = "1.0.0"

    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値設定"""
        return {
            'grade_a_threshold': 0.85,
            'grade_b_threshold': 0.70,
            'grade_c_threshold': 0.55,
            'grade_d_threshold': 0.40,
            'center_alignment_weight': 0.35,
            'mass_distribution_weight': 0.25,
            'margin_balance_weight': 0.25,
            'composition_weight': 0.15,
            'perfect_center_radius': 0.1,    # 完璧な中央の半径（画像対角線比）
            'good_center_radius': 0.2,       # 良好な中央の半径
            'acceptable_center_radius': 0.35, # 許容できる中央の半径
            'min_margin_ratio': 0.05,        # 最小マージン比
            'ideal_margin_ratio': 0.15,      # 理想マージン比
            'mass_center_tolerance': 0.15,   # 重心中央許容範囲
        }

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """中央配置判定の実行"""
        start_time = time.time()

        if not self.validate_input(input_data):
            return self._create_error_result("Invalid input data", start_time)

        try:
            image = input_data.image
            mask = input_data.mask
            issues = []
            recommendations = []
            metrics = {}

            # 画像とキャラクターの位置分析
            positioning_analysis = self._analyze_character_positioning(image, mask)
            metrics.update(positioning_analysis)

            # 1. 中央配置評価
            center_alignment_score = self._evaluate_center_alignment(
                positioning_analysis, issues, recommendations
            )
            metrics['center_alignment_score'] = center_alignment_score

            # 2. 質量分布評価
            mass_distribution_score = self._evaluate_mass_distribution(
                positioning_analysis, image, mask, issues, recommendations
            )
            metrics['mass_distribution_score'] = mass_distribution_score

            # 3. マージンバランス評価
            margin_balance_score = self._evaluate_margin_balance(
                positioning_analysis, issues, recommendations
            )
            metrics['margin_balance_score'] = margin_balance_score

            # 4. 構図評価
            composition_score = self._evaluate_composition(
                positioning_analysis, image, issues, recommendations
            )
            metrics['composition_score'] = composition_score

            # 重み付き総合スコア計算
            weights = self._current_thresholds
            overall_score = (
                center_alignment_score * weights['center_alignment_weight'] +
                mass_distribution_score * weights['mass_distribution_weight'] +
                margin_balance_score * weights['margin_balance_weight'] +
                composition_score * weights['composition_weight']
            )

            metrics['overall_central_score'] = overall_score

            # グレード判定
            grade = self._calculate_grade(overall_score)

            # 信頼度計算
            score_variance = np.var([center_alignment_score, mass_distribution_score,
                                   margin_balance_score, composition_score])
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
            return self._create_error_result(f"Central positioning error: {str(e)}", start_time)

    def _analyze_character_positioning(self, image: np.ndarray,
                                     mask: Optional[np.ndarray]) -> Dict[str, float]:
        """キャラクター配置の分析"""
        height, width = image.shape[:2]
        image_center_x = width / 2
        image_center_y = height / 2
        diagonal = math.sqrt(width**2 + height**2)

        if mask is not None:
            binary_mask = (mask > 0.5).astype(np.uint8)
            character_pixels = np.sum(binary_mask)

            # 重心計算
            moments = cv2.moments(binary_mask)
            if moments['m00'] > 0:
                center_of_mass_x = moments['m10'] / moments['m00']
                center_of_mass_y = moments['m01'] / moments['m00']
            else:
                center_of_mass_x = image_center_x
                center_of_mass_y = image_center_y

            # バウンディングボックス
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2
            else:
                x = y = w = h = 0
                bbox_center_x = image_center_x
                bbox_center_y = image_center_y
        else:
            # マスクがない場合の推定
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 閾値処理で前景を推定
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary)

            if moments['m00'] > 0:
                center_of_mass_x = moments['m10'] / moments['m00']
                center_of_mass_y = moments['m01'] / moments['m00']
            else:
                center_of_mass_x = image_center_x
                center_of_mass_y = image_center_y

            # 簡易バウンディングボックス
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2
            else:
                x = y = w = h = 0
                bbox_center_x = image_center_x
                bbox_center_y = image_center_y

            character_pixels = np.sum(binary > 0)

        # 距離計算
        mass_distance_from_center = math.sqrt(
            (center_of_mass_x - image_center_x)**2 + (center_of_mass_y - image_center_y)**2
        )
        bbox_distance_from_center = math.sqrt(
            (bbox_center_x - image_center_x)**2 + (bbox_center_y - image_center_y)**2
        )

        # 正規化距離（対角線比）
        mass_distance_normalized = mass_distance_from_center / diagonal
        bbox_distance_normalized = bbox_distance_from_center / diagonal

        # マージン計算
        margin_left = x
        margin_right = width - (x + w)
        margin_top = y
        margin_bottom = height - (y + h)

        return {
            'image_width': width,
            'image_height': height,
            'image_center_x': image_center_x,
            'image_center_y': image_center_y,
            'character_pixels': character_pixels,
            'center_of_mass_x': center_of_mass_x,
            'center_of_mass_y': center_of_mass_y,
            'bbox_center_x': bbox_center_x,
            'bbox_center_y': bbox_center_y,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'mass_distance_from_center': mass_distance_from_center,
            'bbox_distance_from_center': bbox_distance_from_center,
            'mass_distance_normalized': mass_distance_normalized,
            'bbox_distance_normalized': bbox_distance_normalized,
            'margin_left': margin_left,
            'margin_right': margin_right,
            'margin_top': margin_top,
            'margin_bottom': margin_bottom
        }

    def _evaluate_center_alignment(self, positioning_analysis: Dict, issues: List[str],
                                  recommendations: List[str]) -> float:
        """中央配置の評価"""
        mass_distance = positioning_analysis['mass_distance_normalized']
        bbox_distance = positioning_analysis['bbox_distance_normalized']

        # 重心とバウンディングボックス中心の平均距離
        avg_distance = (mass_distance + bbox_distance) / 2.0

        perfect_radius = self._current_thresholds['perfect_center_radius']
        good_radius = self._current_thresholds['good_center_radius']
        acceptable_radius = self._current_thresholds['acceptable_center_radius']

        alignment_score = 0.0

        if avg_distance <= perfect_radius:
            alignment_score = 1.0
            recommendations.append("Perfect center alignment achieved")
        elif avg_distance <= good_radius:
            alignment_score = 0.9
            recommendations.append("Good center alignment")
        elif avg_distance <= acceptable_radius:
            alignment_score = 0.7
            issues.append(f"Character slightly off-center: {avg_distance:.2f}")
            recommendations.append("Consider minor repositioning for better centering")
        else:
            alignment_score = 0.4
            issues.append(f"Character significantly off-center: {avg_distance:.2f}")
            recommendations.append("Recenter character in frame")

        return alignment_score

    def _evaluate_mass_distribution(self, positioning_analysis: Dict, image: np.ndarray,
                                   mask: Optional[np.ndarray], issues: List[str],
                                   recommendations: List[str]) -> float:
        """質量分布の評価"""
        mass_center_x = positioning_analysis['center_of_mass_x']
        mass_center_y = positioning_analysis['center_of_mass_y']
        image_center_x = positioning_analysis['image_center_x']
        image_center_y = positioning_analysis['image_center_y']

        # 質量中心と幾何中心の差
        mass_geometric_diff = math.sqrt(
            (mass_center_x - positioning_analysis['bbox_center_x'])**2 +
            (mass_center_y - positioning_analysis['bbox_center_y'])**2
        )

        diagonal = math.sqrt(
            positioning_analysis['image_width']**2 + positioning_analysis['image_height']**2
        )
        mass_geometric_diff_normalized = mass_geometric_diff / diagonal

        tolerance = self._current_thresholds['mass_center_tolerance']

        if mass_geometric_diff_normalized <= tolerance:
            mass_score = 1.0
        elif mass_geometric_diff_normalized <= tolerance * 2:
            mass_score = 0.7
            issues.append("Uneven mass distribution detected")
            recommendations.append("Check for pose asymmetry")
        else:
            mass_score = 0.4
            issues.append("Significant mass distribution imbalance")
            recommendations.append("Character pose may be heavily asymmetric")

        # 四象限分析
        quadrant_score = self._analyze_quadrant_distribution(
            image, mask, positioning_analysis, issues, recommendations
        )

        return (mass_score + quadrant_score) / 2.0

    def _analyze_quadrant_distribution(self, image: np.ndarray, mask: Optional[np.ndarray],
                                     positioning_analysis: Dict, issues: List[str],
                                     recommendations: List[str]) -> float:
        """四象限分布の分析"""
        center_x = int(positioning_analysis['image_center_x'])
        center_y = int(positioning_analysis['image_center_y'])
        height, width = image.shape[:2]

        if mask is not None:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            binary_mask = (gray > 30).astype(np.uint8)

        # 四象限の質量計算
        q1 = np.sum(binary_mask[:center_y, center_x:])      # 右上
        q2 = np.sum(binary_mask[:center_y, :center_x])      # 左上
        q3 = np.sum(binary_mask[center_y:, :center_x])      # 左下
        q4 = np.sum(binary_mask[center_y:, center_x:])      # 右下

        total_mass = q1 + q2 + q3 + q4
        if total_mass == 0:
            return 0.0

        # 各象限の比率
        ratios = np.array([q1, q2, q3, q4]) / total_mass

        # 理想的には各象限が0.25前後
        ideal_ratio = 0.25
        deviations = np.abs(ratios - ideal_ratio)
        max_deviation = np.max(deviations)

        if max_deviation < 0.1:
            quadrant_score = 1.0
        elif max_deviation < 0.15:
            quadrant_score = 0.8
        elif max_deviation < 0.25:
            quadrant_score = 0.6
            issues.append("Unbalanced quadrant distribution")
        else:
            quadrant_score = 0.3
            issues.append("Highly unbalanced quadrant distribution")
            recommendations.append("Consider reframing for better balance")

        return quadrant_score

    def _evaluate_margin_balance(self, positioning_analysis: Dict, issues: List[str],
                                recommendations: List[str]) -> float:
        """マージンバランスの評価"""
        left = positioning_analysis['margin_left']
        right = positioning_analysis['margin_right']
        top = positioning_analysis['margin_top']
        bottom = positioning_analysis['margin_bottom']

        width = positioning_analysis['image_width']
        height = positioning_analysis['image_height']

        # マージン比率
        left_ratio = left / width
        right_ratio = right / width
        top_ratio = top / height
        bottom_ratio = bottom / height

        min_margin = self._current_thresholds['min_margin_ratio']
        ideal_margin = self._current_thresholds['ideal_margin_ratio']

        margin_score = 0.0

        # 最小マージンチェック
        margins = [left_ratio, right_ratio, top_ratio, bottom_ratio]
        min_actual_margin = min(margins)

        if min_actual_margin < min_margin:
            issues.append(f"Insufficient margin: {min_actual_margin*100:.1f}%")
            recommendations.append("Increase margins around character")
            margin_score = 0.3
        elif min_actual_margin >= ideal_margin:
            margin_score = 1.0
        else:
            margin_score = 0.7

        # 左右バランス
        horizontal_imbalance = abs(left_ratio - right_ratio)
        vertical_imbalance = abs(top_ratio - bottom_ratio)

        if horizontal_imbalance > 0.1:
            issues.append(f"Horizontal margin imbalance: {horizontal_imbalance:.2f}")
            recommendations.append("Center character horizontally")
            margin_score *= 0.8

        if vertical_imbalance > 0.1:
            issues.append(f"Vertical margin imbalance: {vertical_imbalance:.2f}")
            recommendations.append("Center character vertically")
            margin_score *= 0.8

        return margin_score

    def _evaluate_composition(self, positioning_analysis: Dict, image: np.ndarray,
                            issues: List[str], recommendations: List[str]) -> float:
        """構図評価"""
        # 黄金比と三分割法の評価
        width = positioning_analysis['image_width']
        height = positioning_analysis['image_height']
        center_x = positioning_analysis['bbox_center_x']
        center_y = positioning_analysis['bbox_center_y']

        # 三分割法のポイント
        third_x1 = width / 3
        third_x2 = width * 2 / 3
        third_y1 = height / 3
        third_y2 = height * 2 / 3

        # 中央と三分割点への距離
        distances_to_thirds = [
            abs(center_x - third_x1),
            abs(center_x - third_x2),
            abs(center_y - third_y1),
            abs(center_y - third_y2)
        ]

        min_distance_to_third = min(distances_to_thirds)
        min_distance_normalized = min_distance_to_third / min(width, height)

        # 中央配置を重視しつつ、三分割法も考慮
        center_distance = math.sqrt(
            (center_x - width/2)**2 + (center_y - height/2)**2
        )
        center_distance_normalized = center_distance / math.sqrt(width**2 + height**2)

        composition_score = 0.0

        # 中央配置が最優先
        if center_distance_normalized < 0.1:
            composition_score = 1.0
        elif center_distance_normalized < 0.2:
            composition_score = 0.9
        elif min_distance_normalized < 0.1:  # 三分割法に近い
            composition_score = 0.7
            recommendations.append("Good composition using rule of thirds")
        else:
            composition_score = 0.5
            recommendations.append("Consider central or rule-of-thirds positioning")

        return composition_score

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