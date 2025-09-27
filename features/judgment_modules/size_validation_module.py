"""
KIRO-012: サイズ検証専用モジュール

size方式の評価を専門に行うモジュール
キャラクターサイズと画像内での適切な配置を評価
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


class SizeValidationModule(ConfigurableModule):
    """サイズ検証専用モジュール - size方式実装"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.module_name = "SizeValidation"
        self.version = "1.0.0"

    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値設定"""
        return {
            'grade_a_threshold': 0.85,
            'grade_b_threshold': 0.70,
            'grade_c_threshold': 0.55,
            'grade_d_threshold': 0.40,
            'min_character_pixels': 2000,
            'max_character_pixels': 500000,
            'ideal_character_ratio': 0.25,
            'min_character_ratio': 0.03,
            'max_character_ratio': 0.85,
            'ideal_aspect_ratio': 0.6,  # height/width for character
            'min_aspect_ratio': 0.3,
            'max_aspect_ratio': 3.0,
            'resolution_quality_threshold': 64,  # minimum character height in pixels
            'size_consistency_weight': 0.3,
            'aspect_ratio_weight': 0.25,
            'resolution_weight': 0.25,
            'positioning_weight': 0.2
        }

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """サイズ検証判定の実行"""
        start_time = time.time()

        if not self.validate_input(input_data):
            return self._create_error_result("Invalid input data", start_time)

        try:
            image = input_data.image
            mask = input_data.mask
            issues = []
            recommendations = []
            metrics = {}

            # 画像とキャラクターの基本情報取得
            image_info = self._analyze_image_dimensions(image)
            character_info = self._analyze_character_dimensions(image, mask)

            metrics.update(image_info)
            metrics.update(character_info)

            # 1. サイズ一貫性評価
            size_consistency = self._evaluate_size_consistency(
                character_info, image_info, issues, recommendations
            )
            metrics['size_consistency'] = size_consistency

            # 2. アスペクト比評価
            aspect_ratio_score = self._evaluate_aspect_ratio(
                character_info, issues, recommendations
            )
            metrics['aspect_ratio_score'] = aspect_ratio_score

            # 3. 解像度品質評価
            resolution_score = self._evaluate_resolution_quality(
                character_info, issues, recommendations
            )
            metrics['resolution_score'] = resolution_score

            # 4. 配置評価
            positioning_score = self._evaluate_character_positioning(
                character_info, image_info, issues, recommendations
            )
            metrics['positioning_score'] = positioning_score

            # 重み付き総合スコア計算
            weights = self._current_thresholds
            overall_score = (
                size_consistency * weights['size_consistency_weight'] +
                aspect_ratio_score * weights['aspect_ratio_weight'] +
                resolution_score * weights['resolution_weight'] +
                positioning_score * weights['positioning_weight']
            )

            metrics['overall_size_score'] = overall_score

            # グレード判定
            grade = self._calculate_grade(overall_score)

            # 信頼度計算
            score_variance = np.var([size_consistency, aspect_ratio_score,
                                   resolution_score, positioning_score])
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
            return self._create_error_result(f"Size validation error: {str(e)}", start_time)

    def _analyze_image_dimensions(self, image: np.ndarray) -> Dict[str, float]:
        """画像の基本情報分析"""
        height, width = image.shape[:2]
        total_pixels = height * width
        aspect_ratio = height / width if width > 0 else 0

        return {
            'image_width': width,
            'image_height': height,
            'total_pixels': total_pixels,
            'image_aspect_ratio': aspect_ratio
        }

    def _analyze_character_dimensions(self, image: np.ndarray,
                                    mask: Optional[np.ndarray]) -> Dict[str, float]:
        """キャラクターの寸法分析"""
        if mask is not None:
            # マスクを使用したキャラクター領域分析
            binary_mask = (mask > 0.5).astype(np.uint8)
            character_pixels = np.sum(binary_mask)

            # バウンディングボックス取得
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox_area = w * h
                char_aspect_ratio = h / w if w > 0 else 0

                # 中心座標
                center_x = x + w / 2
                center_y = y + h / 2
            else:
                x = y = w = h = 0
                bbox_area = 0
                char_aspect_ratio = 0
                center_x = center_y = 0
        else:
            # マスクがない場合はグレースケール閾値で推定
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            character_pixels = np.sum(gray > 30)  # 暗い背景を想定

            # 簡易的なバウンディングボックス推定
            coords = np.column_stack(np.where(gray > 30))
            if len(coords) > 0:
                y, x = coords.min(axis=0)
                y2, x2 = coords.max(axis=0)
                w, h = x2 - x, y2 - y
                bbox_area = w * h
                char_aspect_ratio = h / w if w > 0 else 0
                center_x = x + w / 2
                center_y = y + h / 2
            else:
                x = y = w = h = 0
                bbox_area = 0
                char_aspect_ratio = 0
                center_x = center_y = 0

        total_pixels = image.shape[0] * image.shape[1]
        character_ratio = character_pixels / total_pixels if total_pixels > 0 else 0

        return {
            'character_pixels': character_pixels,
            'character_ratio': character_ratio,
            'character_bbox_x': x,
            'character_bbox_y': y,
            'character_width': w,
            'character_height': h,
            'character_bbox_area': bbox_area,
            'character_aspect_ratio': char_aspect_ratio,
            'character_center_x': center_x,
            'character_center_y': center_y
        }

    def _evaluate_size_consistency(self, character_info: Dict, image_info: Dict,
                                  issues: List[str], recommendations: List[str]) -> float:
        """サイズ一貫性の評価"""
        character_pixels = character_info['character_pixels']
        character_ratio = character_info['character_ratio']

        # 絶対サイズチェック
        min_pixels = self._current_thresholds['min_character_pixels']
        max_pixels = self._current_thresholds['max_character_pixels']

        size_score = 0.0

        if character_pixels < min_pixels:
            issues.append(f"Character too small: {character_pixels} pixels")
            recommendations.append("Increase image resolution or crop tighter")
            size_score = 0.2
        elif character_pixels > max_pixels:
            issues.append(f"Character too large: {character_pixels} pixels")
            recommendations.append("Reduce image size or add more background")
            size_score = 0.3
        else:
            size_score = 1.0

        # 相対サイズチェック
        min_ratio = self._current_thresholds['min_character_ratio']
        max_ratio = self._current_thresholds['max_character_ratio']
        ideal_ratio = self._current_thresholds['ideal_character_ratio']

        ratio_score = 0.0

        if character_ratio < min_ratio:
            issues.append(f"Character occupies too little space: {character_ratio*100:.1f}%")
            recommendations.append("Crop image tighter around character")
            ratio_score = 0.3
        elif character_ratio > max_ratio:
            issues.append(f"Character occupies too much space: {character_ratio*100:.1f}%")
            recommendations.append("Include more background context")
            ratio_score = 0.4
        else:
            # 理想比率との距離で評価
            ratio_diff = abs(character_ratio - ideal_ratio)
            if ratio_diff < 0.05:
                ratio_score = 1.0
            elif ratio_diff < 0.1:
                ratio_score = 0.8
            else:
                ratio_score = 0.6

        return (size_score + ratio_score) / 2.0

    def _evaluate_aspect_ratio(self, character_info: Dict, issues: List[str],
                              recommendations: List[str]) -> float:
        """アスペクト比の評価"""
        aspect_ratio = character_info['character_aspect_ratio']

        if aspect_ratio == 0:
            issues.append("Cannot determine character aspect ratio")
            return 0.0

        min_aspect = self._current_thresholds['min_aspect_ratio']
        max_aspect = self._current_thresholds['max_aspect_ratio']
        ideal_aspect = self._current_thresholds['ideal_aspect_ratio']

        if aspect_ratio < min_aspect:
            issues.append(f"Character too wide: aspect ratio {aspect_ratio:.2f}")
            recommendations.append("Adjust cropping to include more height")
            return 0.4
        elif aspect_ratio > max_aspect:
            issues.append(f"Character too tall/narrow: aspect ratio {aspect_ratio:.2f}")
            recommendations.append("Adjust cropping to include more width")
            return 0.4
        else:
            # 理想アスペクト比との距離で評価
            aspect_diff = abs(aspect_ratio - ideal_aspect)
            if aspect_diff < 0.1:
                return 1.0
            elif aspect_diff < 0.3:
                return 0.8
            else:
                return 0.6

    def _evaluate_resolution_quality(self, character_info: Dict, issues: List[str],
                                   recommendations: List[str]) -> float:
        """解像度品質の評価"""
        char_height = character_info['character_height']
        char_width = character_info['character_width']

        min_dimension = min(char_height, char_width)
        threshold = self._current_thresholds['resolution_quality_threshold']

        if min_dimension < threshold / 2:
            issues.append(f"Very low character resolution: {min_dimension} pixels")
            recommendations.append("Use higher resolution source image")
            return 0.2
        elif min_dimension < threshold:
            issues.append(f"Low character resolution: {min_dimension} pixels")
            recommendations.append("Consider upscaling or using higher resolution source")
            return 0.5
        elif min_dimension > threshold * 4:
            # 非常に高解像度も問題となる場合がある
            recommendations.append("High resolution - ensure processing efficiency")
            return 0.9
        else:
            return 1.0

    def _evaluate_character_positioning(self, character_info: Dict, image_info: Dict,
                                      issues: List[str], recommendations: List[str]) -> float:
        """キャラクター配置の評価"""
        center_x = character_info['character_center_x']
        center_y = character_info['character_center_y']
        image_width = image_info['image_width']
        image_height = image_info['image_height']

        if image_width == 0 or image_height == 0:
            return 0.0

        # 画像中心からの距離
        image_center_x = image_width / 2
        image_center_y = image_height / 2

        distance_x = abs(center_x - image_center_x) / image_width
        distance_y = abs(center_y - image_center_y) / image_height

        # 中心からの距離に基づく評価
        center_distance = math.sqrt(distance_x**2 + distance_y**2)

        positioning_score = 0.0

        if center_distance < 0.1:
            positioning_score = 1.0  # 完全に中央
        elif center_distance < 0.2:
            positioning_score = 0.9  # ほぼ中央
        elif center_distance < 0.4:
            positioning_score = 0.7  # やや偏心
        else:
            positioning_score = 0.5  # 大きく偏心
            issues.append(f"Character off-center: distance {center_distance:.2f}")
            recommendations.append("Consider recentering the character in frame")

        # 端に近すぎる場合の評価
        margin_x = min(center_x, image_width - center_x) / image_width
        margin_y = min(center_y, image_height - center_y) / image_height

        if margin_x < 0.05 or margin_y < 0.05:
            issues.append("Character too close to image edge")
            recommendations.append("Increase margins around character")
            positioning_score *= 0.7

        return positioning_score

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