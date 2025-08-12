"""
QI-002: キャラクター品質評価器 (CharacterQualityAssessor)

個別キャラクターおよび複数キャラクターシーンの品質を総合的に評価します。
"""

import numpy as np
import cv2
from typing import List, NamedTuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class IndividualCharacterQuality:
    """個別キャラクター品質評価結果"""
    overall_quality_score: float
    completeness_score: float
    clarity_score: float
    size_adequacy_score: float
    shape_quality_score: float
    detail_preservation_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class MultiCharacterQuality:
    """複数キャラクターシーン品質評価結果"""
    character_count: int
    scene_balance_score: float
    character_separation_quality: float
    character_interaction_score: float
    individual_character_scores: List[IndividualCharacterQuality]
    proximity_analysis: Dict[str, float]
    has_character_interaction: bool
    overall_scene_quality: float
    scene_issues: List[str]
    scene_recommendations: List[str]


class CharacterQualityAssessor:
    """キャラクター品質評価を行うクラス"""
    
    def __init__(self,
                 min_character_area: int = 1000,
                 ideal_character_ratio: float = 0.15,
                 interaction_distance_threshold: float = 100.0):
        """
        CharacterQualityAssessor の初期化
        
        Args:
            min_character_area: 最小キャラクター面積
            ideal_character_ratio: 理想的なキャラクター面積比
            interaction_distance_threshold: キャラクター相互作用距離閾値
        """
        self.min_character_area = min_character_area
        self.ideal_character_ratio = ideal_character_ratio
        self.interaction_distance_threshold = interaction_distance_threshold
    
    def assess_individual_character_quality(self, character_image: np.ndarray) -> IndividualCharacterQuality:
        """
        個別キャラクターの品質評価
        
        Args:
            character_image: キャラクター画像 (H, W, C) numpy配列
            
        Returns:
            IndividualCharacterQuality: 品質評価結果
        """
        try:
            issues = []
            recommendations = []
            
            # 1. 完全性スコア（輪郭の完全性）
            completeness_score = self._assess_completeness(character_image, issues, recommendations)
            
            # 2. 明瞭性スコア（エッジとコントラスト）
            clarity_score = self._assess_clarity(character_image, issues, recommendations)
            
            # 3. サイズ適正性スコア
            size_adequacy_score = self._assess_size_adequacy(character_image, issues, recommendations)
            
            # 4. 形状品質スコア
            shape_quality_score = self._assess_shape_quality(character_image, issues, recommendations)
            
            # 5. 詳細保存スコア
            detail_preservation_score = self._assess_detail_preservation(character_image, issues, recommendations)
            
            # 総合スコア計算（重み付き平均）
            overall_quality_score = (
                completeness_score * 0.25 +
                clarity_score * 0.20 +
                size_adequacy_score * 0.20 +
                shape_quality_score * 0.20 +
                detail_preservation_score * 0.15
            )
            
            return IndividualCharacterQuality(
                overall_quality_score=overall_quality_score,
                completeness_score=completeness_score,
                clarity_score=clarity_score,
                size_adequacy_score=size_adequacy_score,
                shape_quality_score=shape_quality_score,
                detail_preservation_score=detail_preservation_score,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            return IndividualCharacterQuality(
                overall_quality_score=0.0,
                completeness_score=0.0,
                clarity_score=0.0,
                size_adequacy_score=0.0,
                shape_quality_score=0.0,
                detail_preservation_score=0.0,
                issues=[f"Assessment error: {str(e)}"],
                recommendations=["Retry with different parameters"]
            )
    
    def assess_multi_character_quality(self, scene_image: np.ndarray) -> MultiCharacterQuality:
        """
        複数キャラクターシーンの品質評価
        
        Args:
            scene_image: シーン画像 (H, W, C) numpy配列
            
        Returns:
            MultiCharacterQuality: シーン品質評価結果
        """
        try:
            # キャラクター検出とセグメンテーション
            from ..detectors.multi_character_detector import MultiCharacterDetector
            from ..detectors.character_separator import CharacterSeparator
            
            detector = MultiCharacterDetector()
            separator = CharacterSeparator()
            
            # 検出と分離
            detection_result = detector.detect_characters(scene_image)
            separation_result = separator.separate_characters(scene_image)
            
            character_count = detection_result.character_count
            scene_issues = []
            scene_recommendations = []
            
            # 個別キャラクター評価
            individual_scores = []
            if separation_result.character_images:
                for char_image in separation_result.character_images:
                    char_quality = self.assess_individual_character_quality(char_image)
                    individual_scores.append(char_quality)
            
            # シーン全体の評価
            scene_balance_score = self._assess_scene_balance(detection_result, scene_issues, scene_recommendations)
            character_separation_quality = separation_result.separation_confidence
            
            # キャラクター相互作用の分析
            proximity_analysis, interaction_score, has_interaction = self._analyze_character_interactions(
                detection_result, scene_issues, scene_recommendations
            )
            
            # 総合シーン品質
            if individual_scores:
                avg_individual_quality = np.mean([s.overall_quality_score for s in individual_scores])
            else:
                avg_individual_quality = 0.0
            
            overall_scene_quality = (
                scene_balance_score * 0.3 +
                character_separation_quality * 0.3 +
                interaction_score * 0.2 +
                avg_individual_quality * 0.2
            )
            
            return MultiCharacterQuality(
                character_count=character_count,
                scene_balance_score=scene_balance_score,
                character_separation_quality=character_separation_quality,
                character_interaction_score=interaction_score,
                individual_character_scores=individual_scores,
                proximity_analysis=proximity_analysis,
                has_character_interaction=has_interaction,
                overall_scene_quality=overall_scene_quality,
                scene_issues=scene_issues,
                scene_recommendations=scene_recommendations
            )
            
        except Exception as e:
            return MultiCharacterQuality(
                character_count=0,
                scene_balance_score=0.0,
                character_separation_quality=0.0,
                character_interaction_score=0.0,
                individual_character_scores=[],
                proximity_analysis={},
                has_character_interaction=False,
                overall_scene_quality=0.0,
                scene_issues=[f"Assessment error: {str(e)}"],
                scene_recommendations=["Check input image format and retry"]
            )
    
    def _assess_completeness(self, image: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """完全性の評価（輪郭の完全性）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 輪郭検出
        contours, _ = cv2.findContours((gray > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            issues.append("No character contour detected")
            recommendations.append("Check image brightness and contrast")
            return 0.0
        
        # 最大輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area < self.min_character_area:
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
        
        return min(1.0, max(0.0, completeness_score))
    
    def _assess_clarity(self, image: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """明瞭性の評価（エッジとコントラスト）"""
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
        if 0.05 <= edge_density <= 0.3:  # 適切なエッジ密度
            clarity_score += 0.4
        elif edge_density < 0.05:
            clarity_score += 0.2
            issues.append("Low edge density - character may be blurry")
            recommendations.append("Improve image sharpness")
        else:
            clarity_score += 0.3
            issues.append("High edge density - may be noisy")
        
        # コントラスト評価
        if contrast > 30:
            clarity_score += 0.3
        elif contrast > 15:
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
        
        return min(1.0, max(0.0, clarity_score))
    
    def _assess_size_adequacy(self, image: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """サイズ適正性の評価"""
        total_area = image.shape[0] * image.shape[1]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        character_pixels = np.sum(gray > 10)
        character_ratio = character_pixels / total_area
        
        # 理想的な比率との比較
        ratio_diff = abs(character_ratio - self.ideal_character_ratio)
        
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
        
        return size_score
    
    def _assess_shape_quality(self, image: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """形状品質の評価"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        contours, _ = cv2.findContours((gray > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 凸包との比較（凹凸の評価）
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(largest_contour)
        
        if hull_area > 0:
            solidity = contour_area / hull_area
            # 人型キャラクターは適度な凹凸を持つ
            if 0.6 <= solidity <= 0.9:
                shape_score = 0.9
            elif 0.4 <= solidity < 0.6:
                shape_score = 0.7
                issues.append("Character shape has many concave regions")
            else:
                shape_score = 0.5
                if solidity > 0.9:
                    issues.append("Character shape too convex")
                else:
                    issues.append("Character shape too fragmented")
        else:
            shape_score = 0.0
        
        return shape_score
    
    def _assess_detail_preservation(self, image: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """詳細保存性の評価"""
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
        
        # テクスチャ評価
        if texture_variance > 100:
            detail_score += 0.5
        elif texture_variance > 50:
            detail_score += 0.3
        else:
            detail_score += 0.1
            issues.append("Low texture detail")
            recommendations.append("Improve image resolution or reduce smoothing")
        
        # エントロピー評価
        normalized_entropy = hist_entropy / 8.0  # log2(256)で正規化
        detail_score += min(0.5, normalized_entropy)
        
        return min(1.0, detail_score)
    
    def _assess_scene_balance(self, detection_result, issues: List[str], recommendations: List[str]) -> float:
        """シーンバランスの評価"""
        if detection_result.character_count <= 1:
            return 1.0
        
        regions = detection_result.character_regions
        
        # サイズバランスの評価
        areas = [r.area for r in regions]
        area_variance = np.var(areas) / (np.mean(areas) ** 2) if areas else 0
        
        size_balance_score = max(0.0, 1.0 - area_variance)
        
        # 配置バランスの評価
        positions = [(r.center_x, r.center_y) for r in regions]
        if len(positions) >= 2:
            # 重心からの分散
            center_x = np.mean([p[0] for p in positions])
            center_y = np.mean([p[1] for p in positions])
            
            distances = [math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions]
            position_balance_score = 1.0 - min(1.0, np.std(distances) / np.mean(distances))
        else:
            position_balance_score = 1.0
        
        overall_balance = (size_balance_score + position_balance_score) / 2
        
        if overall_balance < 0.6:
            issues.append("Poor scene balance")
            recommendations.append("Improve character positioning and size distribution")
        
        return overall_balance
    
    def _analyze_character_interactions(self, detection_result, issues: List[str], recommendations: List[str]):
        """キャラクター相互作用の分析"""
        if detection_result.character_count < 2:
            return {}, 0.0, False
        
        positions = [(r.center_x, r.center_y) for r in detection_result.character_regions]
        distances = []
        
        # 全ペア間距離の計算
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = math.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2 +
                    (positions[i][1] - positions[j][1]) ** 2
                )
                distances.append(dist)
        
        proximity_analysis = {
            'average_distance': np.mean(distances) if distances else 0.0,
            'min_distance': np.min(distances) if distances else 0.0,
            'max_distance': np.max(distances) if distances else 0.0,
            'distance_variance': np.var(distances) if distances else 0.0
        }
        
        # 相互作用の判定
        min_distance = proximity_analysis['min_distance']
        has_interaction = min_distance < self.interaction_distance_threshold
        
        # 相互作用スコア
        if has_interaction:
            # 近すぎず遠すぎない距離が理想的
            ideal_distance = self.interaction_distance_threshold * 0.7
            distance_score = 1.0 - abs(min_distance - ideal_distance) / ideal_distance
            interaction_score = max(0.0, min(1.0, distance_score))
        else:
            interaction_score = 0.5  # 独立配置も一つの価値
        
        if min_distance < 20:
            issues.append("Characters too close - may overlap")
            recommendations.append("Increase character separation")
        elif proximity_analysis['average_distance'] > 300:
            recommendations.append("Consider bringing characters closer for better composition")
        
        return proximity_analysis, interaction_score, has_interaction