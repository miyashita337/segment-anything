#!/usr/bin/env python3
"""
Enhanced Full Body Detector
改良版全身検出システム - 多指標統合による高精度全身判定

Phase 1 P1-003: 全身判定基準の改善
P1-001分析結果とP1-002部分抽出検出を統合した改良システム
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

# 部分抽出検出システムとの統合
from .partial_extraction_detector import PartialExtractionDetector, ExtractionAnalysis


@dataclass
class FullBodyScore:
    """全身判定スコアの詳細情報"""
    total_score: float  # 総合スコア (0.0-1.0)
    aspect_ratio_score: float  # アスペクト比スコア
    body_structure_score: float  # 人体構造スコア
    edge_distribution_score: float  # エッジ分布スコア
    semantic_region_score: float  # セマンティック領域スコア
    completeness_bonus: float  # 完全性ボーナス
    confidence: float  # 判定信頼度
    reasoning: str  # 判定理由


@dataclass
class BodyStructureAnalysis:
    """人体構造分析結果"""
    face_regions: List[Tuple[int, int, int, int]]  # 顔領域 [x, y, w, h]
    torso_density: float  # 胴体密度
    limb_density: float  # 手足密度
    vertical_distribution: float  # 縦方向分布
    body_proportion: float  # 身体比率
    structure_completeness: float  # 構造完全性


class EnhancedFullBodyDetector:
    """改良版全身検出システム"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self.partial_detector = PartialExtractionDetector()
        
        # 重み設定（学習可能）
        self.weights = {
            'aspect_ratio': 0.25,      # アスペクト比（基本指標）
            'body_structure': 0.35,    # 人体構造（最重要）
            'edge_distribution': 0.20,  # エッジ分布
            'semantic_regions': 0.15,   # セマンティック領域
            'completeness_bonus': 0.05  # 完全性ボーナス
        }
        
        # 動的閾値設定
        self.thresholds = {
            'good_fullbody': 0.75,      # 高品質全身
            'acceptable_fullbody': 0.55, # 許容可能全身
            'partial_extraction': 0.35,  # 部分抽出
            'face_only': 0.25           # 顔のみ
        }
        
        # アスペクト比の動的範囲（画像サイズ・スタイル適応）
        self.aspect_ratio_ranges = {
            'ideal': (1.4, 2.2),       # 理想的な全身
            'acceptable': (1.2, 2.8),   # 許容可能
            'extended': (1.0, 3.5)      # 拡張範囲
        }
    
    def evaluate_fullbody_score(self, image: np.ndarray, mask_data: Dict[str, Any]) -> FullBodyScore:
        """
        多指標による全身スコア評価
        
        Args:
            image: 元画像 (BGR)
            mask_data: マスクデータ（bbox, area, マスク等を含む）
            
        Returns:
            FullBodyScore: 詳細な全身判定結果
        """
        h, w = image.shape[:2]
        
        # 1. アスペクト比スコア（改良版）
        aspect_score = self._calculate_aspect_ratio_score(mask_data, h, w)
        
        # 2. 人体構造スコア
        structure_score = self._calculate_body_structure_score(image, mask_data)
        
        # 3. エッジ分布スコア
        edge_score = self._calculate_edge_distribution_score(mask_data, h, w)
        
        # 4. セマンティック領域スコア
        semantic_score = self._calculate_semantic_region_score(image, mask_data)
        
        # 5. 完全性ボーナス（P1-002統合）
        completeness_bonus = self._calculate_completeness_bonus(image, mask_data)
        
        # 重み付き総合スコア計算
        total_score = (
            aspect_score * self.weights['aspect_ratio'] +
            structure_score * self.weights['body_structure'] +
            edge_score * self.weights['edge_distribution'] +
            semantic_score * self.weights['semantic_regions'] +
            completeness_bonus * self.weights['completeness_bonus']
        )
        
        # 信頼度計算（各スコアの分散を考慮）
        scores = [aspect_score, structure_score, edge_score, semantic_score]
        confidence = 1.0 - (np.std(scores) / max(np.mean(scores), 0.1))
        confidence = max(0.0, min(1.0, confidence))
        
        # 判定理由の生成
        reasoning = self._generate_reasoning(
            total_score, aspect_score, structure_score, 
            edge_score, semantic_score, completeness_bonus
        )
        
        return FullBodyScore(
            total_score=total_score,
            aspect_ratio_score=aspect_score,
            body_structure_score=structure_score,
            edge_distribution_score=edge_score,
            semantic_region_score=semantic_score,
            completeness_bonus=completeness_bonus,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _calculate_aspect_ratio_score(self, mask_data: Dict[str, Any], h: int, w: int) -> float:
        """改良版アスペクト比スコア計算"""
        bbox = mask_data.get('bbox', [0, 0, w, h])
        mask_height, mask_width = bbox[3], bbox[2]
        
        if mask_width <= 0:
            return 0.0
        
        aspect_ratio = mask_height / mask_width
        
        # 動的範囲による段階的評価
        ideal_min, ideal_max = self.aspect_ratio_ranges['ideal']
        acceptable_min, acceptable_max = self.aspect_ratio_ranges['acceptable']
        
        if ideal_min <= aspect_ratio <= ideal_max:
            # 理想範囲内：高スコア
            center = (ideal_min + ideal_max) / 2
            deviation = abs(aspect_ratio - center) / ((ideal_max - ideal_min) / 2)
            score = 1.0 - (deviation * 0.2)  # 最大20%減点
        elif acceptable_min <= aspect_ratio <= acceptable_max:
            # 許容範囲内：中程度スコア
            if aspect_ratio < ideal_min:
                score = 0.6 + 0.2 * (aspect_ratio - acceptable_min) / (ideal_min - acceptable_min)
            else:
                score = 0.6 + 0.2 * (acceptable_max - aspect_ratio) / (acceptable_max - ideal_max)
        else:
            # 範囲外：低スコア（ただし完全に0にはしない）
            extended_min, extended_max = self.aspect_ratio_ranges['extended']
            if extended_min <= aspect_ratio <= extended_max:
                score = 0.3
            else:
                score = 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_body_structure_score(self, image: np.ndarray, mask_data: Dict[str, Any]) -> float:
        """人体構造スコア計算"""
        try:
            # マスクの取得
            if 'mask' in mask_data:
                mask = mask_data['mask']
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
            else:
                # bboxからマスク生成
                bbox = mask_data.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                # 座標を整数に変換（slice操作のため）
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                # 境界チェック
                x = max(0, x)
                y = max(0, y)
                x_end = min(x + w, image.shape[1])
                y_end = min(y + h, image.shape[0])
                mask[y:y_end, x:x_end] = 255
            
            # 人体構造分析
            structure_analysis = self._analyze_body_structure(image, mask)
            
            # 構造スコア計算
            structure_score = 0.0
            
            # 顔の存在（20%）
            if len(structure_analysis.face_regions) > 0:
                structure_score += 0.20
            
            # 胴体密度（30%）
            structure_score += structure_analysis.torso_density * 0.30
            
            # 手足密度（25%）
            structure_score += structure_analysis.limb_density * 0.25
            
            # 縦方向分布（15%）
            structure_score += structure_analysis.vertical_distribution * 0.15
            
            # 構造完全性（10%）
            structure_score += structure_analysis.structure_completeness * 0.10
            
            return max(0.0, min(1.0, structure_score))
            
        except Exception as e:
            self.logger.warning(f"Body structure analysis failed: {e}")
            return 0.3  # フォールバック値
    
    def _analyze_body_structure(self, image: np.ndarray, mask: np.ndarray) -> BodyStructureAnalysis:
        """詳細な人体構造分析"""
        h, w = mask.shape
        
        # 顔領域検出
        face_regions = self._detect_face_regions_in_mask(image, mask)
        
        # 領域分割（上部：顔、中央：胴体、下部：足、両端：手）
        upper_region = mask[:int(h*0.4), :]
        middle_region = mask[int(h*0.2):int(h*0.8), :]
        lower_region = mask[int(h*0.6):, :]
        
        # 胴体密度（中央部の密度）
        torso_area = middle_region[int(h*0.1):int(h*0.5), int(w*0.25):int(w*0.75)]
        torso_density = np.sum(torso_area > 0) / max(torso_area.size, 1) if torso_area.size > 0 else 0
        
        # 手足密度（外周部の密度）
        left_edge = mask[:, :int(w*0.2)]
        right_edge = mask[:, int(w*0.8):]
        bottom_edge = lower_region[int(h*0.1):, :]
        
        limb_densities = []
        for edge in [left_edge, right_edge, bottom_edge]:
            if edge.size > 0:
                density = np.sum(edge > 0) / edge.size
                limb_densities.append(density)
        
        limb_density = np.mean(limb_densities) if limb_densities else 0
        
        # 縦方向分布（上下にバランスよく分布しているか）
        vertical_thirds = np.array([
            np.sum(mask[:int(h/3), :] > 0),
            np.sum(mask[int(h/3):int(2*h/3), :] > 0),
            np.sum(mask[int(2*h/3):, :] > 0)
        ])
        
        if np.sum(vertical_thirds) > 0:
            vertical_distribution = 1.0 - (np.std(vertical_thirds) / np.mean(vertical_thirds))
            vertical_distribution = max(0.0, vertical_distribution)
        else:
            vertical_distribution = 0.0
        
        # 身体比率（適切な比率かどうか）
        mask_area = np.sum(mask > 0)
        total_area = h * w
        body_proportion = mask_area / max(total_area, 1)
        
        # 適切な比率範囲（5-40%）での評価
        if 0.05 <= body_proportion <= 0.4:
            proportion_score = 1.0
        else:
            proportion_score = max(0.0, 1.0 - abs(body_proportion - 0.2) / 0.2)
        
        # 構造完全性（各要素の組み合わせ）
        structure_completeness = np.mean([
            min(len(face_regions), 1),  # 顔の存在
            min(torso_density * 2, 1),  # 胴体の存在
            min(limb_density * 3, 1),   # 手足の存在
            proportion_score             # 適切な比率
        ])
        
        return BodyStructureAnalysis(
            face_regions=face_regions,
            torso_density=torso_density,
            limb_density=limb_density,
            vertical_distribution=vertical_distribution,
            body_proportion=body_proportion,
            structure_completeness=structure_completeness
        )
    
    def _detect_face_regions_in_mask(self, image: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """マスク領域内での顔検出"""
        try:
            # マスク領域のみを抽出
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            # 顔検出（OpenCV Cascade）
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(20, 20),
                maxSize=(int(image.shape[0]*0.5), int(image.shape[1]*0.5))
            )
            
            return [(x, y, w, h) for x, y, w, h in faces]
            
        except Exception as e:
            self.logger.warning(f"Face detection in mask failed: {e}")
            return []
    
    def _calculate_edge_distribution_score(self, mask_data: Dict[str, Any], h: int, w: int) -> float:
        """エッジ分布スコア計算"""
        try:
            # マスクからエッジ分布を分析
            if 'mask' in mask_data:
                mask = mask_data['mask']
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
            else:
                bbox = mask_data.get('bbox', [0, 0, w, h])
                mask = np.zeros((h, w), dtype=np.uint8)
                # 座標を整数に変換
                x = int(bbox[0])
                y = int(bbox[1])
                mask_w = int(bbox[2])
                mask_h = int(bbox[3])
                # 境界チェック
                x = max(0, x)
                y = max(0, y)
                x_end = min(x + mask_w, mask.shape[1])
                y_end = min(y + mask_h, mask.shape[0])
                mask[y:y_end, x:x_end] = 255
            
            # エッジ検出
            edges = cv2.Canny(mask, 50, 150)
            
            # 領域別エッジ分布
            regions = {
                'top': edges[:int(h*0.3), :],
                'middle': edges[int(h*0.3):int(h*0.7), :],
                'bottom': edges[int(h*0.7):, :],
                'left': edges[:, :int(w*0.3)],
                'right': edges[:, int(w*0.7):]
            }
            
            # 各領域のエッジ密度
            edge_densities = {}
            for region_name, region in regions.items():
                if region.size > 0:
                    edge_densities[region_name] = np.sum(region > 0) / region.size
                else:
                    edge_densities[region_name] = 0
            
            # 全身らしいエッジ分布の評価
            # 上部（頭）、中央（胴体）、下部（足）にバランスよくエッジがあるか
            vertical_balance = 1.0 - abs(edge_densities['top'] - edge_densities['bottom']) / max(edge_densities['top'] + edge_densities['bottom'], 0.01)
            
            # 左右のバランス
            horizontal_balance = 1.0 - abs(edge_densities['left'] - edge_densities['right']) / max(edge_densities['left'] + edge_densities['right'], 0.01)
            
            # 中央部の安定性（胴体）
            central_stability = edge_densities['middle']
            
            # 総合エッジスコア
            edge_score = np.mean([vertical_balance, horizontal_balance, central_stability])
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            self.logger.warning(f"Edge distribution analysis failed: {e}")
            return 0.5  # フォールバック値
    
    def _calculate_semantic_region_score(self, image: np.ndarray, mask_data: Dict[str, Any]) -> float:
        """セマンティック領域スコア計算"""
        try:
            # マスク領域の色分布分析
            if 'mask' in mask_data:
                mask = mask_data['mask']
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
            else:
                bbox = mask_data.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                # 座標を整数に変換
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                # 境界チェック
                x = max(0, x)
                y = max(0, y)
                x_end = min(x + w, image.shape[1])
                y_end = min(y + h, image.shape[0])
                mask[y:y_end, x:x_end] = 255
            
            # マスク領域の色抽出
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            masked_pixels = masked_image[mask > 0]
            
            if len(masked_pixels) == 0:
                return 0.0
            
            # 色分布の分析
            # 肌色っぽい領域の検出
            skin_score = self._detect_skin_regions(masked_pixels)
            
            # 色の多様性（衣服等）
            color_diversity = self._calculate_color_diversity(masked_pixels)
            
            # 明暗のバランス
            brightness_balance = self._calculate_brightness_balance(masked_pixels)
            
            # セマンティックスコア
            semantic_score = np.mean([skin_score, color_diversity, brightness_balance])
            
            return max(0.0, min(1.0, semantic_score))
            
        except Exception as e:
            self.logger.warning(f"Semantic region analysis failed: {e}")
            return 0.4  # フォールバック値
    
    def _detect_skin_regions(self, pixels: np.ndarray) -> float:
        """肌色領域の検出"""
        if len(pixels) == 0:
            return 0.0
        
        # HSV変換
        bgr_pixels = pixels.reshape(-1, 1, 3)
        hsv_pixels = cv2.cvtColor(bgr_pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        # 肌色範囲（HSV）
        skin_mask = (
            (hsv_pixels[:, 0] >= 0) & (hsv_pixels[:, 0] <= 25) |  # 赤系
            (hsv_pixels[:, 0] >= 160) & (hsv_pixels[:, 0] <= 180)  # 赤系
        ) & (
            (hsv_pixels[:, 1] >= 30) & (hsv_pixels[:, 1] <= 255)  # 彩度
        ) & (
            (hsv_pixels[:, 2] >= 60) & (hsv_pixels[:, 2] <= 255)  # 明度
        )
        
        skin_ratio = np.sum(skin_mask) / len(pixels)
        
        # 適切な肌色比率（10-40%）
        if 0.1 <= skin_ratio <= 0.4:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(skin_ratio - 0.25) / 0.25)
    
    def _calculate_color_diversity(self, pixels: np.ndarray) -> float:
        """色の多様性計算"""
        if len(pixels) == 0:
            return 0.0
        
        # 色空間でのクラスタリング（簡易版）
        unique_colors = len(np.unique(pixels.reshape(-1, 3), axis=0))
        total_pixels = len(pixels)
        
        diversity_ratio = unique_colors / max(total_pixels, 1)
        
        # 適度な多様性（単色過ぎず、複雑過ぎず）
        if 0.1 <= diversity_ratio <= 0.5:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(diversity_ratio - 0.3) / 0.3)
    
    def _calculate_brightness_balance(self, pixels: np.ndarray) -> float:
        """明暗バランスの計算"""
        if len(pixels) == 0:
            return 0.0
        
        # グレースケール変換
        gray_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
        
        # 明暗の分布
        dark_ratio = np.sum(gray_pixels < 85) / len(gray_pixels)    # 暗部
        mid_ratio = np.sum((gray_pixels >= 85) & (gray_pixels < 170)) / len(gray_pixels)  # 中間
        bright_ratio = np.sum(gray_pixels >= 170) / len(gray_pixels)  # 明部
        
        # バランスの良い分布
        balance_score = 1.0 - np.std([dark_ratio, mid_ratio, bright_ratio])
        
        return max(0.0, min(1.0, balance_score))
    
    def _calculate_completeness_bonus(self, image: np.ndarray, mask_data: Dict[str, Any]) -> float:
        """完全性ボーナス計算（P1-002統合）"""
        try:
            # マスクの取得
            if 'mask' in mask_data:
                mask = mask_data['mask']
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
            else:
                bbox = mask_data.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                # 座標を整数に変換
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                # 境界チェック
                x = max(0, x)
                y = max(0, y)
                x_end = min(x + w, image.shape[1])
                y_end = min(y + h, image.shape[0])
                mask[y:y_end, x:x_end] = 255
            
            # 部分抽出検出システムによる分析
            extraction_analysis = self.partial_detector.analyze_extraction(image, mask)
            
            # 完全性スコアをボーナスとして活用
            completeness_bonus = extraction_analysis.completeness_score
            
            # 高品質な場合は追加ボーナス
            if extraction_analysis.quality_assessment == 'good':
                completeness_bonus = min(1.0, completeness_bonus + 0.1)
            
            return completeness_bonus
            
        except Exception as e:
            self.logger.warning(f"Completeness bonus calculation failed: {e}")
            return 0.5  # フォールバック値
    
    def _generate_reasoning(self, total_score: float, aspect_score: float, 
                          structure_score: float, edge_score: float, 
                          semantic_score: float, completeness_bonus: float) -> str:
        """判定理由の生成"""
        reasoning_parts = []
        
        # 総合評価
        if total_score >= self.thresholds['good_fullbody']:
            reasoning_parts.append("高品質全身")
        elif total_score >= self.thresholds['acceptable_fullbody']:
            reasoning_parts.append("許容可能全身")
        elif total_score >= self.thresholds['partial_extraction']:
            reasoning_parts.append("部分抽出")
        else:
            reasoning_parts.append("不完全抽出")
        
        # 主要な要因
        scores = {
            'アスペクト比': aspect_score,
            '人体構造': structure_score,
            'エッジ分布': edge_score,
            'セマンティック': semantic_score,
            '完全性': completeness_bonus
        }
        
        # 最高・最低スコア
        max_factor = max(scores.keys(), key=lambda k: scores[k])
        min_factor = min(scores.keys(), key=lambda k: scores[k])
        
        reasoning_parts.append(f"強み: {max_factor}({scores[max_factor]:.2f})")
        if scores[min_factor] < 0.3:
            reasoning_parts.append(f"弱点: {min_factor}({scores[min_factor]:.2f})")
        
        return "; ".join(reasoning_parts)
    
    def classify_extraction_quality(self, fullbody_score: FullBodyScore) -> str:
        """抽出品質の分類"""
        score = fullbody_score.total_score
        
        if score >= self.thresholds['good_fullbody']:
            return 'excellent_fullbody'
        elif score >= self.thresholds['acceptable_fullbody']:
            return 'good_fullbody'
        elif score >= self.thresholds['partial_extraction']:
            return 'partial_extraction'
        else:
            return 'poor_extraction'
    
    def suggest_improvements(self, fullbody_score: FullBodyScore) -> List[str]:
        """改善提案の生成"""
        suggestions = []
        
        # 各スコアに基づく具体的提案
        if fullbody_score.aspect_ratio_score < 0.5:
            suggestions.append("アスペクト比改善: より縦長の範囲での抽出を試行")
        
        if fullbody_score.body_structure_score < 0.4:
            suggestions.append("人体構造改善: 顔・胴体・手足が含まれる範囲での再抽出")
        
        if fullbody_score.edge_distribution_score < 0.4:
            suggestions.append("境界線改善: マスク拡張やエッジ平滑化処理")
        
        if fullbody_score.semantic_region_score < 0.4:
            suggestions.append("領域改善: コントラスト調整や色空間変換")
        
        if fullbody_score.completeness_bonus < 0.5:
            suggestions.append("完全性改善: 手動ポイント指定や前処理強化")
        
        # 信頼度に基づく提案
        if fullbody_score.confidence < 0.6:
            suggestions.append("判定信頼度向上: 複数手法での検証")
        
        return suggestions


def evaluate_fullbody_enhanced(image: np.ndarray, mask_data: Dict[str, Any]) -> FullBodyScore:
    """
    改良版全身判定の便利関数
    
    Args:
        image: 元画像 (BGR)
        mask_data: マスクデータ
        
    Returns:
        FullBodyScore: 詳細な全身判定結果
    """
    detector = EnhancedFullBodyDetector()
    return detector.evaluate_fullbody_score(image, mask_data)