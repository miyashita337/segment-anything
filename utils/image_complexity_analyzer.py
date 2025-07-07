#!/usr/bin/env python3
"""
Phase 4.1: 画像複雑度分析器
入力画像の複雑度を分析し、最適な処理エンジンを選択するための判定を行う
"""

import cv2
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

class ComplexityLevel(Enum):
    """複雑度レベル"""
    SIMPLE = "simple"      # シンプル：Phase 0.0.3で十分
    COMPLEX = "complex"    # 複雑：Phase 0.0.4が必要
    UNKNOWN = "unknown"    # 不明：両方実行して選択

@dataclass
class ComplexityAnalysis:
    """複雑度分析結果"""
    level: ComplexityLevel
    yolo_detections: int
    overlap_ratio: float
    central_character_ratio: float
    size_variance: float
    confidence_scores: List[float]
    analysis_details: Dict[str, Any]
    reasoning: str

class ImageComplexityAnalyzer:
    """画像複雑度分析器"""
    
    def __init__(self,
                 simple_threshold: float = 0.8,
                 complex_threshold: float = 0.3):
        """
        初期化
        
        Args:
            simple_threshold: シンプル判定の閾値（高いほど厳格）
            complex_threshold: 複雑判定の閾値（低いほど厳格）
        """
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold
        self.logger = logging.getLogger(__name__)
    
    def analyze_complexity(self, 
                          image: np.ndarray,
                          yolo_results: List[Dict[str, Any]]) -> ComplexityAnalysis:
        """
        画像複雑度を分析
        
        Args:
            image: 入力画像
            yolo_results: YOLO検出結果のリスト
                         [{"bbox": (x1, y1, x2, y2), "confidence": float}]
        
        Returns:
            ComplexityAnalysis: 分析結果
        """
        try:
            # 基本情報
            detections_count = len(yolo_results)
            image_height, image_width = image.shape[:2]
            
            # 1. 検出数による基本判定
            if detections_count == 0:
                return self._create_analysis(
                    ComplexityLevel.COMPLEX,
                    yolo_results,
                    reasoning="キャラクター検出なし - Phase 0.0.4の低閾値処理が必要"
                )
            
            if detections_count == 1:
                # 単一キャラクター：面積とレイアウトをチェック
                bbox = yolo_results[0]["bbox"]
                confidence = yolo_results[0]["confidence"]
                
                area_ratio = self._calculate_area_ratio(bbox, image_width, image_height)
                is_central = self._is_central_positioned(bbox, image_width, image_height)
                
                if confidence > 0.7 and area_ratio > 0.1 and is_central:
                    return self._create_analysis(
                        ComplexityLevel.SIMPLE,
                        yolo_results,
                        reasoning="単一キャラクター、高信頼度、適切な配置 - Phase 0.0.3で処理可能"
                    )
                else:
                    return self._create_analysis(
                        ComplexityLevel.UNKNOWN,
                        yolo_results,
                        reasoning="単一キャラクターだが信頼度または配置に問題 - 両方試行"
                    )
            
            # 2. 複数キャラクター：詳細分析
            return self._analyze_multi_character(image, yolo_results)
            
        except Exception as e:
            self.logger.error(f"複雑度分析エラー: {e}")
            return self._create_analysis(
                ComplexityLevel.UNKNOWN,
                yolo_results,
                reasoning=f"分析エラー: {e} - 安全のため両方試行"
            )
    
    def _analyze_multi_character(self, 
                                image: np.ndarray,
                                yolo_results: List[Dict[str, Any]]) -> ComplexityAnalysis:
        """複数キャラクターの詳細分析"""
        image_height, image_width = image.shape[:2]
        
        # バウンディングボックス分析
        bboxes = [result["bbox"] for result in yolo_results]
        confidences = [result["confidence"] for result in yolo_results]
        
        # 重なり度分析
        overlap_ratio = self._calculate_overlap_ratio(bboxes)
        
        # サイズ分散分析
        areas = [self._calculate_area_ratio(bbox, image_width, image_height) for bbox in bboxes]
        size_variance = np.var(areas) if len(areas) > 1 else 0.0
        
        # 中央配置度分析
        central_scores = [self._calculate_central_score(bbox, image_width, image_height) for bbox in bboxes]
        central_character_ratio = max(central_scores) if central_scores else 0.0
        
        # 主キャラクター判定
        primary_character_clear = self._is_primary_character_clear(areas, confidences, central_scores)
        
        # 複雑度判定
        if (primary_character_clear and 
            overlap_ratio < 0.3 and 
            max(confidences) > 0.6):
            level = ComplexityLevel.SIMPLE
            reasoning = "複数キャラクターだが主キャラクター明確 - Phase 0.0.3で処理可能"
        elif (overlap_ratio > 0.7 or 
              size_variance > 0.1 or 
              max(confidences) < 0.4):
            level = ComplexityLevel.COMPLEX
            reasoning = "複数キャラクター、複雑な配置 - Phase 0.0.4の高度処理が必要"
        else:
            level = ComplexityLevel.UNKNOWN
            reasoning = "複数キャラクター、判定困難 - 両方試行して最適選択"
        
        return ComplexityAnalysis(
            level=level,
            yolo_detections=len(yolo_results),
            overlap_ratio=overlap_ratio,
            central_character_ratio=central_character_ratio,
            size_variance=size_variance,
            confidence_scores=confidences,
            analysis_details={
                "areas": areas,
                "central_scores": central_scores,
                "primary_character_clear": primary_character_clear
            },
            reasoning=reasoning
        )
    
    def _calculate_area_ratio(self, bbox: Tuple[int, int, int, int], 
                             image_width: int, image_height: int) -> float:
        """バウンディングボックスの面積比を計算"""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        return bbox_area / image_area
    
    def _is_central_positioned(self, bbox: Tuple[int, int, int, int],
                              image_width: int, image_height: int) -> bool:
        """中央配置かどうかを判定"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 画像中央からの距離
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        distance_x = abs(center_x - image_center_x) / image_width
        distance_y = abs(center_y - image_center_y) / image_height
        
        return distance_x < 0.3 and distance_y < 0.3
    
    def _calculate_central_score(self, bbox: Tuple[int, int, int, int],
                                image_width: int, image_height: int) -> float:
        """中央配置スコアを計算（0-1、1が最も中央）"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        distance_x = abs(center_x - image_center_x) / image_width
        distance_y = abs(center_y - image_center_y) / image_height
        
        # 距離を逆転してスコア化（近いほど高スコア）
        score = 1.0 - np.sqrt(distance_x**2 + distance_y**2)
        return max(0.0, score)
    
    def _calculate_overlap_ratio(self, bboxes: List[Tuple[int, int, int, int]]) -> float:
        """複数バウンディングボックスの重なり率を計算"""
        if len(bboxes) <= 1:
            return 0.0
        
        total_overlap = 0.0
        pairs_count = 0
        
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                overlap = self._calculate_bbox_overlap(bboxes[i], bboxes[j])
                total_overlap += overlap
                pairs_count += 1
        
        return total_overlap / pairs_count if pairs_count > 0 else 0.0
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int],
                               bbox2: Tuple[int, int, int, int]) -> float:
        """2つのバウンディングボックスの重なり率を計算"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 重なり領域の計算
        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
        
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        
        # 各ボックスの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU (Intersection over Union)
        union_area = area1 + area2 - overlap_area
        return overlap_area / union_area if union_area > 0 else 0.0
    
    def _is_primary_character_clear(self, areas: List[float],
                                   confidences: List[float],
                                   central_scores: List[float]) -> bool:
        """主キャラクターが明確かどうかを判定"""
        if len(areas) <= 1:
            return True
        
        # 面積、信頼度、中央配置の重み付きスコア
        scores = []
        for i in range(len(areas)):
            score = (areas[i] * 0.4 + 
                    confidences[i] * 0.4 + 
                    central_scores[i] * 0.2)
            scores.append(score)
        
        # 最高スコアと2番目のスコアの差
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) >= 2:
            score_gap = sorted_scores[0] - sorted_scores[1]
            return score_gap > 0.3  # 30%以上の差があれば明確
        
        return True
    
    def _create_analysis(self,
                        level: ComplexityLevel,
                        yolo_results: List[Dict[str, Any]],
                        reasoning: str = "") -> ComplexityAnalysis:
        """分析結果を作成"""
        confidences = [result["confidence"] for result in yolo_results]
        
        return ComplexityAnalysis(
            level=level,
            yolo_detections=len(yolo_results),
            overlap_ratio=0.0,
            central_character_ratio=0.0,
            size_variance=0.0,
            confidence_scores=confidences,
            analysis_details={},
            reasoning=reasoning
        )