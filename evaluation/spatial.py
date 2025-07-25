#!/usr/bin/env python3
"""
空間評価器 (IoU評価)
GPT-4O設計による改良版IoU計算システム
"""

import numpy as np
import cv2

from typing import Optional, Tuple, Union

from .base import EvaluationResult, EvaluatorBase


class IoUEvaluator(EvaluatorBase):
    """IoU (Intersection over Union) 評価器"""
    
    def __init__(self, device: str = "cuda", threshold: float = 0.3):
        super().__init__(device)
        self.threshold = threshold
        
    def calculate_bbox_iou(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """境界ボックスのIoU計算 (x, y, w, h形式)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 交差領域の計算
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        return intersection_area / max(union_area, 1e-6)
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """マスクのIoU計算"""
        if mask1.shape != mask2.shape:
            # サイズが異なる場合はリサイズ
            h, w = max(mask1.shape[0], mask2.shape[0]), max(mask1.shape[1], mask2.shape[1])
            if mask1.shape != (h, w):
                mask1 = cv2.resize(mask1.astype(np.uint8), (w, h))
            if mask2.shape != (h, w):
                mask2 = cv2.resize(mask2.astype(np.uint8), (w, h))
        
        # バイナリマスクに変換
        mask1_binary = (mask1 > 0).astype(np.uint8)
        mask2_binary = (mask2 > 0).astype(np.uint8)
        
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        union = np.logical_or(mask1_binary, mask2_binary).sum()
        
        return intersection / max(union, 1e-6)
    
    def calculate_center_distance(self, bbox1: Tuple[int, int, int, int],
                                bbox2: Tuple[int, int, int, int]) -> float:
        """中心点間の正規化距離"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        diagonal = np.sqrt(max(w1, w2)**2 + max(h1, h2)**2)
        
        return 1.0 - min(distance / max(diagonal, 1e-6), 1.0)
    
    def calculate_area_ratio(self, bbox1: Tuple[int, int, int, int],
                           bbox2: Tuple[int, int, int, int]) -> float:
        """面積比の類似度 (0-1)"""
        _, _, w1, h1 = bbox1
        _, _, w2, h2 = bbox2
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def evaluate(self, prediction: Union[Tuple, np.ndarray], 
                ground_truth: Union[Tuple, np.ndarray]) -> EvaluationResult:
        """空間評価の実行"""
        if isinstance(prediction, tuple) and isinstance(ground_truth, tuple):
            # 境界ボックス評価
            iou_score = self.calculate_bbox_iou(prediction, ground_truth)
            center_sim = self.calculate_center_distance(prediction, ground_truth)
            area_sim = self.calculate_area_ratio(prediction, ground_truth)
            
            # 統合空間スコア (IoUを重視、補助的に中心・面積類似度を使用)
            spatial_score = 0.7 * iou_score + 0.2 * center_sim + 0.1 * area_sim
            
        elif isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
            # マスク評価
            iou_score = self.calculate_mask_iou(prediction, ground_truth)
            spatial_score = iou_score
            center_sim = area_sim = 0.0
            
        else:
            raise ValueError("prediction と ground_truth は同じ型である必要があります")
        
        success = spatial_score >= self.threshold
        confidence = min(spatial_score * 2, 1.0)  # IoUベースの信頼度
        
        return EvaluationResult(
            success=success,
            spatial_score=spatial_score,
            content_score=0.0,  # 空間評価では内容スコアは0
            integrated_score=spatial_score,  # 空間評価のみの場合
            confidence=confidence,
            metadata={
                'iou': iou_score,
                'center_similarity': center_sim,
                'area_similarity': area_sim,
                'evaluation_type': 'spatial_only'
            }
        )