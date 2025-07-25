#!/usr/bin/env python3
"""
領域マッチャー (ハンガリアン法による最適割当)
GPT-4O設計による空間+内容統合マッチングシステム
"""

import numpy as np

import logging
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Tuple

from .base import EvaluationConfig, EvaluationResult

logger = logging.getLogger(__name__)


class RegionMatcher:
    """領域マッチング器"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def create_cost_matrix(self, iou_matrix: np.ndarray, 
                          similarity_matrix: np.ndarray, 
                          alpha: Optional[float] = None) -> np.ndarray:
        """コスト行列の作成"""
        if alpha is None:
            alpha = self.config.alpha
        
        # IoUと内容類似度を統合 (最大化問題なので負値にする)
        integrated_score = alpha * iou_matrix + (1 - alpha) * similarity_matrix
        cost_matrix = -integrated_score  # ハンガリアン法は最小化問題
        
        return cost_matrix
    
    def match(self, iou_matrix: np.ndarray, 
             similarity_matrix: np.ndarray, 
             alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        最適マッチングの実行
        
        Returns:
            row_indices: マッチした予測のインデックス
            col_indices: マッチしたground truthのインデックス  
            scores: マッチした統合スコア
        """
        # 入力検証
        if iou_matrix.shape != similarity_matrix.shape:
            raise ValueError("IoU matrix and similarity matrix must have the same shape")
        
        # コスト行列作成
        cost_matrix = self.create_cost_matrix(iou_matrix, similarity_matrix, alpha)
        
        # ハンガリアン法で最適割当
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 統合スコアの計算
        alpha = alpha if alpha is not None else self.config.alpha
        scores = alpha * iou_matrix[row_indices, col_indices] + \
                (1 - alpha) * similarity_matrix[row_indices, col_indices]
        
        return row_indices, col_indices, scores
    
    def filter_matches(self, row_indices: np.ndarray, col_indices: np.ndarray, 
                      scores: np.ndarray, iou_matrix: np.ndarray,
                      similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """閾値によるマッチングフィルタリング"""
        valid_mask = (
            (iou_matrix[row_indices, col_indices] >= self.config.iou_threshold) &
            (similarity_matrix[row_indices, col_indices] >= self.config.content_threshold)
        )
        
        return row_indices[valid_mask], col_indices[valid_mask], scores[valid_mask]
    
    def evaluate_matches(self, predictions: List[Any], ground_truths: List[Any],
                        iou_matrix: np.ndarray, similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """マッチング評価の実行"""
        n_pred, n_gt = len(predictions), len(ground_truths)
        
        if n_pred == 0 or n_gt == 0:
            return {
                'matches': [],
                'unmatched_predictions': list(range(n_pred)),
                'unmatched_ground_truths': list(range(n_gt)),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mean_score': 0.0
            }
        
        # 最適マッチング
        row_indices, col_indices, scores = self.match(iou_matrix, similarity_matrix)
        
        # 閾値フィルタリング
        valid_rows, valid_cols, valid_scores = self.filter_matches(
            row_indices, col_indices, scores, iou_matrix, similarity_matrix
        )
        
        # マッチング結果
        matches = []
        for i, (pred_idx, gt_idx, score) in enumerate(zip(valid_rows, valid_cols, valid_scores)):
            matches.append({
                'prediction_index': int(pred_idx),
                'ground_truth_index': int(gt_idx),
                'integrated_score': float(score),
                'iou_score': float(iou_matrix[pred_idx, gt_idx]),
                'content_score': float(similarity_matrix[pred_idx, gt_idx])
            })
        
        # 未マッチの要素
        matched_pred_indices = set(valid_rows)
        matched_gt_indices = set(valid_cols)
        
        unmatched_predictions = [i for i in range(n_pred) if i not in matched_pred_indices]
        unmatched_ground_truths = [i for i in range(n_gt) if i not in matched_gt_indices]
        
        # 評価メトリクス
        tp = len(matches)
        fp = len(unmatched_predictions)
        fn = len(unmatched_ground_truths)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
        
        return {
            'matches': matches,
            'unmatched_predictions': unmatched_predictions,
            'unmatched_ground_truths': unmatched_ground_truths,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_score': mean_score,
            'total_predictions': n_pred,
            'total_ground_truths': n_gt,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }


class MultiCharacterMatcher(RegionMatcher):
    """マルチキャラクター対応マッチャー"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
    
    def calculate_primary_score(self, bbox: Tuple[int, int, int, int], 
                              confidence: float, image_shape: Tuple[int, int]) -> float:
        """主キャラクタースコアの計算"""
        x, y, w, h = bbox
        img_h, img_w = image_shape
        
        # 画像中心からの距離
        center_x, center_y = x + w/2, y + h/2
        img_center_x, img_center_y = img_w/2, img_h/2
        distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        center_score = 1.0 - (distance / max_distance)
        
        # 面積スコア
        area = w * h
        img_area = img_w * img_h
        area_score = min(area / img_area, 1.0)
        
        # 統合スコア
        primary_score = (
            self.config.center_weight * center_score +
            self.config.area_weight * area_score +
            self.config.confidence_weight * confidence
        )
        
        return primary_score
    
    def classify_characters(self, bboxes: List[Tuple[int, int, int, int]], 
                          confidences: List[float], 
                          image_shape: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """主・副キャラクターの分類"""
        if not bboxes:
            return [], []
        
        scores = []
        for bbox, conf in zip(bboxes, confidences):
            score = self.calculate_primary_score(bbox, conf, image_shape)
            scores.append(score)
        
        # 最高スコアを主キャラクター、残りを副キャラクターとする
        primary_idx = np.argmax(scores)
        secondary_indices = [i for i in range(len(bboxes)) if i != primary_idx]
        
        return [primary_idx], secondary_indices
    
    def hierarchical_match(self, predictions: List[Any], ground_truths: List[Any],
                          pred_types: List[str], gt_types: List[str],
                          iou_matrix: np.ndarray, 
                          similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """階層的マッチング（主・副別々に処理）"""
        results = {}
        
        # 主キャラクターマッチング
        primary_pred_indices = [i for i, t in enumerate(pred_types) if t == 'primary']
        primary_gt_indices = [i for i, t in enumerate(gt_types) if t == 'primary']
        
        if primary_pred_indices and primary_gt_indices:
            primary_iou = iou_matrix[np.ix_(primary_pred_indices, primary_gt_indices)]
            primary_sim = similarity_matrix[np.ix_(primary_pred_indices, primary_gt_indices)]
            
            primary_results = self.evaluate_matches(
                [predictions[i] for i in primary_pred_indices],
                [ground_truths[i] for i in primary_gt_indices],
                primary_iou, primary_sim
            )
            results['primary'] = primary_results
        
        # 副キャラクターマッチング（重み調整）
        secondary_pred_indices = [i for i, t in enumerate(pred_types) if t == 'secondary']
        secondary_gt_indices = [i for i, t in enumerate(gt_types) if t == 'secondary']
        
        if secondary_pred_indices and secondary_gt_indices:
            secondary_iou = iou_matrix[np.ix_(secondary_pred_indices, secondary_gt_indices)]
            secondary_sim = similarity_matrix[np.ix_(secondary_pred_indices, secondary_gt_indices)]
            
            # 副キャラクターは重みを下げる
            original_weight = self.config.alpha
            self.config.alpha = original_weight * self.config.secondary_weight
            
            secondary_results = self.evaluate_matches(
                [predictions[i] for i in secondary_pred_indices],
                [ground_truths[i] for i in secondary_gt_indices],
                secondary_iou, secondary_sim
            )
            
            # 重みを元に戻す
            self.config.alpha = original_weight
            results['secondary'] = secondary_results
        
        return results