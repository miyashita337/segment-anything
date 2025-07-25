#!/usr/bin/env python3
"""
評価システム基盤クラス
GPT-4O設計による座標+内容統合評価システム
"""

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class EvaluationResult:
    """評価結果の統一データ構造"""
    success: bool
    spatial_score: float  # IoUスコア
    content_score: float  # 内容類似度スコア
    integrated_score: float  # 統合スコア
    confidence: float  # 信頼度
    metadata: Dict[str, Any]  # 追加情報


class EvaluatorBase(ABC):
    """評価器基底クラス"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @abstractmethod
    def evaluate(self, prediction: Any, ground_truth: Any) -> EvaluationResult:
        """評価メソッド（サブクラスで実装）"""
        pass
        
    def batch_evaluate(self, predictions: list, ground_truths: list) -> list:
        """バッチ評価"""
        results = []
        for pred, gt in zip(predictions, ground_truths):
            result = self.evaluate(pred, gt)
            results.append(result)
        return results


class EvaluationConfig:
    """評価設定クラス"""
    
    def __init__(self):
        # 空間評価設定
        self.iou_threshold = 0.3
        
        # 内容評価設定
        self.content_model = "clip_ViT-B/32"  # or "dinov2"
        self.content_threshold = 0.25
        self.image_size = 224
        self.padding = 32
        
        # 統合評価設定
        self.alpha = 0.5  # IoUとContent類似度の重み (0-1)
        self.success_threshold = 0.5  # 統合スコアの成功閾値
        
        # 処理最適化設定
        self.use_fp16 = True
        self.batch_size = 8
        self.use_pca_dimension = 256  # None で無効化
        
        # マルチキャラクター設定
        self.primary_weight = 1.0
        self.secondary_weight = 0.5
        self.center_weight = 0.3
        self.area_weight = 0.4
        self.confidence_weight = 0.3


# 評価指標計算用ユーティリティ
def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, Recall, F1スコアを計算"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calculate_map(scores: np.ndarray, labels: np.ndarray, thresholds: list = None) -> Dict[str, float]:
    """mAP (mean Average Precision) を計算"""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    map_scores = {}
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        map_scores[f'mAP@{threshold}'] = f1
    
    map_scores['mAP_mean'] = np.mean(list(map_scores.values()))
    return map_scores