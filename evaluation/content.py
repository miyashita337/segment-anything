#!/usr/bin/env python3
"""
内容評価器 (CLIP/DINOv2ベース)
GPT-4O設計による視覚的内容類似度評価システム
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import logging
from PIL import Image
from typing import Optional, Tuple, Union

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

from .base import EvaluationConfig, EvaluationResult, EvaluatorBase

logger = logging.getLogger(__name__)


class ContentEvaluator(EvaluatorBase):
    """視覚的内容類似度評価器"""
    
    def __init__(self, backbone: str = 'clip_ViT-B/32', device: str = "cuda", 
                 config: Optional[EvaluationConfig] = None):
        super().__init__(device)
        self.backbone = backbone
        self.config = config or EvaluationConfig()
        
        # モデル初期化
        self.model = None
        self.preprocess = None
        self._init_model()
        
        # PCA設定
        self.pca_matrix = None
        if self.config.use_pca_dimension:
            self._init_pca()
    
    def _init_model(self):
        """モデル初期化"""
        if self.backbone.startswith('clip') and CLIP_AVAILABLE:
            model_name = self.backbone.replace('clip_', '')
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            
            # FP16設定
            if self.config.use_fp16:
                self.model = self.model.half()
                
            logger.info(f"CLIP model loaded: {model_name}")
            
        elif self.backbone.startswith('dinov2'):
            # DINOv2実装（将来的な拡張用）
            raise NotImplementedError("DINOv2 support not implemented yet")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def _init_pca(self):
        """PCA初期化（ダミー実装）"""
        # 実際の実装では、事前計算されたPCA行列を読み込む
        logger.info(f"PCA dimension reduction: {self.config.use_pca_dimension}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image], 
                        bbox: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """画像前処理"""
        if isinstance(image, np.ndarray):
            if bbox is not None:
                x, y, w, h = bbox
                # パディング追加
                x = max(0, x - self.config.padding)
                y = max(0, y - self.config.padding)
                w = min(image.shape[1] - x, w + 2 * self.config.padding)
                h = min(image.shape[0] - y, h + 2 * self.config.padding)
                
                image = image[y:y+h, x:x+w]
            
            # BGR to RGB変換
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(image)
        
        # CLIPの前処理適用
        if self.preprocess is not None:
            tensor = self.preprocess(image).unsqueeze(0)
        else:
            # 基本的な前処理
            image = image.resize((self.config.image_size, self.config.image_size))
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.to(self.device)
        
        # FP16対応（CLIPモデルがHalfの場合）
        if self.config.use_fp16 and self.model is not None:
            if next(self.model.parameters()).dtype == torch.float16:
                tensor = tensor.half()
        
        return tensor
    
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """画像を特徴ベクトルにエンコード"""
        with torch.no_grad():
            if self.backbone.startswith('clip'):
                features = self.model.encode_image(image_tensor)
                features = F.normalize(features, dim=-1)
            else:
                raise NotImplementedError(f"Encoding not implemented for {self.backbone}")
            
            # PCA次元削減
            if self.pca_matrix is not None:
                features = torch.matmul(features, self.pca_matrix)
        
        return features
    
    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """コサイン類似度計算"""
        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        return float(sim.mean())
    
    def batch_encode(self, image_tensors: list) -> torch.Tensor:
        """バッチエンコーディング"""
        if len(image_tensors) == 0:
            return torch.empty(0, 512 if not self.config.use_pca_dimension else self.config.use_pca_dimension)
        
        batch_tensor = torch.cat(image_tensors, dim=0)
        return self.encode(batch_tensor)
    
    def evaluate_crop_similarity(self, pred_crop: np.ndarray, 
                               gt_crop: np.ndarray) -> float:
        """クロップ画像間の類似度評価"""
        try:
            # 前処理
            pred_tensor = self.preprocess_image(pred_crop)
            gt_tensor = self.preprocess_image(gt_crop)
            
            # エンコーディング
            pred_emb = self.encode(pred_tensor)
            gt_emb = self.encode(gt_tensor)
            
            # 類似度計算
            similarity = self.similarity(pred_emb, gt_emb)
            
            return max(0.0, min(1.0, similarity))  # 0-1にクリップ
            
        except Exception as e:
            logger.error(f"Content similarity evaluation failed: {e}")
            return 0.0
    
    def evaluate(self, prediction: Union[np.ndarray, Tuple], 
                ground_truth: Union[np.ndarray, Tuple]) -> EvaluationResult:
        """内容評価の実行"""
        if isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
            # 画像クロップを直接比較
            content_score = self.evaluate_crop_similarity(prediction, ground_truth)
        else:
            # 境界ボックス情報のみの場合は評価不可
            logger.warning("Content evaluation requires image crops, not just bounding boxes")
            content_score = 0.0
        
        success = content_score >= self.config.content_threshold
        confidence = content_score  # 類似度がそのまま信頼度
        
        return EvaluationResult(
            success=success,
            spatial_score=0.0,  # 内容評価では空間スコアは0
            content_score=content_score,
            integrated_score=content_score,  # 内容評価のみの場合
            confidence=confidence,
            metadata={
                'backbone': self.backbone,
                'content_threshold': self.config.content_threshold,
                'evaluation_type': 'content_only'
            }
        )