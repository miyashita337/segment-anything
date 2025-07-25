#!/usr/bin/env python3
"""
評価オーケストレーター
GPT-4O設計による統合評価システムの中枢
"""

import numpy as np
import cv2

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import EvaluationConfig, EvaluationResult
from .content import ContentEvaluator
from .matcher import MultiCharacterMatcher, RegionMatcher
from .spatial import IoUEvaluator

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """統合評価オーケストレーター"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        
        # 評価器の初期化
        self.spatial_evaluator = IoUEvaluator(
            device=self.config.device if hasattr(self.config, 'device') else 'cuda',
            threshold=self.config.iou_threshold
        )
        
        self.content_evaluator = ContentEvaluator(
            backbone=self.config.content_model,
            device=self.config.device if hasattr(self.config, 'device') else 'cuda',
            config=self.config
        )
        
        self.matcher = RegionMatcher(self.config)
        self.multi_matcher = MultiCharacterMatcher(self.config)
        
        logger.info("EvaluationOrchestrator initialized")
    
    def extract_crops_from_image(self, image_path: Union[str, Path], 
                               bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """画像から境界ボックス領域を切り出し"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        crops = []
        for bbox in bboxes:
            x, y, w, h = bbox
            # パディング追加
            x_pad = max(0, x - self.config.padding)
            y_pad = max(0, y - self.config.padding)
            w_pad = min(image.shape[1] - x_pad, w + 2 * self.config.padding)
            h_pad = min(image.shape[0] - y_pad, h + 2 * self.config.padding)
            
            crop = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            crops.append(crop)
        
        return crops
    
    def calculate_matrices(self, pred_bboxes: List[Tuple[int, int, int, int]],
                          gt_bboxes: List[Tuple[int, int, int, int]],
                          pred_crops: List[np.ndarray],
                          gt_crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """IoU行列と類似度行列の計算"""
        n_pred, n_gt = len(pred_bboxes), len(gt_bboxes)
        
        # IoU行列
        iou_matrix = np.zeros((n_pred, n_gt))
        for i, pred_bbox in enumerate(pred_bboxes):
            for j, gt_bbox in enumerate(gt_bboxes):
                iou_matrix[i, j] = self.spatial_evaluator.calculate_bbox_iou(pred_bbox, gt_bbox)
        
        # 内容類似度行列
        similarity_matrix = np.zeros((n_pred, n_gt))
        if len(pred_crops) == n_pred and len(gt_crops) == n_gt:
            for i, pred_crop in enumerate(pred_crops):
                for j, gt_crop in enumerate(gt_crops):
                    try:
                        similarity_matrix[i, j] = self.content_evaluator.evaluate_crop_similarity(
                            pred_crop, gt_crop
                        )
                    except Exception as e:
                        logger.warning(f"Content similarity failed for ({i},{j}): {e}")
                        similarity_matrix[i, j] = 0.0
        else:
            logger.warning("Crop count mismatch, using zero similarity")
        
        return iou_matrix, similarity_matrix
    
    def run_single_image(self, predictions: Dict[str, Any], 
                        ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """単一画像の評価実行"""
        start_time = time.time()
        
        try:
            # データ抽出
            pred_bboxes = predictions.get('bboxes', [])
            gt_bboxes = ground_truth.get('bboxes', [])
            image_path = predictions.get('image_path') or ground_truth.get('image_path')
            
            if not pred_bboxes or not gt_bboxes:
                return {
                    'success': False,
                    'reason': 'Missing bboxes',
                    'processing_time': time.time() - start_time
                }
            
            # クロップ抽出
            pred_crops = self.extract_crops_from_image(image_path, pred_bboxes)
            gt_crops = self.extract_crops_from_image(image_path, gt_bboxes)
            
            # 行列計算
            iou_matrix, similarity_matrix = self.calculate_matrices(
                pred_bboxes, gt_bboxes, pred_crops, gt_crops
            )
            
            # マッチング実行
            matching_results = self.matcher.evaluate_matches(
                pred_bboxes, gt_bboxes, iou_matrix, similarity_matrix
            )
            
            # 統合スコア計算
            integrated_score = (
                self.config.alpha * matching_results['mean_score'] + 
                (1 - self.config.alpha) * np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0
            )
            
            success = matching_results['f1'] >= self.config.success_threshold
            
            result = {
                'success': success,
                'integrated_score': integrated_score,
                'spatial_score': np.mean(iou_matrix) if iou_matrix.size > 0 else 0,
                'content_score': np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0,
                'matching_results': matching_results,
                'matrices': {
                    'iou_matrix': iou_matrix.tolist(),
                    'similarity_matrix': similarity_matrix.tolist()
                },
                'processing_time': time.time() - start_time,
                'metadata': {
                    'n_predictions': len(pred_bboxes),
                    'n_ground_truths': len(gt_bboxes),
                    'config': {
                        'alpha': self.config.alpha,
                        'iou_threshold': self.config.iou_threshold,
                        'content_threshold': self.config.content_threshold
                    }
                }
            }
            
            logger.debug(f"Single image evaluation completed in {result['processing_time']:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Single image evaluation failed: {e}")
            return {
                'success': False,
                'reason': f'Evaluation error: {str(e)}',
                'processing_time': time.time() - start_time
            }
    
    def run_batch(self, predictions_batch: List[Dict[str, Any]], 
                  ground_truths_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """バッチ評価の実行"""
        start_time = time.time()
        
        if len(predictions_batch) != len(ground_truths_batch):
            raise ValueError("Predictions and ground truths batch sizes must match")
        
        results = []
        success_count = 0
        total_integrated_score = 0
        total_spatial_score = 0
        total_content_score = 0
        
        for i, (pred, gt) in enumerate(zip(predictions_batch, ground_truths_batch)):
            result = self.run_single_image(pred, gt)
            results.append(result)
            
            if result['success']:
                success_count += 1
            
            total_integrated_score += result.get('integrated_score', 0)
            total_spatial_score += result.get('spatial_score', 0) 
            total_content_score += result.get('content_score', 0)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(predictions_batch)} images")
        
        # 集計統計
        n_images = len(results)
        success_rate = success_count / n_images if n_images > 0 else 0
        
        batch_result = {
            'success_rate': success_rate,
            'mean_integrated_score': total_integrated_score / n_images if n_images > 0 else 0,
            'mean_spatial_score': total_spatial_score / n_images if n_images > 0 else 0,
            'mean_content_score': total_content_score / n_images if n_images > 0 else 0,
            'successful_images': success_count,
            'total_images': n_images,
            'individual_results': results,
            'processing_time': time.time() - start_time,
            'avg_time_per_image': (time.time() - start_time) / n_images if n_images > 0 else 0
        }
        
        logger.info(f"Batch evaluation completed: {success_rate:.1%} success rate, "
                   f"{batch_result['avg_time_per_image']:.3f}s per image")
        
        return batch_result
    
    def run_with_visualization(self, predictions: Dict[str, Any], 
                              ground_truth: Dict[str, Any],
                              output_path: Optional[Path] = None) -> Dict[str, Any]:
        """可視化付き評価実行"""
        result = self.run_single_image(predictions, ground_truth)
        
        if output_path and result.get('matching_results'):
            self._save_visualization(predictions, ground_truth, result, output_path)
        
        return result
    
    def _save_visualization(self, predictions: Dict[str, Any], 
                           ground_truth: Dict[str, Any],
                           result: Dict[str, Any], 
                           output_path: Path):
        """可視化結果の保存"""
        try:
            image_path = predictions.get('image_path') or ground_truth.get('image_path')
            if not image_path:
                return
            
            image = cv2.imread(str(image_path))
            if image is None:
                return
            
            # Ground truth を緑で描画
            for bbox in ground_truth.get('bboxes', []):
                x, y, w, h = bbox
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # マッチした予測を青で、未マッチを赤で描画
            matches = result.get('matching_results', {}).get('matches', [])
            matched_pred_indices = {match['prediction_index'] for match in matches}
            
            for i, bbox in enumerate(predictions.get('bboxes', [])):
                x, y, w, h = bbox
                color = (255, 0, 0) if i in matched_pred_indices else (0, 0, 255)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            cv2.imwrite(str(output_path), image)
            logger.info(f"Visualization saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization save failed: {e}")