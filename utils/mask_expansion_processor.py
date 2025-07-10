"""
マスク拡張処理クラス - v0.0.43抽出範囲改善
抽出範囲不適切問題(60%)を解決するための適応的マスク拡張機能
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging

class MaskExpansionProcessor:
    """
    適応的マスク拡張処理クラス
    評価結果に基づく抽出範囲改善機能
    """
    
    def __init__(self, 
                 fullbody_expand_ratio: Tuple[float, float, float, float] = (0.15, 0.10, 0.15, 0.10),
                 upperbody_expand_ratio: Tuple[float, float, float, float] = (0.10, 0.10, 0.20, 0.10),
                 limb_detection_threshold: float = 0.1,
                 min_aspect_ratio: float = 1.2,
                 max_aspect_ratio: float = 2.5):
        """
        初期化
        
        Args:
            fullbody_expand_ratio: 全身用拡張比率 (上, 左, 下, 右)
            upperbody_expand_ratio: 上半身用拡張比率 (上, 左, 下, 右)
            limb_detection_threshold: 手足検出閾値
            min_aspect_ratio: 全身判定最小アスペクト比
            max_aspect_ratio: 全身判定最大アスペクト比
        """
        self.fullbody_expand_ratio = fullbody_expand_ratio
        self.upperbody_expand_ratio = upperbody_expand_ratio
        self.limb_detection_threshold = limb_detection_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        self.logger = logging.getLogger(__name__)
        
    def detect_body_type(self, mask_data: Dict) -> str:
        """
        体型タイプを検出（全身/上半身/その他）
        
        Args:
            mask_data: マスクデータ辞書
            
        Returns:
            'fullbody', 'upperbody', 'other'
        """
        bbox = mask_data['bbox']
        aspect_ratio = bbox[3] / max(bbox[2], 1)  # height / width
        
        # 全身判定
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            return 'fullbody'
        # 上半身判定（横長〜正方形に近い）
        elif 0.8 <= aspect_ratio < self.min_aspect_ratio:
            return 'upperbody'
        else:
            return 'other'
    
    def detect_limb_cutoff(self, mask: np.ndarray, bbox: List[int], image_shape: Tuple[int, int]) -> Dict[str, bool]:
        """
        手足切断を検出
        
        Args:
            mask: セグメンテーションマスク
            bbox: バウンディングボックス [x, y, w, h]
            image_shape: 画像サイズ (height, width)
            
        Returns:
            切断検出結果 {'top': bool, 'bottom': bool, 'left': bool, 'right': bool}
        """
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        cutoff_detection = {
            'top': False,
            'bottom': False,
            'left': False,
            'right': False
        }
        
        # マスクのエッジ密度を計算
        edge_thickness = 3  # エッジ検査領域の厚さ
        
        # 上端チェック
        if y > edge_thickness:
            top_region = mask[max(0, y-edge_thickness):y+edge_thickness, x:x+box_w]
            if np.sum(top_region) > 0:
                edge_density = np.sum(top_region[-edge_thickness:, :]) / np.sum(top_region)
                cutoff_detection['top'] = edge_density > self.limb_detection_threshold
        
        # 下端チェック
        if y + box_h < h - edge_thickness:
            bottom_region = mask[y+box_h-edge_thickness:min(h, y+box_h+edge_thickness), x:x+box_w]
            if np.sum(bottom_region) > 0:
                edge_density = np.sum(bottom_region[:edge_thickness, :]) / np.sum(bottom_region)
                cutoff_detection['bottom'] = edge_density > self.limb_detection_threshold
        
        # 左端チェック
        if x > edge_thickness:
            left_region = mask[y:y+box_h, max(0, x-edge_thickness):x+edge_thickness]
            if np.sum(left_region) > 0:
                edge_density = np.sum(left_region[:, -edge_thickness:]) / np.sum(left_region)
                cutoff_detection['left'] = edge_density > self.limb_detection_threshold
        
        # 右端チェック
        if x + box_w < w - edge_thickness:
            right_region = mask[y:y+box_h, x+box_w-edge_thickness:min(w, x+box_w+edge_thickness)]
            if np.sum(right_region) > 0:
                edge_density = np.sum(right_region[:, :edge_thickness]) / np.sum(right_region)
                cutoff_detection['right'] = edge_density > self.limb_detection_threshold
        
        return cutoff_detection
    
    def expand_mask_adaptive(self, 
                           mask_data: Dict, 
                           image_shape: Tuple[int, int],
                           preserve_a_rating: bool = False) -> Dict:
        """
        適応的マスク拡張
        
        Args:
            mask_data: 元のマスクデータ
            image_shape: 画像サイズ (height, width)
            preserve_a_rating: A評価保護モード
            
        Returns:
            拡張されたマスクデータ
        """
        if preserve_a_rating:
            # A評価の場合は変更しない
            self.logger.info("A評価保護モード: マスク拡張をスキップ")
            return mask_data
        
        h, w = image_shape[:2]
        original_bbox = mask_data['bbox'].copy()
        original_mask = mask_data['segmentation'].copy()
        
        # 体型タイプ検出
        body_type = self.detect_body_type(mask_data)
        self.logger.info(f"体型タイプ検出: {body_type}")
        
        # 手足切断検出
        cutoff_result = self.detect_limb_cutoff(original_mask, original_bbox, image_shape)
        self.logger.info(f"切断検出結果: {cutoff_result}")
        
        # 拡張比率選択
        if body_type == 'fullbody':
            expand_ratios = self.fullbody_expand_ratio
        elif body_type == 'upperbody':
            expand_ratios = self.upperbody_expand_ratio
        else:
            # その他の場合は控えめに拡張
            expand_ratios = (0.05, 0.05, 0.10, 0.05)
        
        # 切断検出に基づく追加拡張
        extra_expand = [0, 0, 0, 0]  # 上, 左, 下, 右
        if cutoff_result['top']:
            extra_expand[0] += 0.05
        if cutoff_result['left']:
            extra_expand[1] += 0.05
        if cutoff_result['bottom']:
            extra_expand[2] += 0.10  # 足元は特に重要
        if cutoff_result['right']:
            extra_expand[3] += 0.05
        
        # 最終拡張比率
        final_expand = tuple(expand_ratios[i] + extra_expand[i] for i in range(4))
        
        # バウンディングボックス拡張
        x, y, box_w, box_h = original_bbox
        
        expand_top = int(box_h * final_expand[0])
        expand_left = int(box_w * final_expand[1])
        expand_bottom = int(box_h * final_expand[2])
        expand_right = int(box_w * final_expand[3])
        
        # 拡張後の座標計算（画像境界を考慮）
        new_x = max(0, x - expand_left)
        new_y = max(0, y - expand_top)
        new_x2 = min(w, x + box_w + expand_right)
        new_y2 = min(h, y + box_h + expand_bottom)
        
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        
        # 新しいマスク作成
        expanded_mask = np.zeros((h, w), dtype=np.uint8)
        expanded_mask[new_y:new_y2, new_x:new_x2] = 255
        
        # 元のマスクと論理和
        final_mask = np.logical_or(expanded_mask > 0, original_mask > 0).astype(np.uint8)
        
        # 拡張されたマスクデータ作成
        expanded_mask_data = mask_data.copy()
        expanded_mask_data['bbox'] = [new_x, new_y, new_w, new_h]
        expanded_mask_data['segmentation'] = final_mask
        expanded_mask_data['area'] = np.sum(final_mask > 0)
        
        # ログ出力
        expansion_info = {
            'original_bbox': original_bbox,
            'expanded_bbox': [new_x, new_y, new_w, new_h],
            'body_type': body_type,
            'expansion_ratios': final_expand,
            'cutoff_detected': any(cutoff_result.values())
        }
        self.logger.info(f"マスク拡張完了: {expansion_info}")
        
        return expanded_mask_data
    
    def process_multiple_masks(self, 
                             masks: List[Dict], 
                             image_shape: Tuple[int, int],
                             quality_scores: Optional[List[float]] = None) -> List[Dict]:
        """
        複数マスクの一括拡張処理
        
        Args:
            masks: マスクデータリスト
            image_shape: 画像サイズ
            quality_scores: 品質スコア（A評価保護用）
            
        Returns:
            拡張されたマスクデータリスト
        """
        expanded_masks = []
        
        for i, mask_data in enumerate(masks):
            # A評価保護判定
            preserve_a = False
            if quality_scores and i < len(quality_scores):
                # 品質スコアが0.8以上をA評価とみなす
                preserve_a = quality_scores[i] >= 0.8
            
            expanded_mask = self.expand_mask_adaptive(
                mask_data, 
                image_shape, 
                preserve_a_rating=preserve_a
            )
            expanded_masks.append(expanded_mask)
        
        return expanded_masks