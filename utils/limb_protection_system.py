"""
手足切断防止システム - v0.0.43
評価結果の「手足切断」問題(30%)を解決する専用システム
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.cluster import DBSCAN

class LimbProtectionSystem:
    """
    手足切断防止システム
    人体構造を考慮した切断検出と自動補完機能
    """
    
    def __init__(self,
                 edge_detection_threshold: int = 50,
                 limb_extension_ratio: float = 0.15,
                 connectivity_threshold: float = 0.3,
                 human_proportions: Dict[str, float] = None):
        """
        初期化
        
        Args:
            edge_detection_threshold: エッジ検出閾値
            limb_extension_ratio: 手足延長比率
            connectivity_threshold: 連結性判定閾値
            human_proportions: 人体比率辞書
        """
        self.edge_threshold = edge_detection_threshold
        self.limb_extension_ratio = limb_extension_ratio
        self.connectivity_threshold = connectivity_threshold
        
        # 標準的な人体比率（頭部を1とした場合）
        self.human_proportions = human_proportions or {
            'head_to_body': 1/7.5,      # 頭部:全身比率
            'arm_to_body': 3/7.5,       # 腕:全身比率
            'leg_to_body': 4/7.5,       # 脚:全身比率
            'torso_to_body': 3/7.5,     # 胴体:全身比率
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_edge_concentration(self, mask: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """
        指定領域のエッジ集中度を検出
        
        Args:
            mask: セグメンテーションマスク
            region: 検査領域 (x, y, w, h)
            
        Returns:
            エッジ集中度 (0.0-1.0)
        """
        x, y, w, h = region
        roi = mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # エッジ検出
        edges = cv2.Canny(roi, self.edge_threshold, self.edge_threshold * 2)
        
        # エッジ密度計算
        edge_pixels = np.sum(edges > 0)
        total_pixels = roi.size
        
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def analyze_limb_structure(self, mask: np.ndarray, bbox: List[int]) -> Dict[str, any]:
        """
        手足構造分析
        
        Args:
            mask: セグメンテーションマスク
            bbox: バウンディングボックス [x, y, w, h]
            
        Returns:
            構造分析結果
        """
        x, y, w, h = bbox
        roi_mask = mask[y:y+h, x:x+w]
        
        # 輪郭検出
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'valid': False}
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # 凸包計算
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(main_contour)
        
        # 凸性（手足の突出を検出）
        convexity = contour_area / hull_area if hull_area > 0 else 0
        
        # アスペクト比
        aspect_ratio = h / w if w > 0 else 0
        
        # 重心計算
        moments = cv2.moments(main_contour)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = w//2, h//2
        
        return {
            'valid': True,
            'convexity': convexity,
            'aspect_ratio': aspect_ratio,
            'center': (cx + x, cy + y),
            'contour_area': contour_area,
            'hull_area': hull_area,
            'main_contour': main_contour
        }
    
    def detect_limb_cutoffs(self, mask: np.ndarray, bbox: List[int], image_shape: Tuple[int, int]) -> Dict[str, Dict]:
        """
        詳細な手足切断検出
        
        Args:
            mask: セグメンテーションマスク
            bbox: バウンディングボックス
            image_shape: 画像サイズ
            
        Returns:
            切断検出詳細結果
        """
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        # 手足構造分析
        structure = self.analyze_limb_structure(mask, bbox)
        
        cutoff_details = {
            'top': {'detected': False, 'severity': 0.0, 'type': 'none'},
            'bottom': {'detected': False, 'severity': 0.0, 'type': 'none'},
            'left': {'detected': False, 'severity': 0.0, 'type': 'none'},
            'right': {'detected': False, 'severity': 0.0, 'type': 'none'}
        }
        
        if not structure['valid']:
            return cutoff_details
        
        # 検査領域サイズ
        check_size = min(box_w, box_h) // 10
        
        # 上端検査（頭部切断）
        if y <= check_size:
            top_region = (x, max(0, y-check_size), box_w, check_size*2)
            edge_density = self.detect_edge_concentration(mask, top_region)
            if edge_density > 0.1:
                cutoff_details['top'] = {
                    'detected': True,
                    'severity': edge_density,
                    'type': 'head_cutoff'
                }
        
        # 下端検査（足切断）
        if y + box_h >= h - check_size:
            bottom_region = (x, y + box_h - check_size, box_w, check_size*2)
            edge_density = self.detect_edge_concentration(mask, bottom_region)
            if edge_density > 0.15:  # 足は特に重要
                cutoff_details['bottom'] = {
                    'detected': True,
                    'severity': edge_density,
                    'type': 'feet_cutoff'
                }
        
        # 左端検査（腕切断）
        if x <= check_size:
            left_region = (max(0, x-check_size), y, check_size*2, box_h)
            edge_density = self.detect_edge_concentration(mask, left_region)
            if edge_density > 0.12:
                cutoff_details['left'] = {
                    'detected': True,
                    'severity': edge_density,
                    'type': 'arm_cutoff'
                }
        
        # 右端検査（腕切断）
        if x + box_w >= w - check_size:
            right_region = (x + box_w - check_size, y, check_size*2, box_h)
            edge_density = self.detect_edge_concentration(mask, right_region)
            if edge_density > 0.12:
                cutoff_details['right'] = {
                    'detected': True,
                    'severity': edge_density,
                    'type': 'arm_cutoff'
                }
        
        return cutoff_details
    
    def generate_limb_extension(self, 
                              mask: np.ndarray, 
                              bbox: List[int], 
                              cutoff_details: Dict[str, Dict],
                              image_shape: Tuple[int, int]) -> np.ndarray:
        """
        手足延長マスク生成
        
        Args:
            mask: 元のマスク
            bbox: バウンディングボックス
            cutoff_details: 切断検出結果
            image_shape: 画像サイズ
            
        Returns:
            延長されたマスク
        """
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        extended_mask = mask.copy()
        
        for direction, details in cutoff_details.items():
            if not details['detected']:
                continue
            
            severity = details['severity']
            cutoff_type = details['type']
            
            # 延長量計算（重要度に応じて調整）
            if cutoff_type == 'feet_cutoff':
                # 足は特に重要なので大きく延長
                extension = int(box_h * self.limb_extension_ratio * 1.5 * severity)
            elif cutoff_type == 'head_cutoff':
                # 頭部も重要
                extension = int(box_h * self.limb_extension_ratio * 1.2 * severity)
            else:
                # 腕は標準延長
                extension = int(min(box_w, box_h) * self.limb_extension_ratio * severity)
            
            # 方向別延長処理
            if direction == 'top' and y > extension:
                # 上方向延長（頭部）
                extended_mask[y-extension:y, x:x+box_w] = 255
                self.logger.info(f"頭部延長: {extension}px")
                
            elif direction == 'bottom' and y + box_h + extension < h:
                # 下方向延長（足）
                extended_mask[y+box_h:y+box_h+extension, x:x+box_w] = 255
                self.logger.info(f"足部延長: {extension}px")
                
            elif direction == 'left' and x > extension:
                # 左方向延長（腕）
                extended_mask[y:y+box_h, x-extension:x] = 255
                self.logger.info(f"左腕延長: {extension}px")
                
            elif direction == 'right' and x + box_w + extension < w:
                # 右方向延長（腕）
                extended_mask[y:y+box_h, x+box_w:x+box_w+extension] = 255
                self.logger.info(f"右腕延長: {extension}px")
        
        return extended_mask
    
    def protect_limbs(self, 
                     mask_data: Dict, 
                     image_shape: Tuple[int, int],
                     enable_protection: bool = True) -> Dict:
        """
        手足保護処理のメイン関数
        
        Args:
            mask_data: マスクデータ
            image_shape: 画像サイズ
            enable_protection: 保護機能有効フラグ
            
        Returns:
            保護処理されたマスクデータ
        """
        if not enable_protection:
            return mask_data
        
        original_mask = mask_data['segmentation'].astype(np.uint8)
        bbox = mask_data['bbox']
        
        # 切断検出
        cutoff_details = self.detect_limb_cutoffs(original_mask, bbox, image_shape)
        
        # 切断が検出された場合のみ処理
        detected_cutoffs = [k for k, v in cutoff_details.items() if v['detected']]
        
        if not detected_cutoffs:
            self.logger.info("手足切断なし: 保護処理スキップ")
            return mask_data
        
        self.logger.info(f"手足切断検出: {detected_cutoffs}")
        
        # 手足延長
        protected_mask = self.generate_limb_extension(
            original_mask, bbox, cutoff_details, image_shape
        )
        
        # 新しいバウンディングボックス計算
        y_indices, x_indices = np.where(protected_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            new_x = x_indices.min()
            new_y = y_indices.min()
            new_w = x_indices.max() - new_x + 1
            new_h = y_indices.max() - new_y + 1
        else:
            new_x, new_y, new_w, new_h = bbox
        
        # 保護されたマスクデータ作成
        protected_mask_data = mask_data.copy()
        protected_mask_data['segmentation'] = protected_mask
        protected_mask_data['bbox'] = [new_x, new_y, new_w, new_h]
        protected_mask_data['area'] = np.sum(protected_mask > 0)
        
        # 保護情報を追加
        protected_mask_data['limb_protection'] = {
            'cutoffs_detected': detected_cutoffs,
            'protection_applied': True,
            'original_bbox': bbox
        }
        
        self.logger.info(f"手足保護完了: {detected_cutoffs}")
        
        return protected_mask_data