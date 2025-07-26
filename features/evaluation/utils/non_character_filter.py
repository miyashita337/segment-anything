"""
Non-Character Element Filter
Filters out masks, speech bubbles, and other non-character elements
"""
import numpy as np
import cv2

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class NonCharacterFilter:
    """Filter for excluding non-character elements like masks and speech bubbles."""
    
    def __init__(self):
        """Initialize non-character filter."""
        pass
    
    def is_speech_bubble(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Detect if mask represents a speech bubble.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Detection results
        """
        result = {
            'is_speech_bubble': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
            
            if roi.size == 0 or mask_roi.size == 0:
                return result
            
            # 1. Shape analysis - speech bubbles are often round or oval
            aspect_ratio = h / max(w, 1)
            if 0.5 <= aspect_ratio <= 1.8:  # Roughly circular/oval
                result['confidence'] += 0.2
                result['reasons'].append('oval_shape')
            
            # 2. Edge analysis - speech bubbles have smooth curves
            mask_roi_uint8 = mask_roi.astype(np.uint8) if mask_roi.dtype != np.uint8 else mask_roi
            edges = cv2.Canny(mask_roi_uint8, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Calculate contour smoothness
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Speech bubbles have fewer vertices when approximated
                if len(approx) < 8:
                    result['confidence'] += 0.3
                    result['reasons'].append('smooth_contour')
            
            # 3. Color analysis - speech bubbles are often white/light
            masked_pixels = roi[mask_roi > 0]
            if len(masked_pixels) > 0:
                # Convert to grayscale for analysis
                if len(masked_pixels.shape) == 2 and masked_pixels.shape[1] == 3:
                    gray_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
                else:
                    gray_pixels = masked_pixels.flatten()
                
                mean_brightness = np.mean(gray_pixels)
                if mean_brightness > 180:  # Very bright (white-ish)
                    result['confidence'] += 0.2
                    result['reasons'].append('bright_color')
            
            # 4. Size analysis - speech bubbles have certain size characteristics
            area = w * h
            if 1000 <= area <= 50000:  # Medium size objects
                result['confidence'] += 0.1
                result['reasons'].append('medium_size')
            
            # 5. Position analysis - speech bubbles are often in upper areas
            image_height = image.shape[0]
            relative_y = y / image_height
            if relative_y < 0.7:  # Upper 70% of image
                result['confidence'] += 0.1
                result['reasons'].append('upper_position')
            
            # Determine if it's a speech bubble
            result['is_speech_bubble'] = result['confidence'] > 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Speech bubble detection failed: {e}")
            return result
    
    def is_mask_object(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Detect if mask represents a mask/face covering object - 強化版.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Detection results
        """
        result = {
            'is_mask_object': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
            
            if roi.size == 0 or mask_roi.size == 0:
                return result
            
            image_height, image_width = image.shape[:2]
            area = w * h
            aspect_ratio = h / max(w, 1)
            
            # 1. 形状分析 - マスクは水平長方形、楕円、または三角形状（キャラクターを除外）
            # キャラクター除外: 縦長（>1.2）や正方形に近い（0.8-1.2）は除外
            if 0.2 <= aspect_ratio <= 0.7 and not (0.8 <= aspect_ratio <= 1.2):  # 明確に横長のみ
                result['confidence'] += 0.25
                result['reasons'].append('horizontal_shape')
                
                # より厳格なマスク比率判定
                if 0.3 <= aspect_ratio <= 0.6:  # 典型的なマスク比率（より厳格）
                    result['confidence'] += 0.15
                    result['reasons'].append('typical_mask_ratio')
            
            # 2. サイズ分析 - 小さなマスク物のみ対象（キャラクター全体を除外）
            relative_area = area / (image_width * image_height)
            
            # 小さな顔マスクのみ対象、大きなキャラクター全体は除外
            if 300 <= area <= 8000 and relative_area <= 0.05:  # より小さなサイズに限定
                result['confidence'] += 0.2
                result['reasons'].append('small_mask_size')
            
            # 極小マスク（装飾品等）
            elif area < 300:
                result['confidence'] += 0.1
                result['reasons'].append('tiny_accessory')
            
            # 3. 位置分析 - 顔マスクのみ検出（全身キャラクターを除外）
            center_x, center_y = x + w/2, y + h/2
            relative_y = center_y / image_height
            relative_x = center_x / image_width
            
            # 顔マスク位置判定: 小さなマスクで顔領域にある場合のみ
            is_in_face_region = (0.15 <= relative_y <= 0.55 and 0.25 <= relative_x <= 0.75)
            is_small_object = relative_area <= 0.05  # 画像の5%以下
            
            if is_in_face_region and is_small_object:
                result['confidence'] += 0.15  # 低めのスコア
                result['reasons'].append('small_face_accessory')
                
                # より厳格な顔中心部判定
                if 0.25 <= relative_y <= 0.45 and 0.4 <= relative_x <= 0.6:
                    result['confidence'] += 0.1
                    result['reasons'].append('face_center_small')
            
            # 4. 色分析の強化
            masked_pixels = roi[mask_roi > 0]
            if len(masked_pixels) > 0:
                if len(masked_pixels.shape) == 2 and masked_pixels.shape[1] == 3:
                    # BGR色空間での分析
                    mean_color = np.mean(masked_pixels, axis=0)
                    b, g, r = mean_color
                    
                    # HSV色空間での分析も追加
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    hsv_masked = hsv_roi[mask_roi > 0]
                    
                    if len(hsv_masked) > 0:
                        h_mean = np.mean(hsv_masked[:, 0])
                        s_mean = np.mean(hsv_masked[:, 1])
                        v_mean = np.mean(hsv_masked[:, 2])
                        
                        # 暗色マスク（黒、濃色）
                        if v_mean < 80:
                            result['confidence'] += 0.2
                            result['reasons'].append('dark_mask')
                        
                        # 白色系マスク（医療用マスクなど）
                        elif v_mean > 200 and s_mean < 50:
                            result['confidence'] += 0.25
                            result['reasons'].append('white_medical_mask')
                        
                        # 単色・低彩度（人工的な質感）
                        if s_mean < 30 and np.std(hsv_masked[:, 1]) < 15:
                            result['confidence'] += 0.15
                            result['reasons'].append('artificial_uniform_color')
                
                # グレースケールでの分析
                gray_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
                mean_brightness = np.mean(gray_pixels)
                brightness_std = np.std(gray_pixels)
                
                # 極端に均一な色（人工的）
                if brightness_std < 25:
                    result['confidence'] += 0.15
                    result['reasons'].append('uniform_texture')
            
            # 5. エッジパターン分析の強化
            mask_roi_uint8 = mask_roi.astype(np.uint8) if mask_roi.dtype != np.uint8 else mask_roi
            edges = cv2.Canny(mask_roi_uint8, 30, 100)  # より敏感な閾値
            
            # 直線検出
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=w//5, maxLineGap=8)
            
            if lines is not None and len(lines) >= 2:
                result['confidence'] += 0.1
                result['reasons'].append('straight_edges')
                
                # 平行線の検出（マスクの上下端）
                if len(lines) >= 4:
                    result['confidence'] += 0.1
                    result['reasons'].append('parallel_edges')
            
            # 6. 輪郭形状分析
            contours, _ = cv2.findContours(mask_roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 凸包解析
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(largest_contour)
                
                if hull_area > 0:
                    convexity = contour_area / hull_area
                    # マスクは比較的単純な凸形状
                    if convexity > 0.85:
                        result['confidence'] += 0.1
                        result['reasons'].append('convex_shape')
            
            # 7. テクスチャ分析（新規追加）
            if roi.size > 0:
                # Local Binary Pattern風の簡易テクスチャ分析
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # テクスチャの単調性チェック
                texture_variance = np.var(gray_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0
                
                # 非常に単調なテクスチャ（人工的なマスク）
                if texture_variance < 100:
                    result['confidence'] += 0.1
                    result['reasons'].append('monotone_texture')
            
            # 閾値を調整（キャラクター保護のため厳格に）
            result['is_mask_object'] = result['confidence'] > 0.8  # より高い閾値でキャラクターを保護
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced mask object detection failed: {e}")
            return result
    
    def is_text_or_effects(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Detect if mask represents text or visual effects.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Detection results
        """
        result = {
            'is_text_effects': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
            
            if roi.size == 0 or mask_roi.size == 0:
                return result
            
            # 1. Aspect ratio - text is often very wide or very tall
            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 0.3 or aspect_ratio > 4.0:  # Very wide or very tall
                result['confidence'] += 0.3
                result['reasons'].append('extreme_aspect_ratio')
            
            # 2. Size analysis - text elements can be very small
            area = w * h
            if area < 500:  # Very small elements
                result['confidence'] += 0.2
                result['reasons'].append('very_small')
            
            # 3. Edge complexity - text has many small details
            mask_roi_uint8 = mask_roi.astype(np.uint8) if mask_roi.dtype != np.uint8 else mask_roi
            edges = cv2.Canny(mask_roi_uint8, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            if edge_density > 0.3:  # High edge density
                result['confidence'] += 0.2
                result['reasons'].append('high_edge_density')
            
            # 4. Fragmentation - text often creates fragmented masks
            mask_roi_uint8 = mask_roi.astype(np.uint8) if mask_roi.dtype != np.uint8 else mask_roi
            contours, _ = cv2.findContours(mask_roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 3:  # Many separate pieces
                result['confidence'] += 0.2
                result['reasons'].append('fragmented')
            
            # 5. Position - text can appear anywhere but effects are often at edges
            image_height, image_width = image.shape[:2]
            relative_x = x / image_width
            relative_y = y / image_height
            
            # Edge positions (likely effects)
            if (relative_x < 0.1 or relative_x > 0.9 or 
                relative_y < 0.1 or relative_y > 0.9):
                result['confidence'] += 0.1
                result['reasons'].append('edge_position')
            
            # Determine if it's text or effects
            result['is_text_effects'] = result['confidence'] > 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Text/effects detection failed: {e}")
            return result
    
    def filter_non_character_elements(self, masks: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Filter out non-character elements from mask list.
        
        Args:
            masks: List of mask candidates
            image: Original image
            
        Returns:
            Filtered list excluding non-character elements
        """
        if not masks:
            return []
        
        filtered_masks = []
        
        logger.info(f"🔍 Non-character filter: {len(masks)} masks")
        
        for i, mask_data in enumerate(masks):
            try:
                # Get mask and bbox
                mask = mask_data.get('segmentation', mask_data.get('mask'))
                bbox = mask_data.get('bbox', [0, 0, 0, 0])
                
                if mask is None or len(bbox) < 4:
                    continue
                
                # Run all detection methods
                speech_result = self.is_speech_bubble(image, mask, bbox)
                mask_result = self.is_mask_object(image, mask, bbox)
                text_result = self.is_text_or_effects(image, mask, bbox)
                
                # Add results to mask data
                mask_data.update({
                    'speech_bubble_detection': speech_result,
                    'mask_object_detection': mask_result,
                    'text_effects_detection': text_result
                })
                
                # Determine if should be filtered out
                is_non_character = (speech_result['is_speech_bubble'] or 
                                  mask_result['is_mask_object'] or 
                                  text_result['is_text_effects'])
                
                logger.info(f"   マスク{i+1}: 吹き出し={speech_result['is_speech_bubble']}, "
                           f"マスク物={mask_result['is_mask_object']}, "
                           f"テキスト={text_result['is_text_effects']}")
                
                if not is_non_character:
                    filtered_masks.append(mask_data)
                    logger.info(f"   ✅ キープ")
                else:
                    # Log why it was filtered
                    reasons = []
                    if speech_result['is_speech_bubble']:
                        reasons.extend([f"吹き出し({speech_result['confidence']:.2f})"])
                    if mask_result['is_mask_object']:
                        reasons.extend([f"マスク物({mask_result['confidence']:.2f})"])
                    if text_result['is_text_effects']:
                        reasons.extend([f"テキスト({text_result['confidence']:.2f})"])
                    
                    logger.info(f"   ❌ 除外: {', '.join(reasons)}")
                
            except Exception as e:
                logger.error(f"Non-character filtering error for mask {i}: {e}")
                # Keep mask if filtering fails (fallback)
                filtered_masks.append(mask_data)
        
        logger.info(f"🎯 Non-character filter result: {len(masks)} → {len(filtered_masks)} masks")
        
        return filtered_masks


def apply_non_character_filter(masks: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Apply non-character filtering to mask list.
    
    Args:
        masks: List of mask candidates
        image: Original image
        
    Returns:
        Filtered mask list
    """
    filter_instance = NonCharacterFilter()
    return filter_instance.filter_non_character_elements(masks, image)


if __name__ == "__main__":
    # Test non-character filter
    logging.basicConfig(level=logging.INFO)
    
    # Create test image and masks
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
    
    # Test speech bubble (white circle)
    cv2.circle(test_image, (100, 100), 40, (255, 255, 255), -1)
    
    # Test mask object (dark horizontal rectangle)
    cv2.rectangle(test_image, (200, 180), (300, 220), (50, 50, 50), -1)
    
    filter_instance = NonCharacterFilter()
    
    # Test speech bubble detection
    bubble_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(bubble_mask, (100, 100), 40, 255, -1)
    bubble_result = filter_instance.is_speech_bubble(test_image, bubble_mask, (60, 60, 80, 80))
    print(f"Speech bubble detection: {bubble_result}")
    
    # Test mask object detection  
    mask_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask_mask, (200, 180), (300, 220), 255, -1)
    mask_result = filter_instance.is_mask_object(test_image, mask_mask, (200, 180, 100, 40))
    print(f"Mask object detection: {mask_result}")