"""
Face Detection Utilities for Character Validation
Validates character masks by detecting facial features
"""
import numpy as np
import cv2

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection for character validation."""
    
    def __init__(self):
        """Initialize face detector with OpenCV cascade."""
        # Try to load face cascade
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        self.face_cascade = None
        for cascade_path in cascade_paths:
            try:
                if Path(cascade_path).exists():
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not self.face_cascade.empty():
                        logger.info(f"Face detector loaded: {cascade_path}")
                        break
            except Exception as e:
                logger.warning(f"Failed to load cascade {cascade_path}: {e}")
        
        if self.face_cascade is None or self.face_cascade.empty():
            logger.warning("Face cascade not loaded, using fallback detection")
            
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if image is None or image.size == 0:
            return []
            
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            faces = []
            
            if self.face_cascade and not self.face_cascade.empty():
                # Use Haar cascade detection
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20),
                    maxSize=(int(image.shape[1] * 0.8), int(image.shape[0] * 0.8))
                )
                faces = [(x, y, w, h) for x, y, w, h in detected_faces]
            else:
                # Fallback: simple skin color detection
                faces = self._fallback_face_detection(image)
                
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _fallback_face_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Fallback face detection using skin color and shape analysis.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Potential face regions
        """
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range (adjusted for anime)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum face area
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check aspect ratio (faces are roughly square to rectangular)
                    aspect_ratio = h / max(w, 1)
                    if 0.8 <= aspect_ratio <= 2.0:
                        faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            logger.error(f"Fallback face detection failed: {e}")
            return []
    
    def validate_character_mask(self, image: np.ndarray, mask: np.ndarray, 
                              bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Validate if mask contains a character by detecting facial features.
        
        Args:
            image: Original image (BGR)
            mask: Character mask
            bbox: Mask bounding box (x, y, w, h)
            
        Returns:
            Validation results dictionary
        """
        result = {
            'has_face': False,
            'face_count': 0,
            'face_coverage': 0.0,
            'is_character': False,
            'confidence': 0.0,
            'faces': [],
            'validation_method': 'unknown'
        }
        
        try:
            # Extract region of interest
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w] if len(image.shape) == 3 else image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return result
                
            # Detect faces in ROI
            faces = self.detect_faces(roi)
            result['faces'] = faces
            result['face_count'] = len(faces)
            result['has_face'] = len(faces) > 0
            result['validation_method'] = 'cascade' if (self.face_cascade and not self.face_cascade.empty()) else 'fallback'
            
            if faces:
                # Calculate face coverage in mask
                mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
                total_face_area = 0
                covered_face_area = 0
                
                for (fx, fy, fw, fh) in faces:
                    face_area = fw * fh
                    total_face_area += face_area
                    
                    # Check how much of face is covered by mask
                    face_mask = mask_roi[fy:fy+fh, fx:fx+fw]
                    if face_mask.size > 0:
                        covered_pixels = np.sum(face_mask > 0)
                        covered_face_area += covered_pixels
                
                if total_face_area > 0:
                    result['face_coverage'] = covered_face_area / total_face_area
                
                # Character validation logic
                result['is_character'] = self._is_character_shape(mask, bbox, faces)
                result['confidence'] = self._calculate_confidence(result, bbox, mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Character validation failed: {e}")
            return result
    
    def _is_character_shape(self, mask: np.ndarray, bbox: Tuple[int, int, int, int], 
                           faces: List[Tuple[int, int, int, int]]) -> bool:
        """
        Determine if mask represents a character based on shape analysis.
        LoRA学習用：顔が見えないキャラクターも含めて判定
        
        Args:
            mask: Character mask
            bbox: Bounding box
            faces: Detected faces
            
        Returns:
            True if likely a character
        """
        try:
            x, y, w, h = bbox
            
            # Basic shape analysis
            aspect_ratio = h / max(w, 1)
            area_ratio = np.sum(mask > 0) / (w * h) if w * h > 0 else 0
            
            # Character criteria (キャラクター抽出最優先で超緩和)
            has_vertical_shape = 0.3 <= aspect_ratio <= 8.0  # 超広いアスペクト比を許可
            has_reasonable_fill = 0.02 <= area_ratio <= 0.95  # 極端に柔軟な塗りつぶし比率
            has_face = len(faces) > 0
            
            # 人体らしいサイズと形状（キャラクター抽出最優先で超緩和）
            is_human_size = w >= 10 and h >= 20  # 最小人体サイズを大幅緩和
            is_reasonable_aspect = 0.1 <= aspect_ratio <= 10.0  # 極端に広い範囲をOK
            
            # Exclude non-character shapes（超緩和）
            is_too_thin = w < 8 or h < 15  # サイズ制限を大幅緩和
            is_too_wide = aspect_ratio < 0.05  # 極端な横長のみ除外
            
            # LoRA学習用判定ロジック：顔が見えなくても人体らしい形状ならOK
            if has_face:
                # 顔が見える場合：従来通り
                return (has_vertical_shape and has_reasonable_fill and 
                       not is_too_thin and not is_too_wide)
            else:
                # 顔が見えない場合：形状とサイズで判定（後ろ向き、横顔、アクションポーズ等）
                return (is_human_size and is_reasonable_aspect and has_reasonable_fill and 
                       not is_too_thin and not is_too_wide)
            
        except Exception as e:
            logger.error(f"Character shape analysis failed: {e}")
            return False
    
    def _calculate_confidence(self, validation_result: Dict[str, Any], 
                            bbox: Tuple[int, int, int, int], 
                            mask: np.ndarray) -> float:
        """
        Calculate confidence score for character detection.
        LoRA学習用：顔が見えない場合でも適切な信頼度を計算
        
        Args:
            validation_result: Validation results
            bbox: Bounding box
            mask: Character mask
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence = 0.0
            x, y, w, h = bbox
            aspect_ratio = h / max(w, 1)
            area = w * h
            mask_area = np.sum(mask > 0)
            fill_ratio = mask_area / area if area > 0 else 0
            
            if validation_result['has_face']:
                # 顔が検出できる場合：従来の重み付け
                face_confidence = min(validation_result['face_coverage'], 1.0)
                confidence += 0.4 * face_confidence
                
                # Shape confidence (30%)
                if 1.2 <= aspect_ratio <= 2.5:
                    shape_confidence = 1.0 - abs(aspect_ratio - 1.8) / 1.3
                    confidence += 0.3 * max(shape_confidence, 0.0)
                
                # Size confidence (20%)
                if 500 <= area <= 100000:
                    size_confidence = 1.0 - abs(area - 10000) / 50000
                    confidence += 0.2 * max(size_confidence, 0.0)
                
                # Mask quality confidence (10%)
                if 0.1 <= fill_ratio <= 0.7:
                    quality_confidence = 1.0 - abs(fill_ratio - 0.4) / 0.3
                    confidence += 0.1 * max(quality_confidence, 0.0)
            else:
                # 顔が検出できない場合：形状・サイズ・品質重視
                # Shape confidence (50% - より重要)
                if 0.8 <= aspect_ratio <= 3.5:
                    shape_confidence = 1.0 - abs(aspect_ratio - 1.5) / 2.0
                    confidence += 0.5 * max(shape_confidence, 0.0)
                
                # Size confidence (30% - 人体らしいサイズ)
                if 300 <= area <= 150000:  # 範囲を拡大
                    size_confidence = 1.0 - abs(area - 15000) / 75000
                    confidence += 0.3 * max(size_confidence, 0.0)
                
                # Mask quality confidence (20% - 塗りつぶし品質)
                if 0.05 <= fill_ratio <= 0.85:  # 範囲を拡大
                    quality_confidence = 1.0 - abs(fill_ratio - 0.4) / 0.4
                    confidence += 0.2 * max(quality_confidence, 0.0)
                
                # 基本的な人体形状ボーナス
                if (w >= 15 and h >= 30 and 
                    0.2 <= aspect_ratio <= 5.0 and 
                    0.05 <= fill_ratio <= 0.85):
                    confidence += 0.1  # 基本ボーナス
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0


def filter_non_character_masks(masks: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Filter out non-character masks using face detection.
    
    Args:
        masks: List of mask candidates with YOLO scores
        image: Original image
        
    Returns:
        Filtered list of character masks
    """
    if not masks:
        return []
    
    face_detector = FaceDetector()
    character_masks = []
    
    logger.info(f"🔍 Validating {len(masks)} masks with face detection")
    
    for i, mask_data in enumerate(masks):
        try:
            # Get mask and bbox
            mask = mask_data.get('segmentation', mask_data.get('mask'))
            bbox = mask_data.get('bbox', [0, 0, 0, 0])
            
            if mask is None or len(bbox) < 4:
                continue
                
            # Validate character
            validation = face_detector.validate_character_mask(image, mask, bbox)
            
            # Add validation info to mask data
            mask_data.update({
                'face_validation': validation,
                'character_confidence': validation['confidence'],
                'has_face': validation['has_face'],
                'is_validated_character': validation['is_character']
            })
            
            logger.info(f"   マスク{i+1}: 顔検出={validation['has_face']}, "
                       f"信頼度={validation['confidence']:.3f}, "
                       f"キャラクター={validation['is_character']}")
            
            # Filter criteria (LoRA学習用：キャラクター抽出最優先で極限まで緩和)
            min_confidence = 0.0  # 信頼度閾値を0に設定（完全緩和）
            
            # 基本的な形状・サイズ条件のみでフィルタリング
            x, y, w, h = bbox
            basic_size_ok = w >= 5 and h >= 10  # 最小限のサイズチェック
            basic_ratio_ok = 0.05 <= (h / max(w, 1)) <= 20.0  # 極端に広い範囲
            
            if (validation['is_character'] and validation['confidence'] >= min_confidence) or (basic_size_ok and basic_ratio_ok):
                character_masks.append(mask_data)
                logger.info(f"   ✅ キャラクターとして認定（信頼度={validation['confidence']:.3f}）")
            else:
                logger.info(f"   ❌ 非キャラクターとして除外（基本条件不満足）")
        
        except Exception as e:
            logger.error(f"Mask validation error for mask {i}: {e}")
            # Keep mask if validation fails (fallback)
            character_masks.append(mask_data)
    
    logger.info(f"🎯 フィルタ結果: {len(masks)} → {len(character_masks)} マスク")
    
    return character_masks


if __name__ == "__main__":
    # Test face detection
    logging.basicConfig(level=logging.INFO)
    
    detector = FaceDetector()
    print(f"Face detector initialized: {detector.face_cascade is not None}")
    
    # Test with a simple image
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[50:150, 75:125] = [200, 180, 160]  # Skin color rectangle
    
    faces = detector.detect_faces(test_image)
    print(f"Detected faces: {faces}")