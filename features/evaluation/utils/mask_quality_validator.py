"""
Mask Quality Validation System
Validates mask completeness and prevents limb/face cutting
"""
import numpy as np
import cv2

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MaskQualityValidator:
    """Validates mask quality and completeness, especially for face preservation."""
    
    def __init__(self):
        """Initialize mask quality validator."""
        # Initialize face detector for validation
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.face_cascade = None
        except:
            self.face_cascade = None
    
    def validate_face_completeness(self, image: np.ndarray, mask: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Validate that face regions are completely included in the mask.
        
        Args:
            image: Original image (BGR)
            mask: Character mask
            bbox: Mask bounding box (x, y, w, h)
            
        Returns:
            Validation results
        """
        result = {
            'face_complete': True,
            'face_coverage': 1.0,
            'missing_face_regions': [],
            'needs_expansion': False,
            'expansion_directions': [],
            'confidence': 1.0
        }
        
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
            
            if roi.size == 0 or mask_roi.size == 0:
                return result
            
            # Detect faces in ROI
            faces = self._detect_faces_in_roi(roi)
            
            if not faces:
                # No faces detected, assume complete
                result['confidence'] = 0.5  # Lower confidence without face detection
                return result
            
            # Check coverage for each detected face
            total_missing_area = 0
            total_face_area = 0
            
            for (fx, fy, fw, fh) in faces:
                face_area = fw * fh
                total_face_area += face_area
                
                # Create face region mask
                face_mask = np.zeros_like(mask_roi)
                face_mask[fy:fy+fh, fx:fx+fw] = 1
                
                # Check overlap with character mask
                overlap = np.logical_and(mask_roi > 0, face_mask > 0)
                overlap_area = np.sum(overlap)
                missing_area = face_area - overlap_area
                
                if missing_area > face_area * 0.1:  # More than 10% missing
                    total_missing_area += missing_area
                    result['face_complete'] = False
                    result['needs_expansion'] = True
                    
                    # Determine expansion direction
                    face_center_x, face_center_y = fx + fw//2, fy + fh//2
                    mask_center_x, mask_center_y = w//2, h//2
                    
                    if face_center_y < mask_center_y - fh//4:  # Face is above mask center
                        result['expansion_directions'].append('up')
                    if face_center_y > mask_center_y + fh//4:  # Face is below mask center
                        result['expansion_directions'].append('down')
                    if face_center_x < mask_center_x - fw//4:  # Face is left of mask center
                        result['expansion_directions'].append('left')
                    if face_center_x > mask_center_x + fw//4:  # Face is right of mask center
                        result['expansion_directions'].append('right')
                    
                    result['missing_face_regions'].append({
                        'bbox': (fx, fy, fw, fh),
                        'missing_ratio': missing_area / face_area,
                        'missing_area': missing_area
                    })
            
            # Calculate overall coverage
            if total_face_area > 0:
                result['face_coverage'] = max(0, 1.0 - (total_missing_area / total_face_area))
            
            # Set confidence based on detection quality
            result['confidence'] = 0.9 if self.face_cascade else 0.6
            
            return result
            
        except Exception as e:
            logger.error(f"Face completeness validation failed: {e}")
            return result
    
    def validate_body_completeness(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Validate that body regions are reasonably complete.
        
        Args:
            mask: Character mask
            bbox: Mask bounding box (x, y, w, h)
            
        Returns:
            Validation results
        """
        result = {
            'body_complete': True,
            'aspect_ratio_ok': True,
            'has_extremities': True,
            'needs_expansion': False,
            'problems': []
        }
        
        try:
            x, y, w, h = bbox
            aspect_ratio = h / max(w, 1)
            
            # Check aspect ratio - characters should be taller than wide
            if aspect_ratio < 1.0:
                result['aspect_ratio_ok'] = False
                result['problems'].append('too_wide')
            
            # Extract mask region
            mask_roi = mask[y:y+h, x:x+w] if mask.ndim == 2 else mask[y:y+h, x:x+w]
            
            if mask_roi.size == 0:
                return result
            
            # Analyze mask distribution - check for gaps that might indicate missing limbs
            # Divide into vertical sections and check for continuity
            sections = 5
            section_height = h // sections
            
            for i in range(sections):
                section_start = i * section_height
                section_end = min((i + 1) * section_height, h)
                section = mask_roi[section_start:section_end, :]
                
                if np.sum(section) == 0:  # Empty section
                    if i == 0:  # Head region empty
                        result['body_complete'] = False
                        result['problems'].append('missing_head')
                    elif i == sections - 1:  # Foot region empty
                        result['problems'].append('missing_feet')
            
            # Check for limb preservation by analyzing mask connectivity
            contours, _ = cv2.findContours((mask_roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 1:  # Disconnected parts might indicate cut limbs
                # Check if disconnected parts are significant
                main_contour = max(contours, key=cv2.contourArea)
                main_area = cv2.contourArea(main_contour)
                
                for contour in contours:
                    if contour is not main_contour:
                        area = cv2.contourArea(contour)
                        if area > main_area * 0.05:  # Significant disconnected part
                            result['has_extremities'] = False
                            result['problems'].append('disconnected_limbs')
                            break
            
            # Determine if expansion is needed
            result['needs_expansion'] = not (result['body_complete'] and result['aspect_ratio_ok'])
            
            return result
            
        except Exception as e:
            logger.error(f"Body completeness validation failed: {e}")
            return result
    
    def expand_mask_for_completeness(self, image: np.ndarray, mask: np.ndarray, 
                                   bbox: Tuple[int, int, int, int],
                                   expansion_directions: List[str],
                                   max_expansion: int = 30) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Expand mask to improve completeness based on validation results.
        
        Args:
            image: Original image
            mask: Character mask to expand
            bbox: Original bounding box
            expansion_directions: Directions to expand ['up', 'down', 'left', 'right']
            max_expansion: Maximum pixels to expand in each direction
            
        Returns:
            (expanded_mask, new_bbox)
        """
        try:
            x, y, w, h = bbox
            img_height, img_width = image.shape[:2]
            
            # Calculate expansion amounts
            expand_up = max_expansion if 'up' in expansion_directions else 10
            expand_down = max_expansion if 'down' in expansion_directions else 10
            expand_left = max_expansion if 'left' in expansion_directions else 10
            expand_right = max_expansion if 'right' in expansion_directions else 10
            
            # Calculate new bounding box
            new_x = max(0, x - expand_left)
            new_y = max(0, y - expand_up)
            new_w = min(img_width - new_x, w + expand_left + expand_right)
            new_h = min(img_height - new_y, h + expand_up + expand_down)
            
            # Create expanded mask
            expanded_mask = np.zeros((img_height, img_width), dtype=mask.dtype)
            
            # Copy original mask
            if mask.ndim == 2:
                expanded_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
            else:
                expanded_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
            
            # Smart expansion using morphological operations
            kernel_size = min(max_expansion // 2, 15)
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                
                # Extract ROI for processing
                roi_mask = expanded_mask[new_y:new_y+new_h, new_x:new_x+new_w]
                
                # Apply morphological closing to fill gaps
                closed_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                
                # Apply slight dilation for expansion
                dilated_mask = cv2.dilate(closed_mask, kernel, iterations=1)
                
                # Update expanded mask
                expanded_mask[new_y:new_y+new_h, new_x:new_x+new_w] = dilated_mask
            
            return expanded_mask, (new_x, new_y, new_w, new_h)
            
        except Exception as e:
            logger.error(f"Mask expansion failed: {e}")
            return mask, bbox
    
    def _detect_faces_in_roi(self, roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in region of interest.
        
        Args:
            roi: Region of interest image
            
        Returns:
            List of face bounding boxes
        """
        if self.face_cascade is None or roi.size == 0:
            return []
        
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                maxSize=(roi.shape[1], roi.shape[0])
            )
            
            return [(x, y, w, h) for x, y, w, h in faces]
            
        except Exception as e:
            logger.error(f"Face detection in ROI failed: {e}")
            return []
    
    def comprehensive_mask_validation(self, image: np.ndarray, mask: np.ndarray, 
                                    bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Perform comprehensive mask quality validation.
        
        Args:
            image: Original image
            mask: Character mask
            bbox: Mask bounding box
            
        Returns:
            Complete validation results
        """
        try:
            # Face completeness validation
            face_results = self.validate_face_completeness(image, mask, bbox)
            
            # Body completeness validation
            body_results = self.validate_body_completeness(mask, bbox)
            
            # Combine results
            validation = {
                'face_validation': face_results,
                'body_validation': body_results,
                'overall_quality': 'good',
                'needs_improvement': False,
                'recommended_actions': []
            }
            
            # Determine overall quality
            face_ok = face_results['face_complete'] and face_results['face_coverage'] > 0.8
            body_ok = body_results['body_complete'] and body_results['aspect_ratio_ok']
            
            if not face_ok or not body_ok:
                validation['overall_quality'] = 'poor' if (not face_ok and not body_ok) else 'fair'
                validation['needs_improvement'] = True
                
                # Add recommendations
                if not face_ok:
                    validation['recommended_actions'].append('expand_for_face_completeness')
                if not body_ok:
                    validation['recommended_actions'].append('expand_for_body_completeness')
            
            return validation
            
        except Exception as e:
            logger.error(f"Comprehensive mask validation failed: {e}")
            return {
                'face_validation': {'face_complete': True, 'confidence': 0.0},
                'body_validation': {'body_complete': True},
                'overall_quality': 'unknown',
                'needs_improvement': False,
                'recommended_actions': []
            }


def validate_and_improve_mask(image: np.ndarray, mask: np.ndarray, 
                            bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, Any]]:
    """
    Validate mask quality and improve if needed.
    
    Args:
        image: Original image
        mask: Character mask
        bbox: Original bounding box
        
    Returns:
        (improved_mask, improved_bbox, validation_results)
    """
    validator = MaskQualityValidator()
    
    # Perform validation
    validation = validator.comprehensive_mask_validation(image, mask, bbox)
    
    improved_mask = mask
    improved_bbox = bbox
    
    # Apply improvements if needed
    if validation['needs_improvement']:
        logger.info(f"🔧 Mask improvement needed: {validation['recommended_actions']}")
        
        # Expand for face completeness
        if 'expand_for_face_completeness' in validation['recommended_actions']:
            face_val = validation['face_validation']
            if face_val['needs_expansion']:
                improved_mask, improved_bbox = validator.expand_mask_for_completeness(
                    image, improved_mask, improved_bbox, 
                    face_val['expansion_directions'], max_expansion=25
                )
                logger.info(f"   🔄 Expanded mask for face completeness: {face_val['expansion_directions']}")
        
        # Additional body expansion if needed
        if 'expand_for_body_completeness' in validation['recommended_actions']:
            improved_mask, improved_bbox = validator.expand_mask_for_completeness(
                image, improved_mask, improved_bbox, 
                ['up', 'down', 'left', 'right'], max_expansion=15
            )
            logger.info(f"   🔄 Expanded mask for body completeness")
    
    return improved_mask, improved_bbox, validation


if __name__ == "__main__":
    # Test mask quality validation
    logging.basicConfig(level=logging.INFO)
    
    validator = MaskQualityValidator()
    
    # Create test image and mask
    test_image = np.ones((300, 200, 3), dtype=np.uint8) * 128
    test_mask = np.zeros((300, 200), dtype=np.uint8)
    
    # Create a mask with potential face region
    cv2.rectangle(test_mask, (50, 50), (150, 250), 255, -1)
    bbox = (50, 50, 100, 200)
    
    # Test validation
    validation = validator.comprehensive_mask_validation(test_image, test_mask, bbox)
    print(f"Validation results: {validation}")
    
    # Test improvement
    improved_mask, improved_bbox, results = validate_and_improve_mask(test_image, test_mask, bbox)
    print(f"Improved bbox: {bbox} -> {improved_bbox}")