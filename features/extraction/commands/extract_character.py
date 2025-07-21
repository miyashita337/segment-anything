from typing import Optional, Tuple
import numpy as np
from ...evaluation.utils.enhanced_solid_fill_processor import EnhancedSolidFillProcessor

def extract_character(image: np.ndarray,
                     yolo_detection: Tuple[int, int, int, int],
                     sam_predictor: object) -> np.ndarray:
    """Extract character with fallback mechanism.
    
    Args:
        image: Input image
        yolo_detection: YOLO bounding box
        sam_predictor: SAM model instance
        
    Returns:
        Extracted character mask
    """
    processor = EnhancedSolidFillProcessor()
    
    # First attempt with solid fill
    mask_with_fill = _extract_with_solid_fill(
        image, yolo_detection, sam_predictor, processor)
    
    # Check extraction quality
    if _calculate_vertical_coverage(mask_with_fill, yolo_detection) < 0.75:
        # Fallback without solid fill
        mask_without_fill = _extract_without_solid_fill(
            image, yolo_detection, sam_predictor)
        
        # Select better result
        if _calculate_iou(mask_without_fill, yolo_detection) > \
           _calculate_iou(mask_with_fill, yolo_detection):
            return mask_without_fill
            
    return mask_with_fill