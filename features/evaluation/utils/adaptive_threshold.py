from typing import Tuple

def calculate_adaptive_threshold(image_width: int, image_height: int) -> float:
    """Calculate adaptive edge threshold based on image dimensions.
    
    Args:
        image_width: Width of input image
        image_height: Height of input image
        
    Returns:
        Adaptive threshold value between 0.0 and 1.0
    """
    base_threshold = min(image_width, image_height) * 0.03
    capped_threshold = min(max(base_threshold, 4.0), 24.0)
    return capped_threshold / min(image_width, image_height)