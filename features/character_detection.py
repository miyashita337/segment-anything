from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from core.sam_predictor import SamPredictor

@dataclass
class DetectionParams:
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 2.5
    solid_fill_threshold: float = 0.85
    min_area_ratio: float = 0.1

class CharacterDetector:
    def __init__(self, sam_predictor: SamPredictor):
        self.predictor = sam_predictor
        self.params = DetectionParams()
    
    def detect_character(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect full character with aspect ratio validation and fallback.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Character mask or None if no valid detection
        """
        initial_mask = self._get_initial_mask(image)
        if initial_mask is None:
            return None
            
        if self._validate_aspect_ratio(initial_mask):
            return initial_mask
            
        return self._fallback_detection(image)
    
    def _validate_aspect_ratio(self, mask: np.ndarray) -> bool:
        h, w = mask.shape[:2]
        aspect = h / w
        return (self.params.min_aspect_ratio <= aspect <= self.params.max_aspect_ratio)
    
    def _get_solid_fill_score(self, mask: np.ndarray) -> float:
        filled = np.sum(mask)
        total = mask.shape[0] * mask.shape[1]
        return filled / total
    
    def _fallback_detection(self, image: np.ndarray) -> Optional[np.ndarray]:
        masks = self.predictor.generate_masks(image)
        for mask in masks:
            if (self._validate_aspect_ratio(mask) and
                self._get_solid_fill_score(mask) >= self.params.solid_fill_threshold):
                return mask
        return None