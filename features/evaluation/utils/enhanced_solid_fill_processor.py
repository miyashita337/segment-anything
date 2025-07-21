from typing import Dict, List, Optional, Tuple
import numpy as np
from .adaptive_threshold import calculate_adaptive_threshold

class EnhancedSolidFillProcessor:
    def __init__(self):
        self.sigma_l_threshold = 6.0
        self.sigma_ab_threshold = 8.0
        self.yolo_sam_prior_weight = 1.2
        
    def process_region(self, 
                      image: np.ndarray,
                      yolo_box: Optional[Tuple[int, int, int, int]] = None,
                      sam_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Process image region with enhanced classification.
        
        Args:
            image: Input image array
            yolo_box: YOLO detection box (x1,y1,x2,y2)
            sam_mask: SAM segmentation mask
            
        Returns:
            Classification scores dictionary
        """
        edge_threshold = calculate_adaptive_threshold(image.shape[1], image.shape[0])
        
        # Calculate base scores
        color_scores = self._calculate_color_uniformity(image)
        edge_scores = self._calculate_edge_scores(image, edge_threshold)
        
        # Integrate detection priors
        if yolo_box is not None and sam_mask is not None:
            detection_score = self._calculate_detection_score(image, yolo_box, sam_mask)
            edge_scores *= (1 + self.yolo_sam_prior_weight * detection_score)
            
        return {
            'character_probability': edge_scores.mean() * color_scores.mean(),
            'edge_quality': edge_scores.mean(),
            'color_uniformity': color_scores.mean()
        }