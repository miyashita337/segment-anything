#!/usr/bin/env python3
"""
SAM Model Wrapper
Segment Anything Model wrapper for character extraction
"""

import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SAMModelWrapper:
    """
    Wrapper class for Segment Anything Model (SAM)
    Provides simplified interface for mask generation and character extraction
    """
    
    def __init__(self, 
                 model_type: str = "vit_h",
                 checkpoint_path: str = "sam_vit_h_4b8939.pth",
                 device: Optional[str] = None):
        """
        Initialize SAM wrapper
        
        Args:
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM checkpoint file
            device: Device to run on (cuda/cpu), auto-detect if None
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sam = None
        self.mask_generator = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load SAM model and initialize mask generator
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if checkpoint exists
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")
            
            # Load SAM model
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            
            # Initialize automatic mask generator with optimized parameters for character extraction
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Avoid tiny regions
            )
            
            self.is_loaded = True
            print(f"✅ SAM {self.model_type} loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ SAM model loading failed: {e}")
            return False
    
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate all possible masks for the given image
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            List of mask dictionaries with segmentation, area, bbox, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("SAM model not loaded. Call load_model() first.")
        
        try:
            masks = self.mask_generator.generate(image)
            return masks
        except Exception as e:
            print(f"❌ Mask generation failed: {e}")
            return []
    
    def filter_character_masks(self, 
                             masks: List[Dict[str, Any]], 
                             min_area: int = 1000,
                             max_area_ratio: float = 0.8,
                             min_aspect_ratio: float = 0.3,
                             max_aspect_ratio: float = 3.0) -> List[Dict[str, Any]]:
        """
        Filter masks that are likely to be characters
        
        Args:
            masks: List of mask dictionaries
            min_area: Minimum mask area
            max_area_ratio: Maximum area ratio relative to image
            min_aspect_ratio: Minimum aspect ratio (height/width)
            max_aspect_ratio: Maximum aspect ratio (height/width)
            
        Returns:
            Filtered list of character-like masks
        """
        if not masks:
            return []
        
        # Get image dimensions from first mask
        first_mask = masks[0]['segmentation']
        image_area = first_mask.shape[0] * first_mask.shape[1]
        max_area = int(image_area * max_area_ratio)
        
        character_masks = []
        
        for mask in masks:
            area = mask['area']
            bbox = mask['bbox']  # [x, y, width, height]
            
            # Area filter
            if area < min_area or area > max_area:
                continue
            
            # Aspect ratio filter
            width, height = bbox[2], bbox[3]
            if width <= 0 or height <= 0:
                continue
                
            aspect_ratio = height / width
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # Stability and IoU thresholds (already filtered by mask generator)
            if mask['stability_score'] < 0.85 or mask['predicted_iou'] < 0.8:
                continue
            
            character_masks.append(mask)
        
        # Sort by area (largest first) and stability score
        character_masks.sort(key=lambda x: (x['area'], x['stability_score']), reverse=True)
        
        return character_masks
    
    def get_mask_bbox(self, mask: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Get bounding box from mask
        
        Args:
            mask: Mask dictionary
            
        Returns:
            Tuple of (x, y, width, height)
        """
        return tuple(mask['bbox'])
    
    def mask_to_binary(self, mask: Dict[str, Any]) -> np.ndarray:
        """
        Convert mask dictionary to binary numpy array
        
        Args:
            mask: Mask dictionary
            
        Returns:
            Binary mask array (0 or 255)
        """
        binary_mask = mask['segmentation'].astype(np.uint8) * 255
        return binary_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'parameters': {
                'points_per_side': 32,
                'pred_iou_thresh': 0.8,
                'stability_score_thresh': 0.85,
                'min_mask_region_area': 100,
            }
        }
    
    def unload_model(self):
        """
        Unload model and free memory
        """
        if self.sam is not None:
            del self.sam
            self.sam = None
        
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
        
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️ SAM model unloaded")


if __name__ == "__main__":
    # Test the wrapper
    import cv2
    
    wrapper = SAMModelWrapper()
    
    if wrapper.load_model():
        print("✅ SAM wrapper test successful")
        print(f"Model info: {wrapper.get_model_info()}")
        
        # Test with a sample image if available
        test_image_path = "../assets/masks1.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            masks = wrapper.generate_masks(image)
            character_masks = wrapper.filter_character_masks(masks)
            
            print(f"Generated {len(masks)} total masks")
            print(f"Filtered to {len(character_masks)} character-like masks")
        
        wrapper.unload_model()
    else:
        print("❌ SAM wrapper test failed")