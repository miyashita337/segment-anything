#!/usr/bin/env python3
"""
YOLO Model Wrapper
YOLOv8 wrapper for character detection and scoring
"""

import os
import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not available, YOLO functionality disabled")


class YOLOModelWrapper:
    """
    Wrapper class for YOLOv8 model
    Provides character detection and scoring capabilities
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.25,
                 device: Optional[str] = None):
        """
        Initialize YOLO wrapper
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            device: Device to run on (cuda/cpu), auto-detect if None
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.is_loaded = False
        
        # COCO class names for person detection
        self.person_class_id = 0  # 'person' class in COCO dataset
        
    def load_model(self) -> bool:
        """
        Load YOLO model
        
        Returns:
            True if successful, False otherwise
        """
        if not YOLO_AVAILABLE:
            print("❌ YOLO unavailable - ultralytics not installed")
            return False
        
        try:
            # Check if model file exists, download if needed
            if not os.path.exists(self.model_path):
                print(f"📥 Downloading YOLO model: {self.model_path}")
            
            # Load YOLO model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            self.is_loaded = True
            print(f"✅ YOLO {self.model_path} loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ YOLO model loading failed: {e}")
            return False
    
    def detect_persons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect person objects in the image
        
        Args:
            image: Input image (BGR or RGB format)
            
        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            persons = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Check if detection is a person
                        class_id = int(box.cls[0])
                        if class_id == self.person_class_id:
                            confidence = float(box.conf[0])
                            
                            # Filter by confidence threshold
                            if confidence >= self.confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                persons.append({
                                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # [x, y, w, h]
                                    'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],   # [x1, y1, x2, y2]
                                    'confidence': confidence,
                                    'area': int((x2-x1) * (y2-y1)),
                                    'class_id': class_id,
                                    'class_name': 'person'
                                })
            
            # Sort by confidence (highest first)
            persons.sort(key=lambda x: x['confidence'], reverse=True)
            
            return persons
            
        except Exception as e:
            print(f"❌ Person detection failed: {e}")
            return []
    
    def calculate_overlap_score(self, 
                              mask_bbox: Tuple[int, int, int, int], 
                              person_bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap score between mask and person detection
        
        Args:
            mask_bbox: Mask bounding box [x, y, w, h]
            person_bbox: Person detection bounding box [x, y, w, h]
            
        Returns:
            Overlap score (IoU - Intersection over Union)
        """
        # Convert to [x1, y1, x2, y2] format
        mask_x1, mask_y1 = mask_bbox[0], mask_bbox[1]
        mask_x2, mask_y2 = mask_x1 + mask_bbox[2], mask_y1 + mask_bbox[3]
        
        person_x1, person_y1 = person_bbox[0], person_bbox[1]
        person_x2, person_y2 = person_x1 + person_bbox[2], person_y1 + person_bbox[3]
        
        # Calculate intersection
        intersection_x1 = max(mask_x1, person_x1)
        intersection_y1 = max(mask_y1, person_y1)
        intersection_x2 = min(mask_x2, person_x2)
        intersection_y2 = min(mask_y2, person_y2)
        
        # Check if there's an intersection
        if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
            return 0.0
        
        # Calculate areas
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        mask_area = mask_bbox[2] * mask_bbox[3]
        person_area = person_bbox[2] * person_bbox[3]
        union_area = mask_area + person_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def score_masks_with_detections(self, 
                                   masks: List[Dict[str, Any]], 
                                   image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Score SAM masks based on YOLO person detections
        
        Args:
            masks: List of SAM mask dictionaries
            image: Input image for person detection
            
        Returns:
            List of masks with added YOLO scores
        """
        if not masks:
            return []
        
        # Get person detections
        persons = self.detect_persons(image)
        
        # Score each mask
        scored_masks = []
        
        for mask in masks:
            mask_bbox = mask['bbox']  # [x, y, w, h]
            best_overlap = 0.0
            best_confidence = 0.0
            
            # Find best matching person detection
            for person in persons:
                overlap = self.calculate_overlap_score(mask_bbox, person['bbox'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_confidence = person['confidence']
            
            # Calculate combined score
            yolo_score = best_overlap * best_confidence
            
            # Add YOLO scoring information to mask
            mask_with_score = mask.copy()
            mask_with_score.update({
                'yolo_overlap': best_overlap,
                'yolo_confidence': best_confidence,
                'yolo_score': yolo_score,
                'combined_score': mask['stability_score'] * 0.6 + yolo_score * 0.4
            })
            
            scored_masks.append(mask_with_score)
        
        # Sort by combined score (highest first)
        scored_masks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return scored_masks
    
    def get_best_character_mask(self, 
                               masks: List[Dict[str, Any]], 
                               image: np.ndarray,
                               min_yolo_score: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get the best character mask based on YOLO scoring
        
        Args:
            masks: List of SAM mask dictionaries
            image: Input image for person detection
            min_yolo_score: Minimum YOLO score threshold
            
        Returns:
            Best character mask or None if no good matches
        """
        scored_masks = self.score_masks_with_detections(masks, image)
        
        # Filter by minimum YOLO score
        good_masks = [m for m in scored_masks if m['yolo_score'] >= min_yolo_score]
        
        if good_masks:
            return good_masks[0]  # Highest scored mask
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'confidence_threshold': self.confidence_threshold,
            'available': YOLO_AVAILABLE,
            'person_class_id': self.person_class_id
        }
    
    def unload_model(self):
        """
        Unload model and free memory
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️ YOLO model unloaded")


if __name__ == "__main__":
    # Test the wrapper
    wrapper = YOLOModelWrapper()
    
    if wrapper.load_model():
        print("✅ YOLO wrapper test successful")
        print(f"Model info: {wrapper.get_model_info()}")
        
        # Test with a sample image if available
        test_image_path = "../assets/masks1.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            
            persons = wrapper.detect_persons(image)
            print(f"Detected {len(persons)} persons")
            
            for i, person in enumerate(persons):
                print(f"  Person {i+1}: confidence={person['confidence']:.3f}, bbox={person['bbox']}")
        
        wrapper.unload_model()
    else:
        print("❌ YOLO wrapper test failed")