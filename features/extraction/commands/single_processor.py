"""Single image processing functionality.

Handles individual image extraction with all quality improvements.
"""
import numpy as np
import cv2

import time
from features.common.hooks.start import get_sam_model, get_yolo_model, initialize_models
from features.common.types import ImageType, MaskType
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional, Tuple

from .base_command import BaseExtractionCommand, ExtractionConfig


class SingleProcessor(BaseExtractionCommand):
    """Handles single image character extraction."""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self._ensure_models_initialized()
    
    def _ensure_models_initialized(self):
        """Ensure SAM and YOLO models are initialized."""
        if get_sam_model() is None or get_yolo_model() is None:
            if self.config.verbose:
                self.logger.info("Initializing models...")
            initialize_models()
    
    def execute(self) -> Dict[str, Any]:
        """Execute single image extraction.
        
        Returns:
            Dict with extraction results
        """
        if not self.validate_config():
            return {"success": False, "error": "Configuration validation failed"}
        
        start_time = time.time()
        
        try:
            # Load and process image
            image = self._load_image()
            if image is None:
                return {"success": False, "error": "Failed to load image"}
            
            # Extract character
            result = self._extract_character_from_image(image)
            
            if result["success"]:
                # Save result
                output_path = Path(self.config.output_path)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                success = self._save_extracted_character(
                    result["extracted_character"], 
                    str(output_path)
                )
                
                if success:
                    processing_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "processing_time": processing_time,
                        "quality_score": result.get("quality_score", 0.0),
                        "mask_area": result.get("mask_area", 0)
                    }
                else:
                    return {"success": False, "error": "Failed to save extracted character"}
            else:
                return {"success": False, "error": result.get("error", "Character extraction failed")}
                
        except Exception as e:
            self.logger.error(f"Single processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_image(self) -> Optional[np.ndarray]:
        """Load image from input path."""
        try:
            image_path = Path(self.config.input_path)
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load with PIL and convert to numpy
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array (RGB -> BGR for OpenCV)
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if self.config.verbose:
                self.logger.info(f"Loaded image: {image.shape}")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None
    
    def _extract_character_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract character from image using SAM+YOLO pipeline.
        
        This is a simplified version focusing on core extraction.
        The full implementation would be imported from the original function.
        """
        try:
            # Import the actual extraction function
            from features.extraction.commands.extract_character import extract_character_from_image

            # Call the existing implementation
            result = extract_character_from_image(
                image=image,
                quality_method="balanced",
                sam_optimization_profile=self.config.sam_optimization_profile,
                enable_quality_monitoring=self.config.enable_quality_monitoring,
                quality_threshold=self.config.quality_threshold
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Character extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_extracted_character(self, extracted_character: np.ndarray, output_path: str) -> bool:
        """Save extracted character to file."""
        try:
            # Convert BGR to RGB for PIL
            if len(extracted_character.shape) == 3:
                extracted_character_rgb = cv2.cvtColor(extracted_character, cv2.COLOR_BGR2RGB)
            else:
                extracted_character_rgb = extracted_character
            
            # Save with PIL
            pil_image = Image.fromarray(extracted_character_rgb)
            pil_image.save(output_path, "PNG")
            
            if self.config.verbose:
                self.logger.info(f"Saved extracted character: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save extracted character: {e}")
            return False