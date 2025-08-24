"""Common type definitions for segment-anything project.

Provides type aliases and common types used across the application.
"""
import numpy as np

from PIL import Image
from typing import Any, Dict, Union

# Image type aliases
ImageType = Union[np.ndarray, Image.Image]
MaskType = np.ndarray

# Model types
SAMModelType = Any
YOLOModelType = Any
PerformanceMonitorType = Any

# Detection types
DetectionResult = Dict[str, Any]
QualityMetrics = Dict[str, float]