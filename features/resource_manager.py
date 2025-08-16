from typing import Dict, Optional
from pathlib import Path
import psutil
import torch
from core.predictor import SamPredictor
from core.automatic_mask_generator import SamAutomaticMaskGenerator

class ResourceManager:
    """Manages computational resources for image processing pipeline.

    Handles memory allocation, model caching, and cleanup for SAM and YOLO models.
    Implements automatic resource optimization based on system load.
    """

    def __init__(self, cache_dir: Path, memory_limit: float = 0.8):
        self.cache_dir = cache_dir
        self.memory_limit = memory_limit
        self.cache: Dict[str, Any] = {}
        self._setup_cache_dir()

    def _setup_cache_dir(self) -> None:
        """Creates cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_available_memory(self) -> float:
        """Returns available system memory in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)

    def cleanup(self) -> None:
        """Performs cleanup of cached resources."""
        torch.cuda.empty_cache()
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()

    def get_predictor(self, model_path: Path) -> SamPredictor:
        """Returns SAM predictor instance, creating new or using cached."""
        cache_key = str(model_path)
        if cache_key not in self.cache:
            if self.get_available_memory() < 2.0:  # 2GB minimum
                self.cleanup()
            predictor = SamPredictor.load_model(model_path)
            self.cache[cache_key] = predictor
        return self.cache[cache_key]

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        self.cleanup()