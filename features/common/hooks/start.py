#!/usr/bin/env python3
"""
Claude Code Hooks - Start
Segment Anything + YOLO Character Extraction

This hook initializes both SAM and YOLO models for character extraction tasks.
It sets up the global model instances that will be used by commands.
"""

import torch

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from features.common.performance.performance import PerformanceMonitor
from features.extraction.models.sam_wrapper import SAMModelWrapper
from features.extraction.models.yolo_wrapper import YOLOModelWrapper

# Global model instances
sam_model = None
yolo_model = None
performance_monitor = None

def start():
    """
    Initialize models and performance monitoring for character extraction.
    This function is called when Claude Code starts.
    """
    global sam_model, yolo_model, performance_monitor
    
    print("🚀 Character Extraction System起動中...")
    
    # Performance monitoring initialization
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()
    
    try:
        # Initialize SAM model
        performance_monitor.start_stage("SAM Model Loading")
        sam_model = SAMModelWrapper()
        sam_model.load_model()
        performance_monitor.end_stage()
        
        print("✅ start()関数による初期化完了（SAMのみ）")
        
        return True
        
    except Exception as e:
        print(f"❌ モデル初期化失敗: {e}")
        return False

def get_sam_model():
    """Get the initialized SAM model instance."""
    return sam_model

def get_yolo_model():
    """Get the initialized YOLO model instance."""
    return yolo_model

def get_performance_monitor():
    """Get the performance monitor instance."""
    return performance_monitor

def initialize_models():
    """Initialize SAM and YOLO models with proper error handling."""
    global sam_model, yolo_model, performance_monitor
    
    try:
        print("🔧 Initializing models...")
        
        # Performance monitor initialization (must be first)
        if performance_monitor is None:
            performance_monitor = PerformanceMonitor()
            performance_monitor.start_monitoring()
            print("✅ Performance monitor initialized")
        
        # SAM model initialization
        sam_model = SAMModelWrapper()
        if not sam_model.load_model():
            raise RuntimeError("SAM model loading failed")
        print("✅ SAM model initialized and loaded")
        
        # YOLO model initialization with load_model() call (汎用モデル + アニメ特化閾値)
        yolo_model = YOLOModelWrapper(model_path="yolov8x6_animeface.pt", confidence_threshold=0.07)
        if not yolo_model.load_model():
            raise RuntimeError("YOLO model loading failed")
        print("✅ YOLO model initialized and loaded")
        
        # Store models globally
        globals()['sam_model'] = sam_model
        globals()['yolo_model'] = yolo_model
        globals()['performance_monitor'] = performance_monitor
        
        print("🎯 Models initialization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

if __name__ == "__main__":
    # For testing purposes
    success = start()
    if success:
        print("🎯 Hook test successful")
    else:
        print("❌ Hook test failed")
        sys.exit(1)