#!/usr/bin/env python3
"""
Claude Code Hooks - Start
Segment Anything + YOLO Character Extraction

This hook initializes both SAM and YOLO models for character extraction tasks.
It sets up the global model instances that will be used by commands.
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from features.extraction.models.sam_wrapper import SAMModelWrapper
from features.extraction.models.yolo_wrapper import YOLOModelWrapper
from features.common.performance.performance import PerformanceMonitor

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
    """Initialize models for Phase 0 new structure compatibility."""
    global sam_model, yolo_model, performance_monitor
    
    try:
        print("🚀 Phase 0対応モデル初期化開始...")
        
        # SAM model initialization with load_model() call
        sam_model = SAMModelWrapper()
        if not sam_model.load_model():
            raise RuntimeError("SAM model loading failed")
        print("✅ SAM model initialized and loaded")
        
        # YOLO model initialization with load_model() call (anime model)
        yolo_model = YOLOModelWrapper(model_path="yolov8x6_animeface.pt")
        if not yolo_model.load_model():
            raise RuntimeError("YOLO model loading failed")
        print("✅ YOLO model initialized and loaded")
        
        # Performance monitor initialization
        performance_monitor = PerformanceMonitor()
        print("✅ Performance monitor initialized")
        
        print("🎉 全モデル初期化完了（load_model()実行済み）")
        return True
        
    except Exception as e:
        print(f"❌ モデル初期化失敗: {e}")
        return False

if __name__ == "__main__":
    # For testing purposes
    success = start()
    if success:
        print("🎯 Hook test successful")
    else:
        print("❌ Hook test failed")
        sys.exit(1)