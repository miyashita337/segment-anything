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
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sam_wrapper import SAMModelWrapper
from models.yolo_wrapper import YOLOModelWrapper
from utils.performance import PerformanceMonitor

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
        
        # Initialize YOLO model
        performance_monitor.start_stage("YOLO Model Loading")
        yolo_model = YOLOModelWrapper()
        yolo_model.load_model()
        performance_monitor.end_stage()
        
        print("✅ モデル初期化完了")
        print(f"   - SAM: {sam_model.model_type}")
        print(f"   - YOLO: {yolo_model.model_path}")
        print(f"   - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
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

if __name__ == "__main__":
    # For testing purposes
    success = start()
    if success:
        print("🎯 Hook test successful")
    else:
        print("❌ Hook test failed")
        sys.exit(1)