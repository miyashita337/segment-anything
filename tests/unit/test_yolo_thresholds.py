#!/usr/bin/env python3
"""
YOLO閾値比較テスト（pytest形式）
CLAUDE.md準拠のUnitTest実装
"""

import pytest
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestYOLOThresholds:
    """YOLO閾値設定に関するテスト"""
    
    @pytest.fixture
    def test_images(self):
        """テスト用画像のパスを提供"""
        return [
            "test_small/img001.jpg",
            "test_small/img002.jpg",
            "test_small/img003.jpg"
        ]
    
    @pytest.fixture
    def yolo_model(self):
        """YOLOモデルのフィクスチャ"""
        from ultralytics import YOLO
        return YOLO('yolov8n.pt')
    
    def test_threshold_007_detection(self, yolo_model, test_images):
        """閾値0.07での検出テスト"""
        threshold = 0.07
        total_detections = 0
        successful_images = 0
        
        for img_path in test_images:
            if os.path.exists(img_path):
                results = yolo_model(img_path, conf=threshold, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                if detections > 0:
                    successful_images += 1
                total_detections += detections
        
        # アサーション
        assert successful_images >= 1, f"閾値{threshold}で少なくとも1画像で検出が必要"
        assert total_detections >= 1, f"閾値{threshold}で少なくとも1つの検出が必要"
    
    def test_threshold_005_detection(self, yolo_model, test_images):
        """閾値0.005での検出テスト"""
        threshold = 0.005
        total_detections = 0
        successful_images = 0
        
        for img_path in test_images:
            if os.path.exists(img_path):
                results = yolo_model(img_path, conf=threshold, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                if detections > 0:
                    successful_images += 1
                total_detections += detections
        
        # アサーション
        assert successful_images >= 2, f"閾値{threshold}で少なくとも2画像で検出が必要"
        assert total_detections >= 10, f"閾値{threshold}で少なくとも10個の検出が必要"
    
    def test_threshold_comparison(self, yolo_model, test_images):
        """閾値0.07と0.005の比較テスト"""
        detections_007 = 0
        detections_005 = 0
        
        for img_path in test_images:
            if os.path.exists(img_path):
                # 0.07での検出
                results_007 = yolo_model(img_path, conf=0.07, verbose=False)
                detections_007 += len(results_007[0].boxes) if results_007[0].boxes is not None else 0
                
                # 0.005での検出
                results_005 = yolo_model(img_path, conf=0.005, verbose=False)
                detections_005 += len(results_005[0].boxes) if results_005[0].boxes is not None else 0
        
        # アサーション
        assert detections_005 > detections_007, "閾値0.005は0.07より多くの検出が必要"
        assert detections_005 >= detections_007 * 2, "閾値0.005は0.07の2倍以上の検出が期待される"
    
    @pytest.mark.parametrize("threshold,min_detections", [
        (0.1, 1),
        (0.07, 4),
        (0.05, 10),
        (0.01, 30),
        (0.005, 40)
    ])
    def test_threshold_range(self, yolo_model, test_images, threshold, min_detections):
        """複数閾値でのパラメトリックテスト"""
        total_detections = 0
        
        for img_path in test_images:
            if os.path.exists(img_path):
                results = yolo_model(img_path, conf=threshold, verbose=False)
                total_detections += len(results[0].boxes) if results[0].boxes is not None else 0
        
        assert total_detections >= min_detections, \
            f"閾値{threshold}で最低{min_detections}個の検出が必要（実際: {total_detections}）"