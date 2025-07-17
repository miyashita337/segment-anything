#!/usr/bin/env python3
"""
Test Suite for Character Extraction
Comprehensive tests for the refactored character extraction system
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# Import modules to test
from models.sam_wrapper import SAMModelWrapper
from models.yolo_wrapper import YOLOModelWrapper
from utils.preprocessing import (
    load_and_validate_image,
    resize_image_if_needed,
    is_color_image,
    preprocess_image_pipeline
)
from utils.postprocessing import (
    enhance_character_mask,
    extract_character_from_image,
    calculate_mask_quality_metrics
)
from utils.text_detection import TextDetector
from utils.performance import PerformanceMonitor
from commands.extract_character import extract_character_from_path


class TestImagePreprocessing(unittest.TestCase):
    """Test image preprocessing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_and_validate_image_success(self):
        """Test successful image loading"""
        image = load_and_validate_image(self.test_image_path)
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # Should be BGR
    
    def test_load_and_validate_image_nonexistent(self):
        """Test loading non-existent image"""
        image = load_and_validate_image("nonexistent.jpg")
        self.assertIsNone(image)
    
    def test_resize_image_if_needed(self):
        """Test image resizing functionality"""
        # Create a large test image
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        
        resized, scale = resize_image_if_needed(large_image, max_size=1024)
        
        self.assertLess(scale, 1.0)  # Should be scaled down
        self.assertLessEqual(max(resized.shape[:2]), 1024)
    
    def test_is_color_image(self):
        """Test color detection"""
        # Create color image
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :, 0] = 255  # Red channel
        
        # Create grayscale image
        gray_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        self.assertTrue(is_color_image(color_image))
        self.assertFalse(is_color_image(gray_image))
    
    def test_preprocess_image_pipeline(self):
        """Test complete preprocessing pipeline"""
        bgr_img, rgb_img, scale = preprocess_image_pipeline(self.test_image_path)
        
        self.assertIsNotNone(bgr_img)
        self.assertIsNotNone(rgb_img)
        self.assertGreater(scale, 0)
        self.assertEqual(bgr_img.shape, rgb_img.shape)


class TestPostprocessing(unittest.TestCase):
    """Test post-processing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test mask
        self.test_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(self.test_mask, (100, 100), 80, 255, -1)
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    def test_enhance_character_mask(self):
        """Test mask enhancement"""
        # Add noise to mask
        noisy_mask = self.test_mask.copy()
        noise = np.random.randint(0, 2, noisy_mask.shape, dtype=np.uint8) * 50
        noisy_mask = np.clip(noisy_mask + noise, 0, 255).astype(np.uint8)
        
        enhanced = enhance_character_mask(noisy_mask)
        
        self.assertEqual(enhanced.shape, self.test_mask.shape)
        self.assertGreaterEqual(np.sum(enhanced > 0), 0)  # Should have some content
    
    def test_extract_character_from_image(self):
        """Test character extraction"""
        extracted = extract_character_from_image(
            self.test_image, 
            self.test_mask,
            background_color=(0, 0, 0)
        )
        
        self.assertEqual(extracted.shape, self.test_image.shape)
    
    def test_calculate_mask_quality_metrics(self):
        """Test mask quality calculation"""
        metrics = calculate_mask_quality_metrics(self.test_mask)
        
        required_keys = [
            'coverage_ratio', 'compactness', 'fill_ratio', 
            'aspect_ratio', 'contour_area', 'mask_pixels', 'total_pixels'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))


class TestTextDetection(unittest.TestCase):
    """Test text detection utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.text_detector = TextDetector(use_easyocr=False)  # Use OpenCV for tests
        
        # Create test image with text-like patterns
        self.test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        # Add horizontal lines (text-like)
        for i in range(50, 150, 20):
            cv2.rectangle(self.test_image, (50, i), (250, i + 5), (0, 0, 0), -1)
    
    def test_detect_text_regions(self):
        """Test text region detection"""
        text_regions = self.text_detector.detect_text_regions(self.test_image)
        
        self.assertIsInstance(text_regions, list)
        # Should detect some text-like patterns
        
        if text_regions:
            region = text_regions[0]
            required_keys = ['bbox', 'mask', 'text', 'confidence']
            for key in required_keys:
                self.assertIn(key, region)
    
    def test_has_significant_text(self):
        """Test significant text detection"""
        has_text = self.text_detector.has_significant_text(self.test_image)
        self.assertIsInstance(has_text, bool)
    
    def test_calculate_text_density_score(self):
        """Test text density calculation"""
        bbox = (40, 40, 220, 120)
        density = self.text_detector.calculate_text_density_score(self.test_image, bbox)
        
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring"""
    
    def test_performance_monitor_basic(self):
        """Test basic performance monitoring functionality"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        monitor.start_stage("Test Stage")
        # Simulate some work
        import time
        time.sleep(0.1)
        monitor.end_stage()
        
        total_time = monitor.get_total_time()
        self.assertGreater(total_time, 0)
        
        summary = monitor.get_stage_summary()
        self.assertIn("Test Stage", summary)
        self.assertGreater(summary["Test Stage"], 0)


class TestModelWrappers(unittest.TestCase):
    """Test model wrapper classes"""
    
    def test_sam_wrapper_initialization(self):
        """Test SAM wrapper initialization"""
        wrapper = SAMModelWrapper(
            model_type="vit_h",
            checkpoint_path="nonexistent.pth"  # Will fail but should handle gracefully
        )
        
        self.assertEqual(wrapper.model_type, "vit_h")
        self.assertFalse(wrapper.is_loaded)
        
        # Test model info
        info = wrapper.get_model_info()
        self.assertIn('model_type', info)
        self.assertIn('is_loaded', info)
    
    def test_yolo_wrapper_initialization(self):
        """Test YOLO wrapper initialization"""
        wrapper = YOLOModelWrapper(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
        
        self.assertEqual(wrapper.confidence_threshold, 0.5)
        self.assertFalse(wrapper.is_loaded)
        
        # Test model info
        info = wrapper.get_model_info()
        self.assertIn('model_path', info)
        self.assertIn('confidence_threshold', info)
    
    def test_yolo_overlap_calculation(self):
        """Test YOLO overlap calculation"""
        wrapper = YOLOModelWrapper()
        
        bbox1 = [10, 10, 50, 50]  # [x, y, w, h]
        bbox2 = [30, 30, 50, 50]  # Overlapping
        bbox3 = [100, 100, 50, 50]  # Non-overlapping
        
        overlap1 = wrapper.calculate_overlap_score(bbox1, bbox2)
        overlap2 = wrapper.calculate_overlap_score(bbox1, bbox3)
        
        self.assertGreater(overlap1, 0)  # Should have overlap
        self.assertEqual(overlap2, 0)  # Should have no overlap


class TestCharacterExtraction(unittest.TestCase):
    """Test complete character extraction pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_character.jpg")
        
        # Create a test manga-like image
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add a character-like shape (dark figure)
        cv2.rectangle(test_image, (200, 100), (300, 350), (50, 50, 50), -1)
        cv2.circle(test_image, (250, 120), 20, (100, 100, 100), -1)  # Head
        
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('hooks.start.get_sam_model')
    @patch('hooks.start.get_yolo_model')
    @patch('hooks.start.get_performance_monitor')
    def test_extract_character_pipeline_mocked(self, mock_perf, mock_yolo, mock_sam):
        """Test extraction pipeline with mocked models"""
        # Mock performance monitor
        mock_monitor = MagicMock()
        mock_perf.return_value = mock_monitor
        
        # Mock SAM model
        mock_sam_instance = MagicMock()
        mock_sam_instance.generate_masks.return_value = [
            {
                'segmentation': np.ones((400, 600), dtype=bool),
                'area': 50000,
                'bbox': [200, 100, 100, 250],
                'stability_score': 0.9,
                'predicted_iou': 0.85
            }
        ]
        mock_sam_instance.filter_character_masks.return_value = [
            {
                'segmentation': np.ones((400, 600), dtype=bool),
                'area': 50000,
                'bbox': [200, 100, 100, 250],
                'stability_score': 0.9,
                'predicted_iou': 0.85
            }
        ]
        mock_sam_instance.mask_to_binary.return_value = np.ones((400, 600), dtype=np.uint8) * 255
        mock_sam.return_value = mock_sam_instance
        
        # Mock YOLO model
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.score_masks_with_detections.return_value = [
            {
                'segmentation': np.ones((400, 600), dtype=bool),
                'area': 50000,
                'bbox': [200, 100, 100, 250],
                'stability_score': 0.9,
                'predicted_iou': 0.85,
                'yolo_score': 0.6,
                'combined_score': 0.75
            }
        ]
        mock_yolo_instance.get_best_character_mask.return_value = {
            'segmentation': np.ones((400, 600), dtype=bool),
            'area': 50000,
            'bbox': [200, 100, 100, 250],
            'stability_score': 0.9,
            'predicted_iou': 0.85,
            'yolo_score': 0.6,
            'combined_score': 0.75
        }
        mock_yolo.return_value = mock_yolo_instance
        
        # Test extraction
        output_path = os.path.join(self.temp_dir, "output")
        result = extract_character_from_path(
            self.test_image_path,
            output_path=output_path,
            verbose=False
        )
        
        # Verify result structure
        self.assertIn('success', result)
        self.assertIn('processing_time', result)
        self.assertIn('mask_quality', result)
        
        # Verify models were called
        mock_sam_instance.generate_masks.assert_called_once()
        mock_yolo_instance.score_masks_with_detections.assert_called_once()


class TestIntegrationWithSampleImage(unittest.TestCase):
    """Integration tests with sample images"""
    
    def test_sample_image_exists(self):
        """Test that sample images exist"""
        sample_paths = [
            "assets/masks1.png",
            "test_small/img001.jpg",
            "test_small/img002.jpg",
            "test_small/img003.jpg"
        ]
        
        existing_samples = []
        for path in sample_paths:
            if os.path.exists(path):
                existing_samples.append(path)
        
        self.assertGreater(len(existing_samples), 0, "At least one sample image should exist")
    
    def test_preprocessing_with_sample_image(self):
        """Test preprocessing with real sample image if available"""
        sample_path = "assets/masks1.png"
        
        if os.path.exists(sample_path):
            bgr_img, rgb_img, scale = preprocess_image_pipeline(sample_path)
            
            self.assertIsNotNone(bgr_img)
            self.assertIsNotNone(rgb_img)
            self.assertGreater(scale, 0)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestImagePreprocessing,
        TestPostprocessing,
        TestTextDetection,
        TestPerformanceMonitor,
        TestModelWrappers,
        TestCharacterExtraction,
        TestIntegrationWithSampleImage
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Character Extraction Test Suite")
    print("=" * 50)
    
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)