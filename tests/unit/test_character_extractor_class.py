#!/usr/bin/env python3
"""
Test for CharacterExtractor class
Phase 0対応: 依存関係問題修正後のテスト
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/extraction'))
sys.path.append(str(project_root / 'features/evaluation'))
sys.path.append(str(project_root / 'features/common'))

from features.extraction.commands.extract_character import extract_character_from_image


class TestCharacterExtractor(unittest.TestCase):
    """Test cases for character extraction functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_extract_character_function_exists(self):
        """Test extract_character_from_image function exists"""
        self.assertTrue(callable(extract_character_from_image))
    
    def test_basic_functionality(self):
        """Test basic functionality is accessible"""
        expected_keys = [
            'enhance_contrast', 'filter_text', 'save_mask', 'save_transparent',
            'min_yolo_score', 'verbose', 'difficult_pose', 'low_threshold',
            'auto_retry', 'high_quality'
        ]
        
        for key in expected_keys:
            self.assertIn(key, self.extractor.default_settings)
        
        # Verify specific default values
        self.assertEqual(self.extractor.default_settings['min_yolo_score'], 0.1)
        self.assertTrue(self.extractor.default_settings['verbose'])
        self.assertTrue(self.extractor.default_settings['filter_text'])
    
    def test_methods_exist(self):
        """Test required methods exist"""
        self.assertTrue(hasattr(self.extractor, 'extract'))
        self.assertTrue(hasattr(self.extractor, 'batch_extract'))
        self.assertTrue(callable(self.extractor.extract))
        self.assertTrue(callable(self.extractor.batch_extract))
    
    def test_extract_method_parameters(self):
        """Test extract method accepts required parameters"""
        # This tests that the method signature is correct
        # Actual functionality testing requires model initialization
        try:
            # Should not raise TypeError for signature
            import inspect
            sig = inspect.signature(self.extractor.extract)
            params = list(sig.parameters.keys())
            
            self.assertIn('image_path', params)
            self.assertIn('output_path', params)
            
        except Exception as e:
            self.fail(f"Extract method signature test failed: {e}")
    
    def test_batch_extract_method_parameters(self):
        """Test batch_extract method accepts required parameters"""
        try:
            import inspect
            sig = inspect.signature(self.extractor.batch_extract)
            params = list(sig.parameters.keys())
            
            self.assertIn('input_dir', params)
            self.assertIn('output_dir', params)
            
        except Exception as e:
            self.fail(f"Batch extract method signature test failed: {e}")


if __name__ == '__main__':
    unittest.main()