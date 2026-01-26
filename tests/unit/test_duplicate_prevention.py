#!/usr/bin/env python3
"""
Test suite for duplicate processing prevention functionality.

Created for: QCC-011 duplicate processing issue resolution
Purpose: Verify file_utils and MultipleCharacterDetector duplicate prevention
"""

import os

# Import the modules we're testing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.file_utils import (
    clean_duplicate_suffixes,
    generate_output_filename,
    get_processing_type_from_filename,
    is_already_processed,
    validate_output_path,
)


class TestDuplicatePrevention(unittest.TestCase):
    """Test duplicate processing prevention functionality."""

    def test_generate_output_filename_basic(self):
        """Test basic output filename generation."""
        # Original file should get prefix
        result = generate_output_filename("kana08_0001.jpg", "extracted")
        self.assertEqual(result, "extracted_kana08_0001.jpg")

        # Already extracted file should get suffix
        result = generate_output_filename("extracted_kana08_0001.jpg", "multi_char_detection")
        self.assertEqual(result, "extracted_kana08_0001_multi_char_detection.jpg")

    def test_generate_output_filename_duplicate_prevention(self):
        """Test duplicate suffix prevention."""
        # Should not duplicate existing suffix
        result = generate_output_filename(
            "extracted_kana08_0001_multi_char_detection.jpg", "multi_char_detection"
        )
        self.assertEqual(result, "extracted_kana08_0001_multi_char_detection.jpg")

        # Should not duplicate extracted prefix
        result = generate_output_filename("extracted_kana08_0001.jpg", "extracted")
        self.assertEqual(result, "extracted_kana08_0001.jpg")

    def test_clean_duplicate_suffixes(self):
        """Test duplicate suffix cleaning."""
        # Clean duplicate multi_char_detection suffixes
        dirty = "extracted_kana08_0001_multi_char_detection_multi_char_detection.jpg"
        clean = clean_duplicate_suffixes(dirty)
        self.assertEqual(clean, "extracted_kana08_0001_multi_char_detection.jpg")

        # Clean duplicate extracted suffixes
        dirty = "extracted_extracted_kana08_0001.jpg"
        clean = clean_duplicate_suffixes(dirty)
        self.assertEqual(clean, "extracted_kana08_0001.jpg")

        # Should not change clean filename
        clean_name = "extracted_kana08_0001_multi_char_detection.jpg"
        result = clean_duplicate_suffixes(clean_name)
        self.assertEqual(result, clean_name)

    def test_get_processing_type_from_filename(self):
        """Test processing type detection from filename."""
        self.assertEqual(get_processing_type_from_filename("kana08_0001.jpg"), "original")
        self.assertEqual(
            get_processing_type_from_filename("extracted_kana08_0001.jpg"), "extraction"
        )
        self.assertEqual(
            get_processing_type_from_filename("extracted_kana08_0001_multi_char_detection.jpg"),
            "multi_char_detection",
        )

    def test_is_already_processed(self):
        """Test existing file detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake output file
            input_file = "kana08_0001.jpg"
            expected_output = generate_output_filename(input_file, "extracted")
            output_path = os.path.join(temp_dir, expected_output)

            # Write some content (>1000 bytes to pass size check)
            with open(output_path, "wb") as f:
                f.write(b"x" * 1500)

            # Should find existing file
            result = is_already_processed(input_file, temp_dir, "extracted")
            self.assertEqual(result, output_path)

            # Should not find non-existent file
            result = is_already_processed(input_file, temp_dir, "nonexistent_suffix")
            self.assertIsNone(result)

    def test_validate_output_path(self):
        """Test output path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Existing directory should validate
            self.assertTrue(validate_output_path(temp_dir))

            # Non-existent directory should be created
            new_dir = os.path.join(temp_dir, "new_subdir")
            self.assertFalse(os.path.exists(new_dir))
            self.assertTrue(validate_output_path(new_dir, create_dirs=True))
            self.assertTrue(os.path.exists(new_dir))

            # File path should validate parent directory
            file_path = os.path.join(new_dir, "test.jpg")
            self.assertTrue(validate_output_path(file_path))

    @patch("features.evaluation.utils.multiple_character_detector.is_already_processed")
    def test_multiple_character_detector_skip_processed(self, mock_is_already_processed):
        """Test MultipleCharacterDetector skips already processed files."""
        # Mock that file is already processed
        mock_is_already_processed.return_value = "/path/to/existing/output.jpg"

        # Import and test MultipleCharacterDetector
        from features.evaluation.utils.multiple_character_detector import (
            detect_multiple_characters_from_image,
        )

        test_image_path = Path("test_image.jpg")
        mock_yolo_wrapper = Mock()

        # Should return skipped result without processing
        result = detect_multiple_characters_from_image(
            test_image_path, mock_yolo_wrapper, save_visualization=False
        )

        self.assertFalse(result.is_multiple)
        self.assertEqual(result.character_count, 0)
        self.assertIn("skipped", result.technical_details)

    def test_duplicate_suffix_in_filename_skip(self):
        """Test skipping files with duplicate suffix in filename."""
        from features.evaluation.utils.multiple_character_detector import (
            detect_multiple_characters_from_image,
        )

        # File with _multi_char_detection in name should be skipped
        test_image_path = Path("extracted_kana08_0001_multi_char_detection.jpg")
        mock_yolo_wrapper = Mock()

        result = detect_multiple_characters_from_image(
            test_image_path, mock_yolo_wrapper, save_visualization=False
        )

        self.assertFalse(result.is_multiple)
        self.assertEqual(result.character_count, 0)

    def test_real_world_duplicate_pattern(self):
        """Test the actual duplicate pattern we encountered."""
        # This is the actual problematic filename pattern from QCC-011
        problematic_filename = "extracted_kana08_0001_multi_char_detection_multi_char_detection.jpg"

        # Should be detected as already processed
        processing_type = get_processing_type_from_filename(problematic_filename)
        self.assertEqual(processing_type, "multi_char_detection")

        # Should be cleaned to single suffix
        cleaned = clean_duplicate_suffixes(problematic_filename)
        self.assertEqual(cleaned, "extracted_kana08_0001_multi_char_detection.jpg")

        # Should not generate new duplicates
        result = generate_output_filename(cleaned, "multi_char_detection")
        self.assertEqual(result, "extracted_kana08_0001_multi_char_detection.jpg")

    def test_edge_cases(self):
        """Test edge cases for robustness."""
        # Empty strings
        self.assertEqual(generate_output_filename("", "test"), "test_.jpg")

        # Files without extension
        result = generate_output_filename("kana08_0001", "extracted")
        self.assertEqual(result, "extracted_kana08_0001.jpg")

        # Files with multiple dots
        result = generate_output_filename("kana08.0001.backup.jpg", "extracted")
        self.assertEqual(result, "extracted_kana08.0001.backup.jpg")

    def test_performance_constraints(self):
        """Test that operations complete within reasonable time."""
        import time

        # Test with 100 filename generations (should complete quickly)
        start_time = time.time()

        for i in range(100):
            generate_output_filename(f"test_file_{i}.jpg", "extracted")
            clean_duplicate_suffixes(f"extracted_test_file_{i}_extracted_extracted.jpg")

        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0, "File operations should complete within 1 second")


class TestIntegrationDuplicatePrevention(unittest.TestCase):
    """Integration tests for duplicate prevention across modules."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_kana08_0001.jpg"

        # Create a minimal test image file
        with open(self.test_image_path, "wb") as f:
            f.write(b"fake_image_data_" * 100)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("cv2.imread")
    @patch("features.evaluation.utils.multiple_character_detector.logger")
    def test_full_duplicate_prevention_workflow(self, mock_logger, mock_cv2_imread):
        """Test complete duplicate prevention workflow."""
        # Mock image loading
        mock_image = MagicMock()
        mock_image.shape = (600, 800, 3)
        mock_cv2_imread.return_value = mock_image

        # Mock YOLO wrapper
        mock_yolo_wrapper = Mock()
        mock_yolo_wrapper.detect_persons.return_value = [
            {"bbox": [100, 100, 200, 300], "confidence": 0.9, "bbox_xyxy": [100, 100, 300, 400]}
        ]

        from features.evaluation.utils.multiple_character_detector import (
            detect_multiple_characters_from_image,
        )

        # First run should process normally
        result1 = detect_multiple_characters_from_image(
            self.test_image_path, mock_yolo_wrapper, save_visualization=True
        )
        self.assertFalse(result1.is_multiple)  # Single character

        # Create the expected output file to simulate first processing
        expected_output = generate_output_filename(
            str(self.test_image_path), "multi_char_detection"
        )
        output_path = self.test_image_path.parent / expected_output
        with open(output_path, "wb") as f:
            f.write(b"x" * 1500)  # Write enough data to pass size check

        # Second run should skip due to existing output
        result2 = detect_multiple_characters_from_image(
            self.test_image_path, mock_yolo_wrapper, save_visualization=True
        )
        self.assertFalse(result2.is_multiple)
        self.assertIn("skipped", result2.technical_details)


if __name__ == "__main__":
    # Run the tests
    print("🧪 Running duplicate prevention tests...")

    # Configure logging to reduce noise during tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run tests with detailed output
    unittest.main(verbosity=2)
