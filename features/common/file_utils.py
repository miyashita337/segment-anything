#!/usr/bin/env python3
"""
File utilities for preventing duplicate processing and filename conflicts.

Created for: QCC-011 duplicate processing issue resolution
Author: Claude Code Integration System
"""

import os
import re
from pathlib import Path
from typing import Optional


def generate_output_filename(input_file: str, suffix: str, extension: str = ".jpg") -> str:
    """
    Generate output filename with duplicate suffix prevention.

    Args:
        input_file: Input file path
        suffix: Suffix to add (e.g., 'extracted', 'multi_char_detection')
        extension: Output file extension (default: .jpg)

    Returns:
        Clean filename without duplicate suffixes

    Examples:
        generate_output_filename("kana08_0001.jpg", "extracted")
        -> "extracted_kana08_0001.jpg"

        generate_output_filename("extracted_kana08_0001.jpg", "multi_char_detection")
        -> "extracted_kana08_0001_multi_char_detection.jpg" (not duplicated)
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Check if suffix already exists in filename
    if f"_{suffix}" in base_name or base_name.startswith(suffix):
        # Already has this suffix, return as-is
        return f"{base_name}{extension}"

    # Add suffix appropriately
    if base_name.startswith("extracted_"):
        # For already extracted files, append new suffix
        return f"{base_name}_{suffix}{extension}"
    else:
        # For original files, add prefix
        return f"{suffix}_{base_name}{extension}"


def clean_duplicate_suffixes(filename: str) -> str:
    """
    Clean duplicate suffixes from filename.

    Args:
        filename: Filename to clean

    Returns:
        Filename with duplicate suffixes removed

    Examples:
        clean_duplicate_suffixes("extracted_kana08_0001_multi_char_detection_multi_char_detection.jpg")
        -> "extracted_kana08_0001_multi_char_detection.jpg"
    """
    # Remove duplicate _multi_char_detection suffixes
    pattern = r"(_multi_char_detection){2,}"
    filename = re.sub(pattern, r"_multi_char_detection", filename)

    # Remove duplicate extracted_ prefixes
    pattern = r"^(extracted_){2,}"
    filename = re.sub(pattern, r"extracted_", filename)

    return filename


def is_already_processed(input_file: str, output_dir: str, suffix: str) -> Optional[str]:
    """
    Check if input file is already processed with given suffix.

    Args:
        input_file: Input file path
        output_dir: Output directory path
        suffix: Processing suffix to check

    Returns:
        Path to existing output file if found, None otherwise
    """
    if not os.path.exists(output_dir):
        return None

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Generate expected output filename
    expected_filename = generate_output_filename(input_file, suffix)
    expected_path = os.path.join(output_dir, expected_filename)

    if (
        os.path.exists(expected_path) and os.path.getsize(expected_path) > 1000
    ):  # Minimum size check
        return expected_path

    return None


def get_processing_type_from_filename(filename: str) -> str:
    """
    Determine processing type from filename.

    Args:
        filename: Filename to analyze

    Returns:
        Processing type string
    """
    if "_multi_char_detection" in filename:
        return "multi_char_detection"
    elif "extracted_" in filename:
        return "extraction"
    else:
        return "original"


def validate_output_path(output_path: str, create_dirs: bool = True) -> bool:
    """
    Validate and optionally create output directory path.

    Args:
        output_path: Output file or directory path
        create_dirs: Whether to create missing directories

    Returns:
        True if path is valid/created, False otherwise
    """
    try:
        output_dir = (
            os.path.dirname(output_path) if os.path.splitext(output_path)[1] else output_path
        )

        if not os.path.exists(output_dir):
            if create_dirs:
                os.makedirs(output_dir, exist_ok=True)
                return True
            else:
                return False

        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test the functions
    print("Testing file_utils functions:")

    # Test duplicate suffix prevention
    test_cases = [
        ("kana08_0001.jpg", "extracted"),
        ("extracted_kana08_0001.jpg", "multi_char_detection"),
        (
            "extracted_kana08_0001_multi_char_detection.jpg",
            "multi_char_detection",
        ),  # Should not duplicate
    ]

    for input_file, suffix in test_cases:
        output = generate_output_filename(input_file, suffix)
        print(f"  {input_file} + {suffix} -> {output}")

    # Test duplicate cleaning
    duplicate_filename = "extracted_kana08_0001_multi_char_detection_multi_char_detection.jpg"
    cleaned = clean_duplicate_suffixes(duplicate_filename)
    print(f"  Clean: {duplicate_filename} -> {cleaned}")
