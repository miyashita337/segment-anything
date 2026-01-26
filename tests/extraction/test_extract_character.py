"""Tests for character extraction command implementation."""
import numpy as np
import torch

import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

# 環境依存テスト: GPU・モデルファイル不在時は全てSkip
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not Path("sam_vit_h_4b8939.pth").exists()
    or not Path("yolov8n.pt").exists(),
    reason="Environment dependent: GPU and model files required",
)

from click.testing import CliRunner
from features.extraction.commands.extract_character import main as extract_character_cli


@pytest.fixture
def mock_models():
    return {"sam": Mock(), "yolo": Mock(), "perf": Mock()}


def test_extract_character_success(tmp_path, mock_models):
    """Character extraction success test (environment-dependent)"""
    import torch

    # Environment check: Skip if GPU/models not available
    if not torch.cuda.is_available():
        pytest.skip("Environment dependent: GPU required")

    if not Path("sam_vit_h_4b8939.pth").exists():
        pytest.skip("Environment dependent: SAM model file required")

    if not Path("yolov8n.pt").exists():
        pytest.skip("Environment dependent: YOLO model file required")

    input_path = tmp_path / "test.jpg"
    output_path = tmp_path / "output"

    # Create test image
    test_image = Image.new("RGB", (100, 100))
    test_image.save(input_path)

    # Use Click CLI runner to test the command
    runner = CliRunner()
    result = runner.invoke(
        extract_character_cli, [str(input_path), "-o", str(output_path), "--verbose"]
    )

    # Check that command ran successfully (exit code 0)
    assert result.exit_code == 0


def test_extract_character_failure(tmp_path, mock_models):
    """Character extraction failure test (environment-dependent)"""
    import torch

    # Environment check: Skip if GPU/models not available
    if not torch.cuda.is_available():
        pytest.skip("Environment dependent: GPU required")

    if not Path("sam_vit_h_4b8939.pth").exists():
        pytest.skip("Environment dependent: SAM model file required")

    input_path = tmp_path / "nonexistent.jpg"
    output_path = tmp_path / "output"

    # Test with non-existent file should handle gracefully
    runner = CliRunner()
    result = runner.invoke(
        extract_character_cli, [str(input_path), "-o", str(output_path), "--verbose"]
    )

    # Should return non-zero exit code for error
    assert result.exit_code != 0


def test_batch_processing(tmp_path, mock_models):
    """Batch processing test (environment-dependent)"""
    import torch

    # Environment check: Skip if GPU/models not available
    if not torch.cuda.is_available():
        pytest.skip("Environment dependent: GPU required")

    if not Path("sam_vit_h_4b8939.pth").exists():
        pytest.skip("Environment dependent: SAM model file required")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Create test images
    for i in range(3):
        test_image = Image.new("RGB", (100, 100))
        test_image.save(input_dir / f"test_{i}.jpg")

    # Test batch processing with CLI
    runner = CliRunner()
    result = runner.invoke(
        extract_character_cli, [str(input_dir), "-o", str(output_dir), "--batch", "--verbose"]
    )

    # Check that command ran successfully (exit code 0)
    assert result.exit_code == 0
