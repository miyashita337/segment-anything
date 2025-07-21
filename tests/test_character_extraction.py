import pytest
import numpy as np
from pathlib import Path

from features.extraction.commands.extract_character import (
    extract_character_from_path,
    _calculate_vertical_coverage,
    _calculate_iou
)

@pytest.fixture
def test_image():
    return str(Path(__file__).parent / 'data' / 'test_character.jpg')

def test_extract_character_success(test_image):
    result = extract_character_from_path(test_image)
    assert result['success']
    assert 'mask' in result
    assert 'metrics' in result

def test_extract_character_invalid_path():
    result = extract_character_from_path('invalid.jpg')
    assert not result['success']
    assert 'error' in result

def test_calculate_vertical_coverage():
    mask = np.ones((100, 100))
    detection = {'bbox': [0, 0, 100, 100]}
    coverage = _calculate_vertical_coverage(mask, detection)
    assert coverage == 1.0

def test_calculate_iou():
    mask = np.ones((100, 100))
    detection = {'bbox': [0, 0, 100, 100]}
    iou = _calculate_iou(mask, detection)
    assert iou == 1.0