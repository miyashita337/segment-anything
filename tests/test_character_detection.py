import pytest
import numpy as np
from features.character_detection import CharacterDetector, DetectionParams

@pytest.fixture
def mock_predictor():
    class MockPredictor:
        def generate_masks(self, image):
            h, w = image.shape[:2]
            return [np.ones((h, w))]
    return MockPredictor()

def test_aspect_ratio_validation():
    detector = CharacterDetector(mock_predictor())
    
    # Valid aspect ratio
    valid_mask = np.ones((300, 200))
    assert detector._validate_aspect_ratio(valid_mask)
    
    # Invalid aspect ratio
    invalid_mask = np.ones((600, 100)) 
    assert not detector._validate_aspect_ratio(invalid_mask)

def test_solid_fill_detection():
    detector = CharacterDetector(mock_predictor())
    
    mask = np.ones((100, 100))
    mask[40:60, 40:60] = 0
    
    score = detector._get_solid_fill_score(mask)
    assert 0.8 <= score <= 0.9

def test_fallback_mechanism():
    detector = CharacterDetector(mock_predictor())
    
    image = np.zeros((400, 200, 3))
    result = detector._fallback_detection(image)
    
    assert result is not None
    assert result.shape == (400, 200)