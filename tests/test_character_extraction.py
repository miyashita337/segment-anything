import pytest
import numpy as np
from features.extraction.commands.extract_character import extract_character
from features.evaluation.utils.enhanced_solid_fill_processor import EnhancedSolidFillProcessor

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def mock_sam_predictor():
    class MockPredictor:
        def predict(self, *args):
            return np.ones((512, 512), dtype=bool)
    return MockPredictor()

def test_full_body_extraction(sample_image, mock_sam_predictor):
    yolo_box = (100, 50, 400, 500)  # Full body box
    result = extract_character(sample_image, yolo_box, mock_sam_predictor)
    
    # Verify vertical coverage
    height = yolo_box[3] - yolo_box[1]
    mask_height = result[yolo_box[1]:yolo_box[3], yolo_box[0]:yolo_box[2]].any(1).sum()
    coverage = mask_height / height
    assert coverage >= 0.75

def test_adaptive_threshold():
    processor = EnhancedSolidFillProcessor()
    image = np.zeros((1000, 800, 3))
    scores = processor.process_region(image)
    assert 0 <= scores['character_probability'] <= 1.0