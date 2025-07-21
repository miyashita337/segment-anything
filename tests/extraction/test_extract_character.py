"""Tests for character extraction command implementation."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from features.extraction.commands.extract_character import extract_character, process_single_image

@pytest.fixture
def mock_models():
    return {
        'sam': Mock(),
        'yolo': Mock(),
        'perf': Mock()
    }

def test_process_single_image_success(tmp_path, mock_models):
    input_path = tmp_path / 'test.jpg'
    output_path = tmp_path / 'output.png'
    
    # Create test image
    test_image = Image.new('RGB', (100, 100))
    test_image.save(input_path)
    
    with patch('features.processing.preprocessing.preprocessing.preprocess_image_pipeline') as mock_preprocess:
        with patch('features.processing.postprocessing.postprocessing.calculate_mask_quality_metrics') as mock_metrics:
            mock_preprocess.return_value = test_image
            mock_metrics.return_value = {'quality_score': 0.9}
            
            result = process_single_image(
                input_path,
                output_path,
                mock_models['sam'],
                mock_models['yolo'],
                mock_models['perf']
            )
            
            assert result is not None
            assert output_path.exists()

def test_process_single_image_failure(tmp_path, mock_models):
    input_path = tmp_path / 'nonexistent.jpg'
    output_path = tmp_path / 'output.png'
    
    result = process_single_image(
        input_path,
        output_path,
        mock_models['sam'],
        mock_models['yolo'],
        mock_models['perf'],
        verbose=True
    )
    
    assert result is None
    assert not output_path.exists()

def test_batch_processing(tmp_path, mock_models):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    
    # Create test images
    for i in range(3):
        test_image = Image.new('RGB', (100, 100))
        test_image.save(input_dir / f'test_{i}.jpg')
    
    with patch('features.extraction.commands.extract_character.get_sam_model') as mock_sam:
        with patch('features.extraction.commands.extract_character.get_yolo_model') as mock_yolo:
            with patch('features.extraction.commands.extract_character.get_performance_monitor') as mock_perf:
                mock_sam.return_value = mock_models['sam']
                mock_yolo.return_value = mock_models['yolo']
                mock_perf.return_value = mock_models['perf']
                
                runner = CliRunner()
                result = runner.invoke(extract_character, [
                    '--batch',
                    str(input_dir),
                    '-o',
                    str(output_dir)
                ])
                
                assert result.exit_code == 0
                assert len(list(output_dir.glob('*.png'))) > 0