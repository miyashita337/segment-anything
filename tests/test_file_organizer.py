"""Tests for file organization utilities."""
import pytest
from pathlib import Path
from tools.file_organizer import get_target_directory, validate_move, move_file

@pytest.fixture
def temp_file(tmp_path):
    test_file = tmp_path / 'test.py'
    test_file.write_text('print("test")')
    return test_file

def test_get_target_directory():
    root = Path().absolute()
    assert get_target_directory(Path('test.py'), 'test') == root / 'tests'
    assert get_target_directory(Path('doc.md'), 'doc') == root / 'docs'
    assert get_target_directory(Path('old.py'), 'deprecated') == root / 'deprecated'

def test_validate_move(temp_file, tmp_path):
    target = tmp_path / 'target.py'
    assert validate_move(temp_file, target)
    assert not validate_move(Path('nonexistent.py'), target)
    
    # Test existing destination
    target.write_text('')
    assert not validate_move(temp_file, target)

def test_move_file(temp_file, tmp_path):
    target = tmp_path / 'subfolder' / 'target.py'
    assert move_file(temp_file, target)
    assert target.exists()
    assert not temp_file.exists()