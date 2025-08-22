"""Tests for file organization utilities."""
import pytest
from pathlib import Path
from tools.file_organizer import validate_file_movement, move_file_safely

@pytest.fixture
def temp_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    return test_file

def test_validate_file_movement(temp_file, tmp_path):
    dst = tmp_path / "subdir" / "test.txt"
    assert validate_file_movement(temp_file, dst)
    
def test_validate_file_movement_nonexistent_source(tmp_path):
    src = tmp_path / "nonexistent.txt"
    dst = tmp_path / "test.txt"
    assert not validate_file_movement(src, dst)

def test_move_file_safely(temp_file, tmp_path):
    dst = tmp_path / "subdir" / "test.txt"
    assert move_file_safely(temp_file, dst)
    assert dst.exists()
    assert not temp_file.exists()
    assert dst.read_text() == "test content"
