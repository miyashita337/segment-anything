"""Tests for file reorganization script."""
import os
import shutil
from pathlib import Path
import pytest
from tools.reorganize_files import get_file_categories, create_directories, move_files

@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory structure."""
    test_files = [
        'test_file1.py',
        'doc_file1.md',
        'deprecated_file1.txt'
    ]
    for f in test_files:
        (tmp_path / f).touch()
    return tmp_path

def test_create_directories(temp_project_dir):
    """Test directory creation."""
    create_directories(temp_project_dir)
    expected_dirs = ['docs', 'tests', 'deprecated', 'features']
    for dir_name in expected_dirs:
        assert (temp_project_dir / dir_name).is_dir()

def test_get_file_categories():
    """Test file categorization."""
    categories = get_file_categories()
    assert 'docs' in categories
    assert 'tests' in categories
    assert 'deprecated' in categories
    assert 'features' in categories
    assert len(categories['docs']) > 0
    assert all(f.endswith('.md') for f in categories['docs'])
    assert all(f.startswith('test_') for f in categories['tests'])

def test_move_files(temp_project_dir):
    """Test file movement."""
    create_directories(temp_project_dir)
    categories = {
        'tests': ['test_file1.py'],
        'docs': ['doc_file1.md'],
        'deprecated': ['deprecated_file1.txt']
    }
    moved_files = move_files(temp_project_dir, categories)
    assert len(moved_files) == 3
    assert (temp_project_dir / 'tests' / 'test_file1.py').exists()
    assert (temp_project_dir / 'docs' / 'doc_file1.md').exists()
    assert (temp_project_dir / 'deprecated' / 'deprecated_file1.txt').exists()