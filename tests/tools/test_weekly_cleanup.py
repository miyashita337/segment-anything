"""Tests for weekly cleanup script."""

import pytest
from pathlib import Path
import shutil
import time
from tools.scripts.weekly_cleanup import CodebaseAnalyzer

@pytest.fixture
def temp_project(tmp_path):
    # Create test project structure
    project_root = tmp_path / 'project'
    (project_root / 'features').mkdir(parents=True)
    (project_root / 'core').mkdir()
    (project_root / 'tools').mkdir()
    
    # Create test files
    (project_root / 'features/old.py').write_text('# Old file')
    (project_root / 'features/new.py').write_text('# New file')
    
    # Set old file mtime to 31 days ago
    old_time = time.time() - (31 * 24 * 60 * 60)
    (project_root / 'features/old.py').touch(exist_ok=True)
    os.utime(project_root / 'features/old.py', (old_time, old_time))
    
    return project_root

def test_find_old_files(temp_project):
    analyzer = CodebaseAnalyzer(temp_project)
    old_files = analyzer.find_old_files()
    
    assert len(old_files) == 1
    assert old_files[0].name == 'old.py'

def test_move_to_deprecated(temp_project):
    analyzer = CodebaseAnalyzer(temp_project)
    old_files = analyzer.find_old_files()
    analyzer.move_to_deprecated(old_files)
    
    assert not (temp_project / 'features/old.py').exists()
    assert (temp_project / 'deprecated/features/old.py').exists()

def test_circular_imports(temp_project):
    # Create files with circular imports
    (temp_project / 'features/a.py').write_text('from features.b import b')
    (temp_project / 'features/b.py').write_text('from features.a import a')
    
    analyzer = CodebaseAnalyzer(temp_project)
    circles = analyzer.find_circular_imports()
    
    assert len(circles) == 1
    assert set(circles[0]) == {'features.a', 'features.b'}

def test_generate_report(temp_project):
    analyzer = CodebaseAnalyzer(temp_project)
    report = analyzer.generate_report()
    
    assert 'old_files' in report
    assert 'circular_imports' in report
    assert 'timestamp' in report