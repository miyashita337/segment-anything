from pathlib import Path
import pytest
from datetime import datetime, timedelta
from features.cleanup.file_analyzer import FileAnalyzer

@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project structure."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    
    # Create test files
    (project_root / "test1.py").touch()
    (project_root / "test2.py").write_text("print('hello')")
    (project_root / "test3.py").write_text("print('hello')")
    
    return project_root

def test_find_outdated_files(temp_project, monkeypatch):
    """Test detection of outdated files."""
    analyzer = FileAnalyzer(temp_project)
    
    # Mock datetime.now()
    fixed_now = datetime.now()
    monkeypatch.setattr('features.cleanup.file_analyzer.datetime',
                        type('MockDateTime', (), {'now': lambda: fixed_now}))
    
    # Set old mtime on test1.py
    old_time = (fixed_now - timedelta(days=31)).timestamp()
    (temp_project / "test1.py").touch()
    Path(temp_project / "test1.py").stat().st_mtime = old_time
    
    outdated = analyzer.find_outdated_files()
    assert len(outdated) == 1
    assert outdated[0].name == "test1.py"

def test_find_duplicate_files(temp_project):
    """Test detection of duplicate files."""
    analyzer = FileAnalyzer(temp_project)
    
    duplicates = analyzer.find_duplicate_files()
    assert len(duplicates) == 1
    assert duplicates[0].name in {"test2.py", "test3.py"}

def test_excluded_directories(temp_project):
    """Test that excluded directories are ignored."""
    (temp_project / "__pycache__").mkdir()
    (temp_project / "__pycache__" / "cache.py").touch()
    
    analyzer = FileAnalyzer(temp_project)
    files = list(analyzer._iter_python_files())
    
    assert not any("__pycache__" in str(f) for f in files)