"""Utilities for organizing project files and validating movements."""
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
import os

def get_target_directory(file_path: Path, content_type: str) -> Path:
    """Determine appropriate target directory based on file type and content.
    
    Args:
        file_path: Original file path
        content_type: Classification of file content
        
    Returns:
        Path to target directory
    """
    root = Path().absolute()
    
    if content_type == 'test':
        return root / 'tests'
    elif content_type == 'doc':
        return root / 'docs'
    elif content_type == 'deprecated':
        return root / 'deprecated'
    else:
        return root / 'features'

def validate_move(src: Path, dst: Path) -> bool:
    """Validate if file can be safely moved to new location.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool indicating if move is valid
    """
    if not src.exists():
        return False
    if dst.exists():
        return False
    return True

def move_file(src: Path, dst: Path) -> bool:
    """Move file while preserving git history.
    
    Args:
        src: Source file path 
        dst: Destination file path
        
    Returns:
        bool indicating success
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.system(f'git mv {src} {dst}')
        return True
    except Exception:
        return False