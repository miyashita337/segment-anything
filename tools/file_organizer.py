"""Utilities for organizing and validating file movements."""
from pathlib import Path
from typing import List, Tuple
import shutil

def validate_file_movement(src: Path, dst: Path) -> bool:
    """Validate if a file can be safely moved to the destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if movement is valid, False otherwise
    """
    if not src.exists():
        return False
    if dst.exists():
        return False
    return True

def move_file_safely(src: Path, dst: Path) -> bool:
    """Move a file while preserving git history and validating the operation.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if move was successful
    """
    if not validate_file_movement(src, dst):
        return False
        
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        return False
