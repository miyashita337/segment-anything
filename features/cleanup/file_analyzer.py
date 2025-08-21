from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set
import hashlib

class FileAnalyzer:
    """Analyzes files in the codebase for cleanup opportunities."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.excluded_dirs = {'.git', '__pycache__', 'venv', 'env'}

    def find_outdated_files(self, days: int = 30) -> List[Path]:
        """Find files not modified in the specified number of days."""
        cutoff = datetime.now() - timedelta(days=days)
        outdated = []

        for file in self._iter_python_files():
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff:
                outdated.append(file)

        return outdated

    def find_unused_files(self) -> List[Path]:
        """Detect Python files with no incoming imports."""
        # Implementation would use AST analysis
        return []

    def find_duplicate_files(self) -> List[Path]:
        """Find files with duplicate content."""
        checksums = {}
        duplicates = []

        for file in self._iter_python_files():
            checksum = self._get_file_hash(file)
            if checksum in checksums:
                duplicates.append(file)
            else:
                checksums[checksum] = file

        return duplicates

    def _iter_python_files(self) -> Set[Path]:
        """Iterate through all Python files in project."""
        for file in self.project_root.rglob("*.py"):
            if not any(excluded in file.parts for excluded in self.excluded_dirs):
                yield file

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()