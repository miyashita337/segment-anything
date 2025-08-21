"""Weekly code cleanup and architecture optimization script.

This script performs automated cleanup tasks including:
- Detecting unused/deprecated files
- Checking architectural boundaries
- Optimizing dependencies
- Improving code quality
"""

from datetime import datetime, timedelta
import ast
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
import pip_autoremove

class CodebaseAnalyzer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deprecated_dir = project_root / 'deprecated'
        self.deprecated_dir.mkdir(exist_ok=True)
        
    def find_old_files(self, days_threshold: int = 30) -> List[Path]:
        """Find files not modified in specified number of days."""
        old_files = []
        threshold = datetime.now() - timedelta(days=days_threshold)
        
        for path in self.project_root.rglob('*.py'):
            if path.stat().st_mtime < threshold.timestamp():
                old_files.append(path)
        return old_files

    def analyze_imports(self) -> nx.DiGraph:
        """Build dependency graph from imports."""
        graph = nx.DiGraph()
        
        for path in self.project_root.rglob('*.py'):
            with open(path) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
                    
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        graph.add_edge(str(path), name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        graph.add_edge(str(path), node.module)
                        
        return graph

    def find_circular_imports(self) -> List[List[str]]:
        """Detect circular import chains."""
        graph = self.analyze_imports()
        return list(nx.simple_cycles(graph))

    def move_to_deprecated(self, files: List[Path]) -> None:
        """Move files to deprecated directory."""
        for file in files:
            rel_path = file.relative_to(self.project_root)
            target = self.deprecated_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target))

    def generate_report(self) -> Dict:
        """Generate analysis report."""
        old_files = self.find_old_files()
        circular_imports = self.find_circular_imports()
        
        return {
            'old_files': [str(f) for f in old_files],
            'circular_imports': circular_imports,
            'timestamp': datetime.now().isoformat()
        }

def main():
    project_root = Path(__file__).parent.parent.parent
    analyzer = CodebaseAnalyzer(project_root)
    
    # Run analysis
    report = analyzer.generate_report()
    
    # Move old files to deprecated
    old_files = [Path(f) for f in report['old_files']]
    analyzer.move_to_deprecated(old_files)
    
    # Save report
    report_path = project_root / 'reports' / f'cleanup_{datetime.now():%Y%m%d}.json'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()