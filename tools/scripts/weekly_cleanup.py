#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Set
import logging
import sys

from features.cleanup.file_analyzer import FileAnalyzer
from features.cleanup.dependency_checker import DependencyChecker
from features.cleanup.usage_stats import UsageStatistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebaseCleanup:
    """Main cleanup orchestrator that manages the codebase optimization process."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.file_analyzer = FileAnalyzer(project_root)
        self.dep_checker = DependencyChecker(project_root)
        self.usage_stats = UsageStatistics(project_root)

    def run_cleanup(self) -> Dict[str, List[Path]]:
        """Execute the cleanup process and return results."""
        try:
            # Analyze files
            outdated = self.file_analyzer.find_outdated_files()
            unused = self.file_analyzer.find_unused_files()
            duplicates = self.file_analyzer.find_duplicate_files()

            # Check dependencies
            circular_deps = self.dep_checker.find_circular_imports()
            unused_deps = self.dep_checker.find_unused_packages()

            # Generate statistics
            self.usage_stats.generate_report()

            return {
                "outdated": outdated,
                "unused": unused,
                "duplicates": duplicates,
                "circular_deps": circular_deps,
                "unused_deps": unused_deps
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

def main():
    project_root = Path(__file__).parent.parent.parent
    cleanup = CodebaseCleanup(project_root)
    
    try:
        results = cleanup.run_cleanup()
        logger.info("Cleanup completed successfully")
        for category, items in results.items():
            logger.info(f"{category}: {len(items)} items found")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()