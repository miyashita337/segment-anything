"""Validates GitHub Actions workflow configurations.

Provides utilities to verify workflow file syntax and run local workflow tests.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
import subprocess


class WorkflowValidator:
    """Validates GitHub Actions workflow files and configurations."""

    def __init__(self, workflow_dir: Path) -> None:
        """Initialize validator with workflow directory.

        Args:
            workflow_dir: Path to .github/workflows directory
        """
        self.workflow_dir = workflow_dir

    def validate_syntax(self, workflow_file: Path) -> bool:
        """Validate YAML syntax of workflow file.

        Args:
            workflow_file: Path to workflow YAML file

        Returns:
            bool: True if syntax is valid

        Raises:
            ValueError: If workflow file is invalid
        """
        try:
            with open(workflow_file) as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid workflow file {workflow_file}: {str(e)}")

    def run_workflow_test(self, workflow_name: str) -> bool:
        """Run workflow test using act tool.

        Args:
            workflow_name: Name of workflow to test

        Returns:
            bool: True if workflow test passes
        """
        result = subprocess.run(
            ["act", "-n", workflow_name],
            cwd=self.workflow_dir.parent.parent,
            capture_output=True,
            text=True
        )
        return result.returncode == 0