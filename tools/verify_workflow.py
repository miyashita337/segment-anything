"""Script to verify GitHub Actions workflow configuration and execution.

Provides utilities to validate workflow files and test execution status.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
import subprocess
import sys


class WorkflowVerifier:
    """Verifies GitHub Actions workflow configuration and execution."""

    def __init__(self, workflow_dir: str = ".github/workflows"):
        """Initialize workflow verifier.

        Args:
            workflow_dir: Directory containing workflow files
        """
        self.workflow_dir = Path(workflow_dir)

    def validate_workflow_files(self) -> List[Dict]:
        """Validate all workflow YAML files.

        Returns:
            List of parsed workflow configurations
        """
        workflows = []
        for file in self.workflow_dir.glob("*.yml"):
            try:
                with open(file) as f:
                    workflow = yaml.safe_load(f)
                    workflows.append(workflow)
            except yaml.YAMLError as e:
                print(f"Error parsing {file}: {e}")
                sys.exit(1)
        return workflows

    def verify_test_execution(self) -> bool:
        """Run test suite and verify execution.

        Returns:
            True if tests pass, False otherwise
        """
        try:
            subprocess.run(["pytest"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def check_workflow_status(self) -> Optional[str]:
        """Check status of latest workflow run.

        Returns:
            Workflow status or None if not found
        """
        try:
            result = subprocess.run(
                ["gh", "run", "list", "--limit", "1", "--json", "status"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return None


def main() -> None:
    """Main entry point for workflow verification."""
    verifier = WorkflowVerifier()
    
    print("Validating workflow files...")
    workflows = verifier.validate_workflow_files()
    print(f"Found {len(workflows)} valid workflow files")

    print("\nVerifying test execution...")
    if verifier.verify_test_execution():
        print("All tests passed")
    else:
        print("Test execution failed")
        sys.exit(1)

    print("\nChecking workflow status...")
    status = verifier.check_workflow_status()
    if status:
        print(f"Latest workflow status: {status}")
    else:
        print("Could not retrieve workflow status")


if __name__ == "__main__":
    main()