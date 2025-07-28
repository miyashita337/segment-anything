"""Script to verify GitHub Actions workflow configuration and execution.

Provides utilities to validate workflow files and test execution status.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
import subprocess
import sys


class WorkflowVerifier:
    def __init__(self, workflow_dir: Path) -> None:
        self.workflow_dir = workflow_dir
        self.workflows: Dict[str, dict] = {}

    def load_workflows(self) -> None:
        """Load all workflow YAML files from .github/workflows directory."""
        for workflow_file in self.workflow_dir.glob("*.yml"):
            with open(workflow_file) as f:
                self.workflows[workflow_file.name] = yaml.safe_load(f)

    def verify_workflow_syntax(self) -> List[str]:
        """Verify syntax of all workflow files.

        Returns:
            List[str]: List of any syntax errors found
        """
        errors = []
        for name, workflow in self.workflows.items():
            try:
                yaml.dump(workflow)
            except yaml.YAMLError as e:
                errors.append(f"Syntax error in {name}: {str(e)}")
        return errors

    def verify_test_execution(self) -> bool:
        """Run test suite to verify workflow test execution.

        Returns:
            bool: True if all tests pass, False otherwise
        """
        try:
            subprocess.run(["pytest"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def verify_script_organization(self) -> List[str]:
        """Verify script organization matches project structure.

        Returns:
            List[str]: List of any organization issues found
        """
        issues = []
        required_dirs = ["core", "features", "tools", "tests"]
        for dir_name in required_dirs:
            if not (Path.cwd() / dir_name).is_dir():
                issues.append(f"Missing required directory: {dir_name}")
        return issues


def main() -> None:
    workflow_dir = Path.cwd() / ".github" / "workflows"
    if not workflow_dir.exists():
        print("Error: .github/workflows directory not found")
        sys.exit(1)

    verifier = WorkflowVerifier(workflow_dir)
    verifier.load_workflows()

    syntax_errors = verifier.verify_workflow_syntax()
    if syntax_errors:
        print("Workflow syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        sys.exit(1)

    org_issues = verifier.verify_script_organization()
    if org_issues:
        print("Script organization issues found:")
        for issue in org_issues:
            print(f"  {issue}")
        sys.exit(1)

    if not verifier.verify_test_execution():
        print("Test execution failed")
        sys.exit(1)

    print("Workflow verification completed successfully")


if __name__ == "__main__":
    main()