"""Tests for workflow verification functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from tools.verify_workflow import WorkflowVerifier


@pytest.fixture
def workflow_verifier():
    return WorkflowVerifier()


def test_validate_workflow_files(workflow_verifier, tmp_path):
    workflow_yaml = """
    name: Test Workflow
    on: [push]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
    """
    
    with patch("builtins.open", mock_open(read_data=workflow_yaml)):
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("test.yml")]
            workflows = workflow_verifier.validate_workflow_files()
            assert len(workflows) == 1
            assert workflows[0]["name"] == "Test Workflow"


def test_verify_test_execution(workflow_verifier):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert workflow_verifier.verify_test_execution() is True

        mock_run.side_effect = subprocess.CalledProcessError(1, [])
        assert workflow_verifier.verify_test_execution() is False


def test_check_workflow_status(workflow_verifier):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "completed"
        assert workflow_verifier.check_workflow_status() == "completed"

        mock_run.side_effect = subprocess.CalledProcessError(1, [])
        assert workflow_verifier.check_workflow_status() is None