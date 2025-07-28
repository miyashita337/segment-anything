"""Tests for workflow validation utilities."""

import pytest
from pathlib import Path
from tools.workflow_validator import WorkflowValidator

@pytest.fixture
def workflow_dir(tmp_path):
    return tmp_path / ".github" / "workflows"

@pytest.fixture
def validator(workflow_dir):
    workflow_dir.parent.mkdir(parents=True)
    return WorkflowValidator(workflow_dir)

def test_validate_syntax_valid(validator, workflow_dir):
    workflow_file = workflow_dir / "test.yml"
    workflow_file.write_text("""
        name: Test
        on: push
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v2
    """)
    
    assert validator.validate_syntax(workflow_file)

def test_validate_syntax_invalid(validator, workflow_dir):
    workflow_file = workflow_dir / "invalid.yml" 
    workflow_file.write_text("invalid: [yaml")
    
    with pytest.raises(ValueError):
        validator.validate_syntax(workflow_file)

def test_run_workflow_test(validator, mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0
    
    assert validator.run_workflow_test("test")
    mock_run.assert_called_once()