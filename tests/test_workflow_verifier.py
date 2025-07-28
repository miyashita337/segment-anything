"""Tests for workflow verification functionality."""

import pytest
from pathlib import Path
from tools.verify_workflow import WorkflowVerifier


@pytest.fixture
def workflow_dir(tmp_path):
    workflow_path = tmp_path / ".github" / "workflows"
    workflow_path.mkdir(parents=True)
    return workflow_path


@pytest.fixture
def valid_workflow(workflow_dir):
    content = """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pytest
"""
    workflow_file = workflow_dir / "ci.yml"
    workflow_file.write_text(content)
    return workflow_file


def test_load_workflows(workflow_dir, valid_workflow):
    verifier = WorkflowVerifier(workflow_dir)
    verifier.load_workflows()
    assert "ci.yml" in verifier.workflows
    assert verifier.workflows["ci.yml"]["name"] == "CI"


def test_verify_workflow_syntax_valid(workflow_dir, valid_workflow):
    verifier = WorkflowVerifier(workflow_dir)
    verifier.load_workflows()
    errors = verifier.verify_workflow_syntax()
    assert not errors


def test_verify_workflow_syntax_invalid(workflow_dir):
    invalid_content = "invalid: ["
    workflow_file = workflow_dir / "invalid.yml"
    workflow_file.write_text(invalid_content)

    verifier = WorkflowVerifier(workflow_dir)
    verifier.load_workflows()
    errors = verifier.verify_workflow_syntax()
    assert len(errors) == 1
    assert "invalid.yml" in errors[0]


def test_verify_script_organization(tmp_path):
    # Create required directories
    for dir_name in ["core", "features", "tools", "tests"]:
        (tmp_path / dir_name).mkdir()

    verifier = WorkflowVerifier(tmp_path / ".github" / "workflows")
    issues = verifier.verify_script_organization()
    assert not issues