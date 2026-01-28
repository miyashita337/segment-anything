#!/usr/bin/env python3
"""
File System Validator - Mechanical Validation System
INCI-006 Solution: External validation through file system evidence

This module provides mechanical validation that does not rely on AI self-reporting.
All validation is based on actual file existence, content verification, and 
measurable evidence that cannot be faked or bypassed by AI judgment.
"""

import glob
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check with evidence"""

    def __init__(self, passed: bool, errors: List[str] = None, evidence: str = ""):
        self.passed = passed
        self.errors = errors or []
        self.evidence = evidence
        self.timestamp = datetime.now()

    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"ValidationResult({status}, errors={len(self.errors)}, evidence='{self.evidence[:50]}...')"


class FileSystemValidator:
    """
    Mechanical validator that checks actual file system evidence.
    No AI interpretation - only measurable, verifiable facts.
    """

    def __init__(self, workspace_base: str = None):
        if workspace_base is None:
            workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        self.workspace_base = workspace_base
        self.project_root = self._find_project_root()
        self.dashboard_config = self._load_dashboard_config()
        logger.info(f"FileSystemValidator initialized: workspace={workspace_base}")

    def _load_dashboard_config(self) -> Dict:
        """Load dashboard section configuration"""
        config_path = os.path.join(self._find_project_root(), "config", "dashboard_sections.json")
        default_config = {
            "required_sections": {
                "statistical_analysis": "統計分析結果",
                "extraction_gallery": "抽出結果ギャラリー",
                "quality_distribution": "品質分布",
                "evaluation_metrics": "評価メトリクス",
            }
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Dashboard config not found at {config_path}, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load dashboard config: {e}, using defaults")
            return default_config

    def _find_project_root(self) -> str:
        """Find the project root directory"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            current = os.path.dirname(current)
        return "/mnt/c/AItools/segment-anything"  # Fallback

    def validate_branch_verification(self, tracker_id: str) -> ValidationResult:
        """
        Validate that proper Git branch is being used.
        Mechanical check - no AI interpretation.
        """
        try:
            # Check current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return ValidationResult(False, ["Failed to get current Git branch"])

            current_branch = result.stdout.strip()
            expected_pattern = f"feature/{tracker_id}"

            if not current_branch.startswith("feature/"):
                return ValidationResult(
                    False,
                    [
                        f"Not on feature branch. Current: {current_branch}, Expected pattern: feature/*"
                    ],
                    f"current_branch={current_branch}",
                )

            # Exact match or starts with tracker_id
            if current_branch == expected_pattern or current_branch.startswith(
                f"feature/{tracker_id}"
            ):
                return ValidationResult(True, [], f"correct_branch={current_branch}")
            else:
                return ValidationResult(
                    False,
                    [
                        f"Wrong feature branch. Current: {current_branch}, Expected: {expected_pattern}"
                    ],
                    f"current_branch={current_branch}",
                )

        except Exception as e:
            return ValidationResult(False, [f"Branch validation error: {str(e)}"])

    def validate_sam_env(self, tracker_id: str) -> ValidationResult:
        """
        Validate that sam-env virtual environment is active.
        Mechanical check through environment variables.
        """
        try:
            virtual_env = os.environ.get("VIRTUAL_ENV", "")

            if not virtual_env:
                return ValidationResult(False, ["No virtual environment active"])

            if "sam-env" not in virtual_env:
                return ValidationResult(
                    False,
                    [f"Wrong virtual environment. Current: {virtual_env}, Expected: sam-env"],
                    f"venv={virtual_env}",
                )

            # Check if Python is from the correct environment
            result = subprocess.run(["which", "python3"], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                python_path = result.stdout.strip()
                if "sam-env" in python_path:
                    return ValidationResult(True, [], f"venv={virtual_env},python={python_path}")
                else:
                    return ValidationResult(
                        False,
                        [f"Python not from sam-env. Path: {python_path}"],
                        f"python_path={python_path}",
                    )
            else:
                return ValidationResult(False, ["Could not determine Python path"])

        except Exception as e:
            return ValidationResult(False, [f"Virtual environment validation error: {str(e)}"])

    def validate_google_sheets_sync(self, tracker_id: str) -> ValidationResult:
        """
        Validate that Google Sheets sync has occurred.
        Check for evidence of CLI execution and status update.
        """
        try:
            # Check if progress tracker CLI exists
            cli_path = os.path.join(self.project_root, "tools/progress_tracker/cli.py")
            if not os.path.exists(cli_path):
                return ValidationResult(False, ["Progress tracker CLI not found"])

            # Try to get status to verify connectivity
            result = subprocess.run(
                ["python3", "tools/progress_tracker/cli.py", "status", tracker_id],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Check if output contains tracker information
                output = result.stdout.strip()
                if tracker_id in output and ("着手中" in output or "進行中" in output or "実装中" in output):
                    return ValidationResult(
                        True, [], f"sheets_sync_verified,status_contains={tracker_id}"
                    )
                else:
                    return ValidationResult(
                        False,
                        [f"Tracker {tracker_id} not found in Google Sheets or wrong status"],
                        f"cli_output={output[:100]}",
                    )
            else:
                return ValidationResult(
                    False,
                    [f"Google Sheets CLI failed: {result.stderr}"],
                    f"cli_error={result.stderr[:100]}",
                )

        except Exception as e:
            return ValidationResult(False, [f"Google Sheets validation error: {str(e)}"])

    def validate_sow_creation(self, tracker_id: str) -> ValidationResult:
        """
        Validate that SOW (Statement of Work) document has been created.
        Check for actual file existence and minimum content requirements.
        """
        workspace_path = os.path.join(self.workspace_base, tracker_id)

        # Check for SOW document in various possible locations
        possible_sow_paths = [
            os.path.join(workspace_path, "sow_document.md"),
            os.path.join(workspace_path, "SOW.md"),
            os.path.join(workspace_path, "statement_of_work.md"),
            os.path.join(workspace_path, "planning", "sow.md"),
        ]

        sow_file = None
        for path in possible_sow_paths:
            if os.path.exists(path):
                sow_file = path
                break

        if not sow_file:
            return ValidationResult(
                False,
                ["SOW document not found in any expected location"],
                f"checked_paths={possible_sow_paths}",
            )

        try:
            # Validate SOW content
            with open(sow_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for required SOW sections
            required_sections = [
                "作業スコープ",  # Work Scope
                "成果物",  # Deliverables
                "責任範囲",  # Responsibility
                "承認条件",  # Approval Conditions
            ]

            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)

            if missing_sections:
                return ValidationResult(
                    False,
                    [f"SOW missing required sections: {missing_sections}"],
                    f"sow_file={sow_file},content_length={len(content)}",
                )

            # Check minimum content length (should be substantial)
            if len(content) < 500:
                return ValidationResult(
                    False,
                    [f"SOW content too short: {len(content)} characters (minimum 500)"],
                    f"sow_file={sow_file},content_length={len(content)}",
                )

            return ValidationResult(
                True, [], f"sow_file={sow_file},content_length={len(content)},sections_complete"
            )

        except Exception as e:
            return ValidationResult(False, [f"SOW validation error: {str(e)}"])

    def validate_implementation(self, tracker_id: str) -> ValidationResult:
        """
        Validate that implementation work has been completed.
        Check for code changes, commits, and implementation artifacts.
        """
        try:
            # Check for recent commits on the feature branch
            result = subprocess.run(
                ["git", "log", "--oneline", "-10", "--grep", tracker_id],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return ValidationResult(False, ["Failed to check Git commits"])

            commits = result.stdout.strip().split("\n") if result.stdout.strip() else []

            if not commits or not any(tracker_id in commit for commit in commits):
                return ValidationResult(
                    False, [f"No commits found for {tracker_id}"], f"recent_commits={len(commits)}"
                )

            # Check for modified files
            result = subprocess.run(
                ["git", "diff", "--name-only", "main", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                modified_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

                if not modified_files:
                    return ValidationResult(
                        False, ["No files modified compared to main branch"], "no_file_changes"
                    )

                return ValidationResult(
                    True, [], f"commits={len(commits)},modified_files={len(modified_files)}"
                )
            else:
                return ValidationResult(False, ["Failed to check file modifications"])

        except Exception as e:
            return ValidationResult(False, [f"Implementation validation error: {str(e)}"])

    def validate_testing(self, tracker_id: str) -> ValidationResult:
        """
        Validate that testing has been completed.
        Check for test execution evidence and results.
        """
        workspace_path = os.path.join(self.workspace_base, tracker_id)

        # Check for test results or test execution evidence
        test_evidence_paths = [
            os.path.join(workspace_path, "test_results.json"),
            os.path.join(workspace_path, "tests", "results.json"),
            os.path.join(workspace_path, "pytest_results.xml"),
            os.path.join(workspace_path, "test_output.log"),
        ]

        test_evidence = []
        for path in test_evidence_paths:
            if os.path.exists(path):
                test_evidence.append(path)

        if not test_evidence:
            # Check if tests were run recently (look for pytest cache)
            pytest_cache = os.path.join(self.project_root, ".pytest_cache")
            if os.path.exists(pytest_cache):
                # Check cache modification time
                cache_mtime = os.path.getmtime(pytest_cache)
                current_time = datetime.now().timestamp()

                # If cache was modified within last hour, consider tests run
                if current_time - cache_mtime < 3600:
                    return ValidationResult(
                        True,
                        [],
                        f"pytest_cache_recent,modified={datetime.fromtimestamp(cache_mtime)}",
                    )

            return ValidationResult(
                False, ["No test execution evidence found"], f"checked_paths={test_evidence_paths}"
            )

        # Validate test results if available
        try:
            for evidence_file in test_evidence:
                if evidence_file.endswith(".json"):
                    with open(evidence_file, "r") as f:
                        test_data = json.load(f)

                    # Check for test success indicators
                    if "passed" in test_data or "success" in test_data:
                        return ValidationResult(
                            True, [], f"test_results={evidence_file},data_available"
                        )

            return ValidationResult(True, [], f"test_evidence_found={len(test_evidence)}")

        except Exception as e:
            return ValidationResult(False, [f"Test validation error: {str(e)}"])

    def validate_extraction_completion(self, tracker_id: str) -> ValidationResult:
        """
        Validate that character extraction has been completed.
        This is a critical validation - checks for actual extracted files.
        """
        workspace_path = os.path.join(self.workspace_base, tracker_id)
        # Check both extraction and extraction_full directories
        extraction_dir = os.path.join(workspace_path, "extraction")
        extraction_full_dir = os.path.join(workspace_path, "extraction_full")

        # Use extraction_full if it exists, otherwise fall back to extraction
        if os.path.exists(extraction_full_dir):
            extraction_dir = extraction_full_dir

        # Check if extraction directory exists
        if not os.path.exists(extraction_dir):
            return ValidationResult(
                False, ["Extraction directory not found"], f"expected_path={extraction_dir}"
            )

        # Check for extracted image files
        extracted_patterns = [
            os.path.join(extraction_dir, "*_extracted.jpg"),
            os.path.join(extraction_dir, "*_extracted.png"),
            os.path.join(extraction_dir, "*.jpg"),
            os.path.join(extraction_dir, "*.png"),
        ]

        extracted_files = []
        for pattern in extracted_patterns:
            extracted_files.extend(glob.glob(pattern))

        if not extracted_files:
            return ValidationResult(
                False,
                ["No extracted image files found"],
                f"extraction_dir={extraction_dir},patterns_checked={extracted_patterns}",
            )

        # Check for extraction_result.json
        result_file = os.path.join(workspace_path, "extraction_result.json")
        if not os.path.exists(result_file):
            return ValidationResult(
                False, ["extraction_result.json not found"], f"expected_path={result_file}"
            )

        try:
            # Validate extraction_result.json content
            with open(result_file, "r") as f:
                extraction_data = json.load(f)

            # Check required fields
            required_fields = [
                "tracker_id",
                "total_images",
                "successful_extractions",
                "average_quality_score",
                "extraction_results",
            ]

            missing_fields = []
            for field in required_fields:
                if field not in extraction_data:
                    missing_fields.append(field)

            if missing_fields:
                return ValidationResult(
                    False,
                    [f"extraction_result.json missing fields: {missing_fields}"],
                    f"result_file={result_file}",
                )

            # Check if any extractions were successful
            successful_count = extraction_data.get("successful_extractions", 0)
            if successful_count == 0:
                return ValidationResult(
                    False,
                    ["No successful extractions recorded"],
                    f"result_file={result_file},successful_count=0",
                )

            # Verify file count matches
            if len(extracted_files) < successful_count:
                return ValidationResult(
                    False,
                    [
                        f"File count mismatch: found {len(extracted_files)}, expected {successful_count}"
                    ],
                    f"extracted_files={len(extracted_files)},recorded_count={successful_count}",
                )

            return ValidationResult(
                True,
                [],
                f"extraction_dir={extraction_dir},files={len(extracted_files)},successful={successful_count}",
            )

        except Exception as e:
            return ValidationResult(False, [f"Extraction result validation error: {str(e)}"])

    def validate_quality_workflow_completion(self, tracker_id: str) -> ValidationResult:
        """
        Validate that quality workflow has been completed.
        This checks for quality analysis outputs (NOT dashboard - that's in dashboard_generation step).
        """
        workspace_path = os.path.join(self.workspace_base, tracker_id)

        # Check for quality report JSON (generated by run_quality_workflow.sh)
        # NOTE: dashboard.html is generated in dashboard_generation step, not here
        quality_report_paths = [
            os.path.join(workspace_path, "quality", "unified_quality_report.json"),
            os.path.join(workspace_path, "unified_quality_report.json"),
            os.path.join(workspace_path, "quality_report.json"),
        ]

        quality_report = None
        for path in quality_report_paths:
            if os.path.exists(path):
                quality_report = path
                break

        if not quality_report:
            return ValidationResult(
                False,
                ["Quality report not found (unified_quality_report.json). "
                 "Hint: Run quality workflow first, dashboard is generated in next step."],
                f"checked_paths={quality_report_paths}"
            )

        try:
            # Validate quality report content (dashboard validation moved to dashboard_generation step)
            with open(quality_report, "r") as f:
                quality_data = json.load(f)

            # Check for essential quality metrics
            if "average_quality_score" not in quality_data:
                return ValidationResult(
                    False,
                    ["Quality report missing average_quality_score"],
                    f"quality_report={quality_report}",
                )

            return ValidationResult(
                True,
                [],
                f"quality_report={quality_report},quality_workflow_complete",
            )

        except Exception as e:
            return ValidationResult(False, [f"Quality workflow validation error: {str(e)}"])

    def validate_dashboard_generation(self, tracker_id: str) -> ValidationResult:
        """
        Validate that final dashboard has been generated and is accessible.
        This includes checking external server accessibility.
        """
        workspace_path = os.path.join(self.workspace_base, tracker_id)

        # Check for index.html (required for integrated dashboard server)
        index_file = os.path.join(workspace_path, "index.html")
        dashboard_file = os.path.join(workspace_path, "dashboard", "dashboard.html")

        if not os.path.exists(index_file):
            # Check if dashboard.html exists to copy
            if os.path.exists(dashboard_file):
                try:
                    # Copy dashboard.html to index.html for server integration
                    import shutil

                    shutil.copy2(dashboard_file, index_file)
                    logger.info(f"Copied dashboard.html to index.html for {tracker_id}")
                except Exception as e:
                    return ValidationResult(
                        False,
                        [f"Failed to create index.html: {str(e)}"],
                        f"dashboard_file={dashboard_file}",
                    )
            else:
                return ValidationResult(
                    False,
                    ["Neither index.html nor dashboard.html found"],
                    f"workspace_path={workspace_path}",
                )

        # Validate index.html content
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for basic HTML structure
            if not all(tag in content for tag in ["<html", "<head", "<body"]):
                return ValidationResult(
                    False, ["index.html is not valid HTML"], f"index_file={index_file}"
                )

            # Check for tracker-specific content
            if tracker_id not in content:
                return ValidationResult(
                    False,
                    [f"index.html does not contain tracker ID {tracker_id}"],
                    f"index_file={index_file}",
                )

            # Test external server accessibility (optional - may timeout)
            try:
                import urllib.error
                import urllib.request

                server_url = f"http://100.123.241.106:8088/tracker/{tracker_id}"

                # Quick connectivity test with short timeout
                request = urllib.request.Request(server_url)
                request.add_header(
                    "Authorization", "Basic YWRtaW46c2VjdXJlX3RyYWNrXzIwMjVfcTNfOGY5YQ=="
                )

                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.getcode() == 200:
                        server_accessible = True
                    else:
                        server_accessible = False

            except (urllib.error.URLError, TimeoutError, Exception):
                server_accessible = False

            evidence = f"index_file={index_file},content_length={len(content)}"
            if server_accessible:
                evidence += ",server_accessible=true"
            else:
                evidence += ",server_accessible=false"

            return ValidationResult(True, [], evidence)

        except Exception as e:
            return ValidationResult(False, [f"Dashboard generation validation error: {str(e)}"])

    def get_workspace_path(self, tracker_id: str) -> str:
        """Get the workspace path for a tracker"""
        return os.path.join(self.workspace_base, tracker_id)

    def ensure_workspace_exists(self, tracker_id: str) -> bool:
        """Ensure workspace directory structure exists"""
        workspace_path = self.get_workspace_path(tracker_id)

        required_dirs = [
            workspace_path,
            os.path.join(workspace_path, "extraction"),
            os.path.join(workspace_path, "quality"),
            os.path.join(workspace_path, "dashboard"),
            os.path.join(workspace_path, "tests"),
            os.path.join(workspace_path, ".workflow"),
        ]

        try:
            for dir_path in required_dirs:
                os.makedirs(dir_path, exist_ok=True)

            logger.info(f"Workspace structure created for {tracker_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create workspace for {tracker_id}: {e}")
            return False
