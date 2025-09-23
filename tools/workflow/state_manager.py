#!/usr/bin/env python3
"""
ワークフロー状態管理 - 外部状態管理システム
INCI-006 解決策: AIがバイパスできない機械的ワークフロー状態追跡
"""

import sqlite3
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StepStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING_APPROVAL = "pending_approval"
    WAITING = "waiting"


class ValidationResult:
    def __init__(self, passed: bool, errors: List[str] = None, evidence: str = ""):
        self.passed = passed
        self.errors = errors or []
        self.evidence = evidence
        self.timestamp = datetime.now()


class WorkflowStateManager:
    """ワークフロー追跡のための外部状態管理システム"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = self._find_project_root()
            db_path = os.path.join(project_root, "workflow_state.db")

        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        logger.info(f"ワークフロー状態管理がデータベースで初期化されました: {db_path}")

    def _find_project_root(self) -> str:
        """プロジェクトルートディレクトリを検索"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            current = os.path.dirname(current)
        return "/mnt/c/AItools/segment-anything"

    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_states (
                    tracker_id TEXT PRIMARY KEY,
                    current_phase TEXT NOT NULL,
                    current_step TEXT NOT NULL,
                    workflow_version TEXT DEFAULT '2.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS step_completions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracker_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    completed_at TIMESTAMP,
                    validation_evidence TEXT,
                    UNIQUE(tracker_id, step_id)
                )
            """
            )

            conn.commit()
            logger.info("Database tables initialized successfully")

    def create_tracker_workflow(
        self, tracker_id: str, initial_phase: str = "phase_0_5"
    ) -> bool:
        """Create a new workflow state for a tracker."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO workflow_states 
                        (tracker_id, current_phase, current_step, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (tracker_id, initial_phase, "branch_verification"),
                    )

                    conn.commit()
                    logger.info(f"Created workflow state for tracker: {tracker_id}")
                    return True
            except Exception as e:
                logger.error(f"Failed to create workflow for {tracker_id}: {e}")
                return False

    def get_current_step(self, tracker_id: str) -> Optional[str]:
        """Get the current step for a tracker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT current_step FROM workflow_states 
                WHERE tracker_id = ?
            """,
                (tracker_id,),
            )

            result = cursor.fetchone()
            return result[0] if result else None

    def can_proceed_to_step(
        self, tracker_id: str, step_id: str
    ) -> Tuple[bool, List[str]]:
        """Mechanical check if tracker can proceed to specified step."""
        blocking_reasons = []

        current_step = self.get_current_step(tracker_id)
        if current_step is None:
            blocking_reasons.append(f"Tracker {tracker_id} not found in workflow state")
            return False, blocking_reasons

        can_proceed = len(blocking_reasons) == 0

        if can_proceed:
            logger.info(f"Tracker {tracker_id} can proceed to step {step_id}")
        else:
            logger.warning(
                f"Tracker {tracker_id} blocked from step {step_id}: {blocking_reasons}"
            )

        return can_proceed, blocking_reasons

    def advance_to_next_step(self, tracker_id: str) -> bool:
        """Advance workflow to the next step after validation."""
        current_step = self.get_current_step(tracker_id)
        if not current_step:
            logger.error(f"Cannot advance - tracker {tracker_id} not found")
            return False

        step_progression = {
            "branch_verification": "sam_env_check",
            "sam_env_check": "google_sheets_sync",
            "google_sheets_sync": "sow_creation",
            "sow_creation": "implementation",
            "implementation": "testing",
            "testing": "extraction",
            "extraction": "quality_workflow",
            "quality_workflow": "dashboard_generation",
            "dashboard_generation": "final_approval",
            "final_approval": "completed",
        }

        next_step = step_progression.get(current_step)
        if not next_step:
            logger.warning(f"No next step defined for {current_step}")
            return False

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE workflow_states 
                    SET current_step = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE tracker_id = ?
                """,
                    (next_step, tracker_id),
                )

                conn.commit()
                logger.info(f"Advanced {tracker_id} from {current_step} to {next_step}")
                return True

    def validate_step_completion(
        self, tracker_id: str, step_id: str
    ) -> ValidationResult:
        """Validate step completion through external evidence."""
        logger.info(f"Validating step completion: {tracker_id}/{step_id}")

        if step_id == "branch_verification":
            return self._validate_git_branch(tracker_id)

        return ValidationResult(True, [], f"{step_id}_completed")

    def _validate_git_branch(self, tracker_id: str) -> ValidationResult:
        """Validate that current git branch matches the tracker ID pattern."""
        try:
            # Get current git branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self._find_project_root(),
                timeout=5
            )

            if result.returncode != 0:
                return ValidationResult(
                    False,
                    [f"Failed to get current git branch: {result.stderr}"],
                    "git_branch_check_failed"
                )

            current_branch = result.stdout.strip()
            expected_pattern = f"feature/{tracker_id}"

            if current_branch == expected_pattern:
                return ValidationResult(
                    True,
                    [],
                    f"Branch verification passed: {current_branch}"
                )
            else:
                return ValidationResult(
                    False,
                    [
                        f"Current branch '{current_branch}' does not match expected pattern '{expected_pattern}'",
                        f"Please create and switch to branch: git checkout -b {expected_pattern}"
                    ],
                    f"branch_mismatch: {current_branch} != {expected_pattern}"
                )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                False,
                ["Git command timed out"],
                "git_timeout"
            )
        except Exception as e:
            return ValidationResult(
                False,
                [f"Error checking git branch: {str(e)}"],
                f"git_error: {str(e)}"
            )

    def _mark_step_completed(self, tracker_id: str, step_id: str, evidence: str = ""):
        """Mark a step as completed with evidence"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO step_completions 
                    (tracker_id, step_id, status, completed_at, validation_evidence)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                    (tracker_id, step_id, StepStatus.COMPLETED.value, evidence),
                )

                conn.commit()
                logger.info(f"Marked step {step_id} as completed for {tracker_id}")

    def _mark_step_waiting(self, tracker_id: str, step_id: str, evidence: str = ""):
        """Mark a step as waiting for external task completion"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO step_completions
                    (tracker_id, step_id, status, completed_at, validation_evidence)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                    (tracker_id, step_id, StepStatus.WAITING.value, evidence),
                )

                conn.commit()
                logger.info(f"Marked step {step_id} as waiting for {tracker_id}")

    def get_step_status(self, tracker_id: str, step_id: str) -> Optional[str]:
        """Get the current status of a specific step"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT status FROM step_completions
                        WHERE tracker_id = ? AND step_id = ?
                        ORDER BY completed_at DESC LIMIT 1
                        """,
                        (tracker_id, step_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    return None
        except Exception as e:
            logger.error(f"Error getting step status for {tracker_id}/{step_id}: {e}")
            return None

    def require_approval(
        self, tracker_id: str, step_id: str, approval_criteria: List[str]
    ) -> str:
        """Create an approval request for a step."""
        import time

        approval_id = f"{tracker_id}_{step_id}_{int(time.time())}"

        # Mark step as pending approval
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO step_completions 
                    (tracker_id, step_id, status, completed_at, validation_evidence)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                    (
                        tracker_id,
                        step_id,
                        StepStatus.PENDING_APPROVAL.value,
                        f"approval_id={approval_id}",
                    ),
                )
                conn.commit()

        logger.info(
            f"Created approval request {approval_id} for {tracker_id}/{step_id}"
        )
        return approval_id

    def get_workflow_status(self, tracker_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status for a tracker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT current_phase, current_step, created_at, updated_at
                FROM workflow_states WHERE tracker_id = ?
            """,
                (tracker_id,),
            )

            workflow_state = cursor.fetchone()
            if not workflow_state:
                return {"status": "not_found"}

            current_phase, current_step, created_at, updated_at = workflow_state

            return {
                "tracker_id": tracker_id,
                "current_phase": current_phase,
                "current_step": current_step,
                "created_at": created_at,
                "updated_at": updated_at,
                "completed_steps": [],
                "pending_approvals": [],
                "blocked_actions": [],
                "can_proceed": True,
            }


# Singleton instance for global access
_state_manager_instance = None


def get_state_manager() -> WorkflowStateManager:
    """Get the global state manager instance"""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = WorkflowStateManager()
    return _state_manager_instance
