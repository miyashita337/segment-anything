#!/usr/bin/env python3
"""
Automatic Workflow Executor - Mechanical Execution System
INCI-006 Solution: Automatic execution of critical steps without AI judgment

This module executes critical workflow steps automatically, removing the
dependency on AI memory, attention, or judgment. Steps are executed
mechanically based on external validation and state management.
"""

import json
import logging
import os
import signal
import subprocess
import threading
import time
from config.workspace_config import get_workspace_config
from dataclasses import dataclass
from datetime import datetime

# SubAgent統合とワークスペース設定
from tools.queue.task_integration import TaskIntegration
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of an automatic execution"""

    success: bool
    message: str
    execution_time: float = 0.0
    process_id: Optional[int] = None
    log_file: Optional[str] = None
    evidence: str = ""


class AutomaticWorkflowExecutor:
    """
    Mechanical executor that runs critical workflow steps automatically
    without relying on AI judgment or intervention.
    """

    def __init__(self, state_manager=None, validator=None):
        self.state_manager = state_manager
        self.validator = validator
        self.project_root = self._find_project_root()
        self.workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        self.running_processes = {}  # Track background processes
        self.lock = threading.Lock()

        logger.info(f"AutomaticWorkflowExecutor initialized: {self.project_root}")

    def _find_project_root(self) -> str:
        """Find the project root directory"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            current = os.path.dirname(current)
        return "/mnt/c/AItools/segment-anything"  # Fallback

    def execute_extraction_step(
        self, tracker_id: str, input_dir: str = None, background: bool = True
    ) -> ExecutionResult:
        """
        Automatically execute character extraction without AI involvement.
        This is mechanical execution - no AI judgment required.
        """
        logger.info(f"Starting automatic extraction for {tracker_id}")

        # Verify prerequisites through state manager
        if self.state_manager:
            can_proceed, blocking_reasons = self.state_manager.can_proceed_to_step(
                tracker_id, "extraction"
            )
            if not can_proceed:
                return ExecutionResult(
                    False,
                    f"Prerequisites not met: {blocking_reasons}",
                    evidence=f"blocked_reasons={blocking_reasons}",
                )

        # Get workspace and verify structure
        workspace = os.path.join(self.workspace_base, tracker_id)
        if not os.path.exists(workspace):
            try:
                os.makedirs(workspace, exist_ok=True)
                os.makedirs(os.path.join(workspace, "extraction"), exist_ok=True)
            except Exception as e:
                return ExecutionResult(False, f"Failed to create workspace: {e}")

        # Get input directory from workspace config
        if input_dir is None:
            workspace_config = get_workspace_config()
            input_dir = workspace_config.get_input_directory(tracker_id)

            if not input_dir:
                return ExecutionResult(
                    False,
                    f"No input directory configured for {tracker_id}",
                    evidence=f"tracker_id={tracker_id}",
                )

        # Verify input directory exists
        if not os.path.exists(input_dir):
            return ExecutionResult(
                False, f"Input directory not found: {input_dir}", evidence=f"input_dir={input_dir}"
            )

        # Prepare extraction using SubAgent queue system
        output_dir = os.path.join(workspace, "extraction")
        log_file = os.path.join(workspace, "extraction.log")

        try:
            start_time = time.time()

            # Initialize TaskIntegration for SubAgent queue
            integration = TaskIntegration(tracker_id)

            # Submit extraction task to SubAgent queue
            task_id = integration.execute_extract_character(
                input_dir=input_dir, output_dir=output_dir, batch=True, max_files=None  # No limit
            )

            execution_time = time.time() - start_time

            logger.info(f"Extraction task submitted to SubAgent queue: {task_id}")

            # Start queue processing to actually execute the task
            integration.start_queue_processing()
            logger.info(f"SubAgent queue processing started for {tracker_id}")

            # Save task ID for later monitoring
            task_info_file = os.path.join(workspace, "subagent_task.json")
            task_info = {
                "task_id": task_id,
                "tracker_id": tracker_id,
                "step": "extraction",
                "submitted_at": datetime.now().isoformat(),
                "input_dir": input_dir,
                "output_dir": output_dir,
            }

            with open(task_info_file, "w") as f:
                json.dump(task_info, f, indent=2)

            # Log task submission and execution start
            with open(log_file, "w") as f:
                f.write(f"SubAgent extraction task submitted at {datetime.now()}\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Queue processing started: Yes\n")
                f.write(f"Input directory: {input_dir}\n")
                f.write(f"Output directory: {output_dir}\n")
                f.write(f"Tracker ID: {tracker_id}\n")

            return ExecutionResult(
                True,
                f"Extraction task started in SubAgent queue: {task_id}",
                execution_time=execution_time,
                evidence=f"task_id={task_id},log_file={log_file},task_info={task_info_file}",
            )

        except Exception as e:
            return ExecutionResult(
                False, f"Extraction execution error: {str(e)}", evidence=f"exception={str(e)}"
            )

    def execute_quality_workflow(self, tracker_id: str) -> ExecutionResult:
        """
        Automatically execute quality workflow with SubAgent task monitoring.
        This checks if extraction is complete and then runs quality analysis.
        """
        logger.info(f"Starting automatic quality workflow for {tracker_id}")

        # Check if SubAgent extraction task is completed
        extraction_task_completed = self._check_extraction_task_status(tracker_id)

        if not extraction_task_completed:
            # Mark as waiting state and return
            if self.state_manager:
                self.state_manager._mark_step_waiting(tracker_id, "quality_workflow")

            return ExecutionResult(
                True,  # Success but waiting
                "Waiting for extraction task to complete",
                evidence="extraction_task_in_progress",
            )

        # Verify extraction results exist
        if self.validator:
            extraction_result = self.validator.validate_extraction_completion(tracker_id)
            if not extraction_result.passed:
                return ExecutionResult(
                    False,
                    f"Extraction validation failed: {extraction_result.errors}",
                    evidence=f"extraction_validation_failed={extraction_result.errors}",
                )

        # Verify prerequisites through state manager
        if self.state_manager:
            can_proceed, blocking_reasons = self.state_manager.can_proceed_to_step(
                tracker_id, "quality_workflow"
            )
            if not can_proceed:
                return ExecutionResult(
                    False,
                    f"Prerequisites not met: {blocking_reasons}",
                    evidence=f"blocked_reasons={blocking_reasons}",
                )

        # Prepare quality workflow command
        workspace = os.path.join(self.workspace_base, tracker_id)
        log_file = os.path.join(workspace, "quality_workflow.log")

        cmd = ["./tools/scripts/run_quality_workflow.sh", tracker_id]

        try:
            start_time = time.time()

            # Run quality workflow with timeout
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    timeout=600,  # 10 minute timeout
                )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Validate quality workflow results
                if self.validator:
                    validation_result = self.validator.validate_quality_workflow_completion(
                        tracker_id
                    )
                    if validation_result.passed:
                        if self.state_manager:
                            self.state_manager._mark_step_completed(
                                tracker_id, "quality_workflow", validation_result.evidence
                            )

                        return ExecutionResult(
                            True,
                            "Quality workflow completed successfully",
                            execution_time=execution_time,
                            log_file=log_file,
                            evidence=validation_result.evidence,
                        )
                    else:
                        return ExecutionResult(
                            False,
                            f"Quality workflow validation failed: {validation_result.errors}",
                            execution_time=execution_time,
                            log_file=log_file,
                            evidence=f"validation_errors={validation_result.errors}",
                        )
                else:
                    return ExecutionResult(
                        True,
                        "Quality workflow completed (validation not available)",
                        execution_time=execution_time,
                        log_file=log_file,
                        evidence="no_validation_available",
                    )
            else:
                return ExecutionResult(
                    False,
                    f"Quality workflow failed with return code {result.returncode}",
                    execution_time=execution_time,
                    log_file=log_file,
                    evidence=f"return_code={result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False, "Quality workflow timeout (10 minutes)", evidence="timeout_10min"
            )
        except Exception as e:
            return ExecutionResult(
                False, f"Quality workflow execution error: {str(e)}", evidence=f"exception={str(e)}"
            )

    def _check_extraction_task_status(self, tracker_id: str) -> bool:
        """Check if SubAgent extraction task is completed"""
        try:
            workspace = os.path.join(self.workspace_base, tracker_id)

            # First check queue status for completion
            queue_status_file = os.path.join(workspace, "queue", "queue_status.json")
            if os.path.exists(queue_status_file):
                with open(queue_status_file, "r") as f:
                    queue_status = json.load(f)

                # Check if status is idle (task completed) or has last_completed_task
                if queue_status.get("status") == "idle":
                    logger.info(f"Queue is idle for {tracker_id} - task completed")
                    return True
                elif queue_status.get("status") == "task_completed":
                    logger.info(f"Task marked as completed in queue for {tracker_id}")
                    return True
                elif queue_status.get("status") == "task_running":
                    logger.info(f"Task still running for {tracker_id}")
                    return False

            # Check if task info file exists
            task_info_file = os.path.join(workspace, "subagent_task.json")
            if not os.path.exists(task_info_file):
                # If no task info, check if extraction has results
                extraction_dir = os.path.join(workspace, "extraction")
                if os.path.exists(extraction_dir):
                    extracted_files = [
                        f for f in os.listdir(extraction_dir) if f.endswith((".jpg", ".png"))
                    ]
                    if len(extracted_files) > 0:
                        logger.info(
                            f"Extraction results found for {tracker_id} - considering task completed"
                        )
                        return True
                logger.warning(f"No SubAgent task info found for {tracker_id}")
                return False

            # Load task information
            with open(task_info_file, "r") as f:
                task_info = json.load(f)

            task_id = task_info.get("task_id")
            if not task_id:
                logger.error(f"No task ID found in task info for {tracker_id}")
                return False

            # Check if extraction directory has results
            extraction_dir = os.path.join(workspace, "extraction")
            if not os.path.exists(extraction_dir):
                logger.info(f"Extraction directory not found for {tracker_id} - task still running")
                return False

            # Check if any extracted files exist
            extracted_files = [
                f for f in os.listdir(extraction_dir) if f.endswith((".jpg", ".png"))
            ]
            if len(extracted_files) == 0:
                logger.info(f"No extracted files found for {tracker_id} - task still running")
                return False

            # Check if extraction_result.json exists (completion indicator)
            result_file = os.path.join(extraction_dir, "extraction_result.json")
            if os.path.exists(result_file):
                logger.info(f"Extraction completed for {tracker_id} - result file found")
                return True

            # If we have extracted files but no result file, check file count
            if len(extracted_files) >= 1:
                logger.info(
                    f"Extraction appears completed for {tracker_id} - {len(extracted_files)} files found"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking extraction task status for {tracker_id}: {e}")
            return False

    def execute_dashboard_generation(self, tracker_id: str) -> ExecutionResult:
        """
        Automatically generate final dashboard and ensure server integration.
        """
        logger.info(f"Starting automatic dashboard generation for {tracker_id}")

        workspace = os.path.join(self.workspace_base, tracker_id)
        dashboard_dir = os.path.join(workspace, "dashboard")
        dashboard_file = os.path.join(dashboard_dir, "dashboard.html")
        index_file = os.path.join(workspace, "index.html")

        try:
            start_time = time.time()

            # Generate dashboard using unified dashboard system
            logger.info(f"Generating dashboard for {tracker_id}")
            dashboard_cmd = [
                "sam-env/bin/python3",
                "tools/scripts/unified_dashboard_wrapper.py",
                tracker_id,
                os.path.join(workspace, "extraction/"),
                workspace,
            ]

            dashboard_result = subprocess.run(
                dashboard_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,  # 5 minute timeout
            )

            if dashboard_result.returncode != 0:
                return ExecutionResult(
                    False,
                    f"Dashboard generation failed: {dashboard_result.stderr}",
                    evidence=f"dashboard_stderr={dashboard_result.stderr}",
                )

            # Verify dashboard.html was created
            if not os.path.exists(dashboard_file):
                return ExecutionResult(
                    False,
                    "Dashboard generation completed but dashboard.html not found",
                    evidence=f"dashboard_file={dashboard_file}",
                )

            # Copy dashboard.html to index.html for server integration
            import shutil

            shutil.copy2(dashboard_file, index_file)
            logger.info(f"Copied dashboard.html to index.html for server integration")

            # Validate dashboard content
            if self.validator:
                validation_result = self.validator.validate_dashboard_generation(tracker_id)
                if validation_result.passed:
                    if self.state_manager:
                        self.state_manager._mark_step_completed(
                            tracker_id, "dashboard_generation", validation_result.evidence
                        )

                    execution_time = time.time() - start_time

                    return ExecutionResult(
                        True,
                        "Dashboard generation completed successfully",
                        execution_time=execution_time,
                        evidence=validation_result.evidence,
                    )
                else:
                    return ExecutionResult(
                        False,
                        f"Dashboard validation failed: {validation_result.errors}",
                        evidence=f"validation_errors={validation_result.errors}",
                    )
            else:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    True,
                    "Dashboard generation completed (validation not available)",
                    execution_time=execution_time,
                    evidence=f"index_file={index_file}",
                )

        except Exception as e:
            return ExecutionResult(
                False, f"Dashboard generation error: {str(e)}", evidence=f"exception={str(e)}"
            )

    def check_process_status(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """Check status of background process for a tracker"""
        with self.lock:
            if tracker_id not in self.running_processes:
                return None

            process_info = self.running_processes[tracker_id]
            process = process_info["process"]

            # Check if process is still running
            poll_result = process.poll()

            if poll_result is None:
                # Still running
                return {
                    "status": "running",
                    "pid": process.pid,
                    "step": process_info["step"],
                    "started_at": process_info["started_at"].isoformat(),
                    "log_file": process_info["log_file"],
                }
            else:
                # Process completed
                completed_info = {
                    "status": "completed",
                    "return_code": poll_result,
                    "pid": process.pid,
                    "step": process_info["step"],
                    "started_at": process_info["started_at"].isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "log_file": process_info["log_file"],
                }

                # Remove from running processes
                del self.running_processes[tracker_id]

                # Validate completion if successful
                if poll_result == 0 and self.validator and process_info["step"] == "extraction":
                    validation_result = self.validator.validate_extraction_completion(tracker_id)
                    if validation_result.passed and self.state_manager:
                        self.state_manager._mark_step_completed(
                            tracker_id, "extraction", validation_result.evidence
                        )
                        completed_info["validation"] = "passed"
                    else:
                        completed_info["validation"] = "failed"
                        completed_info["validation_errors"] = validation_result.errors

                return completed_info

    def wait_for_process_completion(
        self, tracker_id: str, timeout_minutes: int = 60
    ) -> ExecutionResult:
        """
        Wait for background process to complete with timeout.
        This provides synchronous waiting for asynchronous processes.
        """
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds

        while time.time() - start_time < timeout_minutes * 60:
            status = self.check_process_status(tracker_id)

            if status is None:
                return ExecutionResult(
                    False, f"No running process found for {tracker_id}", evidence="no_process_found"
                )

            if status["status"] == "completed":
                execution_time = time.time() - start_time

                if status["return_code"] == 0:
                    return ExecutionResult(
                        True,
                        f"Process completed successfully (return code: {status['return_code']})",
                        execution_time=execution_time,
                        process_id=status["pid"],
                        log_file=status["log_file"],
                        evidence=f"return_code={status['return_code']}",
                    )
                else:
                    return ExecutionResult(
                        False,
                        f"Process failed (return code: {status['return_code']})",
                        execution_time=execution_time,
                        process_id=status["pid"],
                        log_file=status["log_file"],
                        evidence=f"return_code={status['return_code']}",
                    )

            # Still running - wait
            remaining_minutes = timeout_minutes - int((time.time() - start_time) / 60)
            logger.info(
                f"Waiting for {tracker_id} process completion ({remaining_minutes} minutes remaining)"
            )
            time.sleep(check_interval)

        # Timeout - try to terminate process
        self.terminate_process(tracker_id)
        return ExecutionResult(
            False,
            f"Process timeout after {timeout_minutes} minutes",
            evidence=f"timeout_{timeout_minutes}min",
        )

    def terminate_process(self, tracker_id: str) -> bool:
        """Terminate background process for a tracker"""
        with self.lock:
            if tracker_id not in self.running_processes:
                return False

            process_info = self.running_processes[tracker_id]
            process = process_info["process"]

            try:
                # Try graceful termination first
                process.terminate()

                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination failed
                    process.kill()
                    process.wait()

                logger.info(f"Terminated process for {tracker_id} (PID: {process.pid})")
                del self.running_processes[tracker_id]
                return True

            except Exception as e:
                logger.error(f"Error terminating process for {tracker_id}: {e}")
                return False

    def get_running_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all running processes"""
        with self.lock:
            running_info = {}

            for tracker_id, process_info in self.running_processes.items():
                status = self.check_process_status(tracker_id)
                if status and status["status"] == "running":
                    running_info[tracker_id] = status

            return running_info

    def execute_subagent_extraction_step(self, tracker_id: str, **kwargs) -> ExecutionResult:
        """Execute SubAgent extraction step"""
        logger.info(f"Starting SubAgent extraction for {tracker_id}")

        try:
            # Import SubAgent command handler
            from tools.workflow.subagent_command_handler import SubAgentCommandHandler

            handler = SubAgentCommandHandler()
            success = handler.handle_subagent_extraction(tracker_id)

            if success:
                # SubAgent started successfully - state will be managed by monitor
                logger.info(f"SubAgent extraction started for {tracker_id}")

                return ExecutionResult(
                    True,
                    "SubAgent extraction started successfully",
                    evidence=f"tracker_id={tracker_id},subagent_started=true",
                )
            else:
                return ExecutionResult(
                    False,
                    "Failed to start SubAgent extraction",
                    evidence=f"tracker_id={tracker_id},subagent_started=false",
                )

        except Exception as e:
            return ExecutionResult(
                False,
                f"SubAgent extraction error: {str(e)}",
                evidence=f"tracker_id={tracker_id},exception={str(e)}",
            )

    def execute_subagent_validation_step(self, tracker_id: str, **kwargs) -> ExecutionResult:
        """Execute SubAgent validation step"""
        logger.info(f"Validating SubAgent results for {tracker_id}")

        try:
            # Import SubAgent monitor for result validation
            from config.workspace_config import WorkspaceConfig
            from pathlib import Path
            from tools.workflow.subagent_monitor import SubAgentMonitor, SubAgentStatus

            monitor = SubAgentMonitor()
            workspace_base = WorkspaceConfig.get_workspace_base()
            extraction_dir = Path(workspace_base) / tracker_id / "extraction"

            # Primary validation: Check if extraction files exist
            if extraction_dir.exists():
                extracted_files = list(extraction_dir.glob("*.jpg")) + list(
                    extraction_dir.glob("*.png")
                )
                if len(extracted_files) > 0:
                    # Files exist, validation successful
                    return ExecutionResult(
                        True,
                        f"SubAgent validation successful: {len(extracted_files)} files extracted",
                        evidence=f"tracker_id={tracker_id},files_count={len(extracted_files)}",
                    )

            # Secondary validation: Check process status for diagnostic information
            process_status = monitor.check_subagent_status(tracker_id, "extraction")

            if process_status == SubAgentStatus.NOT_STARTED:
                return ExecutionResult(
                    False,
                    "No SubAgent process found for validation",
                    evidence=f"tracker_id={tracker_id},process_found=false",
                )

            # No files extracted, process status indicates failure
            if process_status == SubAgentStatus.FAILED:
                return ExecutionResult(
                    False,
                    f"SubAgent process failed: {process_status.value}",
                    evidence=f"tracker_id={tracker_id},status={process_status.value},files_count=0",
                )

            # Process still running or other states
            if process_status == SubAgentStatus.RUNNING:
                return ExecutionResult(
                    False,
                    "SubAgent process still running, validation pending",
                    evidence=f"tracker_id={tracker_id},status={process_status.value}",
                )

            # Other process states without extracted files
            return ExecutionResult(
                False,
                f"SubAgent validation failed: no files extracted, status={process_status.value}",
                evidence=f"tracker_id={tracker_id},status={process_status.value},files_count=0",
            )

        except Exception as e:
            return ExecutionResult(
                False,
                f"SubAgent validation error: {str(e)}",
                evidence=f"tracker_id={tracker_id},exception={str(e)}",
            )

    def execute_step_automatically(
        self, tracker_id: str, step_id: str, **kwargs
    ) -> ExecutionResult:
        """
        Execute any step automatically based on step_id.
        This is the main entry point for automatic execution.
        """
        logger.info(f"Executing step automatically: {tracker_id}/{step_id}")

        # Map steps to execution methods
        step_executors = {
            "extraction": self.execute_extraction_step,
            "quality_workflow": self.execute_quality_workflow,
            "dashboard_generation": self.execute_dashboard_generation,
            "subagent_extraction": self.execute_subagent_extraction_step,
            "subagent_validation": self.execute_subagent_validation_step,
        }

        executor = step_executors.get(step_id)
        if not executor:
            return ExecutionResult(
                False,
                f"No automatic executor available for step: {step_id}",
                evidence=f"step_id={step_id}",
            )

        try:
            return executor(tracker_id, **kwargs)
        except Exception as e:
            return ExecutionResult(
                False,
                f"Automatic execution error for {step_id}: {str(e)}",
                evidence=f"step_id={step_id},exception={str(e)}",
            )


# Convenience function for getting executor instance
def get_automatic_executor(state_manager=None, validator=None) -> AutomaticWorkflowExecutor:
    """Get automatic executor instance with dependencies"""
    if state_manager is None:
        try:
            import sys

            current_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from tools.workflow.state_manager import get_state_manager

            state_manager = get_state_manager()
        except ImportError:
            logger.warning("State manager not available")

    if validator is None:
        try:
            from tools.validation.file_system_validator import FileSystemValidator

            validator = FileSystemValidator()
        except ImportError:
            logger.warning("File system validator not available")

    return AutomaticWorkflowExecutor(state_manager, validator)
