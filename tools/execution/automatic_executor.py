#!/usr/bin/env python3
"""
Automatic Workflow Executor - Mechanical Execution System
INCI-006 Solution: Automatic execution of critical steps without AI judgment

This module executes critical workflow steps automatically, removing the
dependency on AI memory, attention, or judgment. Steps are executed
mechanically based on external validation and state management.
"""

import os
import subprocess
import json
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import threading

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
    
    def execute_extraction_step(self, tracker_id: str, 
                               input_dir: str = None, 
                               background: bool = True) -> ExecutionResult:
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
                    evidence=f"blocked_reasons={blocking_reasons}"
                )
        
        # Get workspace and verify structure
        workspace = os.path.join(self.workspace_base, tracker_id)
        if not os.path.exists(workspace):
            try:
                os.makedirs(workspace, exist_ok=True)
                os.makedirs(os.path.join(workspace, "extraction"), exist_ok=True)
            except Exception as e:
                return ExecutionResult(False, f"Failed to create workspace: {e}")
        
        # Set default input directory
        if input_dir is None:
            input_dir = "/mnt/c/AItools/lora/train/yado/org/kana05/"
        
        # Verify input directory exists
        if not os.path.exists(input_dir):
            return ExecutionResult(
                False, 
                f"Input directory not found: {input_dir}",
                evidence=f"input_dir={input_dir}"
            )
        
        # Prepare extraction command
        output_dir = os.path.join(workspace, "extraction")
        log_file = os.path.join(workspace, "extraction.log")
        
        cmd = [
            "python3", "features/extraction/commands/extract_character.py",
            input_dir, "-o", output_dir, "--batch", "--verbose"
        ]
        
        # Set environment variables for extraction
        env = os.environ.copy()
        env["MEMORY_LIMIT_DISABLED"] = "true"
        env["TRACKER_ID"] = tracker_id
        
        try:
            start_time = time.time()
            
            if background:
                # Run in background to avoid timeout
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd, 
                        stdout=f, 
                        stderr=subprocess.STDOUT,
                        cwd=self.project_root,
                        env=env
                    )
                
                # Track the process
                with self.lock:
                    self.running_processes[tracker_id] = {
                        'process': process,
                        'step': 'extraction',
                        'started_at': datetime.now(),
                        'log_file': log_file
                    }
                
                # Update state manager
                if self.state_manager:
                    self.state_manager._mark_step_in_progress(tracker_id, "extraction")
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    True, 
                    f"Extraction started in background (PID: {process.pid})",
                    execution_time=execution_time,
                    process_id=process.pid,
                    log_file=log_file,
                    evidence=f"pid={process.pid},log_file={log_file}"
                )
            
            else:
                # Run synchronously with timeout
                with open(log_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=self.project_root,
                        env=env,
                        timeout=1800  # 30 minute timeout
                    )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    # Validate extraction results
                    if self.validator:
                        validation_result = self.validator.validate_extraction_completion(tracker_id)
                        if validation_result.passed:
                            if self.state_manager:
                                self.state_manager._mark_step_completed(
                                    tracker_id, "extraction", validation_result.evidence
                                )
                            
                            return ExecutionResult(
                                True,
                                "Extraction completed successfully",
                                execution_time=execution_time,
                                log_file=log_file,
                                evidence=validation_result.evidence
                            )
                        else:
                            return ExecutionResult(
                                False,
                                f"Extraction validation failed: {validation_result.errors}",
                                execution_time=execution_time,
                                log_file=log_file,
                                evidence=f"validation_errors={validation_result.errors}"
                            )
                    else:
                        return ExecutionResult(
                            True,
                            "Extraction completed (validation not available)",
                            execution_time=execution_time,
                            log_file=log_file,
                            evidence="no_validation_available"
                        )
                else:
                    return ExecutionResult(
                        False,
                        f"Extraction failed with return code {result.returncode}",
                        execution_time=execution_time,
                        log_file=log_file,
                        evidence=f"return_code={result.returncode}"
                    )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False, 
                "Extraction timeout (30 minutes)",
                evidence="timeout_30min"
            )
        except Exception as e:
            return ExecutionResult(
                False, 
                f"Extraction execution error: {str(e)}",
                evidence=f"exception={str(e)}"
            ) 
   
    def execute_quality_workflow(self, tracker_id: str) -> ExecutionResult:
        """
        Automatically execute quality workflow without AI involvement.
        This runs the complete quality analysis and dashboard generation.
        """
        logger.info(f"Starting automatic quality workflow for {tracker_id}")
        
        # Verify extraction completed first
        if self.validator:
            extraction_result = self.validator.validate_extraction_completion(tracker_id)
            if not extraction_result.passed:
                return ExecutionResult(
                    False, 
                    f"Extraction not complete: {extraction_result.errors}",
                    evidence=f"extraction_validation_failed={extraction_result.errors}"
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
                    evidence=f"blocked_reasons={blocking_reasons}"
                )
        
        # Prepare quality workflow command
        workspace = os.path.join(self.workspace_base, tracker_id)
        log_file = os.path.join(workspace, "quality_workflow.log")
        
        cmd = ["./tools/scripts/run_quality_workflow.sh", tracker_id]
        
        try:
            start_time = time.time()
            
            # Run quality workflow with timeout
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    timeout=600  # 10 minute timeout
                )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Validate quality workflow results
                if self.validator:
                    validation_result = self.validator.validate_quality_workflow_completion(tracker_id)
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
                            evidence=validation_result.evidence
                        )
                    else:
                        return ExecutionResult(
                            False,
                            f"Quality workflow validation failed: {validation_result.errors}",
                            execution_time=execution_time,
                            log_file=log_file,
                            evidence=f"validation_errors={validation_result.errors}"
                        )
                else:
                    return ExecutionResult(
                        True,
                        "Quality workflow completed (validation not available)",
                        execution_time=execution_time,
                        log_file=log_file,
                        evidence="no_validation_available"
                    )
            else:
                return ExecutionResult(
                    False,
                    f"Quality workflow failed with return code {result.returncode}",
                    execution_time=execution_time,
                    log_file=log_file,
                    evidence=f"return_code={result.returncode}"
                )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False, 
                "Quality workflow timeout (10 minutes)",
                evidence="timeout_10min"
            )
        except Exception as e:
            return ExecutionResult(
                False, 
                f"Quality workflow execution error: {str(e)}",
                evidence=f"exception={str(e)}"
            )
    
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
            
            # Check if dashboard.html exists
            if not os.path.exists(dashboard_file):
                return ExecutionResult(
                    False,
                    "Dashboard file not found - quality workflow may not have completed",
                    evidence=f"dashboard_file={dashboard_file}"
                )
            
            # Copy dashboard.html to index.html for server integration
            import shutil
            shutil.copy2(dashboard_file, index_file)
            
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
                        evidence=validation_result.evidence
                    )
                else:
                    return ExecutionResult(
                        False,
                        f"Dashboard validation failed: {validation_result.errors}",
                        evidence=f"validation_errors={validation_result.errors}"
                    )
            else:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    True,
                    "Dashboard generation completed (validation not available)",
                    execution_time=execution_time,
                    evidence=f"index_file={index_file}"
                )
        
        except Exception as e:
            return ExecutionResult(
                False,
                f"Dashboard generation error: {str(e)}",
                evidence=f"exception={str(e)}"
            )
    
    def check_process_status(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """Check status of background process for a tracker"""
        with self.lock:
            if tracker_id not in self.running_processes:
                return None
            
            process_info = self.running_processes[tracker_id]
            process = process_info['process']
            
            # Check if process is still running
            poll_result = process.poll()
            
            if poll_result is None:
                # Still running
                return {
                    'status': 'running',
                    'pid': process.pid,
                    'step': process_info['step'],
                    'started_at': process_info['started_at'].isoformat(),
                    'log_file': process_info['log_file']
                }
            else:
                # Process completed
                completed_info = {
                    'status': 'completed',
                    'return_code': poll_result,
                    'pid': process.pid,
                    'step': process_info['step'],
                    'started_at': process_info['started_at'].isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'log_file': process_info['log_file']
                }
                
                # Remove from running processes
                del self.running_processes[tracker_id]
                
                # Validate completion if successful
                if poll_result == 0 and self.validator and process_info['step'] == 'extraction':
                    validation_result = self.validator.validate_extraction_completion(tracker_id)
                    if validation_result.passed and self.state_manager:
                        self.state_manager._mark_step_completed(
                            tracker_id, "extraction", validation_result.evidence
                        )
                        completed_info['validation'] = 'passed'
                    else:
                        completed_info['validation'] = 'failed'
                        completed_info['validation_errors'] = validation_result.errors
                
                return completed_info
    
    def wait_for_process_completion(self, tracker_id: str, timeout_minutes: int = 60) -> ExecutionResult:
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
                    False,
                    f"No running process found for {tracker_id}",
                    evidence="no_process_found"
                )
            
            if status['status'] == 'completed':
                execution_time = time.time() - start_time
                
                if status['return_code'] == 0:
                    return ExecutionResult(
                        True,
                        f"Process completed successfully (return code: {status['return_code']})",
                        execution_time=execution_time,
                        process_id=status['pid'],
                        log_file=status['log_file'],
                        evidence=f"return_code={status['return_code']}"
                    )
                else:
                    return ExecutionResult(
                        False,
                        f"Process failed (return code: {status['return_code']})",
                        execution_time=execution_time,
                        process_id=status['pid'],
                        log_file=status['log_file'],
                        evidence=f"return_code={status['return_code']}"
                    )
            
            # Still running - wait
            remaining_minutes = timeout_minutes - int((time.time() - start_time) / 60)
            logger.info(f"Waiting for {tracker_id} process completion ({remaining_minutes} minutes remaining)")
            time.sleep(check_interval)
        
        # Timeout - try to terminate process
        self.terminate_process(tracker_id)
        return ExecutionResult(
            False,
            f"Process timeout after {timeout_minutes} minutes",
            evidence=f"timeout_{timeout_minutes}min"
        )
    
    def terminate_process(self, tracker_id: str) -> bool:
        """Terminate background process for a tracker"""
        with self.lock:
            if tracker_id not in self.running_processes:
                return False
            
            process_info = self.running_processes[tracker_id]
            process = process_info['process']
            
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
                if status and status['status'] == 'running':
                    running_info[tracker_id] = status
            
            return running_info
    
    def execute_step_automatically(self, tracker_id: str, step_id: str, **kwargs) -> ExecutionResult:
        """
        Execute any step automatically based on step_id.
        This is the main entry point for automatic execution.
        """
        logger.info(f"Executing step automatically: {tracker_id}/{step_id}")
        
        # Map steps to execution methods
        step_executors = {
            'extraction': self.execute_extraction_step,
            'quality_workflow': self.execute_quality_workflow,
            'dashboard_generation': self.execute_dashboard_generation
        }
        
        executor = step_executors.get(step_id)
        if not executor:
            return ExecutionResult(
                False,
                f"No automatic executor available for step: {step_id}",
                evidence=f"step_id={step_id}"
            )
        
        try:
            return executor(tracker_id, **kwargs)
        except Exception as e:
            return ExecutionResult(
                False,
                f"Automatic execution error for {step_id}: {str(e)}",
                evidence=f"step_id={step_id},exception={str(e)}"
            )

# Convenience function for getting executor instance
def get_automatic_executor(state_manager=None, validator=None) -> AutomaticWorkflowExecutor:
    """Get automatic executor instance with dependencies"""
    if state_manager is None:
        try:
            import sys
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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