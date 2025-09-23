#!/usr/bin/env python3
"""
ワークフローコントローラー - ワークフロー強制実行の統合インターフェース
INCI-006 解決策: すべての強制実行コンポーネントを統合する単一インターフェース

このモジュールは、状態管理、検証、承認、自動実行を単一の一貫した
システムに組み合わせ、ワークフローコンプライアンスを機械的に
強制実行する統合インターフェースを提供します。
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """ワークフローステップの定義"""
    step_id: str
    phase: str
    title: str
    description: str
    prerequisites: List[str]
    validation_required: bool
    approval_required: bool
    auto_executable: bool
    approval_criteria: List[str] = None

@dataclass
class StepInstructions:
    """AI用の現在のステップ指示"""
    step_id: str
    title: str
    description: str
    required_actions: List[str]
    validation_criteria: List[str]
    approval_required: bool
    can_proceed: bool
    blocking_reasons: List[str] = None

class StepResult:
    """ステップ実行試行の結果"""
    def __init__(self, status: str, message: str, next_step: str = None, 
                 approval_id: str = None, evidence: str = ""):
        self.status = status  # completed, failed, pending_approval, blocked
        self.message = message
        self.next_step = next_step
        self.approval_id = approval_id
        self.evidence = evidence
    
    @classmethod
    def COMPLETED(cls, next_step: str = None):
        return cls("completed", "Step completed successfully", next_step)
    
    @classmethod
    def FAILED(cls, errors: List[str]):
        return cls("failed", f"Step failed: {'; '.join(errors)}")
    
    @classmethod
    def PENDING_APPROVAL(cls, approval_id: str):
        return cls("pending_approval", f"Waiting for approval: {approval_id}", approval_id=approval_id)
    
    @classmethod
    def BLOCKED(cls, reasons: List[str]):
        return cls("blocked", f"Step blocked: {'; '.join(reasons)}")

    @classmethod
    def WAITING(cls, reasons: List[str]):
        return cls("waiting", f"Step waiting: {'; '.join(reasons)}")

class WorkflowController:
    """
    Unified workflow controller that enforces compliance through
    mechanical state management, validation, and approval systems.
    """
    
    def __init__(self):
        # Initialize all enforcement components
        self.state_manager = self._init_state_manager()
        self.validator = self._init_validator()
        self.approval_controller = self._init_approval_controller()
        self.executor = self._init_executor()
        
        # Load workflow configuration
        self.workflow_config = self._load_workflow_config()
        
        logger.info("WorkflowController initialized with all enforcement components")
    
    def _init_state_manager(self):
        """Initialize state manager"""
        try:
            # Add current directory to Python path
            import sys
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from tools.workflow.state_manager import get_state_manager
            return get_state_manager()
        except ImportError as e:
            logger.error(f"Failed to initialize state manager: {e}")
            return None
    
    def _init_validator(self):
        """Initialize file system validator"""
        try:
            from tools.validation.file_system_validator import FileSystemValidator
            return FileSystemValidator()
        except ImportError as e:
            logger.error(f"Failed to initialize validator: {e}")
            return None
    
    def _init_approval_controller(self):
        """Initialize approval controller"""
        try:
            from tools.approval.approval_gate_controller import ApprovalGateController
            return ApprovalGateController(state_manager=self.state_manager)
        except ImportError as e:
            logger.error(f"Failed to initialize approval controller: {e}")
            return None
    
    def _init_executor(self):
        """Initialize automatic executor"""
        try:
            from tools.execution.automatic_executor import get_automatic_executor
            return get_automatic_executor(self.state_manager, self.validator)
        except ImportError as e:
            logger.error(f"Failed to initialize executor: {e}")
            return None
    
    def _load_workflow_config(self) -> Dict[str, WorkflowStep]:
        """Load workflow step configuration"""
        # Define the standard 13-step workflow
        steps = {
            "branch_verification": WorkflowStep(
                step_id="branch_verification",
                phase="phase_0_5",
                title="Git Branch Verification",
                description="Verify that work is being done on correct feature branch",
                prerequisites=[],
                validation_required=True,
                approval_required=False,
                auto_executable=False
            ),
            "sam_env_check": WorkflowStep(
                step_id="sam_env_check", 
                phase="phase_1",
                title="Virtual Environment Check",
                description="Verify sam-env virtual environment is active",
                prerequisites=["branch_verification"],
                validation_required=True,
                approval_required=False,
                auto_executable=False
            ),
            "google_sheets_sync": WorkflowStep(
                step_id="google_sheets_sync",
                phase="phase_1", 
                title="Google Sheets Synchronization",
                description="Sync tracker status with Google Sheets",
                prerequisites=["sam_env_check"],
                validation_required=True,
                approval_required=False,
                auto_executable=False
            ),
            "sow_creation": WorkflowStep(
                step_id="sow_creation",
                phase="phase_1",
                title="Statement of Work Creation", 
                description="Create comprehensive SOW document",
                prerequisites=["google_sheets_sync"],
                validation_required=True,
                approval_required=True,
                auto_executable=False,
                approval_criteria=[
                    "Work scope is clearly defined and realistic",
                    "Deliverables are specific and measurable",
                    "Timeline and milestones are appropriate",
                    "Responsibility boundaries are clear"
                ]
            ),
            "implementation": WorkflowStep(
                step_id="implementation",
                phase="phase_2",
                title="Implementation Work",
                description="Implement required functionality",
                prerequisites=["sow_creation"],
                validation_required=True,
                approval_required=True,
                auto_executable=False,
                approval_criteria=[
                    "Implementation follows SOW requirements",
                    "Code quality meets project standards",
                    "No breaking changes introduced"
                ]
            ),
            "testing": WorkflowStep(
                step_id="testing",
                phase="phase_2",
                title="Testing Execution",
                description="Execute comprehensive testing",
                prerequisites=["implementation"],
                validation_required=True,
                approval_required=False,
                auto_executable=False
            ),
            "extraction": WorkflowStep(
                step_id="extraction",
                phase="phase_2",
                title="Character Extraction",
                description="Execute character extraction pipeline",
                prerequisites=["testing"],
                validation_required=True,
                approval_required=False,
                auto_executable=True
            ),
            "quality_workflow": WorkflowStep(
                step_id="quality_workflow",
                phase="phase_3",
                title="Quality Workflow Execution",
                description="Execute quality analysis and reporting",
                prerequisites=["extraction"],
                validation_required=True,
                approval_required=False,
                auto_executable=True
            ),
            "dashboard_generation": WorkflowStep(
                step_id="dashboard_generation",
                phase="phase_3",
                title="Dashboard Generation",
                description="Generate final dashboard and ensure server integration",
                prerequisites=["quality_workflow"],
                validation_required=True,
                approval_required=False,
                auto_executable=True
            ),
            "final_approval": WorkflowStep(
                step_id="final_approval",
                phase="phase_4",
                title="Final Approval",
                description="Final review and approval before PR creation",
                prerequisites=["dashboard_generation"],
                validation_required=False,
                approval_required=True,
                auto_executable=False,
                approval_criteria=[
                    "All deliverables completed successfully",
                    "Quality metrics meet requirements",
                    "Dashboard is accessible and complete",
                    "Ready for production deployment"
                ]
            )
        }
        
        return steps
    
    def create_tracker_workflow(self, tracker_id: str) -> bool:
        """
        Create a new tracker workflow with mechanical state management.
        This is the entry point for new trackers.
        """
        logger.info(f"Creating workflow for tracker: {tracker_id}")
        
        if not self.state_manager:
            logger.error("State manager not available")
            return False
        
        # Create workflow state
        success = self.state_manager.create_tracker_workflow(tracker_id)
        
        if success:
            # Ensure workspace exists
            if self.validator:
                self.validator.ensure_workspace_exists(tracker_id)
            
            logger.info(f"Workflow created successfully for {tracker_id}")
        else:
            logger.error(f"Failed to create workflow for {tracker_id}")
        
        return success
    
    def get_current_step_instructions(self, tracker_id: str) -> Optional[StepInstructions]:
        """
        Get simplified, current-step-only instructions for AI.
        This reduces cognitive load by showing only what's needed now.
        """
        if not self.state_manager:
            return None
        
        current_step_id = self.state_manager.get_current_step(tracker_id)
        if not current_step_id:
            return None
        
        step_config = self.workflow_config.get(current_step_id)
        if not step_config:
            return None
        
        # Check if step can proceed
        can_proceed, blocking_reasons = self.state_manager.can_proceed_to_step(
            tracker_id, current_step_id
        )
        
        # Generate step-specific instructions
        required_actions = self._generate_step_actions(step_config)
        validation_criteria = self._generate_validation_criteria(step_config)
        
        return StepInstructions(
            step_id=current_step_id,
            title=step_config.title,
            description=step_config.description,
            required_actions=required_actions,
            validation_criteria=validation_criteria,
            approval_required=step_config.approval_required,
            can_proceed=can_proceed,
            blocking_reasons=blocking_reasons if not can_proceed else None
        )
    
    def _generate_step_actions(self, step_config: WorkflowStep) -> List[str]:
        """Generate specific actions for a step"""
        actions = []
        
        if step_config.step_id == "branch_verification":
            actions = [
                "Verify current Git branch with: git branch --show-current",
                f"Ensure branch follows pattern: feature/{'{tracker_id}'}",
                "Create feature branch if needed: git checkout -b feature/{tracker_id}"
            ]
        elif step_config.step_id == "sam_env_check":
            actions = [
                "Check virtual environment: echo $VIRTUAL_ENV",
                "Activate sam-env if needed: source sam-env/bin/activate",
                "Verify Python path: which python3"
            ]
        elif step_config.step_id == "google_sheets_sync":
            actions = [
                "Update tracker status: python tools/progress_tracker/cli.py update {tracker_id} '着手中'",
                "Verify sync: python tools/progress_tracker/cli.py status {tracker_id}"
            ]
        elif step_config.step_id == "sow_creation":
            actions = [
                "Create SOW document with required sections:",
                "- Work scope definition",
                "- Deliverables specification", 
                "- Responsibility boundaries",
                "- Approval conditions",
                "Save as: workspace/{tracker_id}/sow_document.md"
            ]
        elif step_config.step_id == "implementation":
            actions = [
                "Implement functionality according to SOW",
                "Follow project coding standards",
                "Create appropriate tests",
                "Commit changes with proper messages"
            ]
        elif step_config.step_id == "testing":
            actions = [
                "Execute unit tests: pytest tests/unit/",
                "Execute integration tests: pytest tests/integration/",
                "Verify all tests pass"
            ]
        elif step_config.step_id == "extraction":
            actions = [
                "Extraction will be executed automatically",
                "Monitor progress in extraction.log",
                "Verify results in workspace/{tracker_id}/extraction/"
            ]
        elif step_config.step_id == "quality_workflow":
            actions = [
                "Quality workflow will be executed automatically",
                "Monitor progress in quality_workflow.log",
                "Verify dashboard generation"
            ]
        elif step_config.step_id == "dashboard_generation":
            actions = [
                "Dashboard generation will be executed automatically",
                "Verify index.html creation",
                "Test server accessibility"
            ]
        elif step_config.step_id == "final_approval":
            actions = [
                "Review all completed work",
                "Verify quality metrics",
                "Confirm dashboard accessibility",
                "Wait for approval before proceeding"
            ]
        
        return actions
    
    def _generate_validation_criteria(self, step_config: WorkflowStep) -> List[str]:
        """Generate validation criteria for a step"""
        criteria = []
        
        if step_config.validation_required:
            if step_config.step_id == "branch_verification":
                criteria = ["Current branch matches feature/{tracker_id} pattern"]
            elif step_config.step_id == "sam_env_check":
                criteria = ["VIRTUAL_ENV contains 'sam-env'", "Python path includes sam-env"]
            elif step_config.step_id == "google_sheets_sync":
                criteria = ["Tracker found in Google Sheets", "Status shows '着手中' or similar"]
            elif step_config.step_id == "sow_creation":
                criteria = [
                    "SOW file exists in workspace",
                    "Contains all required sections",
                    "Minimum 500 characters content"
                ]
            elif step_config.step_id == "implementation":
                criteria = [
                    "Git commits exist for tracker",
                    "Files modified compared to main branch"
                ]
            elif step_config.step_id == "testing":
                criteria = [
                    "Test execution evidence found",
                    "No test failures recorded"
                ]
            elif step_config.step_id == "extraction":
                criteria = [
                    "Extraction directory exists",
                    "Extracted image files present",
                    "extraction_result.json valid",
                    "Successful extractions > 0"
                ]
            elif step_config.step_id == "quality_workflow":
                criteria = [
                    "Dashboard HTML file exists",
                    "Contains required sections",
                    "Quality report JSON valid"
                ]
            elif step_config.step_id == "dashboard_generation":
                criteria = [
                    "index.html exists",
                    "Contains tracker-specific content",
                    "Valid HTML structure"
                ]
        
        return criteria
    
    def attempt_step_completion(self, tracker_id: str) -> StepResult:
        """
        Attempt to complete the current step with full enforcement.
        This is the main enforcement entry point.
        """
        if not self.state_manager:
            return StepResult.FAILED(["State manager not available"])
        
        current_step_id = self.state_manager.get_current_step(tracker_id)
        if not current_step_id:
            return StepResult.FAILED([f"No current step found for tracker {tracker_id}"])
        
        step_config = self.workflow_config.get(current_step_id)
        if not step_config:
            return StepResult.FAILED([f"Unknown step configuration: {current_step_id}"])
        
        logger.info(f"Attempting step completion: {tracker_id}/{current_step_id}")

        # Check if current step is in waiting state
        current_status = self.state_manager.get_step_status(tracker_id, current_step_id)
        if current_status == "waiting":
            logger.info(f"Step {current_step_id} is in waiting state for {tracker_id}")
            return StepResult.WAITING([f"Step {current_step_id} is waiting for background task completion"])

        # Check if step can proceed (mechanical enforcement)
        can_proceed, blocking_reasons = self.state_manager.can_proceed_to_step(
            tracker_id, current_step_id
        )
        
        if not can_proceed:
            return StepResult.BLOCKED(blocking_reasons)
        
        # Handle auto-executable steps
        if step_config.auto_executable and self.executor:
            logger.info(f"Executing step automatically: {current_step_id}")
            
            execution_result = self.executor.execute_step_automatically(
                tracker_id, current_step_id
            )
            
            if execution_result.success:
                # Advance to next step
                next_step = self._get_next_step(current_step_id)
                if next_step:
                    self.state_manager.advance_to_next_step(tracker_id)
                    return StepResult.COMPLETED(next_step)
                else:
                    return StepResult.COMPLETED("workflow_complete")
            else:
                return StepResult.FAILED([execution_result.message])
        
        # Handle manual steps with validation
        if step_config.validation_required and self.validator:
            logger.info(f"Validating step completion: {current_step_id}")
            
            validation_result = self.state_manager.validate_step_completion(
                tracker_id, current_step_id
            )
            
            if not validation_result.passed:
                return StepResult.FAILED(validation_result.errors)
        
        # Handle approval requirements
        if step_config.approval_required and self.approval_controller:
            logger.info(f"Checking approval for step: {current_step_id}")

            # First check if already approved
            existing_approval = self.approval_controller.check_existing_approval(tracker_id, current_step_id)
            if existing_approval:
                logger.info(f"Step {current_step_id} is already approved, proceeding...")
                # Continue to step completion below
            else:
                # Need to request approval
                from tools.approval.approval_gate_controller import ApprovalContext

                context = ApprovalContext(
                    step_name=step_config.title,
                    description=step_config.description,
                    artifacts=self._get_step_artifacts(tracker_id, current_step_id),
                    approval_criteria=step_config.approval_criteria or [],
                    priority="high" if current_step_id in ["sow_creation", "final_approval"] else "normal"
                )

                approval_id = self.approval_controller.request_approval(
                    tracker_id, current_step_id, context
                )

                # Check again if it was already approved (request_approval now returns existing approvals)
                if self.approval_controller.check_existing_approval(tracker_id, current_step_id):
                    logger.info(f"Step {current_step_id} has been approved, proceeding...")
                    # Continue to step completion below
                else:
                    return StepResult.PENDING_APPROVAL(approval_id)
        
        # Step completed - advance to next
        next_step = self._get_next_step(current_step_id)
        if next_step:
            self.state_manager.advance_to_next_step(tracker_id)
            return StepResult.COMPLETED(next_step)
        else:
            return StepResult.COMPLETED("workflow_complete")
    
    def _get_step_artifacts(self, tracker_id: str, step_id: str) -> List[str]:
        """Get artifacts for approval review"""
        artifacts = []
        workspace = f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}"
        
        if step_id == "sow_creation":
            artifacts = [f"{workspace}/sow_document.md"]
        elif step_id == "implementation":
            artifacts = ["Git commits and modified files"]
        elif step_id == "final_approval":
            artifacts = [
                f"{workspace}/extraction_result.json",
                f"{workspace}/dashboard/dashboard.html",
                f"{workspace}/index.html"
            ]
        
        return artifacts
    
    def _get_next_step(self, current_step_id: str) -> Optional[str]:
        """Get the next step in the workflow"""
        step_order = [
            "branch_verification", "sam_env_check", "google_sheets_sync",
            "sow_creation", "implementation", "testing", "extraction",
            "quality_workflow", "dashboard_generation", "final_approval"
        ]
        
        try:
            current_index = step_order.index(current_step_id)
            if current_index < len(step_order) - 1:
                return step_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def get_workflow_status(self, tracker_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status"""
        if not self.state_manager:
            return {"error": "State manager not available"}
        
        status = self.state_manager.get_workflow_status(tracker_id)
        
        # Add current step instructions
        instructions = self.get_current_step_instructions(tracker_id)
        if instructions:
            status["current_step_instructions"] = {
                "title": instructions.title,
                "description": instructions.description,
                "required_actions": instructions.required_actions,
                "validation_criteria": instructions.validation_criteria,
                "approval_required": instructions.approval_required,
                "can_proceed": instructions.can_proceed,
                "blocking_reasons": instructions.blocking_reasons
            }
        
        # Add process status if available
        if self.executor:
            process_status = self.executor.check_process_status(tracker_id)
            if process_status:
                status["background_process"] = process_status
        
        return status
    
    def wait_for_approval(self, approval_id: str, timeout_minutes: int = 60) -> StepResult:
        """Wait for approval with timeout"""
        if not self.approval_controller:
            return StepResult.FAILED(["Approval controller not available"])
        
        approval_result = self.approval_controller.wait_for_approval(
            approval_id, timeout_minutes
        )
        
        if approval_result.success:
            return StepResult.COMPLETED("approval_granted")
        else:
            return StepResult.FAILED([approval_result.error_message])

# Global instance for easy access
_workflow_controller = None

def get_workflow_controller() -> WorkflowController:
    """Get the global workflow controller instance"""
    global _workflow_controller
    if _workflow_controller is None:
        _workflow_controller = WorkflowController()
    return _workflow_controller