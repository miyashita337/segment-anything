#!/usr/bin/env python3
"""
承認ゲートコントローラー - 強制承認システム
INCI-006 解決策: AIがバイパスできない機械的承認システム

このモジュールは、明示的な人間の承認を受けるまでAIの進行を物理的に
ブロックする強制承認システムを提供します。推測なし、バイパスなし、
AI判断なし - 機械的強制実行のみ。
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ApprovalContext:
    """承認要求のコンテキスト情報"""
    step_name: str
    description: str
    artifacts: List[str]
    approval_criteria: List[str]
    priority: str = "normal"  # normal, high, critical
    timeout_hours: int = 24

class ApprovalStatus:
    """承認要求の状態"""
    def __init__(self, approved: bool, approved_by: str = None, 
                 approved_at: str = None, comments: str = "", 
                 evidence: str = ""):
        self.approved = approved
        self.approved_by = approved_by
        self.approved_at = approved_at
        self.comments = comments
        self.evidence = evidence

class ApprovalResult:
    """承認待ち操作の結果"""
    def __init__(self, success: bool, status: ApprovalStatus = None, 
                 error_message: str = ""):
        self.success = success
        self.status = status
        self.error_message = error_message

class ApprovalGateController:
    """
    Mechanical approval system that physically blocks AI progress
    until explicit human approval is received through file system.
    """
    
    def __init__(self, approval_dir: str = None, state_manager=None):
        # DO NOT use project root for approvals - causes pollution
        # Approval directory will be set per tracker workspace
        self.base_approval_dir = approval_dir
        self.state_manager = state_manager
        self.lock = threading.Lock()
        
        logger.info(f"承認ゲートコントローラー初期化完了: ワークスペースベース承認")
    
    def _find_project_root(self) -> str:
        """Find the project root directory"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            current = os.path.dirname(current)
        return "/mnt/c/AItools/segment-anything"  # Fallback
    
    def _get_tracker_approval_dir(self, tracker_id: str) -> str:
        """Get approval directory for specific tracker workspace"""
        if self.base_approval_dir:
            return self.base_approval_dir
            
        # Determine workspace base from tracker ID
        if tracker_id.startswith("KIRO"):
            workspace_base = "/mnt/c/AItools/lora/train/kiri/tracker-workspace"
        elif tracker_id.startswith("TRACKER"):
            workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        else:
            # Fallback to generic workspace
            workspace_base = "/mnt/c/AItools/lora/train/generic/tracker-workspace"
        
        approval_dir = os.path.join(workspace_base, tracker_id, ".approvals")
        os.makedirs(approval_dir, exist_ok=True)
        return approval_dir
    
    def request_approval(self, tracker_id: str, step_id: str, 
                        context: ApprovalContext) -> str:
        """
        Create approval request that physically blocks progress.
        Returns approval_id that must be resolved before proceeding.
        """
        approval_id = f"{tracker_id}_{step_id}_{int(time.time())}"
        
        with self.lock:
            # Get tracker-specific approval directory
            approval_dir = self._get_tracker_approval_dir(tracker_id)
            # Create approval request file
            request_file = os.path.join(approval_dir, f"{approval_id}_request.json")
            
            request_data = {
                "approval_id": approval_id,
                "tracker_id": tracker_id,
                "step_id": step_id,
                "step_name": context.step_name,
                "description": context.description,
                "artifacts": context.artifacts,
                "approval_criteria": context.approval_criteria,
                "priority": context.priority,
                "timeout_hours": context.timeout_hours,
                "requested_at": datetime.now().isoformat(),
                "status": "pending"
            }
            
            try:
                with open(request_file, 'w', encoding='utf-8') as f:
                    json.dump(request_data, f, indent=2, ensure_ascii=False)
                
                # Update state manager if available
                if self.state_manager:
                    self.state_manager.require_approval(tracker_id, step_id, {
                        'type': 'step_approval',
                        'title': context.step_name,
                        'description': context.description
                    })
                
                # Display approval request to user
                self._display_approval_request(approval_id, context, request_data)
                
                logger.info(f"承認要求を作成しました: {approval_id}")
                return approval_id
                
            except Exception as e:
                logger.error(f"Failed to create approval request {approval_id}: {e}")
                raise
    
    def _display_approval_request(self, approval_id: str, context: ApprovalContext, 
                                 request_data: Dict[str, Any]):
        """
        Display clear, unmistakable approval request to user.
        This cannot be missed or bypassed by AI.
        """
        print("\n" + "="*80)
        print("🚨 重要: 承認が必要です - すべての進行がブロックされています")
        print("="*80)
        print(f"📋 承認ID: {approval_id}")
        print(f"🎯 ステップ: {context.step_name}")
        print(f"📝 説明: {context.description}")
        print(f"⏰ 優先度: {context.priority.upper()}")
        print(f"⏳ タイムアウト: {context.timeout_hours} 時間")
        
        if context.artifacts:
            print(f"📎 確認対象ファイル:")
            for artifact in context.artifacts:
                print(f"   - {artifact}")
        
        if context.approval_criteria:
            print(f"✅ 承認基準:")
            for i, criterion in enumerate(context.approval_criteria, 1):
                print(f"   {i}. {criterion}")
        
        # Get tracker-specific approval directory for display
        tracker_id = approval_id.split('_')[0]
        approval_dir = self._get_tracker_approval_dir(tracker_id)
        
        print("\n🔧 このステップを承認するには:")
        print(f"   ファイルを作成: {approval_dir}/{approval_id}_approved.json")
        print("   内容:")
        
        approval_template = {
            "approved": True,
            "approved_by": "[あなたの名前]",
            "approved_at": datetime.now().isoformat(),
            "comments": "[任意: 承認理由]",
            "evidence": "[任意: 確認した証拠]"
        }
        
        print(json.dumps(approval_template, indent=2, ensure_ascii=False))
        
        print("\n🚫 このステップを拒否するには:")
        print(f"   ファイルを作成: {approval_dir}/{approval_id}_denied.json")
        print("   内容:")
        
        denial_template = {
            "approved": False,
            "denied_by": "[あなたの名前]",
            "denied_at": datetime.now().isoformat(),
            "reason": "[拒否理由]",
            "required_changes": "[修正が必要な内容]"
        }
        
        print(json.dumps(denial_template, indent=2, ensure_ascii=False))
        
        print("\n⚠️  承認されるまでAIの実行は完全にブロックされます")
        print("⚠️  人間の決定なしには進行できません")
        print("="*80)
        print()
    
    def check_approval_status(self, approval_id: str) -> ApprovalStatus:
        """
        Check if approval has been granted through file system.
        This is mechanical - no AI interpretation.
        """
        # Extract tracker_id from approval_id (format: TRACKER-001_step_timestamp)
        tracker_id = approval_id.split('_')[0]
        approval_dir = self._get_tracker_approval_dir(tracker_id)
        
        # Check for approval file
        approved_file = os.path.join(approval_dir, f"{approval_id}_approved.json")
        denied_file = os.path.join(approval_dir, f"{approval_id}_denied.json")
        
        if os.path.exists(approved_file):
            try:
                with open(approved_file, 'r', encoding='utf-8') as f:
                    approval_data = json.load(f)
                
                return ApprovalStatus(
                    approved=True,
                    approved_by=approval_data.get('approved_by', 'Unknown'),
                    approved_at=approval_data.get('approved_at', ''),
                    comments=approval_data.get('comments', ''),
                    evidence=approval_data.get('evidence', '')
                )
            except Exception as e:
                logger.error(f"Error reading approval file {approved_file}: {e}")
                return ApprovalStatus(False)
        
        elif os.path.exists(denied_file):
            try:
                with open(denied_file, 'r', encoding='utf-8') as f:
                    denial_data = json.load(f)
                
                return ApprovalStatus(
                    approved=False,
                    approved_by=denial_data.get('denied_by', 'Unknown'),
                    approved_at=denial_data.get('denied_at', ''),
                    comments=denial_data.get('reason', ''),
                    evidence=denial_data.get('required_changes', '')
                )
            except Exception as e:
                logger.error(f"Error reading denial file {denied_file}: {e}")
                return ApprovalStatus(False)
        
        else:
            return ApprovalStatus(False)  # Still pending    

    def wait_for_approval(self, approval_id: str, timeout_minutes: int = None) -> ApprovalResult:
        """
        Block execution until approval is received.
        This is the core enforcement mechanism - AI cannot proceed without approval.
        """
        if timeout_minutes is None:
            # Get timeout from request file
            tracker_id = approval_id.split('_')[0]
            approval_dir = self._get_tracker_approval_dir(tracker_id)
            request_file = os.path.join(approval_dir, f"{approval_id}_request.json")
            try:
                with open(request_file, 'r') as f:
                    request_data = json.load(f)
                timeout_minutes = request_data.get('timeout_hours', 24) * 60
            except:
                timeout_minutes = 24 * 60  # Default 24 hours
        
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        logger.info(f"承認待ち: {approval_id} (タイムアウト: {timeout_minutes} 分)")
        
        while time.time() - start_time < timeout_minutes * 60:
            status = self.check_approval_status(approval_id)
            
            if status.approved:
                # Approval granted - unblock progress
                self._process_approval_granted(approval_id, status)
                logger.info(f"承認されました: {approval_id} 承認者: {status.approved_by}")
                return ApprovalResult(True, status)
            
            elif status.approved is False and status.approved_by:
                # Explicitly denied
                self._process_approval_denied(approval_id, status)
                logger.warning(f"承認が拒否されました: {approval_id} 拒否者: {status.approved_by}")
                return ApprovalResult(False, status, f"Approval denied: {status.comments}")
            
            # Still pending - display waiting message
            remaining_minutes = timeout_minutes - int((time.time() - start_time) / 60)
            tracker_id = approval_id.split('_')[0]
            approval_dir = self._get_tracker_approval_dir(tracker_id)
            print(f"⏳ 承認待ち: {approval_id}")
            print(f"   残り時間: {remaining_minutes} 分")
            print(f"   承認するには: {approval_dir}/{approval_id}_approved.json を作成")
            print(f"   拒否するには: {approval_dir}/{approval_id}_denied.json を作成")
            print()
            
            time.sleep(check_interval)
        
        # Timeout reached - escalate
        logger.error(f"Approval timeout: {approval_id}")
        self._escalate_approval_timeout(approval_id)
        return ApprovalResult(False, None, f"Approval timeout after {timeout_minutes} minutes")
    
    def _process_approval_granted(self, approval_id: str, status: ApprovalStatus):
        """Process approval granted - unblock progress"""
        try:
            # Update state manager if available
            if self.state_manager:
                self.state_manager.grant_approval(
                    approval_id, 
                    status.approved_by, 
                    status.evidence
                )
            
            # Move request file to completed
            request_file = os.path.join(self.approval_dir, f"{approval_id}_request.json")
            completed_file = os.path.join(self.approval_dir, f"{approval_id}_completed.json")
            
            if os.path.exists(request_file):
                with open(request_file, 'r') as f:
                    request_data = json.load(f)
                
                request_data.update({
                    "status": "approved",
                    "approved_by": status.approved_by,
                    "approved_at": status.approved_at,
                    "comments": status.comments,
                    "evidence": status.evidence
                })
                
                with open(completed_file, 'w') as f:
                    json.dump(request_data, f, indent=2, ensure_ascii=False)
                
                os.remove(request_file)
            
            print("✅ APPROVAL GRANTED - PROGRESS UNBLOCKED")
            print(f"   Approved by: {status.approved_by}")
            print(f"   Comments: {status.comments}")
            print()
            
        except Exception as e:
            logger.error(f"Error processing approval granted {approval_id}: {e}")
    
    def _process_approval_denied(self, approval_id: str, status: ApprovalStatus):
        """Process approval denied - keep progress blocked"""
        try:
            # Update state manager if available
            if self.state_manager:
                # Keep the blocking in place for denied approvals
                pass
            
            # Move request file to denied
            request_file = os.path.join(self.approval_dir, f"{approval_id}_request.json")
            denied_file = os.path.join(self.approval_dir, f"{approval_id}_denied_final.json")
            
            if os.path.exists(request_file):
                with open(request_file, 'r') as f:
                    request_data = json.load(f)
                
                request_data.update({
                    "status": "denied",
                    "denied_by": status.approved_by,
                    "denied_at": status.approved_at,
                    "reason": status.comments,
                    "required_changes": status.evidence
                })
                
                with open(denied_file, 'w') as f:
                    json.dump(request_data, f, indent=2, ensure_ascii=False)
                
                os.remove(request_file)
            
            print("❌ APPROVAL DENIED - PROGRESS REMAINS BLOCKED")
            print(f"   Denied by: {status.approved_by}")
            print(f"   Reason: {status.comments}")
            print(f"   Required changes: {status.evidence}")
            print()
            
        except Exception as e:
            logger.error(f"Error processing approval denied {approval_id}: {e}")
    
    def _escalate_approval_timeout(self, approval_id: str):
        """Escalate approval timeout to supervisors"""
        try:
            # Create escalation file
            escalation_file = os.path.join(self.approval_dir, f"{approval_id}_escalated.json")
            
            escalation_data = {
                "approval_id": approval_id,
                "escalated_at": datetime.now().isoformat(),
                "reason": "approval_timeout",
                "status": "requires_supervisor_attention"
            }
            
            with open(escalation_file, 'w') as f:
                json.dump(escalation_data, f, indent=2, ensure_ascii=False)
            
            print("🚨 APPROVAL TIMEOUT - ESCALATED TO SUPERVISOR")
            print(f"   Approval ID: {approval_id}")
            print(f"   Escalation file: {escalation_file}")
            print("   Manual intervention required")
            print()
            
            logger.error(f"Approval escalated due to timeout: {approval_id}")
            
        except Exception as e:
            logger.error(f"Error escalating approval timeout {approval_id}: {e}")
    
    def list_pending_approvals(self) -> List[Dict[str, Any]]:
        """List all pending approval requests across all tracker workspaces"""
        pending_approvals = []
        
        try:
            # Scan all possible workspace locations
            workspace_bases = [
                "/mnt/c/AItools/lora/train/kiri/tracker-workspace",
                "/mnt/c/AItools/lora/train/yado/tracker-workspace",
                "/mnt/c/AItools/lora/train/generic/tracker-workspace"
            ]
            
            for workspace_base in workspace_bases:
                if not os.path.exists(workspace_base):
                    continue
                    
                for tracker_id in os.listdir(workspace_base):
                    approval_dir = os.path.join(workspace_base, tracker_id, ".approvals")
                    if not os.path.exists(approval_dir):
                        continue
                        
                    for filename in os.listdir(approval_dir):
                        if filename.endswith('_request.json'):
                            request_file = os.path.join(approval_dir, filename)
                            
                            with open(request_file, 'r') as f:
                                request_data = json.load(f)
                            
                            # Check if still pending
                            approval_id = request_data['approval_id']
                            status = self.check_approval_status(approval_id)
                            
                            if not status.approved and not status.approved_by:
                                # Calculate time remaining
                                requested_at = datetime.fromisoformat(request_data['requested_at'])
                                timeout_hours = request_data.get('timeout_hours', 24)
                                expires_at = requested_at + timedelta(hours=timeout_hours)
                                
                                request_data['expires_at'] = expires_at.isoformat()
                                request_data['time_remaining_hours'] = max(0, 
                                    (expires_at - datetime.now()).total_seconds() / 3600)
                                
                                pending_approvals.append(request_data)
            
            return pending_approvals
            
        except Exception as e:
            logger.error(f"Error listing pending approvals: {e}")
            return []
    
    def cleanup_old_approvals(self, days_old: int = 30):
        """Clean up old approval files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            
            for filename in os.listdir(self.approval_dir):
                if filename.endswith(('.json',)):
                    file_path = os.path.join(self.approval_dir, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old approval file: {filename}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old approvals: {e}")
    
    def force_approve(self, approval_id: str, approved_by: str, reason: str = "Emergency override"):
        """
        Emergency approval override - use with extreme caution.
        This should only be used in emergency situations.
        """
        logger.warning(f"EMERGENCY APPROVAL OVERRIDE: {approval_id} by {approved_by}")
        
        approved_file = os.path.join(self.approval_dir, f"{approval_id}_approved.json")
        
        approval_data = {
            "approved": True,
            "approved_by": f"{approved_by} (EMERGENCY OVERRIDE)",
            "approved_at": datetime.now().isoformat(),
            "comments": f"Emergency override: {reason}",
            "evidence": "emergency_override"
        }
        
        with open(approved_file, 'w') as f:
            json.dump(approval_data, f, indent=2, ensure_ascii=False)
        
        print("🚨 EMERGENCY APPROVAL OVERRIDE EXECUTED")
        print(f"   Approval ID: {approval_id}")
        print(f"   Override by: {approved_by}")
        print(f"   Reason: {reason}")
        print()

# Convenience functions for common approval scenarios
def create_sow_approval(tracker_id: str, sow_file: str, controller: ApprovalGateController = None) -> str:
    """Create SOW approval request"""
    if controller is None:
        controller = ApprovalGateController()
    
    context = ApprovalContext(
        step_name="SOW (Statement of Work) Approval",
        description="Please review and approve the Statement of Work document before proceeding to implementation",
        artifacts=[sow_file],
        approval_criteria=[
            "Work scope is clearly defined and realistic",
            "Deliverables are specific and measurable", 
            "Timeline and milestones are appropriate",
            "Responsibility boundaries are clear",
            "Quality standards are well-defined"
        ],
        priority="high",
        timeout_hours=48
    )
    
    return controller.request_approval(tracker_id, "sow_approval", context)

def create_implementation_approval(tracker_id: str, controller: ApprovalGateController = None) -> str:
    """Create implementation approval request"""
    if controller is None:
        controller = ApprovalGateController()
    
    context = ApprovalContext(
        step_name="Implementation Review Approval",
        description="Please review the implementation work before proceeding to extraction",
        artifacts=[],
        approval_criteria=[
            "Code changes are appropriate for the task",
            "Implementation follows project standards",
            "No breaking changes introduced",
            "Testing has been completed"
        ],
        priority="normal",
        timeout_hours=24
    )
    
    return controller.request_approval(tracker_id, "implementation_approval", context)