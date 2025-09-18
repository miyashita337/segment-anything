# 設計書

## 概要

この設計は、包括的なワークフロードキュメントが一貫したAIエージェントの動作を確保できない根本的な構造問題に対処します。分析により、現在のシステムは「検証ベース制御」（機械的に遵守を強制）ではなく「信頼ベース制御」（AIが指示に従うことを期待）に依存していることが明らかになりました。

必要な核心的なアーキテクチャシフトは、**認知依存**（AI注意、記憶、判断への依存）から**機械依存**（自動チェック、外部強制、物理的ブロック）への移行です。

## アーキテクチャ

### 現在のシステム分析：信頼ベース制御

```
現在の「信頼ベース」アーキテクチャ：
┌─────────────────────────────────────┐
│ ドキュメント層（重い）                │
│ ├── CLAUDE.md（1,200行以上）        │
│ ├── unified_tracker_template.md    │
│ ├── tracker_workflow_checklist.md  │
│ └── 複数の参照ドキュメント            │
└─────────────────────────────────────┘
           ↓ "従ってください"
┌─────────────────────────────────────┐
│ AI認知処理（信頼できない）            │
│ ├── 読み取りと解釈                   │
│ ├── 記憶と優先順位付け               │
│ ├── 判断と決定                      │
│ └── 完了の自己報告                   │
└─────────────────────────────────────┘
           ↓ "やりました"
┌─────────────────────────────────────┐
│ 信頼ベース検証（弱い）               │
│ ├── AIの自己報告を受け入れ           │
│ ├── 遵守への期待                    │
│ └── 失敗への反応                    │
└─────────────────────────────────────┘

問題：
- AIは指示を忘れ、誤解釈し、または迂回する可能性
- 主張された完了の外部検証なし
- 認知過負荷がショートカットにつながる
- セッション間での非冪等的動作
```

### 提案アーキテクチャ：検証ベース制御

```
新しい「検証ベース」アーキテクチャ：
┌─────────────────────────────────────┐
│ 外部制御層（機械的）                 │
│ ├── State Database (SQLite)        │
│ ├── File System Validators         │
│ ├── Approval Gate Controllers      │
│ └── Automatic Executors            │
└─────────────────────────────────────┘
           ↓ "System enforces"
┌─────────────────────────────────────┐
│ AI Interface Layer (Constrained)    │
│ ├── Current Step Only Display      │
│ ├── Forced Validation Calls        │
│ ├── Blocked Action Prevention      │
│ └── Approval Wait States           │
└─────────────────────────────────────┘
           ↓ "System verifies"
┌─────────────────────────────────────┐
│ Mechanical Verification (Strong)    │
│ ├── File existence checks          │
│ ├── Content quality validation     │
│ ├── External approval confirmation │
│ └── Automatic state updates        │
└─────────────────────────────────────┘

Benefits:
- AI cannot bypass mechanical checks
- External verification of all claims
- Automatic execution of critical steps
- Idempotent behavior guaranteed
```

## Components and Interfaces

### 1. Workflow State Database

**Purpose**: External, tamper-proof tracking of workflow progress

```python
# Database Schema
class WorkflowState:
    tracker_id: str
    current_phase: str
    current_step: str
    completed_steps: List[str]
    blocked_actions: List[str]
    pending_approvals: List[str]
    last_validation: datetime
    
class StepValidation:
    step_id: str
    validation_type: str  # file_exists, content_check, quality_gate
    validation_result: bool
    validation_timestamp: datetime
    validation_evidence: str  # ファイルパス、チェックサムなど

class ApprovalRequest:
    approval_id: str
    step_id: str
    requested_at: datetime
    approved_at: Optional[datetime]
    approved_by: Optional[str]
    status: str  # pending, approved, denied, expired
```

**実装**:
```python
class WorkflowStateManager:
    def __init__(self, db_path: str = "workflow_state.db"):
        self.db = sqlite3.connect(db_path)
        self._init_tables()
    
    def can_proceed_to_step(self, tracker_id: str, step_id: str) -> bool:
        """機械的チェック - AI解釈なし"""
        prerequisites = self._get_step_prerequisites(step_id)
        for prereq in prerequisites:
            if not self._is_step_validated(tracker_id, prereq):
                return False
        return True
    
    def validate_step_completion(self, tracker_id: str, step_id: str) -> ValidationResult:
        """ファイルシステムチェックによる外部検証"""
        validators = self._get_step_validators(step_id)
        for validator in validators:
            result = validator.validate(tracker_id)
            if not result.passed:
                return ValidationResult(False, result.errors)
        
        # 検証成功後のみ状態更新
        self._mark_step_completed(tracker_id, step_id)
        return ValidationResult(True, [])
    
    def require_approval(self, tracker_id: str, step_id: str) -> str:
        """進行をブロックする承認要件を作成"""
        approval_id = self._create_approval_request(tracker_id, step_id)
        self._block_actions_until_approved(tracker_id, approval_id)
        return approval_id
```

### 2. ファイルシステム検証器

**目的**: ファイルシステム証拠による機械的検証

```python
class FileSystemValidator:
    def __init__(self, workspace_base: str):
        self.workspace_base = workspace_base
    
    def validate_extraction_completion(self, tracker_id: str) -> ValidationResult:
        """抽出が実際に発生したことを確認"""
        extraction_dir = f"{self.workspace_base}/{tracker_id}/extraction/"
        
        # ディレクトリ存在チェック
        if not os.path.exists(extraction_dir):
            return ValidationResult(False, ["抽出ディレクトリが見つかりません"])
        
        # 実際の抽出ファイルをチェック
        extracted_files = glob.glob(f"{extraction_dir}/*_extracted.jpg")
        if len(extracted_files) == 0:
            return ValidationResult(False, ["抽出ファイルが見つかりません"])
        
        # extraction_result.jsonの存在と有効性をチェック
        result_file = f"{self.workspace_base}/{tracker_id}/extraction_result.json"
        if not os.path.exists(result_file):
            return ValidationResult(False, ["extraction_result.jsonが見つかりません"])
        
        try:
            with open(result_file) as f:
                data = json.load(f)
                if data.get('successful_extractions', 0) == 0:
                    return ValidationResult(False, ["成功した抽出が記録されていません"])
        except Exception as e:
            return ValidationResult(False, [f"無効なextraction_result.json: {e}"])
        
        return ValidationResult(True, [])
    
    def validate_quality_workflow_completion(self, tracker_id: str) -> ValidationResult:
        """品質ワークフローが実際に実行されたことを確認"""
        dashboard_file = f"{self.workspace_base}/{tracker_id}/dashboard/dashboard.html"
        
        if not os.path.exists(dashboard_file):
            return ValidationResult(False, ["ダッシュボードファイルが見つかりません"])
        
        # ダッシュボードに必要なセクションが含まれているかチェック
        with open(dashboard_file) as f:
            content = f.read()
            required_sections = [
                "統計分析結果",
                "基本品質指標", 
                "品質分布",
                "画像ギャラリー"
            ]
            for section in required_sections:
                if section not in content:
                    return ValidationResult(False, [f"ダッシュボードにセクションが不足: {section}"])
        
        return ValidationResult(True, [])
```

### 3. 承認ゲートコントローラー

**目的**: 人間の承認を受けるまでの物理的ブロック

```python
class ApprovalGateController:
    def __init__(self, state_manager: WorkflowStateManager):
        self.state_manager = state_manager
        self.approval_dir = "/tmp/workflow_approvals"
        os.makedirs(self.approval_dir, exist_ok=True)
    
    def request_approval(self, tracker_id: str, step_id: str, context: dict) -> str:
        """Create approval request that physically blocks progress"""
        approval_id = f"{tracker_id}_{step_id}_{int(time.time())}"
        
        # Create approval request file
        request_file = f"{self.approval_dir}/{approval_id}_request.json"
        with open(request_file, 'w') as f:
            json.dump({
                "approval_id": approval_id,
                "tracker_id": tracker_id,
                "step_id": step_id,
                "context": context,
                "requested_at": datetime.now().isoformat(),
                "status": "pending"
            }, f, indent=2)
        
        # Block progress in state database
        self.state_manager.block_actions_until_approved(tracker_id, approval_id)
        
        # Display approval request to user
        self._display_approval_request(approval_id, context)
        
        return approval_id
    
    def check_approval_status(self, approval_id: str) -> ApprovalStatus:
        """Check if approval has been granted"""
        approval_file = f"{self.approval_dir}/{approval_id}_approved.json"
        
        if os.path.exists(approval_file):
            with open(approval_file) as f:
                approval_data = json.load(f)
                return ApprovalStatus(
                    approved=True,
                    approved_by=approval_data.get('approved_by'),
                    approved_at=approval_data.get('approved_at'),
                    comments=approval_data.get('comments', '')
                )
        
        return ApprovalStatus(approved=False)
    
    def wait_for_approval(self, approval_id: str, timeout_minutes: int = 60) -> ApprovalResult:
        """Block execution until approval is received"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_minutes * 60:
            status = self.check_approval_status(approval_id)
            if status.approved:
                # Unblock actions
                self.state_manager.unblock_actions(approval_id)
                return ApprovalResult(True, status)
            
            print(f"⏳ Waiting for approval: {approval_id}")
            print(f"   To approve, create file: {self.approval_dir}/{approval_id}_approved.json")
            time.sleep(30)  # Check every 30 seconds
        
        # Timeout - escalate
        self._escalate_approval_timeout(approval_id)
        return ApprovalResult(False, None, "Approval timeout")
    
    def _display_approval_request(self, approval_id: str, context: dict):
        """Display clear approval request to user"""
        print("\n" + "="*60)
        print("🚨 APPROVAL REQUIRED - WORKFLOW BLOCKED")
        print("="*60)
        print(f"Approval ID: {approval_id}")
        print(f"Step: {context.get('step_name', 'Unknown')}")
        print(f"Description: {context.get('description', 'No description')}")
        print("\nTo approve this step, create the following file:")
        print(f"  {self.approval_dir}/{approval_id}_approved.json")
        print("\nWith content:")
        print(json.dumps({
            "approved": True,
            "approved_by": "[Your Name]",
            "approved_at": datetime.now().isoformat(),
            "comments": "[Optional comments]"
        }, indent=2))
        print("\n⚠️  AI execution is BLOCKED until approval is received")
        print("="*60)
```

### 4. Automatic Workflow Executor

**Purpose**: Execute critical steps automatically without AI judgment

```python
class AutomaticWorkflowExecutor:
    def __init__(self, state_manager: WorkflowStateManager):
        self.state_manager = state_manager
    
    def execute_extraction_step(self, tracker_id: str) -> ExecutionResult:
        """Automatically execute extraction without AI involvement"""
        
        # Verify prerequisites
        if not self.state_manager.can_proceed_to_step(tracker_id, "extraction"):
            return ExecutionResult(False, "Prerequisites not met")
        
        # Get workspace path
        workspace = self._get_workspace_path(tracker_id)
        input_dir = "/mnt/c/AItools/lora/train/yado/org/kana05/"
        output_dir = f"{workspace}/extraction/"
        
        # Verify input directory exists
        if not os.path.exists(input_dir):
            return ExecutionResult(False, f"Input directory not found: {input_dir}")
        
        # Execute extraction command
        cmd = [
            "python3", "features/extraction/commands/extract_character.py",
            input_dir, "-o", output_dir, "--batch", "--verbose"
        ]
        
        try:
            # Run in background to avoid timeout
            log_file = f"{workspace}/extraction.log"
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    cwd="/mnt/c/AItools/segment-anything"
                )
            
            # Don't wait for completion - let it run in background
            self.state_manager.mark_step_in_progress(tracker_id, "extraction")
            
            return ExecutionResult(True, f"Extraction started (PID: {process.pid})")
            
        except Exception as e:
            return ExecutionResult(False, f"Extraction failed: {e}")
    
    def execute_quality_workflow(self, tracker_id: str) -> ExecutionResult:
        """Automatically execute quality workflow"""
        
        # Verify extraction completed first
        validator = FileSystemValidator(self._get_workspace_base())
        extraction_result = validator.validate_extraction_completion(tracker_id)
        if not extraction_result.passed:
            return ExecutionResult(False, f"Extraction not complete: {extraction_result.errors}")
        
        # Execute quality workflow
        cmd = ["./tools/scripts/run_quality_workflow.sh", tracker_id]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd="/mnt/c/AItools/segment-anything"
            )
            
            if result.returncode == 0:
                self.state_manager.mark_step_completed(tracker_id, "quality_workflow")
                return ExecutionResult(True, "Quality workflow completed")
            else:
                return ExecutionResult(False, f"Quality workflow failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, "Quality workflow timeout")
        except Exception as e:
            return ExecutionResult(False, f"Quality workflow error: {e}")
```

### 5. Simplified AI Interface

**Purpose**: Reduce cognitive load by showing only current step requirements

```python
class SimplifiedWorkflowInterface:
    def __init__(self, state_manager: WorkflowStateManager):
        self.state_manager = state_manager
    
    def get_current_step_instructions(self, tracker_id: str) -> StepInstructions:
        """Return only current step - no overwhelming context"""
        
        current_step = self.state_manager.get_current_step(tracker_id)
        
        # Load step-specific instructions
        instructions = self._load_step_instructions(current_step)
        
        return StepInstructions(
            step_id=current_step,
            title=instructions['title'],
            description=instructions['description'],
            required_actions=instructions['actions'],
            validation_criteria=instructions['validation'],
            approval_required=instructions.get('approval_required', False),
            can_proceed=self.state_manager.can_proceed_to_step(tracker_id, current_step)
        )
    
    def attempt_step_completion(self, tracker_id: str) -> StepResult:
        """Attempt to complete current step with automatic validation"""
        
        current_step = self.state_manager.get_current_step(tracker_id)
        
        # Check if step can be auto-executed
        if current_step in ['extraction', 'quality_workflow']:
            executor = AutomaticWorkflowExecutor(self.state_manager)
            if current_step == 'extraction':
                result = executor.execute_extraction_step(tracker_id)
            elif current_step == 'quality_workflow':
                result = executor.execute_quality_workflow(tracker_id)
            
            if result.success:
                return StepResult.COMPLETED(self.get_current_step_instructions(tracker_id))
            else:
                return StepResult.FAILED(result.errors)
        
        # For manual steps, validate completion
        validation_result = self.state_manager.validate_step_completion(tracker_id, current_step)
        
        if not validation_result.passed:
            return StepResult.FAILED(validation_result.errors)
        
        # Check if approval required
        step_config = self._load_step_instructions(current_step)
        if step_config.get('approval_required', False):
            approval_controller = ApprovalGateController(self.state_manager)
            approval_id = approval_controller.request_approval(
                tracker_id, current_step, step_config
            )
            return StepResult.PENDING_APPROVAL(approval_id)
        
        # Mark complete and advance
        self.state_manager.advance_to_next_step(tracker_id)
        return StepResult.COMPLETED(self.get_current_step_instructions(tracker_id))
```

## Data Models

### Workflow Configuration Schema

```json
{
  "workflow_steps": {
    "phase_1_planning": {
      "steps": [
        {
          "step_id": "sam_env_check",
          "title": "sam-env仮想環境確認",
          "description": "仮想環境がアクティベートされていることを確認",
          "validation_type": "command_check",
          "validation_command": "echo $VIRTUAL_ENV | grep sam-env",
          "approval_required": false,
          "auto_executable": false
        },
        {
          "step_id": "sow_creation", 
          "title": "SOW作成",
          "description": "作業範囲確定書を作成",
          "validation_type": "file_exists",
          "validation_path": "{workspace}/sow_document.md",
          "approval_required": true,
          "auto_executable": false
        }
      ]
    },
    "phase_2_implementation": {
      "steps": [
        {
          "step_id": "extraction",
          "title": "抽出実行",
          "description": "キャラクター抽出パイプラインを実行",
          "validation_type": "custom_validator",
          "validation_class": "FileSystemValidator.validate_extraction_completion",
          "approval_required": false,
          "auto_executable": true,
          "auto_executor": "AutomaticWorkflowExecutor.execute_extraction_step"
        }
      ]
    }
  }
}
```

## Error Handling

### Validation Failure Recovery

```python
class ValidationFailureHandler:
    def handle_validation_failure(self, tracker_id: str, step_id: str, errors: List[str]) -> RecoveryPlan:
        """Generate specific recovery instructions"""
        
        recovery_actions = []
        
        for error in errors:
            if "directory not found" in error:
                recovery_actions.append("Create missing directory structure")
            elif "file not found" in error:
                recovery_actions.append("Generate missing files")
            elif "quality threshold" in error:
                recovery_actions.append("Re-run with improved parameters")
        
        return RecoveryPlan(
            failed_step=step_id,
            errors=errors,
            recovery_actions=recovery_actions,
            escalation_threshold=3,
            auto_retry_possible=self._can_auto_retry(step_id, errors)
        )
    
    def attempt_auto_recovery(self, tracker_id: str, recovery_plan: RecoveryPlan) -> RecoveryResult:
        """Attempt automatic recovery where possible"""
        
        if recovery_plan.auto_retry_possible:
            if "extraction" in recovery_plan.failed_step:
                # Retry extraction with different parameters
                executor = AutomaticWorkflowExecutor(self.state_manager)
                return executor.execute_extraction_step(tracker_id)
        
        return RecoveryResult(False, "Manual intervention required")
```

## Testing Strategy

### Integration Testing Approach

1. **End-to-End Workflow Tests**: Complete workflow execution with all enforcement mechanisms
2. **Bypass Prevention Tests**: Verify AI cannot circumvent controls through various methods
3. **State Consistency Tests**: Ensure external state remains accurate across sessions
4. **Approval Flow Tests**: Test human approval integration and timeout handling
5. **Automatic Execution Tests**: Verify critical steps execute without AI involvement

### Performance Considerations

- **Database Operations**: Use SQLite for lightweight, file-based state management
- **File System Checks**: Optimize validation queries to minimize I/O overhead  
- **Background Execution**: Use subprocess for long-running tasks to avoid timeouts
- **Caching**: Cache validation results to avoid repeated expensive checks

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Implement WorkflowStateManager with SQLite backend
- Create FileSystemValidator for extraction and quality validation
- Build ApprovalGateController with file-based approval system
- Integrate with existing `.claude/hooks.json` system

### Phase 2: Automatic Execution (Week 3-4)  
- Implement AutomaticWorkflowExecutor for extraction and quality workflows
- Create SimplifiedWorkflowInterface for reduced cognitive load
- Add comprehensive error handling and recovery mechanisms
- Integrate with existing SubAgent and queue systems

### Phase 3: Testing and Refinement (Week 5-6)
- Comprehensive testing of all enforcement mechanisms
- Performance optimization and caching implementation
- User interface improvements for approval workflows
- Documentation and training materials

### Phase 4: Deployment and Monitoring (Week 7-8)
- Gradual rollout with existing tracker workflows
- Monitor compliance rates and system effectiveness
- Collect user feedback and iterate on design
- Establish long-term maintenance procedures

This design fundamentally shifts from trusting AI behavior to mechanically enforcing it, ensuring consistent workflow compliance regardless of AI cognitive limitations.