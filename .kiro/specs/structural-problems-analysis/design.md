# 設計書

## 概要

この設計は、包括的なワークフロードキュメントと制御メカニズムが一貫したAIエージェントの動作を確保できないアニメキャラクター抽出プロジェクトの根本的な構造問題に対処します。分析により、詳細な13ステップワークフロー、5段階承認システム、広範なチェックリスト、複数の参照ドキュメントがあるにもかかわらず、AIエージェントは非冪等的動作を継続して示すことが明らかになりました - 独立した決定を行い、ステップをスキップし、承認要件を迂回しています。

核心的な問題はアーキテクチャ的です：現在のシステムは手順遵守において本質的に信頼できないAI認知能力（注意、記憶、判断）に依存しています。解決策は、AI依存の制御から外部の機械的強制システムへの移行を必要とします。

## アーキテクチャ

### 現在のシステム分析

既存のシステムは以下の特徴を持つ**認知過負荷アーキテクチャ**を示しています：

```
ドキュメント層（重い）
├── CLAUDE.md（2,000行以上の指示）
├── 13ステップワークフローチェックリスト
├── 5段階承認システム
├── 複数の参照ドキュメント（10以上のファイル）
├── 専門チェックリストとテンプレート
└── 例外処理手順

AIエージェント処理層（信頼できない）
├── 注意メカニズム（確率的）
├── コンテキスト管理（劣化する）
├── 優先度解決（一貫性がない）
├── メモリシステム（セッション制限）
└── 判断呼び出し（可変）

実施層（弱い）
├── 人間承認ポイント（迂回可能）
├── ドキュメント参照（無視可能）
├── チェックリスト項目（スキップ可能）
└── プロセスガイドライン（解釈可能）
```

### 提案アーキテクチャ：外部状態管理

解決策は重要な制御をAIエージェントの認知領域外に移行します：

```
外部制御層（機械的）
├── 状態管理データベース
├── ワークフロー強制エンジン
├── 自動検証ゲート
├── 必須人間チェックポイント
└── 進捗追跡システム

AIエージェントインターフェース層（制約付き）
├── 制限されたアクション範囲
├── 強制検証呼び出し
├── 状態クエリ要件
├── 承認待機状態
└── エラー防止ブロック

人間監視層（強化）
├── 明示的承認インターフェース
├── 進捗可視化ダッシュボード
├── オーバーライドメカニズム
├── 品質ゲート
└── エスカレーション手順
```

## コンポーネントとインターフェース

### 1. ワークフロー状態マネージャー

**目的**: AIの判断によって迂回できないワークフロー進捗の外部追跡

```python
class WorkflowStateManager:
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.current_phase = None
        self.completed_steps = set()
        self.approval_status = {}
        self.blocked_actions = set()
    
    def can_proceed_to_step(self, step_id: str) -> bool:
        """Mechanical check - no AI interpretation allowed"""
        return self._check_prerequisites(step_id)
    
    def require_approval(self, approval_point: str) -> ApprovalStatus:
        """Force human approval - cannot be bypassed"""
        return self._wait_for_human_approval(approval_point)
    
    def validate_completion(self, step_id: str) -> ValidationResult:
        """Automatic validation of step completion"""
        return self._run_validation_checks(step_id)
```

### 2. 承認強制システム

**目的**: スキップまたは仮定できない必須の人間チェックポイント

```python
class ApprovalEnforcement:
    def __init__(self):
        self.pending_approvals = {}
        self.approval_history = []
    
    def request_approval(self, context: ApprovalContext) -> ApprovalToken:
        """Create approval request that blocks progress"""
        token = self._create_approval_request(context)
        self._block_progress_until_approved(token)
        return token
    
    def check_approval_status(self, token: ApprovalToken) -> ApprovalStatus:
        """Non-bypassable approval check"""
        return self._get_approval_status(token)
    
    def wait_for_approval(self, token: ApprovalToken) -> ApprovalResult:
        """Forced wait state - AI cannot proceed without approval"""
        while not self._is_approved(token):
            self._display_waiting_message()
            time.sleep(30)  # Force wait
        return self._get_approval_result(token)
```

### 3. 検証ゲートシステム

**目的**: 適切な完了なしに進行を防ぐ自動チェック

```python
class ValidationGates:
    def __init__(self):
        self.validators = {}
        self.gate_status = {}
    
    def register_gate(self, step_id: str, validator: Callable) -> None:
        """Register automatic validation for a step"""
        self.validators[step_id] = validator
    
    def check_gate(self, step_id: str, context: dict) -> GateResult:
        """Automatic validation - no AI interpretation"""
        validator = self.validators.get(step_id)
        if not validator:
            return GateResult.FAIL("No validator registered")
        
        try:
            result = validator(context)
            return GateResult.PASS() if result else GateResult.FAIL("Validation failed")
        except Exception as e:
            return GateResult.ERROR(str(e))
    
    def enforce_gate(self, step_id: str, context: dict) -> None:
        """Block progress until gate passes"""
        result = self.check_gate(step_id, context)
        if not result.passed:
            raise ValidationError(f"Gate {step_id} failed: {result.message}")
```

### 4. 簡素化ワークフローインターフェース

**目的**: 現在のステップ要件のみを提示することで認知負荷を軽減

```python
class SimplifiedWorkflowInterface:
    def __init__(self, state_manager: WorkflowStateManager):
        self.state_manager = state_manager
    
    def get_current_step(self) -> WorkflowStep:
        """Return only the current step - no overwhelming context"""
        current_step = self.state_manager.get_current_step()
        return WorkflowStep(
            id=current_step.id,
            title=current_step.title,
            description=current_step.description,
            required_actions=current_step.actions,
            validation_criteria=current_step.validation,
            next_step_preview=current_step.next_step
        )
    
    def complete_current_step(self, completion_data: dict) -> StepResult:
        """Attempt to complete current step with validation"""
        current_step = self.get_current_step()
        
        # Automatic validation
        validation_result = self.state_manager.validate_completion(
            current_step.id, completion_data
        )
        
        if not validation_result.passed:
            return StepResult.FAILED(validation_result.errors)
        
        # Check if approval required
        if current_step.requires_approval:
            approval_token = self.state_manager.require_approval(current_step.id)
            return StepResult.PENDING_APPROVAL(approval_token)
        
        # Mark complete and advance
        self.state_manager.mark_complete(current_step.id)
        return StepResult.COMPLETED(self.get_current_step())
```

## データモデル

### ワークフロー状態スキーマ

```json
{
  "tracker_id": "STRUCT-001",
  "workflow_version": "2.0",
  "current_phase": "phase_1",
  "current_step": "step_3",
  "completed_steps": ["step_1", "step_2"],
  "pending_approvals": [
    {
      "approval_id": "approval_001",
      "step_id": "step_3",
      "requested_at": "2025-01-15T10:30:00Z",
      "status": "pending",
      "context": {
        "description": "SOW approval required",
        "artifacts": ["sow_document.md"]
      }
    }
  ],
  "validation_results": {
    "step_1": {"status": "passed", "timestamp": "2025-01-15T09:15:00Z"},
    "step_2": {"status": "passed", "timestamp": "2025-01-15T10:00:00Z"}
  },
  "blocked_actions": ["proceed_to_implementation", "skip_approval"],
  "metadata": {
    "created_at": "2025-01-15T09:00:00Z",
    "last_updated": "2025-01-15T10:30:00Z",
    "ai_session_id": "session_123"
  }
}
```

### 承認要求スキーマ

```json
{
  "approval_id": "approval_001",
  "tracker_id": "STRUCT-001",
  "step_id": "step_3",
  "approval_type": "sow_approval",
  "title": "SOW Document Approval Required",
  "description": "Please review and approve the Statement of Work before proceeding to implementation",
  "artifacts": [
    {
      "name": "sow_document.md",
      "path": "/workspace/STRUCT-001/sow_document.md",
      "checksum": "abc123"
    }
  ],
  "requested_by": "claude_agent",
  "requested_at": "2025-01-15T10:30:00Z",
  "status": "pending",
  "approval_criteria": [
    "SOW scope is clearly defined",
    "Deliverables are specific and measurable",
    "Timeline is realistic"
  ]
}
```

## エラーハンドリング

### 検証失敗復旧

```python
class ValidationFailureHandler:
    def handle_validation_failure(self, step_id: str, errors: List[str]) -> RecoveryPlan:
        """Generate recovery plan for validation failures"""
        return RecoveryPlan(
            failed_step=step_id,
            errors=errors,
            recovery_actions=[
                "Review step requirements",
                "Fix identified issues", 
                "Re-run validation",
                "Request human assistance if needed"
            ],
            escalation_threshold=3  # Escalate after 3 failures
        )
    
    def escalate_to_human(self, context: EscalationContext) -> None:
        """Force human intervention for repeated failures"""
        self._create_escalation_ticket(context)
        self._block_ai_progress()
        self._notify_human_operator()
```

### 承認タイムアウト処理

```python
class ApprovalTimeoutHandler:
    def handle_approval_timeout(self, approval_id: str) -> TimeoutAction:
        """Handle approval requests that timeout"""
        approval = self._get_approval_request(approval_id)
        
        if approval.priority == "critical":
            return TimeoutAction.ESCALATE_IMMEDIATELY
        elif approval.age > timedelta(hours=24):
            return TimeoutAction.SEND_REMINDER
        else:
            return TimeoutAction.CONTINUE_WAITING
```

## テスト戦略

### 単体テストアプローチ

1. **状態マネージャーテスト**: 状態遷移と検証ロジックの確認
2. **承認システムテスト**: 承認要求/応答サイクルのテスト
3. **検証ゲートテスト**: ゲートが無効な進行を適切にブロックすることを確認
4. **インターフェーステスト**: 簡素化されたインターフェースが認知負荷を軽減することを確認

### 統合テストアプローチ

1. **エンドツーエンドワークフローテスト**: 外部制御による完全なワークフロー実行
2. **障害復旧テスト**: 様々な障害条件下でのシステム動作テスト
3. **承認フローテスト**: 人間承認統合のテスト
4. **パフォーマンステスト**: 外部制御がパフォーマンスに大きく影響しないことを確認

### 行動テスト

1. **AI遵守テスト**: AIが制御を迂回できないことを確認
2. **冪等性テスト**: セッション間での一貫した動作を確保
3. **ストレステスト**: 高認知負荷シナリオ下でのシステムテスト
4. **エッジケーステスト**: 異常だが可能なワークフローシナリオのテスト

## 実装フェーズ

### フェーズ1: コアインフラストラクチャ（第1-2週）
- WorkflowStateManagerの実装
- 基本ApprovalEnforcementシステムの作成
- ValidationGatesフレームワークの構築
- 外部状態ストレージの設定

### フェーズ2: ワークフロー統合（第3-4週）
- 既存の13ステップワークフローと新しい制御の統合
- 必須承認ポイントの実装
- 簡素化されたAIインターフェースの作成
- 自動検証チェックの追加

### フェーズ3: テストと改良（第5-6週）
- すべてのコンポーネントの包括的テスト
- パフォーマンス最適化
- ユーザーインターフェースの改善
- ドキュメントの更新

### フェーズ4: デプロイメントと監視（第7-8週）
- 既存ワークフローでの段階的ロールアウト
- AI遵守率の監視
- ユーザーフィードバックの収集
- 実世界の使用に基づく設計の反復

この設計は、重要な決定ポイントをAIの認知領域から除去し、AIの判断や注意の失敗によって迂回できない外部の機械的システムに配置することで、AI非冪等性の根本的問題に対処します。