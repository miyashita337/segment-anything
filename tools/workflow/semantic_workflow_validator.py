"""
INTG-087: セマンティックワークフロー検証システム
意味論的判定による柔軟なワークフロー制御

設計原則:
- ハードコーディング完全排除
- ドキュメント構造変更への自動適応
- 人間理解可能な自然なルール記述
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .dynamic_template_parser import DynamicTemplateParser, SemanticRule, ExecutionContext, ApprovalType

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """検証結果の分類"""
    ALLOWED = "allowed"            # 実行許可
    BLOCKED = "blocked"            # 実行阻止
    REQUIRES_APPROVAL = "requires_approval"  # 承認必要
    PHASE_MISMATCH = "phase_mismatch"        # フェーズ不適合


@dataclass
class ValidationContext:
    """検証コンテキスト情報"""
    current_phase: str
    completed_steps: List[str]
    current_capabilities: List[str]
    approval_status: Dict[str, bool]
    command: str
    execution_context: ExecutionContext


@dataclass
class SemanticValidationResult:
    """セマンティック検証結果"""
    result: ValidationResult
    reason: str
    required_actions: List[str]
    blocking_factors: List[str]
    semantic_analysis: Dict[str, Any]


class SemanticWorkflowValidator:
    """セマンティックワークフロー検証エンジン"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: execution_rules.yaml へのパス
        """
        self.template_parser = DynamicTemplateParser(config_path)
        self.config = self.template_parser.config
        
        # セマンティックルールのキャッシュ
        self._semantic_rules_cache = None
        
        # 現在の検証コンテキスト
        self.current_context = None
    
    def validate_execution(self, command: str, context: ValidationContext = None) -> SemanticValidationResult:
        """
        コマンド実行の検証（セマンティック判定）
        
        Args:
            command: 実行予定のコマンド
            context: 検証コンテキスト
            
        Returns:
            セマンティック検証結果
        """
        # コンテキスト取得・推定
        if context is None:
            context = self._infer_validation_context(command)
        
        self.current_context = context
        
        # セマンティックルール取得
        semantic_rule = self.template_parser.get_execution_requirements(command)
        
        if semantic_rule is None:
            return self._create_default_validation_result(command, context)
        
        # 意味論的検証実行
        return self._perform_semantic_validation(semantic_rule, context)
    
    def _infer_validation_context(self, command: str) -> ValidationContext:
        """コマンドから検証コンテキストを推定"""
        # Phase情報の動的取得
        current_phase = self._get_current_phase()
        
        # 完了ステップの動的確認
        completed_steps = self._get_completed_steps()
        
        # 現在の能力状況
        current_capabilities = self._analyze_current_capabilities()
        
        # 承認状況の動的確認
        approval_status = self._get_approval_status()
        
        # 実行コンテキストの推定
        execution_context = self.template_parser._analyze_command_context(command)
        
        return ValidationContext(
            current_phase=current_phase,
            completed_steps=completed_steps,
            current_capabilities=current_capabilities,
            approval_status=approval_status,
            command=command,
            execution_context=execution_context
        )
    
    def _perform_semantic_validation(
        self, 
        semantic_rule: SemanticRule, 
        context: ValidationContext
    ) -> SemanticValidationResult:
        """セマンティックルールに基づく検証実行"""
        
        # 段階的検証実行
        capability_result = self._validate_capabilities(semantic_rule, context)
        approval_result = self._validate_approvals(semantic_rule, context)
        phase_result = self._validate_phase_dependencies(semantic_rule, context)
        
        # 総合判定
        overall_result = self._synthesize_validation_results(
            capability_result, approval_result, phase_result, semantic_rule, context
        )
        
        return overall_result
    
    def _validate_capabilities(
        self, 
        semantic_rule: SemanticRule, 
        context: ValidationContext
    ) -> Tuple[bool, List[str], List[str]]:
        """必要能力の検証"""
        missing_capabilities = []
        satisfied_capabilities = []
        
        for required_capability in semantic_rule.required_capabilities:
            if self._check_capability_satisfaction(required_capability, context):
                satisfied_capabilities.append(required_capability)
            else:
                missing_capabilities.append(required_capability)
        
        is_satisfied = len(missing_capabilities) == 0
        return is_satisfied, satisfied_capabilities, missing_capabilities
    
    def _validate_approvals(
        self, 
        semantic_rule: SemanticRule, 
        context: ValidationContext
    ) -> Tuple[bool, List[str], List[str]]:
        """承認要件の検証"""
        missing_approvals = []
        satisfied_approvals = []
        
        for required_approval in semantic_rule.approval_requirements:
            approval_key = f"{required_approval.value}_approval"
            
            if context.approval_status.get(approval_key, False):
                satisfied_approvals.append(required_approval.value)
            else:
                missing_approvals.append(required_approval.value)
        
        is_satisfied = len(missing_approvals) == 0
        return is_satisfied, satisfied_approvals, missing_approvals
    
    def _validate_phase_dependencies(
        self, 
        semantic_rule: SemanticRule, 
        context: ValidationContext
    ) -> Tuple[bool, List[str], List[str]]:
        """フェーズ依存関係の検証"""
        missing_phases = []
        satisfied_phases = []
        
        for required_phase in semantic_rule.phase_dependencies:
            if self._check_phase_completion(required_phase, context):
                satisfied_phases.append(required_phase)
            else:
                missing_phases.append(required_phase)
        
        is_satisfied = len(missing_phases) == 0
        return is_satisfied, satisfied_phases, missing_phases
    
    def _check_capability_satisfaction(self, capability: str, context: ValidationContext) -> bool:
        """能力満足状況の確認"""
        # セマンティックパターンマッチング
        patterns = self.template_parser.semantic_patterns
        
        if capability == "implementation_completion":
            # 実装完了の意味論的確認
            impl_patterns = patterns.get("implementation_completion", [])
            return any(
                self._semantic_pattern_match(pattern, context.completed_steps)
                for pattern in impl_patterns
            )
        
        elif capability == "extraction_execution_approval":
            # 抽出実行承認の意味論的確認
            approval_patterns = patterns.get("extraction_approval", [])
            return any(
                self._semantic_pattern_match(pattern, context.current_capabilities)
                for pattern in approval_patterns
            )
        
        elif capability == "extraction_completion":
            # 抽出完了の確認
            return "extraction_completed" in context.current_capabilities
            
        elif capability == "phase_2_completion":
            # Phase 2完了の確認
            return self._check_phase_completion("phase_2", context)
        
        # デフォルト: キーワードベース確認
        return capability in context.current_capabilities
    
    def _check_phase_completion(self, phase: str, context: ValidationContext) -> bool:
        """フェーズ完了状況の確認"""
        if phase == "phase_2_implementation":
            # Phase 2実装完了の意味論的確認
            phase_patterns = self.template_parser.semantic_patterns.get("phase_completion", [])
            return any(
                self._semantic_pattern_match(f"Phase 2.*{pattern}", context.completed_steps)
                for pattern in ["完了", "completion"]
            )
        
        # 一般的なフェーズ完了確認
        return phase in context.completed_steps
    
    def _semantic_pattern_match(self, pattern: str, text_list: List[str]) -> bool:
        """セマンティックパターンマッチング"""
        import re
        
        combined_text = " ".join(text_list)
        return re.search(pattern, combined_text, re.IGNORECASE | re.MULTILINE) is not None
    
    def _synthesize_validation_results(
        self,
        capability_result: Tuple[bool, List[str], List[str]],
        approval_result: Tuple[bool, List[str], List[str]],
        phase_result: Tuple[bool, List[str], List[str]],
        semantic_rule: SemanticRule,
        context: ValidationContext
    ) -> SemanticValidationResult:
        """検証結果の総合判定"""
        
        cap_satisfied, cap_satisfied_list, cap_missing = capability_result
        app_satisfied, app_satisfied_list, app_missing = approval_result
        phase_satisfied, phase_satisfied_list, phase_missing = phase_result
        
        # 意味論的分析情報
        semantic_analysis = {
            "rule_description": semantic_rule.description,
            "execution_context": semantic_rule.context.value,
            "satisfied_capabilities": cap_satisfied_list,
            "missing_capabilities": cap_missing,
            "satisfied_approvals": app_satisfied_list,
            "missing_approvals": app_missing,
            "satisfied_phases": phase_satisfied_list,
            "missing_phases": phase_missing,
            "current_phase": context.current_phase
        }
        
        # 総合判定ロジック
        if cap_satisfied and app_satisfied and phase_satisfied:
            return SemanticValidationResult(
                result=ValidationResult.ALLOWED,
                reason="すべてのセマンティック条件が満たされています",
                required_actions=[],
                blocking_factors=[],
                semantic_analysis=semantic_analysis
            )
        
        # 部分的な条件不満足の場合
        blocking_factors = []
        required_actions = []
        
        if not cap_satisfied:
            blocking_factors.extend([f"能力不足: {cap}" for cap in cap_missing])
            required_actions.extend([f"{cap}の完了が必要" for cap in cap_missing])
        
        if not app_satisfied:
            blocking_factors.extend([f"承認未取得: {app}" for app in app_missing])
            required_actions.extend([f"{app}承認の取得が必要" for app in app_missing])
        
        if not phase_satisfied:
            blocking_factors.extend([f"フェーズ未完了: {phase}" for phase in phase_missing])
            required_actions.extend([f"{phase}の完了が必要" for phase in phase_missing])
        
        # 結果タイプの決定
        if not phase_satisfied:
            result_type = ValidationResult.PHASE_MISMATCH
            reason = f"フェーズ依存関係が満たされていません: {', '.join(phase_missing)}"
        elif not app_satisfied:
            result_type = ValidationResult.REQUIRES_APPROVAL
            reason = f"承認が必要です: {', '.join(app_missing)}"
        else:
            result_type = ValidationResult.BLOCKED
            reason = f"必要条件が満たされていません: {', '.join(cap_missing)}"
        
        return SemanticValidationResult(
            result=result_type,
            reason=reason,
            required_actions=required_actions,
            blocking_factors=blocking_factors,
            semantic_analysis=semantic_analysis
        )
    
    def _create_default_validation_result(
        self, 
        command: str, 
        context: ValidationContext
    ) -> SemanticValidationResult:
        """デフォルト検証結果の作成（ルールが見つからない場合）"""
        
        # フォールバック: 設定ファイルベースの判定
        fallback_rule = self.template_parser._get_fallback_rule(command)
        
        if fallback_rule:
            return self._perform_semantic_validation(fallback_rule, context)
        
        # 最終フォールバック: 保守的な判定
        return SemanticValidationResult(
            result=ValidationResult.BLOCKED,
            reason="該当するセマンティックルールが見つかりません",
            required_actions=["適切なセマンティックルールの定義が必要"],
            blocking_factors=["ルール未定義"],
            semantic_analysis={
                "command": command,
                "fallback_reason": "no_semantic_rule_found",
                "current_phase": context.current_phase
            }
        )
    
    def _get_current_phase(self) -> str:
        """現在のフェーズを動的取得"""
        try:
            # トラッカー状況ファイルから読み込み
            status_files = [
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/status.json",
                "/tmp/current_phase.json"
            ]
            
            from glob import glob
            for pattern in status_files:
                files = glob(pattern)
                if files:
                    with open(files[0], 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        return status.get("current_phase", "phase_0")
            
            # デフォルト値
            return "phase_0"
            
        except Exception as e:
            logger.warning(f"フェーズ情報取得エラー: {e}")
            return "phase_0"
    
    def _get_completed_steps(self) -> List[str]:
        """完了ステップリストを動的取得"""
        try:
            # Google Sheets連携や進捗ファイルから取得
            progress_files = [
                "/tmp/workflow_progress.json",
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/progress.json"
            ]
            
            from glob import glob
            for pattern in progress_files:
                files = glob(pattern)
                if files:
                    with open(files[0], 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                        return progress.get("completed_steps", [])
            
            return []
            
        except Exception as e:
            logger.warning(f"完了ステップ情報取得エラー: {e}")
            return []
    
    def _analyze_current_capabilities(self) -> List[str]:
        """現在の能力状況を分析"""
        try:
            # システム状態の動的解析
            capabilities = []
            
            # 実装完了チェック
            impl_markers = [
                "/tmp/implementation_complete.marker",
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/implementation_complete"
            ]
            
            from glob import glob
            for pattern in impl_markers:
                if glob(pattern):
                    capabilities.append("implementation_completion")
                    break
            
            # 抽出完了チェック
            extraction_outputs = [
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/extraction/*.jpg",
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/extraction/*.png"
            ]
            
            for pattern in extraction_outputs:
                if glob(pattern):
                    capabilities.append("extraction_completion")
                    break
            
            return capabilities
            
        except Exception as e:
            logger.warning(f"能力状況分析エラー: {e}")
            return []
    
    def _get_approval_status(self) -> Dict[str, bool]:
        """承認状況を動的取得"""
        try:
            # 承認ファイルから状況確認
            approval_files = [
                "/tmp/approvals.json",
                "/mnt/c/AItools/lora/train/yado/tracker-workspace/*/approvals.json"
            ]
            
            from glob import glob
            for pattern in approval_files:
                files = glob(pattern)
                if files:
                    with open(files[0], 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            # デフォルト: すべて未承認
            return {
                "planning_approval": False,
                "implementation_approval": False,
                "testing_approval": False,
                "quality_approval": False,
                "final_approval": False
            }
            
        except Exception as e:
            logger.warning(f"承認状況取得エラー: {e}")
            return {}


def validate_command_execution(command: str) -> SemanticValidationResult:
    """
    コマンド実行検証のエントリーポイント
    
    Args:
        command: 検証対象コマンド
        
    Returns:
        セマンティック検証結果
    """
    validator = SemanticWorkflowValidator()
    return validator.validate_execution(command)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    test_commands = [
        "python features/extraction/commands/extract_character.py --batch",
        "bash tools/scripts/run_quality_workflow.sh",
        "python tools/progress_tracker/cli.py update"
    ]
    
    validator = SemanticWorkflowValidator()
    
    for cmd in test_commands:
        print(f"\n=== Testing Command: {cmd} ===")
        result = validator.validate_execution(cmd)
        print(f"Result: {result.result.value}")
        print(f"Reason: {result.reason}")
        if result.required_actions:
            print(f"Required Actions: {', '.join(result.required_actions)}")
        if result.blocking_factors:
            print(f"Blocking Factors: {', '.join(result.blocking_factors)}")