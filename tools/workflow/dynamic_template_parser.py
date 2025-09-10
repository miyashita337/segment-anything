"""
INTG-087: 動的テンプレートパーサーシステム
unified_tracker_template.md からセマンティックルールを動的抽出
ハードコーディング排除・ドキュメント変更への自動対応

設計原則:
- 意味論的パターンマッチング
- テンプレート構造変更への resilience  
- 人間が理解しやすい自然なロジック
"""

import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ApprovalType(Enum):
    """承認の種類を意味論的に分類"""
    PLANNING = "planning"           # 計画承認
    IMPLEMENTATION = "implementation"  # 実装方針承認
    TESTING = "testing"            # テスト結果承認
    QUALITY = "quality"            # 品質ワークフロー承認
    FINAL = "final"                # 最終承認


class ExecutionContext(Enum):
    """実行コンテキストの分類"""
    EXTRACTION = "extraction"      # データ抽出実行
    QUALITY_WORKFLOW = "quality_workflow"  # 品質ワークフロー
    TESTING = "testing"            # テスト実行
    DEPLOYMENT = "deployment"      # デプロイメント


@dataclass
class SemanticRule:
    """セマンティックルールの定義"""
    context: ExecutionContext
    required_capabilities: List[str]
    approval_requirements: List[ApprovalType]
    phase_dependencies: List[str]
    description: str


@dataclass
class ParsedApproval:
    """解析された承認情報"""
    approval_type: ApprovalType
    description: str
    context: str
    requirements: List[str]


class DynamicTemplateParser:
    """動的テンプレート解析エンジン"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: execution_rules.yaml へのパス
        """
        if config_path is None:
            config_path = "/mnt/c/AItools/segment-anything/config/execution_rules.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # セマンティックパターン定義
        self.semantic_patterns = {
            # 実装完了パターン
            "implementation_completion": [
                r"実装完了",
                r"core_implementation.*完了",
                r"実装作業.*完了",
                r"Implementation.*completed",
                r"コア機能実装.*✅"
            ],
            
            # 抽出実行承認パターン
            "extraction_approval": [
                r"抽出実行.*承認",
                r"extraction.*approval",
                r"抽出.*許可",
                r"抽出パイプライン.*実行",
                r"extraction.*execution.*approved"
            ],
            
            # フェーズ完了パターン
            "phase_completion": [
                r"Phase\s+\d+.*完了",
                r"フェーズ.*完了",
                r"Phase.*completion",
                r"ステップ\s+\d+-\d+.*完了"
            ],
            
            # 承認段階パターン
            "approval_stage": [
                r"承認\d+[:：]?\s*(.+?)承認",
                r"Approval\s+\d+[:：]?\s*(.+?)approval",
                r"承認段階\s*\d+[:：]?\s*(.+)"
            ]
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    def parse_unified_template(self, template_path: str = None) -> Dict[str, SemanticRule]:
        """
        unified_tracker_template.md を動的解析してセマンティックルールを抽出
        
        Args:
            template_path: テンプレートファイルのパス
            
        Returns:
            抽出されたセマンティックルール辞書
        """
        if template_path is None:
            template_path = self.config.get(
                'template_parsing', {}
            ).get('unified_tracker_template_path')
            
        if not template_path or not Path(template_path).exists():
            logger.error(f"テンプレートファイルが見つかりません: {template_path}")
            return {}
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # セマンティックルール抽出
            rules = {}
            
            # 抽出実行ルール解析
            extraction_rule = self._parse_extraction_rule(content)
            if extraction_rule:
                rules["extraction"] = extraction_rule
                
            # 品質ワークフロールール解析
            quality_rule = self._parse_quality_workflow_rule(content)
            if quality_rule:
                rules["quality_workflow"] = quality_rule
            
            # 承認システム解析
            approval_rules = self._parse_approval_system(content)
            rules.update(approval_rules)
            
            return rules
            
        except Exception as e:
            logger.error(f"テンプレート解析エラー: {e}")
            return {}
    
    def _parse_extraction_rule(self, content: str) -> Optional[SemanticRule]:
        """抽出実行ルールの解析"""
        # Phase 2 の抽出実行パターンを検索
        phase2_pattern = r"Phase 2.*?抽出実行.*?実装完了後.*?input→output抽出パイプライン実行"
        extraction_match = re.search(phase2_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if extraction_match:
            # 承認2の要求事項を検索
            approval2_pattern = r"承認2.*?抽出実行承認.*?input/output指定.*?バックグラウンド実行許可"
            approval_match = re.search(approval2_pattern, content, re.DOTALL | re.IGNORECASE)
            
            capabilities = ["implementation_completion"]
            approvals = [ApprovalType.IMPLEMENTATION]
            
            if approval_match:
                capabilities.append("extraction_execution_approval")
            
            return SemanticRule(
                context=ExecutionContext.EXTRACTION,
                required_capabilities=capabilities,
                approval_requirements=approvals,
                phase_dependencies=["phase_2_implementation"],
                description="キャラクター抽出実行のセマンティックルール"
            )
        
        return None
    
    def _parse_quality_workflow_rule(self, content: str) -> Optional[SemanticRule]:
        """品質ワークフロールールの解析"""
        # Phase 3 の品質ワークフロー実行パターンを検索
        quality_pattern = r"Phase 3.*?品質ワークフロー.*?run_quality_workflow\.sh"
        quality_match = re.search(quality_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if quality_match:
            return SemanticRule(
                context=ExecutionContext.QUALITY_WORKFLOW,
                required_capabilities=["extraction_completion", "phase_2_completion"],
                approval_requirements=[ApprovalType.TESTING],
                phase_dependencies=["phase_2_implementation"],
                description="品質ワークフロー実行のセマンティックルール"
            )
        
        return None
    
    def _parse_approval_system(self, content: str) -> Dict[str, SemanticRule]:
        """5段階承認システムの解析"""
        rules = {}
        
        # 承認段階を動的抽出
        approval_pattern = r"承認(\d+)[:：]?\s*(.+?)承認.*?-.*?承認基準[:：]?\s*(.+?)(?=###|\n\n|承認\d+|$)"
        
        for match in re.finditer(approval_pattern, content, re.DOTALL | re.IGNORECASE):
            approval_num = int(match.group(1))
            approval_desc = match.group(2).strip()
            criteria = match.group(3).strip()
            
            # 承認の種類を意味論的に分類
            approval_type = self._classify_approval_type(approval_desc, criteria)
            
            rule_key = f"approval_{approval_num}_{approval_type.value}"
            rules[rule_key] = SemanticRule(
                context=ExecutionContext.DEPLOYMENT,  # 承認はデプロイメントコンテキスト
                required_capabilities=[f"{approval_type.value}_completion"],
                approval_requirements=[approval_type],
                phase_dependencies=[],
                description=f"承認{approval_num}: {approval_desc}"
            )
        
        return rules
    
    def _classify_approval_type(self, description: str, criteria: str) -> ApprovalType:
        """承認の説明と基準から承認タイプを分類"""
        text = f"{description} {criteria}".lower()
        
        if any(word in text for word in ["計画", "planning", "設計", "アーキテクチャ"]):
            return ApprovalType.PLANNING
        elif any(word in text for word in ["実装", "implementation", "ワークスペース", "抽出実行"]):
            return ApprovalType.IMPLEMENTATION  
        elif any(word in text for word in ["テスト", "test", "品質メトリクス", "パフォーマンス"]):
            return ApprovalType.TESTING
        elif any(word in text for word in ["品質", "quality", "ダッシュボード", "統計分析"]):
            return ApprovalType.QUALITY
        elif any(word in text for word in ["最終", "final", "total", "PR", "リリース"]):
            return ApprovalType.FINAL
        else:
            return ApprovalType.IMPLEMENTATION  # デフォルト
    
    def get_execution_requirements(self, command: str) -> Optional[SemanticRule]:
        """
        コマンドに対する実行要件をセマンティックに取得
        
        Args:
            command: 実行コマンド
            
        Returns:
            該当するセマンティックルール
        """
        # テンプレートから動的にルールを取得
        rules = self.parse_unified_template()
        
        # コマンドのコンテキストを理解
        context = self._analyze_command_context(command)
        
        # 該当するルールを検索
        for rule_name, rule in rules.items():
            if rule.context == context:
                return rule
        
        # 設定ファイルからのフォールバック
        return self._get_fallback_rule(command)
    
    def _analyze_command_context(self, command: str) -> ExecutionContext:
        """コマンドの実行コンテキストを分析"""
        if "extract_character" in command:
            return ExecutionContext.EXTRACTION
        elif "run_quality_workflow" in command:
            return ExecutionContext.QUALITY_WORKFLOW
        elif "pytest" in command or "test" in command:
            return ExecutionContext.TESTING
        else:
            return ExecutionContext.DEPLOYMENT
    
    def _get_fallback_rule(self, command: str) -> Optional[SemanticRule]:
        """設定ファイルからのフォールバックルール取得"""
        execution_rules = self.config.get('execution_rules', {})
        
        for rule_name, rule_config in execution_rules.items():
            patterns = rule_config.get('command_patterns', [])
            
            if any(self._match_pattern(pattern, command) for pattern in patterns):
                return self._convert_config_to_rule(rule_config)
        
        return None
    
    def _match_pattern(self, pattern: str, text: str) -> bool:
        """パターンマッチング"""
        # シンプルなワイルドカードマッチング
        pattern_regex = pattern.replace('*', '.*')
        return re.search(pattern_regex, text, re.IGNORECASE) is not None
    
    def _convert_config_to_rule(self, rule_config: Dict[str, Any]) -> SemanticRule:
        """設定からSemanticRuleに変換"""
        prerequisites = rule_config.get('semantic_prerequisites', [])
        
        capabilities = []
        approvals = []
        phases = []
        
        for prereq in prerequisites:
            capability = prereq.get('capability')
            if capability:
                capabilities.append(capability)
                
            # 承認タイプの推定
            if 'approval' in capability:
                approvals.append(ApprovalType.IMPLEMENTATION)
            
            # フェーズ依存の推定
            target_phase = prereq.get('target_phase')
            if target_phase:
                phases.append(target_phase)
        
        return SemanticRule(
            context=ExecutionContext.EXTRACTION,  # デフォルト
            required_capabilities=capabilities,
            approval_requirements=approvals,
            phase_dependencies=phases,
            description=rule_config.get('description', '')
        )