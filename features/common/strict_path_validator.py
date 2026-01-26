#!/usr/bin/env python3
"""
厳密パス検証システム (QUAL-033)

ワークフローの意図しない挙動を防ぐため、全てのパス指定を厳密に検証し、
デフォルト値やフォールバックを完全に排除する検証システム。

Created for: QUAL-033 - 厳密パス検証システム実装・全ワークフロー適用・意図しない挙動防止
Author: Claude Code Integration System
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum

# 既存の入力検証との統合
from features.common.input_validation import (
    InputValidationError,
    validate_input_directory,
    validate_output_directory,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 作者別パラメータシステムとの統合
try:
    from features.adaptation.author_parameter_adapter import AuthorParameterAdapter

    AUTHOR_ADAPTER_AVAILABLE = True
except ImportError:
    AUTHOR_ADAPTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """検証結果の重要度"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """検証で発見された問題"""

    severity: ValidationSeverity
    message: str
    suggestion: str
    path: Optional[Path] = None


@dataclass
class AuthorWorkInfo:
    """作者・作品情報"""

    author: Optional[str]
    work: Optional[str]
    detected_pattern: str
    confidence: float


@dataclass
class StrictValidationResult:
    """厳密検証結果"""

    is_valid: bool
    path: Optional[Path]
    issues: List[ValidationIssue]
    author_work_info: Optional[AuthorWorkInfo] = None

    @property
    def has_errors(self) -> bool:
        """エラーレベルの問題があるか"""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """警告レベルの問題があるか"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_error_messages(self) -> List[str]:
        """エラーメッセージのリストを取得"""
        return [
            issue.message for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        ]

    def get_formatted_report(self) -> str:
        """フォーマット済みレポートを取得"""
        if self.is_valid:
            return f"✅ パス検証成功: {self.path}"

        report_lines = [f"❌ パス検証失敗: {self.path}"]
        report_lines.append("")

        # エラー
        errors = [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        if errors:
            report_lines.append("🚨 エラー:")
            for issue in errors:
                report_lines.append(f"   - {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"     💡 対処法: {issue.suggestion}")

        # 警告
        warnings = [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
        if warnings:
            report_lines.append("")
            report_lines.append("⚠️ 警告:")
            for issue in warnings:
                report_lines.append(f"   - {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"     💡 推奨: {issue.suggestion}")

        # 作者・作品情報
        if self.author_work_info:
            report_lines.append("")
            report_lines.append("📋 検出情報:")
            report_lines.append(f"   作者: {self.author_work_info.author or '不明'}")
            report_lines.append(f"   作品: {self.author_work_info.work or '不明'}")
            report_lines.append(f"   パターン: {self.author_work_info.detected_pattern}")
            report_lines.append(f"   信頼度: {self.author_work_info.confidence:.2f}")

        return "\n".join(report_lines)


class StrictPathValidator:
    """
    厳密パス検証システム

    - デフォルト値・フォールバックを完全排除
    - 作者別パス構造の自動検証
    - インタラクティブパス入力機能
    - 完全な権限・構造・内容検証
    """

    def __init__(
        self,
        strict_mode: bool = True,
        require_author_structure: bool = False,
        interactive_mode: bool = False,
    ):
        """
        初期化

        Args:
            strict_mode: 厳密モード（デフォルト値使用禁止）
            require_author_structure: 作者別構造必須
            interactive_mode: 対話的入力モード
        """
        self.strict_mode = strict_mode
        self.require_author_structure = require_author_structure
        self.interactive_mode = interactive_mode

        # 作者パラメータアダプターとの統合
        if AUTHOR_ADAPTER_AVAILABLE:
            self.author_adapter = AuthorParameterAdapter()
        else:
            self.author_adapter = None
            logger.warning("AuthorParameterAdapter利用不可 - 作者別検証機能が制限されます")

    def validate_input_path(
        self, input_path: Optional[Union[str, Path]], context: str = "入力パス"
    ) -> StrictValidationResult:
        """
        入力パスの厳密検証

        Args:
            input_path: 検証対象パス
            context: エラーメッセージ用のコンテキスト

        Returns:
            StrictValidationResult: 検証結果
        """
        issues = []

        # 1. 基本的な存在チェック
        if not input_path:
            if self.strict_mode:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"{context}が指定されていません",
                        suggestion="明示的にパスを指定してください（デフォルト値は無効化されています）",
                    )
                )
                return StrictValidationResult(is_valid=False, path=None, issues=issues)
            else:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"{context}が指定されていません",
                        suggestion="明示的にパスを指定することを推奨します",
                    )
                )

        path = Path(input_path) if input_path else None
        if not path:
            return StrictValidationResult(is_valid=False, path=None, issues=issues)

        # 2. 既存の検証システムとの統合
        try:
            validate_input_directory(path, context, check_images=True)
        except InputValidationError as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=str(e),
                    suggestion="パスの存在確認と権限確認を行ってください",
                    path=path,
                )
            )
            return StrictValidationResult(is_valid=False, path=path, issues=issues)

        # 3. パス構造の検証
        structure_result = self._validate_path_structure(path)
        issues.extend(structure_result.issues)

        # 4. 作者・作品情報の検出
        author_work_info = self._detect_author_work_info(path)

        # 5. 作者別構造要件チェック
        if self.require_author_structure and not author_work_info.author:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="作者別パス構造が検出されませんでした",
                    suggestion="パスを /train/{作者名}/org/{作品名}/ 形式にしてください",
                    path=path,
                )
            )

        # 6. セキュリティチェック
        security_issues = self._validate_security_constraints(path)
        issues.extend(security_issues)

        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return StrictValidationResult(
            is_valid=is_valid, path=path, issues=issues, author_work_info=author_work_info
        )

    def validate_output_path(
        self, output_path: Optional[Union[str, Path]], context: str = "出力パス"
    ) -> StrictValidationResult:
        """
        出力パスの厳密検証

        Args:
            output_path: 検証対象パス
            context: エラーメッセージ用のコンテキスト

        Returns:
            StrictValidationResult: 検証結果
        """
        issues = []

        # 1. 基本的なパス指定チェック
        if not output_path:
            if self.strict_mode:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"{context}が指定されていません",
                        suggestion="明示的に出力パスを指定してください",
                    )
                )
                return StrictValidationResult(is_valid=False, path=None, issues=issues)

        path = Path(output_path) if output_path else None
        if not path:
            return StrictValidationResult(is_valid=False, path=None, issues=issues)

        # 2. 既存の検証システムとの統合
        try:
            validated_path = validate_output_directory(path, context, create_if_missing=False)
        except InputValidationError as e:
            # 出力ディレクトリは存在しなくても良いが、親ディレクトリは存在する必要がある
            parent = path.parent
            if not parent.exists():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"出力パスの親ディレクトリが存在しません: {parent}",
                        suggestion="親ディレクトリを作成するか、既存のパスを指定してください",
                        path=path,
                    )
                )
                return StrictValidationResult(is_valid=False, path=path, issues=issues)

        # 3. 書き込み権限チェック
        if path.exists():
            if not os.access(path, os.W_OK):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"出力ディレクトリに書き込み権限がありません: {path}",
                        suggestion="ディレクトリの権限を確認してください",
                        path=path,
                    )
                )
        else:
            # 親ディレクトリの書き込み権限チェック
            parent = path.parent
            if parent.exists() and not os.access(parent, os.W_OK):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"出力ディレクトリの親に書き込み権限がありません: {parent}",
                        suggestion="親ディレクトリの権限を確認してください",
                        path=path,
                    )
                )

        # 4. セキュリティチェック
        security_issues = self._validate_security_constraints(path)
        issues.extend(security_issues)

        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return StrictValidationResult(is_valid=is_valid, path=path, issues=issues)

    def _validate_path_structure(self, path: Path) -> StrictValidationResult:
        """パス構造の検証"""
        issues = []

        # 絶対パスの推奨
        if not path.is_absolute():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="相対パスよりも絶対パスが推奨されます",
                    suggestion="絶対パスを使用してください",
                    path=path,
                )
            )

        # パス長の検証
        if len(str(path)) > 260:  # Windows制限を考慮
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="パスが長すぎます（260文字制限）",
                    suggestion="より短いパスを使用してください",
                    path=path,
                )
            )

        # 危険文字の検証
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
        path_str = str(path)
        found_dangerous = [char for char in dangerous_chars if char in path_str]
        if found_dangerous:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"パスに危険な文字が含まれています: {', '.join(found_dangerous)}",
                    suggestion="英数字、ハイフン、アンダースコア、スラッシュのみを使用してください",
                    path=path,
                )
            )

        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        return StrictValidationResult(is_valid=is_valid, path=path, issues=issues)

    def _detect_author_work_info(self, path: Path) -> AuthorWorkInfo:
        """作者・作品情報の検出"""
        # 作者パラメータアダプターとの統合
        author = None
        if self.author_adapter:
            author = AuthorParameterAdapter.detect_author_from_path(str(path))

        # パス解析による作者・作品検出
        path_parts = path.parts
        work = None
        detected_pattern = "unknown"
        confidence = 0.0

        # /train/{author}/org/{work}/ パターンの検出
        for i, part in enumerate(path_parts):
            if part.lower() == "train" and i + 3 < len(path_parts):
                potential_author = path_parts[i + 1]
                if path_parts[i + 2].lower() == "org":
                    potential_work = path_parts[i + 3]
                    if not author:
                        author = potential_author
                    work = potential_work
                    detected_pattern = "train/{author}/org/{work}"
                    confidence = 0.9
                    break

        # 他のパターンの検出
        if not author:
            # パス内の既知作者名検索
            known_authors = ["yado", "kiri", "zundamon"]
            for part in path_parts:
                if part.lower() in known_authors:
                    author = part.lower()
                    detected_pattern = "path_component"
                    confidence = 0.5
                    break

        return AuthorWorkInfo(
            author=author, work=work, detected_pattern=detected_pattern, confidence=confidence
        )

    def _validate_security_constraints(self, path: Path) -> List[ValidationIssue]:
        """セキュリティ制約の検証"""
        issues = []

        # 許可ベースパスのチェック
        allowed_bases = [
            Path("/mnt/c/AItools/lora/train"),
            Path("/mnt/c/AItools/segment-anything"),
            Path("/tmp"),
        ]

        is_allowed = False
        for base in allowed_bases:
            try:
                path.relative_to(base)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="許可されたベースパス外のアクセスです",
                    suggestion=f"以下のパス内を使用してください: {', '.join(str(b) for b in allowed_bases)}",
                    path=path,
                )
            )

        # システムディレクトリの回避
        system_dirs = ["/etc", "/sys", "/proc", "/dev"]
        path_str = str(path).lower()
        for sys_dir in system_dirs:
            if path_str.startswith(sys_dir):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"システムディレクトリへのアクセスは禁止されています: {sys_dir}",
                        suggestion="ユーザーディレクトリ内のパスを使用してください",
                        path=path,
                    )
                )

        return issues

    def interactive_path_input(
        self, prompt: str, path_type: str = "input", default_suggestion: Optional[str] = None
    ) -> Optional[Path]:
        """
        対話的パス入力

        Args:
            prompt: ユーザーへのプロンプト
            path_type: パスタイプ（"input" or "output"）
            default_suggestion: デフォルト候補（厳密モードでは使用されない）

        Returns:
            Optional[Path]: 検証済みパス
        """
        if not self.interactive_mode:
            logger.error("インタラクティブモードが無効化されています")
            return None

        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            print(f"\n{prompt}")
            if default_suggestion and not self.strict_mode:
                print(f"デフォルト候補: {default_suggestion}")
                print("Enterキーで候補を使用、または新しいパスを入力:")
            else:
                print("パスを入力してください:")

            try:
                user_input = input("> ").strip()

                # 空入力の処理
                if not user_input:
                    if default_suggestion and not self.strict_mode:
                        user_input = default_suggestion
                    else:
                        print("❌ パスの入力が必要です")
                        attempt += 1
                        continue

                # 検証実行
                if path_type == "input":
                    result = self.validate_input_path(user_input, "対話入力パス")
                else:
                    result = self.validate_output_path(user_input, "対話出力パス")

                if result.is_valid:
                    print(f"✅ パス検証成功: {result.path}")
                    if result.author_work_info and result.author_work_info.author:
                        print(f"📋 検出作者: {result.author_work_info.author}")
                    return result.path
                else:
                    print("\n" + result.get_formatted_report())
                    attempt += 1

            except KeyboardInterrupt:
                print("\n❌ ユーザーによってキャンセルされました")
                return None
            except Exception as e:
                print(f"❌ 入力エラー: {e}")
                attempt += 1

        print(f"❌ {max_attempts}回の試行に失敗しました")
        return None

    def validate_paths_comprehensive(
        self, input_path: Optional[Union[str, Path]], output_path: Optional[Union[str, Path]]
    ) -> Tuple[StrictValidationResult, StrictValidationResult]:
        """
        入力・出力パスの包括的検証

        Args:
            input_path: 入力パス
            output_path: 出力パス

        Returns:
            Tuple[StrictValidationResult, StrictValidationResult]: (入力結果, 出力結果)
        """
        # 対話的入力モードの場合
        if self.interactive_mode:
            if not input_path:
                input_path = self.interactive_path_input("🔍 画像入力ディレクトリを指定してください:", "input")

            if not output_path:
                output_path = self.interactive_path_input("📁 抽出結果出力ディレクトリを指定してください:", "output")

        # 検証実行
        input_result = self.validate_input_path(input_path, "画像入力ディレクトリ")
        output_result = self.validate_output_path(output_path, "抽出結果出力ディレクトリ")

        return input_result, output_result

    def log_validation_summary(
        self,
        input_result: StrictValidationResult,
        output_result: StrictValidationResult,
        script_name: str = "QUAL-033検証",
    ):
        """検証結果サマリーログ"""
        logger.info(f"🔍 {script_name} - 厳密パス検証サマリー")

        # 入力パス結果
        if input_result.is_valid:
            logger.info(f"✅ 入力パス検証成功: {input_result.path}")
            if input_result.author_work_info and input_result.author_work_info.author:
                logger.info(f"📋 検出作者: {input_result.author_work_info.author}")
        else:
            logger.error(f"❌ 入力パス検証失敗: {input_result.path}")
            for error in input_result.get_error_messages():
                logger.error(f"   - {error}")

        # 出力パス結果
        if output_result.is_valid:
            logger.info(f"✅ 出力パス検証成功: {output_result.path}")
        else:
            logger.error(f"❌ 出力パス検証失敗: {output_result.path}")
            for error in output_result.get_error_messages():
                logger.error(f"   - {error}")

        # 警告まとめ
        all_warnings = []
        if input_result.has_warnings:
            all_warnings.extend(
                [
                    f"入力: {issue.message}"
                    for issue in input_result.issues
                    if issue.severity == ValidationSeverity.WARNING
                ]
            )
        if output_result.has_warnings:
            all_warnings.extend(
                [
                    f"出力: {issue.message}"
                    for issue in output_result.issues
                    if issue.severity == ValidationSeverity.WARNING
                ]
            )

        if all_warnings:
            logger.warning("⚠️ 警告事項:")
            for warning in all_warnings:
                logger.warning(f"   - {warning}")

        overall_success = input_result.is_valid and output_result.is_valid
        logger.info(f"📊 全体結果: {'✅ 成功' if overall_success else '❌ 失敗'}")


# 便利な関数
def validate_strict_paths(
    input_path: Optional[Union[str, Path]],
    output_path: Optional[Union[str, Path]],
    strict_mode: bool = True,
    interactive_mode: bool = False,
    require_author_structure: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    厳密パス検証の便利関数

    Args:
        input_path: 入力パス
        output_path: 出力パス
        strict_mode: 厳密モード
        interactive_mode: 対話的入力モード
        require_author_structure: 作者別構造必須

    Returns:
        Tuple[Optional[Path], Optional[Path]]: (検証済み入力パス, 検証済み出力パス)

    Raises:
        InputValidationError: 検証失敗時
    """
    validator = StrictPathValidator(
        strict_mode=strict_mode,
        require_author_structure=require_author_structure,
        interactive_mode=interactive_mode,
    )

    input_result, output_result = validator.validate_paths_comprehensive(input_path, output_path)

    # 検証結果のログ出力
    validator.log_validation_summary(input_result, output_result)

    # エラー時の例外発生
    if not input_result.is_valid:
        raise InputValidationError(input_result.get_formatted_report())

    if not output_result.is_valid:
        raise InputValidationError(output_result.get_formatted_report())

    return input_result.path, output_result.path


if __name__ == "__main__":
    # テスト実行
    import tempfile

    logging.basicConfig(level=logging.INFO)

    print("🧪 厳密パス検証システム テスト")
    print("=" * 50)

    # テスト用パス
    test_paths = [
        "/mnt/c/AItools/lora/train/yado/org/kana05/",
        "/mnt/c/AItools/lora/train/kiri/org/test01/",
        "/nonexistent/path/",
        "",
        None,
    ]

    validator = StrictPathValidator(strict_mode=True, interactive_mode=False)

    for test_path in test_paths:
        print(f"\n🔍 テスト対象: {test_path}")
        result = validator.validate_input_path(test_path)
        print(result.get_formatted_report())
