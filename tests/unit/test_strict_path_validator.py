#!/usr/bin/env python3
"""
厳密パス検証システム単体テスト (QUAL-033)

Created for: QUAL-033 - 厳密パス検証システム実装・全ワークフロー適用・意図しない挙動防止
Author: Claude Code Integration System
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from features.common.strict_path_validator import (
    StrictPathValidator,
    ValidationSeverity,
    ValidationIssue,
    AuthorWorkInfo,
    StrictValidationResult,
    validate_strict_paths
)
from features.common.input_validation import InputValidationError


class TestStrictPathValidator:
    """厳密パス検証システムの単体テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.validator = StrictPathValidator(
            strict_mode=True,
            require_author_structure=False,
            interactive_mode=False
        )
        
        # テスト用一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_dir = Path(self.temp_dir) / "test_input"
        self.test_input_dir.mkdir()
        
        # テスト用画像ファイル作成
        (self.test_input_dir / "test1.jpg").touch()
        (self.test_input_dir / "test2.png").touch()

    def teardown_method(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_input_path_success(self):
        """入力パス検証成功ケース"""
        result = self.validator.validate_input_path(self.test_input_dir)
        
        assert result.is_valid is True
        assert result.path == self.test_input_dir
        assert not result.has_errors

    def test_validate_input_path_nonexistent(self):
        """存在しない入力パス検証"""
        nonexistent_path = Path("/nonexistent/path")
        result = self.validator.validate_input_path(nonexistent_path)
        
        assert result.is_valid is False
        assert result.has_errors
        assert any("存在しません" in issue.message for issue in result.issues)

    def test_validate_input_path_empty_strict_mode(self):
        """厳密モードでの空入力パス"""
        result = self.validator.validate_input_path(None)
        
        assert result.is_valid is False
        assert result.has_errors
        assert any("指定されていません" in issue.message for issue in result.issues)

    def test_validate_input_path_empty_non_strict_mode(self):
        """非厳密モードでの空入力パス"""
        validator = StrictPathValidator(strict_mode=False)
        result = validator.validate_input_path(None)
        
        assert result.has_warnings
        assert any("指定されていません" in issue.message for issue in result.issues)

    def test_validate_output_path_success(self):
        """出力パス検証成功ケース"""
        output_path = Path(self.temp_dir) / "test_output"
        result = self.validator.validate_output_path(output_path)
        
        assert result.is_valid is True
        assert result.path == output_path

    def test_validate_output_path_invalid_parent(self):
        """無効な親ディレクトリの出力パス"""
        invalid_output = Path("/nonexistent/parent/output")
        result = self.validator.validate_output_path(invalid_output)
        
        assert result.is_valid is False
        assert result.has_errors

    def test_author_work_detection_standard_pattern(self):
        """標準的な作者・作品パターンの検出"""
        test_path = Path("/mnt/c/AItools/lora/train/yado/org/kana05/test.jpg")
        author_info = self.validator._detect_author_work_info(test_path)
        
        assert author_info.author == "yado"
        assert author_info.work == "kana05"
        assert "train/{author}/org/{work}" in author_info.detected_pattern
        assert author_info.confidence > 0.8

    def test_author_work_detection_no_pattern(self):
        """パターンなしの作者・作品検出"""
        test_path = Path("/some/random/path/file.jpg")
        author_info = self.validator._detect_author_work_info(test_path)
        
        assert author_info.author is None
        assert author_info.work is None
        assert author_info.confidence < 0.5

    def test_security_constraints_allowed_path(self):
        """許可されたパスのセキュリティチェック"""
        allowed_path = Path("/mnt/c/AItools/lora/train/yado/test")
        issues = self.validator._validate_security_constraints(allowed_path)
        
        # 警告はあるかもしれないが、エラーはないはず
        error_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        assert len(error_issues) == 0

    def test_security_constraints_system_path(self):
        """システムパスのセキュリティチェック"""
        system_path = Path("/etc/passwd")
        issues = self.validator._validate_security_constraints(system_path)
        
        error_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        assert len(error_issues) > 0
        assert any("システムディレクトリ" in issue.message for issue in error_issues)

    def test_path_structure_validation_dangerous_chars(self):
        """危険文字を含むパス構造の検証"""
        dangerous_path = Path("/test/path/with<dangerous>chars")
        result = self.validator._validate_path_structure(dangerous_path)
        
        assert not result.is_valid
        assert any("危険な文字" in issue.message for issue in result.issues)

    def test_path_structure_validation_long_path(self):
        """長すぎるパス構造の検証"""
        long_path = Path("/" + "a" * 300)
        result = self.validator._validate_path_structure(long_path)
        
        warning_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.WARNING]
        assert any("長すぎます" in issue.message for issue in warning_issues)

    def test_validate_paths_comprehensive(self):
        """包括的パス検証"""
        output_path = Path(self.temp_dir) / "output"
        
        input_result, output_result = self.validator.validate_paths_comprehensive(
            self.test_input_dir, output_path
        )
        
        assert input_result.is_valid
        assert output_result.is_valid

    def test_require_author_structure_mode(self):
        """作者別構造必須モード"""
        validator = StrictPathValidator(require_author_structure=True)
        
        # 作者別構造なしのパス
        simple_path = Path("/simple/path")
        result = validator.validate_input_path(simple_path)
        
        assert not result.is_valid
        assert any("作者別パス構造" in issue.message for issue in result.issues)

    @patch('builtins.input')
    def test_interactive_path_input_success(self, mock_input):
        """対話的パス入力成功ケース"""
        validator = StrictPathValidator(interactive_mode=True)
        mock_input.return_value = str(self.test_input_dir)
        
        result_path = validator.interactive_path_input("テストプロンプト", "input")
        
        assert result_path == self.test_input_dir

    @patch('builtins.input')
    def test_interactive_path_input_cancel(self, mock_input):
        """対話的パス入力キャンセル"""
        validator = StrictPathValidator(interactive_mode=True)
        mock_input.side_effect = KeyboardInterrupt()
        
        result_path = validator.interactive_path_input("テストプロンプト", "input")
        
        assert result_path is None

    def test_validation_result_formatting(self):
        """検証結果フォーマット"""
        issues = [
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="テストエラー",
                suggestion="修正してください"
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="テスト警告",
                suggestion="確認してください"
            )
        ]
        
        result = StrictValidationResult(
            is_valid=False,
            path=Path("/test/path"),
            issues=issues
        )
        
        formatted = result.get_formatted_report()
        assert "❌ パス検証失敗" in formatted
        assert "🚨 エラー:" in formatted
        assert "⚠️ 警告:" in formatted
        assert "テストエラー" in formatted
        assert "テスト警告" in formatted


class TestStrictValidationConvenienceFunction:
    """validate_strict_paths 便利関数のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_dir = Path(self.temp_dir) / "input"
        self.test_input_dir.mkdir()
        (self.test_input_dir / "test.jpg").touch()
        
        self.test_output_dir = Path(self.temp_dir) / "output"

    def teardown_method(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_strict_paths_success(self):
        """validate_strict_paths 成功ケース"""
        input_path, output_path = validate_strict_paths(
            self.test_input_dir,
            self.test_output_dir,
            strict_mode=True,
            interactive_mode=False
        )
        
        assert input_path == self.test_input_dir
        assert output_path == self.test_output_dir

    def test_validate_strict_paths_failure(self):
        """validate_strict_paths 失敗ケース"""
        with pytest.raises(InputValidationError):
            validate_strict_paths(
                "/nonexistent/input",
                "/nonexistent/output",
                strict_mode=True,
                interactive_mode=False
            )


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])