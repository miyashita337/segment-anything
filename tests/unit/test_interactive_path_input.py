#!/usr/bin/env python3
"""
インタラクティブパス入力システム単体テスト (QUAL-033)

Created for: QUAL-033 - 厳密パス検証システム実装・全ワークフロー適用・意図しない挙動防止
Author: Claude Code Integration System
"""

import pytest
import tempfile
from features.common.interactive_path_input import (
    InputPrompt,
    InputType,
    InteractivePathInput,
    InteractiveResult,
    interactive_setup,
)
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestInteractivePathInput:
    """インタラクティブパス入力システムの単体テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.interactive = InteractivePathInput(
            use_colors=False, max_attempts=3, auto_suggest=True  # テスト時はカラーなし
        )

        # テスト用一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "test_dir"
        self.test_dir.mkdir()
        (self.test_dir / "test.jpg").touch()

    def teardown_method(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_input_prompt_creation(self):
        """InputPrompt作成テスト"""
        prompt = InputPrompt(
            prompt_text="テストプロンプト",
            input_type=InputType.INPUT_DIRECTORY,
            required=True,
            help_text="テストヘルプ",
        )

        assert prompt.prompt_text == "テストプロンプト"
        assert prompt.input_type == InputType.INPUT_DIRECTORY
        assert prompt.required is True
        assert prompt.help_text == "テストヘルプ"

    def test_interactive_result_creation(self):
        """InteractiveResult作成テスト"""
        result = InteractiveResult(success=True, value="test_value", path=Path("/test/path"))

        assert result.success is True
        assert result.value == "test_value"
        assert result.path == Path("/test/path")
        assert result.cancelled is False

    def test_colorize_with_colors_disabled(self):
        """カラー無効時の色付けテスト"""
        colored_text = self.interactive._colorize("テキスト", "red")
        assert colored_text == "テキスト"  # カラーなしでそのまま

    def test_colorize_with_colors_enabled(self):
        """カラー有効時の色付けテスト"""
        interactive_with_colors = InteractivePathInput(use_colors=True)
        colored_text = interactive_with_colors._colorize("テキスト", "red")
        assert "\033[91m" in colored_text  # 赤色コードが含まれる
        assert "\033[0m" in colored_text  # リセットコードが含まれる

    def test_get_path_suggestions_input_directory(self):
        """入力ディレクトリ候補取得テスト"""
        suggestions = self.interactive._get_path_suggestions(InputType.INPUT_DIRECTORY, "")

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 10  # 最大10候補

    def test_get_path_suggestions_author_name(self):
        """作者名候補取得テスト"""
        suggestions = self.interactive._get_path_suggestions(InputType.AUTHOR_NAME, "ya")

        assert "yado" in suggestions

    def test_get_path_suggestions_work_name(self):
        """作品名候補取得テスト"""
        suggestions = self.interactive._get_path_suggestions(InputType.WORK_NAME, "kana")

        assert any("kana" in suggestion for suggestion in suggestions)

    def test_validate_input_required_empty(self):
        """必須項目で空入力の検証"""
        prompt = InputPrompt(prompt_text="テスト", input_type=InputType.INPUT_DIRECTORY, required=True)

        error = self.interactive._validate_input("", prompt)
        assert error == "入力が必要です"

    def test_validate_input_not_required_empty(self):
        """非必須項目で空入力の検証"""
        prompt = InputPrompt(
            prompt_text="テスト", input_type=InputType.INPUT_DIRECTORY, required=False
        )

        error = self.interactive._validate_input("", prompt)
        assert error is None

    def test_validate_input_custom_function(self):
        """カスタム検証関数テスト"""

        def custom_validator(value):
            if "invalid" in value:
                return "無効な値です"
            return None

        prompt = InputPrompt(
            prompt_text="テスト",
            input_type=InputType.INPUT_DIRECTORY,
            validation_func=custom_validator,
        )

        error = self.interactive._validate_input("invalid_value", prompt)
        assert error == "無効な値です"

        error = self.interactive._validate_input("valid_value", prompt)
        assert error is None

    def test_validate_input_directory_path(self):
        """ディレクトリパス検証テスト"""
        prompt = InputPrompt(prompt_text="テスト", input_type=InputType.INPUT_DIRECTORY)

        # 存在するディレクトリ
        error = self.interactive._validate_input(str(self.test_dir), prompt)
        assert error is None

        # 存在しないディレクトリ
        error = self.interactive._validate_input("/nonexistent/path", prompt)
        assert "存在しません" in error

    def test_validate_input_tracker_id(self):
        """トラッカーID検証テスト"""
        prompt = InputPrompt(prompt_text="テスト", input_type=InputType.TRACKER_ID)

        # 有効なトラッカーID
        error = self.interactive._validate_input("QUAL-033", prompt)
        assert error is None

        # 無効なトラッカーID
        error = self.interactive._validate_input("invalid_id", prompt)
        assert "PREFIX-NUMBER" in error

    @patch("builtins.input")
    def test_prompt_for_input_success(self, mock_input):
        """入力プロンプト成功テスト"""
        mock_input.return_value = str(self.test_dir)

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください", input_type=InputType.INPUT_DIRECTORY, required=True
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is True
        assert result.value == str(self.test_dir)
        assert result.path == self.test_dir

    @patch("builtins.input")
    def test_prompt_for_input_help_command(self, mock_input):
        """ヘルプコマンドテスト"""
        mock_input.side_effect = ["help", str(self.test_dir)]

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください", input_type=InputType.INPUT_DIRECTORY, required=True
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is True
        assert mock_input.call_count == 2

    @patch("builtins.input")
    def test_prompt_for_input_cancel_command(self, mock_input):
        """キャンセルコマンドテスト"""
        mock_input.return_value = "cancel"

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください", input_type=InputType.INPUT_DIRECTORY, required=True
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is False
        assert result.cancelled is True

    @patch("builtins.input")
    def test_prompt_for_input_keyboard_interrupt(self, mock_input):
        """キーボード割り込みテスト"""
        mock_input.side_effect = KeyboardInterrupt()

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください", input_type=InputType.INPUT_DIRECTORY, required=True
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is False
        assert result.cancelled is True

    @patch("builtins.input")
    def test_prompt_for_input_max_attempts(self, mock_input):
        """最大試行回数テスト"""
        mock_input.return_value = "/nonexistent/path"

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください", input_type=InputType.INPUT_DIRECTORY, required=True
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is False
        assert mock_input.call_count == 3  # max_attempts

    @patch("builtins.input")
    def test_prompt_for_input_default_value(self, mock_input):
        """デフォルト値テスト"""
        mock_input.return_value = ""  # 空入力

        prompt = InputPrompt(
            prompt_text="ディレクトリを入力してください",
            input_type=InputType.INPUT_DIRECTORY,
            required=False,
            default_value=str(self.test_dir),
        )

        result = self.interactive.prompt_for_input(prompt)

        assert result.success is True
        assert result.value == str(self.test_dir)

    @patch("builtins.input")
    def test_prompt_for_paths_success(self, mock_input):
        """パス設定プロンプト成功テスト"""
        output_dir = Path(self.temp_dir) / "output"
        mock_input.side_effect = [str(self.test_dir), str(output_dir)]

        results = self.interactive.prompt_for_paths()

        assert results["success"] is True
        assert results["input_path"] == self.test_dir
        assert results["output_path"] == output_dir

    @patch("builtins.input")
    def test_prompt_for_paths_cancel(self, mock_input):
        """パス設定プロンプトキャンセルテスト"""
        mock_input.return_value = "cancel"

        results = self.interactive.prompt_for_paths()

        assert results["success"] is False
        assert results["cancelled"] is True

    @patch("builtins.input")
    def test_prompt_for_tracker_info_success(self, mock_input):
        """トラッカー情報プロンプト成功テスト"""
        mock_input.side_effect = ["QUAL-033", "yado", "kana05"]

        results = self.interactive.prompt_for_tracker_info()

        assert results["success"] is True
        assert results["tracker_id"] == "QUAL-033"
        assert results["author"] == "yado"
        assert results["work"] == "kana05"

    @patch("builtins.input")
    def test_prompt_for_tracker_info_optional_fields(self, mock_input):
        """トラッカー情報プロンプトオプション項目テスト"""
        mock_input.side_effect = ["QUAL-033", "", ""]  # トラッカーIDのみ

        results = self.interactive.prompt_for_tracker_info()

        assert results["success"] is True
        assert results["tracker_id"] == "QUAL-033"
        assert results["author"] is None
        assert results["work"] is None


class TestInteractiveSetupFunction:
    """interactive_setup便利関数のテスト"""

    @patch("builtins.input")
    def test_interactive_setup_success(self, mock_input):
        """完全セットアップ成功テスト"""
        temp_dir = tempfile.mkdtemp()
        test_input = Path(temp_dir) / "input"
        test_input.mkdir()
        (test_input / "test.jpg").touch()
        test_output = Path(temp_dir) / "output"

        try:
            mock_input.side_effect = [
                "QUAL-033",  # トラッカーID
                "yado",  # 作者名
                "test",  # 作品名
                str(test_input),  # 入力パス
                str(test_output),  # 出力パス
            ]

            results = interactive_setup()

            assert results["success"] is True
            assert results["tracker_info"]["tracker_id"] == "QUAL-033"
            assert results["paths"]["input_path"] == test_input

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("builtins.input")
    def test_interactive_setup_cancel(self, mock_input):
        """完全セットアップキャンセルテスト"""
        mock_input.return_value = "cancel"

        results = interactive_setup()

        assert results["success"] is False
        assert results["cancelled"] is True


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
