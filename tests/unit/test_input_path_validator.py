#!/usr/bin/env python3
"""
入力パス検証ユーティリティのテスト
"""

import os
import pytest
import sys
import tempfile
from pathlib import Path

# テスト対象をインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.utils.input_path_validator import InputPathValidationError, InputPathValidator


class TestInputPathValidator:
    """入力パス検証ユーティリティのテストクラス"""

    def test_valid_directory_path(self):
        """有効なディレクトリパスの検証テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid, error_msg = InputPathValidator.validate_input_path(temp_dir)
            assert is_valid == True
            assert error_msg is None

    def test_nonexistent_path_raises_error(self):
        """存在しないパスでのエラー発生テスト"""
        nonexistent_path = "/this/path/does/not/exist"

        with pytest.raises(InputPathValidationError) as exc_info:
            InputPathValidator.validate_input_path(nonexistent_path)

        error_msg = str(exc_info.value)
        assert "❌ エラー: 入力ディレクトリが存在しません" in error_msg
        assert nonexistent_path in error_msg
        assert "🔧 対処方法:" in error_msg
        assert "⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です" in error_msg

    def test_file_path_instead_of_directory(self):
        """ファイルパスをディレクトリとして指定した場合のエラーテスト"""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(InputPathValidationError) as exc_info:
                InputPathValidator.validate_input_path(temp_file.name)

            error_msg = str(exc_info.value)
            assert "❌ エラー: 指定されたパスはディレクトリではありません" in error_msg
            assert temp_file.name in error_msg

    def test_error_message_format(self):
        """エラーメッセージの統一フォーマット確認"""
        test_path = "/mnt/c/AItools/lora/train/yado/org/TEST-001/"
        error_msg = InputPathValidator._generate_error_message(test_path)

        # 必須要素の確認
        assert "❌ エラー: 入力ディレクトリが存在しません" in error_msg
        assert f"パス: {test_path}" in error_msg
        assert "🔧 対処方法:" in error_msg
        assert "1. パスの確認:" in error_msg
        assert "2. 正しいパスの指定" in error_msg
        assert "3. 必要に応じてディレクトリ作成" in error_msg
        assert "⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です" in error_msg

    def test_claude_md_compliance(self):
        """CLAUDE.md仕様準拠の確認テスト"""
        # 存在しないパスでの検証
        nonexistent_path = "/nonexistent/test/path"

        with pytest.raises(InputPathValidationError):
            InputPathValidator.validate_input_path(nonexistent_path)

        # チェックリスト項目の確認:
        # ✅ 1. 入力パス存在確認完了 - validate_input_path内で実行
        # ✅ 2. 存在しない場合、即座停止実行 - InputPathValidationError発生
        # ✅ 3. 統一エラーメッセージ使用 - _generate_error_message使用
        # ✅ 4. 代替案提案を回避 - エラー発生のみ、代替案なし

        # エラーメッセージに代替案提案が含まれていないことを確認
        try:
            InputPathValidator.validate_input_path(nonexistent_path)
        except InputPathValidationError as e:
            error_msg = str(e)
            # 代替案提案の禁止文言が含まれていないことを確認
            assert "代替案" not in error_msg
            assert "kana05" not in error_msg
            assert "とりあえず" not in error_msg
            assert "一旦" not in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
