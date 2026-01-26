#!/usr/bin/env python3
"""
入力パス検証ユーティリティ

CLAUDE.md準拠の厳格な入力パス検証を実装
docs/checklists/input_path_validation_checklist.md のチェックリストに従った検証
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple


class InputPathValidationError(Exception):
    """入力パス検証エラー"""

    pass


class InputPathValidator:
    """入力パス検証クラス - CLAUDE.md準拠"""

    @staticmethod
    def validate_input_path(input_path: str) -> Tuple[bool, Optional[str]]:
        """
        入力パス検証（CLAUDE.md準拠）

        Args:
            input_path: 検証対象の入力パス

        Returns:
            (検証結果, エラーメッセージ)

        Raises:
            InputPathValidationError: パスが存在しない場合
        """
        # 空文字列チェック（Phase 1前の事前チェック）
        if not input_path or not input_path.strip():
            error_msg = """❌ エラー: 入力パスが空です
   パス: (空文字列)

🔧 対処方法:
   1. 有効なディレクトリパスを指定
   2. パス形式の確認

⚠️ 注意: 空のパスは無効です"""
            raise InputPathValidationError(error_msg)

        # Phase 1: 入力パス受信時の即座検証
        path = Path(input_path)

        # チェック1: 入力パス存在確認完了
        if not path.exists():
            # チェック2: 存在しない場合、即座停止実行
            error_msg = InputPathValidator._generate_error_message(input_path)
            # チェック3: 統一エラーメッセージ使用
            # チェック4: 代替案提案を回避（エラーのみ返却）
            raise InputPathValidationError(error_msg)

        if not path.is_dir():
            error_msg = f"""❌ エラー: 指定されたパスはディレクトリではありません
   パス: {input_path}

🔧 対処方法:
   1. パスの確認: ls {path.parent}
   2. 正しいディレクトリパスの指定

⚠️ 注意: ファイルパスではなくディレクトリパスが必要です"""
            raise InputPathValidationError(error_msg)

        return True, None

    @staticmethod
    def _generate_error_message(input_path: str) -> str:
        """
        統一エラーメッセージ生成

        Args:
            input_path: 存在しないパス

        Returns:
            統一フォーマットのエラーメッセージ
        """
        parent_path = Path(input_path).parent

        return f"""❌ エラー: 入力ディレクトリが存在しません
   パス: {input_path}

🔧 対処方法:
   1. パスの確認: ls {parent_path}
   2. 正しいパスの指定
   3. 必要に応じてディレクトリ作成

⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です"""

    @staticmethod
    def validate_and_exit_on_error(input_path: str) -> None:
        """
        入力パス検証実行（エラー時は即座終了）

        Args:
            input_path: 検証対象の入力パス
        """
        try:
            InputPathValidator.validate_input_path(input_path)
        except InputPathValidationError as e:
            print(str(e))
            sys.exit(1)


def main():
    """CLI実行用メイン関数"""
    if len(sys.argv) != 2:
        print("Usage: python input_path_validator.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        is_valid, error_msg = InputPathValidator.validate_input_path(input_path)
        if is_valid:
            print(f"✅ 入力パス検証成功: {input_path}")
        else:
            print(error_msg)
            sys.exit(1)
    except InputPathValidationError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
