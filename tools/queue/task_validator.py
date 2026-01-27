#!/usr/bin/env python3
"""
KIRO-015: タスク検証モジュール
subagent_wrapper.pyから分割
"""

import logging
import psutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ExtractCharacterTaskValidator:
    """extract_character.py 直接実行防止システム"""

    @staticmethod
    def check_subagent_execution(command: str) -> Tuple[bool, str]:
        """
        SubAgent実行必須チェック

        Args:
            command: 実行されるコマンド

        Returns:
            Tuple[bool, str]: (実行許可, メッセージ)
        """
        if "extract_character.py" in command:
            # 直接実行かSubAgent経由かの判定
            if not ExtractCharacterTaskValidator._is_subagent_execution():
                return False, (
                    "❌ extract_character.py の直接実行は禁止されています。\n"
                    "SubAgentキューシステム経由で実行してください。\n"
                    "\n"
                    "🔧 正しい実行方法:\n"
                    "1. python tools/queue/subagent_wrapper.py enqueue extract_character\n"
                    "2. python tools/queue/subagent_wrapper.py execute\n"
                    "\n"
                    "詳細: docs/claude-code-hooks-guide.md を参照"
                )

        return True, "実行許可"

    @staticmethod
    def _is_subagent_execution() -> bool:
        """SubAgent実行環境の検出"""
        # 環境変数による検出
        subagent_env = ["SUBAGENT_TASK_ID", "SUBAGENT_EXECUTION", "CLAUDE_SUBAGENT_MODE"]

        import os

        for env_var in subagent_env:
            if os.getenv(env_var):
                return True

        # プロセス階層による検出
        try:
            current_process = psutil.Process()
            for parent in current_process.parents():
                if "subagent" in parent.name().lower():
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return False


# INTG-089: SubAgentTaskQueue拡張機能
