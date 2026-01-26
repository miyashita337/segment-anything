#!/usr/bin/env python3
"""
KIRO-015: 後方互換性ラッパー

このファイルは後方互換性のために維持されています。
新規コードでは以下のインポートを推奨します:

    from tools.queue import SubAgentTaskQueue
    from tools.queue import IntegratedSubAgentSystem
    # etc.

または直接:

    from tools.queue.task_queue import SubAgentTaskQueue
    from tools.queue.integrated_system import IntegratedSubAgentSystem
    # etc.
"""

# 後方互換性のための再エクスポート
from .task_queue import SubAgentTaskQueue
from .task_validator import ExtractCharacterTaskValidator
from .enhanced_task_queue import EnhancedSubAgentTaskQueue
from .integrated_system import IntegratedSubAgentSystem
from .system_coordinator import SubAgentSystemCoordinator

# main関数も移行（system_coordinatorから）
from .system_coordinator import main

__all__ = [
    "SubAgentTaskQueue",
    "ExtractCharacterTaskValidator",
    "EnhancedSubAgentTaskQueue",
    "IntegratedSubAgentSystem",
    "SubAgentSystemCoordinator",
    "main",
]
