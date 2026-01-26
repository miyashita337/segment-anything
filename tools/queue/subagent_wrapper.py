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

import sys
from pathlib import Path

# 直接実行時のパス設定
if __name__ == "__main__":
    # プロジェクトルートをパスに追加
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 後方互換性のための再エクスポート（絶対インポート）
from tools.queue.task_queue import SubAgentTaskQueue
from tools.queue.task_validator import ExtractCharacterTaskValidator
from tools.queue.enhanced_task_queue import EnhancedSubAgentTaskQueue
from tools.queue.integrated_system import IntegratedSubAgentSystem
from tools.queue.system_coordinator import SubAgentSystemCoordinator, main

__all__ = [
    "SubAgentTaskQueue",
    "ExtractCharacterTaskValidator",
    "EnhancedSubAgentTaskQueue",
    "IntegratedSubAgentSystem",
    "SubAgentSystemCoordinator",
    "main",
]

if __name__ == "__main__":
    main()
