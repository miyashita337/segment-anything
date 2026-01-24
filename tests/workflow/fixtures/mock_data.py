"""
ワークフローテスト用モックデータ定義
実際のDB・API呼び出しを模擬するためのテストデータを提供
"""

from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class MockTask:
    """モックタスクデータ"""
    tracker_id: str
    description: str
    status: str
    created_date: str
    updated_date: str


class MockData:
    """テスト用モックデータコレクション"""

    # 標準的なトラッカーID
    VALID_TRACKER_IDS = [
        "TRACKER-001",
        "KIRO-006",
        "QUAL-044",
        "TEST-123",
        "INTG-088"
    ]

    # 無効なトラッカーID
    INVALID_TRACKER_IDS = [
        "",
        "invalid",
        "TRACKER_001",  # ハイフンなし
        "tracker-001",  # 小文字
        "TR-",          # 番号なし
        "123-TRACKER",  # 逆順
    ]

    # モックタスクデータ
    MOCK_TASKS = {
        "TRACKER-001": MockTask(
            tracker_id="TRACKER-001",
            description="テスト用タスク1",
            status="planning",
            created_date="2025-09-27 10:00:00",
            updated_date="2025-09-27 10:00:00"
        ),
        "KIRO-006": MockTask(
            tracker_id="KIRO-006",
            description="ワークフロー強制実行システム",
            status="in_progress",
            created_date="2025-09-27 09:00:00",
            updated_date="2025-09-27 11:00:00"
        ),
        "QUAL-044": MockTask(
            tracker_id="QUAL-044",
            description="品質保証ワークフロー",
            status="completed",
            created_date="2025-09-26 15:00:00",
            updated_date="2025-09-27 12:00:00"
        )
    }

    # ワークフロー状態データ
    MOCK_WORKFLOW_STATUS = {
        "TRACKER-001": {
            "tracker_id": "TRACKER-001",
            "current_phase": "phase_0_5",
            "current_step": "branch_verification",
            "can_proceed": True,
            "completed_steps": [
                {"step_id": "planning", "completed_at": "2025-09-27 10:00:00"}
            ],
            "pending_approvals": [],
            "blocked_actions": [],
            "current_step_instructions": {
                "title": "ブランチ検証",
                "description": "feature/TRACKER-001 ブランチで作業していることを確認",
                "required_actions": ["ブランチ確認", "初期コミット"],
                "validation_criteria": ["正しいブランチ", "コミット履歴確認"],
                "approval_required": False,
                "can_proceed": True,
                "blocking_reasons": []
            }
        },
        "KIRO-006": {
            "tracker_id": "KIRO-006",
            "current_phase": "phase_2",
            "current_step": "implementation",
            "can_proceed": False,
            "completed_steps": [
                {"step_id": "planning", "completed_at": "2025-09-27 09:00:00"},
                {"step_id": "design", "completed_at": "2025-09-27 09:30:00"}
            ],
            "pending_approvals": [
                {
                    "approval_id": "APP-001",
                    "title": "実装承認",
                    "tracker_id": "KIRO-006",
                    "step_name": "implementation",
                    "priority": "high",
                    "requested_at": "2025-09-27 10:00:00",
                    "time_remaining_hours": 23.5,
                    "approval_criteria": ["設計レビュー完了", "技術仕様確認"]
                }
            ],
            "blocked_actions": [
                {
                    "action": "step_completion",
                    "reason": "承認待ちのため進行できません"
                }
            ],
            "current_step_instructions": {
                "title": "実装フェーズ",
                "description": "ワークフロー強制実行システムの実装",
                "required_actions": ["コード実装", "ユニットテスト作成"],
                "validation_criteria": ["テスト通過", "コードレビュー"],
                "approval_required": True,
                "can_proceed": False,
                "blocking_reasons": ["承認待ち"]
            }
        }
    }

    # ワークスペース設定データ
    MOCK_WORKSPACE_CONFIG = {
        "TRACKER-001": {
            "tracker_id": "TRACKER-001",
            "author_name": "yado",
            "workspace_path": "/mnt/c/AItools/lora/train/yado/tracker-workspace/TRACKER-001",
            "input_path": "/mnt/c/AItools/lora/train/yado/org/kana05"
        },
        "KIRO-006": {
            "tracker_id": "KIRO-006",
            "author_name": "kiri",
            "workspace_path": "/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-006",
            "input_path": "/mnt/c/AItools/lora/train/kiri/org/dataset"
        }
    }

    # SubAgent状態データ
    MOCK_SUBAGENT_STATUS = {
        "TRACKER-001": {
            "tracker_id": "TRACKER-001",
            "process_type": "extraction",
            "status": "running",
            "pid": 12345,
            "started_at": "2025-09-27 10:30:00",
            "command": "python features/extraction/commands/extract_character.py",
            "log_file": "/tmp/subagent_TRACKER-001.log",
            "expected_duration": 3600
        },
        "KIRO-006": {
            "tracker_id": "KIRO-006",
            "process_type": "extraction",
            "status": "completed",
            "pid": None,
            "started_at": "2025-09-27 09:00:00",
            "completed_at": "2025-09-27 09:45:00",
            "command": "python features/extraction/commands/extract_character.py",
            "log_file": "/tmp/subagent_KIRO-006.log",
            "return_code": 0,
            "validation": "success"
        }
    }

    # エラーケース用データ
    ERROR_CASES = {
        "google_sheets_error": {
            "error_type": "ConnectionError",
            "message": "Google Sheets API connection failed"
        },
        "sqlite_error": {
            "error_type": "DatabaseError",
            "message": "SQLite database connection failed"
        },
        "workspace_not_found": {
            "error_type": "FileNotFoundError",
            "message": "Workspace directory not found"
        },
        "process_not_found": {
            "error_type": "ProcessLookupError",
            "message": "SubAgent process not found"
        }
    }

    @classmethod
    def get_mock_task(cls, tracker_id: str) -> MockTask:
        """指定されたトラッカーIDのモックタスクを取得"""
        return cls.MOCK_TASKS.get(tracker_id)

    @classmethod
    def get_mock_workflow_status(cls, tracker_id: str) -> Dict[str, Any]:
        """指定されたトラッカーIDのワークフロー状態を取得"""
        return cls.MOCK_WORKFLOW_STATUS.get(tracker_id, {
            "error": "Tracker not found",
            "status": "not_found"
        })

    @classmethod
    def get_mock_workspace_config(cls, tracker_id: str) -> Dict[str, Any]:
        """指定されたトラッカーIDのワークスペース設定を取得"""
        return cls.MOCK_WORKSPACE_CONFIG.get(tracker_id)

    @classmethod
    def get_mock_subagent_status(cls, tracker_id: str) -> Dict[str, Any]:
        """指定されたトラッカーIDのSubAgent状態を取得"""
        return cls.MOCK_SUBAGENT_STATUS.get(tracker_id)


# 定数定義
SAMPLE_TRACKER_ID = "TEST-001"
SAMPLE_SUMMARY = "テスト用概要"
SAMPLE_DETAILS = "これはテスト用の詳細説明です。" * 10  # 適度な長さ
SAMPLE_LONG_DETAILS = "これは非常に長い詳細説明です。" * 1000  # 制限超過用
SAMPLE_AUTHOR_NAME = "yado"
SAMPLE_PRIORITY = "medium"

# プロセス関連のモックデータ
MOCK_PROCESS_INFO = {
    "pid": 12345,
    "status": "running",
    "cpu_percent": 25.5,
    "memory_percent": 15.2,
    "create_time": 1695798600.0
}