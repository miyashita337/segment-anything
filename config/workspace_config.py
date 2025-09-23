#!/usr/bin/env python3
"""
ワークスペース設定管理 - 作者名・ワークフローパス管理システム
KIRO-006 解決策: ハードコードされたパスの動的設定化
"""

import sqlite3
import os
from typing import Optional, Dict, Any
from pathlib import Path
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkspaceConfig:
    """ワークスペース設定の管理クラス"""

    # 後方互換性のための定数（integrated_dashboard_server.py等で使用）
    BASE_TRAIN_PATH = "/mnt/c/AItools/lora/train"

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = self._find_project_root()
            db_path = os.path.join(project_root, "workspace_config.db")

        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        logger.info(f"ワークスペース設定管理が初期化されました: {db_path}")

    def _find_project_root(self) -> str:
        """プロジェクトルートディレクトリを探索"""
        current = os.path.abspath(__file__)
        while current != os.path.dirname(current):
            current = os.path.dirname(current)
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
        return os.path.dirname(os.path.abspath(__file__))

    def _init_database(self):
        """データベースとテーブルを初期化"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_configs (
                        tracker_id TEXT PRIMARY KEY,
                        author_name TEXT NOT NULL,
                        workspace_path TEXT NOT NULL,
                        input_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Workspace config database tables initialized successfully")

    def set_workspace_config(
        self,
        tracker_id: str,
        author_name: str,
        workspace_path: str,
        input_path: Optional[str] = None
    ) -> bool:
        """トラッカーのワークスペース設定を保存"""
        try:
            # パスの検証
            if not self._validate_workspace_path(workspace_path, author_name, tracker_id):
                return False

            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO workspace_configs
                        (tracker_id, author_name, workspace_path, input_path, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (tracker_id, author_name, workspace_path, input_path))
                    conn.commit()

            logger.info(f"Workspace config saved: {tracker_id} -> {author_name}@{workspace_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save workspace config: {e}")
            return False

    def get_workspace_config(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """トラッカーのワークスペース設定を取得"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT tracker_id, author_name, workspace_path, input_path,
                               created_at, updated_at
                        FROM workspace_configs
                        WHERE tracker_id = ?
                    """, (tracker_id,))
                    row = cursor.fetchone()

            if row:
                return {
                    'tracker_id': row[0],
                    'author_name': row[1],
                    'workspace_path': row[2],
                    'input_path': row[3],
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get workspace config: {e}")
            return None

    def get_author_workspace_base(self, author_name: str) -> str:
        """作者名に基づくワークスペースベースパスを生成"""
        return f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace"

    def get_default_input_path(self, author_name: str) -> str:
        """作者名に基づくデフォルト入力パスを生成"""
        return f"/mnt/c/AItools/lora/train/{author_name}/org/kana05"

    def get_input_directory(self, tracker_id: str) -> Optional[str]:
        """トラッカーの入力ディレクトリを取得"""
        try:
            config = self.get_workspace_config(tracker_id)
            if not config:
                logger.error(f"No workspace config found for {tracker_id}")
                return None

            input_path = config.get('input_path')
            if input_path and os.path.exists(input_path):
                return input_path

            # フォールバック: 作者名からデフォルトパスを生成
            author_name = config.get('author_name')
            if author_name:
                default_path = self.get_default_input_path(author_name)
                if os.path.exists(default_path):
                    logger.info(f"Using default input path for {tracker_id}: {default_path}")
                    return default_path

            logger.error(f"No valid input directory found for {tracker_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get input directory for {tracker_id}: {e}")
            return None

    def _validate_workspace_path(self, workspace_path: str, author_name: str, tracker_id: str) -> bool:
        """ワークスペースパスの妥当性を検証"""
        expected_pattern = f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace/{tracker_id}"

        if workspace_path != expected_pattern:
            logger.error(f"Invalid workspace path: {workspace_path}")
            logger.error(f"Expected: {expected_pattern}")
            return False

        return True

    def ensure_workspace_exists(self, tracker_id: str) -> bool:
        """ワークスペースディレクトリの存在を確保"""
        config = self.get_workspace_config(tracker_id)
        if not config:
            logger.error(f"No workspace config found for {tracker_id}")
            return False

        workspace_path = Path(config['workspace_path'])
        try:
            workspace_path.mkdir(parents=True, exist_ok=True)

            # 必要なサブディレクトリを作成
            subdirs = ['extraction', 'dashboard', '.approvals', 'logs']
            for subdir in subdirs:
                (workspace_path / subdir).mkdir(exist_ok=True)

            logger.info(f"Workspace directory ensured: {workspace_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create workspace directory: {e}")
            return False

    def list_author_trackers(self, author_name: str) -> list:
        """指定した作者のトラッカー一覧を取得"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT tracker_id, workspace_path, created_at
                        FROM workspace_configs
                        WHERE author_name = ?
                        ORDER BY created_at DESC
                    """, (author_name,))
                    rows = cursor.fetchall()

            return [
                {
                    'tracker_id': row[0],
                    'workspace_path': row[1],
                    'created_at': row[2]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to list author trackers: {e}")
            return []

    def delete_workspace_config(self, tracker_id: str) -> bool:
        """ワークスペース設定を削除"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM workspace_configs WHERE tracker_id = ?
                    """, (tracker_id,))
                    conn.commit()

            logger.info(f"Workspace config deleted: {tracker_id}")
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete workspace config: {e}")
            return False

    @classmethod
    def export_environment_variables(cls, tracker_id: str = None) -> Dict[str, str]:
        """環境変数として設定情報をエクスポート"""
        config = cls()
        env_vars = {}

        # トラッカーIDが指定されている場合はその設定を取得
        if tracker_id:
            workspace_info = config.get_workspace_config(tracker_id)
            if workspace_info:
                env_vars['TRACKER_ID'] = tracker_id
                env_vars['AUTHOR_NAME'] = workspace_info.get('author_name', '')
                env_vars['WORKSPACE_PATH'] = workspace_info.get('workspace_path', '')
                env_vars['INPUT_PATH'] = workspace_info.get('input_path', '')

        # デフォルト値も設定
        env_vars.setdefault('WORKSPACE_BASE', '/mnt/c/AItools/lora/train/yado/tracker-workspace')

        return env_vars


# グローバルインスタンス
_workspace_config = None


def get_workspace_config() -> WorkspaceConfig:
    """ワークスペース設定のグローバルインスタンスを取得"""
    global _workspace_config
    if _workspace_config is None:
        _workspace_config = WorkspaceConfig()
    return _workspace_config


def validate_tracker_setup(tracker_id: str) -> Dict[str, Any]:
    """トラッカーのセットアップ状況を検証"""
    config = get_workspace_config()
    workspace_config = config.get_workspace_config(tracker_id)

    result = {
        'tracker_id': tracker_id,
        'is_configured': workspace_config is not None,
        'errors': [],
        'warnings': []
    }

    if not workspace_config:
        result['errors'].append(f"❌ ワークスペース設定が未設定です: {tracker_id}")
        result['errors'].append("🔧 以下のコマンドで設定してください:")
        result['errors'].append(f"python tools/workflow/workflow_cli.py plan {tracker_id} <概要> <詳細> <作者名>")
        return result

    result['config'] = workspace_config

    # ワークスペースディレクトリの存在確認
    workspace_path = Path(workspace_config['workspace_path'])
    if not workspace_path.exists():
        result['warnings'].append(f"⚠️ ワークスペースディレクトリが存在しません: {workspace_path}")

    return result


if __name__ == "__main__":
    # テスト用
    config = WorkspaceConfig()

    # テスト設定
    test_tracker = "TEST-001"
    test_author = "test_user"
    test_workspace = f"/mnt/c/AItools/lora/train/{test_author}/tracker-workspace/{test_tracker}"

    # 設定保存テスト
    success = config.set_workspace_config(test_tracker, test_author, test_workspace)
    print(f"設定保存: {'成功' if success else '失敗'}")

    # 設定取得テスト
    retrieved = config.get_workspace_config(test_tracker)
    print(f"設定取得: {retrieved}")

    # 検証テスト
    validation = validate_tracker_setup(test_tracker)
    print(f"検証結果: {validation}")