#!/usr/bin/env python3
"""
Claude実行権限管理システム
暴走防止のための権限レベル制御機能
"""

import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum, IntEnum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """権限レベル定義"""

    READ_ONLY = 1  # 読み取り専用
    PLAN_ONLY = 2  # 計画モードのみ
    EXECUTE_STEP_BY_STEP = 3  # 段階実行（各操作に確認必要）
    EXECUTE_FULL = 4  # 完全実行権限


class ActionType(Enum):
    """アクション種別"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    GIT_OPERATION = "git_operation"
    SYSTEM_COMMAND = "system_command"
    CONFIG_CHANGE = "config_change"


class PermissionViolationError(Exception):
    """権限違反エラー"""

    pass


class ExecutionPermissionManager:
    """実行権限管理クラス"""

    # デフォルト設定
    DEFAULT_LEVEL = PermissionLevel.EXECUTE_FULL  # 互換性のためデフォルトは完全権限
    STATE_FILE = ".claude_execution_state.json"
    BACKUP_FILE = ".claude_execution_state.backup.json"

    def __init__(self, state_file: Optional[str] = None):
        """
        初期化

        Args:
            state_file: 状態ファイルパス（省略時はデフォルト）
        """
        self.state_file = Path(state_file or self.STATE_FILE)
        self.backup_file = Path(self.state_file.parent / self.BACKUP_FILE)
        self.session_id = str(uuid.uuid4())
        self.state = self._load_state()

        # フィーチャーフラグチェック
        self.enabled = os.getenv("CLAUDE_PERMISSION_ENABLED", "false").lower() == "true"

        if self.enabled:
            logger.info(f"権限管理システム有効化 (セッション: {self.session_id})")
            logger.info(f"現在の権限レベル: {self.get_current_level().name}")
        else:
            logger.debug("権限管理システムは無効です (CLAUDE_PERMISSION_ENABLED=false)")

    def _load_state(self) -> Dict[str, Any]:
        """状態ファイル読み込み"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    logger.debug(f"状態ファイル読み込み成功: {self.state_file}")
                    return state
        except Exception as e:
            logger.warning(f"状態ファイル読み込みエラー: {e}")

            # バックアップから復旧試行
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                        logger.info("バックアップから状態を復旧しました")
                        return state
                except Exception as backup_error:
                    logger.error(f"バックアップ復旧失敗: {backup_error}")

        # デフォルト状態
        env_level = os.getenv("CLAUDE_PERMISSION_LEVEL", "EXECUTE_FULL")
        try:
            default_level = PermissionLevel[env_level]
        except KeyError:
            default_level = self.DEFAULT_LEVEL
            logger.warning(f"無効な権限レベル: {env_level}, デフォルト使用: {default_level.name}")

        return {
            "current_level": default_level.name,
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "audit_log": [],
            "user_confirmations": {},
        }

    def _save_state(self) -> None:
        """状態ファイル保存"""
        try:
            # バックアップ作成
            if self.state_file.exists():
                import shutil

                shutil.copy2(self.state_file, self.backup_file)

            # 状態保存
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            logger.debug(f"状態ファイル保存成功: {self.state_file}")
        except Exception as e:
            logger.error(f"状態ファイル保存エラー: {e}")

    def get_current_level(self) -> PermissionLevel:
        """現在の権限レベル取得"""
        level_name = self.state.get("current_level", self.DEFAULT_LEVEL.name)
        try:
            return PermissionLevel[level_name]
        except KeyError:
            logger.warning(f"無効な権限レベル: {level_name}")
            return self.DEFAULT_LEVEL

    def set_permission_level(self, level: PermissionLevel) -> None:
        """
        権限レベル設定

        Args:
            level: 新しい権限レベル
        """
        old_level = self.get_current_level()
        self.state["current_level"] = level.name
        self._save_state()

        # 監査ログ記録
        self._add_audit_log(
            "permission_change", {"old_level": old_level.name, "new_level": level.name}
        )

        logger.info(f"権限レベル変更: {old_level.name} -> {level.name}")

    def check_permission(self, action: ActionType, target: Optional[str] = None) -> bool:
        """
        権限チェック

        Args:
            action: アクション種別
            target: 対象（ファイルパス、コマンド等）

        Returns:
            bool: 権限がある場合True
        """
        # 無効時は常に許可
        if not self.enabled:
            return True

        current_level = self.get_current_level()

        # 権限マトリックス
        permission_matrix = {
            PermissionLevel.READ_ONLY: [ActionType.READ],
            PermissionLevel.PLAN_ONLY: [ActionType.READ],
            PermissionLevel.EXECUTE_STEP_BY_STEP: [
                ActionType.READ,
                ActionType.WRITE,
                ActionType.EXECUTE,
                ActionType.CONFIG_CHANGE,
            ],
            PermissionLevel.EXECUTE_FULL: [
                ActionType.READ,
                ActionType.WRITE,
                ActionType.DELETE,
                ActionType.EXECUTE,
                ActionType.GIT_OPERATION,
                ActionType.SYSTEM_COMMAND,
                ActionType.CONFIG_CHANGE,
            ],
        }

        allowed_actions = permission_matrix.get(current_level, [])
        is_allowed = action in allowed_actions

        # 監査ログ記録
        self._add_audit_log(
            "permission_check",
            {
                "action": action.value,
                "target": target,
                "level": current_level.name,
                "allowed": is_allowed,
            },
        )

        if not is_allowed:
            logger.warning(f"権限違反: {action.value} は {current_level.name} では許可されていません")

        return is_allowed

    def require_confirmation(self, action: str, details: Optional[str] = None) -> bool:
        """
        ユーザー確認要求

        Args:
            action: アクション説明
            details: 詳細情報

        Returns:
            bool: ユーザーが承認した場合True
        """
        # EXECUTE_STEP_BY_STEPの場合のみ確認
        if self.get_current_level() != PermissionLevel.EXECUTE_STEP_BY_STEP:
            return True

        # 環境変数で自動承認（CI/CD用）
        if os.getenv("CLAUDE_AUTO_APPROVE", "false").lower() == "true":
            logger.info(f"自動承認: {action}")
            return True

        print("\n" + "=" * 60)
        print("🔒 実行確認が必要です")
        print(f"アクション: {action}")
        if details:
            print(f"詳細: {details}")
        print("=" * 60)

        try:
            response = input("実行しますか？ (y/N): ").strip().lower()
            approved = response == "y"

            # 監査ログ記録
            self._add_audit_log(
                "user_confirmation", {"action": action, "details": details, "approved": approved}
            )

            return approved
        except (KeyboardInterrupt, EOFError):
            logger.info("ユーザーによる中断")
            return False

    def _add_audit_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        監査ログ追加

        Args:
            event_type: イベント種別
            data: イベントデータ
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "data": data,
        }

        # メモリ上のログに追加
        if "audit_log" not in self.state:
            self.state["audit_log"] = []

        self.state["audit_log"].append(log_entry)

        # ログローテーション（最新1000件のみ保持）
        if len(self.state["audit_log"]) > 1000:
            self.state["audit_log"] = self.state["audit_log"][-1000:]

        # 定期的に保存（10エントリごと）
        if len(self.state["audit_log"]) % 10 == 0:
            self._save_state()

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        監査ログ取得

        Args:
            limit: 取得件数上限

        Returns:
            List[Dict]: 監査ログエントリ
        """
        audit_log = self.state.get("audit_log", [])
        return audit_log[-limit:] if audit_log else []

    def enforce_permission(self, action: ActionType, target: Optional[str] = None) -> None:
        """
        権限強制チェック（違反時は例外発生）

        Args:
            action: アクション種別
            target: 対象

        Raises:
            PermissionViolationError: 権限がない場合
        """
        if not self.check_permission(action, target):
            current_level = self.get_current_level()
            error_msg = f"権限違反: {action.value} は現在の権限レベル " f"{current_level.name} では実行できません。"
            if target:
                error_msg += f" (対象: {target})"

            raise PermissionViolationError(error_msg)


def require_permission(action_type: ActionType):
    """
    権限チェックデコレータ

    Args:
        action_type: 必要な権限種別

    Usage:
        @require_permission(ActionType.WRITE)
        def update_file(path):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_permission_manager()  # グローバルインスタンス使用

            # 権限チェック
            target = None
            if args and len(args) > 0:
                # 第一引数を文字列化してtargetとする
                first_arg = args[0]
                # オブジェクトの場合はスキップ
                if not hasattr(first_arg, "__dict__"):
                    target = str(first_arg)

            manager.enforce_permission(action_type, target)

            # EXECUTE_STEP_BY_STEPの場合は確認
            if manager.get_current_level() == PermissionLevel.EXECUTE_STEP_BY_STEP:
                func_name = func.__name__
                if not manager.require_confirmation(f"関数実行: {func_name}", target):
                    raise PermissionViolationError(f"ユーザーが実行を拒否しました: {func_name}")

            # 実行
            return func(*args, **kwargs)

        return wrapper

    return decorator


# グローバルインスタンス（シングルトン）
_global_manager: Optional[ExecutionPermissionManager] = None


def get_permission_manager() -> ExecutionPermissionManager:
    """グローバル権限マネージャー取得"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ExecutionPermissionManager()
    return _global_manager
