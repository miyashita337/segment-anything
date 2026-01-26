#!/usr/bin/env python3
"""
INTG-089 Pushoverクライアント用モッククラス
テスト環境でのPushover API通信をシミュレート
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MockPushoverResponse:
    """Pushover APIレスポンスのモック"""

    def __init__(self, status_code: int = 200, response_data: Dict[str, Any] = None):
        self.status_code = status_code
        self.response_data = response_data or {"status": 1}

    def json(self) -> Dict[str, Any]:
        """JSON形式のレスポンスデータを返す"""
        return self.response_data


class MockPushoverClient:
    """Pushoverクライアントのモッククラス"""

    def __init__(self, config_path: Optional[str] = None, simulate_network_errors: bool = False):
        """
        初期化

        Args:
            config_path: 設定ファイルパス（実際は使用しない）
            simulate_network_errors: ネットワークエラーをシミュレートするか
        """
        self.config_path = config_path
        self.simulate_network_errors = simulate_network_errors

        # デフォルト設定
        self.config = {"user_key": "mock_user_key", "api_token": "mock_api_token"}

        self.api_url = "https://api.pushover.net/1/messages.json"

        # 送信履歴記録
        self.sent_messages: List[Dict[str, Any]] = []
        self.call_count = 0

        # エラーシミュレーション設定
        self.network_error_rate = 0.0  # 0.0-1.0 (0%から100%)
        self.api_error_rate = 0.0

        # レスポンス遅延シミュレーション
        self.response_delay = 0.0  # 秒

    def send_notification(self, title: str, message: str, priority: int = 0, **kwargs) -> bool:
        """
        通知送信のモック実装

        Args:
            title: 通知タイトル
            message: 通知メッセージ
            priority: 優先度
            **kwargs: その他のパラメータ

        Returns:
            bool: 送信成功かどうか
        """
        self.call_count += 1

        # レスポンス遅延シミュレーション
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        # ネットワークエラーシミュレーション
        if self.simulate_network_errors and self._should_simulate_error(self.network_error_rate):
            raise Exception(f"Simulated network error (call #{self.call_count})")

        # APIエラーシミュレーション
        if self._should_simulate_error(self.api_error_rate):
            self._record_message(title, message, priority, success=False, error="API Error")
            return False

        # 正常送信の記録
        self._record_message(title, message, priority, success=True)
        return True

    def _should_simulate_error(self, error_rate: float) -> bool:
        """エラーをシミュレートするかどうかの判定"""
        import random

        return random.random() < error_rate

    def _record_message(
        self, title: str, message: str, priority: int, success: bool, error: Optional[str] = None
    ):
        """送信メッセージの記録"""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "message": message,
            "priority": priority,
            "success": success,
            "call_number": self.call_count,
        }

        if error:
            record["error"] = error

        self.sent_messages.append(record)

    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """送信済みメッセージ一覧を取得"""
        return self.sent_messages.copy()

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """最後に送信したメッセージを取得"""
        return self.sent_messages[-1] if self.sent_messages else None

    def clear_history(self):
        """送信履歴をクリア"""
        self.sent_messages.clear()
        self.call_count = 0

    def set_network_error_rate(self, rate: float):
        """ネットワークエラー発生率を設定 (0.0-1.0)"""
        self.network_error_rate = max(0.0, min(1.0, rate))

    def set_api_error_rate(self, rate: float):
        """APIエラー発生率を設定 (0.0-1.0)"""
        self.api_error_rate = max(0.0, min(1.0, rate))

    def set_response_delay(self, delay: float):
        """レスポンス遅延を設定 (秒)"""
        self.response_delay = max(0.0, delay)

    def get_statistics(self) -> Dict[str, Any]:
        """送信統計を取得"""
        if not self.sent_messages:
            return {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "success_rate": 0.0}

        successful = sum(1 for msg in self.sent_messages if msg["success"])
        failed = len(self.sent_messages) - successful

        return {
            "total_calls": self.call_count,
            "successful_calls": successful,
            "failed_calls": failed,
            "success_rate": successful / len(self.sent_messages) if self.sent_messages else 0.0,
            "first_message_time": self.sent_messages[0]["timestamp"]
            if self.sent_messages
            else None,
            "last_message_time": self.sent_messages[-1]["timestamp"]
            if self.sent_messages
            else None,
        }


class MockNotificationBridge:
    """NotificationBridge全体のモッククラス"""

    def __init__(self, workspace_path: str, pushover_config_path: Optional[str] = None):
        """
        初期化

        Args:
            workspace_path: ワークスペースパス
            pushover_config_path: Pushover設定ファイルパス
        """
        self.workspace_path = workspace_path
        self.pushover_config_path = pushover_config_path

        # モックPushoverクライアント
        self.pushover = MockPushoverClient(config_path=pushover_config_path)

        # 重複防止用履歴
        self.notification_history: Dict[str, Dict[str, Any]] = {}

        # 通知優先度設定
        self.notification_priorities = {
            "critical": {"priority": 2, "min_interval": 0},
            "high": {"priority": 1, "min_interval": 60},
            "normal": {"priority": 0, "min_interval": 300},
            "low": {"priority": -1, "min_interval": 1800},
        }

    def send_enhanced_notification(
        self,
        title: str,
        message: str,
        notification_type: str = "normal",
        priority_level: str = "normal",
    ) -> bool:
        """
        強化通知送信のモック実装

        Args:
            title: タイトル
            message: メッセージ
            notification_type: 通知タイプ
            priority_level: 優先度レベル

        Returns:
            bool: 送信成功かどうか
        """
        # ハッシュ生成（重複防止用）
        notification_hash = self._generate_notification_hash(title, message, notification_type)

        # 重複チェック
        if self._should_suppress_notification(notification_hash, priority_level):
            return False  # 重複により送信抑制

        # 優先度設定取得
        priority_config = self.notification_priorities.get(
            priority_level, self.notification_priorities["normal"]
        )
        pushover_priority = priority_config["priority"]

        # Pushover送信
        success = self.pushover.send_notification(title, message, priority=pushover_priority)

        # 履歴更新
        if success:
            self.notification_history[notification_hash] = {
                "last_sent": time.time(),
                "count": self.notification_history.get(notification_hash, {}).get("count", 0) + 1,
                "title": title,
                "message": message,
                "priority_level": priority_level,
            }

        return success

    def _generate_notification_hash(self, title: str, message: str, notification_type: str) -> str:
        """通知ハッシュ生成"""
        import hashlib

        content = f"{title}|{message}|{notification_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def _should_suppress_notification(self, notification_hash: str, priority_level: str) -> bool:
        """重複抑制判定"""
        if notification_hash not in self.notification_history:
            return False

        last_sent = self.notification_history[notification_hash]["last_sent"]
        priority_config = self.notification_priorities.get(
            priority_level, self.notification_priorities["normal"]
        )
        min_interval = priority_config["min_interval"]

        # 最小間隔経過チェック
        return (time.time() - last_sent) < min_interval

    def get_notification_statistics(self) -> Dict[str, Any]:
        """通知統計取得"""
        pushover_stats = self.pushover.get_statistics()

        unique_notifications = len(self.notification_history)
        total_notifications = sum(
            history["count"] for history in self.notification_history.values()
        )

        return {
            "pushover_stats": pushover_stats,
            "unique_notifications": unique_notifications,
            "total_notifications": total_notifications,
            "duplicate_prevention_rate": (total_notifications - pushover_stats["total_calls"])
            / total_notifications
            if total_notifications > 0
            else 0.0,
        }


class MockPushoverTestScenarios:
    """テストシナリオ用のヘルパークラス"""

    @staticmethod
    def create_high_error_rate_client() -> MockPushoverClient:
        """高エラー率クライアント作成"""
        client = MockPushoverClient()
        client.set_network_error_rate(0.3)  # 30%のネットワークエラー
        client.set_api_error_rate(0.2)  # 20%のAPIエラー
        return client

    @staticmethod
    def create_slow_response_client(delay: float = 2.0) -> MockPushoverClient:
        """レスポンス遅延クライアント作成"""
        client = MockPushoverClient()
        client.set_response_delay(delay)
        return client

    @staticmethod
    def create_reliable_client() -> MockPushoverClient:
        """信頼性の高いクライアント作成"""
        client = MockPushoverClient()
        client.set_network_error_rate(0.0)
        client.set_api_error_rate(0.0)
        client.set_response_delay(0.1)
        return client


# グローバルモックインスタンス管理
_mock_pushover_instance: Optional[MockPushoverClient] = None
_mock_bridge_instance: Optional[MockNotificationBridge] = None


def get_mock_pushover_client() -> MockPushoverClient:
    """グローバルモックPushoverクライアント取得"""
    global _mock_pushover_instance
    if _mock_pushover_instance is None:
        _mock_pushover_instance = MockPushoverClient()
    return _mock_pushover_instance


def get_mock_notification_bridge(workspace_path: str) -> MockNotificationBridge:
    """グローバルモックNotificationBridge取得"""
    global _mock_bridge_instance
    if _mock_bridge_instance is None:
        _mock_bridge_instance = MockNotificationBridge(workspace_path=workspace_path)
    return _mock_bridge_instance


def reset_mock_pushover_client():
    """グローバルモックPushoverクライアントリセット"""
    global _mock_pushover_instance
    if _mock_pushover_instance:
        _mock_pushover_instance.clear_history()
    _mock_pushover_instance = None


def reset_mock_notification_bridge():
    """グローバルモックNotificationBridgeリセット"""
    global _mock_bridge_instance
    _mock_bridge_instance = None


# テスト用便利関数
def assert_notification_sent(mock_client: MockPushoverClient, title: str, message: str) -> bool:
    """通知が送信されたことをアサート"""
    for msg in mock_client.get_sent_messages():
        if msg["title"] == title and msg["message"] == message and msg["success"]:
            return True
    return False


def get_notification_count_by_priority(mock_client: MockPushoverClient, priority: int) -> int:
    """指定優先度の通知数を取得"""
    return sum(
        1
        for msg in mock_client.get_sent_messages()
        if msg["priority"] == priority and msg["success"]
    )


def simulate_notification_burst(bridge: MockNotificationBridge, count: int = 10) -> Dict[str, int]:
    """通知バースト送信シミュレーション"""
    results = {"sent": 0, "suppressed": 0}

    for i in range(count):
        success = bridge.send_enhanced_notification(
            f"Burst Test #{i}",
            "Duplicate test message",
            notification_type="burst_test",
            priority_level="normal",
        )

        if success:
            results["sent"] += 1
        else:
            results["suppressed"] += 1

    return results
