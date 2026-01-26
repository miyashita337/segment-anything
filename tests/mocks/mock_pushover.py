#!/usr/bin/env python3
"""
Mock Pushover notification system for workflow testing

本物のPushover APIを使用せずに通知機能をテストするためのモックシステム
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PushoverMessage:
    """Pushover メッセージ データクラス"""

    message: str
    title: str
    priority: int
    device: Optional[str]
    timestamp: datetime
    user_key: str
    api_token: str
    attachment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返却"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class MockPushoverClient:
    """Mock Pushover クライアント - 実際の通知なしでテスト可能"""

    def __init__(self):
        self.sent_messages: List[PushoverMessage] = []
        self.message_log_file = Path("tests/fixtures/mock_pushover_messages.json")
        self.api_call_count = 0
        self.failure_mode = False
        self.failure_count = 0

    def send_notification(
        self,
        message: str,
        title: str = "Test Notification",
        priority: int = 0,
        device: Optional[str] = None,
        user_key: str = "mock_user_key",
        api_token: str = "mock_api_token",
        attachment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mock通知送信

        Args:
            message: 通知メッセージ
            title: 通知タイトル
            priority: 優先度 (-2 to 2)
            device: デバイス指定
            user_key: ユーザーキー
            api_token: APIトークン
            attachment: 添付ファイルパス

        Returns:
            送信結果の辞書
        """
        self.api_call_count += 1

        # 失敗モードの場合
        if self.failure_mode:
            self.failure_count += 1
            return {"status": 0, "error": "Mock API failure", "request": str(self.api_call_count)}

        # 成功時の処理
        notification = PushoverMessage(
            message=message,
            title=title,
            priority=priority,
            device=device,
            timestamp=datetime.now(),
            user_key=user_key,
            api_token=api_token,
            attachment=attachment,
        )

        self.sent_messages.append(notification)

        # メッセージログファイルに保存
        self._save_message_log()

        return {
            "status": 1,
            "request": str(self.api_call_count),
            "message_id": f"mock_msg_{self.api_call_count}",
            "timestamp": notification.timestamp.isoformat(),
        }

    def send_extraction_complete_notification(
        self,
        tracker_id: str,
        total_images: int,
        successful_extractions: int,
        success_rate: float,
        workspace_path: str,
        attachment_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        抽出完了通知（特化版）

        Args:
            tracker_id: トラッカーID
            total_images: 総画像数
            successful_extractions: 成功抽出数
            success_rate: 成功率
            workspace_path: ワークスペースパス
            attachment_images: 添付画像リスト

        Returns:
            送信結果の辞書
        """
        message = f"""🎯 {tracker_id} 抽出完了

📊 結果:
• 総画像数: {total_images}枚
• 成功抽出: {successful_extractions}枚  
• 成功率: {success_rate:.1f}%

📁 出力: {workspace_path}"""

        title = f"Claude Code - {tracker_id} Complete"

        # 添付ファイル処理（最大10枚）
        attachment_count = len(attachment_images) if attachment_images else 0
        if attachment_count > 0:
            message += f"\n\n📎 添付画像: {min(attachment_count, 10)}枚"

        return self.send_notification(
            message=message,
            title=title,
            priority=1,  # 高優先度
            attachment=str(attachment_count) if attachment_count > 0 else None,
        )

    def send_approval_request(
        self, tracker_id: str, stage: str, details: str, approval_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        承認依頼通知

        Args:
            tracker_id: トラッカーID
            stage: 承認ステージ
            details: 詳細情報
            approval_url: 承認用URL

        Returns:
            送信結果の辞書
        """
        message = f"""⚠️ {tracker_id} 承認依頼

🔄 ステージ: {stage}

📋 詳細:
{details}"""

        if approval_url:
            message += f"\n\n🔗 承認URL: {approval_url}"

        return self.send_notification(message=message, title=f"承認依頼 - {tracker_id}", priority=1)

    def send_quality_alert(
        self, tracker_id: str, quality_score: float, threshold: float, failed_images: List[str]
    ) -> Dict[str, Any]:
        """
        品質アラート通知

        Args:
            tracker_id: トラッカーID
            quality_score: 品質スコア
            threshold: 閾値
            failed_images: 失敗画像リスト

        Returns:
            送信結果の辞書
        """
        message = f"""⚠️ {tracker_id} 品質アラート

📉 品質スコア: {quality_score:.3f}
📏 閾値: {threshold:.3f}

❌ 失敗画像: {len(failed_images)}件"""

        if len(failed_images) <= 5:
            message += f"\n\n失敗ファイル:\n" + "\n".join([f"• {img}" for img in failed_images])
        else:
            message += f"\n\n失敗ファイル:\n" + "\n".join([f"• {img}" for img in failed_images[:3]])
            message += f"\n• ... 他{len(failed_images)-3}件"

        return self.send_notification(
            message=message, title=f"品質アラート - {tracker_id}", priority=2  # 緊急
        )

    def get_sent_messages(self, limit: Optional[int] = None) -> List[PushoverMessage]:
        """送信済みメッセージ取得"""
        messages = self.sent_messages.copy()
        if limit:
            messages = messages[-limit:]
        return messages

    def get_messages_by_title_pattern(self, pattern: str) -> List[PushoverMessage]:
        """タイトルパターンでメッセージ検索"""
        import re

        matching_messages = []
        for msg in self.sent_messages:
            if re.search(pattern, msg.title):
                matching_messages.append(msg)
        return matching_messages

    def clear_message_history(self):
        """メッセージ履歴クリア"""
        self.sent_messages.clear()
        self.api_call_count = 0
        self.failure_count = 0
        if self.message_log_file.exists():
            self.message_log_file.unlink()

    def set_failure_mode(self, enabled: bool):
        """失敗モード設定"""
        self.failure_mode = enabled
        if enabled:
            self.failure_count = 0

    def get_api_statistics(self) -> Dict[str, int]:
        """API統計情報取得"""
        return {
            "total_calls": self.api_call_count,
            "successful_calls": self.api_call_count - self.failure_count,
            "failed_calls": self.failure_count,
            "messages_sent": len(self.sent_messages),
        }

    def _save_message_log(self):
        """メッセージログファイル保存"""
        self.message_log_file.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            "messages": [msg.to_dict() for msg in self.sent_messages],
            "statistics": self.get_api_statistics(),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.message_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


# グローバルモックインスタンス（シングルトン）
_mock_pushover_instance = None


def get_mock_pushover_client() -> MockPushoverClient:
    """Mock Pushoverクライアントのシングルトン取得"""
    global _mock_pushover_instance
    if _mock_pushover_instance is None:
        _mock_pushover_instance = MockPushoverClient()
    return _mock_pushover_instance


def reset_mock_pushover_client():
    """Mock Pushoverクライアントのリセット"""
    global _mock_pushover_instance
    if _mock_pushover_instance:
        _mock_pushover_instance.clear_message_history()
    _mock_pushover_instance = None
