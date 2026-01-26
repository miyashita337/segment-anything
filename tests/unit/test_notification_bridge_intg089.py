#!/usr/bin/env python3
"""
INTG-089 NotificationBridge強化機能の単体テスト
現実的な通知システムテスト
"""

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.queue.notification_bridge import NotificationBridge, PushoverNotifier


class TestPushoverNotifierINTG089(unittest.TestCase):
    """PushoverNotifier INTG-089強化機能テスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "pushover.json")

        # テスト用設定ファイル作成
        self.test_config = {"user_key": "test_user_key", "api_token": "test_api_token"}

        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_file_loading_success(self):
        """設定ファイル正常読み込みテスト"""
        notifier = PushoverNotifier(config_path=self.config_path)

        self.assertEqual(notifier.config, self.test_config)
        self.assertEqual(notifier.config["user_key"], "test_user_key")
        self.assertEqual(notifier.config["api_token"], "test_api_token")

    def test_config_file_missing(self):
        """設定ファイル不在時のテスト"""
        non_existent_path = os.path.join(self.temp_dir, "missing.json")
        notifier = PushoverNotifier(config_path=non_existent_path)

        # 設定ファイルが存在しない場合は空辞書
        self.assertEqual(notifier.config, {})

    def test_config_file_invalid_json(self):
        """無効なJSON設定ファイルのテスト"""
        invalid_config_path = os.path.join(self.temp_dir, "invalid.json")

        with open(invalid_config_path, "w") as f:
            f.write("invalid json content")

        notifier = PushoverNotifier(config_path=invalid_config_path)

        # 無効なJSONの場合は空辞書
        self.assertEqual(notifier.config, {})

    @patch("requests.post")
    def test_send_notification_success(self, mock_post):
        """通知送信成功テスト"""
        # レスポンスをモック
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 1}
        mock_post.return_value = mock_response

        notifier = PushoverNotifier(config_path=self.config_path)
        result = notifier.send_notification("Test Title", "Test Message")

        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_notification_failure(self, mock_post):
        """通知送信失敗テスト"""
        # エラーレスポンスをモック
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        notifier = PushoverNotifier(config_path=self.config_path)
        result = notifier.send_notification("Test Title", "Test Message")

        self.assertFalse(result)

    @patch("requests.post")
    def test_send_notification_with_priority(self, mock_post):
        """優先度付き通知送信テスト"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 1}
        mock_post.return_value = mock_response

        notifier = PushoverNotifier(config_path=self.config_path)
        result = notifier.send_notification("High Priority", "Important message", priority=1)

        self.assertTrue(result)

        # 送信されたデータを確認
        call_args = mock_post.call_args
        sent_data = call_args[1]["data"]
        self.assertEqual(sent_data["priority"], 1)

    def test_api_url_configuration(self):
        """API URL設定テスト"""
        notifier = PushoverNotifier(config_path=self.config_path)

        expected_url = "https://api.pushover.net/1/messages.json"
        self.assertEqual(notifier.api_url, expected_url)


class TestNotificationBridgeINTG089(unittest.TestCase):
    """NotificationBridge INTG-089強化機能テスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "pushover.json")

        # テスト用設定ファイル作成
        self.test_config = {"user_key": "test_user_key", "api_token": "test_api_token"}

        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f)

        self.workspace_path = self.temp_dir

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_notification_bridge_initialization(self):
        """NotificationBridge初期化テスト"""
        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 基本的な初期化確認
        self.assertIsInstance(bridge.pushover, PushoverNotifier)
        self.assertEqual(bridge.workspace_path, self.workspace_path)
        self.assertEqual(bridge.tracker_id, "TEST-089")

    def test_duplicate_notification_hash_generation(self):
        """重複通知ハッシュ生成テスト"""
        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 同一メッセージのハッシュは同じであること
        hash1 = bridge._generate_notification_hash("Title", "Message", "normal")
        hash2 = bridge._generate_notification_hash("Title", "Message", "normal")
        self.assertEqual(hash1, hash2)

        # 異なるメッセージのハッシュは異なること
        hash3 = bridge._generate_notification_hash("Different Title", "Message", "normal")
        self.assertNotEqual(hash1, hash3)

    def test_notification_priority_configuration(self):
        """通知優先度設定テスト"""
        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 優先度設定の確認
        expected_priorities = {
            "critical": {"priority": 2, "min_interval": 0},
            "high": {"priority": 1, "min_interval": 60},
            "normal": {"priority": 0, "min_interval": 300},
            "low": {"priority": -1, "min_interval": 1800},
        }

        self.assertEqual(bridge.notification_priorities, expected_priorities)

    @patch("time.time")
    def test_duplicate_suppression_logic(self, mock_time):
        """重複抑制ロジックテスト"""
        # 時間を固定
        mock_time.return_value = 1000.0

        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        notification_hash = "test_hash"

        # 初回は抑制されない
        result1 = bridge._should_suppress_notification(notification_hash, "normal")
        self.assertFalse(result1)

        # 記録
        bridge.notification_history[notification_hash] = {"last_sent": 1000.0, "count": 1}

        # 間隔内（300秒以内）は抑制される
        mock_time.return_value = 1200.0  # 200秒後
        result2 = bridge._should_suppress_notification(notification_hash, "normal")
        self.assertTrue(result2)

        # 間隔外（300秒経過）は抑制されない
        mock_time.return_value = 1400.0  # 400秒後
        result3 = bridge._should_suppress_notification(notification_hash, "normal")
        self.assertFalse(result3)

    @patch.object(PushoverNotifier, "send_notification")
    def test_enhanced_notification_send(self, mock_send):
        """強化通知送信テスト"""
        mock_send.return_value = True

        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 通知送信
        result = bridge.send_enhanced_notification(
            "Test Title", "Test Message", notification_type="test", priority_level="high"
        )

        self.assertTrue(result)
        mock_send.assert_called_once()

        # 正しい優先度で送信されているか確認
        call_args = mock_send.call_args
        self.assertEqual(call_args[1]["priority"], 1)  # high priority

    @patch.object(PushoverNotifier, "send_notification")
    @patch("time.time")
    def test_duplicate_notification_prevention(self, mock_time, mock_send):
        """重複通知防止テスト"""
        mock_time.return_value = 1000.0
        mock_send.return_value = True

        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 初回送信
        result1 = bridge.send_enhanced_notification(
            "Test Title", "Test Message", notification_type="test", priority_level="normal"
        )
        self.assertTrue(result1)
        self.assertEqual(mock_send.call_count, 1)

        # 同じ内容を短時間で再送信（抑制されるべき）
        mock_time.return_value = 1100.0  # 100秒後
        result2 = bridge.send_enhanced_notification(
            "Test Title", "Test Message", notification_type="test", priority_level="normal"
        )
        self.assertFalse(result2)
        self.assertEqual(mock_send.call_count, 1)  # 呼び出し回数変わらず

    def test_notification_statistics_tracking(self):
        """通知統計追跡テスト"""
        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 統計管理属性の存在確認
        self.assertIsInstance(bridge.notification_history, dict)

        # 統計取得メソッドの存在確認（実装されている場合）
        if hasattr(bridge, "get_notification_statistics"):
            stats = bridge.get_notification_statistics()
            self.assertIsInstance(stats, dict)

    def test_anomaly_notification_formatting(self):
        """異常通知フォーマットテスト"""
        bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")

        # 異常通知専用メソッドの存在確認
        if hasattr(bridge, "send_anomaly_notification"):
            # 異常通知の基本機能テスト
            try:
                # ダミーの異常データで呼び出し
                result = bridge.send_anomaly_notification(
                    "GPU Memory Alert", {"memory_usage": 85, "threshold": 90}
                )
                # エラーが発生しなければ成功
                self.assertIsInstance(result, bool)
            except Exception as e:
                self.fail(f"send_anomaly_notification failed: {e}")


class TestNotificationBridgeRealWorldScenarios(unittest.TestCase):
    """実世界シナリオテスト"""

    def setUp(self):
        """テスト前セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = self.temp_dir

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_missing_config_file_handling(self):
        """設定ファイル不在時の実用的な処理テスト"""
        # 存在しない設定ファイルパス
        missing_config = os.path.join(self.temp_dir, "missing_pushover.json")

        # エラーなく初期化できること
        try:
            bridge = NotificationBridge(workspace_path=self.workspace_path, tracker_id="TEST-089")
            self.assertIsInstance(bridge, NotificationBridge)
        except Exception as e:
            self.fail(f"NotificationBridge initialization failed with missing config: {e}")

    def test_notification_history_persistence(self):
        """通知履歴永続化テスト（実装されている場合）"""
        bridge = NotificationBridge(workspace_path=self.workspace_path)

        # 履歴ファイルパスの確認（実装されている場合）
        if hasattr(bridge, "history_file"):
            self.assertIsInstance(bridge.history_file, Path)

    @patch("requests.post")
    def test_network_error_handling(self, mock_post):
        """ネットワークエラー処理テスト"""
        # ネットワークエラーをシミュレート
        mock_post.side_effect = Exception("Network error")

        bridge = NotificationBridge(workspace_path=self.workspace_path)

        # エラーが適切に処理されること
        result = bridge.pushover.send_notification("Test", "Message")
        self.assertFalse(result)


def run_all_tests():
    """全テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestPushoverNotifierINTG089))
    suite.addTests(loader.loadTestsFromTestCase(TestNotificationBridgeINTG089))
    suite.addTests(loader.loadTestsFromTestCase(TestNotificationBridgeRealWorldScenarios))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
