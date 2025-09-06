#!/usr/bin/env python3
"""
通知ブリッジシステム
QUAL-044: Pushover通知とPlanModeエスカレーション統合

タスク完了・失敗時の通知とエラー時のPlanMode連携
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PushoverNotifier:
    """Pushover通知クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: Pushover設定ファイルパス
        """
        if config_path is None:
            config_path = "/mnt/c/AItools/segment-anything/config/pushover.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.api_url = "https://api.pushover.net/1/messages.json"
        
        logger.info("PushoverNotifier initialized")
    
    def _load_config(self) -> Dict[str, str]:
        """設定ファイル読み込み"""
        if not self.config_path.exists():
            logger.warning(f"Pushover config not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Pushover config: {e}")
            return {}
    
    def send_notification(self,
                         title: str,
                         message: str,
                         priority: int = 0,
                         url: Optional[str] = None) -> bool:
        """
        通知送信
        
        Args:
            title: 通知タイトル
            message: 通知メッセージ
            priority: 優先度 (-2 to 2)
            url: 関連URL
            
        Returns:
            送信成功フラグ
        """
        if not self.config:
            logger.warning("Pushover not configured, skipping notification")
            return False
        
        payload = {
            'token': self.config.get('api_token'),
            'user': self.config.get('user_key'),
            'title': title,
            'message': message,
            'priority': priority,
            'timestamp': int(datetime.now().timestamp())
        }
        
        if url:
            payload['url'] = url
            payload['url_title'] = "View Details"
        
        try:
            response = requests.post(self.api_url, data=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Pushover notification sent: {title}")
                return True
            else:
                logger.error(f"Pushover API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Pushover notification: {e}")
            return False
    
    def send_task_completed(self, task_id: str, task_type: str, details: Dict[str, Any]) -> bool:
        """
        タスク完了通知
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            details: 詳細情報
            
        Returns:
            送信成功フラグ
        """
        title = f"✅ Task Completed: {task_type}"
        
        message_parts = [
            f"Task ID: {task_id}",
            f"Type: {task_type}",
            f"Status: Completed Successfully"
        ]
        
        if task_type == "pytest":
            if 'results' in details:
                results = details['results']
                message_parts.extend([
                    f"Tests: {results.get('total_tests', 0)}",
                    f"Passed: {results.get('passed', 0)}",
                    f"Failed: {results.get('failed', 0)}"
                ])
        
        elif task_type == "extract_character":
            if 'results' in details:
                results = details['results']
                message_parts.extend([
                    f"Images: {results.get('total_images', 0)}",
                    f"Success: {results.get('successful', 0)}",
                    f"Rate: {results.get('success_rate', 0):.1f}%"
                ])
        
        message = "\n".join(message_parts)
        
        return self.send_notification(title, message, priority=0)
    
    def send_task_failed(self, task_id: str, task_type: str, error: str, retry_count: int) -> bool:
        """
        タスク失敗通知
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            error: エラー内容
            retry_count: リトライ回数
            
        Returns:
            送信成功フラグ
        """
        title = f"❌ Task Failed: {task_type}"
        
        message = f"""Task ID: {task_id}
Type: {task_type}
Status: Failed after {retry_count} retries
Error: {error[:200]}

⚠️ Manual intervention required
Consider switching to PlanMode for resolution"""
        
        return self.send_notification(title, message, priority=1)
    
    def send_queue_status(self, queue_length: int, current_task: Optional[str] = None) -> bool:
        """
        キュー状態通知
        
        Args:
            queue_length: キュー長
            current_task: 現在実行中のタスク
            
        Returns:
            送信成功フラグ
        """
        title = "📊 Queue Status Update"
        
        message_parts = [
            f"Queue Length: {queue_length} tasks",
            f"Current Task: {current_task or 'None'}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        message = "\n".join(message_parts)
        
        return self.send_notification(title, message, priority=-1)


class PlanModeEscalator:
    """PlanModeエスカレーター"""
    
    def __init__(self, workspace_path: str):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
        """
        self.workspace = Path(workspace_path)
        self.escalation_file = self.workspace / "planmode_escalation.json"
        logger.info("PlanModeEscalator initialized")
    
    def create_escalation(self,
                         task_id: str,
                         task_type: str,
                         error: str,
                         retry_count: int,
                         command: str) -> Dict[str, Any]:
        """
        PlanModeエスカレーション作成
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            error: エラー内容
            retry_count: リトライ回数
            command: 実行コマンド
            
        Returns:
            エスカレーション情報
        """
        escalation = {
            'task_id': task_id,
            'task_type': task_type,
            'error': error,
            'retry_count': retry_count,
            'command': command,
            'created_at': datetime.now().isoformat(),
            'status': 'pending_review',
            'suggested_actions': self._suggest_actions(task_type, error)
        }
        
        # ファイル保存
        try:
            with open(self.escalation_file, 'w') as f:
                json.dump(escalation, f, indent=2)
            logger.info(f"PlanMode escalation created for {task_id}")
        except Exception as e:
            logger.error(f"Failed to create escalation file: {e}")
        
        return escalation
    
    def _suggest_actions(self, task_type: str, error: str) -> List[str]:
        """
        推奨アクション生成
        
        Args:
            task_type: タスクタイプ
            error: エラー内容
            
        Returns:
            推奨アクションリスト
        """
        suggestions = []
        
        # 共通アクション
        suggestions.append("Review error logs for root cause")
        suggestions.append("Check system resources (memory, disk space)")
        
        # タスクタイプ別アクション
        if task_type == "pytest":
            suggestions.extend([
                "Verify test dependencies are installed",
                "Check for import errors in test files",
                "Run individual test files to isolate issue",
                "Consider reducing test scope"
            ])
        
        elif task_type == "extract_character":
            suggestions.extend([
                "Verify input images exist and are valid",
                "Check CUDA/GPU availability",
                "Reduce batch size or max_files",
                "Try different quality_method parameter",
                "Ensure SAM and YOLO models are loaded"
            ])
        
        # エラー内容別アクション
        if "memory" in error.lower() or "oom" in error.lower():
            suggestions.append("Increase available memory or reduce batch size")
        
        if "cuda" in error.lower() or "gpu" in error.lower():
            suggestions.append("Switch to CPU mode or verify GPU drivers")
        
        if "timeout" in error.lower():
            suggestions.append("Increase timeout limit or split into smaller tasks")
        
        if "permission" in error.lower():
            suggestions.append("Check file/directory permissions")
        
        return suggestions
    
    def get_escalation_prompt(self, escalation: Dict[str, Any]) -> str:
        """
        PlanMode用プロンプト生成
        
        Args:
            escalation: エスカレーション情報
            
        Returns:
            PlanModeプロンプト
        """
        prompt = f"""🚨 タスク失敗のためPlanModeレビューが必要です

## エラー情報
- **タスクID**: {escalation['task_id']}
- **タスクタイプ**: {escalation['task_type']}
- **リトライ回数**: {escalation['retry_count']}
- **作成時刻**: {escalation['created_at']}

## エラー詳細
```
{escalation['error']}
```

## 実行コマンド
```bash
{escalation['command']}
```

## 推奨アクション
"""
        
        for i, action in enumerate(escalation['suggested_actions'], 1):
            prompt += f"{i}. {action}\n"
        
        prompt += """
## 次のステップ
1. エラーの根本原因を分析
2. 推奨アクションから適切な対応を選択
3. 必要に応じてコード修正やパラメータ調整
4. タスクを再実行

このエラーをどのように解決すべきか検討してください。
"""
        
        return prompt


class NotificationBridge:
    """通知ブリッジ統合クラス"""
    
    def __init__(self, workspace_path: str, tracker_id: str = "QUAL-044"):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
            tracker_id: トラッカーID
        """
        self.workspace_path = workspace_path
        self.tracker_id = tracker_id
        
        # コンポーネント初期化
        self.pushover = PushoverNotifier()
        self.escalator = PlanModeEscalator(workspace_path)
        
        logger.info(f"NotificationBridge initialized for {tracker_id}")
    
    def handle_task_completion(self,
                              task_id: str,
                              task_type: str,
                              results: Dict[str, Any]) -> None:
        """
        タスク完了ハンドリング
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            results: 実行結果
        """
        logger.info(f"Handling task completion: {task_id}")
        
        # Pushover通知送信
        self.pushover.send_task_completed(
            task_id=task_id,
            task_type=task_type,
            details={'results': results}
        )
        
        # 成功ログ記録
        self._log_event({
            'event': 'task_completed',
            'task_id': task_id,
            'task_type': task_type,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    def handle_task_failure(self,
                          task_id: str,
                          task_type: str,
                          error: str,
                          retry_count: int,
                          command: str) -> Dict[str, Any]:
        """
        タスク失敗ハンドリング
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            error: エラー内容
            retry_count: リトライ回数
            command: 実行コマンド
            
        Returns:
            エスカレーション情報
        """
        logger.info(f"Handling task failure: {task_id}")
        
        # Pushover通知送信
        self.pushover.send_task_failed(
            task_id=task_id,
            task_type=task_type,
            error=error,
            retry_count=retry_count
        )
        
        # PlanModeエスカレーション作成
        escalation = self.escalator.create_escalation(
            task_id=task_id,
            task_type=task_type,
            error=error,
            retry_count=retry_count,
            command=command
        )
        
        # 失敗ログ記録
        self._log_event({
            'event': 'task_failed',
            'task_id': task_id,
            'task_type': task_type,
            'error': error,
            'retry_count': retry_count,
            'escalation_created': True,
            'timestamp': datetime.now().isoformat()
        })
        
        return escalation
    
    def handle_queue_update(self, queue_status: Dict[str, Any]) -> None:
        """
        キュー状態更新ハンドリング
        
        Args:
            queue_status: キュー状態
        """
        queue_length = queue_status.get('queue_length', 0)
        current_task = None
        
        if 'current_task' in queue_status:
            current_task = queue_status['current_task'].get('task_id')
        
        # 定期的な状態通知（キューが空でない場合のみ）
        if queue_length > 0 or current_task:
            self.pushover.send_queue_status(queue_length, current_task)
    
    def _log_event(self, event_data: Dict[str, Any]) -> None:
        """
        イベントログ記録
        
        Args:
            event_data: イベントデータ
        """
        log_file = Path(self.workspace_path) / "logs" / "notification_events.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


def demonstrate_notification_bridge():
    """通知ブリッジデモンストレーション"""
    print("🔔 通知ブリッジシステムデモンストレーション")
    print("=" * 50)
    
    workspace = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-044"
    bridge = NotificationBridge(workspace, "QUAL-044")
    
    print("\n1️⃣ タスク完了通知")
    print("   - Pushover通知送信")
    print("   - 成功ログ記録")
    
    # デモ: タスク完了
    demo_results = {
        'total_images': 39,
        'successful': 34,
        'success_rate': 87.2
    }
    bridge.handle_task_completion(
        task_id="extract_20250830_164000",
        task_type="extract_character",
        results=demo_results
    )
    
    print("\n2️⃣ タスク失敗通知とPlanModeエスカレーション")
    print("   - Pushover通知送信")
    print("   - PlanModeエスカレーション作成")
    print("   - 推奨アクション生成")
    
    # デモ: タスク失敗
    escalation = bridge.handle_task_failure(
        task_id="pytest_20250830_164500",
        task_type="pytest",
        error="ImportError: No module named 'test_module'",
        retry_count=2,
        command="python -m pytest tests/"
    )
    
    # エスカレーションプロンプト表示
    prompt = bridge.escalator.get_escalation_prompt(escalation)
    print("\n📝 生成されたPlanModeプロンプト:")
    print("-" * 40)
    print(prompt[:500] + "...")  # 最初の500文字のみ表示
    
    print("\n✅ 通知ブリッジの特徴:")
    print("   1. Pushover通知（トークン効率的）")
    print("   2. PlanModeエスカレーション")
    print("   3. 推奨アクション自動生成")
    print("   4. イベントログ記録")
    print("   5. キュー状態通知")
    
    return True


def main():
    """CLI実行用メイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        # デモンストレーション実行
        demonstrate_notification_bridge()
    else:
        command = sys.argv[1]
        workspace = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-044"
        bridge = NotificationBridge(workspace)
        
        if command == "test-pushover":
            # Pushoverテスト送信
            success = bridge.pushover.send_notification(
                title="🧪 Test Notification",
                message="QUAL-044 Notification Bridge Test",
                priority=0
            )
            print(f"Pushover test: {'✅ Success' if success else '❌ Failed'}")
        
        elif command == "test-escalation":
            # エスカレーションテスト
            escalation = bridge.handle_task_failure(
                task_id="test_task_001",
                task_type="test",
                error="Test error for demonstration",
                retry_count=2,
                command="echo 'test'"
            )
            print(f"Escalation created: {json.dumps(escalation, indent=2)}")
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python notification_bridge.py [test-pushover|test-escalation]")


if __name__ == "__main__":
    main()