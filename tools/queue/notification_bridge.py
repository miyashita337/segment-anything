#!/usr/bin/env python3
"""
通知ブリッジシステム
QUAL-044: Pushover通知とPlanModeエスカレーション統合

タスク完了・失敗時の通知とエラー時のPlanMode連携
"""

import json
import os
import re
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging
import hashlib

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

        # INTG-089: 重複通知防止・優先度管理機能
        self.notification_history: Set[str] = set()  # 通知ハッシュ履歴
        self.last_notification_times: Dict[str, float] = {}  # 最後の通知時間

        # INTG-089 効率化: ハッシュ生成キャッシュ（メモリ最適化）
        self._hash_cache: Dict[str, str] = {}
        self._cache_max_size = 1000  # キャッシュサイズ上限

        # INTG-089 効率化: コンパイル済み正規表現（高速化）
        self._timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        self._percentage_pattern = re.compile(r'\b\d+\.\d+%\b')
        self._size_pattern = re.compile(r'\b\d+\.\d+ (MB|GB|KB)\b')
        self._whitespace_pattern = re.compile(r'\s+')

        # INTG-089 統計: パフォーマンス監視
        self._hash_cache_hits = 0
        self._hash_cache_misses = 0
        
        # INTG-089: 精密化された優先度制御
        self.burst_counters: Dict[str, int] = {}  # バーストカウンター  
        self.burst_windows: Dict[str, float] = {}   # バーストウィンドウ開始時刻
        self.adaptive_intervals: Dict[str, float] = {}  # アダプティブ間隔
        self.notification_priorities = {
            'critical': {
                'priority': 2, 
                'min_interval': 30,
                'burst_allowance': 3,  # バースト許可数
                'burst_window': 300,   # バーストウィンドウ（5分）
                'adaptive_factor': 1.5,  # アダプティブ係数
                'max_interval': 300     # 最大間隔（5分）
            },
            'high': {
                'priority': 1, 
                'min_interval': 300,
                'burst_allowance': 2,
                'burst_window': 600,   # 10分
                'adaptive_factor': 1.3,
                'max_interval': 900    # 15分
            },
            'normal': {
                'priority': 0, 
                'min_interval': 600,
                'burst_allowance': 1,
                'burst_window': 900,   # 15分
                'adaptive_factor': 1.2,
                'max_interval': 1800   # 30分
            },
            'low': {
                'priority': -1, 
                'min_interval': 1800,
                'burst_allowance': 1,
                'burst_window': 3600,  # 60分
                'adaptive_factor': 1.1,
                'max_interval': 7200   # 120分
            },
        }
        self.deduplication_window = 3600  # 1時間の重複チェックウィンドウ

        logger.info(f"NotificationBridge initialized for {tracker_id}")
        logger.info("INTG-089: Enhanced notification management enabled")
    
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

    # INTG-089: 拡張通知機能
    def _generate_notification_hash(self, title: str, message: str, notification_type: str) -> str:
        """
        通知の重複チェック用ハッシュ生成（高度効率化版）

        Args:
            title: 通知タイトル
            message: 通知メッセージ
            notification_type: 通知タイプ

        Returns:
            str: 通知ハッシュ
        """
        # 効率化1: キャッシュキー生成（入力文字列の事前チェック）
        cache_key = f"{title[:50]}:{message[:100]}:{notification_type}"

        # 効率化2: キャッシュチェック（同一内容の再計算を回避）
        if cache_key in self._hash_cache:
            self._hash_cache_hits += 1
            return self._hash_cache[cache_key]

        self._hash_cache_misses += 1

        # 効率化3: コンパイル済み正規表現で高速正規化
        normalized_title = self._whitespace_pattern.sub(' ', title.strip())
        normalized_message = self._whitespace_pattern.sub(' ', message.strip())

        # 効率化4: 変動値の標準化（コンパイル済みパターン使用）
        cleaned_message = self._timestamp_pattern.sub('[TS]', normalized_message)
        cleaned_message = self._percentage_pattern.sub('[PCT]', cleaned_message)
        cleaned_message = self._size_pattern.sub('[SZ]', cleaned_message)

        # 効率化5: 高速ハッシュ生成（短縮版で collision 確率を最小化）
        content = f"{normalized_title}:{cleaned_message}:{notification_type}"
        hash_result = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

        # 効率化6: キャッシュ管理（メモリリーク防止）
        if len(self._hash_cache) >= self._cache_max_size:
            # LRU方式でキャッシュクリア（最も古いエントリを削除）
            oldest_key = next(iter(self._hash_cache))
            del self._hash_cache[oldest_key]

        self._hash_cache[cache_key] = hash_result
        return hash_result

    def get_hash_cache_statistics(self) -> Dict[str, Any]:
        """
        ハッシュキャッシュ効率化統計取得

        Returns:
            dict: キャッシュ統計情報
        """
        total_requests = self._hash_cache_hits + self._hash_cache_misses
        hit_rate = (self._hash_cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self._hash_cache_hits,
            'cache_misses': self._hash_cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._hash_cache),
            'cache_max_size': self._cache_max_size,
            'memory_efficiency': f"{len(self._hash_cache)}/{self._cache_max_size} ({len(self._hash_cache)/self._cache_max_size*100:.1f}%)"
        }

    def _should_suppress_notification(self, notification_hash: str, priority_level: str) -> bool:
        """
        重複通知抑制判定（精密化版）

        Args:
            notification_hash: 通知ハッシュ
            priority_level: 優先度レベル

        Returns:
            bool: True=抑制する, False=送信する
        """
        current_time = time.time()

        # 重複チェック
        if notification_hash in self.notification_history:
            logger.info(f"Notification suppressed: duplicate detected ({notification_hash[:8]})")
            return True

        # 優先度設定取得
        priority_config = self.notification_priorities.get(priority_level, self.notification_priorities['normal'])
        
        # アダプティブ間隔計算
        effective_interval = self._calculate_adaptive_interval(priority_level, priority_config)
        
        # バースト制御チェック
        if self._is_burst_allowed(priority_level, priority_config, current_time):
            logger.info(f"Notification allowed: burst mode for {priority_level} priority")
            return False
        
        # 通常の間隔チェック
        if priority_level in self.last_notification_times:
            time_since_last = current_time - self.last_notification_times[priority_level]
            if time_since_last < effective_interval:
                logger.info(f"Notification suppressed: too frequent for {priority_level} priority ({time_since_last:.0f}s < {effective_interval:.0f}s)")
                return True

        return False
        
    def _calculate_adaptive_interval(self, priority_level: str, priority_config: Dict[str, Any]) -> float:
        """
        アダプティブ間隔計算
        
        Args:
            priority_level: 優先度レベル
            priority_config: 優先度設定
            
        Returns:
            float: 効果的な間隔（秒）
        """
        base_interval = priority_config['min_interval']
        adaptive_factor = priority_config['adaptive_factor']
        max_interval = priority_config['max_interval']
        
        # 初回または履歴なしの場合は基本間隔を使用
        if priority_level not in self.adaptive_intervals:
            self.adaptive_intervals[priority_level] = base_interval
            return base_interval
        
        # 前回の通知からの経過時間に基づくアダプティブ調整
        current_time = time.time()
        if priority_level in self.last_notification_times:
            time_since_last = current_time - self.last_notification_times[priority_level]
            
            # 長期間通知がない場合は間隔をリセット
            if time_since_last > (max_interval * 2):
                self.adaptive_intervals[priority_level] = base_interval
            # 頻繁な通知の場合は間隔を増加
            elif time_since_last < base_interval:
                current_adaptive = self.adaptive_intervals[priority_level]
                new_adaptive = min(current_adaptive * adaptive_factor, max_interval)
                self.adaptive_intervals[priority_level] = new_adaptive
            # 適切な間隔の場合は徐々に基本間隔に戻す
            else:
                current_adaptive = self.adaptive_intervals[priority_level]
                new_adaptive = max(current_adaptive * 0.9, base_interval)
                self.adaptive_intervals[priority_level] = new_adaptive
        
        return self.adaptive_intervals[priority_level]
    
    def _is_burst_allowed(self, priority_level: str, priority_config: Dict[str, Any], current_time: float) -> bool:
        """
        バースト許可判定
        
        Args:
            priority_level: 優先度レベル
            priority_config: 優先度設定
            current_time: 現在時刻
            
        Returns:
            bool: True=バースト許可, False=バースト不可
        """
        burst_allowance = priority_config['burst_allowance']
        burst_window = priority_config['burst_window']
        
        # バースト許可が1以下の場合はバースト機能無効
        if burst_allowance <= 1:
            return False
        
        # バーストウィンドウ初期化または期限切れチェック
        if (priority_level not in self.burst_windows or 
            current_time - self.burst_windows[priority_level] > burst_window):
            # 新しいバーストウィンドウを開始
            self.burst_windows[priority_level] = current_time
            self.burst_counters[priority_level] = 0
        
        # バーストカウンター確認
        current_burst_count = self.burst_counters.get(priority_level, 0)
        
        if current_burst_count < burst_allowance:
            # バースト許可範囲内
            self.burst_counters[priority_level] = current_burst_count + 1
            return True
        
        return False

    def send_enhanced_notification(self, title: str, message: str, notification_type: str = 'normal',
                                   priority_level: str = 'normal') -> bool:
        """
        拡張通知送信（重複防止・優先度管理付き）

        Args:
            title: 通知タイトル
            message: 通知メッセージ
            notification_type: 通知タイプ
            priority_level: 優先度レベル (critical/high/normal/low)

        Returns:
            bool: 送信成功フラグ
        """
        # 通知ハッシュ生成
        notification_hash = self._generate_notification_hash(title, message, notification_type)

        # 重複・間隔チェック
        if self._should_suppress_notification(notification_hash, priority_level):
            return False

        # 優先度設定
        priority_config = self.notification_priorities.get(priority_level, self.notification_priorities['normal'])
        pushover_priority = priority_config['priority']

        # 通知送信
        success = self.pushover.send_notification(title, message, priority=pushover_priority)

        if success:
            # 履歴更新
            current_time = time.time()
            self.notification_history.add(notification_hash)
            self.last_notification_times[priority_level] = current_time

            # 重複チェック履歴のクリーンアップ
            self._cleanup_notification_history()

            logger.info(f"Enhanced notification sent: {priority_level} priority, hash={notification_hash[:8]}")

        return success

    def send_anomaly_notification(self, anomaly_data: Dict[str, Any]) -> bool:
        """
        異常検知通知（INTG-089専用）

        Args:
            anomaly_data: 異常検知データ

        Returns:
            bool: 送信成功フラグ
        """
        anomaly_count = anomaly_data.get('anomaly_count', 0)
        anomalies = anomaly_data.get('anomalies', [])

        if anomaly_count == 0:
            return True  # 異常なしの場合は送信不要

        # 異常の重要度判定
        priority_level = 'normal'
        if any(anomaly['type'] == 'gpu' for anomaly in anomalies):
            priority_level = 'high'
        if any('critical' in anomaly['message'].lower() for anomaly in anomalies):
            priority_level = 'critical'

        # 通知メッセージ生成
        title = f"⚠️ System Anomaly Detected ({anomaly_count} issues)"

        message_parts = [f"Tracker: {self.tracker_id}", f"Anomalies: {anomaly_count}"]

        for anomaly in anomalies:
            anomaly_type = anomaly['type'].upper()
            anomaly_msg = anomaly['message'][:50] + "..." if len(anomaly['message']) > 50 else anomaly['message']
            message_parts.append(f"• {anomaly_type}: {anomaly_msg}")

        # システム統計追加
        stats = anomaly_data.get('system_stats', {})
        if stats:
            message_parts.append(f"\nSystem: CPU {stats.get('cpu_percent', 0):.1f}%, RAM {stats.get('memory_percent', 0):.1f}%")

        message = "\n".join(message_parts)

        # 拡張通知送信
        return self.send_enhanced_notification(title, message, 'anomaly', priority_level)

    def send_process_status_notification(self, process_status: Dict[str, Any], priority_level: str = 'normal') -> bool:
        """
        プロセス状態通知

        Args:
            process_status: プロセス状態データ
            priority_level: 優先度レベル

        Returns:
            bool: 送信成功フラグ
        """
        process_id = process_status.get('process_id')
        status = process_status.get('status', 'unknown')
        details = process_status.get('details', {})

        title = f"🔄 Process Status: {status}"

        message_parts = [
            f"Tracker: {self.tracker_id}",
            f"Process ID: {process_id}",
            f"Status: {status}"
        ]

        # 詳細情報追加
        if 'cpu_percent' in details:
            message_parts.append(f"CPU: {details['cpu_percent']:.1f}%")
        if 'memory_mb' in details:
            message_parts.append(f"Memory: {details['memory_mb']}MB")
        if 'runtime_hours' in details:
            message_parts.append(f"Runtime: {details['runtime_hours']:.1f}h")

        message = "\n".join(message_parts)

        return self.send_enhanced_notification(title, message, 'process_status', priority_level)

    def _cleanup_notification_history(self) -> None:
        """通知履歴クリーンアップ（メモリ使用量制御）"""
        # 履歴サイズ制限（最大1000件）
        if len(self.notification_history) > 1000:
            # 古い履歴を削除（簡易実装：セットをクリアして再構築）
            self.notification_history.clear()
            logger.info("Notification history cleaned up")

    def get_notification_stats(self) -> Dict[str, Any]:
        """
        通知統計取得

        Returns:
            Dict[str, Any]: 通知統計
        """
        return {
            'total_notifications_sent': len(self.notification_history),
            'last_notification_times': dict(self.last_notification_times),
            'priority_levels': list(self.notification_priorities.keys()),
            'deduplication_window_hours': self.deduplication_window / 3600
        }


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