#!/usr/bin/env python3
"""
エラー検出・自動停止システム - Claude暴走防止のための安全システム

KIRO-012解決策: Claude暴走問題の根本解決
- エラーループ検出
- 矛盾指示検出
- 自動停止機能
- ユーザー承認待機
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorPattern:
    """エラーパターンの定義"""
    pattern_id: str
    error_type: str  # "loop", "contradiction", "permission", "execution"
    pattern_regex: str
    severity: str    # "critical", "warning", "info"
    max_occurrences: int
    time_window_minutes: int
    auto_stop: bool
    description: str

@dataclass
class ErrorOccurrence:
    """エラー発生記録"""
    occurrence_id: str
    pattern_id: str
    tracker_id: str
    timestamp: datetime
    error_message: str
    context: Dict[str, Any]
    step_id: str
    user_action_required: bool

@dataclass
class AutoStopTrigger:
    """自動停止トリガーの記録"""
    trigger_id: str
    tracker_id: str
    trigger_type: str  # "error_loop", "contradiction", "manual"
    triggered_at: datetime
    pattern_ids: List[str]
    stop_reason: str
    approval_required: bool
    resolution_status: str  # "pending", "approved", "rejected", "timeout"

class ErrorDetectionSystem:
    """
    Claude暴走防止のためのエラー検出・自動停止システム
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "error_detection.db")

        self.db_path = db_path
        self._init_database()
        self._load_error_patterns()

        # 停止フラグファイル
        self.stop_flags_dir = Path(__file__).parent / "stop_flags"
        self.stop_flags_dir.mkdir(exist_ok=True)

        logger.info(f"ErrorDetectionSystem initialized with db: {self.db_path}")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_occurrences (
                    occurrence_id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL,
                    tracker_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    context TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    user_action_required INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS auto_stop_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    tracker_id TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    pattern_ids TEXT NOT NULL,
                    stop_reason TEXT NOT NULL,
                    approval_required INTEGER NOT NULL,
                    resolution_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_tracker_time
                ON error_occurrences(tracker_id, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stop_tracker_status
                ON auto_stop_triggers(tracker_id, resolution_status)
            """)

    def _load_error_patterns(self):
        """エラーパターンの定義読み込み"""
        self.error_patterns = {
            "claude_error_loop": ErrorPattern(
                pattern_id="claude_error_loop",
                error_type="loop",
                pattern_regex=r"(同じ.*エラー|繰り返し.*失敗|再度.*試行)",
                severity="critical",
                max_occurrences=3,
                time_window_minutes=10,
                auto_stop=True,
                description="Claude が同じエラーを繰り返している"
            ),
            "contradiction_detected": ErrorPattern(
                pattern_id="contradiction_detected",
                error_type="contradiction",
                pattern_regex=r"(矛盾.*指示|相反.*要件|対立.*条件)",
                severity="critical",
                max_occurrences=2,
                time_window_minutes=5,
                auto_stop=True,
                description="矛盾する指示が検出された"
            ),
            "permission_violation": ErrorPattern(
                pattern_id="permission_violation",
                error_type="permission",
                pattern_regex=r"(許可.*なし|権限.*不足|アクセス.*拒否)",
                severity="warning",
                max_occurrences=5,
                time_window_minutes=15,
                auto_stop=True,
                description="権限関連エラーが多発している"
            ),
            "excessive_automation": ErrorPattern(
                pattern_id="excessive_automation",
                error_type="execution",
                pattern_regex=r"(自動.*実行|連続.*処理|一括.*変更)",
                severity="warning",
                max_occurrences=10,
                time_window_minutes=5,
                auto_stop=True,
                description="過度な自動化が検出された"
            ),
            "lost_in_middle": ErrorPattern(
                pattern_id="lost_in_middle",
                error_type="context",
                pattern_regex=r"(コンテキスト.*不足|情報.*過多|指示.*不明確)",
                severity="critical",
                max_occurrences=2,
                time_window_minutes=3,
                auto_stop=True,
                description="Lost-in-the-Middle問題が発生している"
            )
        }

    def check_for_errors(self, tracker_id: str, step_id: str,
                        message: str, context: Dict[str, Any] = None) -> List[str]:
        """
        エラーパターンをチェックし、必要に応じて自動停止を実行

        Returns:
            発生したパターンIDのリスト
        """
        if context is None:
            context = {}

        detected_patterns = []

        # 各パターンをチェック
        for pattern in self.error_patterns.values():
            if self._match_pattern(message, pattern):
                detected_patterns.append(pattern.pattern_id)

                # エラー発生を記録
                occurrence = ErrorOccurrence(
                    occurrence_id=self._generate_id("error"),
                    pattern_id=pattern.pattern_id,
                    tracker_id=tracker_id,
                    timestamp=datetime.now(),
                    error_message=message,
                    context=context,
                    step_id=step_id,
                    user_action_required=pattern.auto_stop
                )

                self._record_error_occurrence(occurrence)

                # 自動停止判定
                if pattern.auto_stop:
                    should_stop = self._should_trigger_auto_stop(tracker_id, pattern)
                    if should_stop:
                        self._trigger_auto_stop(tracker_id, [pattern.pattern_id],
                                              pattern.description)

        return detected_patterns

    def _match_pattern(self, message: str, pattern: ErrorPattern) -> bool:
        """パターンマッチング"""
        import re
        return bool(re.search(pattern.pattern_regex, message, re.IGNORECASE))

    def _should_trigger_auto_stop(self, tracker_id: str, pattern: ErrorPattern) -> bool:
        """自動停止をトリガーすべきかチェック"""
        # 時間窓内のエラー発生回数をチェック
        time_threshold = datetime.now() - timedelta(minutes=pattern.time_window_minutes)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM error_occurrences
                WHERE tracker_id = ? AND pattern_id = ? AND timestamp > ?
            """, (tracker_id, pattern.pattern_id, time_threshold.isoformat()))

            count = cursor.fetchone()[0]

        return count >= pattern.max_occurrences

    def _trigger_auto_stop(self, tracker_id: str, pattern_ids: List[str], reason: str):
        """自動停止を実行"""
        trigger = AutoStopTrigger(
            trigger_id=self._generate_id("stop"),
            tracker_id=tracker_id,
            trigger_type="error_loop",
            triggered_at=datetime.now(),
            pattern_ids=pattern_ids,
            stop_reason=reason,
            approval_required=True,
            resolution_status="pending"
        )

        # データベースに記録
        self._record_auto_stop_trigger(trigger)

        # 停止フラグファイル作成
        self._create_stop_flag(tracker_id, trigger)

        logger.critical(f"🚨 AUTO-STOP TRIGGERED: {tracker_id} - {reason}")

        # 承認要求通知
        self._notify_approval_required(tracker_id, trigger)

    def _create_stop_flag(self, tracker_id: str, trigger: AutoStopTrigger):
        """停止フラグファイル作成"""
        stop_flag_path = self.stop_flags_dir / f"{tracker_id}.stop"

        stop_flag_data = {
            "trigger_id": trigger.trigger_id,
            "tracker_id": tracker_id,
            "triggered_at": trigger.triggered_at.isoformat(),
            "reason": trigger.stop_reason,
            "pattern_ids": trigger.pattern_ids,
            "approval_required": trigger.approval_required,
            "status": "STOPPED",
            "message": "Claude自動実行が安全のため停止されました。承認が必要です。"
        }

        with open(stop_flag_path, 'w', encoding='utf-8') as f:
            json.dump(stop_flag_data, f, indent=2, ensure_ascii=False)

        logger.info(f"停止フラグファイル作成: {stop_flag_path}")

    def is_stopped(self, tracker_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """トラッカーが停止状態かチェック"""
        stop_flag_path = self.stop_flags_dir / f"{tracker_id}.stop"

        if not stop_flag_path.exists():
            return False, None

        try:
            with open(stop_flag_path, 'r', encoding='utf-8') as f:
                stop_data = json.load(f)
            return True, stop_data
        except Exception as e:
            logger.error(f"停止フラグファイル読み取りエラー: {e}")
            return False, None

    def approve_continuation(self, tracker_id: str, approver: str = "user") -> bool:
        """継続承認処理"""
        is_stopped, stop_data = self.is_stopped(tracker_id)

        if not is_stopped:
            logger.warning(f"トラッカー {tracker_id} は停止していません")
            return False

        # 承認記録をデータベースに保存
        trigger_id = stop_data.get("trigger_id")
        if trigger_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE auto_stop_triggers
                    SET resolution_status = 'approved',
                        resolved_at = CURRENT_TIMESTAMP,
                        resolver = ?
                    WHERE trigger_id = ?
                """, (approver, trigger_id))

        # 停止フラグファイル削除
        stop_flag_path = self.stop_flags_dir / f"{tracker_id}.stop"
        stop_flag_path.unlink()

        logger.info(f"✅ トラッカー {tracker_id} の継続が承認されました (承認者: {approver})")
        return True

    def _record_error_occurrence(self, occurrence: ErrorOccurrence):
        """エラー発生をデータベースに記録"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO error_occurrences
                (occurrence_id, pattern_id, tracker_id, timestamp,
                 error_message, context, step_id, user_action_required)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                occurrence.occurrence_id,
                occurrence.pattern_id,
                occurrence.tracker_id,
                occurrence.timestamp.isoformat(),
                occurrence.error_message,
                json.dumps(occurrence.context, ensure_ascii=False),
                occurrence.step_id,
                1 if occurrence.user_action_required else 0
            ))

    def _record_auto_stop_trigger(self, trigger: AutoStopTrigger):
        """自動停止トリガーをデータベースに記録"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO auto_stop_triggers
                (trigger_id, tracker_id, trigger_type, triggered_at,
                 pattern_ids, stop_reason, approval_required, resolution_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trigger.trigger_id,
                trigger.tracker_id,
                trigger.trigger_type,
                trigger.triggered_at.isoformat(),
                json.dumps(trigger.pattern_ids),
                trigger.stop_reason,
                1 if trigger.approval_required else 0,
                trigger.resolution_status
            ))

    def _notify_approval_required(self, tracker_id: str, trigger: AutoStopTrigger):
        """承認要求通知"""
        approval_dir = Path(__file__).parent / "approval_requests"
        approval_dir.mkdir(exist_ok=True)

        approval_file = approval_dir / f"{tracker_id}_approval_request.json"

        approval_data = {
            "trigger_id": trigger.trigger_id,
            "tracker_id": tracker_id,
            "triggered_at": trigger.triggered_at.isoformat(),
            "reason": trigger.stop_reason,
            "pattern_ids": trigger.pattern_ids,
            "message": "Claude暴走防止システムが自動停止を実行しました。",
            "instructions": [
                "1. 停止理由を確認してください",
                "2. 根本原因を分析してください",
                "3. 修正が完了したら承認してください",
                f"4. 承認コマンド: python tools/safety/error_detection_system.py approve {tracker_id}"
            ],
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }

        with open(approval_file, 'w', encoding='utf-8') as f:
            json.dump(approval_data, f, indent=2, ensure_ascii=False)

        logger.info(f"承認要求ファイル作成: {approval_file}")

    def get_error_statistics(self, tracker_id: str = None,
                           hours: int = 24) -> Dict[str, Any]:
        """エラー統計情報取得"""
        time_threshold = datetime.now() - timedelta(hours=hours)

        where_clause = "WHERE timestamp > ?"
        params = [time_threshold.isoformat()]

        if tracker_id:
            where_clause += " AND tracker_id = ?"
            params.append(tracker_id)

        with sqlite3.connect(self.db_path) as conn:
            # パターン別集計
            cursor = conn.execute(f"""
                SELECT pattern_id, COUNT(*) as count
                FROM error_occurrences
                {where_clause}
                GROUP BY pattern_id
                ORDER BY count DESC
            """, params)

            pattern_stats = dict(cursor.fetchall())

            # 停止トリガー情報
            cursor = conn.execute(f"""
                SELECT trigger_type, resolution_status, COUNT(*) as count
                FROM auto_stop_triggers
                WHERE triggered_at > ?
                GROUP BY trigger_type, resolution_status
            """, [time_threshold.isoformat()])

            stop_stats = cursor.fetchall()

        return {
            "time_window_hours": hours,
            "tracker_id": tracker_id,
            "pattern_statistics": pattern_stats,
            "stop_trigger_statistics": stop_stats,
            "total_errors": sum(pattern_stats.values()),
            "generated_at": datetime.now().isoformat()
        }

    def _generate_id(self, prefix: str) -> str:
        """ユニークID生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

def main():
    """CLI エントリーポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="エラー検出・自動停止システム")
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')

    # 承認コマンド
    approve_parser = subparsers.add_parser('approve', help='継続承認')
    approve_parser.add_argument('tracker_id', help='トラッカーID')
    approve_parser.add_argument('--approver', default='user', help='承認者名')

    # 状態確認コマンド
    status_parser = subparsers.add_parser('status', help='停止状態確認')
    status_parser.add_argument('tracker_id', help='トラッカーID')

    # 統計コマンド
    stats_parser = subparsers.add_parser('stats', help='エラー統計')
    stats_parser.add_argument('--tracker-id', help='特定トラッカーの統計')
    stats_parser.add_argument('--hours', type=int, default=24, help='集計時間（時間）')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    system = ErrorDetectionSystem()

    if args.command == 'approve':
        success = system.approve_continuation(args.tracker_id, args.approver)
        if success:
            print(f"✅ {args.tracker_id} の継続が承認されました")
        else:
            print(f"❌ {args.tracker_id} の承認に失敗しました")

    elif args.command == 'status':
        is_stopped, stop_data = system.is_stopped(args.tracker_id)
        if is_stopped:
            print(f"🚨 {args.tracker_id} は停止中です")
            print(f"理由: {stop_data.get('reason', '不明')}")
            print(f"停止時刻: {stop_data.get('triggered_at', '不明')}")
        else:
            print(f"✅ {args.tracker_id} は正常稼働中です")

    elif args.command == 'stats':
        stats = system.get_error_statistics(args.tracker_id, args.hours)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()