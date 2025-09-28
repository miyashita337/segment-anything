#!/usr/bin/env python3
"""
矛盾指示検出システム - Claude暴走防止のための高度な矛盾分析

KIRO-012解決策: 矛盾する指示による暴走防止
- セマンティック矛盾検出
- 時系列矛盾分析
- 優先度競合検出
- 自動アラート機能
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import hashlib
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Instruction:
    """指示の構造化データ"""
    instruction_id: str
    tracker_id: str
    timestamp: datetime
    content: str
    category: str  # "requirement", "constraint", "action", "priority"
    priority: int  # 1(highest) to 5(lowest)
    source: str    # "user", "claude", "system"
    step_id: str
    metadata: Dict[str, Any]

@dataclass
class ContradictionPattern:
    """矛盾パターンの定義"""
    pattern_id: str
    pattern_type: str  # "semantic", "temporal", "priority", "logical"
    keywords_set_a: List[str]
    keywords_set_b: List[str]
    conflict_description: str
    severity: str  # "critical", "high", "medium", "low"
    auto_alert: bool

@dataclass
class ContradictionDetection:
    """矛盾検出結果"""
    detection_id: str
    tracker_id: str
    pattern_id: str
    instruction_a: str
    instruction_b: str
    contradiction_type: str
    confidence_score: float
    detected_at: datetime
    resolution_status: str  # "unresolved", "acknowledged", "resolved"
    alert_sent: bool

class ContradictionDetector:
    """
    矛盾指示検出システム
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "contradiction_detector.db")

        self.db_path = db_path
        self._init_database()
        self._load_contradiction_patterns()

        # アラートファイル保存ディレクトリ
        self.alerts_dir = Path(__file__).parent / "contradiction_alerts"
        self.alerts_dir.mkdir(exist_ok=True)

        logger.info(f"ContradictionDetector initialized with db: {self.db_path}")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS instructions (
                    instruction_id TEXT PRIMARY KEY,
                    tracker_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS contradiction_detections (
                    detection_id TEXT PRIMARY KEY,
                    tracker_id TEXT NOT NULL,
                    pattern_id TEXT NOT NULL,
                    instruction_a TEXT NOT NULL,
                    instruction_b TEXT NOT NULL,
                    contradiction_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolution_status TEXT NOT NULL DEFAULT 'unresolved',
                    alert_sent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_instructions_tracker_time
                ON instructions(tracker_id, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contradictions_tracker_status
                ON contradiction_detections(tracker_id, resolution_status)
            """)

    def _load_contradiction_patterns(self):
        """矛盾パターンの定義読み込み"""
        self.contradiction_patterns = {
            "add_remove_conflict": ContradictionPattern(
                pattern_id="add_remove_conflict",
                pattern_type="semantic",
                keywords_set_a=["追加", "作成", "新規", "インストール", "導入"],
                keywords_set_b=["削除", "除去", "アンインストール", "撤去", "廃止"],
                conflict_description="追加と削除の指示が同時に存在",
                severity="critical",
                auto_alert=True
            ),
            "enable_disable_conflict": ContradictionPattern(
                pattern_id="enable_disable_conflict",
                pattern_type="semantic",
                keywords_set_a=["有効", "オン", "起動", "開始", "アクティブ"],
                keywords_set_b=["無効", "オフ", "停止", "終了", "非アクティブ"],
                conflict_description="有効化と無効化の指示が競合",
                severity="high",
                auto_alert=True
            ),
            "priority_contradiction": ContradictionPattern(
                pattern_id="priority_contradiction",
                pattern_type="priority",
                keywords_set_a=["最優先", "緊急", "即座", "至急"],
                keywords_set_b=["後回し", "延期", "低優先", "保留"],
                conflict_description="優先度の競合指示",
                severity="medium",
                auto_alert=True
            ),
            "version_conflict": ContradictionPattern(
                pattern_id="version_conflict",
                pattern_type="semantic",
                keywords_set_a=["最新", "アップデート", "新しい", "アップグレード"],
                keywords_set_b=["古い", "ダウングレード", "前の", "旧バージョン"],
                conflict_description="バージョン管理の矛盾",
                severity="high",
                auto_alert=True
            ),
            "automation_manual_conflict": ContradictionPattern(
                pattern_id="automation_manual_conflict",
                pattern_type="logical",
                keywords_set_a=["自動", "バッチ", "一括", "自動実行"],
                keywords_set_b=["手動", "個別", "確認", "承認"],
                conflict_description="自動化と手動実行の指示が競合",
                severity="critical",
                auto_alert=True
            )
        }

    def record_instruction(self, tracker_id: str, content: str, category: str = "requirement",
                          priority: int = 3, source: str = "user", step_id: str = "unknown",
                          metadata: Dict[str, Any] = None) -> str:
        """
        指示を記録し、即座に矛盾検出を実行

        Returns:
            instruction_id
        """
        if metadata is None:
            metadata = {}

        instruction = Instruction(
            instruction_id=self._generate_id("inst"),
            tracker_id=tracker_id,
            timestamp=datetime.now(),
            content=content,
            category=category,
            priority=priority,
            source=source,
            step_id=step_id,
            metadata=metadata
        )

        # データベースに記録
        self._save_instruction(instruction)

        # 矛盾検出実行
        contradictions = self.detect_contradictions(tracker_id, instruction)

        # アラート送信
        for contradiction in contradictions:
            if contradiction.confidence_score > 0.7:  # 高信頼度の矛盾のみアラート
                self._send_contradiction_alert(contradiction)

        return instruction.instruction_id

    def detect_contradictions(self, tracker_id: str, new_instruction: Instruction) -> List[ContradictionDetection]:
        """新しい指示と既存指示の矛盾を検出"""
        contradictions = []

        # 過去24時間の指示を取得
        recent_instructions = self._get_recent_instructions(tracker_id, hours=24)

        for existing_instruction in recent_instructions:
            if existing_instruction.instruction_id == new_instruction.instruction_id:
                continue

            for pattern in self.contradiction_patterns.values():
                contradiction = self._check_pattern_contradiction(
                    new_instruction, existing_instruction, pattern
                )

                if contradiction:
                    contradictions.append(contradiction)
                    self._save_contradiction_detection(contradiction)

        return contradictions

    def _check_pattern_contradiction(self, inst_a: Instruction, inst_b: Instruction,
                                   pattern: ContradictionPattern) -> Optional[ContradictionDetection]:
        """特定パターンでの矛盾チェック"""
        content_a = inst_a.content.lower()
        content_b = inst_b.content.lower()

        # パターンAとBのキーワードが両方の指示に含まれているかチェック
        has_a_in_first = any(keyword in content_a for keyword in pattern.keywords_set_a)
        has_b_in_first = any(keyword in content_a for keyword in pattern.keywords_set_b)
        has_a_in_second = any(keyword in content_b for keyword in pattern.keywords_set_a)
        has_b_in_second = any(keyword in content_b for keyword in pattern.keywords_set_b)

        # 矛盾パターンを検出
        contradiction_found = False
        confidence = 0.0

        if (has_a_in_first and has_b_in_second) or (has_b_in_first and has_a_in_second):
            contradiction_found = True
            # 信頼度計算
            keyword_matches = sum([has_a_in_first, has_b_in_first, has_a_in_second, has_b_in_second])
            confidence = min(keyword_matches / 2.0, 1.0)

            # 時間的近接性による信頼度補正
            time_diff = abs((inst_a.timestamp - inst_b.timestamp).total_seconds())
            if time_diff < 300:  # 5分以内
                confidence += 0.2
            elif time_diff < 3600:  # 1時間以内
                confidence += 0.1

            # 優先度競合による信頼度補正
            if pattern.pattern_type == "priority" and abs(inst_a.priority - inst_b.priority) > 2:
                confidence += 0.2

            confidence = min(confidence, 1.0)

        if contradiction_found and confidence > 0.5:
            return ContradictionDetection(
                detection_id=self._generate_id("contra"),
                tracker_id=inst_a.tracker_id,
                pattern_id=pattern.pattern_id,
                instruction_a=inst_a.instruction_id,
                instruction_b=inst_b.instruction_id,
                contradiction_type=pattern.pattern_type,
                confidence_score=confidence,
                detected_at=datetime.now(),
                resolution_status="unresolved",
                alert_sent=False
            )

        return None

    def _send_contradiction_alert(self, contradiction: ContradictionDetection):
        """矛盾アラートを送信"""
        pattern = self.contradiction_patterns.get(contradiction.pattern_id)
        if not pattern or not pattern.auto_alert:
            return

        # 指示の詳細情報取得
        inst_a = self._get_instruction_by_id(contradiction.instruction_a)
        inst_b = self._get_instruction_by_id(contradiction.instruction_b)

        if not inst_a or not inst_b:
            return

        alert_data = {
            "detection_id": contradiction.detection_id,
            "tracker_id": contradiction.tracker_id,
            "alert_type": "contradiction_detected",
            "severity": pattern.severity,
            "detected_at": contradiction.detected_at.isoformat(),
            "pattern": {
                "id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "description": pattern.conflict_description
            },
            "contradicting_instructions": {
                "instruction_a": {
                    "id": inst_a.instruction_id,
                    "content": inst_a.content,
                    "timestamp": inst_a.timestamp.isoformat(),
                    "step_id": inst_a.step_id,
                    "source": inst_a.source
                },
                "instruction_b": {
                    "id": inst_b.instruction_id,
                    "content": inst_b.content,
                    "timestamp": inst_b.timestamp.isoformat(),
                    "step_id": inst_b.step_id,
                    "source": inst_b.source
                }
            },
            "confidence_score": contradiction.confidence_score,
            "recommended_actions": [
                "1. 矛盾する指示を確認してください",
                "2. 優先すべき指示を決定してください",
                "3. 不要な指示を取り消してください",
                "4. 矛盾解決後に承認してください"
            ]
        }

        # アラートファイル保存
        alert_file = self.alerts_dir / f"{contradiction.tracker_id}_contradiction_{contradiction.detection_id}.json"
        with open(alert_file, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, indent=2, ensure_ascii=False)

        # データベース更新
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE contradiction_detections
                SET alert_sent = 1
                WHERE detection_id = ?
            """, (contradiction.detection_id,))

        logger.warning(f"🚨 矛盾アラート送信: {contradiction.tracker_id} - {pattern.conflict_description}")
        logger.warning(f"   信頼度: {contradiction.confidence_score:.2f}")
        logger.warning(f"   アラートファイル: {alert_file}")

    def get_active_contradictions(self, tracker_id: str) -> List[Dict[str, Any]]:
        """未解決の矛盾一覧取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT detection_id, pattern_id, instruction_a, instruction_b,
                       contradiction_type, confidence_score, detected_at, alert_sent
                FROM contradiction_detections
                WHERE tracker_id = ? AND resolution_status = 'unresolved'
                ORDER BY confidence_score DESC, detected_at DESC
            """, (tracker_id,))

            contradictions = []
            for row in cursor.fetchall():
                detection_id, pattern_id, inst_a_id, inst_b_id, \
                contradiction_type, confidence, detected_at, alert_sent = row

                # 指示の詳細情報取得
                inst_a = self._get_instruction_by_id(inst_a_id)
                inst_b = self._get_instruction_by_id(inst_b_id)

                if inst_a and inst_b:
                    contradictions.append({
                        "detection_id": detection_id,
                        "pattern_id": pattern_id,
                        "contradiction_type": contradiction_type,
                        "confidence_score": confidence,
                        "detected_at": detected_at,
                        "alert_sent": bool(alert_sent),
                        "instruction_a": {
                            "id": inst_a.instruction_id,
                            "content": inst_a.content,
                            "timestamp": inst_a.timestamp.isoformat(),
                            "step_id": inst_a.step_id
                        },
                        "instruction_b": {
                            "id": inst_b.instruction_id,
                            "content": inst_b.content,
                            "timestamp": inst_b.timestamp.isoformat(),
                            "step_id": inst_b.step_id
                        }
                    })

        return contradictions

    def resolve_contradiction(self, detection_id: str, resolution: str = "resolved") -> bool:
        """矛盾解決マーク"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE contradiction_detections
                SET resolution_status = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE detection_id = ?
            """, (resolution, detection_id))

            return cursor.rowcount > 0

    def _get_recent_instructions(self, tracker_id: str, hours: int = 24) -> List[Instruction]:
        """最近の指示を取得"""
        time_threshold = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT instruction_id, tracker_id, timestamp, content, category,
                       priority, source, step_id, metadata
                FROM instructions
                WHERE tracker_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (tracker_id, time_threshold.isoformat()))

            instructions = []
            for row in cursor.fetchall():
                inst_id, tracker_id, timestamp, content, category, \
                priority, source, step_id, metadata_json = row

                instructions.append(Instruction(
                    instruction_id=inst_id,
                    tracker_id=tracker_id,
                    timestamp=datetime.fromisoformat(timestamp),
                    content=content,
                    category=category,
                    priority=priority,
                    source=source,
                    step_id=step_id,
                    metadata=json.loads(metadata_json)
                ))

        return instructions

    def _get_instruction_by_id(self, instruction_id: str) -> Optional[Instruction]:
        """指示IDから指示を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT instruction_id, tracker_id, timestamp, content, category,
                       priority, source, step_id, metadata
                FROM instructions
                WHERE instruction_id = ?
            """, (instruction_id,))

            row = cursor.fetchone()
            if not row:
                return None

            inst_id, tracker_id, timestamp, content, category, \
            priority, source, step_id, metadata_json = row

            return Instruction(
                instruction_id=inst_id,
                tracker_id=tracker_id,
                timestamp=datetime.fromisoformat(timestamp),
                content=content,
                category=category,
                priority=priority,
                source=source,
                step_id=step_id,
                metadata=json.loads(metadata_json)
            )

    def _save_instruction(self, instruction: Instruction):
        """指示をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO instructions
                (instruction_id, tracker_id, timestamp, content, category,
                 priority, source, step_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instruction.instruction_id,
                instruction.tracker_id,
                instruction.timestamp.isoformat(),
                instruction.content,
                instruction.category,
                instruction.priority,
                instruction.source,
                instruction.step_id,
                json.dumps(instruction.metadata, ensure_ascii=False)
            ))

    def _save_contradiction_detection(self, contradiction: ContradictionDetection):
        """矛盾検出をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO contradiction_detections
                (detection_id, tracker_id, pattern_id, instruction_a, instruction_b,
                 contradiction_type, confidence_score, detected_at, resolution_status, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                contradiction.detection_id,
                contradiction.tracker_id,
                contradiction.pattern_id,
                contradiction.instruction_a,
                contradiction.instruction_b,
                contradiction.contradiction_type,
                contradiction.confidence_score,
                contradiction.detected_at.isoformat(),
                contradiction.resolution_status,
                1 if contradiction.alert_sent else 0
            ))

    def _generate_id(self, prefix: str) -> str:
        """ユニークID生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

def main():
    """CLI エントリーポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="矛盾指示検出システム")
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')

    # 指示記録コマンド
    record_parser = subparsers.add_parser('record', help='指示を記録')
    record_parser.add_argument('tracker_id', help='トラッカーID')
    record_parser.add_argument('content', help='指示内容')
    record_parser.add_argument('--category', default='requirement', help='カテゴリ')
    record_parser.add_argument('--priority', type=int, default=3, help='優先度 (1-5)')
    record_parser.add_argument('--source', default='user', help='ソース')
    record_parser.add_argument('--step-id', default='unknown', help='ステップID')

    # 矛盾確認コマンド
    check_parser = subparsers.add_parser('check', help='矛盾確認')
    check_parser.add_argument('tracker_id', help='トラッカーID')

    # 矛盾解決コマンド
    resolve_parser = subparsers.add_parser('resolve', help='矛盾解決')
    resolve_parser.add_argument('detection_id', help='検出ID')
    resolve_parser.add_argument('--resolution', default='resolved', help='解決状況')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    detector = ContradictionDetector()

    if args.command == 'record':
        instruction_id = detector.record_instruction(
            tracker_id=args.tracker_id,
            content=args.content,
            category=args.category,
            priority=args.priority,
            source=args.source,
            step_id=args.step_id
        )
        print(f"✅ 指示を記録しました: {instruction_id}")

    elif args.command == 'check':
        contradictions = detector.get_active_contradictions(args.tracker_id)
        if contradictions:
            print(f"🚨 {len(contradictions)} 件の矛盾が検出されています:")
            for i, contradiction in enumerate(contradictions, 1):
                print(f"\n{i}. 検出ID: {contradiction['detection_id']}")
                print(f"   パターン: {contradiction['pattern_id']}")
                print(f"   信頼度: {contradiction['confidence_score']:.2f}")
                print(f"   指示A: {contradiction['instruction_a']['content'][:50]}...")
                print(f"   指示B: {contradiction['instruction_b']['content'][:50]}...")
        else:
            print(f"✅ {args.tracker_id} に矛盾は検出されていません")

    elif args.command == 'resolve':
        success = detector.resolve_contradiction(args.detection_id, args.resolution)
        if success:
            print(f"✅ 矛盾が解決済みとしてマークされました")
        else:
            print(f"❌ 矛盾の解決に失敗しました")

if __name__ == "__main__":
    main()