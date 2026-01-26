#!/usr/bin/env python3
"""
トラッカーID機能分類分析スクリプト
137件のトラッカーIDを機能別に詳細分類
"""

import json
import re
from collections import defaultdict
from datetime import datetime


def analyze_tracker_functionality():
    """トラッカーIDを機能別に詳細分類"""

    # 機能分類ルール定義
    functional_categories = {
        "quality_management": {
            "patterns": [r"QI-\d+", r"QCC-\d+", r"QCA-\d+", r"QC-.*"],
            "description": "品質管理・改善システム",
            "trackers": [],
        },
        "phase_work": {
            "patterns": [r"PH\d+-\d+", r"P1-A\d+", r"PHASE-.*"],
            "description": "フェーズ別開発作業",
            "trackers": [],
        },
        "optimization": {
            "patterns": [r"P1-\d+", r"P1-B\d+", r"CLAUDE-OPT-\d+", r".*OPT.*"],
            "description": "最適化・改善",
            "trackers": [],
        },
        "integration": {
            "patterns": [r"CI-.*", r"INTEGRATE-.*", r"MERGE-\d+", r"UNIFY-\d+"],
            "description": "統合・CI/CD",
            "trackers": [],
        },
        "testing": {
            "patterns": [r"T-\d+", r"TEST-\d+", r"BAT-\d+"],
            "description": "テスト・検証",
            "trackers": [],
        },
        "maintenance": {
            "patterns": [r"TDR-\d+", r"METRICS-.*", r"CLEANUP-.*", r"CRON-.*"],
            "description": "保守・メンテナンス",
            "trackers": [],
        },
        "documentation": {
            "patterns": [r"DOC-\d+", r".*DOC.*"],
            "description": "ドキュメント作成",
            "trackers": [],
        },
        "baseline_calculation": {
            "patterns": [r"BASELINE-.*", r"RECALC-.*"],
            "description": "ベースライン計算",
            "trackers": [],
        },
        "infrastructure": {
            "patterns": [r"UNTRACKED-.*", r"COMPOSITE-.*"],
            "description": "インフラ・基盤",
            "trackers": [],
        },
    }

    # 実際の137件トラッカーIDリスト（Google Sheetsから取得したデータベース）
    all_tracker_ids = [
        # Phase Work系
        "PHS-007",
        "PHS-005",
        "PHS-001",
        "PHS-002",
        "PHS-003",
        "PHS-004",
        "PHS-006",
        "PHS-013",
        "PHS-014",
        "PHS-015",
        "PHS-016",
        "PHS-017",
        "PHS-018",
        "PHS-019",
        "PHS-012",
        "PHS-021",
        "PHS-022",
        "PHS-023",
        "PHS-024",
        "PHS-025",
        "PHS-026",
        "PHS-027",
        "PHS-028",
        "PHS-029",
        "PHS-030",
        "PHS-031",
        "PHS-032",
        "PHS-033",
        "PHS-034",
        "PHS-035",
        "PHS-036",
        "PHS-037",
        "PHS-011",
        "PHS-010",
        "PHS-008",
        "PHS-020",
        "PHS-008-RESOURCE",
        # Quality Management系
        "QTY-016",
        "QTY-017",
        "QTY-018",
        "QTY-019",
        "QTY-020",
        "QTY-021",
        "QTY-022",
        "QTY-003",
        "QTY-004",
        "QTY-005",
        "QTY-006",
        "QTY-007",
        "QTY-008",
        "QTY-009",
        "QTY-010",
        "QTY-011",
        "QTY-012",
        "QTY-013",
        "QTY-014",
        "QTY-015",
        "QCC-FIX-001",
        "QCC-FIX-002",
        "QCC-FIX-003",
        "QCC-STAGE2-001",
        "QCC-STAGE3-001",
        "QCC-STAGE4-001",
        "QCC-SAFETY-001",
        "QCC-STATS-001",
        "QCC-MONITOR-001",
        "QCC-DASH-001",
        "QTY-002",
        "QTY-001",
        # Optimization系
        "OPTETETETETETETET-010",
        "OPTETETETETETET-010",
        "OPTETETETETET-010",
        "OPTETETETET-010",
        "OPTETETET-010",
        "OPTETET-010",
        "OPTET-010",
        "OPT-010",
        "OPTET-011",
        "OPTET-012",
        "OPTET-013",
        "OPTET-014",
        "OPTET-015",
        "OPTET-016",
        "OPT-017",
        "OPT-018",
        "OPT-019",
        "OPT-020",
        "OPT-021",
        "OPT-022",
        "OPT-023",
        "OPT-024",
        "OPT-024-1",
        "OPT-024-2",
        "OPT-024-3",
        "OPT-028",
        "OPT-029",
        "OPT-030",
        "OPT-031",
        "OPT-032",
        "OPT-033",
        "OPT-034",
        "OPT-035",
        "OPT-036",
        "OPTETETETETETETETETET-010",
        "OPTETETETETETETETET-010",
        # Integration系
        "INTETETETETETETETETET-010",
        "INTETETETETETETETET-010",
        "INTETETETETETETET-010",
        "INTETETETETETET-010",
        # Testing系
        "TETETETETETETETETET-010",
        "TETETETETETETETET-010",
        "TETETETETETETET-010",
        "TETETETETETET-010",
        "TETETETETET-010",
        "TETETETET-010",
        "TETETET-010",
        "TETET-010",
        "TET-010",
        "TET-011",
        "TET-012",
        "TET-013",
        "TET-014",
        "TET-015",
        "TET-016",
        "TESTETETETETETETETETET-010",
        "TETETETETETETETETETET-010",
        # Maintenance系
        "MNT-003",
        "MNT-004",
        "MNT-005",
        "MNT-002",
        "MNT-001",
        # Documentation系
        "DOC-001",
        "DOC-002",
        # Baseline Calculation系
        "BSL-001",
        # Infrastructure系
        "INF-002",
        "INF-001",
    ]

    # 分類実行
    unclassified = []

    for tracker_id in all_tracker_ids:
        classified = False

        for category, config in functional_categories.items():
            for pattern in config["patterns"]:
                if re.match(pattern, tracker_id):
                    config["trackers"].append({"id": tracker_id, "matched_pattern": pattern})
                    classified = True
                    break
            if classified:
                break

        if not classified:
            unclassified.append(tracker_id)

    # 結果統計
    total_trackers = len(all_tracker_ids)
    classified_count = total_trackers - len(unclassified)

    print(f"\n📊 フェーズ1.1完了 - 機能分類結果統計:")
    print(f"総トラッカー数: {total_trackers}")
    print(f"分類済み: {classified_count}")
    print(f"未分類: {len(unclassified)}")
    print(f"分類率: {classified_count/total_trackers*100:.1f}%")

    if unclassified:
        print(f"\n❌ 未分類トラッカーID: {', '.join(unclassified)}")

    print(f"\n📋 機能別分類結果:")
    for category, config in functional_categories.items():
        count = len(config["trackers"])
        if count > 0:
            print(f"  ✅ {category}: {count}件 - {config['description']}")
            sample_ids = [t["id"] for t in config["trackers"][:3]]
            print(f"     例: {', '.join(sample_ids)}{'...' if count > 3 else ''}")

    # JSON出力
    result = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_count": total_trackers,
        "classified_count": classified_count,
        "unclassified_count": len(unclassified),
        "classification_rate": round(classified_count / total_trackers * 100, 1),
        "unclassified_trackers": unclassified,
        "functional_categories": functional_categories,
        "summary": {
            category: {
                "count": len(config["trackers"]),
                "description": config["description"],
                "tracker_ids": [t["id"] for t in config["trackers"]],
            }
            for category, config in functional_categories.items()
            if len(config["trackers"]) > 0
        },
    }

    return result


if __name__ == "__main__":
    result = analyze_tracker_functionality()

    # 次フェーズへの提案生成
    print(f"\n🎯 フェーズ1.2への移行準備完了")
    print(f"   次ステップ: ユーザーとの新フォーマット設計相談")

    # 提案する統一フォーマット戦略
    strategies = {
        "A": "完全統一フォーマット（例: SEG-00001, SEG-00002...）",
        "B": "機能別カテゴリ型（例: QUA-001, OPTETETETETETETETETET-010, INTETETETETETETETETET-010...）",
        "C": "ハイブリッド型（既存優先+新規統一）",
    }

    print(f"\n💡 提案する戦略オプション:")
    for key, strategy in strategies.items():
        print(f"   戦略{key}: {strategy}")
