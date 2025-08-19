#!/usr/bin/env python3
"""
トラッカーID標準化マッピング生成スクリプト
136件のトラッカーIDを機能別カテゴリ型に変換
"""

import json
import re
from datetime import datetime
from collections import defaultdict

def generate_tracker_mapping():
    """機能別カテゴリ型への完全マッピング生成"""
    
    # 既存トラッカーIDの機能分類（フェーズ1.1の結果）
    tracker_categories = {
        "quality_management": [
            "QTY-016", "QTY-017", "QTY-018", "QTY-019", "QTY-020", "QTY-021", "QTY-022",
            "QTY-003", "QTY-004", "QTY-005", "QTY-006", "QTY-007", "QTY-008",
            "QTY-009", "QTY-010", "QTY-011", "QTY-012", "QTY-013", "QTY-014",
            "QTY-015", "QTY-002", "QTY-001"
        ],
        "phase_work": [
            "PHS-001", "PHS-002", "PHS-003", "PHS-004", "PHS-005", "PHS-006",
            "PHS-007", "PHS-008", "PHS-010", "PHS-011", "PHS-012", "PHS-013",
            "PHS-014", "PHS-015", "PHS-016", "PHS-017", "PHS-018", "PHS-019",
            "PHS-020", "PHS-021", "PHS-022", "PHS-023", "PHS-024", "PHS-025",
            "PHS-026", "PHS-027", "PHS-028", "PHS-029", "PHS-030", "PHS-031",
            "PHS-032", "PHS-033", "PHS-034", "PHS-035", "PHS-036", "PHS-037",
            "PHS-008-RESOURCE"
        ],
        "optimization": [
            "OPTETETETETETETET-010", "OPTETETETETETET-010", "OPTETETETETET-010", "OPTETETETET-010", "OPTETETET-010", "OPTETET-010", "OPTET-010",
            "OPT-010", "OPTET-011", "OPTET-012", "OPTET-013", "OPTET-014", "OPTET-015", "OPTET-016",
            "OPT-017", "OPT-018", "OPT-019", "OPT-020", "OPT-021", "OPT-022",
            "OPT-023", "OPT-024", "OPT-024-1", "OPT-024-2", "OPT-024-3", 
            "OPT-028", "OPT-029", "OPT-030", "OPT-031", "OPT-032", "OPT-033",
            "OPT-034", "OPT-035", "OPT-036", "OPTETETETETETETETETET-010", "OPTETETETETETETETET-010"
        ],
        "integration": [
            "INTETETETETETETETETET-010", "INTETETETETETETETET-010", "INTETETETETETETET-010", "INTETETETETETET-010"
        ],
        "testing": [
            "TETETETETETETETETET-010", "TETETETETETETETET-010", "TETETETETETETET-010", "TETETETETETET-010", "TETETETETET-010", "TETETETET-010", "TETETET-010",
            "TETET-010", "TET-010", "TET-011", "TET-012", "TET-013", "TET-014", "TET-015",
            "TET-016", "TESTETETETETETETETETET-010", "TETETETETETETETETETET-010"
        ],
        "maintenance": [
            "MNT-003", "MNT-004", "MNT-005", "MNT-002", "MNT-001"
        ],
        "documentation": [
            "DOC-001", "DOC-002"
        ],
        "baseline_calculation": [
            "BSL-001"
        ],
        "infrastructure": [
            "INF-002", "INF-001"
        ]
    }
    
    # 新カテゴリコード（ユーザー決定版）
    category_codes = {
        "quality_management": "QTY",
        "phase_work": "PHS", 
        "optimization": "OPT",
        "integration": "INT",
        "testing": "TET",
        "maintenance": "MNT",
        "documentation": "DOC",
        "baseline_calculation": "BSL",
        "infrastructure": "INF"
    }
    
    # マッピング生成
    mapping = {}
    counters = defaultdict(int)
    
    print("🔄 トラッカーID標準化マッピング生成中...")
    
    for category, tracker_list in tracker_categories.items():
        code = category_codes[category]
        print(f"\n📂 {category} ({code}) - {len(tracker_list)}件:")
        
        for old_id in sorted(tracker_list):
            counters[code] += 1
            new_id = f"{code}-{counters[code]:03d}"
            mapping[old_id] = new_id
            print(f"   {old_id:20} → {new_id}")
    
    # 統計情報
    total_trackers = sum(len(trackers) for trackers in tracker_categories.values())
    print(f"\n📊 マッピング生成完了:")
    print(f"   総トラッカー数: {total_trackers}")
    print(f"   カテゴリ数: {len(category_codes)}")
    
    # 結果をJSONで保存
    result = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "total_trackers": total_trackers,
            "categories": len(category_codes),
            "strategy": "functional_categorization",
            "user_approved_codes": category_codes
        },
        "category_mapping": category_codes,
        "tracker_categories": tracker_categories,
        "id_mapping": mapping,
        "reverse_mapping": {v: k for k, v in mapping.items()},
        "category_counts": {code: counters[code] for code in category_codes.values()}
    }
    
    output_file = "tools/trackers/QUAL_032_tracker_standardization/tracker_id_mapping.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ マッピングファイル保存: {output_file}")
    
    # 検証用サマリー出力
    print(f"\n📋 カテゴリ別統計:")
    for code, count in sorted(counters.items()):
        category_name = [k for k, v in category_codes.items() if v == code][0]
        print(f"   {code}: {count:3d}件 - {category_name}")
    
    return result

if __name__ == "__main__":
    mapping_result = generate_tracker_mapping()
    print(f"\n🎯 次ステップ: フェーズ2.2（安全性確保システム設計）")