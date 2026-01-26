#!/usr/bin/env python3
"""
トラッカーID機能別マッピング生成スクリプト
全137件のトラッカーIDを QUAL/OPTM/TEST/INTG に分類

ユーザー指定:
- QUAL: 品質管理
- OPTM: 最適化  
- TEST: テスト
- INTG: 統合
"""

import json
import re
from pathlib import Path


def load_tracker_analysis():
    """分析済みトラッカーデータ読み込み"""
    analysis_file = Path("/tmp/tracker_analysis.json")
    if analysis_file.exists():
        with open(analysis_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def classify_tracker_to_function(tracker_id, description=""):
    """
    トラッカーIDを機能別カテゴリに分類

    ユーザー指定の4カテゴリ:
    - QUAL: 品質管理
    - OPTM: 最適化
    - TEST: テスト
    - INTG: 統合
    """
    tracker_id = tracker_id.upper()
    description = description.lower()

    # 1. QCCパターン → QUAL (品質管理)
    if re.match(r"^QCC-", tracker_id):
        return "QUAL"

    # 2. QIパターン → QUAL (品質改善)
    if re.match(r"^QI-", tracker_id):
        return "QUAL"

    # 3. 最適化パターン → OPTM
    if re.match(r"^(CLAUDE-OPT|OPTM)", tracker_id):
        return "OPTM"
    if any(keyword in description for keyword in ["最適化", "高速化", "optimize", "performance"]):
        return "OPTM"

    # 4. テストパターン → TEST
    if re.match(r"^(T-|TEST-)", tracker_id):
        return "TEST"
    if any(keyword in description for keyword in ["テスト", "test", "検証", "verify"]):
        return "TEST"

    # 5. 統合パターン → INTG
    if re.match(r"^(CI-|INTEGRATE-)", tracker_id):
        return "INTG"
    if any(keyword in description for keyword in ["統合", "integration", "ci", "cd", "pipeline"]):
        return "INTG"

    # 6. フェーズ作業 → デフォルトでINTG（統合プロジェクト）
    if re.match(r"^PH\d+-", tracker_id):
        return "INTG"

    # 7. P1系統の分類
    if re.match(r"^P1-", tracker_id):
        # 品質関連のP1 → QUAL
        if any(keyword in description for keyword in ["品質", "quality", "評価"]):
            return "QUAL"
        # それ以外のP1 → INTG（一般的な統合作業）
        return "INTG"

    # 8. その他の特定パターン
    special_patterns = {
        "MERGE-": "INTG",  # マージ作業
        "DOC-": "INTG",  # ドキュメント（プロジェクト統合の一環）
        "BAT-": "OPTM",  # バッチ処理最適化
        "UNIFY-": "INTG",  # 統一化作業
        "CLEANUP-": "OPTM",  # クリーンアップ（最適化）
        "MAINT-": "OPTM",  # メンテナンス（最適化）
        "METRICS-": "QUAL",  # メトリクス（品質管理）
        "BASELINE-": "QUAL",  # ベースライン（品質管理）
        "COMPOSITE-": "INTG",  # 複合（統合）
        "CRON-": "OPTM",  # Cron設定（最適化）
        "UNTRACKED-": "OPTM",  # 未追跡ファイル（最適化）
    }

    for pattern, category in special_patterns.items():
        if tracker_id.startswith(pattern):
            return category

    # 9. デフォルト: INTGに分類（統合プロジェクトとして扱う）
    return "INTG"


def generate_function_mapping():
    """全137件のトラッカーIDの機能別マッピング生成"""

    # 分析データ読み込み
    analysis_data = load_tracker_analysis()
    if not analysis_data:
        print("❌ 分析データが見つかりません")
        return None

    mapping = {"QUAL": [], "OPTM": [], "TEST": [], "INTG": []}  # 品質管理  # 最適化  # テスト  # 統合

    total_count = 0

    # 各カテゴリのトラッカーを処理
    for category_name, category_data in analysis_data["categories"].items():
        for tracker in category_data["trackers"]:
            tracker_id = tracker["id"]
            description = tracker.get("description", "")

            # 機能分類実行
            function_category = classify_tracker_to_function(tracker_id, description)

            # マッピング追加
            mapping[function_category].append(
                {
                    "original_id": tracker_id,
                    "new_id": None,  # 後で連番付与
                    "description": description,
                    "original_category": category_name,
                }
            )

            total_count += 1

    # 連番付与
    for func_code, trackers in mapping.items():
        for i, tracker in enumerate(trackers, 1):
            tracker["new_id"] = f"{func_code}-{i:03d}"

    # 統計情報
    stats = {
        "total_trackers": total_count,
        "QUAL_count": len(mapping["QUAL"]),
        "OPTM_count": len(mapping["OPTM"]),
        "TEST_count": len(mapping["TEST"]),
        "INTG_count": len(mapping["INTG"]),
    }

    result = {
        "mapping": mapping,
        "statistics": stats,
        "generation_timestamp": "2025-08-17T17:30:00",
    }

    return result


def save_mapping_to_file(mapping_data, output_file="/tmp/tracker_function_mapping.json"):
    """マッピングデータをファイルに保存"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    print(f"✅ マッピングデータ保存: {output_file}")


def print_mapping_summary(mapping_data):
    """マッピング結果のサマリー表示"""
    stats = mapping_data["statistics"]
    mapping = mapping_data["mapping"]

    print("\n📊 トラッカーID機能別マッピング結果")
    print("=" * 50)
    print(f"総トラッカー数: {stats['total_trackers']}件")
    print()

    for func_code in ["QUAL", "OPTM", "TEST", "INTG"]:
        count = stats[f"{func_code}_count"]
        print(f"🎯 {func_code} (件数: {count})")

        # 上位5件を表示
        for i, tracker in enumerate(mapping[func_code][:5]):
            print(f"   {tracker['original_id']:15} → {tracker['new_id']}")

        if count > 5:
            print(f"   ... 他{count-5}件")
        print()


if __name__ == "__main__":
    print("🔄 トラッカーID機能別マッピング生成開始...")

    # マッピング生成
    mapping_data = generate_function_mapping()

    if mapping_data:
        # ファイル保存
        save_mapping_to_file(mapping_data)

        # サマリー表示
        print_mapping_summary(mapping_data)

        print("✅ 機能別マッピング生成完了！")
    else:
        print("❌ マッピング生成失敗")
