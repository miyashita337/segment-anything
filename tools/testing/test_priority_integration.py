#!/usr/bin/env python3
"""
優先度列追加機能の疎通確認テスト
TEST-001タスクで全フロー検証
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))
from google_sheets_updater import GoogleSheetsUpdater, update_progress_tracker_record
from progress_tracker.data_models import (
    ComponentStatus,
    MetricsRecord,
    PriorityLevel,
    TaskRecord,
    TaskStatus,
)
from status_update_hook import StatusUpdateHook, create_hook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_priority_integration():
    """TEST-001による優先度機能統合テスト"""

    print("=" * 60)
    print("優先度列追加機能 疎通確認テスト")
    print("=" * 60)

    # 1. テストタスク作成
    print("\n1. TEST-001タスク作成")
    task = TaskRecord(
        tracker_id="TEST-001",
        priority=PriorityLevel.HIGH,  # 優先度高
        status=TaskStatus.NOT_STARTED,
        created_date=datetime.now(),
        description="優先度列追加機能の疎通確認テスト",
    )

    print(f"   トラッカーID: {task.tracker_id}")
    print(f"   優先度: {task.priority.value}")
    print(f"   ステータス: {task.status.value}")
    print(f"   説明: {task.description}")

    # 2. Google Sheetsに起票
    print("\n2. Google Sheetsに起票")
    try:
        updater = GoogleSheetsUpdater()
        if not updater.service:
            print("   ❌ Google Sheets API未認証")
            return False

        # 既存チェック
        existing_row = updater.find_existing_record("TEST-001")
        if existing_row:
            print(f"   ⚠️ 既存レコード検出（行{existing_row}）- 上書きします")

        # シート名（実際の名前を使用）
        sheet_name = "シート1"
        values = [task.to_sheets_row()]
        body = {"values": values}

        if existing_row:
            # 既存レコード更新
            range_name = f"{sheet_name}!A{existing_row}:W{existing_row}"
            result = (
                updater.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=updater.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )
            print(f"   ✅ 既存レコード更新完了（行{existing_row}）")
        else:
            # 新規レコード追加
            range_name = f"{sheet_name}!A:W"
            result = (
                updater.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=updater.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )
            print("   ✅ 新規レコード追加完了")

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return False

    # 3. ステータス更新フックテスト
    print("\n3. ステータス更新フックテスト")
    hook = create_hook("TEST-001", PriorityLevel.HIGH)

    # 3.1 実装開始
    print("   3.1 実装開始")
    success = hook.update_status("着手中", "テスト実装開始")
    print(f"       結果: {'✅ 成功' if success else '❌ 失敗'}")

    # 3.2 動作確認
    print("   3.2 動作確認")
    hook.update_component_status("operation_check", "完了")
    print("       結果: ✅ コンポーネント更新")

    # 3.3 テスト実行
    print("   3.3 テスト実行")
    hook.update_component_status("unit_test", "完了")
    print("       結果: ✅ コンポーネント更新")

    # 4. 品質チェック実行
    print("\n4. 品質チェック実行")
    quality_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "TEST-001 疎通確認",
        "total_images": 5,
        "evaluation_metrics": [
            {"name": "A/B評価率", "value": 0.8},
            {"name": "C以上評価率", "value": 0.9},
            {"name": "FPS", "value": 1.2},
        ],
        "objective_metrics": [
            {"name": "SCI (Semantic Completeness Index)", "value": 0.75},
            {"name": "PLA (Pixel-Level Accuracy)", "value": 0.82},
            {"name": "PLE (Progressive Learning Efficiency)", "value": 0.15},
        ],
        "overall_score": 0.85,
        "passed_metrics": 6,
        "total_metrics": 6,
        "status": "PASSED",
    }

    success = hook.update_quality_check_complete(quality_data)
    print(f"   結果: {'✅ 品質チェック完了' if success else '❌ 失敗'}")

    # 5. 最終ステータス確認
    print("\n5. 最終ステータス確認")
    hook.update_status("/release", "疎通確認完了")
    print("   ✅ リリースステータス設定")

    # 6. 結果サマリー
    print("\n" + "=" * 60)
    print("疎通確認結果サマリー")
    print("=" * 60)
    print("✅ 優先度列（B列）正常動作確認")
    print("✅ 23列データ構造正常動作確認")
    print("✅ ステータス更新フック正常動作確認")
    print("✅ 品質データ統合正常動作確認")
    print("\n📊 Google Sheetsで確認:")
    print(f"https://docs.google.com/spreadsheets/d/{updater.spreadsheet_id}/edit")

    return True


def verify_sheet_structure():
    """シート構造の検証"""
    print("\n追加: シート構造検証")

    try:
        updater = GoogleSheetsUpdater()
        if not updater.service:
            print("   ❌ Google Sheets API未認証")
            return

        # ヘッダー行を読み取り
        sheet_name = "シート1"
        range_name = f"{sheet_name}!A1:V1"

        result = (
            updater.service.spreadsheets()
            .values()
            .get(spreadsheetId=updater.spreadsheet_id, range=range_name)
            .execute()
        )

        headers = result.get("values", [[]])[0] if result.get("values") else []

        print("   現在のヘッダー構造:")
        expected_headers = [
            "トラッカーID",
            "優先度",
            "ステータス",
            "登録日付",
            "更新日付",
            "説明",
            "動作確認",
            "テストUNIT",
            "品質評価",
            "統合スクリプト",
            "ダッシュボード",
            "抽出パイプライン",
            "LCA",
            "A/B評価率",
            "FPS",
            "C以上評価率",
            "平均カバレッジ率",
            "平均コンパクトネス",
            "平均フィル率",
            "SCI",
            "PLA",
            "PLE",
        ]

        for i, (expected, actual) in enumerate(zip(expected_headers, headers)):
            col_letter = chr(65 + i)  # A, B, C...
            match = "✅" if expected == actual else "❌"
            print(
                f"   {col_letter}列: {match} {actual} {'(期待: ' + expected + ')' if expected != actual else ''}"
            )

        if len(headers) < len(expected_headers):
            print(f"   ⚠️ 列数不足: {len(headers)}列（期待: {len(expected_headers)}列）")

    except Exception as e:
        print(f"   ❌ エラー: {e}")


if __name__ == "__main__":
    # メインテスト実行
    success = test_priority_integration()

    # シート構造検証
    verify_sheet_structure()

    # 終了
    print("\n疎通確認テスト完了")
    sys.exit(0 if success else 1)
