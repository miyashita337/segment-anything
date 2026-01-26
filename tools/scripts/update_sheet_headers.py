#!/usr/bin/env python3
"""
Google Sheetsヘッダー更新スクリプト
21列形式から22列形式（優先度列追加）へ移行
"""

import logging
import sys
from pathlib import Path

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))
from google_sheets_updater import GoogleSheetsUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_sheet_headers():
    """シートヘッダーを22列形式に更新"""

    print("Google Sheetsヘッダー更新")
    print("=" * 60)

    try:
        updater = GoogleSheetsUpdater()
        if not updater.service:
            print("❌ Google Sheets API未認証")
            return False

        sheet_name = "シート1"

        # 新しいヘッダー（22列）
        new_headers = [
            "トラッカーID",  # A列
            "優先度",  # B列（新規追加）
            "ステータス",  # C列（旧B列）
            "登録日付",  # D列（旧C列）
            "更新日付",  # E列（旧D列）
            "概要",  # F列（旧E列）
            "動作確認",  # G列（旧F列）
            "テストUNIT",  # H列（旧G列）
            "品質評価",  # I列（旧H列）
            "統合実行スクリプト",  # J列（旧I列）
            "ダッシュボード生成",  # K列（旧J列）
            "抽出パイプライン",  # L列（旧K列）
            "LCA",  # M列（旧L列）
            "A/B評価率",  # N列（旧M列）
            "FPS",  # O列（旧N列）
            "C以上評価率",  # P列（旧O列）
            "平均カバレッジ率",  # Q列（旧P列）
            "平均コンパクトネス",  # R列（旧Q列）
            "平均フィル率",  # S列（旧R列）
            "SCI",  # T列（旧S列）
            "PLA",  # U列（旧T列）
            "PLE",  # V列（旧U列）
        ]

        # 1. ヘッダー行更新
        print("\n1. ヘッダー行更新")
        range_name = f"{sheet_name}!A1:V1"
        body = {"values": [new_headers]}

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
        print("   ✅ ヘッダー更新完了")

        # 2. 既存データの移行（B列に優先度挿入）
        print("\n2. 既存データの移行")

        # 全データ読み取り
        read_range = f"{sheet_name}!A2:U"
        result = (
            updater.service.spreadsheets()
            .values()
            .get(spreadsheetId=updater.spreadsheet_id, range=read_range)
            .execute()
        )

        existing_rows = result.get("values", [])
        print(f"   既存レコード数: {len(existing_rows)}")

        if existing_rows:
            # 各行にB列（優先度）を挿入
            migrated_rows = []
            for row in existing_rows:
                # 21列を確保（不足分は空文字）
                row = row + [""] * (21 - len(row))

                # 新しい行を構築（B列に優先度挿入）
                new_row = [
                    row[0],  # A: トラッカーID
                    "優先度中",  # B: 優先度（デフォルト）
                    row[1],  # C: ステータス
                    row[2],  # D: 登録日付
                    row[3],  # E: 更新日付
                    row[4],  # F: 概要
                    row[5],  # G: 動作確認
                    row[6],  # H: テストUNIT
                    row[7],  # I: 品質評価
                    row[8],  # J: 統合実行スクリプト
                    row[9],  # K: ダッシュボード生成
                    row[10],  # L: 抽出パイプライン
                    row[11],  # M: LCA
                    row[12],  # N: A/B評価率
                    row[13],  # O: FPS
                    row[14],  # P: C以上評価率
                    row[15],  # Q: 平均カバレッジ率
                    row[16],  # R: 平均コンパクトネス
                    row[17],  # S: 平均フィル率
                    row[18],  # T: SCI
                    row[19],  # U: PLA
                    row[20],  # V: PLE
                ]
                migrated_rows.append(new_row)

            # データ書き戻し
            write_range = f"{sheet_name}!A2:V{len(migrated_rows) + 1}"
            body = {"values": migrated_rows}

            result = (
                updater.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=updater.spreadsheet_id,
                    range=write_range,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )
            print(f"   ✅ {len(migrated_rows)}レコードの移行完了")
        else:
            print("   ℹ️ 既存データなし")

        # 3. 優先度列のドロップダウン設定
        print("\n3. 優先度列（B列）のドロップダウン設定")

        # データ検証ルール作成
        validation_rule = {
            "condition": {
                "type": "ONE_OF_LIST",
                "values": [
                    {"userEnteredValue": "優先度最高"},
                    {"userEnteredValue": "優先度高"},
                    {"userEnteredValue": "優先度中"},
                    {"userEnteredValue": "優先度低"},
                ],
            },
            "showCustomUi": True,
        }

        # B列全体に適用
        requests = [
            {
                "setDataValidation": {
                    "range": {
                        "sheetId": 0,  # 最初のシート
                        "startColumnIndex": 1,  # B列
                        "endColumnIndex": 2,  # B列のみ
                        "startRowIndex": 1,  # 2行目以降
                    },
                    "rule": validation_rule,
                }
            }
        ]

        body = {"requests": requests}

        try:
            result = (
                updater.service.spreadsheets()
                .batchUpdate(spreadsheetId=updater.spreadsheet_id, body=body)
                .execute()
            )
            print("   ✅ ドロップダウン設定完了")
        except Exception as e:
            print(f"   ⚠️ ドロップダウン設定エラー（手動設定推奨）: {e}")

        print("\n" + "=" * 60)
        print("✅ シート構造更新完了")
        print("📊 Google Sheetsで確認:")
        print(f"https://docs.google.com/spreadsheets/d/{updater.spreadsheet_id}/edit")

        return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


if __name__ == "__main__":
    success = update_sheet_headers()
    sys.exit(0 if success else 1)
