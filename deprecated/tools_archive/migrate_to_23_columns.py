#!/usr/bin/env python3
"""
Google Sheets 22列→23列移行スクリプト
G列に「詳細」列追加、H-W列データシフト
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.google_sheets_updater import GoogleSheetsUpdater

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ColumnMigration23:
    """23列移行ツール"""

    def __init__(self):
        """初期化"""
        self.updater = GoogleSheetsUpdater()

    def migrate_to_23_columns(self) -> bool:
        """
        22列→23列移行実行

        Returns:
            bool: 移行成功/失敗
        """
        try:
            logger.info("Google Sheets 22列→23列移行開始")

            # 現在のヘッダー更新
            success_header = self._update_headers()
            if not success_header:
                logger.error("ヘッダー更新失敗")
                return False

            # 既存データ移行
            success_data = self._migrate_existing_data()
            if not success_data:
                logger.error("データ移行失敗")
                return False

            logger.info("✅ 22列→23列移行完了")
            return True

        except Exception as e:
            logger.error(f"移行エラー: {e}")
            return False

    def _update_headers(self) -> bool:
        """ヘッダー行更新"""
        try:
            # 新しいヘッダー（23列）
            headers = [
                "トラッカーID",  # A
                "優先度",  # B
                "ステータス",  # C
                "登録日付",  # D
                "更新日付",  # E
                "概要",  # F
                "詳細",  # G ← 新規追加
                "動作確認",  # H
                "テストUNIT",  # I
                "品質評価",  # J
                "統合実行スクリプト",  # K
                "ダッシュボード生成",  # L
                "抽出パイプライン",  # M
                "LCA",  # N
                "A/B評価率",  # O
                "FPS",  # P
                "C以上評価率",  # Q
                "平均カバレッジ率",  # R
                "平均コンパクトネス",  # S
                "平均フィル率",  # T
                "SCI",  # U
                "PLA",  # V
                "PLE",  # W
            ]

            # ヘッダー行更新
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A1:W1"

            update_body = {"values": [headers]}

            result = (
                self.updater.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=self.updater.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=update_body,
                )
                .execute()
            )

            logger.info("✅ ヘッダー行更新完了（23列）")
            return True

        except Exception as e:
            logger.error(f"ヘッダー更新エラー: {e}")
            return False

    def _migrate_existing_data(self) -> bool:
        """既存データ移行"""
        try:
            # 現在のデータ取得（A2:V行まで）
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A2:V"

            result = (
                self.updater.service.spreadsheets()
                .values()
                .get(spreadsheetId=self.updater.spreadsheet_id, range=range_name)
                .execute()
            )

            existing_data = result.get("values", [])
            if not existing_data:
                logger.info("移行対象データなし")
                return True

            logger.info(f"移行対象: {len(existing_data)}行")

            # 各行を23列形式に変換
            migrated_rows = []
            for i, row in enumerate(existing_data, start=2):
                # 22列に補完
                while len(row) < 22:
                    row.append("")

                # 23列形式に変換（G列に空文字追加、H-W列をシフト）
                new_row = [
                    row[0],  # A: トラッカーID
                    row[1],  # B: 優先度
                    row[2],  # C: ステータス
                    row[3],  # D: 登録日付
                    row[4],  # E: 更新日付
                    row[5],  # F: 概要
                    "",  # G: 詳細（新規追加・空白）
                    row[6],  # H: 動作確認（旧G列）
                    row[7],  # I: テストUNIT（旧H列）
                    row[8],  # J: 品質評価（旧I列）
                    row[9],  # K: 統合実行スクリプト（旧J列）
                    row[10],  # L: ダッシュボード生成（旧K列）
                    row[11],  # M: 抽出パイプライン（旧L列）
                    row[12],  # N: LCA（旧M列）
                    row[13],  # O: A/B評価率（旧N列）
                    row[14],  # P: FPS（旧O列）
                    row[15],  # Q: C以上評価率（旧P列）
                    row[16],  # R: 平均カバレッジ率（旧Q列）
                    row[17],  # S: 平均コンパクトネス（旧R列）
                    row[18],  # T: 平均フィル率（旧S列）
                    row[19],  # U: SCI（旧T列）
                    row[20],  # V: PLA（旧U列）
                    row[21],  # W: PLE（旧V列）
                ]

                migrated_rows.append(new_row)

                if (i - 1) % 10 == 0:
                    logger.info(f"進捗: {i-1}/{len(existing_data)}行変換完了")

            # 変換データを一括更新
            write_range = f"{sheet_name}!A2:W{len(migrated_rows) + 1}"
            update_body = {"values": migrated_rows}

            result = (
                self.updater.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=self.updater.spreadsheet_id,
                    range=write_range,
                    valueInputOption="RAW",
                    body=update_body,
                )
                .execute()
            )

            logger.info(f"✅ データ移行完了: {len(migrated_rows)}行")
            return True

        except Exception as e:
            logger.error(f"データ移行エラー: {e}")
            return False

    def test_23_column_operation(self) -> bool:
        """23列動作テスト"""
        try:
            logger.info("23列動作テスト開始")

            # 全データ取得テスト
            all_data = self.updater.get_all_sheet_data()
            if not all_data:
                logger.error("データ取得テスト失敗")
                return False

            logger.info(f"✅ データ取得テスト成功: {len(all_data)}行")

            # サンプル行確認
            if len(all_data) > 0:
                sample_row = all_data[0]
                logger.info(f"サンプル行列数: {len(sample_row)}")

                if len(sample_row) >= 23:
                    logger.info("✅ 23列構造確認")
                    logger.info(f"  G列（詳細）: '{sample_row[6]}'")
                else:
                    logger.warning(f"⚠️ 列数不足: {len(sample_row)}/23")

            return True

        except Exception as e:
            logger.error(f"動作テストエラー: {e}")
            return False


def main():
    """メイン実行"""
    logger.info("Google Sheets 22列→23列移行ツール")

    migrator = ColumnMigration23()

    # 移行実行
    success = migrator.migrate_to_23_columns()

    if success:
        # 動作テスト
        test_success = migrator.test_23_column_operation()

        if test_success:
            logger.info("✅ 22列→23列移行・動作確認完了")
            return 0
        else:
            logger.error("❌ 動作確認失敗")
            return 1
    else:
        logger.error("❌ 移行失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
