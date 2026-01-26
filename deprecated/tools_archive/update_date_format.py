#!/usr/bin/env python3
"""
Google Sheets既存データの日付フォーマット更新スクリプト
yyyy-mm-dd → yyyy-mm-dd hh:mm:ss
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


class DateFormatUpdater:
    """日付フォーマット更新ツール"""

    def __init__(self):
        """初期化"""
        self.updater = GoogleSheetsUpdater()

    def update_all_dates(self) -> bool:
        """
        全データの日付フォーマットを更新

        Returns:
            bool: 更新成功/失敗
        """
        try:
            logger.info("既存データの日付フォーマット更新開始")

            # 全データを取得
            all_data = self.updater.get_all_sheet_data()
            if not all_data:
                logger.error("データ取得に失敗")
                return False

            logger.info(f"取得データ: {len(all_data)}行")

            # 更新対象データを特定
            update_count = 0
            skipped_count = 0

            for row_num, row_data in enumerate(all_data, start=2):  # ヘッダー行は1行目
                # 列数が不十分な場合はスキップ
                if len(row_data) < 5:
                    skipped_count += 1
                    continue

                tracker_id = row_data[0] if row_data[0] else f"Row-{row_num}"
                created_date = row_data[3] if len(row_data) > 3 else ""
                updated_date = row_data[4] if len(row_data) > 4 else ""

                needs_update = False
                updated_row = list(row_data)

                # 登録日付（D列）の更新チェック
                if created_date and self._needs_format_update(created_date):
                    new_format = self._convert_to_new_format(created_date)
                    if new_format:
                        updated_row[3] = new_format
                        needs_update = True
                        logger.info(f"登録日付更新: {tracker_id} {created_date} → {new_format}")

                # 更新日付（E列）の更新チェック
                if updated_date and self._needs_format_update(updated_date):
                    new_format = self._convert_to_new_format(updated_date)
                    if new_format:
                        updated_row[4] = new_format
                        needs_update = True
                        logger.info(f"更新日付更新: {tracker_id} {updated_date} → {new_format}")

                # 更新が必要な場合のみGoogle Sheetsを更新
                if needs_update:
                    # 22列に補完
                    while len(updated_row) < 22:
                        updated_row.append("")

                    success = self._update_sheet_row(row_num, updated_row)
                    if success:
                        update_count += 1
                        logger.info(f"✅ 行{row_num}更新完了: {tracker_id}")
                    else:
                        logger.error(f"❌ 行{row_num}更新失敗: {tracker_id}")
                else:
                    skipped_count += 1

            logger.info("=" * 50)
            logger.info("日付フォーマット更新完了")
            logger.info(f"更新対象: {update_count}行")
            logger.info(f"スキップ: {skipped_count}行")
            logger.info(f"総処理: {update_count + skipped_count}行")

            return True

        except Exception as e:
            logger.error(f"日付フォーマット更新エラー: {e}")
            return False

    def _needs_format_update(self, date_str: str) -> bool:
        """
        日付フォーマット更新が必要かチェック

        Args:
            date_str: 日付文字列

        Returns:
            bool: 更新必要/不要
        """
        if not date_str or date_str.strip() == "":
            return False

        # 既に新フォーマット（時刻付き）の場合は更新不要
        if " " in date_str and ":" in date_str:
            return False

        # 旧フォーマット（日付のみ）の場合は更新必要
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _convert_to_new_format(self, date_str: str) -> str:
        """
        日付を新フォーマットに変換

        Args:
            date_str: 旧フォーマット日付

        Returns:
            str: 新フォーマット日付
        """
        try:
            # 旧フォーマットをパース
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            # デフォルト時刻（00:00:00）を追加
            return date_obj.strftime("%Y-%m-%d 00:00:00")

        except ValueError:
            logger.warning(f"日付変換失敗: {date_str}")
            return date_str

    def _update_sheet_row(self, row_number: int, row_data: List[str]) -> bool:
        """
        指定行をGoogle Sheetsで更新

        Args:
            row_number: 行番号（1開始）
            row_data: 行データ

        Returns:
            bool: 更新成功/失敗
        """
        try:
            if not self.updater.service:
                logger.error("Google Sheets API未初期化")
                return False

            sheet_name = "シート1"
            range_name = f"{sheet_name}!A{row_number}:V{row_number}"

            update_body = {"values": [row_data]}

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

            return True

        except Exception as e:
            logger.error(f"行更新エラー（行{row_number}）: {e}")
            return False


def main():
    """メイン実行"""
    logger.info("Google Sheets日付フォーマット更新ツール")

    updater = DateFormatUpdater()
    success = updater.update_all_dates()

    if success:
        logger.info("✅ 日付フォーマット更新完了")
        return 0
    else:
        logger.error("❌ 日付フォーマット更新失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
