#!/usr/bin/env python3
"""
重複トラッカーID修正スクリプト
P1-A002-1, P1-A002-2, P1-A002-3の重複行を削除
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.google_sheets_updater import GoogleSheetsUpdater

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateTrackerIDFixer:
    """重複トラッカーID修正ツール"""
    
    def __init__(self):
        """初期化"""
        self.updater = GoogleSheetsUpdater()
        self.duplicate_ids = ['P1-A002-1', 'P1-A002-2', 'P1-A002-3']
    
    def fix_duplicates(self) -> bool:
        """
        重複トラッカーID修正
        
        Returns:
            bool: 修正成功/失敗
        """
        try:
            logger.info("重複トラッカーID修正開始")
            
            # 全データ取得
            all_data = self.updater.get_all_sheet_data()
            if not all_data:
                logger.error("データ取得失敗")
                return False
            
            # 重複分析
            duplicates_found = self._analyze_duplicates(all_data)
            
            if not duplicates_found:
                logger.info("重複なし、修正不要")
                return True
            
            # 重複行削除
            success = self._remove_duplicate_rows(duplicates_found)
            
            return success
            
        except Exception as e:
            logger.error(f"重複修正エラー: {e}")
            return False
    
    def _analyze_duplicates(self, all_data: List[List[str]]) -> Dict[str, List[int]]:
        """重複分析"""
        duplicates = {}
        
        for i, row in enumerate(all_data, start=1):
            if len(row) > 0:
                tracker_id = row[0]
                
                if tracker_id in self.duplicate_ids:
                    if tracker_id not in duplicates:
                        duplicates[tracker_id] = []
                    duplicates[tracker_id].append(i)
        
        # 重複があるIDのみ残す
        actual_duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
        
        logger.info("重複分析結果:")
        for tracker_id, rows in actual_duplicates.items():
            logger.info(f"  {tracker_id}: 行{rows}")
        
        return actual_duplicates
    
    def _remove_duplicate_rows(self, duplicates: Dict[str, List[int]]) -> bool:
        """重複行削除（後の行を削除）"""
        try:
            # 削除対象行を特定（各IDの2番目以降）
            rows_to_delete = []
            for tracker_id, row_numbers in duplicates.items():
                # 最初の行は残し、2番目以降を削除
                to_delete = row_numbers[1:]
                rows_to_delete.extend(to_delete)
                logger.info(f"{tracker_id}: 行{to_delete}を削除予定（行{row_numbers[0]}は保持）")
            
            # 降順ソート（後ろから削除）
            rows_to_delete.sort(reverse=True)
            
            logger.info(f"削除対象行: {rows_to_delete}")
            
            # 行削除実行
            for row_num in rows_to_delete:
                success = self._delete_row(row_num)
                if success:
                    logger.info(f"✅ 行{row_num}削除完了")
                else:
                    logger.error(f"❌ 行{row_num}削除失敗")
                    return False
            
            logger.info("✅ 重複行削除完了")
            return True
            
        except Exception as e:
            logger.error(f"重複行削除エラー: {e}")
            return False
    
    def _delete_row(self, row_number: int) -> bool:
        """指定行削除"""
        try:
            # Google SheetsのbatchUpdate APIを使用
            requests = [{
                'deleteDimension': {
                    'range': {
                        'sheetId': 0,  # シート1のID
                        'dimension': 'ROWS',
                        'startIndex': row_number - 1,  # 0ベースインデックス
                        'endIndex': row_number
                    }
                }
            }]
            
            body = {'requests': requests}
            
            result = self.updater.service.spreadsheets().batchUpdate(
                spreadsheetId=self.updater.spreadsheet_id,
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"行削除エラー（行{row_number}）: {e}")
            return False
    
    def verify_fix(self) -> bool:
        """修正検証"""
        try:
            logger.info("修正検証開始")
            
            # 全データ再取得
            all_data = self.updater.get_all_sheet_data()
            if not all_data:
                logger.error("検証用データ取得失敗")
                return False
            
            # 重複チェック
            duplicates = self._analyze_duplicates(all_data)
            
            if duplicates:
                logger.error(f"修正後も重複が残存: {duplicates}")
                return False
            
            # 各IDが1つずつ存在することを確認
            found_ids = set()
            for row in all_data:
                if len(row) > 0 and row[0] in self.duplicate_ids:
                    found_ids.add(row[0])
            
            if found_ids == set(self.duplicate_ids):
                logger.info("✅ 修正検証成功: 各IDが1つずつ存在")
                return True
            else:
                missing = set(self.duplicate_ids) - found_ids
                logger.error(f"修正後に不足ID: {missing}")
                return False
            
        except Exception as e:
            logger.error(f"検証エラー: {e}")
            return False


def main():
    """メイン実行"""
    logger.info("重複トラッカーID修正ツール")
    
    fixer = DuplicateTrackerIDFixer()
    
    # 修正実行
    success = fixer.fix_duplicates()
    
    if success:
        # 検証
        verify_success = fixer.verify_fix()
        
        if verify_success:
            logger.info("=" * 50)
            logger.info("重複トラッカーID修正完了")
            logger.info("=" * 50)
            logger.info("修正されたID:")
            for tracker_id in fixer.duplicate_ids:
                logger.info(f"  - {tracker_id}: 重複削除、1つに統一")
            
            return 0
        else:
            logger.error("❌ 修正検証失敗")
            return 1
    else:
        logger.error("❌ 修正失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())