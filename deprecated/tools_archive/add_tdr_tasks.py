#!/usr/bin/env python3
"""
Tools Directory Refactoring タスク追加スクリプト
TDR-001, TDR-002, TDR-003をGoogle Sheetsに追加
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
from tools.progress_tracker.data_models import PriorityLevel, TaskRecord, TaskStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TDRTaskCreator:
    """Tools Directory Refactoring タスク作成ツール"""
    
    def __init__(self):
        """初期化"""
        self.updater = GoogleSheetsUpdater()
        
        # 3つのTDRタスク定義
        self.tdr_tasks = [
            {
                'tracker_id': 'TDR-001',
                'description': 'Tools Directory Phase 1: 分類・整理',
                'details': '使い捨てスクリプト削除と機能別ディレクトリ再編成。split_p1_a002_tasks.py、update_date_format.py、fix_duplicate_tracker_ids.py、migrate_to_23_columns.pyをdeprecated/に移動。core/、batch/、testing/、scripts/ディレクトリ作成。45ファイル→40程度への削減を実現し、将来の100ファイル超え回避基盤を構築する。'
            },
            {
                'tracker_id': 'TDR-002', 
                'description': 'Tools Directory Phase 2: 統合ツール作成',
                'details': '統合管理CLI（tools/manager.py）作成。Google Sheets操作統合（read/update/format）、バッチ処理統合、一時スクリプト自動クリーンアップ機能を実装。tools/scripts/下の自動期限管理、実行後の自動deprecated移動機能を構築し、使い捨てスクリプトの適切なライフサイクル管理を確立する。'
            },
            {
                'tracker_id': 'TDR-003',
                'description': 'Tools Directory Phase 3: ガバナンス確立',
                'details': 'ファイル作成ルール確立と定期メンテナンス戦略構築。一時作業はtools/scripts/限定、継続ツールはtools/core/配置ルールの運用開始。月次不要ファイル確認、deprecated/移動戦略確立。自動クリーンアップ導入により、ファイル数増加抑制・再利用性向上・メンテナンス負荷軽減を実現する。'
            }
        ]
    
    def create_all_tdr_tasks(self) -> bool:
        """
        全TDRタスク作成
        
        Returns:
            bool: 作成成功/失敗
        """
        try:
            logger.info("Tools Directory Refactoring タスク作成開始")
            
            success_count = 0
            for task_data in self.tdr_tasks:
                success = self._create_tdr_task(task_data)
                if success:
                    success_count += 1
                else:
                    logger.error(f"タスク作成失敗: {task_data['tracker_id']}")
            
            if success_count == len(self.tdr_tasks):
                logger.info("✅ 全TDRタスク作成完了")
                return True
            else:
                logger.error(f"❌ 一部タスク作成失敗: {success_count}/{len(self.tdr_tasks)}")
                return False
            
        except Exception as e:
            logger.error(f"TDRタスク作成エラー: {e}")
            return False
    
    def _create_tdr_task(self, task_data: Dict[str, str]) -> bool:
        """
        TDRタスク作成
        
        Args:
            task_data: タスクデータ
            
        Returns:
            bool: 作成成功/失敗
        """
        try:
            # 既存タスク確認
            existing_task = self.updater.get_task_by_id(task_data['tracker_id'])
            if existing_task:
                logger.info(f"タスク既存のためスキップ: {task_data['tracker_id']}")
                return True
            
            # TaskRecord作成
            task = TaskRecord(
                tracker_id=task_data['tracker_id'],
                priority=PriorityLevel.HIGHEST,  # 優先度最高
                status=TaskStatus.NOT_STARTED,
                created_date=datetime.now(),
                description=task_data['description'],
                details=task_data['details']
            )
            
            # Google Sheetsに追加
            task_row = task.to_sheets_row()
            
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A:W"
            
            body = {
                'values': [task_row]
            }
            
            result = self.updater.service.spreadsheets().values().append(
                spreadsheetId=self.updater.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"✅ TDRタスク作成完了: {task_data['tracker_id']}")
            logger.info(f"  概要: {task_data['description']}")
            
            return True
            
        except Exception as e:
            logger.error(f"TDRタスク作成エラー ({task_data['tracker_id']}): {e}")
            return False
    
    def verify_tdr_tasks(self) -> bool:
        """TDRタスク作成検証"""
        try:
            logger.info("TDRタスク作成検証開始")
            
            all_created = True
            for task_data in self.tdr_tasks:
                task = self.updater.get_task_by_id(task_data['tracker_id'])
                if not task:
                    logger.error(f"TDRタスク未作成: {task_data['tracker_id']}")
                    all_created = False
                else:
                    status = task[2]  # C列：ステータス
                    priority = task[1]  # B列：優先度
                    logger.info(f"✅ {task_data['tracker_id']}: {status}, {priority}")
            
            if all_created:
                logger.info("✅ 全TDRタスク検証完了")
                return True
            else:
                logger.error("❌ TDRタスク検証失敗")
                return False
            
        except Exception as e:
            logger.error(f"TDRタスク検証エラー: {e}")
            return False


def main():
    """メイン実行"""
    logger.info("Tools Directory Refactoring タスク追加ツール")
    
    creator = TDRTaskCreator()
    
    # TDRタスク作成
    success = creator.create_all_tdr_tasks()
    
    if success:
        # 検証
        verify_success = creator.verify_tdr_tasks()
        
        if verify_success:
            logger.info("=" * 60)
            logger.info("Tools Directory Refactoring タスク追加完了")
            logger.info("=" * 60)
            logger.info("追加されたタスク:")
            for task in creator.tdr_tasks:
                logger.info(f"  - {task['tracker_id']}: {task['description']}")
            logger.info("")
            logger.info("優先度: 優先度最高")
            logger.info("ステータス: 着手前")
            logger.info("Tools Directory 大規模リファクタリングの実装準備が完了しました")
            
            return 0
        else:
            logger.error("❌ 検証失敗")
            return 1
    else:
        logger.error("❌ タスク作成失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())