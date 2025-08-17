#!/usr/bin/env python3
"""
既存PROGRESS_TRACKER.mdからGoogle Sheetsへのデータ移行
"""

import re
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskRecord, TaskStatus, ComponentStatus

logger = logging.getLogger(__name__)


class MarkdownMigrationTool:
    """PROGRESS_TRACKER.md移行ツール"""
    
    def __init__(self, markdown_path: str):
        """初期化"""
        self.markdown_path = Path(markdown_path)
        self.config = get_default_config()
        self.manager = ProgressManager(self.config)
    
    def parse_markdown_tasks(self) -> List[Dict]:
        """Markdownファイルからタスク情報を抽出"""
        if not self.markdown_path.exists():
            raise FileNotFoundError(f"PROGRESS_TRACKER.mdが見つかりません: {self.markdown_path}")
        
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tasks = []
        
        # Phase 1完了済みタスクの抽出
        phase1_pattern = r'(PH1-\d+):\s*"([^"]+)"\s*\n\s*status:\s*✅\s*COMPLETED\s*\n\s*achievement:\s*([^\n]+)\s*\n\s*completion_date:\s*([^\n]+)'
        phase1_matches = re.findall(phase1_pattern, content, re.MULTILINE)
        
        for match in phase1_matches:
            tracker_id, description, achievement, completion_date = match
            tasks.append({
                'tracker_id': tracker_id,
                'description': description.strip(),
                'status': TaskStatus.RELEASE,
                'completion_date': completion_date.strip(),
                'achievement': achievement.strip()
            })
        
        # Phase 2タスクの抽出
        phase2_pattern = r'(PH2-\d+):\s*"([^"]+)"\s*\n\s*status:\s*(✅\s*COMPLETED|🔄\s*PLANNED|⏳\s*IN_PROGRESS|❌\s*FAILED)'
        phase2_matches = re.findall(phase2_pattern, content, re.MULTILINE)
        
        for match in phase2_matches:
            tracker_id, description, status_str = match
            
            # ステータス変換
            if '✅ COMPLETED' in status_str:
                status = TaskStatus.RELEASE
            elif '🔄 PLANNED' in status_str:
                status = TaskStatus.NOT_STARTED
            elif '⏳ IN_PROGRESS' in status_str:
                status = TaskStatus.IN_PROGRESS
            elif '❌ FAILED' in status_str:
                status = TaskStatus.IN_PROGRESS  # 失敗も着手中として扱う
            else:
                status = TaskStatus.NOT_STARTED
            
            tasks.append({
                'tracker_id': tracker_id,
                'description': description.strip(),
                'status': status,
                'completion_date': None,
                'achievement': None
            })
        
        logger.info(f"Markdownから{len(tasks)}件のタスクを抽出しました")
        return tasks
    
    def migrate_tasks(self, dry_run: bool = True) -> List[TaskRecord]:
        """タスクをGoogle Sheetsに移行"""
        try:
            # Markdownからタスク抽出
            markdown_tasks = self.parse_markdown_tasks()
            
            # 既存タスクとの重複チェック
            existing_tasks = self.manager.get_all_tasks()
            existing_ids = {task.tracker_id for task in existing_tasks}
            
            migrated_tasks = []
            skipped_tasks = []
            
            for task_data in markdown_tasks:
                tracker_id = task_data['tracker_id']
                
                if tracker_id in existing_ids:
                    skipped_tasks.append(tracker_id)
                    logger.info(f"スキップ (既存): {tracker_id}")
                    continue
                
                # TaskRecordオブジェクト作成
                task = TaskRecord(
                    tracker_id=tracker_id,
                    description=task_data['description'],
                    status=task_data['status']
                )
                
                # 完了日の設定
                if task_data['completion_date']:
                    try:
                        completion_date = datetime.strptime(task_data['completion_date'], '%Y-%m-%d')
                        task.created_date = completion_date
                        task.updated_date = completion_date
                    except ValueError:
                        logger.warning(f"日付解析失敗: {task_data['completion_date']}")
                
                # ドライラン確認
                if dry_run:
                    print(f"[DRY RUN] 移行予定: {tracker_id} - {task_data['description']}")
                    migrated_tasks.append(task)
                else:
                    # 実際の移行
                    try:
                        self.manager.client.update_task(task)
                        migrated_tasks.append(task)
                        logger.info(f"移行完了: {tracker_id}")
                    except Exception as e:
                        logger.error(f"移行失敗: {tracker_id} - {e}")
            
            # 結果サマリー
            print(f"\n📊 移行結果サマリー:")
            print(f"  移行対象: {len(markdown_tasks)}件")
            print(f"  移行{'予定' if dry_run else '完了'}: {len(migrated_tasks)}件")
            print(f"  スキップ: {len(skipped_tasks)}件")
            
            if skipped_tasks:
                print(f"  スキップタスク: {', '.join(skipped_tasks)}")
            
            if dry_run:
                print(f"\n💡 実際の移行を実行するには --execute フラグを使用してください")
            
            return migrated_tasks
            
        except Exception as e:
            logger.error(f"移行エラー: {e}")
            raise
    
    def create_initial_tasks(self) -> None:
        """初期タスクセットの作成"""
        try:
            # 基本的なPhase 2タスクを作成
            initial_tasks = [
                {
                    'tracker_id': 'PHS-005',
                    'description': 'システム全体性能評価・ボトルネック特定',
                    'status': TaskStatus.RELEASE
                },
                {
                    'tracker_id': 'PHS-006', 
                    'description': 'アーキテクチャ最適化・安定性確保',
                    'status': TaskStatus.QUALITY_CHECK
                },
                {
                    'tracker_id': 'PHS-007',
                    'description': 'Google Sheets進捗管理システム実装',
                    'status': TaskStatus.IN_PROGRESS
                }
            ]
            
            for task_data in initial_tasks:
                task = TaskRecord(
                    tracker_id=task_data['tracker_id'],
                    description=task_data['description'],
                    status=task_data['status']
                )
                
                # 既存チェック
                existing_task = self.manager.get_task(task.tracker_id)
                if existing_task:
                    logger.info(f"スキップ (既存): {task.tracker_id}")
                    continue
                
                self.manager.client.update_task(task)
                logger.info(f"初期タスク作成: {task.tracker_id}")
            
            print("✅ 初期タスクセット作成完了")
            
        except Exception as e:
            logger.error(f"初期タスク作成エラー: {e}")
            raise


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PROGRESS_TRACKER.md移行ツール")
    parser.add_argument('--markdown-path', default='docs/workflows/PROGRESS_TRACKER.md',
                       help='PROGRESS_TRACKER.mdのパス')
    parser.add_argument('--execute', action='store_true',
                       help='実際の移行を実行（デフォルトはドライラン）')
    parser.add_argument('--create-initial', action='store_true',
                       help='初期タスクセットを作成')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        tool = MarkdownMigrationTool(args.markdown_path)
        
        if args.create_initial:
            tool.create_initial_tasks()
        else:
            tool.migrate_tasks(dry_run=not args.execute)
        
        return 0
        
    except Exception as e:
        logger.error(f"移行ツール実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())