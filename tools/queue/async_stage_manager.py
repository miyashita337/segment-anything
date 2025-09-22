#!/usr/bin/env python3
"""
SubAgent非同期多段階実行管理システム
QUAL-044: Claude Code 2分制限対応の段階的実行制御

Usage:
    python tools/queue/async_stage_manager.py register QUAL-044 /input/dir --task-type extraction
    python tools/queue/async_stage_manager.py monitor QUAL-044
    python tools/queue/async_stage_manager.py collect QUAL-044
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.queue.task_integration import TaskOrchestrator
from config.workspace_config import WorkspaceConfig

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncStageManager:
    """非同期多段階実行管理クラス"""
    
    def __init__(self, tracker_id: str):
        """
        初期化
        
        Args:
            tracker_id: トラッカーID
        """
        self.tracker_id = tracker_id
        self.orchestrator = TaskOrchestrator(tracker_id)
        
        # ワークスペース設定
        config = WorkspaceConfig()
        self.workspace = Path(config.get_tracker_workspace(tracker_id))
        self.stage_file = self.workspace / "async_stage_status.json"
        
        logger.info(f"AsyncStageManager initialized for {tracker_id}")
        logger.info(f"Workspace: {self.workspace}")
    
    def save_stage_status(self, stage_data: Dict[str, Any]) -> None:
        """
        段階状態を永続化
        
        Args:
            stage_data: 段階データ
        """
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            
            # 既存データと統合
            existing_data = {}
            if self.stage_file.exists():
                try:
                    with open(self.stage_file, 'r') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Existing stage file contains invalid JSON")
            
            # データ更新
            existing_data.update(stage_data)
            existing_data['last_updated'] = time.time()
            
            with open(self.stage_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
            logger.info(f"Stage status saved: {stage_data.get('stage', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to save stage status: {e}")
    
    def load_stage_status(self) -> Dict[str, Any]:
        """
        段階状態を読み込み
        
        Returns:
            段階データ
        """
        try:
            if self.stage_file.exists():
                with open(self.stage_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Stage status loaded: {data.get('stage', 'unknown')}")
                return data
            else:
                logger.info("No existing stage file found")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in stage file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load stage status: {e}")
            return {}
    
    def stage_register(self, 
                      input_dir: str,
                      task_type: str = "extraction",
                      **kwargs) -> Dict[str, Any]:
        """
        段階1: タスク登録実行
        
        Args:
            input_dir: 入力ディレクトリ
            task_type: タスクタイプ
            **kwargs: その他のオプション
            
        Returns:
            登録結果
        """
        logger.info("=" * 50)
        logger.info("🚀 Stage 1: Task Registration")
        logger.info("=" * 50)
        
        try:
            # 入力パス検証
            input_path = Path(input_dir)
            if not input_path.exists():
                error_result = {
                    'stage': 'register',
                    'status': 'error',
                    'error': f'入力ディレクトリが存在しません: {input_dir}',
                    'message': 'パスを確認してください'
                }
                self.save_stage_status(error_result)
                return error_result
            
            logger.info(f"📂 Input directory: {input_dir}")
            logger.info(f"🎯 Task type: {task_type}")
            
            # 既存TaskOrchestratorの非同期登録メソッド活用
            result = self.orchestrator.register_async_task(
                input_dir=input_dir,
                task_type=task_type,
                **kwargs
            )
            
            # 段階情報追加
            result['stage'] = 'register'
            result['input_dir'] = input_dir
            result['tracker_id'] = self.tracker_id
            
            # 状態永続化
            self.save_stage_status(result)
            
            logger.info(f"✅ Registration result: {result['status']}")
            if result['status'] == 'registered':
                logger.info(f"📊 Task ID: {result['task_id']}")
                logger.info("🔄 Background processing started")
            
            return result
            
        except Exception as e:
            error_result = {
                'stage': 'register',
                'status': 'error',
                'error': str(e),
                'message': f'登録処理でエラーが発生しました: {e}'
            }
            logger.error(f"Registration failed: {e}")
            self.save_stage_status(error_result)
            return error_result
    
    def stage_monitor(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        段階2: タスク監視実行
        
        Args:
            task_id: 監視するタスクID
            
        Returns:
            監視結果
        """
        logger.info("=" * 50)
        logger.info("👁️ Stage 2: Task Monitoring")
        logger.info("=" * 50)
        
        try:
            # 既存段階状態読み込み
            stage_status = self.load_stage_status()
            if not task_id and stage_status.get('task_id'):
                task_id = stage_status['task_id']
            
            logger.info(f"🔍 Monitoring task: {task_id or 'latest'}")
            
            # 既存TaskOrchestratorの状態確認メソッド活用
            result = self.orchestrator.check_task_status(task_id)
            
            # 段階情報追加
            result['stage'] = 'monitor'
            result['tracker_id'] = self.tracker_id
            
            # 詳細ログ出力
            logger.info(f"📊 Current status: {result['status']}")
            logger.info(f"💬 Message: {result['message']}")
            
            if result['status'] == 'running':
                logger.info("⏳ Task is still running in background")
                logger.info("🔄 Continue monitoring with next command")
            elif result['status'] == 'completed':
                logger.info("🎉 Task completed successfully")
                logger.info("📋 Ready for result collection")
            elif result['status'] == 'failed':
                logger.info("❌ Task failed")
                if result.get('details', {}).get('requires_manual_review'):
                    logger.info("⚠️ Manual review required")
            
            # 状態更新
            stage_status.update(result)
            self.save_stage_status(stage_status)
            
            return result
            
        except Exception as e:
            error_result = {
                'stage': 'monitor',
                'status': 'error',
                'error': str(e),
                'message': f'監視処理でエラーが発生しました: {e}'
            }
            logger.error(f"Monitoring failed: {e}")
            self.save_stage_status(error_result)
            return error_result
    
    def stage_collect(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        段階3: 結果収集実行
        
        Args:
            task_id: 収集するタスクID
            
        Returns:
            収集結果
        """
        logger.info("=" * 50)
        logger.info("📊 Stage 3: Result Collection")
        logger.info("=" * 50)
        
        try:
            # 既存段階状態読み込み
            stage_status = self.load_stage_status()
            if not task_id and stage_status.get('task_id'):
                task_id = stage_status['task_id']
            
            logger.info(f"📋 Collecting results for: {task_id or 'latest'}")
            
            # 既存TaskOrchestratorの結果収集メソッド活用
            result = self.orchestrator.collect_task_results(task_id)
            
            # 段階情報追加
            result['stage'] = 'collect'
            result['tracker_id'] = self.tracker_id
            
            # 詳細ログ出力
            logger.info(f"📊 Collection status: {result['status']}")
            logger.info(f"💬 Message: {result['message']}")
            
            if result['status'] == 'completed':
                logger.info("✅ Results collected successfully")
                logger.info(f"📄 Report: {result.get('report_path')}")
                logger.info(f"🖼️ Result files: {result.get('result_files', 0)}")
                logger.info(f"📂 Extraction dir: {result.get('extraction_dir')}")
            elif result['status'] == 'not_ready':
                logger.info("⏳ Task not ready for collection")
                logger.info(f"Current status: {result.get('current_status', {}).get('status')}")
            
            # 状態更新
            stage_status.update(result)
            self.save_stage_status(stage_status)
            
            return result
            
        except Exception as e:
            error_result = {
                'stage': 'collect',
                'status': 'error',
                'error': str(e),
                'message': f'収集処理でエラーが発生しました: {e}'
            }
            logger.error(f"Collection failed: {e}")
            self.save_stage_status(error_result)
            return error_result
    
    def get_full_status(self) -> Dict[str, Any]:
        """
        全段階の状態取得
        
        Returns:
            完全な状態情報
        """
        stage_status = self.load_stage_status()
        
        # 現在の監視状況も確認
        if stage_status.get('task_id'):
            current_status = self.orchestrator.check_task_status(stage_status['task_id'])
            stage_status['current_monitor'] = current_status
        
        return {
            'tracker_id': self.tracker_id,
            'workspace': str(self.workspace),
            'stage_file': str(self.stage_file),
            'full_status': stage_status,
            'last_updated': stage_status.get('last_updated'),
            'next_recommended_action': self._recommend_next_action(stage_status)
        }
    
    def _recommend_next_action(self, stage_status: Dict[str, Any]) -> str:
        """
        次のアクション推奨
        
        Args:
            stage_status: 段階状態
            
        Returns:
            推奨アクション
        """
        last_stage = stage_status.get('stage')
        last_status = stage_status.get('status')
        
        if not last_stage:
            return "register: 最初にタスクを登録してください"
        elif last_stage == 'register' and last_status == 'registered':
            return "monitor: タスクの状態を監視してください"
        elif last_stage == 'monitor' and last_status == 'running':
            return "monitor: まだ実行中です。再度監視してください"
        elif last_stage == 'monitor' and last_status == 'completed':
            return "collect: タスクが完了しました。結果を収集してください"
        elif last_stage == 'collect' and last_status == 'completed':
            return "complete: 全ての段階が完了しました"
        elif last_status == 'failed':
            return "review: タスクが失敗しました。エラー内容を確認してください"
        elif last_status == 'error':
            return "fix: エラーが発生しました。問題を修正してください"
        else:
            return "unknown: 状態を確認して適切なアクションを選択してください"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="SubAgent非同期多段階実行管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 段階1: タスク登録
  python tools/queue/async_stage_manager.py register QUAL-044 /mnt/c/AItools/lora/train/kiri/aichikan/
  
  # 段階2: 状態監視
  python tools/queue/async_stage_manager.py monitor QUAL-044
  
  # 段階3: 結果収集
  python tools/queue/async_stage_manager.py collect QUAL-044
  
  # 状態確認
  python tools/queue/async_stage_manager.py status QUAL-044
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='実行コマンド')
    
    # register サブコマンド
    register_parser = subparsers.add_parser('register', help='タスク登録')
    register_parser.add_argument('tracker_id', help='トラッカーID')
    register_parser.add_argument('input_dir', help='入力ディレクトリ')
    register_parser.add_argument('--task-type', default='extraction', help='タスクタイプ')
    register_parser.add_argument('--max-files', type=int, help='最大処理ファイル数')
    register_parser.add_argument('--quality-method', default='balanced', help='品質評価手法')
    
    # monitor サブコマンド
    monitor_parser = subparsers.add_parser('monitor', help='タスク監視')
    monitor_parser.add_argument('tracker_id', help='トラッカーID')
    monitor_parser.add_argument('--task-id', help='監視するタスクID')
    
    # collect サブコマンド
    collect_parser = subparsers.add_parser('collect', help='結果収集')
    collect_parser.add_argument('tracker_id', help='トラッカーID')
    collect_parser.add_argument('--task-id', help='収集するタスクID')
    
    # status サブコマンド
    status_parser = subparsers.add_parser('status', help='状態確認')
    status_parser.add_argument('tracker_id', help='トラッカーID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 段階管理システム初期化
    manager = AsyncStageManager(args.tracker_id)
    
    # コマンド実行
    if args.command == 'register':
        kwargs = {}
        if args.max_files:
            kwargs['max_files'] = args.max_files
        if args.quality_method:
            kwargs['quality_method'] = args.quality_method
            
        result = manager.stage_register(
            input_dir=args.input_dir,
            task_type=args.task_type,
            **kwargs
        )
    elif args.command == 'monitor':
        result = manager.stage_monitor(task_id=args.task_id)
    elif args.command == 'collect':
        result = manager.stage_collect(task_id=args.task_id)
    elif args.command == 'status':
        result = manager.get_full_status()
    
    # 結果出力
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    
    # 終了コード
    if result.get('status') in ['completed', 'registered', 'running']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()