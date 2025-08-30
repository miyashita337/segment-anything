#!/usr/bin/env python3
"""
SubAgent監視システム
QUAL-044: 同一セッション内でのタスク監視と次アクション実行

Task toolを活用して、同一セッション内でqueue_status.jsonを監視し、
タスク完了時に自動的に次のアクションを実行する
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubAgentMonitor:
    """SubAgent監視クラス"""
    
    def __init__(self, workspace_path: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-044"):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
        """
        self.workspace = Path(workspace_path)
        self.queue_dir = self.workspace / "queue"
        self.status_file = self.queue_dir / "queue_status.json"
        
        # 監視設定
        self.check_interval = 5  # 5秒間隔でチェック
        self.is_monitoring = False
        self.last_status: Optional[Dict[str, Any]] = None
        
        # コールバック登録
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        
        logger.info(f"SubAgentMonitor initialized for {self.workspace}")
    
    def read_status_file(self) -> Optional[Dict[str, Any]]:
        """状態ファイル読み込み"""
        if not self.status_file.exists():
            return None
        
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read status file: {e}")
            return None
    
    def start_monitoring(self, task_id: str) -> Dict[str, Any]:
        """
        タスク監視開始（同一セッション内実行）
        
        Args:
            task_id: 監視対象タスクID
            
        Returns:
            監視結果
        """
        logger.info(f"Starting monitoring for task: {task_id}")
        self.is_monitoring = True
        
        monitoring_result = {
            'task_id': task_id,
            'monitoring_started': time.time(),
            'status_changes': [],
            'final_status': None,
            'next_action_executed': False
        }
        
        try:
            while self.is_monitoring:
                status = self.read_status_file()
                
                if not status:
                    time.sleep(self.check_interval)
                    continue
                
                # 状態変化検出
                if self.last_status != status:
                    monitoring_result['status_changes'].append({
                        'timestamp': time.time(),
                        'status': status.get('status'),
                        'task_id': status.get('task_id')
                    })
                    
                    logger.info(f"Status change detected: {status.get('status')}")
                    
                    # タスク完了検出
                    if status.get('status') == 'task_completed' and status.get('task_id') == task_id:
                        logger.info(f"Task completed: {task_id}")
                        monitoring_result['final_status'] = 'completed'
                        
                        # 次アクション実行
                        if self.on_task_complete:
                            next_action_result = self.on_task_complete(status)
                            monitoring_result['next_action_result'] = next_action_result
                            monitoring_result['next_action_executed'] = True
                        
                        self.is_monitoring = False
                        break
                    
                    # タスク失敗検出
                    elif status.get('status') == 'task_failed' and status.get('task_id') == task_id:
                        logger.error(f"Task failed: {task_id}")
                        monitoring_result['final_status'] = 'failed'
                        
                        # PlanMode連携が必要
                        if status.get('requires_planmode_review'):
                            monitoring_result['requires_planmode'] = True
                            logger.info("Task requires PlanMode review")
                        
                        if self.on_task_failed:
                            self.on_task_failed(status)
                        
                        self.is_monitoring = False
                        break
                    
                    # エラー検出
                    elif status.get('status') == 'task_error' and status.get('task_id') == task_id:
                        logger.error(f"Task error: {task_id}")
                        monitoring_result['final_status'] = 'error'
                        monitoring_result['error'] = status.get('error')
                        
                        if self.on_task_error:
                            self.on_task_error(status)
                        
                        self.is_monitoring = False
                        break
                    
                    self.last_status = status
                
                time.sleep(self.check_interval)
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            monitoring_result['final_status'] = 'interrupted'
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            monitoring_result['final_status'] = 'error'
            monitoring_result['error'] = str(e)
        
        finally:
            monitoring_result['monitoring_ended'] = time.time()
            monitoring_result['duration'] = monitoring_result['monitoring_ended'] - monitoring_result['monitoring_started']
            
        return monitoring_result
    
    def stop_monitoring(self) -> None:
        """監視停止"""
        logger.info("Stopping monitoring")
        self.is_monitoring = False
    
    def register_callbacks(self, 
                          on_complete: Optional[Callable] = None,
                          on_failed: Optional[Callable] = None,
                          on_error: Optional[Callable] = None) -> None:
        """
        コールバック登録
        
        Args:
            on_complete: 完了時コールバック
            on_failed: 失敗時コールバック
            on_error: エラー時コールバック
        """
        self.on_task_complete = on_complete
        self.on_task_failed = on_failed
        self.on_task_error = on_error
        logger.info("Callbacks registered")
    
    def execute_next_action(self, task_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        次アクション実行（デフォルト実装）
        
        Args:
            task_status: タスク完了状態
            
        Returns:
            実行結果
        """
        logger.info("Executing next action based on task completion")
        
        result = {
            'action': 'next_task',
            'previous_task': task_status.get('task_id'),
            'timestamp': time.time()
        }
        
        # タスクタイプに応じた次アクション判定
        if 'pytest' in str(task_status.get('task_id', '')):
            # pytest完了後は結果分析
            result['next_action'] = 'analyze_test_results'
            result['details'] = 'Analyzing pytest results for failures and performance'
            
        elif 'extract_character' in str(task_status.get('task_id', '')):
            # extract_character完了後は品質評価
            result['next_action'] = 'evaluate_extraction_quality'
            result['details'] = 'Running quality assessment on extracted characters'
        
        else:
            # デフォルトアクション
            result['next_action'] = 'review_output'
            result['details'] = 'Reviewing task output for next steps'
        
        logger.info(f"Next action determined: {result['next_action']}")
        return result


class SubAgentIntegration:
    """SubAgent統合クラス（Claude Code内で使用）"""
    
    def __init__(self):
        """初期化"""
        self.monitor = SubAgentMonitor()
        self.current_context: Dict[str, Any] = {}
        logger.info("SubAgentIntegration initialized")
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        作業コンテキスト設定
        
        Args:
            context: 現在の作業コンテキスト（トラッカーID、Todoリスト等）
        """
        self.current_context = context
        logger.info(f"Context set: {context.get('tracker_id', 'unknown')}")
    
    def monitor_long_task(self, task_id: str, task_command: str) -> Dict[str, Any]:
        """
        長時間タスクの監視（同一セッション内）
        
        Args:
            task_id: タスクID
            task_command: 実行コマンド
            
        Returns:
            監視結果と次アクション
        """
        logger.info(f"Monitoring long task in same session: {task_id}")
        
        # コンテキスト継承の確認
        logger.info(f"Current context: {self.current_context}")
        
        # 完了時の自動アクション設定
        def on_complete(status):
            logger.info(f"Task {task_id} completed successfully")
            
            # コンテキストを保持したまま次アクション実行
            next_action = self.monitor.execute_next_action(status)
            next_action['context'] = self.current_context
            
            # Todoリスト更新（もし存在すれば）
            if 'todo_list' in self.current_context:
                logger.info("Updating Todo list with completion")
                # ここでTodoリスト更新処理
            
            return next_action
        
        # 失敗時のPlanMode連携
        def on_failed(status):
            logger.error(f"Task {task_id} failed")
            
            if status.get('requires_planmode_review'):
                logger.info("Switching to PlanMode for error review")
                # ここでPlanMode切り替え処理
                return {
                    'action': 'switch_to_planmode',
                    'reason': 'task_failure',
                    'task_id': task_id,
                    'error': status.get('error')
                }
        
        # コールバック登録
        self.monitor.register_callbacks(
            on_complete=on_complete,
            on_failed=on_failed
        )
        
        # 監視開始
        result = self.monitor.start_monitoring(task_id)
        
        # 結果にコンテキスト情報追加
        result['session_context'] = self.current_context
        result['same_session'] = True
        
        return result


def demonstrate_subagent_monitoring():
    """
    SubAgent監視のデモンストレーション
    実際のClaude Code内での使用例
    """
    print("🎯 SubAgent監視デモンストレーション")
    print("=" * 50)
    
    # SubAgent統合初期化
    integration = SubAgentIntegration()
    
    # 現在の作業コンテキスト設定（Claude Codeから継承）
    integration.set_context({
        'tracker_id': 'QUAL-044',
        'current_task': 'long_task_queue_implementation',
        'todo_list': ['implement_queue', 'test_system', 'deploy'],
        'session_id': 'current_claude_session'
    })
    
    print("📋 コンテキスト設定完了:")
    print(f"   トラッカー: QUAL-044")
    print(f"   セッション: 同一セッション内実行")
    print()
    
    # 長時間タスクの監視例
    print("🔄 長時間タスク監視開始...")
    print("   タスク: pytest実行")
    print("   監視: SubAgentによる同一セッション監視")
    print()
    
    # 実際の監視（デモ用にシミュレート）
    demo_result = {
        'task_id': 'pytest_20250830_161234',
        'monitoring_started': time.time(),
        'status_changes': [
            {'timestamp': time.time(), 'status': 'task_running'},
            {'timestamp': time.time() + 10, 'status': 'task_completed'}
        ],
        'final_status': 'completed',
        'next_action_executed': True,
        'next_action_result': {
            'action': 'analyze_test_results',
            'details': 'Analyzing pytest results for failures and performance',
            'context': {
                'tracker_id': 'QUAL-044',
                'session_id': 'current_claude_session'
            }
        },
        'session_context': {
            'tracker_id': 'QUAL-044',
            'current_task': 'long_task_queue_implementation',
            'todo_list': ['implement_queue', 'test_system', 'deploy'],
            'session_id': 'current_claude_session'
        },
        'same_session': True,
        'duration': 10.0
    }
    
    print("✅ 監視結果:")
    print(f"   タスク状態: {demo_result['final_status']}")
    print(f"   次アクション: {demo_result['next_action_result']['action']}")
    print(f"   セッション継続: {demo_result['same_session']}")
    print(f"   コンテキスト保持: ✅")
    print()
    
    print("🎯 重要な特徴:")
    print("   1. 同一セッション内で監視・実行")
    print("   2. コンテキスト（トラッカーID、Todo）完全継承")
    print("   3. 自動的な次アクション判定・実行")
    print("   4. PlanMode連携（エラー時）")
    print()
    
    return demo_result


def main():
    """CLI実行用メイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        # デモンストレーション実行
        demonstrate_subagent_monitoring()
    else:
        task_id = sys.argv[1]
        monitor = SubAgentMonitor()
        
        print(f"📍 Monitoring task: {task_id}")
        print("Press Ctrl+C to stop monitoring...")
        
        result = monitor.start_monitoring(task_id)
        
        print("\n📊 Monitoring Result:")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()