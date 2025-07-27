#!/usr/bin/env python3
"""
ワークフロー統合スクリプト
run_quality_workflow.shと連携して自動で進捗・品質指標を更新
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskStatus, ComponentStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowIntegrator:
    """ワークフロー統合クラス"""
    
    def __init__(self):
        """初期化"""
        self.config = get_default_config()
        self.manager = ProgressManager(self.config)
        
    def run_quality_workflow_with_tracking(self, tracker_id: str, **kwargs) -> Dict[str, Any]:
        """品質ワークフロー実行＋進捗追跡"""
        try:
            # タスク開始マーク
            task = self.manager.get_task(tracker_id)
            if not task:
                task = self.manager.create_task(tracker_id, "品質ワークフロー実行")
            
            self.manager.update_task_status(tracker_id, TaskStatus.IN_PROGRESS)
            logger.info(f"品質ワークフロー開始: {tracker_id}")
            
            # 統合品質チェッカー実行
            results = self._run_unified_quality_checker(**kwargs)
            
            # 結果に基づく進捗更新
            success = results.get('success', False)
            
            if success:
                # 品質評価完了マーク
                self.manager.update_component_status(tracker_id, 'quality_evaluation', ComponentStatus.COMPLETED)
                
                # 10指標データ自動更新
                if 'quality_results' in results:
                    self.manager.update_from_quality_checker_results(tracker_id, results['quality_results'])
                
                # ステータス進行
                self.manager.update_task_status(tracker_id, TaskStatus.QUALITY_CHECK)
                logger.info(f"品質ワークフロー成功: {tracker_id}")
            else:
                # 失敗マーク
                self.manager.update_component_status(tracker_id, 'quality_evaluation', ComponentStatus.FAILED)
                logger.warning(f"品質ワークフロー失敗: {tracker_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"品質ワークフロー実行エラー: {e}")
            self.manager.update_component_status(tracker_id, 'quality_evaluation', ComponentStatus.FAILED)
            return {'success': False, 'error': str(e)}
    
    def _run_unified_quality_checker(self, **kwargs) -> Dict[str, Any]:
        """統合品質チェッカー実行"""
        try:
            # unified_quality_checker.pyを直接インポートして実行
            sys.path.append(str(Path(__file__).parent.parent))
            from unified_quality_checker import UnifiedQualityChecker
            
            # チェッカー実行
            checker = UnifiedQualityChecker()
            results = checker.evaluate_all_metrics(**kwargs)
            
            return {
                'success': True,
                'quality_results': results,
                'message': '品質評価完了'
            }
            
        except Exception as e:
            logger.error(f"統合品質チェッカー実行エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_extraction_pipeline_with_tracking(self, tracker_id: str, **kwargs) -> Dict[str, Any]:
        """抽出パイプライン実行＋進捗追跡"""
        try:
            # 抽出パイプライン開始
            self.manager.update_component_status(tracker_id, 'extraction_pipeline', ComponentStatus.IN_PROGRESS)
            
            # extract_kana03.py実行（例）
            cmd = ['python', 'extract_kana03.py']
            if 'quality_method' in kwargs:
                cmd.extend(['--quality_method', kwargs['quality_method']])
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            success = result.returncode == 0
            
            if success:
                self.manager.update_component_status(tracker_id, 'extraction_pipeline', ComponentStatus.COMPLETED)
                self.manager.update_task_status(tracker_id, TaskStatus.EXTRACTION_PIPELINE)
                logger.info(f"抽出パイプライン成功: {tracker_id}")
            else:
                self.manager.update_component_status(tracker_id, 'extraction_pipeline', ComponentStatus.FAILED)
                logger.error(f"抽出パイプライン失敗: {tracker_id}")
            
            return {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except Exception as e:
            logger.error(f"抽出パイプライン実行エラー: {e}")
            self.manager.update_component_status(tracker_id, 'extraction_pipeline', ComponentStatus.FAILED)
            return {'success': False, 'error': str(e)}
    
    def complete_workflow(self, tracker_id: str) -> None:
        """ワークフロー完了処理"""
        try:
            # 全コンポーネント状態確認
            task = self.manager.get_task(tracker_id)
            if not task:
                logger.warning(f"タスクが見つかりません: {tracker_id}")
                return
            
            all_completed = all([
                task.operation_check == ComponentStatus.COMPLETED,
                task.unit_test == ComponentStatus.COMPLETED,
                task.quality_evaluation == ComponentStatus.COMPLETED,
                task.integration_script == ComponentStatus.COMPLETED,
                task.dashboard_generation == ComponentStatus.COMPLETED,
                task.extraction_pipeline == ComponentStatus.COMPLETED
            ])
            
            if all_completed:
                self.manager.update_task_status(tracker_id, TaskStatus.RELEASE)
                logger.info(f"ワークフロー完全完了: {tracker_id}")
            
        except Exception as e:
            logger.error(f"ワークフロー完了処理エラー: {e}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ワークフロー統合実行")
    parser.add_argument('tracker_id', help='トラッカーID')
    parser.add_argument('--mode', choices=['quality', 'extraction', 'complete'], 
                       default='quality', help='実行モード')
    parser.add_argument('--quality-method', default='balanced', 
                       help='品質評価手法')
    
    args = parser.parse_args()
    
    try:
        integrator = WorkflowIntegrator()
        
        if args.mode == 'quality':
            results = integrator.run_quality_workflow_with_tracking(
                args.tracker_id, 
                quality_method=args.quality_method
            )
            print(f"📊 品質ワークフロー結果: {results}")
            
        elif args.mode == 'extraction':
            results = integrator.run_extraction_pipeline_with_tracking(
                args.tracker_id,
                quality_method=args.quality_method
            )
            print(f"🔧 抽出パイプライン結果: {results}")
            
        elif args.mode == 'complete':
            integrator.complete_workflow(args.tracker_id)
            print(f"✅ ワークフロー完了: {args.tracker_id}")
        
        return 0
        
    except Exception as e:
        logger.error(f"ワークフロー統合実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())