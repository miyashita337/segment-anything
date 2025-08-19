#!/usr/bin/env python3
"""
進捗管理システムメインクラス修正版
Google Sheetsとの連携による進捗追跡機能
v0.9.21 - QUAL-034対応（MetricsRecord -> StatisticalRecord）
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .data_models import (
    TaskRecord, TaskStatus, ComponentStatus, StatisticalRecord,
    ProgressTrackerConfig, ProgressTrackerError
)
from .sheets_client import GoogleSheetsClient

logger = logging.getLogger(__name__)


class ProgressManager:
    """進捗管理システム"""
    
    def __init__(self, config: ProgressTrackerConfig):
        """初期化"""
        self.config = config
        self.client = GoogleSheetsClient(config)
        
        # 初回初期化チェック
        self._ensure_sheet_initialized()
    
    def _ensure_sheet_initialized(self) -> None:
        """シート初期化確認"""
        try:
            # ヘッダー行確認（23列）
            headers = self.client.get_sheet_values("A1:W1")
            if not headers:
                logger.info("シートを初期化しています...")
                self.client.initialize_sheet()
            
        except Exception as e:
            logger.warning(f"シート初期化確認エラー: {e}")
    
    def create_task(self, tracker_id: str, description: str = "") -> TaskRecord:
        """新規タスク作成"""
        try:
            # 既存チェック
            existing_task = self.get_task(tracker_id)
            if existing_task:
                raise ProgressTrackerError(f"タスクが既に存在します: {tracker_id}")
            
            # 新規タスク作成
            task = TaskRecord(
                tracker_id=tracker_id,
                description=description,
                status=TaskStatus.NOT_STARTED
            )
            
            # Google Sheetsに追加
            self.client.update_task(task)
            logger.info(f"新規タスク作成: {tracker_id}")
            
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"タスク作成失敗: {e}")
    
    def get_task(self, tracker_id: str) -> Optional[TaskRecord]:
        """タスク取得"""
        try:
            return self.client.get_task(tracker_id)
        except Exception as e:
            raise ProgressTrackerError(f"タスク取得失敗: {e}")
    
    def update_task_status(self, tracker_id: str, status: TaskStatus) -> TaskRecord:
        """タスクステータス更新"""
        try:
            task = self.get_task(tracker_id)
            if not task:
                raise ProgressTrackerError(f"タスクが見つかりません: {tracker_id}")
            
            task.update_status(status)
            self.client.update_task(task)
            
            logger.info(f"ステータス更新: {tracker_id} -> {status.value}")
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"ステータス更新失敗: {e}")
    
    def update_component_status(self, tracker_id: str, component: str, 
                              status: ComponentStatus) -> TaskRecord:
        """コンポーネントステータス更新"""
        try:
            task = self.get_task(tracker_id)
            if not task:
                raise ProgressTrackerError(f"タスクが見つかりません: {tracker_id}")
            
            task.update_component(component, status)
            self.client.update_task(task)
            
            logger.info(f"コンポーネント更新: {tracker_id}.{component} -> {status.value}")
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"コンポーネント更新失敗: {e}")
    
    def get_all_tasks(self) -> List[TaskRecord]:
        """全タスク取得"""
        try:
            return self.client.get_all_tasks()
        except Exception as e:
            raise ProgressTrackerError(f"全タスク取得失敗: {e}")
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskRecord]:
        """ステータス別タスク取得"""
        try:
            all_tasks = self.get_all_tasks()
            return [task for task in all_tasks if task.status == status]
        except Exception as e:
            raise ProgressTrackerError(f"ステータス別タスク取得失敗: {e}")
    
    def get_active_tasks(self) -> List[TaskRecord]:
        """アクティブなタスク取得（着手中～品質チェック）"""
        try:
            active_statuses = [
                TaskStatus.IN_PROGRESS,
                TaskStatus.IMPLEMENTATION_DONE,
                TaskStatus.OPERATION_CHECK,
                TaskStatus.UNIT_TEST,
                TaskStatus.QUALITY_CHECK,
                TaskStatus.EXTRACTION_PIPELINE
            ]
            
            all_tasks = self.get_all_tasks()
            return [task for task in all_tasks if task.status in active_statuses]
            
        except Exception as e:
            raise ProgressTrackerError(f"アクティブタスク取得失敗: {e}")
    
    def progress_workflow_step(self, tracker_id: str, step: str, 
                             success: bool = True) -> TaskRecord:
        """ワークフローステップ進行"""
        try:
            task = self.get_task(tracker_id)
            if not task:
                raise ProgressTrackerError(f"タスクが見つかりません: {tracker_id}")
            
            # ステップに応じてステータス更新
            status_map = {
                'implementation': TaskStatus.IMPLEMENTATION_DONE,
                'operation_check': TaskStatus.OPERATION_CHECK,
                'unit_test': TaskStatus.UNIT_TEST,
                'quality_check': TaskStatus.QUALITY_CHECK,
                'extraction_pipeline': TaskStatus.EXTRACTION_PIPELINE,
                'release': TaskStatus.RELEASE
            }
            
            # コンポーネントステータス更新
            component_status = ComponentStatus.COMPLETED if success else ComponentStatus.FAILED
            
            if step in ['operation_check', 'unit_test', 'quality_evaluation', 
                       'integration_script', 'dashboard_generation', 'extraction_pipeline']:
                task.update_component(step, component_status)
            
            # メインステータス更新
            if step in status_map:
                task.update_status(status_map[step])
            
            self.client.update_task(task)
            
            logger.info(f"ワークフローステップ進行: {tracker_id}.{step} -> {'成功' if success else '失敗'}")
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"ワークフローステップ進行失敗: {e}")
    
    def bulk_update_from_workflow(self, tracker_id: str, workflow_results: Dict[str, bool]) -> TaskRecord:
        """ワークフロー結果の一括更新"""
        try:
            task = self.get_task(tracker_id)
            if not task:
                # 新規タスクとして作成
                task = self.create_task(tracker_id, f"自動生成: {tracker_id}")
            
            # 各コンポーネントの結果を反映
            component_map = {
                'extraction_pipeline': 'extraction_pipeline',
                'quality_evaluation': 'quality_evaluation',
                'dashboard_generation': 'dashboard_generation',
                'unit_test': 'unit_test',
                'integration_script': 'integration_script',
                'operation_check': 'operation_check'
            }
            
            for workflow_key, component_field in component_map.items():
                if workflow_key in workflow_results:
                    success = workflow_results[workflow_key]
                    component_status = ComponentStatus.COMPLETED if success else ComponentStatus.FAILED
                    task.update_component(component_field, component_status)
            
            # 全体ステータス判定
            all_success = all(workflow_results.values())
            if all_success:
                task.update_status(TaskStatus.QUALITY_CHECK)
            else:
                task.update_status(TaskStatus.IN_PROGRESS)
            
            self.client.update_task(task)
            
            logger.info(f"ワークフロー結果一括更新: {tracker_id}")
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"ワークフロー結果一括更新失敗: {e}")
    
    def generate_status_report(self) -> Dict[str, Any]:
        """ステータスレポート生成"""
        try:
            all_tasks = self.get_all_tasks()
            
            # ステータス別集計
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = len([t for t in all_tasks if t.status == status])
            
            # アクティブタスク
            active_tasks = self.get_active_tasks()
            
            # 完了率計算
            total_tasks = len(all_tasks)
            completed_tasks = len([t for t in all_tasks if t.status == TaskStatus.RELEASE])
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # 最近の活動
            recent_tasks = sorted(all_tasks, key=lambda t: t.updated_date or datetime.min, reverse=True)[:5]
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'active_tasks': len(active_tasks),
                    'completion_rate': completion_rate
                },
                'status_breakdown': status_counts,
                'active_task_ids': [t.tracker_id for t in active_tasks],
                'recent_activity': [
                    {
                        'tracker_id': t.tracker_id,
                        'status': t.status.value,
                        'updated_date': t.updated_date.isoformat() if t.updated_date else None
                    }
                    for t in recent_tasks
                ]
            }
            
            return report
            
        except Exception as e:
            raise ProgressTrackerError(f"ステータスレポート生成失敗: {e}")
    
    def print_status_summary(self) -> None:
        """ステータスサマリー表示"""
        try:
            report = self.generate_status_report()
            
            print(f"\n{'='*60}")
            print(f"📊 進捗管理システム - ステータスサマリー")
            print(f"{'='*60}")
            print(f"📅 更新日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📋 総タスク数: {report['summary']['total_tasks']}")
            print(f"✅ 完了タスク: {report['summary']['completed_tasks']}")
            print(f"🔄 アクティブタスク: {report['summary']['active_tasks']}")
            print(f"📈 完了率: {report['summary']['completion_rate']:.1f}%")
            
            print(f"\n🎯 ステータス内訳:")
            for status, count in report['status_breakdown'].items():
                if count > 0:
                    print(f"  {status}: {count}件")
            
            if report['active_task_ids']:
                print(f"\n🔄 アクティブタスク:")
                for task_id in report['active_task_ids']:
                    print(f"  - {task_id}")
            
            print(f"\n🕒 最近の活動:")
            for activity in report['recent_activity']:
                updated = activity['updated_date']
                if updated:
                    updated_str = datetime.fromisoformat(updated).strftime('%m-%d %H:%M')
                else:
                    updated_str = "未更新"
                print(f"  {activity['tracker_id']}: {activity['status']} ({updated_str})")
            
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"ステータスサマリー表示エラー: {e}")
            print(f"❌ ステータスサマリー表示エラー: {e}")
    
    def update_task_metrics(self, tracker_id: str, stats_dict: Dict[str, float]) -> TaskRecord:
        """統計指標データ更新 (旧10指標システムから新統計指標システムへ移行)"""
        try:
            task = self.get_task(tracker_id)
            if not task:
                raise ProgressTrackerError(f"タスクが見つかりません: {tracker_id}")
            
            # 統計辞書からStatisticalRecordを作成
            statistical_record = StatisticalRecord(
                current_score=stats_dict.get('current_score'),
                baseline_score=stats_dict.get('baseline_score'),
                p_value=stats_dict.get('p_value'),
                cohens_d=stats_dict.get('cohens_d'),
                improvement_rate=stats_dict.get('improvement_rate'),
                statistical_significance=stats_dict.get('statistical_significance')
            )
            
            task.statistical_record = statistical_record
            task.updated_date = datetime.now()
            
            self.client.update_task(task)
            
            logger.info(f"統計指標更新: {tracker_id}")
            return task
            
        except Exception as e:
            raise ProgressTrackerError(f"統計指標更新失敗: {e}")
    
    def update_from_quality_checker_results(self, tracker_id: str, quality_results: Dict[str, Any]) -> TaskRecord:
        """統合品質チェッカー結果から自動更新"""
        try:
            # 品質チェッカーの結果形式を解析
            stats_dict = {}
            
            if 'statistical_analysis' in quality_results:
                stats_analysis = quality_results['statistical_analysis']
                stats_dict = {
                    'current_score': stats_analysis.get('current_score'),
                    'baseline_score': stats_analysis.get('baseline_score'),
                    'p_value': stats_analysis.get('p_value'),
                    'cohens_d': stats_analysis.get('cohens_d'),
                    'improvement_rate': stats_analysis.get('improvement_rate'),
                    'statistical_significance': stats_analysis.get('statistical_significance')
                }
            
            return self.update_task_metrics(tracker_id, stats_dict)
            
        except Exception as e:
            raise ProgressTrackerError(f"品質チェッカー結果更新失敗: {e}")