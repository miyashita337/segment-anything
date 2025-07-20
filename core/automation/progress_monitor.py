#!/usr/bin/env python3
"""
進捗監視システム
5分毎の自動ステータスレポート生成
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from .background_executor import get_task_manager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProgressMonitor:
    """進捗監視・レポート生成クラス"""
    
    def __init__(self, report_interval: int = 300):
        """
        初期化
        
        Args:
            report_interval: レポート生成間隔（秒）デフォルト5分
        """
        self.report_interval = report_interval
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.task_manager = get_task_manager()
        self.report_callbacks = []
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'running_tasks': 0,
            'processing_times': [],
            'error_types': {}
        }
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring:
            logger.warning("Progress monitoring is already running")
            return
            
        self.monitoring = True
        self.start_time = datetime.now()
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ProgressMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Progress monitoring started")
        
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
            
        logger.info("Progress monitoring stopped")
        
    def add_report_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """レポートコールバック追加"""
        self.report_callbacks.append(callback)
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """現在のメトリクス取得"""
        with self.lock:
            return self.metrics.copy()
            
    def _monitor_loop(self):
        """監視ループ"""
        last_report_time = time.time()
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                # タスクステータス更新
                self._update_task_metrics()
                
                # レポート生成タイミング
                if current_time - last_report_time >= self.report_interval:
                    self._generate_report()
                    last_report_time = current_time
                    
                # 1秒待機
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                
    def _update_task_metrics(self):
        """タスクメトリクス更新"""
        all_tasks = self.task_manager.executor.get_all_status()
        
        with self.lock:
            self.metrics['total_tasks'] = len(all_tasks)
            self.metrics['completed_tasks'] = sum(
                1 for t in all_tasks.values() if t['status'] == 'completed'
            )
            self.metrics['failed_tasks'] = sum(
                1 for t in all_tasks.values() if t['status'] == 'failed'
            )
            self.metrics['running_tasks'] = sum(
                1 for t in all_tasks.values() if t['status'] == 'running'
            )
            
            # エラータイプ集計
            for task in all_tasks.values():
                if task['status'] == 'failed' and 'error' in task:
                    error_type = type(task.get('error', '')).__name__
                    self.metrics['error_types'][error_type] = \
                        self.metrics['error_types'].get(error_type, 0) + 1
                        
            # 処理時間計算
            for task in all_tasks.values():
                if task['status'] == 'completed' and 'started_at' in task and 'completed_at' in task:
                    duration = (task['completed_at'] - task['started_at']).total_seconds()
                    self.metrics['processing_times'].append(duration)
                    
    def _generate_report(self):
        """レポート生成"""
        report = self._create_report()
        
        # コンソール出力
        self._print_report(report)
        
        # ファイル保存
        self._save_report(report)
        
        # コールバック実行
        for callback in self.report_callbacks:
            try:
                callback(report)
            except Exception as e:
                logger.error(f"Report callback error: {e}")
                
    def _create_report(self) -> Dict[str, Any]:
        """レポート作成"""
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time if self.start_time else timedelta(0)
        
        with self.lock:
            metrics = self.metrics.copy()
            
        # 平均処理時間計算
        avg_processing_time = 0
        if metrics['processing_times']:
            avg_processing_time = sum(metrics['processing_times']) / len(metrics['processing_times'])
            
        # 成功率計算
        success_rate = 0
        if metrics['total_tasks'] > 0:
            success_rate = (metrics['completed_tasks'] / metrics['total_tasks']) * 100
            
        report = {
            'timestamp': current_time.isoformat(),
            'elapsed_time': str(elapsed_time),
            'elapsed_minutes': elapsed_time.total_seconds() / 60,
            'metrics': {
                'total_tasks': metrics['total_tasks'],
                'completed_tasks': metrics['completed_tasks'],
                'failed_tasks': metrics['failed_tasks'],
                'running_tasks': metrics['running_tasks'],
                'pending_tasks': metrics['total_tasks'] - metrics['completed_tasks'] - 
                               metrics['failed_tasks'] - metrics['running_tasks'],
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time,
                'error_types': metrics['error_types']
            },
            'system_info': self._get_system_info()
        }
        
        return report
        
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            info = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
            
            # GPU情報（利用可能な場合）
            try:
                import torch
                if torch.cuda.is_available():
                    info['gpu_available'] = True
                    info['gpu_count'] = torch.cuda.device_count()
                    info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                else:
                    info['gpu_available'] = False
            except:
                info['gpu_available'] = False
                
            return info
            
        except Exception as e:
            logger.error(f"System info error: {e}")
            return {}
            
    def _print_report(self, report: Dict[str, Any]):
        """レポートコンソール出力"""
        print("\n" + "="*60)
        print(f"📊 進捗レポート - {report['timestamp']}")
        print("="*60)
        
        metrics = report['metrics']
        print(f"\n⏱️ 経過時間: {report['elapsed_minutes']:.1f}分")
        print(f"\n📈 タスク統計:")
        print(f"   総タスク数: {metrics['total_tasks']}")
        print(f"   完了: {metrics['completed_tasks']} ✅")
        print(f"   実行中: {metrics['running_tasks']} 🔄")
        print(f"   待機中: {metrics['pending_tasks']} ⏳")
        print(f"   失敗: {metrics['failed_tasks']} ❌")
        print(f"   成功率: {metrics['success_rate']:.1f}%")
        print(f"   平均処理時間: {metrics['avg_processing_time']:.1f}秒")
        
        if metrics['error_types']:
            print(f"\n⚠️ エラータイプ:")
            for error_type, count in metrics['error_types'].items():
                print(f"   {error_type}: {count}件")
                
        sys_info = report.get('system_info', {})
        if sys_info:
            print(f"\n💻 システム状況:")
            print(f"   CPU使用率: {sys_info.get('cpu_usage', 0):.1f}%")
            print(f"   メモリ使用率: {sys_info.get('memory_usage', 0):.1f}%")
            print(f"   利用可能メモリ: {sys_info.get('memory_available_gb', 0):.1f}GB")
            
            if sys_info.get('gpu_available'):
                print(f"   GPU数: {sys_info.get('gpu_count', 0)}")
                print(f"   GPU使用メモリ: {sys_info.get('gpu_memory_allocated_gb', 0):.1f}GB")
                
        print("="*60 + "\n")
        
    def _save_report(self, report: Dict[str, Any]):
        """レポートファイル保存"""
        report_dir = Path("temp/logs/progress_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # タイムスタンプベースのファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"progress_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


# シングルトンインスタンス
_progress_monitor = None


def get_progress_monitor() -> ProgressMonitor:
    """進捗モニターのシングルトンインスタンス取得"""
    global _progress_monitor
    if _progress_monitor is None:
        _progress_monitor = ProgressMonitor()
    return _progress_monitor