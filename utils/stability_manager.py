"""
安定性管理システム - v0.0.43
Windowsハングアップ防止とシステム安定性確保機能
"""

import psutil
import time
import threading
import logging
import gc
import torch
from typing import Dict, Optional, Callable
from dataclasses import dataclass

@dataclass
class SystemStatus:
    """システム状態データクラス"""
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_usage_percent: float
    processing_time: float
    is_stable: bool
    warnings: list

class StabilityManager:
    """
    システム安定性管理クラス
    メモリ監視、タイムアウト保護、段階的品質低下機能
    """
    
    def __init__(self,
                 memory_limit_mb: int = 2048,
                 gpu_memory_limit_mb: int = 8192,
                 cpu_limit_percent: float = 90.0,
                 timeout_seconds: int = 300,
                 monitoring_interval: float = 5.0):
        """
        初期化
        
        Args:
            memory_limit_mb: メモリ使用量上限(MB)
            gpu_memory_limit_mb: GPU メモリ上限(MB)
            cpu_limit_percent: CPU使用率上限(%)
            timeout_seconds: 処理タイムアウト(秒)
            monitoring_interval: 監視間隔(秒)
        """
        self.memory_limit = memory_limit_mb
        self.gpu_memory_limit = gpu_memory_limit_mb
        self.cpu_limit = cpu_limit_percent
        self.timeout = timeout_seconds
        self.monitoring_interval = monitoring_interval
        
        # 監視状態
        self.monitoring_active = False
        self.current_task = None
        self.task_start_time = None
        self.monitor_thread = None
        self.emergency_stop_flag = False
        
        # 統計情報
        self.memory_peak = 0
        self.gpu_memory_peak = 0
        self.processing_times = []
        
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得(MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_usage(self) -> float:
        """現在のGPUメモリ使用量を取得(MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_cpu_usage(self) -> float:
        """現在のCPU使用率を取得(%)"""
        return psutil.cpu_percent()
    
    def get_system_status(self) -> SystemStatus:
        """現在のシステム状態を取得"""
        memory_mb = self.get_memory_usage()
        gpu_memory_mb = self.get_gpu_memory_usage()
        cpu_percent = self.get_cpu_usage()
        
        # 処理時間計算
        processing_time = 0.0
        if self.task_start_time:
            processing_time = time.time() - self.task_start_time
        
        # 警告チェック
        warnings = []
        is_stable = True
        
        if memory_mb > self.memory_limit * 0.8:
            warnings.append(f"メモリ使用量警告: {memory_mb:.1f}MB")
            if memory_mb > self.memory_limit:
                is_stable = False
        
        if gpu_memory_mb > self.gpu_memory_limit * 0.8:
            warnings.append(f"GPU メモリ警告: {gpu_memory_mb:.1f}MB")
            if gpu_memory_mb > self.gpu_memory_limit:
                is_stable = False
        
        if cpu_percent > self.cpu_limit:
            warnings.append(f"CPU使用率警告: {cpu_percent:.1f}%")
            is_stable = False
        
        if processing_time > self.timeout * 0.8:
            warnings.append(f"処理時間警告: {processing_time:.1f}秒")
            if processing_time > self.timeout:
                is_stable = False
        
        return SystemStatus(
            memory_usage_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_usage_percent=cpu_percent,
            processing_time=processing_time,
            is_stable=is_stable,
            warnings=warnings
        )
    
    def force_memory_cleanup(self):
        """強制メモリクリーンアップ"""
        self.logger.warning("強制メモリクリーンアップ実行")
        
        # Python ガベージコレクション
        gc.collect()
        
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 統計更新
        time.sleep(1)  # クリーンアップ処理完了待ち
        memory_after = self.get_memory_usage()
        self.logger.info(f"メモリクリーンアップ後: {memory_after:.1f}MB")
    
    def _monitoring_loop(self):
        """監視ループ（別スレッドで実行）"""
        while self.monitoring_active and not self.emergency_stop_flag:
            try:
                status = self.get_system_status()
                
                # 統計更新
                self.memory_peak = max(self.memory_peak, status.memory_usage_mb)
                self.gpu_memory_peak = max(self.gpu_memory_peak, status.gpu_memory_mb)
                
                # 警告ログ出力
                for warning in status.warnings:
                    self.logger.warning(warning)
                
                # 緊急停止判定
                if not status.is_stable:
                    self.logger.error("システム不安定: 緊急停止実行")
                    self.emergency_stop_flag = True
                    
                    # 緊急メモリクリーンアップ
                    self.force_memory_cleanup()
                    break
                
                # 予防的メモリクリーンアップ
                if (status.memory_usage_mb > self.memory_limit * 0.7 or 
                    status.gpu_memory_mb > self.gpu_memory_limit * 0.7):
                    self.force_memory_cleanup()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"監視エラー: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_monitoring(self, task_name: str = "unknown"):
        """監視開始"""
        self.current_task = task_name
        self.task_start_time = time.time()
        self.emergency_stop_flag = False
        self.monitoring_active = True
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"監視開始: {task_name}")
    
    def stop_monitoring(self) -> Dict:
        """監視停止と統計取得"""
        self.monitoring_active = False
        
        if self.task_start_time:
            total_time = time.time() - self.task_start_time
            self.processing_times.append(total_time)
        
        # スレッド終了待ち
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # 最終統計
        final_status = self.get_system_status()
        statistics = {
            'task_name': self.current_task,
            'total_processing_time': total_time if self.task_start_time else 0,
            'memory_peak_mb': self.memory_peak,
            'gpu_memory_peak_mb': self.gpu_memory_peak,
            'final_status': final_status,
            'emergency_stopped': self.emergency_stop_flag,
            'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        }
        
        self.logger.info(f"監視停止: {self.current_task}, 統計: {statistics}")
        
        # リセット
        self.current_task = None
        self.task_start_time = None
        
        return statistics
    
    def check_emergency_stop(self) -> bool:
        """緊急停止フラグチェック"""
        return self.emergency_stop_flag
    
    def safe_execute(self, func: Callable, *args, **kwargs):
        """
        安全実行ラッパー
        
        Args:
            func: 実行する関数
            *args, **kwargs: 関数の引数
            
        Returns:
            実行結果または None（エラー時）
        """
        task_name = f"{func.__name__}"
        self.start_monitoring(task_name)
        
        try:
            # 実行前チェック
            initial_status = self.get_system_status()
            if not initial_status.is_stable:
                self.logger.error(f"実行前システム不安定: {initial_status.warnings}")
                return None
            
            # 関数実行
            result = func(*args, **kwargs)
            
            # 緊急停止チェック
            if self.check_emergency_stop():
                self.logger.error("実行中に緊急停止")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"実行エラー: {e}")
            return None
            
        finally:
            stats = self.stop_monitoring()
            
            # 最終クリーンアップ
            self.force_memory_cleanup()
    
    def get_performance_report(self) -> Dict:
        """パフォーマンスレポート生成"""
        return {
            'memory_peak_mb': self.memory_peak,
            'gpu_memory_peak_mb': self.gpu_memory_peak,
            'total_tasks': len(self.processing_times),
            'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'min_processing_time': min(self.processing_times) if self.processing_times else 0,
            'memory_limit_mb': self.memory_limit,
            'gpu_memory_limit_mb': self.gpu_memory_limit,
            'timeout_seconds': self.timeout
        }