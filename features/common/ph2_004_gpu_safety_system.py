#!/usr/bin/env python3
"""
PH2-004 GPU安全策システム: GPU使用中チェック・競合回避・軽量モード実装

ユーザーコメント対応:
- ボトルネック問題: 軽量監視モード実装
- リスク問題: GPU使用中チェック・アクティブプロセス検出
"""

import torch

import gc
import logging
import os
import psutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SafetyStatus:
    """安全性ステータス"""
    gpu_in_use: bool = False
    active_processes: List[str] = None
    safe_for_optimization: bool = True
    warning_message: Optional[str] = None
    
    def __post_init__(self):
        if self.active_processes is None:
            self.active_processes = []


class GPUSafetyChecker:
    """GPU使用中チェック・安全性確認システム"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.monitored_processes = {
            'extract_character.py',
            'sam_yolo_character_segment.py', 
            'run_auto_pipeline.py',
            'python', 'python3'  # Python プロセス一般
        }
        self.gpu_memory_threshold = 100  # MB, GPU使用判定閾値
        
    def check_gpu_usage(self) -> Dict[str, any]:
        """詳細なGPU使用状況チェック"""
        if not self.gpu_available:
            return {"in_use": False, "memory_allocated": 0, "memory_reserved": 0}
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            
            # GPU使用判定: 割り当てメモリが閾値超過
            gpu_in_use = memory_allocated > self.gpu_memory_threshold
            
            return {
                "in_use": gpu_in_use,
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
            }
            
        except Exception as e:
            logger.warning(f"GPU使用状況チェックエラー: {e}")
            return {"in_use": True, "error": str(e)}  # エラー時は安全側で判定
    
    def detect_active_ml_processes(self) -> List[Dict[str, any]]:
        """ML関連アクティブプロセス検出"""
        active_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name']
                    cmdline = ' '.join(proc_info['cmdline'] or [])
                    
                    # ML関連プロセス判定
                    is_ml_process = any(
                        keyword in cmdline.lower() or keyword in proc_name.lower()
                        for keyword in self.monitored_processes
                    )
                    
                    if is_ml_process and proc_info['memory_percent'] > 1.0:  # 1%以上のメモリ使用
                        active_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_name,
                            'cmdline': cmdline,
                            'memory_percent': proc_info['memory_percent'],
                            'cpu_percent': proc_info['cpu_percent'],
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"プロセス検出エラー: {e}")
        
        return active_processes
    
    def check_extraction_conflicts(self) -> List[str]:
        """キャラクター抽出プロセス競合チェック"""
        conflicts = []
        active_processes = self.detect_active_ml_processes()
        
        extraction_keywords = [
            'extract_character', 'sam_yolo', 'character_segment',
            'run_auto_pipeline', 'batch_extraction'
        ]
        
        for proc in active_processes:
            cmdline = proc['cmdline'].lower()
            if any(keyword in cmdline for keyword in extraction_keywords):
                conflicts.append(f"PID {proc['pid']}: {proc['name']} - {proc['memory_percent']:.1f}% メモリ使用")
        
        return conflicts
    
    def get_comprehensive_safety_status(self) -> SafetyStatus:
        """包括的安全性ステータス取得"""
        gpu_status = self.check_gpu_usage()
        active_processes = self.detect_active_ml_processes()
        conflicts = self.check_extraction_conflicts()
        
        gpu_in_use = gpu_status.get("in_use", False)
        has_conflicts = len(conflicts) > 0
        
        # 安全性判定
        safe_for_optimization = not (gpu_in_use or has_conflicts)
        
        # 警告メッセージ生成
        warning_message = None
        if gpu_in_use:
            allocated = gpu_status.get("memory_allocated", 0)
            warning_message = f"GPU使用中 ({allocated:.1f}MB割り当て済み) - 最適化危険"
        elif has_conflicts:
            warning_message = f"抽出プロセス競合検出 ({len(conflicts)}件) - 最適化危険"
        
        return SafetyStatus(
            gpu_in_use=gpu_in_use,
            active_processes=[f"{p['name']} (PID {p['pid']})" for p in active_processes],
            safe_for_optimization=safe_for_optimization,
            warning_message=warning_message
        )


class LightweightResourceMonitor:
    """軽量リソース監視システム (ボトルネック対策)"""
    
    def __init__(self, check_interval: float = 10.0):
        """
        Args:
            check_interval: チェック間隔（秒）- デフォルト10秒（従来2秒から軽量化）
        """
        self.check_interval = check_interval
        self.safety_checker = GPUSafetyChecker()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_lightweight_monitoring(self) -> bool:
        """軽量監視開始"""
        if self.is_monitoring:
            return False
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._lightweight_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"軽量リソース監視開始 (間隔: {self.check_interval}秒)")
        return True
    
    def stop_monitoring(self) -> bool:
        """監視停止"""
        if not self.is_monitoring:
            return False
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval * 2)
        
        logger.info("軽量リソース監視停止")
        return True
    
    def _lightweight_monitor_loop(self):
        """軽量監視ループ"""
        while self.is_monitoring:
            try:
                safety_status = self.safety_checker.get_comprehensive_safety_status()
                
                # 危険状態のみログ出力（軽量化）
                if not safety_status.safe_for_optimization:
                    logger.warning(f"安全性警告: {safety_status.warning_message}")
                
            except Exception as e:
                logger.error(f"軽量監視エラー: {e}")
            
            time.sleep(self.check_interval)


class SafeOptimizationManager:
    """安全な最適化管理システム"""
    
    def __init__(self, force_mode: bool = False):
        """
        Args:
            force_mode: 強制モード（安全チェック無視）
        """
        self.safety_checker = GPUSafetyChecker()
        self.force_mode = force_mode
        
    @contextmanager
    def safe_optimization_context(self, operation_name: str = "optimization"):
        """安全な最適化コンテキスト"""
        logger.info(f"安全な最適化開始: {operation_name}")
        
        # 事前安全性チェック
        if not self.force_mode:
            safety_status = self.safety_checker.get_comprehensive_safety_status()
            
            if not safety_status.safe_for_optimization:
                logger.warning(f"最適化をスキップ: {safety_status.warning_message}")
                yield False  # 最適化実行不可を示す
                return
        
        try:
            yield True  # 最適化実行可能を示す
            logger.info(f"安全な最適化完了: {operation_name}")
        except Exception as e:
            logger.error(f"最適化エラー: {operation_name} - {e}")
            raise
    
    def safe_gpu_cleanup(self) -> bool:
        """安全なGPUクリーンアップ"""
        gpu_status = self.safety_checker.check_gpu_usage()
        
        if not self.force_mode and gpu_status["in_use"]:
            logger.warning(f"GPU使用中のためクリーンアップスキップ ({gpu_status['memory_allocated']:.1f}MB)")
            return False
        
        if not torch.cuda.is_available():
            return False
        
        try:
            before_memory = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after_memory = torch.cuda.memory_allocated() / 1024**2
            
            freed = before_memory - after_memory
            logger.info(f"安全なGPUクリーンアップ完了: {freed:.1f}MB解放")
            return True
            
        except Exception as e:
            logger.error(f"GPUクリーンアップエラー: {e}")
            return False
    
    def safe_memory_cleanup(self) -> bool:
        """安全なメモリクリーンアップ"""
        conflicts = self.safety_checker.check_extraction_conflicts()
        
        if not self.force_mode and conflicts:
            logger.warning(f"抽出プロセス競合のためメモリクリーンアップスキップ ({len(conflicts)}件)")
            return False
        
        try:
            collected = gc.collect()
            logger.info(f"安全なメモリクリーンアップ完了: {collected}オブジェクト回収")
            return True
            
        except Exception as e:
            logger.error(f"メモリクリーンアップエラー: {e}")
            return False
    
    def generate_safety_report(self) -> Dict[str, any]:
        """安全性レポート生成"""
        safety_status = self.safety_checker.get_comprehensive_safety_status()
        gpu_status = self.safety_checker.check_gpu_usage()
        active_processes = self.safety_checker.detect_active_ml_processes()
        conflicts = self.safety_checker.check_extraction_conflicts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "safety_status": {
                "safe_for_optimization": safety_status.safe_for_optimization,
                "gpu_in_use": safety_status.gpu_in_use,
                "warning_message": safety_status.warning_message,
                "active_process_count": len(safety_status.active_processes),
            },
            "gpu_status": gpu_status,
            "active_processes": active_processes,
            "conflicts": conflicts,
            "recommendations": self._generate_safety_recommendations(safety_status, conflicts),
        }
    
    def _generate_safety_recommendations(self, safety_status: SafetyStatus, conflicts: List[str]) -> List[str]:
        """安全性改善推奨事項生成"""
        recommendations = []
        
        if safety_status.gpu_in_use:
            recommendations.append("GPU使用中 - 抽出処理完了後に最適化実行を推奨")
        
        if conflicts:
            recommendations.append(f"競合プロセス {len(conflicts)}件検出 - プロセス完了待ちを推奨")
        
        if safety_status.safe_for_optimization:
            recommendations.append("安全な最適化実行が可能 - 積極的なリソース管理を推奨")
        else:
            recommendations.append("最適化を延期し、軽量監視モードでの待機を推奨")
        
        return recommendations


# 便利な関数群
def check_gpu_safety() -> bool:
    """GPU安全性チェック"""
    checker = GPUSafetyChecker()
    status = checker.get_comprehensive_safety_status()
    return status.safe_for_optimization


def safe_resource_optimization(force: bool = False) -> bool:
    """安全なリソース最適化"""
    manager = SafeOptimizationManager(force_mode=force)
    
    with manager.safe_optimization_context("resource_optimization") as can_optimize:
        if not can_optimize:
            return False
        
        # GPU クリーンアップ
        gpu_cleaned = manager.safe_gpu_cleanup()
        
        # メモリクリーンアップ  
        memory_cleaned = manager.safe_memory_cleanup()
        
        return gpu_cleaned or memory_cleaned


def create_lightweight_monitor(interval: float = 10.0) -> LightweightResourceMonitor:
    """軽量監視システム作成"""
    return LightweightResourceMonitor(check_interval=interval)


if __name__ == "__main__":
    # テスト実行
    print("🛡️ PH2-004 GPU安全策システムテスト")
    print("=" * 60)
    
    # 安全性チェック
    checker = GPUSafetyChecker()
    safety_status = checker.get_comprehensive_safety_status()
    
    print(f"GPU使用中: {'はい' if safety_status.gpu_in_use else 'いいえ'}")
    print(f"最適化安全: {'はい' if safety_status.safe_for_optimization else 'いいえ'}")
    print(f"アクティブプロセス: {len(safety_status.active_processes)}件")
    
    if safety_status.warning_message:
        print(f"⚠️ 警告: {safety_status.warning_message}")
    
    # 安全な最適化テスト
    print(f"\n🔧 安全な最適化テスト:")
    optimization_success = safe_resource_optimization()
    print(f"最適化実行: {'成功' if optimization_success else '安全のためスキップ'}")
    
    # 軽量監視テスト
    print(f"\n📊 軽量監視システムテスト:")
    monitor = create_lightweight_monitor(interval=2.0)
    monitor.start_lightweight_monitoring()
    
    print("2秒間の軽量監視中...")
    time.sleep(3)
    
    monitor.stop_monitoring()
    print("✅ 軽量監視停止")
    
    print("\n✅ PH2-004 GPU安全策システム初期化完了")