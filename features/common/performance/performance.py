#!/usr/bin/env python3
"""
Performance Monitoring Utilities
Extracted from original sam_yolo_character_segment.py
"""

import time
import psutil
import torch
import gc
from typing import Dict, Optional


class PerformanceMonitor:
    """
    処理パフォーマンスを監視するクラス
    """
    
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.memory_usage = {}
        self.current_stage = None
        
    def start_monitoring(self):
        """モニタリング開始"""
        self.start_time = time.time()
        self.log_system_info()
        
    def start_stage(self, stage_name: str):
        """処理段階の開始"""
        if self.current_stage:
            self.end_stage()
        
        self.current_stage = stage_name
        self.stage_times[stage_name] = time.time()
        
        # メモリ使用量記録
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0
        
        print(f"🔄 開始: {stage_name} (RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB)")
        
    def end_stage(self):
        """処理段階の終了"""
        if not self.current_stage:
            return
            
        elapsed = time.time() - self.stage_times[self.current_stage]
        
        # メモリ使用量記録
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0
        
        print(f"✅ 完了: {self.current_stage} ({elapsed:.2f}秒, RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB)")
        
        self.stage_times[self.current_stage] = elapsed
        self.current_stage = None
        
        # ガベージコレクション実行
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_gpu_memory(self) -> float:
        """GPU メモリ使用量を取得"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0
    
    def log_system_info(self):
        """システム情報をログ出力"""
        print("🖥️ システム情報:")
        
        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   CPU: {cpu_count}コア, 使用率: {cpu_percent}%")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024 / 1024 / 1024
        memory_available_gb = memory.available / 1024 / 1024 / 1024
        print(f"   メモリ: {memory_available_gb:.1f}GB / {memory_gb:.1f}GB 利用可能")
        
        # GPU情報
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024 / 1024 / 1024
            print(f"   GPU: {gpu_name} ({gpu_count}基), VRAM: {gpu_memory_total:.1f}GB")
        else:
            print("   GPU: 利用不可 (CPUモード)")
    
    def get_total_time(self) -> float:
        """総処理時間を取得"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_stage_summary(self) -> Dict[str, float]:
        """各段階の処理時間サマリを取得"""
        return self.stage_times.copy()
    
    def print_summary(self):
        """処理時間サマリを出力"""
        total_time = self.get_total_time()
        
        print("\n📊 処理時間サマリ:")
        print(f"   総処理時間: {total_time:.2f}秒")
        
        for stage, duration in self.stage_times.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"   {stage}: {duration:.2f}秒 ({percentage:.1f}%)")
        
        # メモリ使用量
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0
        
        print(f"\n💾 最終メモリ使用量:")
        print(f"   RAM: {memory_mb:.1f}MB")
        if torch.cuda.is_available():
            print(f"   GPU: {gpu_memory:.1f}MB")


class ResourceManager:
    """
    システムリソース管理クラス
    """
    
    @staticmethod
    def check_available_memory() -> Dict[str, float]:
        """利用可能メモリをチェック"""
        memory = psutil.virtual_memory()
        
        result = {
            'total_gb': memory.total / 1024 / 1024 / 1024,
            'available_gb': memory.available / 1024 / 1024 / 1024,
            'percent': memory.percent
        }
        
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            gpu_memory_free = gpu_memory_total - gpu_memory_allocated
            
            result.update({
                'gpu_total_gb': gpu_memory_total,
                'gpu_allocated_gb': gpu_memory_allocated,
                'gpu_free_gb': gpu_memory_free
            })
        
        return result
    
    @staticmethod
    def cleanup_memory():
        """メモリクリーンアップ"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def check_system_requirements() -> bool:
        """システム要件をチェック"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024 / 1024 / 1024
        
        # 最低4GB必要
        if available_gb < 4.0:
            print(f"❌ メモリ不足: {available_gb:.1f}GB (最低4GB必要)")
            return False
        
        print(f"✅ システム要件OK: {available_gb:.1f}GB利用可能")
        return True


if __name__ == "__main__":
    # Test performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    monitor.start_stage("Test Stage 1")
    time.sleep(1)
    monitor.end_stage()
    
    monitor.start_stage("Test Stage 2")
    time.sleep(0.5)
    monitor.end_stage()
    
    monitor.print_summary()
    
    # Test resource manager
    print("\n🔍 Resource Check:")
    resources = ResourceManager.check_available_memory()
    for key, value in resources.items():
        print(f"   {key}: {value}")
    
    print(f"✅ Performance module test completed")