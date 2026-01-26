"""
メモリ使用最適化システム
大規模データセット対応のためのメモリ効率化
"""

import numpy as np
import torch

import gc
import logging
import psutil
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """メモリ監視クラス"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self._monitoring = False
        self._monitor_thread = None

    def get_memory_usage(self) -> Dict[str, float]:
        """現在のメモリ使用量を取得"""
        memory_info = self.process.memory_info()

        result = {
            "ram_mb": memory_info.rss / (1024 * 1024),
            "virtual_mb": memory_info.vms / (1024 * 1024),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        }

        # GPU メモリ情報
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

            result.update(
                {
                    "gpu_allocated_mb": gpu_memory,
                    "gpu_reserved_mb": gpu_reserved,
                    "gpu_total_mb": gpu_total,
                    "gpu_free_mb": gpu_total - gpu_reserved,
                }
            )

        return result

    def start_monitoring(self, interval: float = 1.0):
        """メモリ監視を開始"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        logger.info("📊 メモリ監視開始")

    def stop_monitoring(self):
        """メモリ監視を停止"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("📊 メモリ監視停止")

    def _monitor_loop(self, interval: float):
        """監視ループ"""
        while self._monitoring:
            current = self.get_memory_usage()

            # ピークメモリ更新
            if current["ram_mb"] > self.peak_memory["ram_mb"]:
                self.peak_memory = current.copy()

            # メモリ不足警告
            if current["available_mb"] < 1000:  # 1GB未満
                logger.warning(f"⚠️ RAM不足: 残り {current['available_mb']:.1f}MB")

            if "gpu_free_mb" in current and current["gpu_free_mb"] < 500:  # 500MB未満
                logger.warning(f"⚠️ GPU メモリ不足: 残り {current['gpu_free_mb']:.1f}MB")

            time.sleep(interval)

    def get_peak_memory(self) -> Dict[str, float]:
        """ピークメモリ使用量を取得"""
        return self.peak_memory.copy()


class MemoryOptimizer:
    """メモリ最適化管理クラス"""

    def __init__(self, enable_aggressive_gc: bool = True):
        self.enable_aggressive_gc = enable_aggressive_gc
        self.monitor = MemoryMonitor()
        self._optimization_stats = {"gc_calls": 0, "memory_freed_mb": 0.0, "cache_clears": 0}

    def optimize_memory(self, force: bool = False) -> Dict[str, float]:
        """メモリ最適化を実行"""
        before_memory = self.monitor.get_memory_usage()

        # Python ガベージコレクション
        if self.enable_aggressive_gc or force:
            self._run_garbage_collection()

        # PyTorch キャッシュクリア
        if torch.cuda.is_available():
            self._clear_cuda_cache()

        # NumPy メモリ最適化
        self._optimize_numpy_memory()

        after_memory = self.monitor.get_memory_usage()

        # 統計更新
        freed_ram = before_memory["ram_mb"] - after_memory["ram_mb"]
        self._optimization_stats["memory_freed_mb"] += max(0, freed_ram)

        logger.info(f"🧹 メモリ最適化完了: RAM {freed_ram:+.1f}MB")

        return {"before": before_memory, "after": after_memory, "freed_mb": freed_ram}

    def _run_garbage_collection(self):
        """ガベージコレクション実行"""
        collected = gc.collect()
        self._optimization_stats["gc_calls"] += 1
        logger.debug(f"GC実行: {collected}個のオブジェクトを回収")

    def _clear_cuda_cache(self):
        """CUDA キャッシュクリア"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._optimization_stats["cache_clears"] += 1
            logger.debug("CUDA キャッシュクリア完了")

    def _optimize_numpy_memory(self):
        """NumPy メモリ最適化"""
        # NumPy の内部メモリプールを最適化
        # 実際の実装では、大きな配列の削除後にメモリを解放
        pass

    @contextmanager
    def memory_context(self, auto_optimize: bool = True):
        """メモリ管理コンテキスト"""
        initial_memory = self.monitor.get_memory_usage()
        logger.info(f"📊 メモリコンテキスト開始: RAM {initial_memory['ram_mb']:.1f}MB")

        try:
            yield self
        finally:
            if auto_optimize:
                self.optimize_memory()

            final_memory = self.monitor.get_memory_usage()
            logger.info(f"📊 メモリコンテキスト終了: RAM {final_memory['ram_mb']:.1f}MB")

    def memory_efficient_decorator(self, auto_optimize: bool = True):
        """メモリ効率デコレータ"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.memory_context(auto_optimize=auto_optimize):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計を取得"""
        return {
            "optimization_stats": self._optimization_stats.copy(),
            "current_memory": self.monitor.get_memory_usage(),
            "peak_memory": self.monitor.get_peak_memory(),
        }


class BatchMemoryManager:
    """バッチ処理用メモリ管理"""

    def __init__(self, max_memory_mb: Optional[float] = None):
        self.optimizer = MemoryOptimizer()
        self.max_memory_mb = max_memory_mb or self._get_safe_memory_limit()
        self.processed_items = 0
        self.optimization_interval = 10  # 10アイテムごとに最適化

        # P1-015: 大規模データセット対応機能
        self.large_dataset_mode = False
        self.adaptive_batch_size = True
        self.current_batch_size = 1
        self.max_batch_size = 5
        self.memory_pressure_threshold = 0.85  # 85%使用でメモリ圧迫
        self.consecutive_pressure_count = 0
        self.processing_history = []  # 処理時間履歴

    def _get_safe_memory_limit(self) -> float:
        """安全なメモリ制限を計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        # P1-015: 大規模データセット対応での動的制限
        if getattr(self, "large_dataset_mode", False):
            # 大規模データセットでは更に保守的に60%
            return total_memory * 0.6
        else:
            # 通常は70%を上限とする
            return total_memory * 0.7

    def should_optimize_memory(self) -> bool:
        """メモリ最適化が必要かチェック"""
        current_memory = self.optimizer.monitor.get_memory_usage()

        # メモリ制限を超えている場合
        if current_memory["ram_mb"] > self.max_memory_mb:
            return True

        # P1-015: メモリ圧迫検知
        if self._is_memory_pressure():
            return True

        # 定期的な最適化
        if self.processed_items % self.optimization_interval == 0:
            return True

        return False

    def process_batch_item(self, process_func: Callable, *args, **kwargs):
        """バッチアイテムを処理"""
        self.processed_items += 1

        # メモリ最適化チェック
        if self.should_optimize_memory():
            self.optimizer.optimize_memory()

        # アイテム処理
        try:
            result = process_func(*args, **kwargs)
            return result
        finally:
            # 処理後の軽量最適化
            if self.processed_items % 5 == 0:
                gc.collect()

    def enable_large_dataset_mode(self, max_batch_size: int = 10):
        """大規模データセットモード有効化"""
        self.large_dataset_mode = True
        self.max_batch_size = max_batch_size
        self.max_memory_mb = self._get_safe_memory_limit()  # 制限値再計算
        self.optimization_interval = 5  # より頻繁な最適化
        logger.info(f"🚀 大規模データセットモード有効化: バッチサイズ上限{max_batch_size}")

    def disable_large_dataset_mode(self):
        """大規模データセットモード無効化"""
        self.large_dataset_mode = False
        self.current_batch_size = 1
        self.max_memory_mb = self._get_safe_memory_limit()  # 制限値再計算
        self.optimization_interval = 10  # 通常間隔に戻す
        logger.info("📴 大規模データセットモード無効化")

    def _is_memory_pressure(self) -> bool:
        """メモリ圧迫状態の検知"""
        current_memory = self.optimizer.monitor.get_memory_usage()
        total_memory = psutil.virtual_memory().total / (1024 * 1024)

        usage_ratio = current_memory["ram_mb"] / total_memory

        if usage_ratio > self.memory_pressure_threshold:
            self.consecutive_pressure_count += 1
            logger.warning(f"⚠️ メモリ圧迫検知: {usage_ratio:.1%} 使用中")
            return True
        else:
            self.consecutive_pressure_count = 0
            return False

    def _adapt_batch_size(self):
        """動的バッチサイズ調整"""
        if not self.adaptive_batch_size:
            return

        # メモリ圧迫が続く場合はバッチサイズを削減
        if self.consecutive_pressure_count >= 3:
            self.current_batch_size = max(1, self.current_batch_size - 1)
            logger.info(f"📉 バッチサイズ削減: {self.current_batch_size}")
            self.consecutive_pressure_count = 0

        # メモリに余裕がある場合は増加を検討
        elif self.consecutive_pressure_count == 0 and len(self.processing_history) >= 10:
            recent_avg_time = sum(self.processing_history[-5:]) / 5
            if recent_avg_time < 30.0 and self.current_batch_size < self.max_batch_size:
                self.current_batch_size += 1
                logger.info(f"📈 バッチサイズ増加: {self.current_batch_size}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """メモリ統計を取得"""
        stats = self.optimizer.get_optimization_stats()
        stats.update(
            {
                "processed_items": self.processed_items,
                "memory_limit_mb": self.max_memory_mb,
                "optimization_interval": self.optimization_interval,
                # P1-015: 大規模データセット関連統計
                "large_dataset_mode": self.large_dataset_mode,
                "current_batch_size": self.current_batch_size,
                "max_batch_size": self.max_batch_size,
                "memory_pressure_threshold": self.memory_pressure_threshold,
                "consecutive_pressure_count": self.consecutive_pressure_count,
                "processing_history_length": len(self.processing_history),
            }
        )
        return stats


# グローバルインスタンス
global_memory_optimizer = MemoryOptimizer()
memory_efficient = global_memory_optimizer.memory_efficient_decorator()


def optimize_for_large_dataset(func: Callable) -> Callable:
    """大規模データセット用最適化デコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # スコープエラーを回避するため、必要なインポートをここで行う
        # これにより内部関数からの呼び出しでも正常に動作する
        with global_memory_optimizer.memory_context(auto_optimize=True):
            return func(*args, **kwargs)

    return wrapper


def get_memory_usage_summary() -> str:
    """メモリ使用量サマリーを取得"""
    memory = global_memory_optimizer.monitor.get_memory_usage()

    summary = f"💾 メモリ使用量: RAM {memory['ram_mb']:.1f}MB"
    if "gpu_allocated_mb" in memory:
        summary += f", GPU {memory['gpu_allocated_mb']:.1f}MB"

    return summary
