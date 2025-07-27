#!/usr/bin/env python3
"""
リソース管理最適化システム
PH2-002: メモリ・GPU・CPUリソースの効率的な管理
"""

import gc
import os
import psutil
import torch
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

from .error_handling import (
    InsufficientMemoryError,
    GPUNotAvailableError,
    BaseCustomError,
    ErrorSeverity,
    ErrorCategory,
)


@dataclass
class ResourceUsage:
    """リソース使用状況"""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
        }


class ResourceManager:
    """リソース管理クラス"""

    def __init__(
        self,
        memory_threshold_mb: float = 1000,
        gpu_memory_threshold_mb: float = 500,
        monitoring_interval: float = 5.0,
        auto_cleanup: bool = True,
    ):
        """
        Args:
            memory_threshold_mb: メモリ警告閾値（MB）
            gpu_memory_threshold_mb: GPUメモリ警告閾値（MB）
            monitoring_interval: モニタリング間隔（秒）
            auto_cleanup: 自動クリーンアップ有効化
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.gpu_memory_threshold_mb = gpu_memory_threshold_mb
        self.monitoring_interval = monitoring_interval
        self.auto_cleanup = auto_cleanup

        self.usage_history: List[ResourceUsage] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.cleanup_callbacks: List[Callable] = []

        self.logger = logging.getLogger(__name__)

        # GPU可用性チェック
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name(self.gpu_device)}")

    def initialize(self) -> bool:
        """リソースマネージャーの初期化"""
        try:
            # 初期リソース状況の取得
            initial_usage = self.get_current_usage()
            self.usage_history.append(initial_usage)

            self.logger.info("ResourceManager initialized successfully")
            self.logger.info(
                f"Initial CPU: {initial_usage.cpu_percent}%, Memory: {initial_usage.memory_percent}%"
            )

            if self.gpu_available:
                self.logger.info(f"GPU Memory: {initial_usage.gpu_memory_mb:.1f}MB")

            return True
        except Exception as e:
            self.logger.error(f"ResourceManager initialization failed: {e}")
            return False

    def get_current_usage(self) -> ResourceUsage:
        """現在のリソース使用状況を取得"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # メモリ使用量
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent

        # GPU使用量
        gpu_memory_mb = None
        gpu_utilization = None

        if self.gpu_available:
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # GPU利用率取得（nvidia-smiが必要）
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        gpu_utilization = float(result.stdout.strip())
                except:
                    pass
            except Exception as e:
                self.logger.warning(f"GPU usage collection failed: {e}")

        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
        )

    def check_memory_availability(self, required_mb: float) -> bool:
        """必要なメモリが利用可能かチェック"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)

        if available_mb < required_mb:
            if self.auto_cleanup:
                self.logger.warning(f"Memory shortage detected. Running cleanup...")
                self.cleanup_memory()
                # 再チェック
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)

            if available_mb < required_mb:
                raise InsufficientMemoryError(required_mb=required_mb, available_mb=available_mb)

        return True

    def cleanup_memory(self, deep: bool = False):
        """メモリクリーンアップ"""
        self.logger.info("Starting memory cleanup...")

        # カスタムクリーンアップコールバック実行
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {e}")

        # ガベージコレクション
        gc.collect()

        if deep:
            # 強制的な完全ガベージコレクション
            gc.collect(2)

        # GPU メモリクリア
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 使用状況ログ
        usage = self.get_current_usage()
        self.logger.info(
            f"Cleanup completed. Memory: {usage.memory_mb:.1f}MB ({usage.memory_percent:.1f}%), "
            f"GPU: {usage.gpu_memory_mb:.1f}MB"
            if usage.gpu_memory_mb
            else ""
        )

    def get_usage_summary(self) -> Dict[str, Any]:
        """現在のリソース使用状況のサマリを取得"""
        current_usage = self.get_current_usage()

        summary = {
            "cpu_percent": current_usage.cpu_percent,
            "memory_mb": current_usage.memory_mb,
            "memory_percent": current_usage.memory_percent,
            "timestamp": current_usage.timestamp.isoformat(),
        }

        if self.gpu_available and current_usage.gpu_memory_mb is not None:
            summary.update(
                {
                    "gpu_memory_mb": current_usage.gpu_memory_mb,
                    "gpu_utilization": current_usage.gpu_utilization,
                    "gpu_available": True,
                }
            )
        else:
            summary["gpu_available"] = False

        # 履歴統計（直近10件）
        if len(self.usage_history) > 1:
            recent_history = self.usage_history[-10:]
            cpu_avg = sum(u.cpu_percent for u in recent_history) / len(recent_history)
            memory_avg = sum(u.memory_percent for u in recent_history) / len(recent_history)

            summary.update(
                {
                    "cpu_avg_recent": cpu_avg,
                    "memory_avg_recent": memory_avg,
                    "history_count": len(self.usage_history),
                }
            )

            if self.gpu_available:
                gpu_memory_values = [
                    u.gpu_memory_mb for u in recent_history if u.gpu_memory_mb is not None
                ]
                if gpu_memory_values:
                    gpu_memory_avg = sum(gpu_memory_values) / len(gpu_memory_values)
                    summary["gpu_memory_avg_recent"] = gpu_memory_avg

        return summary

    def register_cleanup_callback(self, callback: Callable):
        """クリーンアップコールバックを登録"""
        self.cleanup_callbacks.append(callback)

    def start_monitoring(self):
        """リソースモニタリング開始"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """リソースモニタリング停止"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """モニタリングループ"""
        while self.is_monitoring:
            try:
                usage = self.get_current_usage()
                self.usage_history.append(usage)

                # 履歴サイズ制限（最新1000件のみ保持）
                if len(self.usage_history) > 1000:
                    self.usage_history = self.usage_history[-1000:]

                # 閾値チェック
                if usage.memory_mb > self.memory_threshold_mb:
                    self.logger.warning(
                        f"Memory usage high: {usage.memory_mb:.1f}MB "
                        f"(threshold: {self.memory_threshold_mb}MB)"
                    )
                    if self.auto_cleanup:
                        self.cleanup_memory()

                if (
                    self.gpu_available
                    and usage.gpu_memory_mb
                    and usage.gpu_memory_mb > self.gpu_memory_threshold_mb
                ):
                    self.logger.warning(
                        f"GPU memory usage high: {usage.gpu_memory_mb:.1f}MB "
                        f"(threshold: {self.gpu_memory_threshold_mb}MB)"
                    )
                    if self.auto_cleanup:
                        torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    def get_usage_summary(self) -> Dict[str, Any]:
        """使用状況サマリーを取得"""
        if not self.usage_history:
            return {}

        recent_usages = self.usage_history[-10:]

        return {
            "current": self.get_current_usage().to_dict(),
            "average": {
                "cpu_percent": sum(u.cpu_percent for u in recent_usages) / len(recent_usages),
                "memory_mb": sum(u.memory_mb for u in recent_usages) / len(recent_usages),
                "gpu_memory_mb": (
                    sum(u.gpu_memory_mb for u in recent_usages if u.gpu_memory_mb)
                    / len([u for u in recent_usages if u.gpu_memory_mb])
                    if self.gpu_available and any(u.gpu_memory_mb for u in recent_usages)
                    else None
                ),
            },
            "peak": {
                "cpu_percent": max(u.cpu_percent for u in self.usage_history),
                "memory_mb": max(u.memory_mb for u in self.usage_history),
                "gpu_memory_mb": (
                    max(u.gpu_memory_mb for u in self.usage_history if u.gpu_memory_mb)
                    if self.gpu_available and any(u.gpu_memory_mb for u in self.usage_history)
                    else None
                ),
            },
        }

    def save_usage_report(self, filepath: Path):
        """使用状況レポートを保存"""
        report = {
            "summary": self.get_usage_summary(),
            "history": [u.to_dict() for u in self.usage_history[-100:]],  # 最新100件
            "settings": {
                "memory_threshold_mb": self.memory_threshold_mb,
                "gpu_memory_threshold_mb": self.gpu_memory_threshold_mb,
                "monitoring_interval": self.monitoring_interval,
                "auto_cleanup": self.auto_cleanup,
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Usage report saved to {filepath}")


@contextmanager
def managed_resources(
    memory_mb: Optional[float] = None,
    gpu_memory_mb: Optional[float] = None,
    cleanup_on_exit: bool = True,
):
    """リソース管理コンテキストマネージャー"""
    manager = ResourceManager()

    # メモリチェック
    if memory_mb:
        manager.check_memory_availability(memory_mb)

    # モニタリング開始
    manager.start_monitoring()

    try:
        yield manager
    finally:
        # モニタリング停止
        manager.stop_monitoring()

        # クリーンアップ
        if cleanup_on_exit:
            manager.cleanup_memory()


class BatchProcessor:
    """バッチ処理最適化クラス"""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        memory_fraction: float = 0.8,
    ):
        """
        Args:
            initial_batch_size: 初期バッチサイズ
            min_batch_size: 最小バッチサイズ
            max_batch_size: 最大バッチサイズ
            memory_fraction: 使用可能メモリの割合
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_fraction = memory_fraction

        self.resource_manager = ResourceManager()
        self.logger = logging.getLogger(__name__)

    def get_optimal_batch_size(self, item_memory_mb: float) -> int:
        """最適なバッチサイズを計算"""
        # 利用可能メモリ取得
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024) * self.memory_fraction

        # GPU メモリも考慮
        if self.resource_manager.gpu_available:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_total_mb = gpu_props.total_memory / (1024 * 1024)
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_available_mb = (gpu_total_mb - gpu_allocated_mb) * self.memory_fraction

            # より制限的な方を使用
            available_mb = min(available_mb, gpu_available_mb)

        # バッチサイズ計算
        optimal_size = int(available_mb / item_memory_mb)
        optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))

        self.logger.info(
            f"Optimal batch size: {optimal_size} "
            f"(available memory: {available_mb:.1f}MB, item size: {item_memory_mb:.1f}MB)"
        )

        return optimal_size

    def adjust_batch_size(self, success: bool, memory_error: bool = False):
        """処理結果に基づいてバッチサイズを調整"""
        if success and not memory_error:
            # 成功した場合は徐々に増やす
            self.current_batch_size = min(int(self.current_batch_size * 1.1), self.max_batch_size)
        elif memory_error:
            # メモリエラーの場合は大幅に減らす
            self.current_batch_size = max(int(self.current_batch_size * 0.5), self.min_batch_size)
        else:
            # その他のエラーの場合は少し減らす
            self.current_batch_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)

        self.logger.info(f"Batch size adjusted to: {self.current_batch_size}")


# グローバルリソースマネージャー
global_resource_manager = ResourceManager()


def cleanup_global_resources():
    """グローバルリソースのクリーンアップ"""
    global_resource_manager.cleanup_memory(deep=True)
