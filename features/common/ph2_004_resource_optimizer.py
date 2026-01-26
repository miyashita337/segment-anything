#!/usr/bin/env python3
"""
PH2-004: リソース管理最適化システム
CPU/メモリ/GPU使用率改善とパフォーマンス最適化
"""

import torch

import gc
import json
import logging
import os
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .error_handling import (
    BaseCustomError,
    ErrorCategory,
    ErrorSeverity,
    GPUNotAvailableError,
    InsufficientMemoryError,
    ResourceError,
)


@dataclass
class OptimizedResourceUsage:
    """最適化されたリソース使用状況"""

    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    memory_mb: float
    memory_percent: float
    memory_available_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    process_count: int = 0
    optimization_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_utilization": self.gpu_utilization,
            "gpu_temperature": self.gpu_temperature,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_mb": self.network_io_mb,
            "process_count": self.process_count,
            "optimization_suggestions": self.optimization_suggestions,
        }


@dataclass
class ResourceThresholds:
    """リソース閾値設定"""

    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 70.0
    memory_critical: float = 90.0
    gpu_memory_warning: float = 80.0
    gpu_memory_critical: float = 95.0
    gpu_temperature_warning: float = 80.0
    gpu_temperature_critical: float = 90.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0


class PH2004ResourceOptimizer:
    """PH2-004: 高度なリソース管理最適化システム"""

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        monitoring_interval: float = 2.0,
        history_retention_hours: int = 24,
        auto_optimization: bool = True,
        aggressive_cleanup: bool = False,
    ):
        """
        Args:
            thresholds: リソース閾値設定
            monitoring_interval: モニタリング間隔（秒）
            history_retention_hours: 履歴保持時間
            auto_optimization: 自動最適化有効化
            aggressive_cleanup: アグレッシブクリーンアップ
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.monitoring_interval = monitoring_interval
        self.history_retention_hours = history_retention_hours
        self.auto_optimization = auto_optimization
        self.aggressive_cleanup = aggressive_cleanup

        self.usage_history: List[OptimizedResourceUsage] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.optimization_callbacks: List[Callable] = []

        self.logger = logging.getLogger(__name__)

        # システム情報取得
        self.cpu_count = psutil.cpu_count()
        self.memory_total_gb = psutil.virtual_memory().total / 1024**3

        # GPU検出と初期化
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.gpu_device)
            self.gpu_memory_total_mb = (
                torch.cuda.get_device_properties(self.gpu_device).total_memory / 1024**2
            )
            self.logger.info(f"GPU detected: {self.gpu_name} ({self.gpu_memory_total_mb:.0f} MB)")

        # 初期化完了ログ
        self.logger.info(f"PH2-004 ResourceOptimizer initialized")
        self.logger.info(f"CPU cores: {self.cpu_count}, RAM: {self.memory_total_gb:.1f}GB")

    def get_comprehensive_usage(self) -> OptimizedResourceUsage:
        """包括的なリソース使用状況を取得"""
        timestamp = datetime.now()

        # CPU情報
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = self.cpu_count

        # メモリ情報
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024**2
        memory_percent = memory.percent
        memory_available_mb = memory.available / 1024**2

        # ディスク情報
        disk = psutil.disk_usage("/")
        disk_usage_percent = disk.percent

        # ネットワークI/O
        network = psutil.net_io_counters()
        network_io_mb = (network.bytes_sent + network.bytes_recv) / 1024**2

        # プロセス数
        process_count = len(psutil.pids())

        # GPU情報
        gpu_memory_mb = None
        gpu_memory_total_mb = None
        gpu_utilization = None
        gpu_temperature = None

        if self.gpu_available:
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_total_mb = self.gpu_memory_total_mb

                # NVIDIA-MLを使用してGPU利用率と温度を取得（可能な場合）
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu

                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature = temp

                except ImportError:
                    # pynvmlが利用できない場合は基本情報のみ
                    gpu_utilization = min(gpu_memory_mb / gpu_memory_total_mb * 100, 100.0)

            except Exception as e:
                self.logger.warning(f"GPU情報取得エラー: {e}")

        # 最適化提案の生成
        optimization_suggestions = self._generate_optimization_suggestions(
            cpu_percent, memory_percent, gpu_memory_mb, gpu_memory_total_mb, disk_usage_percent
        )

        return OptimizedResourceUsage(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            disk_usage_percent=disk_usage_percent,
            network_io_mb=network_io_mb,
            process_count=process_count,
            optimization_suggestions=optimization_suggestions,
        )

    def _generate_optimization_suggestions(
        self,
        cpu_percent: float,
        memory_percent: float,
        gpu_memory_mb: Optional[float],
        gpu_memory_total_mb: Optional[float],
        disk_usage_percent: float,
    ) -> List[str]:
        """最適化提案の生成"""
        suggestions = []

        # CPU最適化提案
        if cpu_percent > self.thresholds.cpu_critical:
            suggestions.append("🚨 CPU使用率が危険レベル - プロセス優先度調整を推奨")
        elif cpu_percent > self.thresholds.cpu_warning:
            suggestions.append("⚠️ CPU使用率が高い - 並列処理数の調整を検討")
        elif cpu_percent < 20:
            suggestions.append("📈 CPU使用率が低い - 並列処理の活用余地あり")

        # メモリ最適化提案
        if memory_percent > self.thresholds.memory_critical:
            suggestions.append("🚨 メモリ使用率が危険レベル - 即座にクリーンアップ必要")
        elif memory_percent > self.thresholds.memory_warning:
            suggestions.append("⚠️ メモリ使用率が高い - ガベージコレクション推奨")

        # GPU最適化提案
        if gpu_memory_mb and gpu_memory_total_mb:
            gpu_usage_percent = (gpu_memory_mb / gpu_memory_total_mb) * 100
            if gpu_usage_percent > self.thresholds.gpu_memory_critical:
                suggestions.append("🚨 GPU メモリが危険レベル - バッチサイズ削減必要")
            elif gpu_usage_percent > self.thresholds.gpu_memory_warning:
                suggestions.append("⚠️ GPU メモリ使用率が高い - キャッシュクリア推奨")
            elif gpu_usage_percent < 30:
                suggestions.append("📈 GPU活用率が低い - バッチサイズ増加を検討")

        # ディスク最適化提案
        if disk_usage_percent > self.thresholds.disk_critical:
            suggestions.append("🚨 ディスク容量が危険レベル - 不要ファイル削除必要")
        elif disk_usage_percent > self.thresholds.disk_warning:
            suggestions.append("⚠️ ディスク使用率が高い - 容量確認を推奨")

        return suggestions

    def start_monitoring(self) -> bool:
        """リソース監視開始"""
        if self.is_monitoring:
            self.logger.warning("監視は既に開始されています")
            return False

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("PH2-004 リソース監視開始")
        return True

    def stop_monitoring(self) -> bool:
        """リソース監視停止"""
        if not self.is_monitoring:
            return False

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.monitoring_interval * 2)

        self.logger.info("PH2-004 リソース監視停止")
        return True

    def _monitoring_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                usage = self.get_comprehensive_usage()
                self.usage_history.append(usage)

                # 履歴の剪定
                self._prune_history()

                # 自動最適化の実行
                if self.auto_optimization:
                    self._auto_optimize(usage)

                # 警告チェック
                self._check_warnings(usage)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")

            time.sleep(self.monitoring_interval)

    def _prune_history(self):
        """履歴の剪定"""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        self.usage_history = [
            usage for usage in self.usage_history if usage.timestamp > cutoff_time
        ]

    def _auto_optimize(self, usage: OptimizedResourceUsage):
        """自動最適化"""
        if usage.memory_percent > self.thresholds.memory_warning:
            self.force_cleanup()

        if self.gpu_available and usage.gpu_memory_mb:
            gpu_usage_percent = (usage.gpu_memory_mb / usage.gpu_memory_total_mb) * 100
            if gpu_usage_percent > self.thresholds.gpu_memory_warning:
                self.cleanup_gpu_memory()

    def _check_warnings(self, usage: OptimizedResourceUsage):
        """警告チェック"""
        warnings = []

        if usage.cpu_percent > self.thresholds.cpu_critical:
            warnings.append(f"CPU使用率危険: {usage.cpu_percent:.1f}%")

        if usage.memory_percent > self.thresholds.memory_critical:
            warnings.append(f"メモリ使用率危険: {usage.memory_percent:.1f}%")

        if (
            usage.gpu_temperature
            and usage.gpu_temperature > self.thresholds.gpu_temperature_critical
        ):
            warnings.append(f"GPU温度危険: {usage.gpu_temperature:.1f}°C")

        for warning in warnings:
            self.logger.warning(f"PH2-004 リソース警告: {warning}")

    @contextmanager
    def resource_monitoring_context(self, task_name: str = "task"):
        """リソース監視コンテキスト"""
        start_usage = self.get_comprehensive_usage()
        start_time = time.time()

        self.logger.info(f"PH2-004 タスク開始: {task_name}")
        self.logger.info(
            f"開始時リソース - CPU: {start_usage.cpu_percent:.1f}%, Memory: {start_usage.memory_percent:.1f}%"
        )

        try:
            yield start_usage
        finally:
            end_usage = self.get_comprehensive_usage()
            end_time = time.time()
            duration = end_time - start_time

            self.logger.info(f"PH2-004 タスク完了: {task_name} ({duration:.2f}秒)")
            self.logger.info(
                f"終了時リソース - CPU: {end_usage.cpu_percent:.1f}%, Memory: {end_usage.memory_percent:.1f}%"
            )

            # リソース使用量の変化を計算
            memory_delta = end_usage.memory_mb - start_usage.memory_mb
            if abs(memory_delta) > 100:  # 100MB以上の変化
                self.logger.info(f"メモリ使用量変化: {memory_delta:+.1f}MB")

    def force_cleanup(self):
        """強制クリーンアップ"""
        self.logger.info("PH2-004 強制クリーンアップ実行")

        # Python ガベージコレクション
        collected = gc.collect()

        # GPU メモリクリーンアップ
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # アグレッシブクリーンアップ（必要な場合）
        if self.aggressive_cleanup:
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)

        self.logger.info(f"クリーンアップ完了 - 回収オブジェクト: {collected}")

    def cleanup_gpu_memory(self):
        """GPU メモリクリーンアップ"""
        if not self.gpu_available:
            return

        before_memory = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_memory = torch.cuda.memory_allocated() / 1024**2

        freed_memory = before_memory - after_memory
        self.logger.info(f"GPU メモリクリーンアップ - 解放: {freed_memory:.1f}MB")

    def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポートの生成"""
        if not self.usage_history:
            return {"error": "履歴データなし"}

        latest_usage = self.usage_history[-1]

        # 平均値計算
        recent_history = self.usage_history[-60:]  # 直近60回
        avg_cpu = sum(u.cpu_percent for u in recent_history) / len(recent_history)
        avg_memory = sum(u.memory_percent for u in recent_history) / len(recent_history)

        report = {
            "timestamp": latest_usage.timestamp.isoformat(),
            "current_status": {
                "cpu_percent": latest_usage.cpu_percent,
                "memory_percent": latest_usage.memory_percent,
                "memory_available_gb": latest_usage.memory_available_mb / 1024,
                "gpu_memory_mb": latest_usage.gpu_memory_mb,
                "disk_usage_percent": latest_usage.disk_usage_percent,
                "process_count": latest_usage.process_count,
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
            },
            "system_info": {
                "cpu_cores": self.cpu_count,
                "memory_total_gb": self.memory_total_gb,
                "gpu_available": self.gpu_available,
                "gpu_name": getattr(self, "gpu_name", None),
                "gpu_memory_total_mb": getattr(self, "gpu_memory_total_mb", None),
            },
            "optimization_suggestions": latest_usage.optimization_suggestions,
            "thresholds": {
                "cpu_warning": self.thresholds.cpu_warning,
                "memory_warning": self.thresholds.memory_warning,
                "gpu_memory_warning": self.thresholds.gpu_memory_warning,
            },
            "performance_recommendations": self._generate_performance_recommendations(
                avg_cpu, avg_memory
            ),
        }

        return report

    def _generate_performance_recommendations(self, avg_cpu: float, avg_memory: float) -> List[str]:
        """パフォーマンス推奨事項の生成"""
        recommendations = []

        if avg_cpu < 30:
            recommendations.append(f"CPU活用率が低い（{avg_cpu:.1f}%）- 並列処理数の増加を検討")
        elif avg_cpu > 85:
            recommendations.append(f"CPU使用率が高い（{avg_cpu:.1f}%）- 処理負荷の分散を検討")

        if avg_memory > 80:
            recommendations.append(f"メモリ使用率が高い（{avg_memory:.1f}%）- メモリ効率の改善を推奨")

        if self.gpu_available:
            recommendations.append("GPU活用最適化 - バッチサイズとメモリ使用量のバランス調整")

        recommendations.append(f"{self.cpu_count}コアCPU環境での並列処理最適化を活用")

        return recommendations

    def save_optimization_report(self, output_path: Path) -> bool:
        """最適化レポートの保存"""
        try:
            report = self.get_optimization_report()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"最適化レポート保存: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"レポート保存エラー: {e}")
            return False


# ユーティリティ関数
def create_optimized_resource_manager(**kwargs) -> PH2004ResourceOptimizer:
    """最適化されたリソースマネージャーの作成"""
    return PH2004ResourceOptimizer(**kwargs)


def monitor_resource_usage(func):
    """リソース使用量監視デコレータ"""

    def wrapper(*args, **kwargs):
        optimizer = create_optimized_resource_manager()

        with optimizer.resource_monitoring_context(func.__name__):
            result = func(*args, **kwargs)

        return result

    return wrapper


if __name__ == "__main__":
    # テスト実行
    optimizer = PH2004ResourceOptimizer()

    print("🎯 PH2-004 リソース最適化システムテスト")
    print("=" * 60)

    # 現在の使用状況
    usage = optimizer.get_comprehensive_usage()
    print(f"CPU: {usage.cpu_percent:.1f}% ({usage.cpu_count}コア)")
    print(f"メモリ: {usage.memory_percent:.1f}% ({usage.memory_mb:.0f}MB使用)")
    print(f"GPU: {'利用可能' if optimizer.gpu_available else '利用不可'}")

    if optimizer.gpu_available:
        print(f"GPU メモリ: {usage.gpu_memory_mb:.0f}MB / {usage.gpu_memory_total_mb:.0f}MB")

    print(f"\n📊 最適化提案:")
    for suggestion in usage.optimization_suggestions:
        print(f"   {suggestion}")

    # 最適化レポート生成
    optimizer.start_monitoring()
    time.sleep(5)  # 5秒間監視
    optimizer.stop_monitoring()

    report = optimizer.get_optimization_report()
    print(f"\n📈 パフォーマンス推奨事項:")
    for rec in report.get("performance_recommendations", []):
        print(f"   • {rec}")

    print("\n✅ PH2-004 リソース最適化システム初期化完了")
