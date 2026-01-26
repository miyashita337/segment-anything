#!/usr/bin/env python3
"""
PH2-004: リソース管理最適化システム スタンドアロンテスト
"""

import torch

import gc
import json
import logging
import os
import psutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """リソースメトリクス"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_available: bool = False


class PH2004StandaloneOptimizer:
    """PH2-004 スタンドアロン最適化システム"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.cpu_count = psutil.cpu_count()
        self.memory_total_gb = psutil.virtual_memory().total / 1024**3

        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.gpu_device)

        logger.info(f"PH2-004 最適化システム初期化")
        logger.info(f"CPU: {self.cpu_count}コア, RAM: {self.memory_total_gb:.1f}GB")
        logger.info(f"GPU: {'利用可能' if self.gpu_available else '利用不可'}")

    def get_current_metrics(self) -> ResourceMetrics:
        """現在のリソースメトリクス取得"""
        memory = psutil.virtual_memory()

        gpu_memory_mb = None
        if self.gpu_available:
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
            except:
                gpu_memory_mb = None

        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024**2,
            gpu_memory_mb=gpu_memory_mb,
            gpu_available=self.gpu_available,
        )

    def optimize_resources(self) -> Dict[str, Any]:
        """リソース最適化実行"""
        logger.info("🚀 PH2-004 リソース最適化実行開始")

        # 最適化前の状態
        before_metrics = self.get_current_metrics()

        # 最適化処理
        optimizations_performed = []

        # 1. Python ガベージコレクション
        collected_objects = gc.collect()
        optimizations_performed.append(f"ガベージコレクション: {collected_objects}オブジェクト回収")

        # 2. GPU メモリクリーンアップ
        if self.gpu_available:
            before_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            freed_gpu_memory = before_gpu_memory - after_gpu_memory
            optimizations_performed.append(f"GPU メモリクリーンアップ: {freed_gpu_memory:.1f}MB解放")

        # 3. 強制ガベージコレクション（全世代）
        for generation in range(3):
            collected = gc.collect(generation)
            if collected > 0:
                optimizations_performed.append(f"ガベージコレクション第{generation}世代: {collected}オブジェクト回収")

        # 最適化後の状態
        after_metrics = self.get_current_metrics()

        # 改善計算
        memory_improvement_mb = before_metrics.memory_mb - after_metrics.memory_mb
        cpu_change = after_metrics.cpu_percent - before_metrics.cpu_percent

        return {
            "before_metrics": {
                "cpu_percent": before_metrics.cpu_percent,
                "memory_mb": before_metrics.memory_mb,
                "memory_percent": before_metrics.memory_percent,
                "gpu_memory_mb": before_metrics.gpu_memory_mb,
            },
            "after_metrics": {
                "cpu_percent": after_metrics.cpu_percent,
                "memory_mb": after_metrics.memory_mb,
                "memory_percent": after_metrics.memory_percent,
                "gpu_memory_mb": after_metrics.gpu_memory_mb,
            },
            "improvements": {
                "memory_freed_mb": memory_improvement_mb,
                "cpu_change_percent": cpu_change,
                "optimizations_performed": optimizations_performed,
            },
            "system_info": {
                "cpu_cores": self.cpu_count,
                "memory_total_gb": self.memory_total_gb,
                "gpu_available": self.gpu_available,
                "gpu_name": getattr(self, "gpu_name", None),
            },
        }

    def run_benchmark_test(self) -> Dict[str, Any]:
        """ベンチマークテスト実行"""
        logger.info("📊 PH2-004 ベンチマークテスト開始")

        # テスト用データ生成（メモリ使用）
        test_data = []

        # ベースライン測定
        baseline_start = time.time()
        baseline_metrics = self.get_current_metrics()

        # メモリ負荷テスト
        for i in range(1000):
            test_data.append([j * i for j in range(100)])

        baseline_end = time.time()
        baseline_duration = baseline_end - baseline_start

        # 最適化実行
        optimization_result = self.optimize_resources()

        # 最適化後テスト
        optimized_start = time.time()
        optimized_metrics = self.get_current_metrics()

        # 同様の処理を実行
        test_data_2 = []
        for i in range(1000):
            test_data_2.append([j * i for j in range(100)])

        optimized_end = time.time()
        optimized_duration = optimized_end - optimized_start

        # クリーンアップ
        del test_data, test_data_2
        gc.collect()

        return {
            "baseline": {
                "duration": baseline_duration,
                "metrics": {
                    "cpu_percent": baseline_metrics.cpu_percent,
                    "memory_percent": baseline_metrics.memory_percent,
                    "memory_mb": baseline_metrics.memory_mb,
                },
            },
            "optimized": {
                "duration": optimized_duration,
                "metrics": {
                    "cpu_percent": optimized_metrics.cpu_percent,
                    "memory_percent": optimized_metrics.memory_percent,
                    "memory_mb": optimized_metrics.memory_mb,
                },
            },
            "optimization_result": optimization_result,
            "performance_improvement": {
                "duration_improvement_percent": (
                    (baseline_duration - optimized_duration) / baseline_duration
                )
                * 100
                if baseline_duration > 0
                else 0,
                "memory_efficiency": optimization_result["improvements"]["memory_freed_mb"],
            },
        }


def run_ph2_004_test():
    """PH2-004テスト実行メイン関数"""
    print("🎯 PH2-004 リソース管理最適化システムテスト")
    print("=" * 60)

    optimizer = PH2004StandaloneOptimizer()

    # 初期状態表示
    initial_metrics = optimizer.get_current_metrics()
    print(f"📊 初期リソース状況:")
    print(f"   CPU: {initial_metrics.cpu_percent:.1f}% ({optimizer.cpu_count}コア)")
    print(f"   メモリ: {initial_metrics.memory_percent:.1f}% ({initial_metrics.memory_mb:.0f}MB)")
    if initial_metrics.gpu_available and initial_metrics.gpu_memory_mb:
        print(f"   GPU メモリ: {initial_metrics.gpu_memory_mb:.0f}MB")

    # ベンチマークテスト実行
    benchmark_results = optimizer.run_benchmark_test()

    # 結果表示
    print(f"\n📈 ベンチマーク結果:")
    baseline = benchmark_results["baseline"]
    optimized = benchmark_results["optimized"]
    improvement = benchmark_results["performance_improvement"]

    print(f"   ベースライン処理時間: {baseline['duration']:.4f}秒")
    print(f"   最適化後処理時間: {optimized['duration']:.4f}秒")
    print(f"   処理時間改善: {improvement['duration_improvement_percent']:.1f}%")
    print(f"   メモリ効率化: {improvement['memory_efficiency']:.1f}MB解放")

    # 最適化詳細
    opt_result = benchmark_results["optimization_result"]
    print(f"\n🚀 実行された最適化:")
    for opt in opt_result["improvements"]["optimizations_performed"]:
        print(f"   • {opt}")

    # システム情報
    sys_info = opt_result["system_info"]
    print(f"\n💻 システム情報:")
    print(f"   CPU: {sys_info['cpu_cores']}コア")
    print(f"   RAM: {sys_info['memory_total_gb']:.1f}GB")
    print(f"   GPU: {sys_info['gpu_name'] if sys_info['gpu_available'] else '利用不可'}")

    # テスト結果をファイルに保存
    output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-004/tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "ph2_004_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📁 結果保存: {results_file}")
    print("\n✅ PH2-004 リソース管理最適化システムテスト完了")

    return benchmark_results


if __name__ == "__main__":
    results = run_ph2_004_test()

    # 成功判定
    improvement = results["performance_improvement"]["duration_improvement_percent"]
    memory_freed = results["performance_improvement"]["memory_efficiency"]

    if improvement > 0 or memory_freed > 0:
        print(f"\n🎉 テスト成功 - リソース最適化が確認されました")
    else:
        print(f"\n⚠️ テスト完了 - 最適化効果は限定的でした")
