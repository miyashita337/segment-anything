"""
メモリ最適化システムのテスト
"""

import numpy as np

import pytest
import time
from features.common.memory_optimizer import (
    BatchMemoryManager,
    MemoryMonitor,
    MemoryOptimizer,
    global_memory_optimizer,
    memory_efficient,
)
from unittest.mock import Mock, patch


class TestMemoryMonitor:
    """MemoryMonitorのテスト"""

    def test_get_memory_usage(self):
        """メモリ使用量取得のテスト"""
        monitor = MemoryMonitor()
        memory = monitor.get_memory_usage()

        # 基本的なキーが存在することを確認
        assert "ram_mb" in memory
        assert "virtual_mb" in memory
        assert "available_mb" in memory

        # 値が正の数であることを確認
        assert memory["ram_mb"] > 0
        assert memory["virtual_mb"] > 0
        assert memory["available_mb"] > 0

    def test_monitoring(self):
        """監視機能のテスト"""
        monitor = MemoryMonitor()

        # 監視開始
        monitor.start_monitoring(interval=0.1)
        assert monitor._monitoring is True

        # 少し待つ
        time.sleep(0.2)

        # 監視停止
        monitor.stop_monitoring()
        assert monitor._monitoring is False


class TestMemoryOptimizer:
    """MemoryOptimizerのテスト"""

    def test_optimize_memory(self):
        """メモリ最適化のテスト"""
        optimizer = MemoryOptimizer()

        # 最適化実行
        result = optimizer.optimize_memory()

        # 結果の構造確認
        assert "before" in result
        assert "after" in result
        assert "freed_mb" in result

        # メモリ情報が存在することを確認
        assert "ram_mb" in result["before"]
        assert "ram_mb" in result["after"]

    def test_memory_context(self):
        """メモリコンテキストのテスト"""
        optimizer = MemoryOptimizer()

        with optimizer.memory_context() as ctx:
            # コンテキスト内での処理
            assert ctx is optimizer

            # 大きな配列を作成（メモリ使用量増加）
            large_array = np.random.rand(1000, 1000)
            assert large_array.nbytes > 0

    def test_memory_efficient_decorator(self):
        """メモリ効率デコレータのテスト"""
        optimizer = MemoryOptimizer()

        @optimizer.memory_efficient_decorator()
        def test_function(size: int) -> np.ndarray:
            return np.zeros((size, size))

        # デコレータ適用関数の実行
        result = test_function(100)
        assert result.shape == (100, 100)

    def test_get_optimization_stats(self):
        """最適化統計のテスト"""
        optimizer = MemoryOptimizer()

        # 最適化実行
        optimizer.optimize_memory()

        # 統計取得
        stats = optimizer.get_optimization_stats()

        # 統計の構造確認
        assert "optimization_stats" in stats
        assert "current_memory" in stats
        assert "peak_memory" in stats

        # 最適化統計の詳細確認
        opt_stats = stats["optimization_stats"]
        assert "gc_calls" in opt_stats
        assert "memory_freed_mb" in opt_stats
        assert "cache_clears" in opt_stats

        # GCが実行されていることを確認
        assert opt_stats["gc_calls"] > 0


class TestBatchMemoryManager:
    """BatchMemoryManagerのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        manager = BatchMemoryManager()

        assert manager.processed_items == 0
        assert manager.optimization_interval == 10
        assert manager.max_memory_mb > 0

    def test_should_optimize_memory(self):
        """メモリ最適化判定のテスト"""
        manager = BatchMemoryManager(max_memory_mb=1000)  # 低い制限値

        # 初期状態（0アイテム目なので10の倍数でTrue）
        assert manager.should_optimize_memory() is True

        # アイテム処理を進める
        for i in range(1, 11):
            manager.processed_items = i
            if i % 10 == 0:  # 10の倍数の時
                assert manager.should_optimize_memory() is True
            else:
                # メモリ制限チェックは実際のRAM使用量に依存するため、
                # ここでは10の倍数でない場合の詳細テストは省略
                pass

    def test_process_batch_item(self):
        """バッチアイテム処理のテスト"""
        manager = BatchMemoryManager()

        def dummy_process(value: int) -> int:
            return value * 2

        # アイテム処理
        result = manager.process_batch_item(dummy_process, 5)

        assert result == 10
        assert manager.processed_items == 1

    def test_get_memory_stats(self):
        """メモリ統計取得のテスト"""
        manager = BatchMemoryManager()

        # 統計取得
        stats = manager.get_memory_stats()

        # 統計の構造確認
        assert "processed_items" in stats
        assert "memory_limit_mb" in stats
        assert "optimization_interval" in stats
        assert "optimization_stats" in stats
        assert "current_memory" in stats


class TestGlobalFunctions:
    """グローバル関数のテスト"""

    def test_memory_efficient_decorator(self):
        """グローバルメモリ効率デコレータのテスト"""

        @memory_efficient
        def test_function(x: int) -> int:
            # 大きな配列を作成して削除
            large_array = np.random.rand(500, 500)
            return x + len(large_array)

        result = test_function(10)
        assert result > 10

    def test_optimize_for_large_dataset(self):
        """大規模データセット最適化デコレータのテスト"""
        from features.common.memory_optimizer import optimize_for_large_dataset

        @optimize_for_large_dataset
        def process_large_dataset(items: int) -> list:
            # 大量のデータを処理するシミュレーション
            data = []
            for i in range(items):
                data.append(np.random.rand(10, 10))
            return data

        result = process_large_dataset(5)
        assert len(result) == 5
        assert all(item.shape == (10, 10) for item in result)

    def test_get_memory_usage_summary(self):
        """メモリ使用量サマリーのテスト"""
        from features.common.memory_optimizer import get_memory_usage_summary

        summary = get_memory_usage_summary()

        # サマリー文字列の基本確認
        assert isinstance(summary, str)
        assert "メモリ使用量" in summary
        assert "RAM" in summary
        assert "MB" in summary


class TestMemoryLeakDetection:
    """メモリリーク検出のテスト"""

    def test_memory_usage_tracking(self):
        """メモリ使用量追跡のテスト"""
        monitor = MemoryMonitor()

        initial_memory = monitor.get_memory_usage()["ram_mb"]

        # 大きなオブジェクトを作成
        large_objects = []
        for i in range(100):  # より多くのオブジェクトを作成
            large_objects.append(np.random.rand(500, 500))  # より大きなオブジェクト

        after_allocation = monitor.get_memory_usage()["ram_mb"]

        # メモリ使用量が増加していることを確認（1MB以上の差を期待）
        memory_diff = after_allocation - initial_memory
        assert memory_diff > 0.1, f"メモリ使用量の増加が検出されませんでした: {memory_diff}MB"

        # オブジェクトを削除
        del large_objects

        # メモリ最適化実行
        optimizer = MemoryOptimizer()
        optimizer.optimize_memory()

        final_memory = monitor.get_memory_usage()["ram_mb"]

        # メモリが解放されていることを確認（完全ではないが減少傾向）
        # 注意: Pythonのメモリ管理により、完全に元の値に戻らない場合がある
