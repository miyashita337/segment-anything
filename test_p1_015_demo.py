#!/usr/bin/env python3
"""
P1-015 メモリ使用最適化のデモスクリプト
大規模データセット対応のためのメモリ効率化
"""

import numpy as np

import logging
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.common.memory_optimizer import (
    BatchMemoryManager,
    MemoryMonitor,
    MemoryOptimizer,
    get_memory_usage_summary,
    memory_efficient,
    optimize_for_large_dataset,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_memory_monitoring():
    """メモリ監視機能のデモ"""
    print("\n=== メモリ監視機能のデモ ===")

    monitor = MemoryMonitor()

    # 初期メモリ使用量
    initial_memory = monitor.get_memory_usage()
    print(f"💾 初期メモリ使用量: RAM {initial_memory['ram_mb']:.1f}MB")
    if "gpu_allocated_mb" in initial_memory:
        print(f"    GPU {initial_memory['gpu_allocated_mb']:.1f}MB")

    # 監視開始
    monitor.start_monitoring(interval=0.5)

    # 大きなデータを作成
    print("📈 大容量データ作成中...")
    large_arrays = []
    for i in range(5):
        arr = np.random.rand(500, 500).astype(np.float32)
        large_arrays.append(arr)
        time.sleep(0.2)

    # 現在のメモリ使用量
    current_memory = monitor.get_memory_usage()
    print(f"💾 データ作成後: RAM {current_memory['ram_mb']:.1f}MB")

    # データを削除
    del large_arrays

    # 監視停止
    monitor.stop_monitoring()

    # ピークメモリ表示
    peak_memory = monitor.get_peak_memory()
    print(f"📊 ピークメモリ: RAM {peak_memory['ram_mb']:.1f}MB")


def demo_memory_optimization():
    """メモリ最適化機能のデモ"""
    print("\n=== メモリ最適化機能のデモ ===")

    optimizer = MemoryOptimizer()

    # 大きなオブジェクトを作成
    print("📈 大容量オブジェクト作成...")
    test_data = []
    for i in range(10):
        data = {"id": i, "array": np.random.rand(300, 300), "text": f"Test data {i}" * 100}
        test_data.append(data)

    # 最適化前のメモリ状態
    before_stats = optimizer.get_optimization_stats()
    print(f"💾 最適化前: RAM {before_stats['current_memory']['ram_mb']:.1f}MB")

    # データを削除（参照は残る可能性）
    del test_data

    # メモリ最適化実行
    print("🧹 メモリ最適化実行中...")
    result = optimizer.optimize_memory()

    print(f"✅ 最適化完了: {result['freed_mb']:+.1f}MB解放")

    # 最適化統計
    stats = optimizer.get_optimization_stats()
    opt_stats = stats["optimization_stats"]
    print(f"📊 最適化統計: GC {opt_stats['gc_calls']}回実行")


def demo_memory_context():
    """メモリコンテキスト機能のデモ"""
    print("\n=== メモリコンテキスト機能のデモ ===")

    optimizer = MemoryOptimizer()

    def memory_intensive_operation():
        """メモリ集約的な処理"""
        data = []
        for i in range(20):
            arr = np.random.rand(200, 200)
            data.append(arr)

        # 何らかの処理をシミュレート
        result = sum(arr.mean() for arr in data)
        return result

    # メモリコンテキストを使用
    with optimizer.memory_context() as ctx:
        result = memory_intensive_operation()
        print(f"🔢 処理結果: {result:.3f}")

    print("✅ メモリコンテキスト完了（自動最適化実行済み）")


@memory_efficient
def demo_memory_efficient_decorator():
    """メモリ効率デコレータのデモ"""
    print("\n=== メモリ効率デコレータのデモ ===")

    # デコレータにより自動的にメモリ最適化される
    large_matrix = np.random.rand(800, 800)

    # 行列演算
    result = np.linalg.norm(large_matrix)

    print(f"🔢 行列ノルム: {result:.3f}")
    return result


@optimize_for_large_dataset
def demo_large_dataset_processing(dataset_size: int):
    """大規模データセット処理のデモ"""
    print(f"\n=== 大規模データセット処理のデモ (サイズ: {dataset_size}) ===")

    # 大規模データセットの処理をシミュレート
    processed_items = []

    for i in range(dataset_size):
        # 各アイテムの処理
        item_data = np.random.rand(100, 100)
        processed_value = item_data.mean()
        processed_items.append(processed_value)

        if i % 10 == 0:
            print(f"📝 処理進捗: {i}/{dataset_size}")

    result = np.array(processed_items).mean()
    print(f"✅ 処理完了: 平均値 {result:.6f}")
    return result


def demo_batch_memory_manager():
    """バッチメモリ管理のデモ"""
    print("\n=== バッチメモリ管理のデモ ===")

    def process_item(item_id: int) -> dict:
        """単一アイテムの処理"""
        # 処理をシミュレート
        data = np.random.rand(150, 150)
        result = {"id": item_id, "mean": float(data.mean()), "std": float(data.std())}
        return result

    # バッチメモリ管理を使用
    manager = BatchMemoryManager()

    results = []
    for i in range(25):
        result = manager.process_batch_item(process_item, i)
        results.append(result)

        if i % 5 == 0:
            print(f"📝 バッチ処理: {i}/25")

    # 統計表示
    stats = manager.get_memory_stats()
    print(f"📊 バッチ統計: {stats['processed_items']}アイテム処理")
    print(f"💾 現在のメモリ: {stats['current_memory']['ram_mb']:.1f}MB")

    return results


def main():
    """メイン実行関数"""
    print("💾 P1-015 メモリ使用最適化デモ")

    # 各種デモを実行
    demo_memory_monitoring()
    demo_memory_optimization()
    demo_memory_context()
    demo_memory_efficient_decorator()
    demo_large_dataset_processing(50)
    demo_batch_memory_manager()

    # 最終メモリサマリー
    print(f"\n📊 最終メモリ状態: {get_memory_usage_summary()}")

    print("\n✅ デモ完了！")


if __name__ == "__main__":
    main()
