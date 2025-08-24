#!/usr/bin/env python3
"""
リソース管理最適化の使用例
PH2-002: 実際の処理でのリソース管理デモ
"""

import numpy as np
import torch

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from features.common.error_handling import global_error_handler, with_error_handling
from features.common.resource_manager import (
    BatchProcessor,
    ResourceManager,
    cleanup_global_resources,
    managed_resources,
)


def simulate_memory_intensive_task(size_mb: int = 100):
    """メモリ集約的なタスクのシミュレーション"""
    # 指定サイズのデータを作成
    data_size = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    data = np.random.random(data_size).astype(np.float32)
    
    # 何か処理をシミュレート
    result = np.mean(data)
    
    return result


@with_error_handling(global_error_handler)
def example_basic_resource_management():
    """基本的なリソース管理の例"""
    print("\n=== 基本的なリソース管理の例 ===")
    
    # リソースマネージャー作成
    manager = ResourceManager(
        memory_threshold_mb=500,
        auto_cleanup=True
    )
    
    # 現在の使用状況確認
    usage = manager.get_current_usage()
    print(f"現在のメモリ使用量: {usage.memory_mb:.1f}MB ({usage.memory_percent:.1f}%)")
    print(f"CPU使用率: {usage.cpu_percent:.1f}%")
    
    if usage.gpu_memory_mb is not None:
        print(f"GPU メモリ使用量: {usage.gpu_memory_mb:.1f}MB")
    
    # メモリチェック
    required_mb = 200
    print(f"\n{required_mb}MBのメモリが必要です...")
    
    if manager.check_memory_availability(required_mb):
        print("✅ メモリ十分です")
        
        # タスク実行
        result = simulate_memory_intensive_task(150)
        print(f"タスク完了: result={result:.6f}")
    
    # クリーンアップ
    manager.cleanup_memory()
    
    # 使用後の状況
    usage_after = manager.get_current_usage()
    print(f"\nクリーンアップ後のメモリ: {usage_after.memory_mb:.1f}MB")


def example_context_manager():
    """コンテキストマネージャーを使った例"""
    print("\n=== コンテキストマネージャーの例 ===")
    
    # managed_resourcesコンテキストマネージャー使用
    with managed_resources(memory_mb=300, cleanup_on_exit=True) as manager:
        print("リソース管理されたコンテキスト内")
        
        # 複数のタスクを実行
        for i in range(3):
            print(f"\nタスク {i+1}/3 実行中...")
            result = simulate_memory_intensive_task(50)
            
            # 使用状況モニタリング
            usage = manager.get_current_usage()
            print(f"  メモリ: {usage.memory_mb:.1f}MB")
    
    print("\nコンテキスト終了 - 自動クリーンアップ完了")


def example_batch_processing():
    """バッチ処理最適化の例"""
    print("\n=== バッチ処理最適化の例 ===")
    
    # バッチプロセッサー作成
    batch_processor = BatchProcessor(
        initial_batch_size=16,
        min_batch_size=1,
        max_batch_size=64,
        memory_fraction=0.7
    )
    
    # アイテムあたりのメモリ使用量（MB）
    item_memory_mb = 10
    
    # 最適なバッチサイズ取得
    optimal_batch_size = batch_processor.get_optimal_batch_size(item_memory_mb)
    print(f"最適バッチサイズ: {optimal_batch_size}")
    
    # バッチ処理シミュレーション
    total_items = 100
    processed = 0
    
    while processed < total_items:
        batch_size = min(batch_processor.current_batch_size, total_items - processed)
        
        try:
            print(f"\nバッチ処理: {batch_size}アイテム")
            
            # バッチ処理実行
            simulate_memory_intensive_task(batch_size * item_memory_mb)
            
            processed += batch_size
            print(f"✅ 成功 (進捗: {processed}/{total_items})")
            
            # 成功したのでバッチサイズ調整
            batch_processor.adjust_batch_size(success=True)
            
        except MemoryError:
            print(f"❌ メモリエラー - バッチサイズ縮小")
            batch_processor.adjust_batch_size(success=False, memory_error=True)
        
        except Exception as e:
            print(f"❌ エラー: {e}")
            batch_processor.adjust_batch_size(success=False)
    
    print(f"\n処理完了: {processed}アイテム")


def example_resource_monitoring():
    """リソースモニタリングの例"""
    print("\n=== リソースモニタリングの例 ===")
    
    manager = ResourceManager(
        memory_threshold_mb=300,
        monitoring_interval=2.0,
        auto_cleanup=True
    )
    
    # カスタムクリーンアップコールバック登録
    def custom_cleanup():
        print("  🧹 カスタムクリーンアップ実行")
    
    manager.register_cleanup_callback(custom_cleanup)
    
    # モニタリング開始
    manager.start_monitoring()
    print("モニタリング開始（10秒間）...")
    
    # 10秒間、定期的にメモリを使用
    for i in range(5):
        time.sleep(2)
        print(f"\n[{i+1}/5] メモリ集約タスク実行")
        simulate_memory_intensive_task(100)
        
        # 現在のサマリー表示
        summary = manager.get_usage_summary()
        if summary:
            print(f"  平均CPU: {summary['average']['cpu_percent']:.1f}%")
            print(f"  平均メモリ: {summary['average']['memory_mb']:.1f}MB")
    
    # モニタリング停止
    manager.stop_monitoring()
    
    # レポート保存
    report_path = Path("resource_usage_report.json")
    manager.save_usage_report(report_path)
    print(f"\n📊 使用状況レポート保存: {report_path}")


def example_gpu_resource_management():
    """GPU リソース管理の例"""
    print("\n=== GPU リソース管理の例 ===")
    
    if not torch.cuda.is_available():
        print("❌ GPUが利用できません")
        return
    
    manager = ResourceManager(
        gpu_memory_threshold_mb=1000,
        auto_cleanup=True
    )
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # GPU メモリ使用前
    usage_before = manager.get_current_usage()
    print(f"GPU メモリ使用量（前）: {usage_before.gpu_memory_mb:.1f}MB")
    
    # GPU でテンソル作成
    size = 1000
    device = torch.device('cuda')
    
    tensors = []
    for i in range(5):
        tensor = torch.randn(size, size, device=device)
        tensors.append(tensor)
        
        usage = manager.get_current_usage()
        print(f"テンソル {i+1} 作成後: {usage.gpu_memory_mb:.1f}MB")
    
    # クリーンアップ
    print("\nGPU メモリクリーンアップ実行...")
    del tensors
    manager.cleanup_memory()
    
    # GPU メモリ使用後
    usage_after = manager.get_current_usage()
    print(f"GPU メモリ使用量（後）: {usage_after.gpu_memory_mb:.1f}MB")


def main():
    """メイン処理"""
    print("🔧 リソース管理最適化デモ")
    print("=" * 60)
    
    # 各例を実行
    example_basic_resource_management()
    example_context_manager()
    example_batch_processing()
    example_resource_monitoring()
    
    if torch.cuda.is_available():
        example_gpu_resource_management()
    
    # グローバルクリーンアップ
    print("\n🧹 グローバルリソースクリーンアップ...")
    cleanup_global_resources()
    
    print("\n✅ 全ての例が完了しました")


if __name__ == "__main__":
    main()