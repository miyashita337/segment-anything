#!/usr/bin/env python3
"""
スケーラビリティ統合例
PH2-002: 既存パイプラインへのスケーラビリティ改善統合デモ
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from features.common.scalability import (
    ParallelProcessor, AsyncProcessor, PipelineProcessor,
    GPUParallelProcessor, ScalabilityManager,
    ProcessingTask, TaskResult
)
from features.common.resource_manager import (
    ResourceManager, managed_resources, BatchProcessor
)
from features.common.error_handling import (
    global_error_handler, with_error_handling
)


def simulate_image_preprocessing(image_path: str) -> Dict[str, Any]:
    """画像前処理のシミュレーション"""
    # 実際の処理時間をシミュレート
    time.sleep(0.1 + np.random.random() * 0.05)
    
    return {
        'path': image_path,
        'shape': (1024, 1024, 3),
        'processed': True,
        'processing_time': np.random.random() * 0.1
    }


def simulate_yolo_detection(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """YOLO検出のシミュレーション"""
    time.sleep(0.2 + np.random.random() * 0.1)
    
    return {
        'detections': [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.8}
        ],
        'detection_time': np.random.random() * 0.2,
        'source': preprocessed_data['path']
    }


def simulate_sam_segmentation(detection_data: Dict[str, Any]) -> Dict[str, Any]:
    """SAMセグメンテーションのシミュレーション"""
    time.sleep(0.3 + np.random.random() * 0.15)
    
    masks = []
    for detection in detection_data['detections']:
        masks.append({
            'mask': np.random.random((256, 256)) > 0.5,
            'quality_score': np.random.random(),
            'bbox': detection['bbox']
        })
    
    return {
        'masks': masks,
        'segmentation_time': np.random.random() * 0.3,
        'source': detection_data['source']
    }


async def async_file_operation(filepath: str) -> Dict[str, Any]:
    """非同期ファイル操作のシミュレーション"""
    import asyncio
    
    # I/O待機をシミュレート
    await asyncio.sleep(0.05 + np.random.random() * 0.02)
    
    return {
        'filepath': filepath,
        'size_mb': np.random.randint(1, 10),
        'loaded': True
    }


def create_test_tensor_batch(batch_size: int = 8) -> List[torch.Tensor]:
    """テスト用テンソルバッチ作成"""
    tensors = []
    for i in range(batch_size):
        tensor = torch.randn(3, 224, 224)
        tensors.append(tensor)
    return tensors


def gpu_model_inference(batch_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """GPU モデル推論のシミュレーション"""
    # バッチテンソルをスタック
    if isinstance(batch_tensors, list):
        batch = torch.stack(batch_tensors)
    else:
        batch = batch_tensors
    
    # 簡単な処理をシミュレート
    result = torch.nn.functional.avg_pool2d(batch, kernel_size=2)
    
    # リストに戻す
    return [result[i] for i in range(result.size(0))]


@with_error_handling(global_error_handler)
def example_parallel_preprocessing():
    """並列前処理の例"""
    print("\n=== 並列前処理の例 ===")
    
    # テスト画像パスリスト
    image_paths = [f"test_image_{i:03d}.jpg" for i in range(50)]
    
    # 並列プロセッサー作成
    parallel_processor = ParallelProcessor(
        max_workers=4,
        use_processes=True,
        chunk_size=10
    )
    
    # 並列処理実行
    start_time = time.time()
    results = parallel_processor.process_batch(
        simulate_image_preprocessing,
        image_paths
    )
    total_time = time.time() - start_time
    
    # 結果分析
    success_count = sum(1 for r in results if r.success)
    avg_processing_time = sum(r.execution_time for r in results if r.success) / success_count
    
    print(f"✅ 並列前処理完了:")
    print(f"  - 処理画像数: {len(image_paths)}")
    print(f"  - 成功数: {success_count}/{len(image_paths)}")
    print(f"  - 総処理時間: {total_time:.2f}秒")
    print(f"  - 平均処理時間: {avg_processing_time:.3f}秒/画像")
    print(f"  - スループット: {success_count/total_time:.1f}画像/秒")
    
    return results


@with_error_handling(global_error_handler)
async def example_async_file_loading():
    """非同期ファイル読み込みの例"""
    print("\n=== 非同期ファイル読み込みの例 ===")
    
    # ファイルパスリスト
    filepaths = [f"data/file_{i:03d}.txt" for i in range(30)]
    
    # 非同期プロセッサー作成
    async_processor = AsyncProcessor(max_concurrent=8)
    
    # 非同期処理実行
    start_time = time.time()
    results = await async_processor.process_async_batch(
        async_file_operation,
        filepaths
    )
    total_time = time.time() - start_time
    
    # 結果分析
    success_count = sum(1 for r in results if r.success)
    
    print(f"✅ 非同期ファイル読み込み完了:")
    print(f"  - 処理ファイル数: {len(filepaths)}")
    print(f"  - 成功数: {success_count}/{len(filepaths)}")
    print(f"  - 総処理時間: {total_time:.2f}秒")
    print(f"  - スループット: {success_count/total_time:.1f}ファイル/秒")
    
    return results


@with_error_handling(global_error_handler)
def example_pipeline_processing():
    """パイプライン処理の例"""
    print("\n=== パイプライン処理の例 ===")
    
    # パイプラインステージ定義
    stages = [
        simulate_image_preprocessing,
        simulate_yolo_detection,
        simulate_sam_segmentation
    ]
    
    # パイプラインプロセッサー作成
    pipeline_processor = PipelineProcessor(
        stages=stages,
        buffer_size=20,
        max_workers_per_stage=2
    )
    
    # 入力データ準備
    inputs = [f"pipeline_image_{i:03d}.jpg" for i in range(20)]
    
    # パイプライン処理実行
    start_time = time.time()
    results = pipeline_processor.process_pipeline(inputs)
    total_time = time.time() - start_time
    
    print(f"✅ パイプライン処理完了:")
    print(f"  - 入力数: {len(inputs)}")
    print(f"  - 出力数: {len(results)}")
    print(f"  - 総処理時間: {total_time:.2f}秒")
    print(f"  - スループット: {len(results)/total_time:.1f}アイテム/秒")
    
    return results


@with_error_handling(global_error_handler)
def example_gpu_batch_processing():
    """GPU バッチ処理の例"""
    print("\n=== GPU バッチ処理の例 ===")
    
    if not torch.cuda.is_available():
        print("❌ GPUが利用できません - スキップします")
        return []
    
    # GPU プロセッサー作成
    gpu_processor = GPUParallelProcessor(batch_size=16)
    
    # テストテンソル作成
    test_tensors = create_test_tensor_batch(64)
    
    # GPU バッチ処理実行
    start_time = time.time()
    results = gpu_processor.process_tensors_batch(
        gpu_model_inference,
        test_tensors
    )
    total_time = time.time() - start_time
    
    # 結果分析
    success_count = sum(1 for r in results if r.success)
    
    print(f"✅ GPU バッチ処理完了:")
    print(f"  - 入力テンソル数: {len(test_tensors)}")
    print(f"  - 成功バッチ数: {success_count}/{len(results)}")
    print(f"  - 総処理時間: {total_time:.2f}秒")
    print(f"  - スループット: {len(test_tensors)/total_time:.1f}テンソル/秒")
    
    return results


@with_error_handling(global_error_handler)
def example_optimal_strategy_selection():
    """最適戦略選択の例"""
    print("\n=== 最適戦略選択の例 ===")
    
    scalability_manager = ScalabilityManager()
    
    # 異なるタスクタイプでの戦略選択テスト
    test_cases = [
        {
            'task_type': 'image_preprocessing',
            'data_size': 100,
            'memory_intensive': False,
            'io_intensive': True,
            'gpu_compatible': False
        },
        {
            'task_type': 'model_inference',
            'data_size': 50,
            'memory_intensive': True,
            'io_intensive': False,
            'gpu_compatible': True
        },
        {
            'task_type': 'file_operations',
            'data_size': 200,
            'memory_intensive': False,
            'io_intensive': True,
            'gpu_compatible': False
        },
        {
            'task_type': 'data_analysis',
            'data_size': 1000,
            'memory_intensive': True,
            'io_intensive': False,
            'gpu_compatible': False
        }
    ]
    
    print("最適戦略選択結果:")
    for case in test_cases:
        strategy = scalability_manager.choose_optimal_processing_strategy(**case)
        print(f"  - {case['task_type']}: {strategy}")
    
    # パフォーマンス推奨事項
    recommendations = scalability_manager.get_performance_recommendations()
    if recommendations:
        print("\nパフォーマンス改善推奨事項:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("\n✅ 現在のリソース使用状況は最適です")


@with_error_handling(global_error_handler)
def performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")
    
    # テストデータ準備
    test_data = [f"test_item_{i:03d}" for i in range(100)]
    
    # シーケンシャル処理
    print("\n1. シーケンシャル処理:")
    start_time = time.time()
    sequential_results = []
    for item in test_data:
        result = simulate_image_preprocessing(item)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"  時間: {sequential_time:.2f}秒")
    print(f"  スループット: {len(test_data)/sequential_time:.1f}アイテム/秒")
    
    # 並列処理（スレッド）
    print("\n2. 並列処理（スレッド）:")
    parallel_thread = ParallelProcessor(max_workers=4, use_processes=False)
    start_time = time.time()
    thread_results = parallel_thread.process_batch(
        simulate_image_preprocessing,
        test_data
    )
    thread_time = time.time() - start_time
    success_count = sum(1 for r in thread_results if r.success)
    print(f"  時間: {thread_time:.2f}秒")
    print(f"  スループット: {success_count/thread_time:.1f}アイテム/秒")
    print(f"  改善率: {sequential_time/thread_time:.2f}x")
    
    # 並列処理（プロセス）
    print("\n3. 並列処理（プロセス）:")
    parallel_process = ParallelProcessor(max_workers=4, use_processes=True)
    start_time = time.time()
    process_results = parallel_process.process_batch(
        simulate_image_preprocessing,
        test_data
    )
    process_time = time.time() - start_time
    success_count = sum(1 for r in process_results if r.success)
    print(f"  時間: {process_time:.2f}秒")
    print(f"  スループット: {success_count/process_time:.1f}アイテム/秒")
    print(f"  改善率: {sequential_time/process_time:.2f}x")
    
    # 最適戦略比較
    print(f"\n最適戦略:")
    if thread_time < process_time:
        print(f"  スレッド並列が最適 ({thread_time:.2f}秒)")
    else:
        print(f"  プロセス並列が最適 ({process_time:.2f}秒)")


async def main():
    """メイン処理"""
    print("🚀 スケーラビリティ統合デモ")
    print("=" * 60)
    
    # リソース管理されたコンテキストで実行
    with managed_resources(memory_mb=1000, cleanup_on_exit=True) as manager:
        print(f"初期リソース使用量:")
        usage = manager.get_current_usage()
        print(f"  - CPU: {usage.cpu_percent:.1f}%")
        print(f"  - メモリ: {usage.memory_mb:.1f}MB ({usage.memory_percent:.1f}%)")
        if usage.gpu_memory_mb:
            print(f"  - GPU: {usage.gpu_memory_mb:.1f}MB")
        
        # 各例を実行
        example_parallel_preprocessing()
        await example_async_file_loading()
        example_pipeline_processing()
        example_gpu_batch_processing()
        example_optimal_strategy_selection()
        performance_comparison()
        
        # 最終リソース状況
        print(f"\n最終リソース使用量:")
        final_usage = manager.get_current_usage()
        print(f"  - CPU: {final_usage.cpu_percent:.1f}%")
        print(f"  - メモリ: {final_usage.memory_mb:.1f}MB ({final_usage.memory_percent:.1f}%)")
        if final_usage.gpu_memory_mb:
            print(f"  - GPU: {final_usage.gpu_memory_mb:.1f}MB")
    
    print("\n✅ 全ての統合例が完了しました")
    print("\n📊 スケーラビリティ改善の効果:")
    print("  • 並列処理により最大4倍の性能向上")
    print("  • 非同期処理でI/O待機時間を大幅削減")
    print("  • パイプライン処理で連続処理の効率化")
    print("  • GPU バッチ処理で大量データの高速処理")
    print("  • 自動戦略選択でタスクに最適な処理方式を選択")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())