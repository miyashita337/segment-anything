#!/usr/bin/env python3
"""
修正版パフォーマンステスト
API戻り値型統一対応版
"""

import time
import psutil
import sys
from pathlib import Path
sys.path.append(str(Path('.')))

from features.evaluation.enhanced_detection_systems import EnhancedFaceDetector, EnhancedPoseDetector
import cv2

def run_performance_test():
    """修正版パフォーマンステスト実行"""
    print('パフォーマンステスト開始（修正版）')
    print('='*50)

    # テスト画像読み込み
    test_image_path = '/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix/kana08_0000_cover.jpg'
    image = cv2.imread(test_image_path)
    if image is None:
        print('テスト画像読み込み失敗')
        return False

    print(f'テスト画像: {test_image_path}')
    print(f'画像サイズ: {image.shape}')

    # メモリ使用量測定開始
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 顔検出パフォーマンステスト
    print('\n顔検出パフォーマンステスト:')
    face_detector = EnhancedFaceDetector()

    face_times = []
    face_detections = []
    for i in range(3):
        start_time = time.time()
        face_result = face_detector.detect_faces_comprehensive(image, efficient_mode=True)
        elapsed = time.time() - start_time
        face_times.append(elapsed)
        
        # 戻り値の型確認
        detected = len(face_result) > 0 if isinstance(face_result, list) else bool(face_result.detected) if hasattr(face_result, 'detected') else False
        face_detections.append(detected)
        
        print(f'  テスト{i+1}: 検出数={len(face_result) if isinstance(face_result, list) else 1}, 処理時間={elapsed:.3f}秒')

    avg_face_time = sum(face_times) / len(face_times)
    face_detection_rate = sum(face_detections) / len(face_detections)
    print(f'  平均処理時間: {avg_face_time:.3f}秒')
    print(f'  検出成功率: {face_detection_rate:.1%}')

    # ポーズ検出パフォーマンステスト  
    print('\nポーズ検出パフォーマンステスト:')
    pose_detector = EnhancedPoseDetector()

    pose_times = []
    pose_detections = []
    for i in range(3):
        start_time = time.time()
        pose_result = pose_detector.detect_pose_comprehensive(image, efficient_mode=True)
        elapsed = time.time() - start_time
        pose_times.append(elapsed)
        
        # PoseDetectionResultの場合
        detected = pose_result.detected if hasattr(pose_result, 'detected') else False
        pose_detections.append(detected)
        
        print(f'  テスト{i+1}: 検出={detected}, 処理時間={elapsed:.3f}秒')

    avg_pose_time = sum(pose_times) / len(pose_times)
    pose_detection_rate = sum(pose_detections) / len(pose_detections)
    print(f'  平均処理時間: {avg_pose_time:.3f}秒')
    print(f'  検出成功率: {pose_detection_rate:.1%}')

    # メモリ使用量測定終了
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print('\nメモリ使用量測定:')
    print(f'  初期メモリ: {initial_memory:.1f} MB')
    print(f'  最終メモリ: {final_memory:.1f} MB')
    print(f'  増加量: {memory_increase:.1f} MB')

    # パフォーマンス要件チェック
    print('\nパフォーマンス要件チェック:')
    total_time = avg_face_time + avg_pose_time
    target_time = 10.0  # 目標: 10秒以下
    memory_ok = memory_increase < 1000  # 1GB以下
    
    print(f'  統合処理時間: {total_time:.3f}秒')
    print(f'  目標時間: {target_time}秒以下')
    print(f'  時間要件達成: {"YES" if total_time <= target_time else "NO"}')
    print(f'  メモリ増加: {memory_increase:.1f}MB')
    print(f'  メモリ要件達成: {"YES" if memory_ok else "NO"}')
    
    # 検出率要件チェック
    print(f'  顔検出率: {face_detection_rate:.1%} (目標90%以上)')
    print(f'  ポーズ検出率: {pose_detection_rate:.1%} (目標80%以上)')
    
    face_ok = face_detection_rate >= 0.90
    pose_ok = pose_detection_rate >= 0.80
    
    print(f'  顔検出要件達成: {"YES" if face_ok else "NO"}')
    print(f'  ポーズ検出要件達成: {"YES" if pose_ok else "NO"}')

    # 総合判定
    all_ok = (total_time <= target_time) and memory_ok and face_ok and pose_ok
    print(f'\n総合要件達成: {"✅ YES" if all_ok else "❌ NO"}')

    print('\n✅ パフォーマンステスト完了（修正版）')
    return all_ok

if __name__ == "__main__":
    try:
        success = run_performance_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)