#!/usr/bin/env python3
"""
YOLO閾値最適化テスト
アニメキャラクター検出に最適な閾値を発見
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_yolo_thresholds():
    """複数の閾値でYOLO検出テスト"""
    
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # テスト用画像（失敗した3枚を含む）
    test_images = [
        "kaname07_0001.jpg",  # 失敗
        "kaname07_0002.jpg",  # 失敗
        "kaname07_0003.jpg"   # 失敗
    ]
    
    # テスト閾値（より緩い設定）
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
    
    print(f"🎯 YOLO閾値最適化テスト開始")
    print(f"テスト画像: {len(test_images)}枚")
    print(f"テスト閾値: {thresholds}")
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        results = {}
        
        for threshold in thresholds:
            print(f"\n🔄 閾値 {threshold} でテスト中...")
            success_count = 0
            
            for i, image_name in enumerate(test_images, 1):
                image_file = Path(input_path) / image_name
                
                if not image_file.exists():
                    print(f"   ⚠️ ファイル不存在: {image_name}")
                    continue
                
                try:
                    output_file = Path(output_path) / f"threshold_{threshold}_{i:02d}_{image_file.stem}.jpg"
                    
                    start_time = time.time()
                    
                    result = extract_character_from_path(
                        str(image_file),
                        output_path=str(output_file),
                        multi_character_criteria='fullbody_priority_enhanced',
                        enhance_contrast=True,
                        filter_text=True,
                        save_mask=False,  # 高速化のためマスク保存無効
                        save_transparent=False,  # 高速化のため透明版無効
                        verbose=False,
                        high_quality=True,
                        difficult_pose=True,
                        adaptive_learning=True,
                        manga_mode=True,
                        effect_removal=True,
                        min_yolo_score=threshold  # 閾値変更
                    )
                    
                    proc_time = time.time() - start_time
                    
                    if result.get('success', False):
                        success_count += 1
                        print(f"   ✅ {image_name}: 成功 ({proc_time:.1f}秒)")
                    else:
                        print(f"   ❌ {image_name}: 失敗 - {result.get('error', '不明')}")
                        
                except Exception as e:
                    print(f"   💥 {image_name}: 例外 - {str(e)}")
            
            success_rate = success_count / len(test_images) * 100
            results[threshold] = {
                'success_count': success_count,
                'success_rate': success_rate
            }
            
            print(f"   📊 閾値 {threshold}: {success_count}/{len(test_images)} ({success_rate:.1f}%)")
        
        # 最適閾値決定
        print(f"\n📈 YOLO閾値最適化結果:")
        print(f"{'閾値':<8} {'成功数':<6} {'成功率':<8}")
        print(f"{'-'*25}")
        
        best_threshold = None
        best_rate = 0
        
        for threshold, data in results.items():
            rate = data['success_rate']
            count = data['success_count']
            print(f"{threshold:<8} {count:<6} {rate:<8.1f}%")
            
            if rate > best_rate:
                best_rate = rate
                best_threshold = threshold
        
        print(f"\n🎯 推奨設定:")
        if best_threshold:
            print(f"   最適閾値: {best_threshold}")
            print(f"   成功率: {best_rate:.1f}%")
            
            if best_rate >= 80:
                print(f"   ✨ 優秀な結果 - バッチ処理推奨")
            elif best_rate >= 60:
                print(f"   ✅ 良好な結果 - 実用可能")
            else:
                print(f"   ⚠️ 要改善 - さらなる調整が必要")
        else:
            print(f"   ❌ すべての閾値で失敗 - アルゴリズム見直しが必要")
        
        return best_threshold, best_rate
        
    except Exception as e:
        print(f"❌ 致命的エラー: {str(e)}")
        return None, 0

if __name__ == "__main__":
    try:
        threshold, rate = test_yolo_thresholds()
        if threshold and rate >= 60:
            print(f"\n🎉 YOLO閾値最適化成功: {threshold} ({rate:.1f}%)")
        else:
            print(f"\n🚨 YOLO閾値最適化失敗")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ ユーザー中断")
        sys.exit(0)