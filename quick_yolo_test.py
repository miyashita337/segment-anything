#!/usr/bin/env python3
"""
クイックYOLO検出テスト
1枚だけでYOLO閾値を調整
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def quick_yolo_test():
    """1枚の画像で複数閾値テスト"""
    
    # 失敗した画像1枚でテスト
    test_image = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07/kaname07_0001.jpg"
    output_dir = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # より緩い閾値設定
    thresholds = [0.001, 0.005, 0.01, 0.02]  
    
    print(f"⚡ クイックYOLO閾値テスト")
    print(f"テスト画像: {Path(test_image).name}")
    print(f"テスト閾値: {thresholds}")
    
    if not Path(test_image).exists():
        print(f"❌ テスト画像が存在しません: {test_image}")
        return False
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        for threshold in thresholds:
            print(f"\n🔄 閾値 {threshold} テスト中...")
            
            try:
                output_file = Path(output_dir) / f"quick_test_{threshold}.jpg"
                
                start_time = time.time()
                
                result = extract_character_from_path(
                    test_image,
                    output_path=str(output_file),
                    multi_character_criteria='fullbody_priority_enhanced',
                    enhance_contrast=True,
                    filter_text=False,  # 高速化
                    save_mask=False,    # 高速化
                    save_transparent=False,  # 高速化
                    verbose=False,
                    high_quality=False,  # 高速化
                    difficult_pose=False,  # 高速化
                    adaptive_learning=False,  # 高速化
                    manga_mode=True,
                    effect_removal=False,  # 高速化
                    min_yolo_score=threshold  # 閾値変更
                )
                
                proc_time = time.time() - start_time
                
                if result.get('success', False):
                    quality = result.get('quality_score', 0)
                    print(f"   ✅ 成功: 品質={quality:.3f} ({proc_time:.1f}秒)")
                    
                    # この閾値で成功した場合、推奨値として設定
                    print(f"\n🎯 推奨設定発見!")
                    print(f"   min_yolo_score={threshold}")
                    print(f"   処理時間: {proc_time:.1f}秒")
                    print(f"   品質スコア: {quality:.3f}")
                    
                    return threshold
                else:
                    error = result.get('error', '不明')
                    print(f"   ❌ 失敗: {error}")
                    
            except Exception as e:
                print(f"   💥 例外: {str(e)}")
        
        print(f"\n⚠️ すべての閾値で失敗")
        print(f"   より緩い閾値が必要、またはアルゴリズム調整が必要")
        return False
        
    except Exception as e:
        print(f"❌ 致命的エラー: {str(e)}")
        return False

if __name__ == "__main__":
    success = quick_yolo_test()
    if success:
        print(f"\n🎉 最適閾値発見完了")
    else:
        print(f"\n🚨 閾値調整が必要")
        sys.exit(1)