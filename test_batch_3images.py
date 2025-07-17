#!/usr/bin/env python3
"""
Phase 1バッチ動作確認テスト（3枚限定）
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_batch_3images():
    """3枚限定でのバッチ動作確認"""
    
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # 最初の3枚でテスト
    image_files = list(Path(input_path).glob("*.jpg"))[:3]
    
    print(f"🧪 Phase 1バッチテスト実行（3枚限定）")
    print(f"テスト画像: {len(image_files)}枚")
    for img in image_files:
        print(f"  - {img.name}")
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")
            
            try:
                image_start = time.time()
                
                output_file = Path(output_path) / f"batch_test_{i:02d}_{image_file.stem}.jpg"
                
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file),
                    multi_character_criteria='fullbody_priority_enhanced',
                    enhance_contrast=True,
                    filter_text=True,
                    save_mask=True,
                    save_transparent=True,
                    verbose=False,  # バッチなので詳細ログ抑制
                    high_quality=True,
                    difficult_pose=True,
                    adaptive_learning=True,
                    manga_mode=True,
                    effect_removal=True,
                    min_yolo_score=0.05
                )
                
                image_time = time.time() - image_start
                
                if result.get('success', False):
                    success_count += 1
                    print(f"✅ 成功: {output_file.name} ({image_time:.1f}秒)")
                    
                    # Phase 1機能動作確認
                    if 'quality_score' in result:
                        print(f"   品質: {result['quality_score']:.3f}")
                    
                    if 'extraction_analysis' in result:
                        analysis = result['extraction_analysis']
                        completeness = analysis.get('completeness_score', 0)
                        print(f"   完全性: {completeness:.3f}")
                        
                        features = []
                        if analysis.get('has_face', False):
                            features.append("顔")
                        if analysis.get('has_torso', False):
                            features.append("胴体")
                        if analysis.get('has_limbs', False):
                            features.append("手足")
                        print(f"   検出: {', '.join(features) if features else 'なし'}")
                    
                else:
                    error_count += 1
                    print(f"❌ 失敗: {image_file.name}")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ 例外: {image_file.name} - {str(e)}")
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Phase 1バッチテスト結果:")
        print(f"   処理数: {len(image_files)}")
        print(f"   成功: {success_count}")
        print(f"   失敗: {error_count}")
        print(f"   成功率: {success_count/len(image_files)*100:.1f}%")
        print(f"   総時間: {total_time:.1f}秒")
        print(f"   平均時間: {total_time/len(image_files):.1f}秒/画像")
        
        # 生成ファイル確認
        generated_files = list(Path(output_path).glob("batch_test_*"))
        print(f"   生成ファイル: {len(generated_files)}個")
        
        return success_count == len(image_files)
        
    except Exception as e:
        print(f"❌ 致命的エラー: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_batch_3images()
    if success:
        print("\n✨ Phase 1バッチテスト成功!")
    else:
        print("\n💥 Phase 1バッチテスト失敗!")
        sys.exit(1)