#!/usr/bin/env python3
"""
Phase 1単体画像動作確認テスト
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_single_image_phase1():
    """単体画像でのPhase 1動作確認"""
    
    # テスト画像選択
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # 最初の画像でテスト
    image_files = list(Path(input_path).glob("*.jpg"))
    if not image_files:
        print("❌ テスト画像が見つかりません")
        return False
    
    test_image = image_files[0]
    print(f"🧪 Phase 1単体テスト実行")
    print(f"テスト画像: {test_image.name}")
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        output_file = Path(output_path) / f"test_phase1_{test_image.stem}.jpg"
        
        print("📋 実行設定:")
        print("  - fullbody_priority_enhanced (P1-003)")
        print("  - enhanced_screentone (P1-004)")
        print("  - mosaic_boundary (P1-005)")
        print("  - solid_fill_enhancement (P1-006)")
        print("  - partial_extraction_check (P1-002)")
        
        start_time = time.time()
        
        # Phase 1最高品質設定での抽出実行
        result = extract_character_from_path(
            str(test_image),
            output_path=str(output_file),
            multi_character_criteria='fullbody_priority_enhanced',  # P1-003
            enhance_contrast=True,
            filter_text=True,
            save_mask=True,
            save_transparent=True,
            verbose=True,  # 詳細ログ
            high_quality=True,
            difficult_pose=True,
            adaptive_learning=True,
            manga_mode=True,
            effect_removal=True,
            min_yolo_score=0.05
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n⏱️  処理時間: {processing_time:.2f}秒")
        
        if result.get('success', False):
            print(f"✅ 成功: {output_file.name}")
            
            # 詳細結果表示
            if 'quality_score' in result:
                print(f"   品質スコア: {result['quality_score']:.3f}")
            
            if 'extraction_analysis' in result:
                analysis = result['extraction_analysis']
                print(f"   完全性スコア: {analysis.get('completeness_score', 0):.3f}")
                print(f"   顔検出: {analysis.get('has_face', False)}")
                print(f"   胴体検出: {analysis.get('has_torso', False)}")
                print(f"   手足検出: {analysis.get('has_limbs', False)}")
            
            # 出力ファイル確認
            output_files = list(Path(output_path).glob(f"test_phase1_{test_image.stem}*"))
            print(f"   生成ファイル: {len(output_files)}個")
            for f in output_files:
                print(f"     - {f.name}")
            
            return True
        else:
            print(f"❌ 失敗: {result.get('error', '不明なエラー')}")
            return False
            
    except Exception as e:
        print(f"❌ 例外: {str(e)}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_single_image_phase1()
    if success:
        print("\n✨ Phase 1動作確認テスト成功!")
    else:
        print("\n💥 Phase 1動作確認テスト失敗!")
        sys.exit(1)