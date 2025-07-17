#!/usr/bin/env python3
"""
最適化設定での最終品質確認テスト
YOLO閾値0.001で5枚テスト
"""

import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def optimized_final_test():
    """最適化設定での最終品質確認"""
    
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    print(f"🚀 最適化設定での最終品質確認テスト")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    print(f"最適化: min_yolo_score=0.001")
    
    # パス確認
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)
    
    # 画像数確認
    image_files = list(Path(input_path).glob("*.jpg"))
    print(f"📊 検出画像: {len(image_files)}枚")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 最初の5枚で最適化テスト
    test_images = image_files[:5]
    print(f"🧪 テスト対象: {len(test_images)}枚（最適化設定）")
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        total_start = time.time()
        
        for i, image_file in enumerate(test_images, 1):
            print(f"\\n🔄 [{i}/{len(test_images)}] テスト: {image_file.name}")
            
            try:
                output_file = Path(output_path) / f"optimized_test_{i:02d}_{image_file.stem}.jpg"
                
                start_time = time.time()
                
                # 最適化設定（YOLO閾値0.001）
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file),
                    multi_character_criteria='fullbody_priority_enhanced',  # P1-003
                    enhance_contrast=True,
                    filter_text=True,
                    save_mask=True,
                    save_transparent=True,
                    verbose=False,
                    high_quality=True,
                    difficult_pose=True,
                    adaptive_learning=True,
                    manga_mode=True,
                    effect_removal=True,
                    min_yolo_score=0.001  # 最適化された閾値
                )
                
                proc_time = time.time() - start_time
                
                if result.get('success', False):
                    success_count += 1
                    print(f"✅ 成功 ({proc_time:.1f}秒)")
                    
                    # Phase 1品質メトリクス表示
                    quality = result.get('quality_score', 0)
                    print(f"   品質スコア: {quality:.3f}")
                    
                    if 'extraction_analysis' in result:
                        analysis = result['extraction_analysis']
                        completeness = analysis.get('completeness_score', 0)
                        print(f"   P1-002完全性: {completeness:.3f}")
                        
                        # 検出部位確認
                        features = []
                        if analysis.get('has_face', False):
                            features.append("顔")
                        if analysis.get('has_torso', False):
                            features.append("胴体")
                        if analysis.get('has_limbs', False):
                            features.append("手足")
                        print(f"   検出部位: {', '.join(features) if features else 'なし'}")
                    
                    # 生成ファイル確認
                    output_files = list(Path(output_path).glob(f"optimized_test_{i:02d}_*"))
                    print(f"   生成ファイル: {len(output_files)}個")
                    
                else:
                    print(f"❌ 失敗: {result.get('error', '不明')}")
                    
            except Exception as e:
                print(f"❌ 例外: {str(e)}")
        
        total_time = time.time() - total_start
        
        print(f"\\n📊 最適化設定テスト結果:")
        print(f"   テスト実行: {len(test_images)}枚")
        print(f"   成功: {success_count}枚") 
        print(f"   成功率: {success_count/len(test_images)*100:.1f}%")
        print(f"   総処理時間: {total_time:.1f}秒")
        print(f"   平均処理時間: {total_time/len(test_images):.1f}秒/画像")
        
        # 全体画像数での推定
        estimated_total_time = (total_time / len(test_images)) * len(image_files)
        estimated_success = len(image_files) * success_count / len(test_images)
        
        print(f"\\n📈 全体処理推定（最適化設定）:")
        print(f"   全画像数: {len(image_files)}枚")
        print(f"   推定処理時間: {estimated_total_time/60:.1f}分")
        print(f"   推定成功数: {estimated_success:.0f}枚")
        print(f"   推定成功率: {success_count/len(test_images)*100:.1f}%")
        
        # 実行可能性の判定
        if success_count >= len(test_images) * 0.8:  # 80%以上成功
            print(f"\\n✨ Phase 1最適化システム動作確認: 優秀")
            print(f"   指定パスでの高品質バッチ処理の実行を強く推奨")
            
            print(f"\\n🚀 実行推奨コマンド:")
            print(f"   python3 run_kaname07_highest_quality.py")
            print(f"   予想処理時間: {estimated_total_time/60:.0f}分")
            print(f"   予想成功数: {estimated_success:.0f}枚")
            
            return True
            
        elif success_count >= len(test_images) * 0.6:  # 60%以上成功
            print(f"\\n✅ Phase 1最適化システム動作確認: 良好")
            print(f"   実用可能レベル、バッチ処理実行可能")
            return True
        else:
            print(f"\\n⚠️ Phase 1最適化システム動作確認: 要改善")
            print(f"   さらなる調整が必要")
            return False
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False

if __name__ == "__main__":
    try:
        success = optimized_final_test()
        if success:
            print(f"\\n🎉 Phase 1最適化確認完了: バッチ処理実行可能")
        else:
            print(f"\\n🚨 Phase 1最適化確認: さらなる調整が必要")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\\n⏹️ ユーザー中断")
        sys.exit(0)