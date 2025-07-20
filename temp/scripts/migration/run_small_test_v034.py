#!/usr/bin/env python3
"""
v0.3.4小規模テスト
最初の10枚で v0.3.4 の動作確認
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_small_test_v034():
    """v0.3.4小規模テスト実行（最初の10枚）"""
    
    # パス設定
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname09"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname09_0_3_4_test"
    
    print("🚀 v0.3.4小規模テスト実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    print("✨ 処理対象: 最初の10枚")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル取得（最初の10枚）
    all_image_files = sorted(list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png")))
    image_files = all_image_files[:10]  # 最初の10枚のみ
    
    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")
    
    # バッチ処理実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")
            
            try:
                image_start = time.time()
                
                # 出力ファイル名設定（入力ファイル名と同じ）
                output_filename = image_file.name
                output_file_path = Path(output_path) / output_filename
                
                # シンプルな設定での抽出実行
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file_path),
                    multi_character_criteria='balanced',
                    enhance_contrast=True,
                    save_mask=True,
                    save_transparent=True,
                    verbose=False,
                    high_quality=True,
                    min_yolo_score=0.01,
                )
                
                image_time = time.time() - image_start
                
                if result.get('success', False):
                    success_count += 1
                    print(f"✅ 成功: {output_filename}")
                    if 'quality_score' in result:
                        print(f"   品質スコア: {result['quality_score']:.3f}")
                    print(f"   処理時間: {image_time:.2f}秒")
                else:
                    error_count += 1
                    error_msg = result.get('error', '不明なエラー')
                    print(f"❌ 失敗: {output_filename} - {error_msg}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 処理エラー: {image_file.name} - {str(e)}")
        
        # 結果サマリー
        total_time = time.time() - start_time
        success_rate = success_count / len(image_files) * 100
        
        print("\n" + "="*60)
        print("📊 v0.3.4小規模テスト結果")
        print("="*60)
        print(f"処理数: {len(image_files)}枚")
        print(f"成功: {success_count}枚")
        print(f"失敗: {error_count}枚")
        print(f"成功率: {success_rate:.1f}%")
        print(f"総処理時間: {total_time:.1f}秒")
        
        print(f"✅ v0.3.4小規模テスト完了")
        
        return success_rate >= 50.0  # 50%以上で成功判定
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")
        return False


def main():
    """メイン実行関数"""
    print("🚀 v0.3.4小規模テスト開始")
    
    success = run_small_test_v034()
    
    if success:
        print("\n✅ 小規模テスト成功!")
        sys.exit(0)
    else:
        print("\n❌ 小規模テスト失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()