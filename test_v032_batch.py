#!/usr/bin/env python3
"""
v0.3.2 バッチテスト - 最初の10枚のみ
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_batch_test():
    """最初の10枚でバッチテスト実行"""
    input_dir = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_dir = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力画像のリストを取得
    input_images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:10]
    
    print(f"🚀 v0.3.2 バッチテスト開始")
    print(f"入力パス: {input_dir}")
    print(f"出力パス: {output_dir}")
    print(f"📊 処理対象: {len(input_images)}枚（最初の10枚）")
    
    success_count = 0
    failure_count = 0
    
    for i, image_file in enumerate(input_images, 1):
        print(f"\n📷 処理中 ({i}/{len(input_images)}): {image_file}")
        
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        try:
            # 個別実行
            cmd = [
                "python3", "extract_kaname07.py",
                "--input_path", input_path,
                "--output_path", output_path,
                "--quality_method", "size_priority",
                "--enable_region_priority"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"✅ 成功: {image_file} ({end_time - start_time:.1f}秒)")
                success_count += 1
            else:
                print(f"❌ 失敗: {image_file}")
                if result.stderr:
                    print(f"   エラー: {result.stderr[:200]}...")
                failure_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏰ タイムアウト: {image_file}")
            failure_count += 1
        except Exception as e:
            print(f"❌ 例外: {image_file} - {str(e)}")
            failure_count += 1
    
    print(f"\n📊 処理結果:")
    print(f"✅ 成功: {success_count}枚")
    print(f"❌ 失敗: {failure_count}枚")
    print(f"📈 成功率: {success_count/(success_count+failure_count)*100:.1f}%")

if __name__ == "__main__":
    run_batch_test()