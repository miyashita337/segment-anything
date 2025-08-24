#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元仕様確認用バックグラウンド抽出スクリプト
TEST-20250803で全26枚を処理
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# UTF-8エンコーディング設定
os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    print("Starting background extraction - Original specification check")
    
    input_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/TEST-20250803/extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル取得
    image_files = list(input_dir.glob("*.jpg"))
    image_files.sort()
    
    print(f"Target images: {len(image_files)} files")
    
    success_count = 0
    failed_files = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i:2d}/{len(image_files)}] Processing: {image_file.name}")
        
        base_name = image_file.stem
        output_file = output_dir / f"{base_name}_extracted.jpg"
        
        try:
            # extract_character.py実行（最適化なし、元仕様）
            cmd = [
                "sam-env/bin/python3",
                "features/extraction/commands/extract_character.py",
                str(image_file),
                "-o", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0 and output_file.exists():
                success_count += 1
                print(f"  SUCCESS")
            else:
                failed_files.append(image_file.name)
                print(f"  FAILED")
                    
        except subprocess.TimeoutExpired:
            failed_files.append(image_file.name)
            print(f"  TIMEOUT")
        except Exception as e:
            failed_files.append(image_file.name)
            print(f"  ERROR: {e}")
    
    # 結果サマリー
    total_time = time.time() - start_time
    success_rate = success_count / len(image_files) if len(image_files) > 0 else 0
    
    print("\n" + "="*60)
    print("Batch extraction completed")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{len(image_files)} files)")
    print(f"Processing time: {total_time:.1f} seconds")
    
    if failed_files:
        print(f"Failed files: {', '.join(failed_files)}")
    
    print("="*60)
    
    # 結果をファイルに保存
    result_file = output_dir / "extraction_result.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Original specification check result\n")
        f.write(f"Success rate: {success_rate:.1%} ({success_count}/{len(image_files)} files)\n")
        f.write(f"Processing time: {total_time:.1f} seconds\n")
        if failed_files:
            f.write(f"Failed files: {', '.join(failed_files)}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())