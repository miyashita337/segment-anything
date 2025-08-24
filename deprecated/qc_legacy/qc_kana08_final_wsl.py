#!/usr/bin/env python3
"""
QC-KANA08完全復元版 - WSLネイティブ環境で20枚抽出
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    print("QC-KANA08完全復元版 - WSLネイティブ環境で20枚抽出開始")
    
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QC-KANA08-FINAL-WSL")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 最初の20枚を取得
    image_files = sorted(list(input_dir.glob("*.jpg")))[:20]
    
    print(f"対象画像: {len(image_files)}枚")
    
    success_count = 0
    failed_files = []
    start_time = time.time()
    
    # PYTHONPATHを設定
    env = os.environ.copy()
    env['PYTHONPATH'] = '/mnt/c/AItools/segment-anything'
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i:2d}/20] 処理中: {image_file.name}")
        
        base_name = image_file.stem
        # QC-KANA08形式: extracted_kana08_XXXX.png
        output_file = output_dir / f"extracted_{base_name}.png"
        
        try:
            # WSLネイティブPython3で実行
            cmd = [
                "python3",
                "features/extraction/commands/extract_character.py",
                str(image_file),
                "-o", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            
            if result.returncode == 0 and output_file.exists():
                success_count += 1
                print(f"  成功")
            else:
                failed_files.append(image_file.name)
                print(f"  失敗")
                    
        except subprocess.TimeoutExpired:
            failed_files.append(image_file.name)
            print(f"  タイムアウト")
        except Exception as e:
            failed_files.append(image_file.name)
            print(f"  例外: {e}")
    
    # 結果サマリー
    total_time = time.time() - start_time
    success_rate = success_count / 20
    
    print("\n" + "="*60)
    print("QC-KANA08完全復元版 - WSLネイティブ20枚抽出完了")
    print(f"成功率: {success_rate:.1%} ({success_count}/20枚)")
    print(f"処理時間: {total_time:.1f}秒")
    
    if failed_files:
        print(f"失敗ファイル: {', '.join(failed_files)}")
    else:
        print("全20枚で抽出成功！QC-KANA08完全復元完了")
    
    print("="*60)
    
    return 0 if success_rate >= 0.8 else 1

if __name__ == "__main__":
    exit(main())