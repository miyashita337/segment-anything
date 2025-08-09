#!/usr/bin/env python3
"""
QC-KANA08復元版で5枚テスト
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    input_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/QC-KANA08-TEST-5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 最初の5枚
    image_files = sorted(list(input_dir.glob("*.jpg")))[:5]
    
    print(f"処理対象: {len(image_files)}枚")
    success_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/5] {image_file.name}")
        
        base_name = image_file.stem
        output_file = output_dir / f"extracted_{base_name}.png"
        
        try:
            cmd = [
                "sam-env/bin/python3",
                "features/extraction/commands/extract_character.py",
                str(image_file),
                "-o", str(output_file),
                "--verbose"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_file.exists():
                success_count += 1
                print(f"  成功")
            else:
                print(f"  失敗")
                    
        except subprocess.TimeoutExpired:
            print(f"  タイムアウト")
        except Exception as e:
            print(f"  例外: {e}")
    
    print(f"\n結果: {success_count}/5枚成功")
    return 0 if success_count >= 4 else 1

if __name__ == "__main__":
    exit(main())