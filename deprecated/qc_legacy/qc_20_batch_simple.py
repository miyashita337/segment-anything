#!/usr/bin/env python3
"""
QC-KANA08復元版で20枚抽出
簡素な実装・決定論的実行確保
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    print("QC-KANA08復元版で20枚抽出開始")
    
    input_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/QC-KANA08-RESTORED-BATCH")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 最初の20枚を取得（kana08_0000_cover.jpg から）
    image_files = list(input_dir.glob("*.jpg"))
    image_files.sort()
    image_files = image_files[:20]  # 最初の20枚のみ
    
    print(f"対象画像: {len(image_files)}枚")
    
    success_count = 0
    failed_files = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i:2d}/20] 処理中: {image_file.name}")
        
        base_name = image_file.stem
        output_file = output_dir / f"extracted_{base_name}.png"
        
        try:
            # QC-KANA08復元版のextract_character.py実行
            cmd = [
                "sam-env/bin/python3",
                "features/extraction/commands/extract_character.py",
                str(image_file),
                "-o", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and output_file.exists():
                success_count += 1
                print(f"  成功")
            else:
                failed_files.append(image_file.name)
                print(f"  失敗")
                if result.stderr:
                    print(f"     エラー: {result.stderr[:100]}...")
                    
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
    print("QC-KANA08復元版20枚抽出完了")
    print(f"成功率: {success_rate:.1%} ({success_count}/20枚)")
    print(f"処理時間: {total_time:.1f}秒")
    
    if failed_files:
        print(f"失敗ファイル: {', '.join(failed_files)}")
    else:
        print("全20枚で抽出成功！QC-KANA08復元完了")
    
    print("="*60)
    
    return 0 if success_rate >= 0.8 else 1

if __name__ == "__main__":
    exit(main())