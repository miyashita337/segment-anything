#!/usr/bin/env python3
"""
元仕様確認用バックグラウンド抽出スクリプト
TEST-20250803で全26枚を処理
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    print("🚀 元仕様確認 - バックグラウンド抽出開始")
    
    input_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/TEST-20250803/extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル取得
    image_files = list(input_dir.glob("*.jpg"))
    image_files.sort()
    
    print(f"📊 対象画像: {len(image_files)}枚")
    
    success_count = 0
    failed_files = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i:2d}/{len(image_files)}] 処理中: {image_file.name}")
        
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
                print(f"  ✅ 成功")
            else:
                failed_files.append(image_file.name)
                print(f"  ❌ 失敗")
                    
        except subprocess.TimeoutExpired:
            failed_files.append(image_file.name)
            print(f"  ⏰ タイムアウト")
        except Exception as e:
            failed_files.append(image_file.name)
            print(f"  ❌ 例外: {e}")
    
    # 結果サマリー
    total_time = time.time() - start_time
    success_rate = success_count / len(image_files)
    
    print("\n" + "="*60)
    print("🎯 元仕様確認バッチ抽出完了")
    print(f"📊 成功率: {success_rate:.1%} ({success_count}/{len(image_files)}枚)")
    print(f"⏱️ 処理時間: {total_time:.1f}秒")
    
    if failed_files:
        print(f"❌ 失敗ファイル: {', '.join(failed_files)}")
    
    print("="*60)
    
    # 結果をファイルに保存
    result_file = output_dir / "extraction_result.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"元仕様確認結果\n")
        f.write(f"成功率: {success_rate:.1%} ({success_count}/{len(image_files)}枚)\n")
        f.write(f"処理時間: {total_time:.1f}秒\n")
        if failed_files:
            f.write(f"失敗ファイル: {', '.join(failed_files)}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())