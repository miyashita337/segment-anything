#!/usr/bin/env python3
"""
最高品質キャラクター抽出バッチ実行
Phase 0リファクタリング後の新構造対応版
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_batch_extraction(input_path, output_path):
    """バッチ抽出実行"""
    
    print("Starting high-quality character extraction batch...")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    
    # 入力パス検証
    if not Path(input_path).exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # 画像ファイル取得
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    
    if not image_files:
        print(f"ERROR: No image files found in: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {len(image_files)} image files")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 新構造での抽出実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\nProcessing ({i}/{len(image_files)}): {image_file.name}")
            
            try:
                # 出力ファイル名生成
                output_file = Path(output_path) / image_file.name
                
                # 最高品質設定で抽出実行
                extract_character_from_path(
                    image_path=str(image_file),
                    output_path=str(output_file),
                    enhance_contrast=True,
                    filter_text=True,
                    save_mask=False,
                    save_transparent=False,
                    min_yolo_score=0.05,  # 高感度
                    verbose=False,
                    difficult_pose=True,  # 複雑姿勢対応
                    low_threshold=True,   # 低閾値
                    auto_retry=True,      # 自動リトライ
                    high_quality=True     # 高品質処理
                )
                
                # 出力ファイル確認
                if output_file.exists():
                    print(f"SUCCESS: {output_file.name}")
                    success_count += 1
                else:
                    print(f"FAILED: Output file not created")
                    error_count += 1
                    
            except Exception as e:
                print(f"ERROR: {e}")
                error_count += 1
                
        print(f"\nProcessing completed")
        print(f"成功: {success_count}個")
        print(f"失敗: {error_count}個")
        print(f"成功率: {success_count/(success_count+error_count)*100:.1f}%")
        
        if error_count > 0:
            print("WARNING: Some images failed to process")
        else:
            print("All images processed successfully")
            
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print("必要なモジュールが見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch character extraction")
    parser.add_argument("--input_dir", required=True, help="Input directory path")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    args = parser.parse_args()
    
    run_batch_extraction(args.input_dir, args.output_dir)