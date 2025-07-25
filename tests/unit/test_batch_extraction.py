#!/usr/bin/env python3
"""
バッチ抽出テスト
入力パス・出力パス検証とエラーハンドリングテスト
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_input_path_validation():
    """入力パス検証テスト"""
    input_path = "/mnt/c/AItools/lora/train/yado/org/kaname06"
    
    if not Path(input_path).exists():
        print(f"❌ 入力パスが存在しません: {input_path}")
        return False
    
    # 画像ファイルの存在確認
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    
    if not image_files:
        print(f"❌ 入力パスに画像ファイルがありません: {input_path}")
        return False
    
    print(f"✅ 入力パス確認: {len(image_files)}個の画像ファイル")
    return True

def test_output_path_preparation():
    """出力パス準備テスト"""
    output_path = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname06"
    
    # 出力ディレクトリの作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    if not Path(output_path).exists():
        print(f"❌ 出力パスの作成に失敗: {output_path}")
        return False
    
    print(f"✅ 出力パス準備完了: {output_path}")
    return True

if __name__ == "__main__":
    print("🧪 バッチ抽出前テスト実行中...")
    
    tests = [
        ("入力パス検証", test_input_path_validation),
        ("出力パス準備", test_output_path_preparation)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\n✅ 事前テスト完了 - バッチ実行可能")
    else:
        print("\n❌ 事前テスト失敗 - バッチ実行中止")
        sys.exit(1)