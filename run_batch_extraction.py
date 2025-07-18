#!/usr/bin/env python3
"""
最高品質キャラクター抽出バッチ実行
Phase 0リファクタリング後の新構造対応版
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_batch_extraction():
    """バッチ抽出実行"""
    
    # パス設定
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname06"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname06"
    
    print("🚀 最高品質キャラクター抽出バッチ実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    
    # 入力パス検証
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)
    
    # 画像ファイル取得
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    
    if not image_files:
        print(f"❌ エラー: 入力パスに画像ファイルがありません: {input_path}")
        sys.exit(1)
    
    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 新構造での抽出実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n📷 処理中 ({i}/{len(image_files)}): {image_file.name}")
            
            try:
                # 出力ファイル名生成
                output_file = Path(output_path) / f"{image_file.stem}_character.jpg"
                
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
                    print(f"✅ 成功: {output_file.name}")
                    success_count += 1
                else:
                    print(f"❌ 失敗: 出力ファイル未作成")
                    error_count += 1
                    
            except Exception as e:
                print(f"❌ エラー: {e}")
                error_count += 1
                
        print(f"\n📊 処理完了")
        print(f"成功: {success_count}個")
        print(f"失敗: {error_count}個")
        print(f"成功率: {success_count/(success_count+error_count)*100:.1f}%")
        
        if error_count > 0:
            print("⚠️ 一部の画像で処理に失敗しました")
        else:
            print("🎉 全ての画像の処理が完了しました")
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("必要なモジュールが見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_batch_extraction()