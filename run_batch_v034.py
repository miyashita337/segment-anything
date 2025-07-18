#!/usr/bin/env python3
"""
v0.3.4最高品質キャラクター抽出バッチ実行
Phase 1品質評価システム完全統合版 - 5つの新システム統合実行

新機能:
P1-018: 滑らかさ評価指標の実装
P1-020: 切断検出アルゴリズム
P1-022: 混入率定量化システム
P1-016: フィードバックループ構築
P1-010: 効率的サンプリングアルゴリズム
"""

import sys
import os
import time
import json
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_batch_v034():
    """v0.3.4最高品質バッチ抽出実行（Phase 1完全統合）"""
    
    # パス設定（ユーザー指定）
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname09"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname09_0_3_4"
    
    print("🚀 v0.3.4最高品質キャラクター抽出バッチ実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    print(f"バージョン: v0.3.4 (Phase 1完全統合版)")
    print("✨ 新機能: 滑らかさ評価、切断検出、混入定量化、フィードバック学習、効率サンプリング")
    
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
                
                # v0.3.4最高品質設定での抽出実行
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
        
        # バッチ処理完了統計
        total_time = time.time() - start_time
        success_rate = success_count / len(image_files) * 100
        
        # 結果レポート出力
        print("\n" + "="*80)
        print("📊 v0.3.4バッチ処理完了レポート")
        print("="*80)
        
        print(f"\n📈 処理結果:")
        print(f"  総処理数: {len(image_files)}枚")
        print(f"  成功: {success_count}枚")
        print(f"  失敗: {error_count}枚")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  総処理時間: {total_time:.1f}秒")
        
        print(f"\n✅ v0.3.4バッチ処理完了!")
        
        return success_rate >= 70.0  # 70%以上の成功率で成功判定
        
    except Exception as e:
        print(f"❌ バッチ処理で重大エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")
        return False


def main():
    """メイン実行関数"""
    print("🚀 v0.3.4最高品質バッチ処理開始")
    
    success = run_batch_v034()
    
    if success:
        print("\n✅ バッチ処理成功!")
        sys.exit(0)
    else:
        print("\n❌ バッチ処理失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()