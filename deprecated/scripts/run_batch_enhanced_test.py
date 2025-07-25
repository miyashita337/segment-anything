#!/usr/bin/env python3
"""
Enhanced Filtering System Test - 問題のあった5画像で効果検証
"""
import os
import sys
import time
from pathlib import Path

# プロジェクトルートに追加
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
from features.common.notification.notification import PushoverNotifier


def main():
    # 入力・出力ディレクトリ
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_enhanced_system_test")
    
    # テスト対象ファイル（評価でF評価だった問題ファイル）
    test_files = [
        "kana08_0006.jpg",  # 吹き出し誤抽出
        "kana08_0007.jpg",  # マスク誤抽出
        "kana08_0011.jpg",  # マスク誤抽出
        "kana08_0022.jpg",  # マスク誤抽出
        "kana08_0000_cover.jpg"  # 部分抽出問題
    ]
    
    # 出力ディレクトリを確保
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 強化フィルタリングシステム テスト開始")
    print(f"📁 入力: {input_dir}")
    print(f"📁 出力: {output_dir}")
    print(f"📊 テスト対象: {len(test_files)}枚（問題ファイル）")
    
    # カウンター
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 各画像を処理
    for i, filename in enumerate(test_files, 1):
        print(f"\n{'='*60}")
        print(f"📸 処理中 [{i}/{len(test_files)}]: {filename}")
        
        image_path = input_dir / filename
        output_path = output_dir / filename
        
        if not image_path.exists():
            print(f"❌ ファイル未検出: {image_path}")
            failed += 1
            continue
        
        try:
            # CLI経由で抽出実行
            cmd = [
                'python3', '-m', 'features.extraction.commands.extract_character',
                str(image_path),
                '-o', str(output_path),
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), timeout=120)
            
            if output_path.exists():
                successful += 1
                print(f"✅ 成功: {output_path.name}")
                
                # 重要な情報を抽出
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines:
                    if 'Final masks for selection:' in line:
                        print(f"   フィルタ結果: {line.strip()}")
                    elif 'Selected mask validation:' in line:
                        print(f"   検証: {line.strip()}")
                    elif 'Character extracted:' in line and 'size:' in line:
                        print(f"   出力: {line.strip()}")
                    elif 'Mask quality:' in line:
                        print(f"   品質: {line.strip()}")
            else:
                failed += 1
                print(f"❌ 失敗: {output_path.name}")
                if result.stderr:
                    print(f"   エラー: {result.stderr.strip()[-200:]}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"❌ タイムアウト: {filename}")
        except Exception as e:
            failed += 1
            print(f"❌ エラー: {filename} - {e}")
    
    # 処理時間計算
    total_time = time.time() - start_time
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"🎯 強化システムテスト完了")
    print(f"✅ 成功: {successful}/{len(test_files)} ({successful/len(test_files)*100:.1f}%)")
    print(f"❌ 失敗: {failed}")
    print(f"⏱️  処理時間: {total_time:.1f}秒 (平均: {total_time/len(test_files):.1f}秒/画像)")
    
    # 効果の評価
    if successful >= 3:
        print("🎉 強化システムは問題ファイルに対して効果的です！")
    elif successful >= 2:
        print("🔧 強化システムは部分的に効果的、さらなる調整が必要")
    else:
        print("⚠️ 強化システムの効果が限定的、追加の改善が必要")

if __name__ == "__main__":
    main()