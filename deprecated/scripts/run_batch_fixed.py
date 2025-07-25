#!/usr/bin/env python3
"""
修正版バッチ処理スクリプト - 26枚の画像をfullbody_priorityで処理
"""
import os
import shutil
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
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_claude_uni_13_9_fixed")
    
    # 出力ディレクトリを確保
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルリスト取得
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    total = len(image_files)
    
    print(f"🚀 修正版バッチ処理開始")
    print(f"📁 入力: {input_dir}")
    print(f"📁 出力: {output_dir}")
    print(f"📊 総数: {total}枚")
    
    # カウンター
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 各画像を処理
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"📸 処理中 [{i}/{total}]: {image_path.name}")
        
        # 出力パス（元のファイル名を保持）
        output_path = output_dir / image_path.name
        
        try:
            # CLI経由で抽出実行
            cmd = [
                'python3', '-m', 'features.extraction.commands.extract_character',
                str(image_path),
                '-o', str(output_path),
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if output_path.exists():
                successful += 1
                print(f"✅ 成功: {output_path.name}")
                if result.stdout:
                    print(f"   出力: {result.stdout.strip()[-100:]}")  # 最後の100文字のみ
            else:
                failed += 1
                print(f"❌ 失敗: {output_path.name}")
                if result.stderr:
                    print(f"   エラー: {result.stderr.strip()[-100:]}")
                
        except Exception as e:
            failed += 1
            print(f"❌ エラー: {image_path.name} - {e}")
            import traceback
            traceback.print_exc()
    
    # 処理時間計算
    total_time = time.time() - start_time
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"🎯 バッチ処理完了")
    print(f"✅ 成功: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ 失敗: {failed}")
    print(f"⏱️  処理時間: {total_time:.1f}秒 (平均: {total_time/total:.1f}秒/画像)")
    
    # Pushover通知
    try:
        notifier = PushoverNotifier()
        notifier.send_batch_complete_with_images(
            successful=successful,
            total=total,
            failed=failed,
            total_time=total_time,
            image_dir=output_dir
        )
        print("📱 Pushover通知送信完了")
    except Exception as e:
        print(f"⚠️ Pushover通知失敗: {e}")

if __name__ == "__main__":
    main()