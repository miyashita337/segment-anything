#!/usr/bin/env python3
"""
kaname04データセット専用バッチ処理スクリプト
extract_kaname03.pyベースでkaname04用に調整
"""

import sys
import os
sys.path.append('.')

from utils.notification import send_batch_notification

def main():
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    # バッチ処理実行
    print("🚀 kaname04バッチ処理開始...")
    from commands.extract_character import batch_extract_characters
    
    input_dir = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname04"
    output_dir = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04"
    
    # デフォルト設定（マスクと透明背景はOFF）
    extract_args = {
        'enhance_contrast': False,
        'filter_text': True,
        'save_mask': False,
        'save_transparent': False,
        'min_yolo_score': 0.1,
        'verbose': False
    }
    
    result = batch_extract_characters(input_dir, output_dir, **extract_args)
    
    print(f"\n📊 kaname04最終結果:")
    print(f"   成功: {result['successful']}/{result['total_files']} ({result['success_rate']:.1%})")
    print(f"   失敗: {result['failed']}")
    
    # 結果を進捗ファイルに記録
    try:
        import json
        import time
        
        progress_file = "progress_req_4_exe_202507120307.json"
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        progress["test_results"]["method4_pipeline"] = f"SUCCESS - {result['successful']}/{result['total_files']} images processed"
        progress["completed_steps"].append("method4_pipeline")
        progress["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✅ 進捗ファイル更新完了")
        
    except Exception as e:
        print(f"⚠️ 進捗ファイル更新エラー: {e}")
    
    return result

if __name__ == "__main__":
    result = main()
    print(f"\n🎯 kaname04処理完了: {result['success_rate']:.1%} 成功率")