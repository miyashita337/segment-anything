#!/usr/bin/env python3
"""
kaname03データセット専用バッチ処理スクリプト
モデル初期化とバッチ処理を一度に実行
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
    print("🚀 kaname03バッチ処理開始...")
    from commands.extract_character import batch_extract_characters
    
    input_dir = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname03"
    output_dir = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname03"
    
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
    
    print(f"\n📊 kaname03最終結果:")
    print(f"   成功: {result['successful']}/{result['total']} ({result['success_rate']:.1f}%)")
    print(f"   失敗: {result['failed']}")
    print(f"   処理時間: {result['total_time']:.2f}秒")
    
    # Pushover通知送信
    print("\n📱 通知送信中...")
    notification_sent = send_batch_notification(
        successful=result['successful'],
        total=result['total'],
        failed=result['failed'],
        total_time=result['total_time']
    )
    
    if notification_sent:
        print("✅ Pushover通知送信完了")
    else:
        print("⚠️ Pushover通知送信失敗またはスキップ")

if __name__ == "__main__":
    main()