#!/usr/bin/env python3
"""
チャンク分割バッチ処理スクリプト
タイムアウト対策で小分けして処理し、最後にまとめて通知
"""

import sys
import os
import json
import time
from pathlib import Path
sys.path.append('.')

from utils.notification import send_batch_notification

# 進捗ファイル
PROGRESS_FILE = "batch_progress.json"

def load_progress():
    """進捗ファイルを読み込み"""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "successful": 0, "failed": 0, "start_time": time.time()}

def save_progress(progress):
    """進捗ファイルを保存"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def process_chunk(start_idx, chunk_size=10):
    """指定範囲の画像を処理"""
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    # バッチ処理
    from commands.extract_character import batch_extract_characters
    
    input_dir = "/mnt/c/AItools/lora/train/diff_aichi/org_aichikan1"
    output_dir = "/mnt/c/AItools/lora/train/diff_aichi/auto_extracted_v5"
    
    # 全ファイルリスト取得
    input_path = Path(input_dir)
    image_files = sorted([f for f in input_path.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    total_files = len(image_files)
    end_idx = min(start_idx + chunk_size, total_files)
    chunk_files = image_files[start_idx:end_idx]
    
    print(f"🚀 チャンク処理開始: {start_idx+1}-{end_idx}/{total_files}")
    
    # チャンク処理用の一時ディレクトリ作成
    chunk_input_dir = Path("temp_chunk_input")
    chunk_input_dir.mkdir(exist_ok=True)
    
    # チャンク内ファイルを一時ディレクトリにリンク
    for i, file in enumerate(chunk_files):
        link_path = chunk_input_dir / file.name
        if link_path.exists():
            link_path.unlink()  # 既存リンクを削除
        link_path.symlink_to(file.absolute())
    
    try:
        # デフォルト設定
        extract_args = {
            'enhance_contrast': False,
            'filter_text': True,
            'save_mask': False,
            'save_transparent': False,
            'min_yolo_score': 0.1,
            'verbose': False
        }
        
        result = batch_extract_characters(str(chunk_input_dir), output_dir, **extract_args)
        
        return {
            'successful': result['successful'],
            'failed': result['failed'],
            'total': len(chunk_files),
            'total_time': result.get('total_time', 0),
            'processed_files': [f.name for f in chunk_files]
        }
    
    finally:
        # 一時ディレクトリクリーンアップ
        for link_file in chunk_input_dir.iterdir():
            link_file.unlink()
        chunk_input_dir.rmdir()

def main():
    """メイン処理"""
    progress = load_progress()
    
    # 全ファイル数取得
    input_dir = "/mnt/c/AItools/lora/train/diff_aichi/org_aichikan1"
    input_path = Path(input_dir)
    all_files = sorted([f for f in input_path.iterdir() 
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    total_files = len(all_files)
    
    # 未処理ファイルから開始位置決定
    processed_files = set(progress.get("processed", []))
    start_idx = 0
    for i, file in enumerate(all_files):
        if file.name not in processed_files:
            start_idx = i
            break
    else:
        # 全て処理済み
        print("✅ 全ファイル処理済み")
        # 最終通知送信
        elapsed_time = time.time() - progress["start_time"]
        send_batch_notification(
            successful=progress["successful"],
            total=total_files,
            failed=progress["failed"],
            total_time=elapsed_time
        )
        Path(PROGRESS_FILE).unlink(missing_ok=True)
        return
    
    print(f"📊 進捗状況: {len(processed_files)}/{total_files} 完了")
    print(f"⏯️ 位置 {start_idx+1} から再開")
    
    # チャンク処理（5ファイルずつ）
    chunk_size = 5
    chunk_result = process_chunk(start_idx, chunk_size)
    
    # 進捗更新
    progress["successful"] += chunk_result["successful"]
    progress["failed"] += chunk_result["failed"]
    progress["processed"].extend(chunk_result["processed_files"])
    
    save_progress(progress)
    
    # 進捗表示
    completed = len(progress["processed"])
    success_rate = (progress["successful"] / completed * 100) if completed > 0 else 0
    
    print(f"\n📊 チャンク完了:")
    print(f"   今回: 成功{chunk_result['successful']}/失敗{chunk_result['failed']}")
    print(f"   累計: {progress['successful']}/{completed} ({success_rate:.1f}%)")
    print(f"   残り: {total_files - completed}ファイル")
    
    # 全処理完了チェック
    if completed >= total_files:
        print("\n🎉 全バッチ処理完了!")
        elapsed_time = time.time() - progress["start_time"]
        
        # 最終通知送信
        print("📱 最終通知送信中...")
        notification_sent = send_batch_notification(
            successful=progress["successful"],
            total=total_files,
            failed=progress["failed"],
            total_time=elapsed_time
        )
        
        if notification_sent:
            print("✅ Pushover通知送信完了")
        else:
            print("⚠️ Pushover通知送信失敗")
        
        # 進捗ファイル削除
        Path(PROGRESS_FILE).unlink(missing_ok=True)
    else:
        print(f"\n⏭️ 次回は位置 {start_idx + chunk_size + 1} から継続")

if __name__ == "__main__":
    main()