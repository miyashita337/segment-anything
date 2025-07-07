#!/usr/bin/env python3
"""
全バッチ処理自動実行スクリプト
チャンク処理を繰り返し実行して全ファイルを処理
"""

import subprocess
import time
import json
from pathlib import Path

PROGRESS_FILE = "batch_progress.json"

def get_progress():
    """現在の進捗を取得"""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return None

def main():
    """メイン処理"""
    print("🚀 全バッチ処理自動実行開始")
    
    total_files = 153
    chunk_count = 0
    
    while True:
        chunk_count += 1
        print(f"\n📦 チャンク {chunk_count} 実行中...")
        
        try:
            # チャンク処理実行
            result = subprocess.run(
                ["python3", "batch_extract_chunked.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5分タイムアウト
            )
            
            print(f"チャンク {chunk_count} 完了:")
            if result.stdout:
                # 重要な行のみ表示
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in [
                        '📊 チャンク完了:', '累計:', '残り:', 
                        '🎉 全バッチ処理完了!', '✅ Pushover通知送信完了',
                        '⚠️'
                    ]):
                        print(f"  {line}")
            
            if result.stderr:
                print(f"⚠️ エラー出力: {result.stderr}")
            
            # 進捗確認
            progress = get_progress()
            if progress:
                completed = len(progress['processed'])
                success_rate = (progress['successful'] / completed * 100) if completed > 0 else 0
                print(f"  進捗: {completed}/{total_files} ({completed/total_files*100:.1f}%)")
                print(f"  成功率: {success_rate:.1f}%")
                
                if completed >= total_files:
                    print("🎉 全処理完了!")
                    break
            else:
                print("⚠️ 進捗ファイルが見つかりません")
                break
            
            # 次のチャンクまで少し待機
            print("⏳ 次のチャンクまで3秒待機...")
            time.sleep(3)
            
        except subprocess.TimeoutExpired:
            print(f"⏰ チャンク {chunk_count} タイムアウト (5分)")
            continue
        except Exception as e:
            print(f"❌ チャンク {chunk_count} エラー: {e}")
            break
        
        # 安全装置: 最大50チャンクで停止
        if chunk_count >= 50:
            print("⚠️ 最大チャンク数に到達。処理を停止します。")
            break
    
    print(f"\n📊 全体処理完了 (実行チャンク数: {chunk_count})")
    
    # 最終状況表示
    final_progress = get_progress()
    if final_progress:
        completed = len(final_progress['processed'])
        success_rate = (final_progress['successful'] / completed * 100) if completed > 0 else 0
        print(f"最終結果:")
        print(f"  処理済み: {completed}/{total_files} ({completed/total_files*100:.1f}%)")
        print(f"  成功: {final_progress['successful']}")
        print(f"  失敗: {final_progress['failed']}")
        print(f"  成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main()