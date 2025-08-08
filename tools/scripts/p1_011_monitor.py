#!/usr/bin/env python3
"""
P1-011抽出進捗監視スクリプト
"""

import subprocess
import time
from pathlib import Path


def monitor_extraction(tracker_id: str = "P1-011"):
    """抽出進捗監視"""
    workspace_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}")
    extraction_dir = workspace_dir / "extraction"
    log_file = workspace_dir / "extraction_log.txt"
    
    print(f"🔍 {tracker_id} 抽出進捗監視開始")
    print(f"📁 監視ディレクトリ: {extraction_dir}")
    print(f"📋 ログファイル: {log_file}")
    print("="*50)
    
    while True:
        # プロセス確認
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            running = "p1_011_real_extraction" in result.stdout
            
            if running:
                status = "🔄 実行中"
            else:
                status = "⏹️ 停止中"
            
        except Exception:
            status = "❓ 不明"
        
        # 出力ファイル数確認
        output_files = []
        if extraction_dir.exists():
            output_files = list(extraction_dir.glob("*.png")) + list(extraction_dir.glob("*.jpg"))
        
        # ログファイルサイズ確認
        log_size = 0
        if log_file.exists():
            log_size = log_file.stat().st_size
        
        # 現在時刻
        current_time = time.strftime("%H:%M:%S")
        
        print(f"[{current_time}] {status} | 出力ファイル: {len(output_files)}個 | ログ: {log_size}bytes")
        
        # 完了判定
        if not running and len(output_files) > 0:
            print(f"✅ 抽出完了の可能性（出力ファイル{len(output_files)}個検出）")
            break
        elif not running and log_size > 100:
            print("⚠️ プロセス停止・ログあり（エラーの可能性）")
            if log_file.exists():
                print("📋 ログ内容:")
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        print(f.read()[-500:])  # 最後の500文字
                except Exception as e:
                    print(f"ログ読み取りエラー: {e}")
            break
        
        time.sleep(10)  # 10秒間隔で監視


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-011抽出進捗監視")
    parser.add_argument("--tracker_id", type=str, default="P1-011", help="トラッカーID")
    
    args = parser.parse_args()
    
    try:
        monitor_extraction(args.tracker_id)
    except KeyboardInterrupt:
        print("\n⚠️ 監視中断")


if __name__ == "__main__":
    main()