#!/usr/bin/env python3
"""
シンプル進捗監視システム

Google Sheets連携なしでバッチ処理を監視
- プロセス監視
- ログファイル分析
- 推定進捗計算
"""

import time
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class SimpleProgressMonitor:
    """シンプル進捗監視"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.project_root = Path(__file__).parent.parent.parent
        self.log_file = self.project_root / "kana05_production.log"
        self.start_time = datetime.now()
        
    def find_batch_process(self):
        """バッチ処理プロセスを検索"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('simple_batch_runner.py' in arg for arg in cmdline):
                    if self.tracker_id in ' '.join(cmdline):
                        return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def find_sam_process(self):
        """SAM処理プロセスを検索"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('sam_yolo_character_segment.py' in arg for arg in cmdline):
                    return {
                        'pid': proc.info['pid'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_info'].rss / 1024 / 1024 / 1024  # GB
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def parse_log_progress(self):
        """ログファイルから進捗解析"""
        if not self.log_file.exists():
            return {"status": "ログファイルなし", "progress": 0}
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_batch = 0
            total_batches = 0
            processed_images = 0
            failed_images = 0
            
            for line in lines:
                # バッチ進捗の解析
                if "=== バッチ" in line:
                    # "=== バッチ 2/39 ===" のような行を解析
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "/" in part:
                            batch_info = part.split("/")
                            if len(batch_info) == 2:
                                current_batch = int(batch_info[0])
                                total_batches = int(batch_info[1])
                
                # 成功・失敗の解析
                if "バッチ処理成功:" in line and "枚" in line:
                    # "バッチ処理成功: 0枚, 57.1秒" のような行から枚数を抽出
                    parts = line.split(":")
                    if len(parts) > 1:
                        success_part = parts[1].split("枚")[0].strip()
                        try:
                            processed_images += int(success_part)
                        except ValueError:
                            pass
                
                if "バッチ処理失敗" in line or "バッチ処理タイムアウト" in line:
                    failed_images += 1
            
            progress_rate = (current_batch / total_batches * 100) if total_batches > 0 else 0
            
            return {
                "status": "実行中",
                "current_batch": current_batch,
                "total_batches": total_batches,
                "progress": progress_rate,
                "processed_images": processed_images,
                "failed_images": failed_images
            }
            
        except Exception as e:
            return {"status": f"ログ解析エラー: {e}", "progress": 0}
    
    def estimate_remaining_time(self, progress_info):
        """残り時間推定"""
        if progress_info["progress"] <= 0:
            return "不明"
        
        elapsed = datetime.now() - self.start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        # 進捗率から残り時間を推定
        remaining_rate = 100 - progress_info["progress"]
        estimated_remaining_hours = (elapsed_hours * remaining_rate) / progress_info["progress"]
        
        remaining_td = timedelta(hours=estimated_remaining_hours)
        remaining_str = str(remaining_td).split('.')[0]  # ミリ秒削除
        
        estimated_completion = datetime.now() + remaining_td
        
        return {
            "remaining_time": remaining_str,
            "estimated_completion": estimated_completion.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def display_status(self):
        """ステータス表示"""
        print(f"\n{'='*60}")
        print(f"🔍 {self.tracker_id} バッチ処理監視 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # プロセス状況
        batch_pid = self.find_batch_process()
        sam_info = self.find_sam_process()
        
        if batch_pid:
            print(f"📋 バッチプロセス: PID {batch_pid} (稼働中)")
        else:
            print(f"📋 バッチプロセス: 見つかりません")
        
        if sam_info:
            print(f"🔧 SAM処理: PID {sam_info['pid']}")
            print(f"   CPU使用率: {sam_info['cpu']:.1f}%")
            print(f"   メモリ使用量: {sam_info['memory']:.1f}GB")
        else:
            print(f"🔧 SAM処理: アイドル状態")
        
        # 進捗情報
        progress_info = self.parse_log_progress()
        print(f"\n📈 進捗状況:")
        print(f"   ステータス: {progress_info['status']}")
        
        if progress_info.get('total_batches', 0) > 0:
            print(f"   バッチ進捗: {progress_info['current_batch']}/{progress_info['total_batches']} ({progress_info['progress']:.1f}%)")
            print(f"   処理成功: {progress_info['processed_images']}枚")
            print(f"   処理失敗: {progress_info['failed_images']}枚")
            
            # 残り時間推定
            time_est = self.estimate_remaining_time(progress_info)
            if time_est != "不明":
                print(f"   推定残り時間: {time_est['remaining_time']}")
                print(f"   完了予定: {time_est['estimated_completion']}")
        
        # 実行時間
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        print(f"   実行時間: {elapsed_str}")
        
        print(f"{'='*60}\n")
    
    def monitor(self, interval_seconds: int = 60):
        """監視開始"""
        print(f"🚀 {self.tracker_id} のシンプル監視を開始します")
        print(f"⏰ 監視間隔: {interval_seconds}秒")
        print(f"📄 ログファイル: {self.log_file}")
        print("Ctrl+C で監視を停止できます\n")
        
        try:
            while True:
                self.display_status()
                
                # プロセス生存確認
                batch_pid = self.find_batch_process()
                if not batch_pid:
                    print("📋 バッチプロセスが終了しました")
                    break
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 監視が中断されました")
        except Exception as e:
            print(f"\n❌ 監視エラー: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="シンプル進捗監視システム")
    parser.add_argument("tracker_id", help="監視対象トラッカーID")
    parser.add_argument("--interval", type=int, default=60, help="監視間隔（秒）")
    
    args = parser.parse_args()
    
    monitor = SimpleProgressMonitor(args.tracker_id)
    monitor.monitor(args.interval)


if __name__ == "__main__":
    main()