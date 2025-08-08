#!/usr/bin/env python3
"""
P1-019: プロセス安定性向上 - チェックポイント・自動再開・マイクロバッチ処理
長時間処理での中断対策と100%処理完了率を実現
"""

import json
import logging
import os
import psutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class ProcessHealthMonitor:
    """プロセス健全性監視クラス"""
    
    def __init__(self, max_memory_mb: int = 8000, max_runtime_minutes: int = 180):
        self.max_memory_mb = max_memory_mb
        self.max_runtime_minutes = max_runtime_minutes
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def check_health(self) -> Tuple[bool, str]:
        """プロセス健全性チェック"""
        try:
            # メモリ使用量チェック
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            if memory_mb > self.max_memory_mb:
                return False, f"メモリ使用量超過: {memory_mb:.1f}MB > {self.max_memory_mb}MB"
            
            # 実行時間チェック
            runtime_minutes = (time.time() - self.start_time) / 60
            if runtime_minutes > self.max_runtime_minutes:
                return False, f"実行時間超過: {runtime_minutes:.1f}分 > {self.max_runtime_minutes}分"
            
            return True, "正常"
            
        except Exception as e:
            return False, f"健全性チェックエラー: {str(e)}"


class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "processing_checkpoint.json"
        self.logger = logging.getLogger(__name__)
        
    def save_progress(self, processed_files: List[str], total_files: List[str], 
                     current_stats: Dict[str, Any]) -> None:
        """処理進捗を保存"""
        checkpoint_data = {
            "processed_files": processed_files,
            "total_files": total_files,
            "remaining_files": [f for f in total_files if f not in processed_files],
            "stats": current_stats,
            "timestamp": time.time(),
            "version": "P1-019-v1.0"
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"チェックポイント保存: {len(processed_files)}/{len(total_files)} 完了")
        except Exception as e:
            self.logger.error(f"チェックポイント保存エラー: {str(e)}")
    
    def load_progress(self) -> Optional[Dict[str, Any]]:
        """保存された進捗を読み込み"""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # バージョンチェック
            if checkpoint_data.get("version") != "P1-019-v1.0":
                self.logger.warning("チェックポイントバージョン不一致、新規開始")
                return None
                
            self.logger.info(f"チェックポイント読み込み: {len(checkpoint_data['processed_files'])} 件処理済み")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"チェックポイント読み込みエラー: {str(e)}")
            return None
    
    def clear_checkpoint(self) -> None:
        """チェックポイントクリア"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            self.logger.info("チェックポイントクリア完了")


class StableBatchProcessor:
    """
    P1-019: 安定バッチ処理システム
    チェックポイント・自動再開・マイクロバッチ処理による100%完了率実現
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 micro_batch_size: int = 3,
                 max_retries: int = 3,
                 retry_delay_seconds: int = 30):
        self.checkpoint_manager = CheckpointManager(Path(checkpoint_dir))
        self.health_monitor = ProcessHealthMonitor()
        self.micro_batch_size = micro_batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        
        # ロギング設定
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # 統計情報
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "success_count": 0,
            "error_count": 0,
            "retry_count": 0,
            "start_time": time.time()
        }
    
    def process_with_checkpoint(self, 
                               files: List[str], 
                               process_function: Callable[[str], Tuple[bool, str]],
                               output_dir: str,
                               resume: bool = True) -> Dict[str, Any]:
        """
        チェックポイント機能付きバッチ処理
        
        Args:
            files: 処理対象ファイルリスト
            process_function: 単一ファイル処理関数 (file_path) -> (success, message)
            output_dir: 出力ディレクトリ
            resume: 再開するかどうか
            
        Returns:
            処理結果統計
        """
        self.stats["total_files"] = len(files)
        processed_files = []
        
        # チェックポイントから再開
        if resume:
            checkpoint_data = self.checkpoint_manager.load_progress()
            if checkpoint_data:
                processed_files = checkpoint_data["processed_files"]
                self.stats.update(checkpoint_data["stats"])
                remaining_files = checkpoint_data["remaining_files"]
                self.logger.info(f"📋 チェックポイントから再開: {len(processed_files)} 件完了済み")
            else:
                remaining_files = files
        else:
            remaining_files = files
            self.checkpoint_manager.clear_checkpoint()
        
        self.logger.info(f"🚀 安定バッチ処理開始: {len(remaining_files)} 件処理予定")
        
        # マイクロバッチ処理
        micro_batches = [remaining_files[i:i + self.micro_batch_size] 
                        for i in range(0, len(remaining_files), self.micro_batch_size)]
        
        for batch_idx, micro_batch in enumerate(micro_batches):
            self.logger.info(f"📦 マイクロバッチ {batch_idx + 1}/{len(micro_batches)}: {len(micro_batch)} 件")
            
            # 健全性チェック
            is_healthy, health_message = self.health_monitor.check_health()
            if not is_healthy:
                self.logger.warning(f"⚠️ プロセス健全性低下: {health_message}")
                self.logger.info("💾 現在の進捗を保存して終了")
                self._save_current_progress(processed_files, files)
                return self._generate_result(False, "健全性問題により中断")
            
            # マイクロバッチ内の各ファイル処理
            for file_path in micro_batch:
                retry_count = 0
                
                while retry_count <= self.max_retries:
                    try:
                        self.logger.info(f"🔄 処理中: {os.path.basename(file_path)}")
                        
                        # 単一ファイル処理実行
                        success, message = process_function(file_path)
                        
                        if success:
                            processed_files.append(file_path)
                            self.stats["success_count"] += 1
                            self.stats["processed_files"] += 1
                            self.logger.info(f"✅ 成功: {os.path.basename(file_path)} - {message}")
                            break
                        else:
                            if retry_count < self.max_retries:
                                retry_count += 1
                                self.stats["retry_count"] += 1
                                self.logger.warning(f"🔄 リトライ {retry_count}/{self.max_retries}: {message}")
                                time.sleep(self.retry_delay_seconds)
                            else:
                                self.stats["error_count"] += 1
                                self.stats["processed_files"] += 1
                                self.logger.error(f"❌ 失敗: {os.path.basename(file_path)} - {message}")
                                processed_files.append(file_path)  # 失敗も処理済みとしてマーク
                                break
                                
                    except Exception as e:
                        if retry_count < self.max_retries:
                            retry_count += 1
                            self.stats["retry_count"] += 1
                            self.logger.warning(f"🔄 例外リトライ {retry_count}/{self.max_retries}: {str(e)}")
                            time.sleep(self.retry_delay_seconds)
                        else:
                            self.stats["error_count"] += 1
                            self.stats["processed_files"] += 1
                            self.logger.error(f"❌ 例外失敗: {os.path.basename(file_path)} - {str(e)}")
                            processed_files.append(file_path)  # 失敗も処理済みとしてマーク
                            break
            
            # マイクロバッチ完了後チェックポイント保存
            self._save_current_progress(processed_files, files)
            
            # 進捗表示
            progress_percent = (len(processed_files) / len(files)) * 100
            self.logger.info(f"📊 進捗: {len(processed_files)}/{len(files)} ({progress_percent:.1f}%)")
        
        # 処理完了
        self.checkpoint_manager.clear_checkpoint()
        self.logger.info(f"🎯 バッチ処理完了: 成功 {self.stats['success_count']}, エラー {self.stats['error_count']}")
        
        return self._generate_result(True, "正常完了")
    
    def _save_current_progress(self, processed_files: List[str], total_files: List[str]) -> None:
        """現在の進捗を保存"""
        self.checkpoint_manager.save_progress(processed_files, total_files, self.stats)
    
    def _generate_result(self, success: bool, message: str) -> Dict[str, Any]:
        """結果辞書生成"""
        end_time = time.time()
        runtime_minutes = (end_time - self.stats["start_time"]) / 60
        
        return {
            "success": success,
            "message": message,
            "stats": {
                **self.stats,
                "end_time": end_time,
                "runtime_minutes": runtime_minutes,
                "completion_rate": (self.stats["processed_files"] / self.stats["total_files"] * 100) 
                                   if self.stats["total_files"] > 0 else 0
            }
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """進捗サマリー取得"""
        checkpoint_data = self.checkpoint_manager.load_progress()
        if not checkpoint_data:
            return {"status": "no_checkpoint", "message": "チェックポイントなし"}
        
        processed = len(checkpoint_data["processed_files"])
        total = len(checkpoint_data["total_files"])
        remaining = len(checkpoint_data["remaining_files"])
        
        return {
            "status": "in_progress",
            "processed": processed,
            "total": total,
            "remaining": remaining,
            "completion_rate": (processed / total * 100) if total > 0 else 0,
            "timestamp": checkpoint_data["timestamp"]
        }


# テスト用サンプル処理関数
def sample_processing_function(file_path: str) -> Tuple[bool, str]:
    """サンプル処理関数（テスト用）"""
    import random

    # 90%の確率で成功をシミュレーション
    if random.random() > 0.1:
        time.sleep(0.5)  # 処理時間シミュレーション
        return True, f"正常処理完了"
    else:
        return False, "処理エラー（テスト）"


if __name__ == "__main__":
    # 使用例・テスト
    print("=== P1-019 安定バッチ処理システム テスト ===")
    
    # テスト用ファイルリスト作成
    test_files = [f"test_file_{i:03d}.jpg" for i in range(1, 16)]
    
    # 安定バッチ処理実行
    processor = StableBatchProcessor(
        checkpoint_dir="/tmp/p1_019_checkpoint",
        micro_batch_size=3,
        max_retries=2
    )
    
    result = processor.process_with_checkpoint(
        files=test_files,
        process_function=sample_processing_function,
        output_dir="/tmp/test_output"
    )
    
    print(f"\n📊 処理結果:")
    print(f"成功: {result['success']}")
    print(f"メッセージ: {result['message']}")
    print(f"統計: {result['stats']}")