#!/usr/bin/env python3
"""
P1-011キュー管理システム テスト実行
実際の抽出処理の代わりにモック処理でテスト
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.queue_manager import (
    ProcessingQueue,
    QueueConfig,
    QueueTask,
    TaskPriority,
    TaskStatus,
    ProcessingMode
)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockExtractionQueue(ProcessingQueue):
    """モック抽出キュー（テスト用）"""
    
    def __init__(self, config=None, output_dir=None):
        super().__init__(config)
        self.output_dir = output_dir or Path("./test_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _process_task(self, task: QueueTask) -> Any:
        """モック処理（実際の抽出の代わり）"""
        image_path = Path(task.image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {task.image_path}")
        
        logger.info(f"Mock processing: {image_path.name} (Priority: {task.priority.name})")
        
        # 処理時間をシミュレート（優先度によって変わる）
        if task.priority == TaskPriority.HIGH:
            processing_time = 0.1  # HIGH優先度は高速
        elif task.priority == TaskPriority.NORMAL:
            processing_time = 0.2  # NORMAL優先度は中程度
        else:
            processing_time = 0.3  # LOW優先度は低速
        
        time.sleep(processing_time)
        
        # モック品質スコア（ファイルサイズベース）
        file_size = image_path.stat().st_size
        quality_score = min((file_size / (1024 * 1024)) / 2.0, 1.0)  # 2MB=1.0
        
        # モック出力ファイル作成
        output_file = self.output_dir / f"{image_path.stem}_extracted.txt"
        output_file.write_text(f"Mock extraction result for {image_path.name}\n"
                              f"Quality: {quality_score:.3f}\n"
                              f"Processing time: {processing_time:.1f}s\n")
        
        return {
            "status": "success",
            "processing_time": processing_time,
            "quality_score": quality_score,
            "output_file": str(output_file),
            "mock": True
        }


def test_p1_011_queue_system():
    """P1-011キューシステムテスト"""
    print("🚀 P1-011キュー管理システムテスト開始")
    
    # テスト用設定
    config = QueueConfig(
        max_workers=2,
        processing_mode=ProcessingMode.ADAPTIVE,
        memory_threshold_mb=4000.0,
        timeout_seconds=60.0,
        enable_retry=True,
        enable_statistics=True,
        auto_priority=True,
        batch_size=5
    )
    
    # ワークスペース準備
    workspace_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-011")
    test_output_dir = workspace_dir / "test_extraction"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # モックキュー作成
    queue = MockExtractionQueue(config, test_output_dir)
    
    # 入力画像リスト
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana05")
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False
    
    # 画像ファイル収集（テスト用に最初の10枚のみ）
    image_paths = list(input_dir.glob("*.jpg"))[:10]
    
    if not image_paths:
        print(f"❌ テスト画像が見つかりません: {input_dir}")
        return False
    
    print(f"📊 テスト対象画像: {len(image_paths)}枚")
    
    # バッチタスク追加
    start_time = datetime.now()
    task_ids = queue.add_batch_tasks([str(p) for p in image_paths])
    
    print(f"✅ キューにタスク追加完了: {len(task_ids)}タスク")
    
    # ワーカー開始
    queue.start_workers(config.max_workers)
    print(f"🔧 ワーカー開始: {config.max_workers}スレッド")
    
    # 処理監視
    try:
        processed_count = 0
        while processed_count < len(task_ids):
            time.sleep(1)
            
            status = queue.get_queue_status()
            new_processed = status["completed_count"] + status["failed_count"]
            
            if new_processed > processed_count:
                processed_count = new_processed
                progress = (processed_count / len(task_ids)) * 100
                print(f"📈 進捗: {progress:.1f}% ({processed_count}/{len(task_ids)}) "
                      f"- 成功:{status['completed_count']}, 失敗:{status['failed_count']}")
            
            # タイムアウト防止（30秒最大）
            if (datetime.now() - start_time).total_seconds() > 30:
                print("⏱️ タイムアウト（30秒）")
                break
        
        # 最終結果
        final_status = queue.get_queue_status()
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 出力ファイル確認
        output_files = list(test_output_dir.glob("*.txt"))
        
        # queue_statisticsをシリアライザブルに変換
        serializable_stats = {}
        for key, value in final_status["statistics"].items():
            if isinstance(value, datetime):
                serializable_stats[key] = value.isoformat()
            else:
                serializable_stats[key] = value
        
        # 結果集計
        result = {
            "test_name": "P1-011 Queue System Test",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time,
            "input_images": len(image_paths),
            "total_tasks": len(task_ids),
            "completed_tasks": final_status["completed_count"],
            "failed_tasks": final_status["failed_count"],
            "output_files": len(output_files),
            "success_rate": (final_status["completed_count"] / len(task_ids)) * 100,
            "queue_statistics": serializable_stats,
            "config": {
                "max_workers": config.max_workers,
                "processing_mode": config.processing_mode.value,
                "auto_priority": config.auto_priority,
                "batch_size": config.batch_size
            }
        }
        
        # 結果保存
        result_file = test_output_dir / "test_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 結果表示
        print("\n" + "="*60)
        print("🏁 P1-011キュー管理システムテスト結果")
        print("="*60)
        print(f"⏱️  実行時間: {total_time:.2f}秒")
        print(f"📊 成功率: {result['success_rate']:.1f}%")
        print(f"✅ 完了タスク: {result['completed_tasks']}/{result['total_tasks']}")
        print(f"❌ 失敗タスク: {result['failed_tasks']}")
        print(f"📁 出力ファイル: {result['output_files']}個")
        print(f"💾 結果保存: {result_file}")
        print("="*60)
        
        # 成功判定
        success = result['success_rate'] >= 80.0
        print(f"🎯 テスト結果: {'✅ 成功' if success else '❌ 失敗'}")
        
        return success
        
    finally:
        queue.stop_workers()


def main():
    """メイン実行"""
    try:
        success = test_p1_011_queue_system()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        print(f"❌ テストエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())