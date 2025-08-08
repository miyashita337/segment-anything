#!/usr/bin/env python3
"""
P1-011処理キュー管理を使った抽出パイプライン
大量画像の効率的処理順序制御による抽出実行
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from features.common.retry_handler import RetryHandler, RetryConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CharacterExtractionQueue(ProcessingQueue):
    """キャラクター抽出特化のキュー"""
    
    def __init__(self, config: Optional[QueueConfig] = None, output_dir: Optional[Path] = None):
        super().__init__(config)
        self.output_dir = output_dir or Path("./extraction_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # リトライハンドラー初期化
        retry_config = RetryConfig(
            max_retries=2,  # P1-011: 効率重視で2回まで
            initial_delay=3.0,
            exponential_backoff=True,
            quality_retry_enabled=True,
            quality_threshold=0.3  # 低めの閾値で実用性重視
        )
        self.retry_handler = RetryHandler(retry_config)
        
        logger.info(f"CharacterExtractionQueue initialized: output_dir={self.output_dir}")
    
    def _process_task(self, task: QueueTask) -> Any:
        """実際のキャラクター抽出処理"""
        image_path = Path(task.image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {task.image_path}")
        
        logger.info(f"抽出開始: {image_path.name} (Priority: {task.priority.name})")
        
        # 出力ディレクトリ準備
        task_output_dir = self.output_dir / f"task_{task.task_id}"
        task_output_dir.mkdir(exist_ok=True)
        
        try:
            # SAM+YOLO抽出コマンド実行
            result = self._execute_extraction(image_path, task_output_dir)
            
            # 品質評価
            quality_score = self._evaluate_extraction_quality(task_output_dir)
            result["quality_score"] = quality_score
            
            # 成功時は結果をメイン出力ディレクトリに移動
            if quality_score > 0.3:  # 品質閾値
                self._move_results_to_main_output(task_output_dir, image_path.stem)
                result["moved_to_main"] = True
            else:
                result["moved_to_main"] = False
                logger.warning(f"低品質のため移動しません: {image_path.name} (score: {quality_score:.3f})")
            
            logger.info(f"抽出完了: {image_path.name} (score: {quality_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"抽出失敗: {image_path.name} - {str(e)}")
            raise
        finally:
            # 一時ディレクトリクリーンアップ
            self._cleanup_temp_dir(task_output_dir)
    
    def _execute_extraction(self, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        """SAM+YOLO抽出実行"""
        start_time = time.time()
        
        # 単一画像用の一時ディレクトリ作成
        temp_input_dir = output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        # 画像をシンボリックリンクで一時ディレクトリにコピー
        temp_image_path = temp_input_dir / image_path.name
        if not temp_image_path.exists():
            try:
                temp_image_path.symlink_to(image_path)
            except OSError:
                # シンボリックリンク失敗時は実ファイルコピー
                import shutil
                shutil.copy2(image_path, temp_image_path)
        
        # 抽出コマンド（一時ディレクトリを入力とする）
        command = [
            "python3", "tools/core/sam_yolo_character_segment.py",
            "--mode", "reproduce-auto",
            "--input_dir", str(temp_input_dir),
            "--output_dir", str(output_dir),
            "--score_threshold", "0.07"
        ]
        
        # プロジェクトルートで実行
        project_root = Path(__file__).parent.parent.parent
        
        logger.debug(f"抽出コマンド実行: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=180,  # 3分タイムアウト
                env=dict(os.environ, PYTHONPATH=str(project_root))
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "stdout": result.stdout,
                    "command": " ".join(command)
                }
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stdout, result.stderr
                )
                
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"抽出処理タイムアウト: {image_path.name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"抽出コマンドエラー: {e.stderr}")
        finally:
            # 一時入力ディレクトリクリーンアップ
            try:
                import shutil
                if temp_input_dir.exists():
                    shutil.rmtree(temp_input_dir)
            except Exception:
                pass
    
    def _evaluate_extraction_quality(self, output_dir: Path) -> float:
        """抽出品質評価"""
        try:
            # 出力ファイル数をカウント
            extracted_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            
            if not extracted_files:
                return 0.0
            
            # ファイル数ベースの基本品質スコア
            file_count_score = min(len(extracted_files) / 3.0, 1.0)  # 最大3ファイルで1.0
            
            # ファイルサイズベース品質（小さすぎるファイルは低品質）
            size_scores = []
            for file_path in extracted_files:
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 1024:  # 1KB以上
                        size_scores.append(min(file_size / (100*1024), 1.0))  # 100KB=1.0
                    else:
                        size_scores.append(0.1)  # 小さすぎるファイルは低評価
                except Exception:
                    size_scores.append(0.1)
            
            avg_size_score = sum(size_scores) / len(size_scores) if size_scores else 0.0
            
            # 総合品質スコア
            final_score = (file_count_score * 0.7) + (avg_size_score * 0.3)
            
            logger.debug(f"品質評価: files={len(extracted_files)}, "
                        f"file_score={file_count_score:.3f}, "
                        f"size_score={avg_size_score:.3f}, "
                        f"final={final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            logger.warning(f"品質評価エラー: {e}")
            return 0.1  # エラー時は低品質として扱う
    
    def _move_results_to_main_output(self, task_output_dir: Path, image_stem: str):
        """結果をメイン出力ディレクトリに移動"""
        try:
            extracted_files = list(task_output_dir.glob("*.png")) + list(task_output_dir.glob("*.jpg"))
            
            for i, file_path in enumerate(extracted_files):
                # ファイル名に画像名を含める
                new_name = f"{image_stem}_extracted_{i:02d}{file_path.suffix}"
                dest_path = self.output_dir / new_name
                
                file_path.rename(dest_path)
                logger.debug(f"移動: {file_path.name} -> {dest_path.name}")
                
        except Exception as e:
            logger.warning(f"ファイル移動エラー: {e}")
    
    def _cleanup_temp_dir(self, temp_dir: Path):
        """一時ディレクトリクリーンアップ"""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"一時ディレクトリ削除: {temp_dir}")
        except Exception as e:
            logger.warning(f"一時ディレクトリ削除エラー: {e}")


class P1011ExtractionRunner:
    """P1-011抽出実行管理"""
    
    def __init__(self, tracker_id: str = "P1-011"):
        self.tracker_id = tracker_id
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace_dir = self.workspace_base / tracker_id
        self.extraction_dir = self.workspace_dir / "extraction"
        
        # ディレクトリ作成
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # キュー設定
        self.queue_config = QueueConfig(
            max_workers=2,  # P1-011: 並列処理で効率化
            processing_mode=ProcessingMode.ADAPTIVE,
            memory_threshold_mb=6000.0,  # 6GB閾値
            timeout_seconds=240.0,  # 4分タイムアウト
            enable_retry=True,
            enable_statistics=True,
            auto_priority=True,  # P1-011: サイズベース自動優先度
            batch_size=8
        )
        
        # 抽出キュー初期化
        self.extraction_queue = CharacterExtractionQueue(
            config=self.queue_config,
            output_dir=self.extraction_dir
        )
        
        logger.info(f"P1011ExtractionRunner initialized: {tracker_id}")
        logger.info(f"  workspace: {self.workspace_dir}")
        logger.info(f"  extraction: {self.extraction_dir}")
    
    def run_extraction(self, input_dir: Path) -> Dict[str, Any]:
        """キュー管理による抽出実行"""
        if not input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")
        
        # 画像ファイル収集
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_paths = []
        
        for extension in image_extensions:
            paths = list(input_dir.glob(extension))
            image_paths.extend(paths)
        
        if not image_paths:
            raise ValueError(f"処理可能な画像が見つかりません: {input_dir}")
        
        logger.info(f"抽出対象画像: {len(image_paths)}枚")
        
        # バッチタスク追加
        task_ids = self.extraction_queue.add_batch_tasks([str(p) for p in image_paths])
        
        logger.info(f"キューにタスク追加完了: {len(task_ids)}タスク")
        
        # 処理開始
        start_time = datetime.now()
        self.extraction_queue.start_workers(self.queue_config.max_workers)
        
        try:
            # 処理完了まで監視
            self._monitor_processing_progress(len(task_ids))
            
            # 最終結果集計
            final_status = self.extraction_queue.get_queue_status()
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 結果ファイル数カウント
            extracted_files = list(self.extraction_dir.glob("*.png")) + list(self.extraction_dir.glob("*.jpg"))
            
            result = {
                "tracker_id": self.tracker_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "processing_time_seconds": processing_time,
                "input_images": len(image_paths),
                "total_tasks": len(task_ids),
                "completed_tasks": final_status["completed_count"],
                "failed_tasks": final_status["failed_count"],
                "extracted_files": len(extracted_files),
                "success_rate": (final_status["completed_count"] / len(task_ids)) * 100,
                "extraction_rate": (len(extracted_files) / len(image_paths)) * 100,
                "queue_statistics": final_status["statistics"],
                "workspace_dir": str(self.workspace_dir),
                "extraction_dir": str(self.extraction_dir)
            }
            
            logger.info(f"P1-011抽出完了: {result['success_rate']:.1f}%成功率, "
                       f"{result['extraction_rate']:.1f}%抽出率, "
                       f"{processing_time:.1f}秒")
            
            return result
            
        finally:
            self.extraction_queue.stop_workers()
    
    def _monitor_processing_progress(self, total_tasks: int):
        """処理進捗監視"""
        logger.info("処理進捗監視開始")
        
        last_progress = 0
        max_wait_time = 30 * 60  # 30分最大待機
        start_monitor_time = time.time()
        
        while True:
            status = self.extraction_queue.get_queue_status()
            
            completed = status["completed_count"] + status["failed_count"]
            progress = (completed / total_tasks) * 100 if total_tasks > 0 else 0
            
            # 進捗表示（10%刻み）
            if progress >= last_progress + 10:
                logger.info(f"処理進捗: {progress:.1f}% ({completed}/{total_tasks})")
                logger.info(f"  成功: {status['completed_count']}, "
                           f"失敗: {status['failed_count']}, "
                           f"処理中: {status['processing_count']}")
                last_progress = int(progress / 10) * 10
            
            # 完了判定
            if completed >= total_tasks:
                logger.info("全タスク処理完了")
                break
            
            # タイムアウト判定
            if time.time() - start_monitor_time > max_wait_time:
                logger.warning(f"最大待機時間({max_wait_time}秒)に達しました")
                break
            
            time.sleep(5)  # 5秒間隔で監視


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-011処理キュー管理による抽出パイプライン")
    parser.add_argument("--tracker_id", type=str, default="P1-011", help="トラッカーID")
    parser.add_argument("--input_dir", type=str, 
                        default="/mnt/c/AItools/lora/train/yado/org/kana05",
                        help="入力ディレクトリ")
    parser.add_argument("--workers", type=int, default=2, help="ワーカー数")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="ログレベル")
    
    args = parser.parse_args()
    
    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # P1-011実行システム初期化
        runner = P1011ExtractionRunner(args.tracker_id)
        runner.queue_config.max_workers = args.workers
        
        # 抽出実行
        input_dir = Path(args.input_dir)
        result = runner.run_extraction(input_dir)
        
        # 結果出力
        print("\n" + "="*60)
        print("P1-011キュー管理抽出結果")
        print("="*60)
        print(f"トラッカーID: {result['tracker_id']}")
        print(f"処理時間: {result['processing_time_seconds']:.1f}秒")
        print(f"入力画像数: {result['input_images']}枚")
        print(f"成功率: {result['success_rate']:.1f}%")
        print(f"抽出ファイル数: {result['extracted_files']}個")
        print(f"出力先: {result['extraction_dir']}")
        print("="*60)
        
        # 成功判定
        if result['success_rate'] >= 70.0:
            print("✅ P1-011抽出成功")
            return 0
        else:
            print("❌ P1-011抽出失敗（成功率70%未満）")
            return 1
            
    except Exception as e:
        logger.error(f"P1-011抽出エラー: {e}")
        print(f"❌ エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())