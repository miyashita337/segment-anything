#!/usr/bin/env python3
"""
シンプルバッチ実行システム（同期処理版）

非同期処理の複雑性を排除し、確実な処理を優先
- 同期的な順次処理
- psutilによる確実なプロセス管理
- 実用的なタイムアウト設定
"""

import os
import sys
import time
import json
import psutil
import signal
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.api_config import get_api_config
from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskStatus, ComponentStatus
from features.common.notification.global_pushover import (
    send_pushover_notification, notify_success, notify_error, 
    notify_warning, notify_process_complete
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessManager:
    """確実なプロセス管理"""
    
    @staticmethod
    def kill_process_tree(pid: int):
        """プロセスツリー全体を確実に終了"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # 子プロセスから終了
            for child in children:
                try:
                    logger.info(f"子プロセス終了: PID {child.pid}")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 少し待機
            psutil.wait_procs(children, timeout=5)
            
            # 残っている子プロセスを強制終了
            for child in children:
                try:
                    if child.is_running():
                        logger.warning(f"子プロセス強制終了: PID {child.pid}")
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # 親プロセス終了
            try:
                parent.terminate()
                parent.wait(timeout=5)
            except psutil.TimeoutExpired:
                logger.warning(f"親プロセス強制終了: PID {pid}")
                parent.kill()
            except psutil.NoSuchProcess:
                pass
                
            logger.info(f"プロセスツリー終了完了: PID {pid}")
            return True
            
        except Exception as e:
            logger.error(f"プロセスツリー終了エラー: {e}")
            return False
    
    @staticmethod
    def cleanup_sam_processes():
        """SAM関連の残留プロセスをクリーンアップ"""
        cleaned = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('sam_yolo_character_segment.py' in arg for arg in cmdline):
                    logger.warning(f"残留SAMプロセス検出: PID {proc.info['pid']}")
                    ProcessManager.kill_process_tree(proc.info['pid'])
                    cleaned += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if cleaned > 0:
            logger.info(f"{cleaned}個の残留プロセスをクリーンアップしました")
        return cleaned


class SimpleBatchRunner:
    """シンプルバッチ実行システム"""
    
    def __init__(self, tracker_id: str, use_google_sheets: bool = False, 
                 use_pushover: bool = True, pushover_interval_minutes: int = 30):
        self.tracker_id = tracker_id
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace_dir = self.workspace_base / tracker_id
        
        # Google Sheets連携
        self.use_google_sheets = use_google_sheets
        self.progress_manager = None
        if use_google_sheets:
            try:
                self.progress_manager = ProgressManager()
                logger.info("Google Sheets連携を有効化")
            except Exception as e:
                logger.warning(f"Google Sheets連携の初期化に失敗: {e}")
                self.use_google_sheets = False
        
        # Pushover通知設定
        self.use_pushover = use_pushover
        self.pushover_interval = pushover_interval_minutes * 60  # 秒に変換
        self.last_pushover_time = datetime.now()
        self.pushover_milestones = [25, 50, 75]  # 通知するマイルストーン（％）
        self.notified_milestones = set()
        
        # 実用的な設定
        self.max_images_per_batch = 2  # 安定性重視
        self.timeout_per_image = 300  # 5分/枚（実測ベース）
        self.batch_overhead = 60  # オーバーヘッド1分
        
        # 統計
        self.stats = {
            'start_time': datetime.now(),
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'timeout_images': 0,
            'batch_times': []  # 各バッチの処理時間を記録
        }
        
        logger.info(f"シンプルバッチランナー初期化: {tracker_id}")
    
    def calculate_timeout(self, num_images: int) -> int:
        """現実的なタイムアウト計算"""
        base_timeout = num_images * self.timeout_per_image + self.batch_overhead
        # 50%の余裕を追加
        return int(base_timeout * 1.5)
    
    def calculate_estimated_completion(self) -> Dict[str, Any]:
        """推定完了時刻を計算"""
        if not self.stats['batch_times'] or self.stats['total_images'] == 0:
            return {
                'estimated_completion': None,
                'remaining_time': None,
                'avg_time_per_image': None
            }
        
        # 平均処理時間を計算
        avg_batch_time = sum(self.stats['batch_times']) / len(self.stats['batch_times'])
        avg_time_per_image = avg_batch_time / self.max_images_per_batch
        
        # 残り画像数と推定時間
        remaining_images = self.stats['total_images'] - self.stats['processed_images'] - self.stats['failed_images']
        remaining_batches = (remaining_images + self.max_images_per_batch - 1) // self.max_images_per_batch
        estimated_remaining_seconds = remaining_batches * avg_batch_time
        
        # 推定完了時刻
        estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
        
        return {
            'estimated_completion': estimated_completion,
            'remaining_time': timedelta(seconds=estimated_remaining_seconds),
            'avg_time_per_image': avg_time_per_image,
            'remaining_images': remaining_images
        }
    
    def send_progress_notification(self, force: bool = False, is_milestone: bool = False):
        """進捗通知を送信"""
        if not self.use_pushover:
            return
        
        # 時間チェック（マイルストーン以外）
        if not force and not is_milestone:
            elapsed = (datetime.now() - self.last_pushover_time).total_seconds()
            if elapsed < self.pushover_interval:
                return
        
        # 進捗率計算
        progress_rate = (self.stats['processed_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
        
        # 推定完了時刻
        estimation = self.calculate_estimated_completion()
        
        # メッセージ構築
        title = f"📊 バッチ処理進捗 [{self.stats['processed_images']}/{self.stats['total_images']}]"
        
        message_lines = [
            f"{'━' * 30}",
            f"✅ 進捗: {progress_rate:.1f}% ({self.stats['processed_images']}/{self.stats['total_images']}枚)",
        ]
        
        # 経過時間
        elapsed_time = datetime.now() - self.stats['start_time']
        elapsed_str = str(elapsed_time).split('.')[0]  # ミリ秒削除
        message_lines.append(f"⏱️ 経過: {elapsed_str}")
        
        # 推定完了時刻
        if estimation['estimated_completion']:
            message_lines.append(f"🎯 推定完了: {estimation['estimated_completion'].strftime('%H:%M')}")
            remaining_str = str(estimation['remaining_time']).split('.')[0]
            message_lines.append(f"⏳ 残り時間: {remaining_str}")
        
        # 成功率
        if self.stats['processed_images'] > 0:
            success_rate = ((self.stats['processed_images'] - self.stats['failed_images']) / self.stats['processed_images'] * 100)
            message_lines.append(f"📈 成功率: {success_rate:.1f}% ({self.stats['processed_images'] - self.stats['failed_images']}/{self.stats['processed_images']})")
        
        # 処理速度
        if estimation['avg_time_per_image']:
            message_lines.append(f"💨 処理速度: {estimation['avg_time_per_image']:.0f}秒/枚")
        
        # 失敗情報
        if self.stats['failed_images'] > 0:
            message_lines.append(f"❌ 失敗: {self.stats['failed_images']}枚")
        
        message_lines.append(f"{'━' * 30}")
        
        # マイルストーン通知の場合は優先度を上げる
        priority = 1 if is_milestone else 0
        
        # 通知送信
        message = "\n".join(message_lines)
        send_pushover_notification(title, message, priority=priority)
        
        self.last_pushover_time = datetime.now()
        logger.info(f"Pushover通知送信: 進捗 {progress_rate:.1f}%")
    
    def process_batch(self, image_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
        """バッチ処理実行"""
        batch_result = {
            'input_images': len(image_paths),
            'processed': 0,
            'failed': 0,
            'timeout': False,
            'output_files': []
        }
        
        # 一時ディレクトリ準備
        temp_dir = output_dir / f"batch_temp_{int(time.time())}"
        temp_input = temp_dir / "input"
        temp_output = temp_dir / "output"
        
        try:
            # ディレクトリ作成
            temp_input.mkdir(parents=True, exist_ok=True)
            temp_output.mkdir(parents=True, exist_ok=True)
            
            # 画像をコピー
            import shutil
            for img_path in image_paths:
                shutil.copy2(img_path, temp_input / img_path.name)
            
            # コマンド準備
            command = [
                sys.executable,  # 現在のPythonインタープリタ
                "tools/core/sam_yolo_character_segment.py",
                "--mode", "reproduce-auto",
                "--input_dir", str(temp_input),
                "--output_dir", str(temp_output),
                "--score_threshold", "0.07"
            ]
            
            # タイムアウト計算
            timeout = self.calculate_timeout(len(image_paths))
            logger.info(f"バッチ処理開始: {len(image_paths)}枚, タイムアウト: {timeout}秒")
            
            # プロセス実行
            start_time = time.time()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                env=dict(os.environ, PYTHONPATH=str(self.project_root)),
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            
            try:
                # タイムアウト付き待機
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
                if return_code == 0:
                    # 成功: 結果を移動
                    for output_file in temp_output.glob("*"):
                        if output_file.is_file():
                            shutil.move(str(output_file), str(output_dir / output_file.name))
                            batch_result['output_files'].append(output_file.name)
                            batch_result['processed'] += 1
                    
                    elapsed = time.time() - start_time
                    # バッチ処理時間を記録
                    self.stats['batch_times'].append(elapsed)
                    logger.info(f"バッチ処理成功: {batch_result['processed']}枚, {elapsed:.1f}秒")
                else:
                    batch_result['failed'] = len(image_paths)
                    logger.error(f"バッチ処理失敗: リターンコード {return_code}")
                    if stderr:
                        logger.error(f"エラー出力: {stderr.decode()[:500]}")
                        
            except subprocess.TimeoutExpired:
                batch_result['timeout'] = True
                batch_result['failed'] = len(image_paths)
                elapsed = time.time() - start_time
                logger.error(f"バッチ処理タイムアウト: {elapsed:.1f}秒経過")
                
                # プロセス終了
                ProcessManager.kill_process_tree(process.pid)
                
        except Exception as e:
            batch_result['failed'] = len(image_paths)
            logger.error(f"バッチ処理エラー: {e}")
            
        finally:
            # 一時ディレクトリクリーンアップ
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info("一時ディレクトリクリーンアップ完了")
            except Exception as e:
                logger.warning(f"一時ディレクトリクリーンアップエラー: {e}")
        
        return batch_result
    
    def run(self, input_dir: Path) -> bool:
        """メイン実行"""
        # 事前クリーンアップ
        ProcessManager.cleanup_sam_processes()
        
        # Google Sheets開始通知
        if self.use_google_sheets and self.progress_manager:
            try:
                self.progress_manager.update_status(
                    self.tracker_id, 
                    TaskStatus.EXTRACTION_PIPELINE,
                    ComponentStatus.IN_PROGRESS,
                    "extraction_pipeline"
                )
                logger.info(f"Google Sheets更新: {self.tracker_id} - 抽出パイプライン開始")
            except Exception as e:
                logger.warning(f"Google Sheets更新エラー: {e}")
        
        # 画像リスト取得
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(input_dir.glob(ext))
        
        if not image_paths:
            logger.error(f"処理対象画像が見つかりません: {input_dir}")
            self._update_sheets_failure("入力画像が見つかりません")
            return False
        
        self.stats['total_images'] = len(image_paths)
        logger.info(f"処理対象: {len(image_paths)}枚")
        
        # 出力ディレクトリ準備
        output_dir = self.workspace_dir / "extraction"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 開始通知を送信
        if self.use_pushover:
            title = "🚀 バッチ処理開始"
            message = f"トラッカー: {self.tracker_id}\n画像数: {len(image_paths)}枚\n推定時間: {len(image_paths) * self.timeout_per_image / 3600:.1f}時間"
            send_pushover_notification(title, message, priority=0, sound="cosmic")
            logger.info("Pushover開始通知送信")
        
        # バッチ処理実行
        processed_total = 0
        failed_total = 0
        
        for i in range(0, len(image_paths), self.max_images_per_batch):
            batch_images = image_paths[i:i + self.max_images_per_batch]
            batch_num = (i // self.max_images_per_batch) + 1
            total_batches = (len(image_paths) + self.max_images_per_batch - 1) // self.max_images_per_batch
            
            logger.info(f"\n=== バッチ {batch_num}/{total_batches} ===")
            
            result = self.process_batch(batch_images, output_dir)
            
            processed_total += result['processed']
            failed_total += result['failed']
            
            if result['timeout']:
                self.stats['timeout_images'] += len(batch_images)
            
            # 統計を更新
            self.stats['processed_images'] = processed_total
            self.stats['failed_images'] = failed_total
            
            # 進捗表示
            progress = (processed_total + failed_total) / len(image_paths) * 100
            logger.info(f"進捗: {progress:.1f}% ({processed_total + failed_total}/{len(image_paths)})")
            
            # マイルストーン通知チェック
            for milestone in self.pushover_milestones:
                if progress >= milestone and milestone not in self.notified_milestones:
                    self.send_progress_notification(force=True, is_milestone=True)
                    self.notified_milestones.add(milestone)
                    break
            else:
                # 定期通知チェック
                self.send_progress_notification()
            
            # Google Sheets進捗更新（10枚ごと）
            self._update_sheets_progress(processed_total, failed_total)
            
            # 短い休憩（GPU冷却）
            if batch_num < total_batches:
                time.sleep(5)
        
        # 最終統計
        self.stats['processed_images'] = processed_total
        self.stats['failed_images'] = failed_total
        self.stats['end_time'] = datetime.now()
        
        # レポート生成
        report_data = self.generate_report(output_dir)
        
        # Google Sheets最終更新
        success_rate = (processed_total / len(image_paths) * 100) if image_paths else 0
        if self.use_google_sheets and self.progress_manager:
            self._update_sheets_completion(report_data, success_rate >= 50.0)
        
        # 事後クリーンアップ
        ProcessManager.cleanup_sam_processes()
        
        logger.info(f"\n処理完了: 成功率 {success_rate:.1f}% ({processed_total}/{len(image_paths)})")
        
        # 完了通知を送信
        if self.use_pushover:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            notify_process_complete(
                title=f"🎉 {self.tracker_id} 処理完了",
                successful=processed_total,
                total=len(image_paths),
                failed=failed_total,
                duration=duration
            )
            logger.info("Pushover完了通知送信")
        
        return success_rate >= 50.0
    
    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """処理レポート生成"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        report = {
            'tracker_id': self.tracker_id,
            'execution_time': self.stats['start_time'].isoformat(),
            'end_time': self.stats['end_time'].isoformat(),
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'total_images': self.stats['total_images'],
            'processed_images': self.stats['processed_images'],
            'failed_images': self.stats['failed_images'],
            'timeout_images': self.stats['timeout_images'],
            'success_rate': (self.stats['processed_images'] / self.stats['total_images'] * 100) 
                           if self.stats['total_images'] > 0 else 0,
            'avg_time_per_image': duration / self.stats['total_images'] 
                                 if self.stats['total_images'] > 0 else 0,
            'fps': (self.stats['total_images'] / duration * 3600) 
                  if duration > 0 else 0  # 枚/時間
        }
        
        report_path = output_dir / "simple_batch_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"レポート生成: {report_path}")
        except Exception as e:
            logger.warning(f"レポート生成エラー: {e}")
        
        return report
    
    def _update_sheets_failure(self, error_message: str):
        """Google Sheets失敗時更新"""
        if not (self.use_google_sheets and self.progress_manager):
            return
        
        try:
            self.progress_manager.update_status(
                self.tracker_id,
                TaskStatus.EXTRACTION_PIPELINE, 
                ComponentStatus.FAILED,
                "extraction_pipeline"
            )
            
            # 詳細情報更新
            details = {
                "error": error_message,
                "timestamp": datetime.now().isoformat(),
                "total_images": self.stats.get('total_images', 0)
            }
            self.progress_manager.update_details(self.tracker_id, json.dumps(details, ensure_ascii=False))
            
            logger.info(f"Google Sheets失敗更新完了: {self.tracker_id}")
        except Exception as e:
            logger.warning(f"Google Sheets失敗更新エラー: {e}")
    
    def _update_sheets_completion(self, report_data: Dict[str, Any], is_success: bool):
        """Google Sheets完了時更新"""
        if not (self.use_google_sheets and self.progress_manager):
            return
        
        try:
            # ステータス更新
            final_status = TaskStatus.RELEASE if is_success else TaskStatus.EXTRACTION_PIPELINE
            component_status = ComponentStatus.COMPLETED if is_success else ComponentStatus.FAILED
            
            self.progress_manager.update_status(
                self.tracker_id,
                final_status,
                component_status,
                "extraction_pipeline"
            )
            
            # メトリクス更新
            success_rate = f"{report_data['success_rate']:.1f}%"
            fps_value = f"{report_data['fps']:.1f}"
            
            # Google Sheetsの各フィールドに更新
            updates = {
                'ab_evaluation_rate': success_rate,  # O列
                'fps': fps_value,  # P列
                'details': json.dumps({
                    "total_images": report_data['total_images'],
                    "processed": report_data['processed_images'], 
                    "failed": report_data['failed_images'],
                    "timeout": report_data['timeout_images'],
                    "execution_time": f"{report_data['duration_hours']:.1f}時間",
                    "avg_time_per_image": f"{report_data['avg_time_per_image']:.1f}秒",
                    "success_rate": success_rate,
                    "end_time": report_data['end_time']
                }, ensure_ascii=False)
            }
            
            for field, value in updates.items():
                try:
                    if field == 'details':
                        self.progress_manager.update_details(self.tracker_id, value)
                    else:
                        # メトリクス更新のための専用メソッド呼び出し
                        self.progress_manager.update_metrics(self.tracker_id, {field: value})
                except Exception as field_error:
                    logger.warning(f"フィールド {field} 更新エラー: {field_error}")
            
            logger.info(f"Google Sheets完了更新完了: {self.tracker_id}, 成功率: {success_rate}")
            
        except Exception as e:
            logger.warning(f"Google Sheets完了更新エラー: {e}")
    
    def _update_sheets_progress(self, current_processed: int, current_failed: int):
        """Google Sheets進捗更新（10枚ごと）"""
        if not (self.use_google_sheets and self.progress_manager):
            return
        
        if (current_processed + current_failed) % 10 != 0:
            return  # 10枚ごとのみ更新
        
        try:
            progress_rate = ((current_processed + current_failed) / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
            current_success_rate = (current_processed / (current_processed + current_failed) * 100) if (current_processed + current_failed) > 0 else 0
            
            progress_details = {
                "progress_rate": f"{progress_rate:.1f}%",
                "current_success_rate": f"{current_success_rate:.1f}%", 
                "processed": current_processed,
                "failed": current_failed,
                "remaining": self.stats['total_images'] - current_processed - current_failed,
                "timestamp": datetime.now().isoformat()
            }
            
            self.progress_manager.update_details(self.tracker_id, json.dumps(progress_details, ensure_ascii=False))
            logger.info(f"Google Sheets進捗更新: {progress_rate:.1f}%完了 ({current_processed}成功/{current_failed}失敗)")
            
        except Exception as e:
            logger.warning(f"Google Sheets進捗更新エラー: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="シンプルバッチ実行システム")
    parser.add_argument("tracker_id", help="トラッカーID")
    parser.add_argument("--input-dir", required=True, help="入力ディレクトリ")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--timeout", type=int, default=300, help="画像あたりタイムアウト（秒）")
    parser.add_argument("--google-sheets", action="store_true", help="Google Sheets連携を有効化")
    parser.add_argument("--no-pushover", action="store_true", help="Pushover通知を無効化")
    parser.add_argument("--pushover-interval", type=int, default=30, help="Pushover通知間隔（分）")
    
    args = parser.parse_args()
    
    runner = SimpleBatchRunner(
        args.tracker_id, 
        use_google_sheets=args.google_sheets,
        use_pushover=not args.no_pushover,
        pushover_interval_minutes=args.pushover_interval
    )
    if args.batch_size:
        runner.max_images_per_batch = args.batch_size
    if args.timeout:
        runner.timeout_per_image = args.timeout
    
    try:
        success = runner.run(Path(args.input_dir))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        ProcessManager.cleanup_sam_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()