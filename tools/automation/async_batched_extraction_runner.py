#!/usr/bin/env python3
"""
非同期バッチ分割抽出実行システム (完全asyncio対応版)

完全非同期処理による無制限タイムアウト対策
- asyncio: I/O処理の完全非同期化
- aiofiles: ファイル操作の非同期化
- asyncio.subprocess: サブプロセスの非同期実行
- セマフォによる同時実行数制御
"""

import aiofiles
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.api_config import get_api_config

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncGPUMonitor:
    """非同期GPU監視クラス"""
    
    @staticmethod
    async def check_gpu_available() -> Tuple[bool, Dict[str, Any]]:
        """GPU利用可能性を非同期チェック"""
        try:
            # GPU情報取得を別プロセスで実行
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor, AsyncGPUMonitor._check_gpu_sync
                )
            return result
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    def _check_gpu_sync() -> Tuple[bool, Dict[str, Any]]:
        """同期GPU情報取得（ProcessPoolExecutor用）"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return False, {'error': 'CUDA not available'}
            
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            gpu_info = {
                'device_count': device_count,
                'current_device': current_device,
                'device_name': torch.cuda.get_device_name(current_device),
                'memory_allocated': torch.cuda.memory_allocated(current_device),
                'memory_reserved': torch.cuda.memory_reserved(current_device),
                'memory_total': torch.cuda.get_device_properties(current_device).total_memory
            }
            
            memory_usage = gpu_info['memory_reserved'] / gpu_info['memory_total']
            gpu_info['memory_usage_percent'] = memory_usage * 100
            
            return True, gpu_info
            
        except ImportError:
            return False, {'error': 'PyTorch not available'}
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    async def cleanup_gpu_memory():
        """非同期GPU メモリクリーンアップ"""
        try:
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(executor, AsyncGPUMonitor._cleanup_gpu_sync)
            logger.info("GPU メモリクリーンアップ完了")
            return True
        except Exception as e:
            logger.warning(f"GPU メモリクリーンアップ失敗: {e}")
            return False
    
    @staticmethod
    def _cleanup_gpu_sync():
        """同期GPU メモリクリーンアップ（ProcessPoolExecutor用）"""
        try:
            import torch

            import gc
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass


class AsyncImageBatch:
    """非同期画像バッチ管理クラス"""
    
    def __init__(self, batch_id: int, image_paths: List[Path], estimated_time: float = 120):
        self.batch_id = batch_id
        self.image_paths = image_paths
        self.estimated_time = estimated_time
        self.actual_time = None
        self.status = 'pending'  # pending, running, completed, failed
        self.results = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        
    def __len__(self):
        return len(self.image_paths)
    
    def __str__(self):
        return f"AsyncBatch-{self.batch_id} ({len(self.image_paths)} images)"


class AsyncBatchedExtractionRunner:
    """非同期バッチ分割抽出実行システム"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace_dir = self.workspace_base / tracker_id
        
        # API設定取得
        self.api_config = get_api_config()
        
        # GPU情報は非同期で初期化
        self.gpu_info = {}
        
        # バッチ設定（現実的な時間に調整）
        self.batch_size = 4  # デフォルト、初期化後に動的調整（8→4に変更）
        self.max_concurrent_batches = 2  # 同時実行バッチ数
        self.max_batch_timeout = 600  # 10分/バッチ（300→600秒に延長）
        self.max_image_timeout = 120   # 2分/画像（60→120秒に延長）
        
        # 統計
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'total_batches': 0,
            'completed_batches': 0,
            'failed_batches': 0,
            'start_time': None,
            'end_time': None
        }
        
        # シャットダウンイベント
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"非同期バッチ抽出システム初期化: {tracker_id}")
    
    async def initialize(self):
        """非同期初期化"""
        # GPU情報取得
        gpu_available, self.gpu_info = await AsyncGPUMonitor.check_gpu_available()
        if not gpu_available:
            logger.warning(f"GPU利用不可: {self.gpu_info}")
        
        # 最適バッチサイズ計算
        self.batch_size = self._calculate_optimal_batch_size()
        logger.info(f"最適バッチサイズ: {self.batch_size}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """最適なバッチサイズを計算"""
        if not self.gpu_info.get('memory_total'):
            return 4  # デフォルト
        
        memory_gb = self.gpu_info['memory_total'] / (1024**3)
        
        # より保守的なバッチサイズ（安定性重視）
        if memory_gb >= 12:
            return 4   # 12GB以上: 4枚バッチ（8→4に削減）
        elif memory_gb >= 8:
            return 3   # 8GB以上: 3枚バッチ（6→3に削減）
        elif memory_gb >= 6:
            return 2   # 6GB以上: 2枚バッチ（4→2に削減）
        else:
            return 1   # 6GB未満: 1枚バッチ（2→1に削減）
    
    async def _create_batches(self, image_paths: List[Path]) -> List[AsyncImageBatch]:
        """画像リストをバッチに分割"""
        batches = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_images = image_paths[i:i + self.batch_size]
            
            # バッチ処理時間推定（現実的な時間: 画像数 × 60秒 + オーバーヘッド60秒）
            estimated_time = len(batch_images) * 60 + 60
            
            batch = AsyncImageBatch(
                batch_id=len(batches) + 1,
                image_paths=batch_images,
                estimated_time=estimated_time
            )
            batches.append(batch)
        
        logger.info(f"非同期バッチ作成完了: {len(batches)}バッチ ({len(image_paths)}枚)")
        return batches
    
    async def _create_batch_symlinks(self, batch: AsyncImageBatch, batch_input_dir: Path):
        """バッチ用シンボリックリンクを非同期作成"""
        for img_path in batch.image_paths:
            link_path = batch_input_dir / img_path.name
            if not link_path.exists():
                try:
                    # シンボリックリンク作成
                    link_path.symlink_to(img_path)
                except OSError:
                    # シンボリックリンク失敗時は非同期コピー
                    try:
                        async with aiofiles.open(img_path, 'rb') as src:
                            async with aiofiles.open(link_path, 'wb') as dst:
                                while True:
                                    chunk = await src.read(8192)
                                    if not chunk:
                                        break
                                    await dst.write(chunk)
                    except Exception as e:
                        logger.warning(f"ファイルコピー失敗: {img_path} -> {link_path}: {e}")
    
    async def _execute_extraction_command(self, batch: AsyncImageBatch, batch_input_dir: Path, batch_output_dir: Path) -> Tuple[int, str, str]:
        """抽出コマンドを非同期実行"""
        command = [
            "python3", "tools/core/sam_yolo_character_segment.py",
            "--mode", "reproduce-auto",
            "--input_dir", str(batch_input_dir),
            "--output_dir", str(batch_output_dir),
            "--score_threshold", "0.07"
        ]
        
        # タイムアウト計算（より余裕を持たせる）
        timeout = max(int(batch.estimated_time * 2.0), self.max_batch_timeout)
        
        logger.info(f"{batch} 非同期コマンド実行開始: タイムアウト{timeout}秒")
        
        process = None
        try:
            # 非同期サブプロセス実行
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=str(self.project_root))
            )
            
            # タイムアウト付き実行
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')
            
        except asyncio.TimeoutError:
            # 改善されたプロセス強制終了処理
            if process:
                logger.warning(f"{batch} タイムアウト発生、プロセス強制終了開始")
                
                try:
                    # まずTERMシグナルで穏やかに終了を試みる
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                    logger.info(f"{batch} プロセス正常終了")
                except asyncio.TimeoutError:
                    # 10秒待って終了しない場合はKILL
                    logger.warning(f"{batch} TERM無効、KILL実行")
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                        logger.info(f"{batch} プロセス強制終了完了")
                    except asyncio.TimeoutError:
                        logger.error(f"{batch} プロセス終了失敗（ゾンビプロセス化の可能性）")
                except Exception as e:
                    logger.error(f"{batch} プロセス終了エラー: {e}")
            raise
    
    async def _process_single_batch(self, batch: AsyncImageBatch, input_dir: Path, output_dir: Path) -> bool:
        """単一バッチを非同期処理"""
        batch.status = 'running'
        batch.start_time = time.time()
        
        logger.info(f"{batch} 非同期処理開始 (推定時間: {batch.estimated_time}秒)")
        
        # 一時ディレクトリの事前クリーンアップ
        await self._cleanup_existing_temp_dirs(output_dir, batch.batch_id)
        
        try:
            # バッチ用の一時ディレクトリ作成
            batch_temp_dir = output_dir / f"async_batch_{batch.batch_id}_temp"
            batch_input_dir = batch_temp_dir / "input"
            batch_output_dir = batch_temp_dir / "output"
            
            # ディレクトリ作成
            batch_temp_dir.mkdir(exist_ok=True)
            batch_input_dir.mkdir(exist_ok=True)
            batch_output_dir.mkdir(exist_ok=True)
            
            # シンボリックリンク作成
            await self._create_batch_symlinks(batch, batch_input_dir)
            
            # GPU メモリクリーンアップ
            await AsyncGPUMonitor.cleanup_gpu_memory()
            
            # 抽出コマンド実行
            returncode, stdout, stderr = await self._execute_extraction_command(
                batch, batch_input_dir, batch_output_dir
            )
            
            batch.end_time = time.time()
            batch.actual_time = batch.end_time - batch.start_time
            
            if returncode == 0:
                # 成功: 結果をメイン出力ディレクトリに移動
                if batch_output_dir.exists():
                    async for file_path in self._async_glob(batch_output_dir, "*"):
                        dest_path = output_dir / file_path.name
                        await asyncio.get_event_loop().run_in_executor(
                            None, shutil.move, str(file_path), str(dest_path)
                        )
                
                # 処理済み画像数カウント
                processed_count = 0
                for ext in ["*.jpg", "*.png"]:
                    async for _ in self._async_glob(batch_output_dir, ext):
                        processed_count += 1
                
                batch.results = [f"processed_{i}" for i in range(processed_count)]
                batch.status = 'completed'
                
                logger.info(f"{batch} 完了: {processed_count}枚処理 ({batch.actual_time:.1f}秒)")
                
                # 一時ディレクトリクリーンアップ（成功時）
                await self._safe_cleanup_temp_dir(batch_temp_dir, f"{batch}_success")
                
                return True
            else:
                # 失敗
                batch.status = 'failed'
                batch.errors.append(f"Return code: {returncode}")
                batch.errors.append(f"Stderr: {stderr}")
                
                logger.error(f"{batch} 失敗: {stderr[:200]}...")
                return False
        
        except asyncio.TimeoutError:
            batch.status = 'failed'
            batch.end_time = time.time()
            batch.actual_time = batch.end_time - batch.start_time
            batch.errors.append(f"タイムアウト ({batch.estimated_time * 1.5:.0f}秒)")
            
            logger.error(f"{batch} タイムアウト")
            # タイムアウト時の一時ディレクトリクリーンアップ
            if 'batch_temp_dir' in locals():
                await self._safe_cleanup_temp_dir(batch_temp_dir, f"{batch}_timeout")
            return False
            
        except Exception as e:
            batch.status = 'failed'
            batch.end_time = time.time()
            batch.actual_time = batch.end_time - batch.start_time
            batch.errors.append(f"処理エラー: {str(e)}")
            
            logger.error(f"{batch} エラー: {e}")
            # エラー時の一時ディレクトリクリーンアップ
            if 'batch_temp_dir' in locals():
                await self._safe_cleanup_temp_dir(batch_temp_dir, f"{batch}_error")
            return False
    
    async def _async_glob(self, path: Path, pattern: str):
        """非同期glob（アイテムを順次yield）"""
        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(None, lambda: list(path.glob(pattern)))
        for item in items:
            yield item
    
    async def _cleanup_existing_temp_dirs(self, output_dir: Path, current_batch_id: int):
        """既存の一時ディレクトリを事前クリーンアップ"""
        try:
            temp_pattern = f"async_batch_{current_batch_id}_temp"
            existing_temp = output_dir / temp_pattern
            
            if existing_temp.exists():
                logger.warning(f"既存の一時ディレクトリを検出、削除します: {existing_temp}")
                await self._safe_cleanup_temp_dir(existing_temp, f"existing_batch_{current_batch_id}")
                
        except Exception as e:
            logger.warning(f"事前クリーンアップエラー: {e}")
    
    async def _safe_cleanup_temp_dir(self, temp_dir: Path, context: str):
        """安全な一時ディレクトリクリーンアップ"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if not temp_dir.exists():
                    return True
                
                logger.info(f"一時ディレクトリクリーンアップ [{context}] (試行 {attempt + 1}/{max_retries}): {temp_dir}")
                
                # 読み取り専用属性を削除
                await asyncio.get_event_loop().run_in_executor(
                    None, self._remove_readonly_recursive, temp_dir
                )
                
                # ディレクトリ削除
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.rmtree, temp_dir, True
                )
                
                # 削除確認
                if not temp_dir.exists():
                    logger.info(f"一時ディレクトリクリーンアップ完了 [{context}]: {temp_dir}")
                    return True
                    
            except Exception as e:
                logger.warning(f"一時ディレクトリクリーンアップ失敗 [{context}] (試行 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数バックオフ
        
        logger.error(f"一時ディレクトリクリーンアップ最終失敗 [{context}]: {temp_dir}")
        return False
    
    def _remove_readonly_recursive(self, path: Path):
        """読み取り専用属性を再帰的に削除（Windows対応）"""
        try:
            for root, dirs, files in os.walk(path):
                # ディレクトリの権限変更
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.chmod(dir_path, 0o755)
                    except (OSError, PermissionError):
                        pass
                
                # ファイルの権限変更
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    try:
                        os.chmod(file_path, 0o755)
                    except (OSError, PermissionError):
                        pass
        except Exception as e:
            logger.warning(f"読み取り専用属性削除エラー: {e}")
    
    async def _process_individual_images(self, failed_batch: AsyncImageBatch, input_dir: Path, output_dir: Path) -> int:
        """失敗したバッチの個別画像を非同期処理"""
        logger.info(f"{failed_batch} 個別画像非同期処理開始")
        
        success_count = 0
        individual_temp_dir = output_dir / f"async_individual_{failed_batch.batch_id}_temp"
        individual_temp_dir.mkdir(exist_ok=True)
        
        # セマフォで同時実行数制御
        semaphore = asyncio.Semaphore(2)  # 最大2枚同時処理
        
        async def process_single_image(i: int, img_path: Path) -> bool:
            async with semaphore:
                try:
                    logger.info(f"個別非同期処理: {img_path.name} ({i+1}/{len(failed_batch.image_paths)})")
                    
                    # 1枚用の一時ディレクトリ
                    single_input_dir = individual_temp_dir / f"input_{i}"
                    single_output_dir = individual_temp_dir / f"output_{i}"
                    single_input_dir.mkdir(exist_ok=True)
                    single_output_dir.mkdir(exist_ok=True)
                    
                    # 非同期ファイルコピー
                    dest_path = single_input_dir / img_path.name
                    async with aiofiles.open(img_path, 'rb') as src:
                        async with aiofiles.open(dest_path, 'wb') as dst:
                            while True:
                                chunk = await src.read(8192)
                                if not chunk:
                                    break
                                await dst.write(chunk)
                    
                    # 個別実行
                    command = [
                        "python3", "tools/core/sam_yolo_character_segment.py",
                        "--mode", "reproduce-auto",
                        "--input_dir", str(single_input_dir),
                        "--output_dir", str(single_output_dir),
                        "--score_threshold", "0.07"
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=self.project_root,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=dict(os.environ, PYTHONPATH=str(self.project_root))
                    )
                    
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.max_image_timeout
                    )
                    
                    if process.returncode == 0:
                        # 成功: 結果を移動
                        async for file_path in self._async_glob(single_output_dir, "*"):
                            dest_path = output_dir / file_path.name
                            await asyncio.get_event_loop().run_in_executor(
                                None, shutil.move, str(file_path), str(dest_path)
                            )
                        
                        logger.info(f"個別処理成功: {img_path.name}")
                        return True
                    else:
                        logger.warning(f"個別処理失敗: {img_path.name} - {stderr.decode()[:100]}")
                        return False
                    
                except asyncio.TimeoutError:
                    logger.warning(f"個別処理タイムアウト: {img_path.name}")
                    return False
                except Exception as e:
                    logger.warning(f"個別処理エラー: {img_path.name} - {e}")
                    return False
                finally:
                    # 個別一時ディレクトリクリーンアップ
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, shutil.rmtree, single_input_dir, True
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None, shutil.rmtree, single_output_dir, True
                        )
                    except Exception:
                        pass
        
        # 全画像を並列処理
        tasks = [
            process_single_image(i, img_path)
            for i, img_path in enumerate(failed_batch.image_paths)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for result in results if result is True)
        
        # 全体一時ディレクトリクリーンアップ
        await asyncio.get_event_loop().run_in_executor(
            None, shutil.rmtree, individual_temp_dir, True
        )
        
        logger.info(f"{failed_batch} 個別処理完了: {success_count}/{len(failed_batch.image_paths)} 成功")
        return success_count
    
    async def run_async_batched_extraction(self, input_dir: Path, output_dir: Path) -> bool:
        """非同期バッチ分割抽出実行"""
        self.stats['start_time'] = datetime.now()
        
        # 入力ディレクトリ確認
        if not input_dir.exists():
            logger.error(f"入力ディレクトリが存在しません: {input_dir}")
            return False
        
        # 画像ファイルリスト取得
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_paths = []
        for extension in image_extensions:
            paths = await asyncio.get_event_loop().run_in_executor(
                None, lambda ext=extension: list(input_dir.glob(ext))
            )
            image_paths.extend(paths)
        
        if not image_paths:
            logger.error(f"処理可能な画像が見つかりません: {input_dir}")
            return False
        
        self.stats['total_images'] = len(image_paths)
        logger.info(f"処理対象画像: {len(image_paths)}枚")
        
        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # バッチ作成
        batches = await self._create_batches(image_paths)
        self.stats['total_batches'] = len(batches)
        
        # 非同期バッチ処理実行
        failed_batches = []
        
        print(f"\n🚀 非同期バッチ抽出処理開始")
        print(f"📊 {len(batches)}バッチ ({len(image_paths)}枚) を最大{self.max_concurrent_batches}並列で処理")
        print(f"⚙️  バッチサイズ: {self.batch_size}枚/バッチ")
        
        # セマフォで同時実行数制御
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch_with_semaphore(batch: AsyncImageBatch) -> Tuple[AsyncImageBatch, bool]:
            async with semaphore:
                success = await self._process_single_batch(batch, input_dir, output_dir)
                return batch, success
        
        # 全バッチを非同期実行
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        
        # 完了したタスクを順次処理
        for coro in asyncio.as_completed(tasks):
            batch, success = await coro
            
            if success:
                self.stats['completed_batches'] += 1
                self.stats['processed_images'] += len(batch)
            else:
                self.stats['failed_batches'] += 1
                failed_batches.append(batch)
            
            # 進捗表示
            completed = self.stats['completed_batches'] + self.stats['failed_batches']
            progress = (completed / len(batches)) * 100
            print(f"   📊 進捗: {progress:.1f}% ({completed}/{len(batches)}バッチ)")
        
        # 失敗バッチの個別処理
        if failed_batches:
            print(f"\n🔄 失敗バッチの個別非同期処理開始: {len(failed_batches)}バッチ")
            
            individual_tasks = [
                self._process_individual_images(failed_batch, input_dir, output_dir)
                for failed_batch in failed_batches
            ]
            
            individual_results = await asyncio.gather(*individual_tasks)
            
            for i, rescued_count in enumerate(individual_results):
                self.stats['processed_images'] += rescued_count
                self.stats['failed_images'] += len(failed_batches[i]) - rescued_count
        
        # 結果集計
        self.stats['end_time'] = datetime.now()
        execution_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # 最終結果確認
        final_output_count = 0
        for ext in ["*.jpg", "*.png"]:
            paths = await asyncio.get_event_loop().run_in_executor(
                None, lambda e=ext: list(output_dir.glob(e))
            )
            final_output_count += len(paths)
        
        success_rate = (final_output_count / self.stats['total_images']) * 100 if self.stats['total_images'] > 0 else 0
        
        # 結果表示
        print(f"\n🏁 非同期バッチ抽出処理完了")
        print(f"⏱️  実行時間: {execution_time:.1f}秒")
        print(f"📊 処理成功率: {success_rate:.1f}% ({final_output_count}/{self.stats['total_images']})")
        print(f"📁 出力先: {output_dir}")
        
        # 統計をJSONで非同期保存
        stats_file = output_dir / "async_batch_extraction_stats.json"
        try:
            stats_data = {
                **self.stats,
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'execution_time_seconds': execution_time,
                'final_output_count': final_output_count,
                'success_rate_percent': success_rate,
                'gpu_info': self.gpu_info,
                'async_processing': True,
                'max_concurrent_batches': self.max_concurrent_batches
            }
            
            async with aiofiles.open(stats_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(stats_data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"統計保存エラー: {e}")
        
        # 成功判定（60%以上の成功率）
        return success_rate >= 60.0


async def main():
    """非同期メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="非同期バッチ分割抽出実行システム")
    parser.add_argument("--tracker_id", type=str, required=True, help="トラッカーID (例: PH3-007)")
    parser.add_argument("--input_dir", type=str, 
                        default="/mnt/c/AItools/lora/train/yado/org/kana05",
                        help="入力ディレクトリ")
    parser.add_argument("--batch_size", type=int, help="バッチサイズ（自動計算を上書き）")
    parser.add_argument("--max_concurrent", type=int, default=2, help="最大同時実行バッチ数")
    parser.add_argument("--log_level", default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="ログレベル")
    
    args = parser.parse_args()
    
    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 非同期バッチ抽出システム作成
    runner = AsyncBatchedExtractionRunner(args.tracker_id)
    runner.max_concurrent_batches = args.max_concurrent
    
    # 非同期初期化
    await runner.initialize()
    
    if args.batch_size:
        runner.batch_size = args.batch_size
        logger.info(f"バッチサイズ上書き: {args.batch_size}")
    
    # 入力・出力ディレクトリ設定
    input_dir = Path(args.input_dir)
    output_dir = runner.workspace_dir / "extraction"
    
    try:
        # 非同期実行
        success = await runner.run_async_batched_extraction(input_dir, output_dir)
        
        # 終了コード
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("非同期処理が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"非同期処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n非同期処理が中断されました")
        sys.exit(1)