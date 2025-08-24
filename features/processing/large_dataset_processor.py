#!/usr/bin/env python3
"""
P1-015: 大規模データセット並列処理システム
メモリ効率化とプログレッシブ処理によるスケーラブルな画像処理
"""

import torch

import logging
import os
import psutil
import subprocess
# プロジェクトルートをパスに追加
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.memory_optimizer import BatchMemoryManager, MemoryOptimizer
from features.common.output_path_manager import OutputCategory, OutputPathManager

logger = logging.getLogger(__name__)


class LargeDatasetConfig:
    """大規模データセット処理設定"""
    
    def __init__(self,
                 chunk_size: int = 50,
                 max_concurrent_chunks: int = 2,
                 memory_threshold_gb: float = 2.0,
                 enable_progressive_processing: bool = True,
                 enable_intermediate_cleanup: bool = True,
                 checkpoint_interval: int = 100):
        self.chunk_size = chunk_size  # チャンクあたりのファイル数
        self.max_concurrent_chunks = max_concurrent_chunks  # 同時処理チャンク数
        self.memory_threshold_gb = memory_threshold_gb  # メモリ閾値(GB)
        self.enable_progressive_processing = enable_progressive_processing
        self.enable_intermediate_cleanup = enable_intermediate_cleanup
        self.checkpoint_interval = checkpoint_interval  # チェックポイント間隔


class LargeDatasetProcessor:
    """大規模データセット処理システム"""
    
    def __init__(self, tracker_id: str, config: LargeDatasetConfig = None):
        self.tracker_id = tracker_id
        self.config = config or LargeDatasetConfig()
        
        # メモリ管理システム
        self.memory_manager = BatchMemoryManager()
        self.memory_manager.enable_large_dataset_mode(max_batch_size=self.config.chunk_size)
        
        # 処理統計
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'memory_optimizations': 0,
            'checkpoints_created': 0
        }
        
        # 出力パス管理
        self.path_manager = OutputPathManager(tracker_id)
        
        # SAM+YOLOスクリプトパス
        self.sam_yolo_script = Path(__file__).parent.parent.parent / "tools/core/sam_yolo_character_segment.py"
        
        logger.info(f"🚀 P1-015 大規模データセット処理システム初期化: {tracker_id}")
    
    def process_large_dataset(self, 
                            input_dir: str, 
                            processing_params: Optional[Dict[str, Any]] = None) -> bool:
        """大規模データセット処理メイン関数"""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"❌ 入力ディレクトリが存在しません: {input_path}")
            return False
        
        # 出力ディレクトリ準備
        output_dir = self.path_manager.ensure_output_dir(OutputCategory.EXTRACTION)
        
        # ファイル収集
        image_files = self._collect_image_files(input_path)
        if not image_files:
            logger.error(f"❌ 処理対象画像が見つかりません: {input_path}")
            return False
        
        self.processing_stats['total_files'] = len(image_files)
        logger.info(f"📊 処理対象: {len(image_files)}ファイル")
        
        # チャンク分割
        chunks = self._create_file_chunks(image_files)
        logger.info(f"🔄 {len(chunks)}個のチャンクに分割 (チャンクサイズ: {self.config.chunk_size})")
        
        # プログレッシブ処理実行
        success = self._process_chunks_progressively(chunks, output_dir, processing_params or {})
        
        # 最終統計
        self._log_final_statistics()
        
        return success
    
    def _collect_image_files(self, input_dir: Path) -> List[Path]:
        """画像ファイル収集"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _create_file_chunks(self, files: List[Path]) -> List[List[Path]]:
        """ファイルをチャンクに分割"""
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(files), chunk_size):
            chunk = files[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _process_chunks_progressively(self, 
                                    chunks: List[List[Path]], 
                                    output_dir: Path,
                                    processing_params: Dict[str, Any]) -> bool:
        """プログレッシブチャンク処理"""
        
        total_success = True
        start_time = time.time()
        
        if self.config.enable_progressive_processing:
            # プログレッシブ処理: チャンクを順次処理
            success = self._process_chunks_sequential(chunks, output_dir, processing_params)
        else:
            # 並列処理: 複数チャンクを同時処理
            success = self._process_chunks_parallel(chunks, output_dir, processing_params)
        
        self.processing_stats['total_processing_time'] = time.time() - start_time
        return success
    
    def _process_chunks_sequential(self, 
                                 chunks: List[List[Path]], 
                                 output_dir: Path,
                                 processing_params: Dict[str, Any]) -> bool:
        """逐次チャンク処理"""
        
        logger.info(f"🔄 プログレッシブ処理開始: {len(chunks)}チャンクを順次処理")
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.info(f"📦 チャンク {chunk_idx}/{len(chunks)} 処理開始 ({len(chunk)}ファイル)")
            
            # メモリ状況確認
            if self._should_optimize_memory():
                logger.info("🧹 メモリ最適化実行中...")
                self.memory_manager.optimizer.optimize_memory(force=True)
                self.processing_stats['memory_optimizations'] += 1
            
            # チャンク処理
            chunk_success = self._process_single_chunk(chunk, output_dir, processing_params)
            
            if chunk_success:
                self.processing_stats['chunks_processed'] += 1
                logger.info(f"✅ チャンク {chunk_idx} 処理完了")
            else:
                logger.error(f"❌ チャンク {chunk_idx} 処理失敗")
            
            # チェックポイント作成
            if chunk_idx % (self.config.checkpoint_interval // self.config.chunk_size) == 0:
                self._create_checkpoint(chunk_idx, len(chunks))
            
            # 中間クリーンアップ
            if self.config.enable_intermediate_cleanup:
                self._intermediate_cleanup()
        
        success_rate = (self.processing_stats['chunks_processed'] / len(chunks)) * 100
        logger.info(f"📊 チャンク処理完了: {success_rate:.1f}% 成功")
        
        return success_rate >= 70.0  # 70%以上で成功とみなす
    
    def _process_chunks_parallel(self, 
                               chunks: List[List[Path]], 
                               output_dir: Path,
                               processing_params: Dict[str, Any]) -> bool:
        """並列チャンク処理"""
        
        max_workers = min(self.config.max_concurrent_chunks, len(chunks))
        logger.info(f"⚡ 並列処理開始: {max_workers}ワーカーで{len(chunks)}チャンク処理")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # チャンク処理タスク投入
            futures = {
                executor.submit(self._process_single_chunk, chunk, output_dir, processing_params): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # 完了待ち
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    success = future.result()
                    if success:
                        self.processing_stats['chunks_processed'] += 1
                        logger.info(f"✅ チャンク {chunk_idx + 1} 処理完了")
                    else:
                        logger.error(f"❌ チャンク {chunk_idx + 1} 処理失敗")
                except Exception as e:
                    logger.error(f"❌ チャンク {chunk_idx + 1} 処理エラー: {e}")
        
        success_rate = (self.processing_stats['chunks_processed'] / len(chunks)) * 100
        logger.info(f"📊 並列処理完了: {success_rate:.1f}% 成功")
        
        return success_rate >= 70.0
    
    def _process_single_chunk(self, 
                            chunk: List[Path], 
                            output_dir: Path,
                            processing_params: Dict[str, Any]) -> bool:
        """単一チャンク処理"""
        
        try:
            # 一時チャンクディレクトリ作成
            with tempfile.TemporaryDirectory() as temp_chunk_dir:
                temp_chunk_path = Path(temp_chunk_dir)
                
                # ファイルコピー
                import shutil
                for file_path in chunk:
                    dst_path = temp_chunk_path / file_path.name
                    shutil.copy2(file_path, dst_path)
                
                # SAM+YOLO処理実行
                cmd = [
                    "python3", str(self.sam_yolo_script),
                    "--mode", "reproduce-auto",
                    "--input_dir", str(temp_chunk_path),
                    "--output_dir", str(output_dir),
                    "--score_threshold", str(processing_params.get('score_threshold', 0.07))
                ]
                
                # チャンク固有の環境設定
                env = dict(os.environ)
                env['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0使用
                
                # 実行
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30分タイムアウト
                    env=env
                )
                
                if result.returncode == 0:
                    # 成功ファイル数カウント
                    success_count = self._count_output_files(output_dir, len(chunk))
                    self.processing_stats['processed_files'] += len(chunk)
                    self.processing_stats['successful_files'] += success_count
                    logger.debug(f"チャンク処理成功: {success_count}/{len(chunk)}")
                    return True
                else:
                    logger.error(f"チャンク処理失敗: {result.stderr[:200]}...")
                    self.processing_stats['failed_files'] += len(chunk)
                    return False
                    
        except Exception as e:
            logger.error(f"チャンク処理エラー: {e}")
            self.processing_stats['failed_files'] += len(chunk)
            return False
    
    def _count_output_files(self, output_dir: Path, expected_count: int) -> int:
        """出力ファイル数カウント"""
        output_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            output_files.extend(output_dir.glob(f"*{ext}"))
            output_files.extend(output_dir.glob(f"*{ext.upper()}"))
        return len(output_files)
    
    def _should_optimize_memory(self) -> bool:
        """メモリ最適化要否判定"""
        memory_info = psutil.virtual_memory()
        used_gb = (memory_info.total - memory_info.available) / (1024**3)
        
        return used_gb > self.config.memory_threshold_gb
    
    def _create_checkpoint(self, current_chunk: int, total_chunks: int):
        """チェックポイント作成"""
        checkpoint_info = {
            'tracker_id': self.tracker_id,
            'current_chunk': current_chunk,
            'total_chunks': total_chunks,
            'processing_stats': self.processing_stats.copy(),
            'timestamp': time.time()
        }
        
        checkpoint_file = self.path_manager.workspace.base_path / f"checkpoint_{current_chunk}.json"
        
        try:
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            self.processing_stats['checkpoints_created'] += 1
            logger.info(f"📍 チェックポイント作成: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"チェックポイント作成失敗: {e}")
    
    def _intermediate_cleanup(self):
        """中間クリーンアップ"""
        # Python ガベージコレクション
        import gc
        collected = gc.collect()
        
        # PyTorch キャッシュクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug(f"🧹 中間クリーンアップ: {collected}オブジェクト回収")
    
    def _log_final_statistics(self):
        """最終統計ログ出力"""
        stats = self.processing_stats
        
        logger.info("="*60)
        logger.info("📊 P1-015 大規模データセット処理完了統計")
        logger.info("="*60)
        logger.info(f"📁 総ファイル数: {stats['total_files']}")
        logger.info(f"✅ 成功ファイル数: {stats['successful_files']}")
        logger.info(f"❌ 失敗ファイル数: {stats['failed_files']}")
        logger.info(f"📦 処理チャンク数: {stats['chunks_processed']}")
        logger.info(f"⏱️ 総処理時間: {stats['total_processing_time']:.1f}秒")
        logger.info(f"🧹 メモリ最適化回数: {stats['memory_optimizations']}")
        logger.info(f"📍 チェックポイント数: {stats['checkpoints_created']}")
        
        success_rate = (stats['successful_files'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        logger.info(f"📈 成功率: {success_rate:.1f}%")
        logger.info("="*60)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計取得"""
        stats = self.processing_stats.copy()
        stats['memory_stats'] = self.memory_manager.get_memory_stats()
        return stats


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-015: 大規模データセット処理")
    parser.add_argument('--tracker-id', default='P1-015', help='トラッカーID')
    parser.add_argument('--input-dir', required=True, help='入力ディレクトリ')
    parser.add_argument('--chunk-size', type=int, default=50, help='チャンクサイズ')
    parser.add_argument('--memory-threshold', type=float, default=2.0, help='メモリ閾値(GB)')
    parser.add_argument('--disable-progressive', action='store_true', help='プログレッシブ処理無効化')
    
    args = parser.parse_args()
    
    # 設定作成
    config = LargeDatasetConfig(
        chunk_size=args.chunk_size,
        memory_threshold_gb=args.memory_threshold,
        enable_progressive_processing=not args.disable_progressive
    )
    
    # 処理システム初期化
    processor = LargeDatasetProcessor(args.tracker_id, config)
    
    # 処理実行
    print(f"🚀 P1-015 大規模データセット処理開始")
    print(f"📁 入力ディレクトリ: {args.input_dir}")
    print(f"📦 チャンクサイズ: {args.chunk_size}")
    print(f"💾 メモリ閾値: {args.memory_threshold}GB")
    
    start_time = time.time()
    success = processor.process_large_dataset(
        input_dir=args.input_dir,
        processing_params={'score_threshold': 0.07}
    )
    end_time = time.time()
    
    if success:
        print(f"✅ P1-015 大規模データセット処理成功")
        print(f"⏱️ 処理時間: {end_time - start_time:.1f}秒")
        return 0
    else:
        print(f"❌ P1-015 大規模データセット処理失敗")
        return 1


if __name__ == "__main__":
    exit(main())