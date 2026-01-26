#!/usr/bin/env python3
"""
P1-014: マルチGPU SAM統合システム
既存のSAM+YOLOパイプラインのマルチGPU対応統合
"""

import torch

import logging
import subprocess

# プロジェクトルートをパスに追加
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.multi_gpu_processor import (
    MultiGPUConfig,
    MultiGPUSAMProcessor,
    ProcessingTask,
)

logger = logging.getLogger(__name__)


class MultiGPUSAMIntegration:
    """マルチGPU SAM統合システム"""

    def __init__(self, tracker_id: str, config: MultiGPUConfig = None):
        """初期化"""
        self.tracker_id = tracker_id
        self.config = config or MultiGPUConfig()
        self.processor = MultiGPUSAMProcessor(tracker_id, config)

        # 既存スクリプトパス
        self.sam_yolo_script = (
            Path(__file__).parent.parent.parent / "tools/core/sam_yolo_character_segment.py"
        )

    def process_with_existing_pipeline(
        self, input_files: List[str], output_dir: str, processing_params: Dict[str, Any] = None
    ) -> bool:
        """既存パイプラインでマルチGPU処理"""

        if not self.processor.gpu_manager.available_gpus:
            logger.warning("GPUが利用できません。標準処理にフォールバック")
            return self._fallback_to_standard_processing(input_files, output_dir, processing_params)

        if len(self.processor.gpu_manager.available_gpus) == 1:
            logger.info("単一GPU環境のため、標準処理を実行")
            return self._single_gpu_processing(input_files, output_dir, processing_params)

        logger.info(f"マルチGPU並列処理開始: {len(self.processor.gpu_manager.available_gpus)}個のGPU")

        # ファイルをGPU数で分割
        gpu_count = len(self.processor.gpu_manager.available_gpus)
        file_chunks = self._split_files_for_gpus(input_files, gpu_count)

        # 各GPUで並列処理
        results = []
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            futures = []

            for gpu_idx, (gpu_info, file_chunk) in enumerate(
                zip(self.processor.gpu_manager.available_gpus, file_chunks)
            ):
                if not file_chunk:  # 空のチャンクはスキップ
                    continue

                # GPU固有の出力ディレクトリ
                gpu_output_dir = Path(output_dir) / f"gpu_{gpu_info.device_id}"
                gpu_output_dir.mkdir(parents=True, exist_ok=True)

                # 並列処理タスク投入
                future = executor.submit(
                    self._process_chunk_on_gpu,
                    file_chunk,
                    str(gpu_output_dir),
                    gpu_info.device_id,
                    processing_params or {},
                )
                futures.append((future, gpu_info.device_id, len(file_chunk)))

            # 完了待ち
            total_success = 0
            total_files = 0

            for future, gpu_id, chunk_size in futures:
                try:
                    success_count = future.result()
                    total_success += success_count
                    total_files += chunk_size
                    logger.info(f"GPU {gpu_id}: {success_count}/{chunk_size} ファイル成功")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 処理エラー: {e}")
                    total_files += chunk_size

        # 結果をメインディレクトリに統合
        self._merge_gpu_outputs(output_dir)

        success_rate = (total_success / total_files) * 100 if total_files > 0 else 0
        logger.info(f"マルチGPU処理完了: {total_success}/{total_files} ({success_rate:.1f}%)")

        return success_rate >= 50.0  # 50%以上で成功とみなす

    def _split_files_for_gpus(self, files: List[str], gpu_count: int) -> List[List[str]]:
        """ファイルをGPU数で分割"""
        chunks = [[] for _ in range(gpu_count)]

        for i, file_path in enumerate(files):
            gpu_index = i % gpu_count
            chunks[gpu_index].append(file_path)

        return chunks

    def _process_chunk_on_gpu(
        self,
        file_chunk: List[str],
        output_dir: str,
        gpu_id: int,
        processing_params: Optional[Dict[str, Any]],
    ) -> int:
        """GPU上でファイルチャンク処理"""
        logger.info(f"GPU {gpu_id}で{len(file_chunk)}ファイルの処理開始")

        try:
            # 一時入力ディレクトリ作成
            with tempfile.TemporaryDirectory() as temp_input_dir:
                temp_input_path = Path(temp_input_dir)

                # ファイルを一時ディレクトリにコピー
                import shutil

                for file_path in file_chunk:
                    src_path = Path(file_path)
                    dst_path = temp_input_path / src_path.name
                    shutil.copy2(src_path, dst_path)

                # 既存のSAM+YOLOスクリプトをGPU指定で実行
                cmd = [
                    "python3",
                    str(self.sam_yolo_script),
                    "--mode",
                    "reproduce-auto",
                    "--input_dir",
                    str(temp_input_path),
                    "--output_dir",
                    output_dir,
                    "--device",
                    f"cuda:{gpu_id}",  # GPU指定
                    "--score_threshold",
                    str((processing_params or {}).get("score_threshold", 0.07)),
                ]

                # 追加パラメータ
                if processing_params and processing_params.get("yolo_model"):
                    cmd.extend(["--yolo_model", processing_params["yolo_model"]])

                logger.debug(f"GPU {gpu_id} 実行コマンド: {' '.join(cmd)}")

                # プロセス実行
                import os

                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # GPU可視性制限

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1800, env=env  # 30分タイムアウト
                )

                if result.returncode == 0:
                    # 出力ファイル数カウント
                    output_path = Path(output_dir)
                    output_files = []
                    for ext in [".jpg", ".jpeg", ".png"]:
                        output_files.extend(output_path.glob(f"*{ext}"))
                        output_files.extend(output_path.glob(f"*{ext.upper()}"))

                    success_count = len(output_files)
                    logger.info(f"GPU {gpu_id} 成功: {success_count}ファイル")
                    return success_count
                else:
                    logger.error(f"GPU {gpu_id} 処理失敗: {result.stderr}")
                    return 0

        except Exception as e:
            logger.error(f"GPU {gpu_id} チャンク処理エラー: {e}")
            return 0

    def _merge_gpu_outputs(self, output_dir: str):
        """GPU別出力をメインディレクトリに統合"""
        output_path = Path(output_dir)

        # GPU別ディレクトリを検索
        gpu_dirs = list(output_path.glob("gpu_*"))

        if not gpu_dirs:
            return

        logger.info(f"GPU出力統合開始: {len(gpu_dirs)}個のGPUディレクトリ")

        import shutil

        for gpu_dir in gpu_dirs:
            if not gpu_dir.is_dir():
                continue

            # GPU別ディレクトリ内のファイルをメインディレクトリに移動
            for file_path in gpu_dir.iterdir():
                if file_path.is_file():
                    dst_path = output_path / file_path.name

                    # 重複ファイル名対応
                    counter = 1
                    while dst_path.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        dst_path = output_path / f"{stem}_{counter}{suffix}"
                        counter += 1

                    shutil.move(str(file_path), str(dst_path))

            # 空のGPUディレクトリ削除
            try:
                gpu_dir.rmdir()
            except OSError:
                logger.warning(f"GPU ディレクトリ削除失敗: {gpu_dir}")

        logger.info("GPU出力統合完了")

    def _single_gpu_processing(
        self, input_files: List[str], output_dir: str, processing_params: Optional[Dict[str, Any]]
    ) -> bool:
        """単一GPU処理"""
        logger.info("単一GPU最適化処理を実行")

        return self._fallback_to_standard_processing(input_files, output_dir, processing_params)

    def _fallback_to_standard_processing(
        self, input_files: List[str], output_dir: str, processing_params: Optional[Dict[str, Any]]
    ) -> bool:
        """標準処理フォールバック"""
        logger.info("標準SAM+YOLO処理にフォールバック")

        try:
            # 一時入力ディレクトリ作成
            with tempfile.TemporaryDirectory() as temp_input_dir:
                temp_input_path = Path(temp_input_dir)

                # ファイルコピー
                import shutil

                for file_path in input_files:
                    src_path = Path(file_path)
                    dst_path = temp_input_path / src_path.name
                    shutil.copy2(src_path, dst_path)

                # 標準コマンド実行
                cmd = [
                    "python3",
                    str(self.sam_yolo_script),
                    "--mode",
                    "reproduce-auto",
                    "--input_dir",
                    str(temp_input_path),
                    "--output_dir",
                    output_dir,
                    "--score_threshold",
                    str((processing_params or {}).get("score_threshold", 0.07)),
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3600  # 1時間タイムアウト
                )

                return result.returncode == 0

        except Exception as e:
            logger.error(f"標準処理エラー: {e}")
            return False

    def get_performance_info(self) -> Dict[str, Any]:
        """パフォーマンス情報取得"""
        gpu_info = []

        for gpu in self.processor.gpu_manager.available_gpus:
            gpu_info.append(
                {
                    "device_id": gpu.device_id,
                    "device_name": gpu.device_name,
                    "total_memory_gb": gpu.total_memory_gb,
                    "available_memory_gb": gpu.available_memory_gb,
                    "is_available": gpu.is_available,
                }
            )

        return {
            "multi_gpu_enabled": self.config.use_multi_gpu,
            "available_gpus": len(self.processor.gpu_manager.available_gpus),
            "gpu_details": gpu_info,
            "load_balancing_strategy": self.config.load_balancing_strategy,
            "batch_size_per_gpu": self.config.batch_size_per_gpu,
        }


def main():
    """メイン実行"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="P1-014: マルチGPU SAM統合処理")
    parser.add_argument("--tracker-id", default="P1-014", help="トラッカーID")
    parser.add_argument("--input-dir", required=True, help="入力ディレクトリ")
    parser.add_argument("--output-dir", help="出力ディレクトリ")
    parser.add_argument("--score-threshold", type=float, default=0.07, help="YOLO閾値")
    parser.add_argument("--max-gpus", type=int, help="最大GPU使用数")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="マルチGPU無効化")

    args = parser.parse_args()

    # 入力ディレクトリ確認
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return 1

    # 入力ファイル収集
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    input_files = []
    for ext in image_extensions:
        input_files.extend(input_dir.glob(f"*{ext}"))
        input_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not input_files:
        print(f"❌ 処理対象画像が見つかりません: {input_dir}")
        return 1

    # 出力ディレクトリ設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from features.common.output_path_manager import OutputCategory, OutputPathManager

        path_manager = OutputPathManager(args.tracker_id)
        output_dir = path_manager.ensure_output_dir(OutputCategory.EXTRACTION)

    output_dir.mkdir(parents=True, exist_ok=True)

    # マルチGPU設定
    config = MultiGPUConfig(use_multi_gpu=not args.disable_multi_gpu, max_gpus=args.max_gpus)

    # 統合システム初期化
    integration = MultiGPUSAMIntegration(args.tracker_id, config)

    # パフォーマンス情報表示
    perf_info = integration.get_performance_info()
    print(f"🖥️ GPU環境情報:")
    print(f"   マルチGPU: {'有効' if perf_info['multi_gpu_enabled'] else '無効'}")
    print(f"   利用可能GPU数: {perf_info['available_gpus']}")
    for gpu in perf_info["gpu_details"]:
        print(f"   GPU {gpu['device_id']}: {gpu['device_name']} ({gpu['total_memory_gb']:.1f}GB)")

    print(f"📊 処理開始: {len(input_files)}ファイル")

    # 処理実行
    processing_params = {"score_threshold": args.score_threshold}

    start_time = time.time()
    success = integration.process_with_existing_pipeline(
        input_files=[str(f) for f in input_files],
        output_dir=str(output_dir),
        processing_params=processing_params,
    )
    end_time = time.time()

    processing_time = end_time - start_time

    if success:
        print(f"✅ P1-014 マルチGPU処理成功")
        print(f"⏱️ 処理時間: {processing_time:.1f}秒")
        print(f"📁 出力先: {output_dir}")
        return 0
    else:
        print(f"❌ P1-014 マルチGPU処理失敗")
        return 1


if __name__ == "__main__":
    exit(main())
