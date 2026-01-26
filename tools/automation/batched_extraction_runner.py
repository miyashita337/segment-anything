#!/usr/bin/env python3
"""
バッチ分割抽出実行システム
タイムアウト問題根本解決のためのバッチ処理実装
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.api_config import get_api_config

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """GPU使用状況監視クラス"""

    @staticmethod
    def check_gpu_available() -> Tuple[bool, Dict[str, Any]]:
        """GPU利用可能性とステータスをチェック"""
        try:
            import torch

            if not torch.cuda.is_available():
                return False, {"error": "CUDA not available"}

            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()

            # GPU情報取得
            gpu_info = {
                "device_count": device_count,
                "current_device": current_device,
                "device_name": torch.cuda.get_device_name(current_device),
                "memory_allocated": torch.cuda.memory_allocated(current_device),
                "memory_reserved": torch.cuda.memory_reserved(current_device),
                "memory_total": torch.cuda.get_device_properties(current_device).total_memory,
            }

            # メモリ使用率計算
            memory_usage = gpu_info["memory_reserved"] / gpu_info["memory_total"]
            gpu_info["memory_usage_percent"] = memory_usage * 100

            return True, gpu_info

        except ImportError:
            return False, {"error": "PyTorch not available"}
        except Exception as e:
            return False, {"error": str(e)}

    @staticmethod
    def cleanup_gpu_memory():
        """GPU メモリクリーンアップ"""
        try:
            import torch

            import gc

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("GPU メモリクリーンアップ完了")
                return True
        except Exception as e:
            logger.warning(f"GPU メモリクリーンアップ失敗: {e}")
        return False


class ImageBatch:
    """画像バッチ管理クラス"""

    def __init__(self, batch_id: int, image_paths: List[Path], estimated_time: float = 120):
        self.batch_id = batch_id
        self.image_paths = image_paths
        self.estimated_time = estimated_time
        self.actual_time = None
        self.status = "pending"  # pending, running, completed, failed
        self.results = []
        self.errors = []

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return f"Batch-{self.batch_id} ({len(self.image_paths)} images)"


class BatchedExtractionRunner:
    """バッチ分割抽出実行システム"""

    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace_dir = self.workspace_base / tracker_id

        # API設定取得
        self.api_config = get_api_config()

        # GPU監視
        gpu_available, self.gpu_info = GPUMonitor.check_gpu_available()
        if not gpu_available:
            logger.warning(f"GPU利用不可: {self.gpu_info}")

        # バッチ設定
        self.batch_size = self._calculate_optimal_batch_size()
        self.max_batch_timeout = 180  # 3分/バッチ
        self.max_image_timeout = 30  # 30秒/画像

        # 統計
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_batches": 0,
            "completed_batches": 0,
            "failed_batches": 0,
            "start_time": None,
            "end_time": None,
        }

        logger.info(f"バッチ抽出システム初期化: {tracker_id}")
        logger.info(f"バッチサイズ: {self.batch_size}")

    def _calculate_optimal_batch_size(self) -> int:
        """最適なバッチサイズを計算"""
        if not self.gpu_info.get("memory_total"):
            return 4  # デフォルト

        # GPU メモリに基づいてバッチサイズを動的計算
        memory_gb = self.gpu_info["memory_total"] / (1024**3)

        if memory_gb >= 12:
            return 8  # 12GB以上: 8枚バッチ
        elif memory_gb >= 8:
            return 6  # 8GB以上: 6枚バッチ
        elif memory_gb >= 6:
            return 4  # 6GB以上: 4枚バッチ
        else:
            return 2  # 6GB未満: 2枚バッチ

    def _create_batches(self, image_paths: List[Path]) -> List[ImageBatch]:
        """画像リストをバッチに分割"""
        batches = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_images = image_paths[i : i + self.batch_size]

            # バッチ処理時間推定（画像数 × 8秒 + オーバーヘッド20秒）
            estimated_time = len(batch_images) * 8 + 20

            batch = ImageBatch(
                batch_id=len(batches) + 1, image_paths=batch_images, estimated_time=estimated_time
            )
            batches.append(batch)

        logger.info(f"バッチ作成完了: {len(batches)}バッチ ({len(image_paths)}枚)")
        return batches

    def _process_single_batch(self, batch: ImageBatch, input_dir: Path, output_dir: Path) -> bool:
        """単一バッチを処理"""
        batch.status = "running"
        start_time = time.time()

        logger.info(f"{batch} 処理開始 (推定時間: {batch.estimated_time}秒)")

        try:
            # バッチ用の一時ディレクトリ作成
            batch_temp_dir = output_dir / f"batch_{batch.batch_id}_temp"
            batch_temp_dir.mkdir(exist_ok=True)

            # バッチ用入力ディレクトリ作成（シンボリックリンク）
            batch_input_dir = batch_temp_dir / "input"
            batch_input_dir.mkdir(exist_ok=True)

            # バッチの画像をシンボリックリンク
            for img_path in batch.image_paths:
                link_path = batch_input_dir / img_path.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(img_path)
                    except OSError:
                        # シンボリックリンク失敗時はコピー
                        import shutil

                        shutil.copy2(img_path, link_path)

            # 抽出コマンド実行
            command = [
                "python3",
                "tools/core/sam_yolo_character_segment.py",
                "--mode",
                "reproduce-auto",
                "--input_dir",
                str(batch_input_dir),
                "--output_dir",
                str(batch_temp_dir / "output"),
                "--score_threshold",
                "0.07",
            ]

            # タイムアウト計算（推定時間 + 余裕50%）
            timeout = int(batch.estimated_time * 1.5)
            timeout = max(timeout, self.max_batch_timeout)  # 最低3分保証

            logger.info(f"{batch} コマンド実行: タイムアウト{timeout}秒")

            # GPU メモリクリーンアップ
            GPUMonitor.cleanup_gpu_memory()

            # コマンド実行
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(os.environ, PYTHONPATH=str(self.project_root)),
            )

            batch.actual_time = time.time() - start_time

            if result.returncode == 0:
                # 成功: 結果をメイン出力ディレクトリに移動
                batch_output_dir = batch_temp_dir / "output"
                if batch_output_dir.exists():
                    import shutil

                    for file_path in batch_output_dir.glob("*"):
                        shutil.move(str(file_path), str(output_dir / file_path.name))

                # 処理済み画像数カウント
                processed_count = len(list(batch_output_dir.glob("*.jpg"))) + len(
                    list(batch_output_dir.glob("*.png"))
                )
                batch.results = [f"processed_{i}" for i in range(processed_count)]

                batch.status = "completed"
                logger.info(f"{batch} 完了: {processed_count}枚処理 ({batch.actual_time:.1f}秒)")

                # 一時ディレクトリクリーンアップ
                shutil.rmtree(batch_temp_dir, ignore_errors=True)

                return True
            else:
                # 失敗
                batch.status = "failed"
                batch.errors.append(f"Return code: {result.returncode}")
                batch.errors.append(f"Stderr: {result.stderr}")

                logger.error(f"{batch} 失敗: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            batch.status = "failed"
            batch.actual_time = time.time() - start_time
            batch.errors.append(f"タイムアウト ({timeout}秒)")

            logger.error(f"{batch} タイムアウト: {timeout}秒")
            return False

        except Exception as e:
            batch.status = "failed"
            batch.actual_time = time.time() - start_time
            batch.errors.append(f"処理エラー: {str(e)}")

            logger.error(f"{batch} エラー: {e}")
            return False

    def _process_individual_images(
        self, failed_batch: ImageBatch, input_dir: Path, output_dir: Path
    ) -> int:
        """失敗したバッチの個別画像処理"""
        logger.info(f"{failed_batch} 個別画像処理開始")

        success_count = 0
        individual_temp_dir = output_dir / f"individual_{failed_batch.batch_id}_temp"
        individual_temp_dir.mkdir(exist_ok=True)

        for i, img_path in enumerate(failed_batch.image_paths):
            try:
                logger.info(f"個別処理: {img_path.name} ({i+1}/{len(failed_batch.image_paths)})")

                # 1枚用の一時ディレクトリ
                single_input_dir = individual_temp_dir / f"input_{i}"
                single_output_dir = individual_temp_dir / f"output_{i}"
                single_input_dir.mkdir(exist_ok=True)
                single_output_dir.mkdir(exist_ok=True)

                # 画像コピー
                import shutil

                shutil.copy2(img_path, single_input_dir / img_path.name)

                # 個別実行
                command = [
                    "python3",
                    "tools/core/sam_yolo_character_segment.py",
                    "--mode",
                    "reproduce-auto",
                    "--input_dir",
                    str(single_input_dir),
                    "--output_dir",
                    str(single_output_dir),
                    "--score_threshold",
                    "0.07",
                ]

                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=self.max_image_timeout,
                    env=dict(os.environ, PYTHONPATH=str(self.project_root)),
                )

                if result.returncode == 0:
                    # 成功: 結果を移動
                    for file_path in single_output_dir.glob("*"):
                        shutil.move(str(file_path), str(output_dir / file_path.name))
                    success_count += 1
                    logger.info(f"個別処理成功: {img_path.name}")
                else:
                    logger.warning(f"個別処理失敗: {img_path.name} - {result.stderr}")

                # 個別一時ディレクトリクリーンアップ
                shutil.rmtree(single_input_dir, ignore_errors=True)
                shutil.rmtree(single_output_dir, ignore_errors=True)

            except subprocess.TimeoutExpired:
                logger.warning(f"個別処理タイムアウト: {img_path.name}")
            except Exception as e:
                logger.warning(f"個別処理エラー: {img_path.name} - {e}")

        # 全体一時ディレクトリクリーンアップ
        import shutil

        shutil.rmtree(individual_temp_dir, ignore_errors=True)

        logger.info(f"{failed_batch} 個別処理完了: {success_count}/{len(failed_batch.image_paths)} 成功")
        return success_count

    def run_batched_extraction(
        self, input_dir: Path, output_dir: Path, max_workers: int = 2
    ) -> bool:
        """バッチ分割抽出実行"""
        self.stats["start_time"] = datetime.now()

        # 入力ディレクトリ確認
        if not input_dir.exists():
            logger.error(f"入力ディレクトリが存在しません: {input_dir}")
            return False

        # 画像ファイルリスト取得
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        image_paths = []
        for extension in image_extensions:
            image_paths.extend(input_dir.glob(extension))

        if not image_paths:
            logger.error(f"処理可能な画像が見つかりません: {input_dir}")
            return False

        self.stats["total_images"] = len(image_paths)
        logger.info(f"処理対象画像: {len(image_paths)}枚")

        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)

        # バッチ作成
        batches = self._create_batches(image_paths)
        self.stats["total_batches"] = len(batches)

        # バッチ処理実行（並列処理）
        failed_batches = []

        print(f"\n🚀 バッチ抽出処理開始")
        print(f"📊 {len(batches)}バッチ ({len(image_paths)}枚) を {max_workers}並列で処理")
        print(f"⚙️  バッチサイズ: {self.batch_size}枚/バッチ")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # バッチを並列実行
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, input_dir, output_dir): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    success = future.result()
                    if success:
                        self.stats["completed_batches"] += 1
                        self.stats["processed_images"] += len(batch)
                    else:
                        self.stats["failed_batches"] += 1
                        failed_batches.append(batch)

                    # 進捗表示
                    completed = self.stats["completed_batches"] + self.stats["failed_batches"]
                    progress = (completed / len(batches)) * 100
                    print(f"   📊 進捗: {progress:.1f}% ({completed}/{len(batches)}バッチ)")

                except Exception as e:
                    logger.error(f"バッチ処理例外: {batch} - {e}")
                    failed_batches.append(batch)
                    self.stats["failed_batches"] += 1

        # 失敗バッチの個別処理
        if failed_batches:
            print(f"\n🔄 失敗バッチの個別処理開始: {len(failed_batches)}バッチ")

            for failed_batch in failed_batches:
                rescued_count = self._process_individual_images(failed_batch, input_dir, output_dir)
                self.stats["processed_images"] += rescued_count
                self.stats["failed_images"] += len(failed_batch) - rescued_count

        # 結果集計
        self.stats["end_time"] = datetime.now()
        execution_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # 最終結果確認
        final_output_count = len(list(output_dir.glob("*.jpg"))) + len(
            list(output_dir.glob("*.png"))
        )
        success_rate = (final_output_count / self.stats["total_images"]) * 100

        # 結果表示
        print(f"\n🏁 バッチ抽出処理完了")
        print(f"⏱️  実行時間: {execution_time:.1f}秒")
        print(f"📊 処理成功率: {success_rate:.1f}% ({final_output_count}/{self.stats['total_images']})")
        print(f"📁 出力先: {output_dir}")

        # 詳細統計
        logger.info("バッチ処理統計:")
        logger.info(f"  総画像数: {self.stats['total_images']}")
        logger.info(f"  総バッチ数: {self.stats['total_batches']}")
        logger.info(f"  成功バッチ: {self.stats['completed_batches']}")
        logger.info(f"  失敗バッチ: {self.stats['failed_batches']}")
        logger.info(f"  最終出力: {final_output_count}枚")
        logger.info(f"  成功率: {success_rate:.1f}%")

        # 統計をJSONで保存
        stats_file = output_dir / "batch_extraction_stats.json"
        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        **self.stats,
                        "start_time": self.stats["start_time"].isoformat(),
                        "end_time": self.stats["end_time"].isoformat(),
                        "execution_time_seconds": execution_time,
                        "final_output_count": final_output_count,
                        "success_rate_percent": success_rate,
                        "gpu_info": self.gpu_info,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning(f"統計保存エラー: {e}")

        # 成功判定（60%以上の成功率）
        return success_rate >= 60.0


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="バッチ分割抽出実行システム")
    parser.add_argument("tracker_id", help="トラッカーID (例: PH3-007)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana05",
        help="入力ディレクトリ",
    )
    parser.add_argument("--batch-size", type=int, help="バッチサイズ（自動計算を上書き）")
    parser.add_argument("--max-workers", type=int, default=2, help="並列実行数")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="ログレベル"
    )

    args = parser.parse_args()

    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # バッチ抽出システム作成
    runner = BatchedExtractionRunner(args.tracker_id)

    if args.batch_size:
        runner.batch_size = args.batch_size
        logger.info(f"バッチサイズ上書き: {args.batch_size}")

    # 入力・出力ディレクトリ設定
    input_dir = Path(args.input_dir)
    output_dir = runner.workspace_dir / "extraction"

    # 実行
    success = runner.run_batched_extraction(input_dir, output_dir, args.max_workers)

    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
