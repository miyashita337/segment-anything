#!/usr/bin/env python3
"""
Phase 2 Stable Background Batch Processor
安定したバックグラウンドバッチ処理システム（nohup対応）
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

import json
import logging
import signal
import time
import traceback
from datetime import datetime

# 品質保護システム
from features.extraction.quality_guard_system import QualityGuardSystem

# 既存のロバスト抽出システム
from features.extraction.robust_extractor import RobustCharacterExtractor
from typing import Any, Dict, List, Optional

# ロギング設定（ファイル出力）
log_file = f"phase2_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class Phase2StableBatchProcessor:
    """Phase 2安定バッチ処理システム"""

    def __init__(self):
        """システム初期化"""
        self.start_time = time.time()
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.protected_count = 0

        # 処理状態管理
        self.processing_stats = {
            "total_images": 0,
            "processed_images": 0,
            "successful_extractions": 0,
            "protected_files": 0,
            "processing_times": [],
            "errors": [],
        }

        # シグナルハンドラー設定（安全終了）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # システム初期化
        try:
            self.robust_extractor = RobustCharacterExtractor()
            self.quality_guard = QualityGuardSystem()
            logger.info("✅ Phase 2安定バッチ処理システム初期化完了")
        except Exception as e:
            logger.error(f"❌ システム初期化エラー: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（安全終了）"""
        logger.info(f"🛑 終了シグナル受信: {signum}")
        self._save_current_progress()
        sys.exit(0)

    def _save_current_progress(self):
        """現在の進捗を保存"""
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "protected_count": self.protected_count,
            "processing_stats": self.processing_stats,
        }

        progress_file = "phase2_batch_progress.json"
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 進捗保存完了: {progress_file}")

    def process_single_image_stable(self, image_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        安定版単一画像処理

        Args:
            image_path: 入力画像パス
            output_path: 出力画像パス

        Returns:
            処理結果
        """
        start_time = time.time()
        filename = image_path.name

        logger.info(f"🚀 処理開始: {filename}")

        result = {
            "filename": filename,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "success": False,
            "protected": False,
            "processing_time": 0.0,
            "quality_score": 0.0,
            "method_used": "unknown",
            "error": None,
        }

        try:
            # 1. 品質保護システムチェック
            should_skip, protected_record = self.quality_guard.should_skip_processing(filename)

            if should_skip and protected_record:
                logger.info(
                    f"✅ 品質保護適用: {filename} (評価={protected_record.rating}, スコア={protected_record.quality_score:.3f})"
                )

                # 既存ファイルをコピー（保護対象）
                self._copy_protected_file(filename, output_path, protected_record)

                result.update(
                    {
                        "success": True,
                        "protected": True,
                        "quality_score": protected_record.quality_score,
                        "method_used": "quality_protection",
                        "protection_reason": f"既存の{protected_record.rating}評価を保護",
                    }
                )

                self.protected_count += 1
                return result

            # 2. ロバスト抽出実行（安定版）
            extraction_result = self.robust_extractor.extract_character_robust(
                image_path, output_path, verbose=False
            )

            # 結果更新
            result.update(
                {
                    "success": extraction_result.get("success", False),
                    "quality_score": extraction_result.get("quality_score", 0.0),
                    "method_used": extraction_result.get("best_method", "robust_extraction"),
                }
            )

            if result["success"]:
                self.success_count += 1
                logger.info(f"✅ 処理成功: {filename} (品質={result['quality_score']:.3f})")
            else:
                logger.warning(f"⚠️ 処理失敗: {filename}")
                self.error_count += 1

        except Exception as e:
            error_msg = str(e)
            logger.error(f"💥 処理エラー: {filename} - {error_msg}")
            logger.error(f"スタックトレース: {traceback.format_exc()}")

            result["error"] = error_msg
            result["success"] = False
            self.error_count += 1

            # エラー統計に追加
            self.processing_stats["errors"].append(
                {"filename": filename, "error": error_msg, "timestamp": datetime.now().isoformat()}
            )

        finally:
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            self.processing_stats["processing_times"].append(processing_time)
            self.processed_count += 1

        return result

    def _copy_protected_file(self, filename: str, output_path: Path, protected_record) -> bool:
        """保護対象ファイルのコピー"""
        try:
            # 既存の高品質結果からコピー
            source_dirs = [
                "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_robust_system_final",
                "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_boundary_enhanced_full",
            ]

            for source_dir in source_dirs:
                source_path = Path(source_dir) / filename
                if source_path.exists():
                    import shutil

                    shutil.copy2(source_path, output_path)
                    logger.info(f"📋 保護ファイルコピー: {source_path} → {output_path}")
                    return True

            logger.warning(f"⚠️ 保護対象ファイルが見つかりません: {filename}")
            return False

        except Exception as e:
            logger.error(f"❌ ファイルコピーエラー: {filename} - {e}")
            return False

    def run_stable_batch(self):
        """安定バッチ処理実行"""
        logger.info("🎬 Phase 2安定バッチ処理開始")
        logger.info(f"📄 ログファイル: {log_file}")

        # 入力・出力ディレクトリ
        input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        output_dir = Path(
            "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_improvement_phase2_final"
        )

        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 出力ディレクトリ: {output_dir}")

        # 入力画像ファイル取得
        image_files = sorted(list(input_dir.glob("*.jpg")))

        if not image_files:
            logger.error(f"❌ 入力画像が見つかりません: {input_dir}")
            return []

        self.processing_stats["total_images"] = len(image_files)
        logger.info(f"📁 処理対象: {len(image_files)}画像")

        # バッチ処理実行
        results = []

        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📸 [{i}/{len(image_files)}] {image_path.name}")

            output_path = output_dir / image_path.name

            # 単一画像処理
            result = self.process_single_image_stable(image_path, output_path)
            results.append(result)

            # 進捗表示
            progress = (i / len(image_files)) * 100
            elapsed_time = time.time() - self.start_time
            avg_time = elapsed_time / i
            eta = avg_time * (len(image_files) - i)

            logger.info(f"📊 進捗: {progress:.1f}% ({i}/{len(image_files)})")
            logger.info(f"⏱️ 経過時間: {elapsed_time/60:.1f}分, 予想残り時間: {eta/60:.1f}分")
            logger.info(
                f"📈 成功: {self.success_count}, 保護: {self.protected_count}, エラー: {self.error_count}"
            )

            # 定期的に進捗保存
            if i % 5 == 0:
                self._save_current_progress()

        # 最終処理
        self._finalize_batch_processing(results, output_dir)

        return results

    def _finalize_batch_processing(self, results: List[Dict[str, Any]], output_dir: Path):
        """バッチ処理完了処理"""
        end_time = time.time()
        total_time = end_time - self.start_time

        # 統計計算
        successful_count = len([r for r in results if r["success"]])
        protected_count = len([r for r in results if r["protected"]])
        error_count = len([r for r in results if r.get("error")])

        success_rate = successful_count / len(results) * 100 if results else 0

        avg_processing_time = (
            np.mean(self.processing_stats["processing_times"])
            if self.processing_stats["processing_times"]
            else 0
        )
        avg_quality = (
            np.mean([r["quality_score"] for r in results if r["success"]])
            if successful_count > 0
            else 0
        )

        # 最終サマリー
        summary = {
            "batch_info": {
                "total_images": len(results),
                "successful_extractions": successful_count,
                "protected_files": protected_count,
                "error_count": error_count,
                "success_rate": success_rate,
                "total_processing_time": total_time,
                "average_processing_time": avg_processing_time,
                "average_quality_score": avg_quality,
            },
            "phase2_improvements": {
                "quality_protection_applied": protected_count > 0,
                "protection_rate": protected_count / len(results) * 100 if results else 0,
                "error_recovery": len(self.processing_stats["errors"]),
            },
            "detailed_results": results,
            "log_file": log_file,
        }

        # 結果保存
        results_path = output_dir / "phase2_batch_results_final.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # 最終ログ出力
        logger.info("🎉 Phase 2安定バッチ処理完了")
        logger.info("=" * 60)
        logger.info(f"📊 最終結果サマリー:")
        logger.info(f"   総画像数: {len(results)}")
        logger.info(f"   成功数: {successful_count} ({success_rate:.1f}%)")
        logger.info(f"   品質保護: {protected_count} ({protected_count/len(results)*100:.1f}%)")
        logger.info(f"   エラー数: {error_count}")
        logger.info(f"   平均品質スコア: {avg_quality:.3f}")
        logger.info(f"   平均処理時間: {avg_processing_time:.1f}秒/枚")
        logger.info(f"   総処理時間: {total_time/60:.1f}分")
        logger.info(f"💾 詳細結果: {results_path}")
        logger.info(f"📄 ログファイル: {log_file}")
        logger.info("=" * 60)

        # 完了通知メッセージ作成
        notification_message = f"""🎉 Phase 2バッチ処理完了！

📊 処理結果:
• 成功率: {success_rate:.1f}% ({successful_count}/{len(results)})
• 品質保護: {protected_count}件
• 平均品質: {avg_quality:.3f}
• 処理時間: {total_time/60:.1f}分

📂 出力: {output_dir.name}
📄 ログ: {log_file}"""

        logger.info("📱 処理完了 - 通知準備完了")

        # 通知メッセージをファイルに保存
        with open("phase2_completion_message.txt", "w", encoding="utf-8") as f:
            f.write(notification_message)


def main():
    """メイン実行"""
    try:
        logger.info("Phase 2安定バッチ処理開始")
        logger.info(f"プロセスID: {os.getpid()}")
        logger.info(f"ログファイル: {log_file}")

        processor = Phase2StableBatchProcessor()
        results = processor.run_stable_batch()

        success_count = len([r for r in results if r["success"]])
        logger.info(f"🎯 最終結果: {success_count}枚の成功処理")

        return success_count

    except KeyboardInterrupt:
        logger.info("🛑 ユーザーによる処理中断")
        return 0

    except Exception as e:
        logger.error(f"💥 バッチ処理致命的エラー: {e}")
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        return 0


if __name__ == "__main__":
    success_count = main()
    print(f"\n🎯 Phase 2バッチ処理完了: {success_count}枚成功")
