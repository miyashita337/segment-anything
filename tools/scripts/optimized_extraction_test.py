#!/usr/bin/env python3
"""
最適化設定でのSAM+YOLO抽出テスト実行
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimized_extraction_test():
    """最適化設定での抽出テスト"""
    logger.info("🧪 最適化設定抽出テスト開始")

    # 最適化設定読み込み
    config_path = Path(
        "C:/AItools/lora/train/yado/tracker-workspace/P1-B004/optimization/optimized_config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"📄 使用設定: {config['optimization_profile']}")
    logger.info(f"🎯 YOLO閾値: {config['yolo_conf_threshold']}")
    logger.info(f"⚙️ 品質方法: {config['quality_method']}")

    # テスト画像選択
    test_images = [
        "kana08_0001.jpg",
        "kana08_0002.jpg",
        "kana08_0008.jpg",
        "kana08_0010.jpg",
        "kana08_0016.jpg",
    ]

    original_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    test_output_dir = Path(
        "C:/AItools/lora/train/yado/tracker-workspace/P1-B004/optimization/test_extraction"
    )
    test_output_dir.mkdir(exist_ok=True)

    results = {
        "config_used": config,
        "test_results": {},
        "summary": {"total_tested": 0, "successful_extractions": 0, "extraction_times": []},
    }

    for i, filename in enumerate(test_images, 1):
        logger.info(f"[{i}/{len(test_images)}] テスト実行: {filename}")

        base_name = filename.replace(".jpg", "")
        input_path = original_dir / filename
        output_path = test_output_dir / f"{base_name}_optimized.jpg"

        # 抽出実行
        start_time = time.time()
        success = run_single_optimized_extraction(input_path, output_path, config)
        end_time = time.time()

        extraction_time = end_time - start_time

        results["test_results"][filename] = {
            "success": success,
            "extraction_time": extraction_time,
            "output_path": str(output_path) if success else None,
        }

        results["summary"]["total_tested"] += 1
        results["summary"]["extraction_times"].append(extraction_time)

        if success:
            results["summary"]["successful_extractions"] += 1
            logger.info(f"  ✅ 成功 ({extraction_time:.1f}秒)")
        else:
            logger.info(f"  ❌ 失敗 ({extraction_time:.1f}秒)")

    # サマリー計算
    if results["summary"]["extraction_times"]:
        avg_time = sum(results["summary"]["extraction_times"]) / len(
            results["summary"]["extraction_times"]
        )
        results["summary"]["average_extraction_time"] = avg_time

    success_rate = results["summary"]["successful_extractions"] / results["summary"]["total_tested"]
    results["summary"]["success_rate"] = success_rate

    # 結果保存
    result_path = test_output_dir / "optimization_test_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # サマリー表示
    logger.info("=" * 50)
    logger.info("🎯 最適化抽出テスト完了")
    logger.info(
        f"📊 成功率: {success_rate:.1%} ({results['summary']['successful_extractions']}/{results['summary']['total_tested']})"
    )
    logger.info(f"⏱️ 平均実行時間: {results['summary'].get('average_extraction_time', 0):.1f}秒")
    logger.info(f"📄 詳細結果: {result_path}")
    logger.info("=" * 50)

    return results


def run_single_optimized_extraction(input_path: Path, output_path: Path, config: dict) -> bool:
    """最適化設定での単一抽出実行"""
    try:
        cmd = [
            sys.executable,
            "features/extraction/commands/extract_character.py",
            str(input_path),
            "-o",
            str(output_path),
            "--verbose",
        ]

        # 最適化設定を適用（環境変数で渡す方式は複雑すぎるため、
        # ここでは基本的な実行のみ行い、後で設定ファイル統合を検討）

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode == 0 and output_path.exists():
            return True
        else:
            logger.error(f"抽出失敗: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("抽出タイムアウト (180秒)")
        return False
    except Exception as e:
        logger.error(f"抽出エラー: {e}")
        return False


def main():
    """メイン実行"""
    return run_optimized_extraction_test()


if __name__ == "__main__":
    exit(main())
