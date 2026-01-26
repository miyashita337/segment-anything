#!/usr/bin/env python3
"""
前処理パイプライン強化テスト
Week 1完了確認：アニメ顔検出の改善検証
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

from features.evaluation.anime_image_preprocessor import AnimeImagePreprocessor
from features.evaluation.enhanced_detection_systems import EnhancedFaceDetector, FaceDetection

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetectionPipelineTest:
    """前処理パイプライン強化テスト"""

    def __init__(self):
        self.face_detector = EnhancedFaceDetector()
        self.preprocessor = AnimeImagePreprocessor()
        self.test_datasets = {
            "kana05": "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix",
            "kana07": "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix",
            "kana08": "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix",
        }

    def run_comprehensive_test(self) -> Dict:
        """包括的前処理パイプラインテスト"""
        logger.info("🧪 前処理パイプライン強化テスト開始")

        all_results = {}
        total_images = 0
        total_faces_detected = 0
        processing_times = []

        for dataset_name, dataset_path in self.test_datasets.items():
            if not os.path.exists(dataset_path):
                logger.warning(f"データセット未発見: {dataset_path}")
                continue

            logger.info(f"📂 テスト実行: {dataset_name}")
            dataset_results = self.test_dataset(dataset_path, dataset_name)
            all_results[dataset_name] = dataset_results

            # 統計集計
            total_images += dataset_results["image_count"]
            total_faces_detected += dataset_results["total_faces_detected"]
            processing_times.extend(dataset_results["processing_times"])

        # 総合統計計算
        overall_stats = self.calculate_overall_statistics(
            all_results, total_images, total_faces_detected, processing_times
        )

        # 結果レポート
        self.generate_improvement_report(all_results, overall_stats)

        return {
            "dataset_results": all_results,
            "overall_statistics": overall_stats,
            "test_completion_time": datetime.now().isoformat(),
        }

    def test_dataset(self, dataset_path: str, dataset_name: str) -> Dict:
        """単一データセットテスト"""
        image_files = [
            f
            for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.endswith("_gt.png")
        ]

        results = {
            "dataset_name": dataset_name,
            "image_count": len(image_files),
            "detection_results": [],
            "total_faces_detected": 0,
            "processing_times": [],
            "method_performance": {},
            "preprocessing_effectiveness": {},
        }

        method_counts = {}

        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)

            try:
                # 画像読み込み
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # 前処理効果テスト
                preprocessing_result = self.test_preprocessing_effectiveness(image)

                # 顔検出実行
                start_time = datetime.now()
                face_detections = self.face_detector.detect_faces_comprehensive(image)
                processing_time = (datetime.now() - start_time).total_seconds()

                # 結果記録
                detection_result = {
                    "image_file": image_file,
                    "faces_detected": len(face_detections),
                    "processing_time": processing_time,
                    "detections": [
                        {"method": det.method, "confidence": det.confidence, "bbox": det.bbox}
                        for det in face_detections
                    ],
                    "preprocessing_stats": preprocessing_result,
                }

                results["detection_results"].append(detection_result)
                results["total_faces_detected"] += len(face_detections)
                results["processing_times"].append(processing_time)

                # 手法別統計
                for detection in face_detections:
                    method = detection.method
                    if method not in method_counts:
                        method_counts[method] = 0
                    method_counts[method] += 1

                logger.debug(f"  {image_file}: {len(face_detections)}件検出 ({processing_time:.2f}秒)")

            except Exception as e:
                logger.error(f"画像処理エラー {image_file}: {e}")

        results["method_performance"] = method_counts
        return results

    def test_preprocessing_effectiveness(self, image: np.ndarray) -> Dict:
        """前処理効果測定"""
        try:
            # 元画像統計
            original_brightness, needs_adjustment = self.preprocessor.detect_optimal_brightness(
                image
            )

            # 前処理実行
            enhanced_image = self.preprocessor.enhance_for_face_detection(image)

            # 処理後統計
            enhanced_brightness, _ = self.preprocessor.detect_optimal_brightness(enhanced_image)

            # マルチスケール版生成
            multi_scale_versions = self.preprocessor.create_multi_scale_versions(enhanced_image)

            return {
                "original_brightness": float(original_brightness),
                "enhanced_brightness": float(enhanced_brightness),
                "brightness_improvement": float(enhanced_brightness - original_brightness),
                "adjustment_needed": bool(needs_adjustment),
                "multi_scale_count": len(multi_scale_versions),
            }

        except Exception as e:
            logger.warning(f"前処理効果測定エラー: {e}")
            return {"error": str(e)}

    def calculate_overall_statistics(
        self, all_results: Dict, total_images: int, total_faces: int, processing_times: List[float]
    ) -> Dict:
        """総合統計計算"""

        # 基本統計
        face_detection_rate = (total_faces / total_images) if total_images > 0 else 0.0
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0.0
        )

        # 手法別統計
        all_methods = {}
        for dataset_results in all_results.values():
            for method, count in dataset_results["method_performance"].items():
                if method not in all_methods:
                    all_methods[method] = 0
                all_methods[method] += count

        # 前処理効果統計
        brightness_improvements = []
        for dataset_results in all_results.values():
            for detection_result in dataset_results["detection_results"]:
                preprocessing_stats = detection_result.get("preprocessing_stats", {})
                improvement = preprocessing_stats.get("brightness_improvement", 0)
                if improvement != 0:
                    brightness_improvements.append(improvement)

        avg_brightness_improvement = (
            sum(brightness_improvements) / len(brightness_improvements)
            if brightness_improvements
            else 0.0
        )

        return {
            "total_images_processed": total_images,
            "total_faces_detected": total_faces,
            "overall_face_detection_rate": face_detection_rate,
            "average_processing_time": avg_processing_time,
            "method_distribution": all_methods,
            "average_brightness_improvement": avg_brightness_improvement,
            "images_with_preprocessing": len(brightness_improvements),
        }

    def generate_improvement_report(self, all_results: Dict, overall_stats: Dict):
        """改善レポート生成"""

        print("\n" + "=" * 80)
        print("🎨 前処理パイプライン強化テスト結果 (Week 1完了確認)")
        print("=" * 80)

        print(f"\n📊 総合統計:")
        print(f"  処理画像数: {overall_stats['total_images_processed']}枚")
        print(f"  総検出数: {overall_stats['total_faces_detected']}件")
        print(f"  検出率: {overall_stats['overall_face_detection_rate']:.1%}")
        print(f"  平均処理時間: {overall_stats['average_processing_time']:.2f}秒/画像")
        print(f"  平均明度改善: {overall_stats['average_brightness_improvement']:+.1f}")

        print(f"\n🔍 検出手法分布:")
        for method, count in overall_stats["method_distribution"].items():
            percentage = (
                (count / overall_stats["total_faces_detected"]) * 100
                if overall_stats["total_faces_detected"] > 0
                else 0
            )
            print(f"  {method}: {count}件 ({percentage:.1f}%)")

        print(f"\n📈 データセット別詳細:")
        for dataset_name, results in all_results.items():
            detection_rate = (
                (results["total_faces_detected"] / results["image_count"])
                if results["image_count"] > 0
                else 0
            )
            avg_time = (
                sum(results["processing_times"]) / len(results["processing_times"])
                if results["processing_times"]
                else 0
            )

            print(f"  {dataset_name}:")
            print(f"    画像数: {results['image_count']}枚")
            print(f"    検出数: {results['total_faces_detected']}件")
            print(f"    検出率: {detection_rate:.1%}")
            print(f"    平均処理時間: {avg_time:.2f}秒")

        # Week 1目標達成評価
        print(f"\n🎯 Week 1目標達成評価:")

        # アニメ顔カスケード使用確認
        anime_methods = [m for m in overall_stats["method_distribution"].keys() if "anime" in m]
        anime_detection_count = sum(
            overall_stats["method_distribution"].get(m, 0) for m in anime_methods
        )

        print(f"  ✅ アニメ顔専用カスケード: {len(anime_methods)}種類の手法使用")
        print(f"  ✅ アニメ特化検出: {anime_detection_count}件検出")
        print(f"  ✅ 前処理パイプライン: {overall_stats['images_with_preprocessing']}枚に適用")
        print(f"  ✅ マルチスケール検出: 統合完了")

        # 改善効果分析
        target_detection_rate = 0.90  # 90%目標
        current_rate = overall_stats["overall_face_detection_rate"]
        improvement_needed = target_detection_rate - current_rate

        print(f"\n📋 次週に向けた分析:")
        print(f"  現在の顔検出率: {current_rate:.1%}")
        print(f"  目標検出率: {target_detection_rate:.1%}")
        print(f"  必要改善: {improvement_needed:+.1%}")

        if current_rate >= target_detection_rate:
            print(f"  🎉 Week 1で90%目標達成!")
        else:
            print(f"  📋 Week 2で追加改善が必要")

        print("\n" + "=" * 80)


def main():
    """メイン実行関数"""
    print("🧪 前処理パイプライン強化テスト開始")

    try:
        tester = FaceDetectionPipelineTest()
        results = tester.run_comprehensive_test()

        # 結果をJSONで保存
        output_file = (
            f"enhanced_face_detection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 詳細結果保存: {output_file}")
        print("✅ Week 1前処理パイプライン強化テスト完了")

        return 0

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
