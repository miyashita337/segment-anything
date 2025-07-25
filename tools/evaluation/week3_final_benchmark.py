#!/usr/bin/env python3
"""
Week 3最終目標達成確認ベンチマーク
アンサンブルシステム + SCI最適化の効果測定

目標指標:
- 顔検出率: 90%以上
- ポーズ検出率: 80%以上  
- SCI総合スコア: 0.70以上
"""

import cv2
import json
import logging
import numpy as np
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.detection_ensemble_system import DetectionEnsembleSystem
from features.evaluation.objective_evaluation_system import ObjectiveEvaluationSystem

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""

    image_path: str
    face_detection_success: bool
    face_confidence: float
    pose_detection_success: bool
    pose_confidence: float
    sci_score: float
    sci_anime_score: float
    ensemble_confidence: float
    processing_time: float
    improvement_metrics: Dict[str, float]


@dataclass
class FinalBenchmarkReport:
    """最終ベンチマークレポート"""

    timestamp: datetime
    total_images: int
    face_detection_rate: float
    pose_detection_rate: float
    sci_average: float
    sci_anime_average: float
    target_achievement: Dict[str, bool]
    detailed_results: List[BenchmarkResult]
    performance_summary: Dict[str, float]
    ensemble_effectiveness: Dict[str, float]


class Week3FinalBenchmark:
    """Week 3最終ベンチマーク実行システム"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Week3FinalBenchmark")

        # システム初期化
        self.ensemble_system = DetectionEnsembleSystem()
        self.evaluation_system = ObjectiveEvaluationSystem()

        # 目標値設定
        self.targets = {
            "face_detection_rate": 0.90,  # 90%
            "pose_detection_rate": 0.80,  # 80%
            "sci_score": 0.70,  # 0.70
        }

        self.logger.info("Week 3最終ベンチマークシステム初期化完了")

    def run_final_benchmark(self, test_images_dir: str = "test_small") -> FinalBenchmarkReport:
        """最終ベンチマーク実行"""
        start_time = datetime.now()
        self.logger.info("Week 3最終ベンチマーク開始")

        # テスト画像の収集
        test_images = self._collect_test_images(test_images_dir)
        if not test_images:
            raise ValueError(f"テスト画像が見つかりません: {test_images_dir}")

        self.logger.info(f"テスト画像数: {len(test_images)}枚")

        # 各画像でのベンチマーク実行
        detailed_results = []
        total_processing_time = 0.0

        for i, image_path in enumerate(test_images, 1):
            self.logger.info(f"処理中 ({i}/{len(test_images)}): {Path(image_path).name}")

            try:
                result = self._benchmark_single_image(image_path)
                detailed_results.append(result)
                total_processing_time += result.processing_time

            except Exception as e:
                self.logger.error(f"画像処理エラー ({image_path}): {e}")
                # エラー時はデフォルト結果を追加
                detailed_results.append(
                    BenchmarkResult(
                        image_path=str(image_path),
                        face_detection_success=False,
                        face_confidence=0.0,
                        pose_detection_success=False,
                        pose_confidence=0.0,
                        sci_score=0.0,
                        sci_anime_score=0.0,
                        ensemble_confidence=0.0,
                        processing_time=0.0,
                        improvement_metrics={},
                    )
                )

        # 統計計算
        statistics = self._calculate_benchmark_statistics(detailed_results)

        # アンサンブル効果分析
        ensemble_effectiveness = self._analyze_ensemble_effectiveness(detailed_results)

        # 目標達成判定
        target_achievement = {
            "face_detection_rate": statistics["face_detection_rate"] >= self.targets["face_detection_rate"],
            "pose_detection_rate": statistics["pose_detection_rate"] >= self.targets["pose_detection_rate"],
            "sci_score": statistics["sci_anime_average"] >= self.targets["sci_score"],
        }

        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Week 3最終ベンチマーク完了 ({total_time:.2f}秒)")

        return FinalBenchmarkReport(
            timestamp=start_time,
            total_images=len(test_images),
            face_detection_rate=statistics["face_detection_rate"],
            pose_detection_rate=statistics["pose_detection_rate"],
            sci_average=statistics["sci_average"],
            sci_anime_average=statistics["sci_anime_average"],
            target_achievement=target_achievement,
            detailed_results=detailed_results,
            performance_summary=statistics,
            ensemble_effectiveness=ensemble_effectiveness,
        )

    def _benchmark_single_image(self, image_path: str) -> BenchmarkResult:
        """単一画像のベンチマーク"""
        start_time = time.time()

        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像読み込み失敗: {image_path}")

        # アンサンブル検出実行
        ensemble_result = self.ensemble_system.detect_comprehensive_ensemble(
            image, target_face_rate=0.90, target_pose_rate=0.80
        )

        # SCI評価（従来 vs アニメ特化）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sci_result_standard = self.evaluation_system.evaluate_single_extraction(rgb_image, anime_optimized=False)
        sci_result_anime = self.evaluation_system.evaluate_single_extraction(rgb_image, anime_optimized=True)

        # 改善指標計算
        improvement_metrics = {
            "ensemble_boost": ensemble_result.ensemble_confidence - max(
                [d.confidence for d in ensemble_result.face_detections], default=0.0
            ),
            "sci_anime_improvement": sci_result_anime.sci_total - sci_result_standard.sci_total,
            "detection_diversity": len(ensemble_result.method_contributions),
        }

        processing_time = time.time() - start_time

        return BenchmarkResult(
            image_path=str(image_path),
            face_detection_success=len(ensemble_result.face_detections) > 0,
            face_confidence=max([d.confidence for d in ensemble_result.face_detections], default=0.0),
            pose_detection_success=ensemble_result.pose_result.detected,
            pose_confidence=ensemble_result.pose_result.confidence,
            sci_score=sci_result_standard.sci_total,
            sci_anime_score=sci_result_anime.sci_total,
            ensemble_confidence=ensemble_result.ensemble_confidence,
            processing_time=processing_time,
            improvement_metrics=improvement_metrics,
        )

    def _collect_test_images(self, test_dir: str) -> List[Path]:
        """テスト画像の収集"""
        test_path = Path(test_dir)
        if not test_path.exists():
            test_path = project_root / test_dir

        if not test_path.exists():
            return []

        # 画像ファイル拡張子
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        images = []
        for ext in image_extensions:
            images.extend(test_path.glob(f"*{ext}"))
            images.extend(test_path.glob(f"*{ext.upper()}"))

        return sorted(images)[:20]  # 最大20枚に制限

    def _calculate_benchmark_statistics(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """ベンチマーク統計計算"""
        if not results:
            return {}

        # 検出成功率
        face_successes = sum(1 for r in results if r.face_detection_success)
        pose_successes = sum(1 for r in results if r.pose_detection_success)

        face_detection_rate = face_successes / len(results)
        pose_detection_rate = pose_successes / len(results)

        # 平均信頼度
        avg_face_confidence = np.mean([r.face_confidence for r in results])
        avg_pose_confidence = np.mean([r.pose_confidence for r in results])
        avg_ensemble_confidence = np.mean([r.ensemble_confidence for r in results])

        # SCI平均
        sci_average = np.mean([r.sci_score for r in results])
        sci_anime_average = np.mean([r.sci_anime_score for r in results])

        # 処理時間統計
        avg_processing_time = np.mean([r.processing_time for r in results])

        return {
            "face_detection_rate": face_detection_rate,
            "pose_detection_rate": pose_detection_rate,
            "avg_face_confidence": avg_face_confidence,
            "avg_pose_confidence": avg_pose_confidence,
            "avg_ensemble_confidence": avg_ensemble_confidence,
            "sci_average": sci_average,
            "sci_anime_average": sci_anime_average,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": sum(r.processing_time for r in results),
        }

    def _analyze_ensemble_effectiveness(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """アンサンブル効果分析"""
        if not results:
            return {}

        # アンサンブルによる改善効果
        ensemble_improvements = [r.improvement_metrics.get("ensemble_boost", 0.0) for r in results]
        avg_ensemble_improvement = np.mean(ensemble_improvements)

        # SCI アニメ特化改善効果
        sci_improvements = [r.improvement_metrics.get("sci_anime_improvement", 0.0) for r in results]
        avg_sci_improvement = np.mean(sci_improvements)

        # 手法多様性
        diversity_scores = [r.improvement_metrics.get("detection_diversity", 1.0) for r in results]
        avg_diversity = np.mean(diversity_scores)

        # 目標達成画像数
        target_achieving_images = sum(
            1
            for r in results
            if r.face_detection_success and r.pose_detection_success and r.sci_anime_score >= 0.70
        )

        target_achievement_rate = target_achieving_images / len(results)

        return {
            "avg_ensemble_improvement": avg_ensemble_improvement,
            "avg_sci_improvement": avg_sci_improvement,
            "avg_method_diversity": avg_diversity,
            "target_achievement_rate": target_achievement_rate,
            "total_target_achieving_images": target_achieving_images,
        }

    def save_benchmark_report(self, report: FinalBenchmarkReport, output_file: str = None) -> str:
        """ベンチマークレポート保存"""
        if output_file is None:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            output_file = f"week3_final_benchmark_report_{timestamp}.json"

        # JSONシリアライズ用にdatetimeとnumpy型を変換
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()
        
        # numpy bool_をPython boolに変換
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        report_dict = convert_numpy_types(report_dict)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"ベンチマークレポート保存: {output_file}")
        return output_file

    def print_benchmark_summary(self, report: FinalBenchmarkReport):
        """ベンチマーク結果サマリー表示"""
        print("\n" + "=" * 60)
        print("🚀 Week 3最終ベンチマーク結果サマリー")
        print("=" * 60)

        print(f"\n📊 基本統計:")
        print(f"  対象画像数: {report.total_images}枚")
        print(f"  処理時間: {report.performance_summary['total_processing_time']:.2f}秒")
        print(f"  平均処理時間: {report.performance_summary['avg_processing_time']:.2f}秒/枚")

        print(f"\n🎯 目標達成度:")
        face_status = "✅" if report.target_achievement["face_detection_rate"] else "❌"
        pose_status = "✅" if report.target_achievement["pose_detection_rate"] else "❌"
        sci_status = "✅" if report.target_achievement["sci_score"] else "❌"

        print(f"  {face_status} 顔検出率: {report.face_detection_rate:.1%} (目標90%)")
        print(f"  {pose_status} ポーズ検出率: {report.pose_detection_rate:.1%} (目標80%)")
        print(f"  {sci_status} SCI総合: {report.sci_anime_average:.3f} (目標0.70)")

        print(f"\n🔧 アンサンブル効果:")
        print(f"  平均信頼度向上: {report.ensemble_effectiveness['avg_ensemble_improvement']:.3f}")
        print(f"  SCI改善効果: {report.ensemble_effectiveness['avg_sci_improvement']:.3f}")
        print(f"  目標達成画像: {report.ensemble_effectiveness['total_target_achieving_images']}/{report.total_images}枚")

        # 総合評価
        all_targets_achieved = all(report.target_achievement.values())
        if all_targets_achieved:
            print(f"\n🎉 Week 3目標完全達成！")
        else:
            print(f"\n⚠️  一部目標未達成")

        print("=" * 60)


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Week 3最終目標達成確認")
    parser.add_argument("--test-dir", default="test_small", help="テスト画像ディレクトリ")
    parser.add_argument("--output", help="レポート出力ファイル名")
    args = parser.parse_args()

    try:
        # ベンチマーク実行
        benchmark = Week3FinalBenchmark()
        report = benchmark.run_final_benchmark(args.test_dir)

        # 結果表示
        benchmark.print_benchmark_summary(report)

        # レポート保存
        output_file = benchmark.save_benchmark_report(report, args.output)
        print(f"\n📄 詳細レポート: {output_file}")

        # 成功判定
        all_achieved = all(report.target_achievement.values())
        sys.exit(0 if all_achieved else 1)

    except Exception as e:
        logger.error(f"ベンチマーク実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()