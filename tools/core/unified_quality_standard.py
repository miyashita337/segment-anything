#!/usr/bin/env python3
"""
OPT-024: 品質基準統一システム
データセット横断的な評価基準の統一実装

PROGRESS_TRACKER.md準拠のワークフロー実装:
- 実装修正 → 動作確認・ユニットTEST → 抽出パイプライン実行（バックグラウンド）
- 品質評価 → 統合実行スクリプト → ダッシュボード生成
"""

import numpy as np

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityStandard:
    """統一品質基準"""

    name: str
    version: str

    # A/B評価基準
    ab_excellent_threshold: float = 0.9  # A評価: 90%以上
    ab_good_threshold: float = 0.7  # B評価: 70%以上
    ab_acceptable_threshold: float = 0.5  # C評価: 50%以上

    # SCI (Structural Completeness Index) 基準
    sci_excellent_threshold: float = 0.85
    sci_good_threshold: float = 0.7
    sci_acceptable_threshold: float = 0.5

    # PLA (Pose Landmark Accuracy) 基準
    pla_excellent_threshold: float = 0.8
    pla_good_threshold: float = 0.65
    pla_acceptable_threshold: float = 0.4

    # PLE (Pose Landmark Extraction) 基準
    ple_excellent_threshold: float = 0.85
    ple_good_threshold: float = 0.7
    ple_acceptable_threshold: float = 0.5

    # 成功率基準
    success_rate_excellent: float = 0.95
    success_rate_good: float = 0.8
    success_rate_acceptable: float = 0.6

    # フィル率基準（マスク品質）
    fill_ratio_excellent: float = 0.9
    fill_ratio_good: float = 0.75
    fill_ratio_acceptable: float = 0.6

    # コンパクトネス基準（形状品質）
    compactness_excellent: float = 0.8
    compactness_good: float = 0.6
    compactness_acceptable: float = 0.4

    # カバレッジ率基準（検出範囲）
    coverage_excellent: float = 0.9
    coverage_good: float = 0.75
    coverage_acceptable: float = 0.6

    # データセット横断的重み設定
    dataset_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.dataset_weights is None:
            self.dataset_weights = {
                "kana03": 1.0,  # 基準データセット
                "kana04": 1.0,
                "kana05": 1.0,
                "kana06": 1.0,
                "kana07": 1.0,
                "kana08": 1.0,  # 主要評価データセット
                "kana09": 0.8,  # 新規データセット
            }


@dataclass
class UnifiedQualityResult:
    """統一品質評価結果"""

    dataset_name: str
    total_processed: int
    successful_extractions: int

    # 主要指標
    ab_evaluation_rate: float
    sci_score: float
    pla_score: float
    ple_score: float
    success_rate: float

    # 補助指標
    avg_fill_ratio: float
    avg_compactness: float
    avg_coverage: float

    # グレード分布
    grade_distribution: Dict[str, int]

    # 統一評価
    unified_score: float
    unified_grade: str
    quality_level: str  # EXCELLENT, GOOD, ACCEPTABLE, POOR

    # 詳細情報
    evaluation_timestamp: str
    processing_time: float
    detailed_metrics: Dict[str, Any]


class UnifiedQualityStandardSystem:
    """統一品質基準システム"""

    def __init__(self, config_path: Optional[Path] = None):
        """初期化"""
        self.project_root = project_root

        # PROGRESS_TRACKER.md準拠の正しいパス
        self.workspace_root = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace")
        self.workspace_dir = self.workspace_root / "OPT-024"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 標準設定パス
        self.config_path = config_path or (self.workspace_dir / "quality_standard_config.json")
        self.standards_file = self.workspace_dir / "quality" / "unified_standards.json"

        # ディレクトリ作成
        for subdir in ["extraction", "quality", "dashboard", "tests"]:
            (self.workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

        # 品質基準読み込み
        self.standard = self._load_or_create_standard()

        print(f"🎯 OPT-024: 品質基準統一システム初期化完了")
        print(f"ワークスペース: {self.workspace_dir}")

    def _load_or_create_standard(self) -> QualityStandard:
        """品質基準読み込みまたは作成"""
        if self.standards_file.exists():
            try:
                with open(self.standards_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"既存品質基準読み込み: {self.standards_file}")
                return QualityStandard(**data)
            except Exception as e:
                logger.warning(f"品質基準読み込みエラー: {e}")

        # デフォルト品質基準作成
        standard = QualityStandard(name="unified_quality_standard_v1", version="1.0.0")

        # 保存
        self.standards_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.standards_file, "w", encoding="utf-8") as f:
            json.dump(asdict(standard), f, indent=2, ensure_ascii=False)

        logger.info(f"新規品質基準作成: {self.standards_file}")
        return standard

    def evaluate_dataset_quality(
        self, dataset_name: str, results_data: Dict[str, Any]
    ) -> UnifiedQualityResult:
        """データセット品質評価（統一基準）"""
        start_time = datetime.now()
        logger.info(f"統一品質評価開始: {dataset_name}")

        # 基本メトリクス抽出
        total_processed = results_data.get("total_processed", 0)
        successful_extractions = results_data.get("successful_extractions", 0)

        # 主要指標
        ab_evaluation_rate = results_data.get("ab_evaluation_rate", 0.0)
        sci_score = results_data.get("sci_score", 0.0)
        pla_score = results_data.get("pla_score", 0.0)
        ple_score = results_data.get("ple_score", 0.0)
        success_rate = successful_extractions / total_processed if total_processed > 0 else 0.0

        # 補助指標
        avg_fill_ratio = results_data.get("avg_fill_ratio", 0.0)
        avg_compactness = results_data.get("avg_compactness", 0.0)
        avg_coverage = results_data.get("avg_coverage", 0.0)

        # グレード分布
        grade_distribution = results_data.get("grade_distribution", {})

        # 統一評価計算
        unified_score = self._calculate_unified_score(
            ab_evaluation_rate,
            sci_score,
            pla_score,
            ple_score,
            success_rate,
            avg_fill_ratio,
            avg_compactness,
            avg_coverage,
            dataset_name,
        )

        unified_grade = self._determine_unified_grade(unified_score)
        quality_level = self._determine_quality_level(unified_score)

        # 処理時間計算
        processing_time = (datetime.now() - start_time).total_seconds()

        result = UnifiedQualityResult(
            dataset_name=dataset_name,
            total_processed=total_processed,
            successful_extractions=successful_extractions,
            ab_evaluation_rate=ab_evaluation_rate,
            sci_score=sci_score,
            pla_score=pla_score,
            ple_score=ple_score,
            success_rate=success_rate,
            avg_fill_ratio=avg_fill_ratio,
            avg_compactness=avg_compactness,
            avg_coverage=avg_coverage,
            grade_distribution=grade_distribution,
            unified_score=unified_score,
            unified_grade=unified_grade,
            quality_level=quality_level,
            evaluation_timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            detailed_metrics=results_data,
        )

        logger.info(f"統一品質評価完了: {dataset_name} -> {quality_level} ({unified_score:.3f})")
        return result

    def _calculate_unified_score(
        self,
        ab_rate: float,
        sci: float,
        pla: float,
        ple: float,
        success_rate: float,
        fill_ratio: float,
        compactness: float,
        coverage: float,
        dataset_name: str,
    ) -> float:
        """統一スコア計算"""
        # データセット重み取得
        dataset_weight = self.standard.dataset_weights.get(dataset_name, 1.0)

        # 主要指標重み（合計1.0）
        weights = {
            "ab_rate": 0.3,
            "sci": 0.2,
            "pla": 0.15,
            "ple": 0.15,
            "success_rate": 0.1,
            "fill_ratio": 0.04,
            "compactness": 0.03,
            "coverage": 0.03,
        }

        # 重み付き合計
        weighted_score = (
            ab_rate * weights["ab_rate"]
            + sci * weights["sci"]
            + pla * weights["pla"]
            + ple * weights["ple"]
            + success_rate * weights["success_rate"]
            + fill_ratio * weights["fill_ratio"]
            + compactness * weights["compactness"]
            + coverage * weights["coverage"]
        )

        # データセット重み適用
        unified_score = weighted_score * dataset_weight

        return min(unified_score, 1.0)  # 1.0で上限

    def _determine_unified_grade(self, score: float) -> str:
        """統一グレード判定"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.65:
            return "B"
        elif score >= 0.55:
            return "C+"
        elif score >= 0.45:
            return "C"
        elif score >= 0.35:
            return "D"
        else:
            return "F"

    def _determine_quality_level(self, score: float) -> str:
        """品質レベル判定"""
        if score >= 0.85:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.5:
            return "ACCEPTABLE"
        else:
            return "POOR"

    def compare_datasets(self, results: List[UnifiedQualityResult]) -> Dict[str, Any]:
        """データセット横断比較"""
        if not results:
            return {}

        logger.info(f"データセット横断比較: {len(results)}件")

        # 統計計算
        scores = [r.unified_score for r in results]
        ab_rates = [r.ab_evaluation_rate for r in results]
        success_rates = [r.success_rate for r in results]

        comparison = {
            "total_datasets": len(results),
            "overall_statistics": {
                "unified_score": {
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                },
                "ab_evaluation_rate": {
                    "mean": np.mean(ab_rates),
                    "median": np.median(ab_rates),
                    "std": np.std(ab_rates),
                },
                "success_rate": {
                    "mean": np.mean(success_rates),
                    "median": np.median(success_rates),
                    "std": np.std(success_rates),
                },
            },
            "quality_level_distribution": {},
            "dataset_ranking": [],
            "recommendations": [],
        }

        # 品質レベル分布
        for result in results:
            level = result.quality_level
            comparison["quality_level_distribution"][level] = (
                comparison["quality_level_distribution"].get(level, 0) + 1
            )

        # データセットランキング
        sorted_results = sorted(results, key=lambda x: x.unified_score, reverse=True)
        for i, result in enumerate(sorted_results):
            comparison["dataset_ranking"].append(
                {
                    "rank": i + 1,
                    "dataset": result.dataset_name,
                    "unified_score": result.unified_score,
                    "quality_level": result.quality_level,
                    "unified_grade": result.unified_grade,
                }
            )

        # 推奨事項生成
        comparison["recommendations"] = self._generate_recommendations(results)

        return comparison

    def _generate_recommendations(self, results: List[UnifiedQualityResult]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        poor_datasets = [r for r in results if r.quality_level == "POOR"]
        if poor_datasets:
            recommendations.append(f"品質改善が必要なデータセット: {[r.dataset_name for r in poor_datasets]}")

        low_ab_datasets = [r for r in results if r.ab_evaluation_rate < 0.6]
        if low_ab_datasets:
            recommendations.append(f"A/B評価率改善が必要: {[r.dataset_name for r in low_ab_datasets]}")

        best_dataset = max(results, key=lambda x: x.unified_score)
        recommendations.append(
            f"最高品質データセット: {best_dataset.dataset_name} ({best_dataset.unified_score:.3f})"
        )

        return recommendations

    def save_evaluation_result(self, result: UnifiedQualityResult) -> Path:
        """評価結果保存"""
        output_file = (
            self.workspace_dir
            / "quality"
            / f"unified_quality_{result.dataset_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        logger.info(f"評価結果保存: {output_file}")
        return output_file

    def save_comparison_result(self, comparison: Dict[str, Any]) -> Path:
        """比較結果保存"""
        output_file = (
            self.workspace_dir
            / "quality"
            / f"dataset_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"比較結果保存: {output_file}")
        return output_file

    def generate_unified_report(self, results: List[UnifiedQualityResult]) -> Dict[str, Any]:
        """統一品質レポート生成"""
        comparison = self.compare_datasets(results)

        report = {
            "report_id": f"P1A002_unified_quality_{datetime.now():%Y%m%d_%H%M%S}",
            "generated_at": datetime.now().isoformat(),
            "standard_version": self.standard.version,
            "summary": {
                "total_datasets_evaluated": len(results),
                "overall_quality_level": self._determine_overall_quality(results),
                "average_unified_score": np.mean([r.unified_score for r in results])
                if results
                else 0.0,
            },
            "individual_results": [asdict(r) for r in results],
            "cross_dataset_comparison": comparison,
            "quality_standard_applied": asdict(self.standard),
        }

        # レポート保存
        report_file = (
            self.workspace_dir
            / "quality"
            / f"unified_quality_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"統一品質レポート生成完了: {report_file}")
        return report

    def _determine_overall_quality(self, results: List[UnifiedQualityResult]) -> str:
        """全体品質レベル判定"""
        if not results:
            return "UNKNOWN"

        avg_score = np.mean([r.unified_score for r in results])
        return self._determine_quality_level(avg_score)


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="OPT-024: 品質基準統一システム")
    parser.add_argument("--dataset", help="データセット名")
    parser.add_argument("--results", help="結果ファイルパス")
    parser.add_argument("--compare", action="store_true", help="データセット比較モード")

    args = parser.parse_args()

    system = UnifiedQualityStandardSystem()

    if args.dataset and args.results:
        # 単一データセット評価
        with open(args.results, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        result = system.evaluate_dataset_quality(args.dataset, results_data)
        system.save_evaluation_result(result)

        print(f"✅ {args.dataset}品質評価完了:")
        print(f"   統一スコア: {result.unified_score:.3f}")
        print(f"   品質レベル: {result.quality_level}")
        print(f"   統一グレード: {result.unified_grade}")

    elif args.compare:
        # サンプル比較実行
        print("📊 品質基準統一システム - 比較デモ実行")

        # サンプルデータで比較デモ
        sample_results = [
            system.evaluate_dataset_quality(
                "kana08",
                {
                    "total_processed": 16,
                    "successful_extractions": 15,
                    "ab_evaluation_rate": 0.8,
                    "sci_score": 0.75,
                    "pla_score": 0.8,
                    "ple_score": 0.85,
                    "avg_fill_ratio": 0.85,
                    "avg_compactness": 0.7,
                    "avg_coverage": 0.8,
                },
            ),
            system.evaluate_dataset_quality(
                "kana05",
                {
                    "total_processed": 36,
                    "successful_extractions": 34,
                    "ab_evaluation_rate": 0.85,
                    "sci_score": 0.8,
                    "pla_score": 0.75,
                    "ple_score": 0.9,
                    "avg_fill_ratio": 0.9,
                    "avg_compactness": 0.75,
                    "avg_coverage": 0.85,
                },
            ),
        ]

        report = system.generate_unified_report(sample_results)
        print(f"✅ 統一品質レポート生成完了")
        print(f"   評価データセット数: {report['summary']['total_datasets_evaluated']}")
        print(f"   全体品質レベル: {report['summary']['overall_quality_level']}")
        print(f"   平均統一スコア: {report['summary']['average_unified_score']:.3f}")

    else:
        print("🎯 OPT-024: 品質基準統一システム")
        print("使用例:")
        print("  python unified_quality_standard.py --dataset kana08 --results results.json")
        print("  python unified_quality_standard.py --compare")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
