#!/usr/bin/env python3
"""
統合品質チェックシステム v1.0.0 (レガシー版)
過去の結果再現用：品質分布ベース評価のみ

このファイルは過去データの一貫性確保のため、
アルゴリズム変更前の計算方法を完全に保持しています。
"""

import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """品質指標の結果"""
    name: str
    value: float
    threshold: Optional[float] = None
    status: str = "measured"
    category: str = "general"
    notes: str = ""
    improvement_suggestions: List[str] = None

    def __post_init__(self):
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []


@dataclass
class UnifiedQualityReport:
    """統合品質レポート"""
    timestamp: str
    dataset_name: str
    total_images: int
    evaluation_metrics: List[QualityMetric]
    mask_metrics: List[QualityMetric]
    objective_metrics: List[QualityMetric]
    overall_score: float
    passed_metrics: int
    total_metrics: int
    status: str
    priority_improvements: List[str]
    technical_recommendations: List[str]


class LegacyUnifiedQualityChecker:
    """統合品質チェッカー レガシー版 (v1.0.0)"""
    
    VERSION = "v1.0.0-legacy"
    
    def __init__(self):
        """初期化 - オリジナルの閾値を使用"""
        self.thresholds = {
            "largest_char_accuracy": 0.80,
            "ab_evaluation_rate": 0.70,  # オリジナル閾値
            "fps": 0.2,
            "sci_score": 0.70,           # オリジナル閾値
            "pla_score": 0.75,
            "ple_score": 0.10
        }
    
    def check_extraction_results(self, results_path: str) -> UnifiedQualityReport:
        """抽出結果ファイルから品質チェック実行（レガシー版）"""
        try:
            results_path = Path(results_path)
            
            if not results_path.exists():
                raise FileNotFoundError(f"結果ファイルが見つかりません: {results_path}")
            
            with open(results_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            logger.info(f"レガシー品質チェック実行: {results_path}")
            
            dataset_name = self._extract_dataset_name(str(results_path))
            
            # レガシー評価のみ実行
            evaluation_metrics = self._check_evaluation_metrics_legacy(extraction_data)
            mask_metrics = []  # レガシー版では簡略化
            objective_metrics = self._check_objective_metrics_legacy(extraction_data)
            
            # 統合レポート作成
            report = self._create_unified_report(
                dataset_name=dataset_name,
                extraction_data=extraction_data,
                evaluation_metrics=evaluation_metrics,
                mask_metrics=mask_metrics,
                objective_metrics=objective_metrics
            )
            
            return report
            
        except Exception as e:
            logger.error(f"レガシー品質チェックエラー: {e}")
            raise
    
    def _extract_dataset_name(self, path: str) -> str:
        """パスからデータセット名を抽出"""
        if "kana08" in path:
            return "kana08"
        elif "kana07" in path:
            return "kana07"
        elif "kana06" in path:
            return "kana06"
        else:
            return "unknown"
    
    def _check_evaluation_metrics_legacy(self, extraction_data: Dict) -> List[QualityMetric]:
        """評価指標システムのチェック（レガシー版：品質分布ベース）"""
        metrics = []
        
        try:
            total_images = extraction_data.get("total_images", 0)
            success_count = extraction_data.get("success_count", 0)
            quality_dist = extraction_data.get("quality_distribution", {})
            avg_processing_time = extraction_data.get("avg_processing_time", 0)
            
            # 1. Largest-Character Accuracy
            accuracy = success_count / total_images if total_images > 0 else 0.0
            metrics.append(QualityMetric(
                name="Largest-Character Accuracy",
                value=accuracy,
                threshold=self.thresholds["largest_char_accuracy"],
                status="passed" if accuracy >= self.thresholds["largest_char_accuracy"] else "failed",
                category="evaluation",
                notes=f"{success_count}/{total_images} 成功",
                improvement_suggestions=["YOLO閾値調整", "SAM後処理改良"] if accuracy < self.thresholds["largest_char_accuracy"] else []
            ))
            
            # 2. A/B評価率（レガシー版：品質分布のみ）
            ab_count = quality_dist.get('A', 0) + quality_dist.get('B', 0)
            ab_rate = ab_count / success_count if success_count > 0 else 0.0
            metrics.append(QualityMetric(
                name="A/B評価率",
                value=ab_rate,
                threshold=self.thresholds["ab_evaluation_rate"],
                status="passed" if ab_rate >= self.thresholds["ab_evaluation_rate"] else "failed",
                category="evaluation",
                notes=f"{ab_count}/{success_count} A/B評価",
                improvement_suggestions=["品質判定基準見直し", "セグメンテーション精度向上"] if ab_rate < self.thresholds["ab_evaluation_rate"] else []
            ))
            
            # 3. FPS
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
            metrics.append(QualityMetric(
                name="FPS",
                value=fps,
                threshold=self.thresholds["fps"],
                status="passed" if fps >= self.thresholds["fps"] else "failed",
                category="evaluation",
                notes=f"平均処理時間: {avg_processing_time:.2f}秒",
                improvement_suggestions=["GPU最適化", "モデル軽量化"] if fps < self.thresholds["fps"] else []
            ))
            
            # 4. C以上評価率
            c_or_better = quality_dist.get('A', 0) + quality_dist.get('B', 0) + quality_dist.get('C', 0)
            c_rate = c_or_better / success_count if success_count > 0 else 0.0
            metrics.append(QualityMetric(
                name="C以上評価率",
                value=c_rate,
                threshold=0.5,
                status="passed" if c_rate >= 0.5 else "failed",
                category="evaluation",
                notes=f"{c_or_better}/{success_count} C以上評価",
                improvement_suggestions=["全体的品質向上", "困難ケース対策"] if c_rate < 0.5 else []
            ))
            
            logger.info(f"レガシー評価指標チェック完了: {len(metrics)}指標")
            
        except Exception as e:
            logger.error(f"レガシー評価指標チェックエラー: {e}")
            metrics.append(QualityMetric(
                name="評価指標システム",
                value=0.0,
                status="error",
                category="evaluation",
                notes=f"エラー: {str(e)}"
            ))
        
        return metrics
    
    def _check_objective_metrics_legacy(self, extraction_data: Dict) -> List[QualityMetric]:
        """客観的指標システムのチェック（レガシー版：品質分布ベース）"""
        metrics = []
        
        try:
            success_count = extraction_data.get("success_count", 0)
            
            if success_count > 0:
                # レガシーSCI計算：品質分布から重み付き平均
                quality_dist = extraction_data.get("quality_distribution", {})
                grade_weights = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'E': 0.2, 'F': 0.0}
                weighted_sum = sum(quality_dist.get(grade, 0) * weight for grade, weight in grade_weights.items())
                sci_estimated = weighted_sum / success_count if success_count > 0 else 0.0
                
                metrics.append(QualityMetric(
                    name="SCI (Semantic Completeness Index)",
                    value=sci_estimated,
                    threshold=self.thresholds["sci_score"],
                    status="passed" if sci_estimated >= self.thresholds["sci_score"] else "failed",
                    category="objective",
                    notes=f"品質分布から推定 ({success_count}枚ベース)",
                    improvement_suggestions=["直接画像分析", "MediaPipe姿勢推定強化"] if sci_estimated < self.thresholds["sci_score"] else []
                ))
            else:
                metrics.append(QualityMetric(
                    name="SCI (Semantic Completeness Index)",
                    value=0.0,
                    threshold=self.thresholds["sci_score"],
                    status="no_data",
                    category="objective",
                    notes="成功データがありません"
                ))
            
            # PLA（簡易版）
            statistics = extraction_data.get("statistics", {})
            avg_sam_score = statistics.get("avg_sam_score", 0.0)
            avg_mask_ratio = statistics.get("avg_mask_ratio", 0.0)
            pla_estimated = (avg_sam_score * 0.7 + min(avg_mask_ratio * 5, 1.0) * 0.3)
            
            metrics.append(QualityMetric(
                name="PLA (Pixel-Level Accuracy)",
                value=pla_estimated,
                threshold=self.thresholds["pla_score"],
                status="passed" if pla_estimated >= self.thresholds["pla_score"] else "failed",
                category="objective",
                notes=f"SAMスコア({avg_sam_score:.3f})とマスク比率({avg_mask_ratio:.3f})から推定",
                improvement_suggestions=["ground truth データ準備", "直接IoU計算"] if pla_estimated < self.thresholds["pla_score"] else []
            ))
            
            # PLE（簡易版）
            metrics.append(QualityMetric(
                name="PLE (Progressive Learning Efficiency)",
                value=0.0,
                threshold=self.thresholds["ple_score"],
                status="baseline_created",
                category="objective",
                notes="レガシー版 - ベースライン作成",
                improvement_suggestions=["継続的実行による履歴蓄積"]
            ))
            
            logger.info(f"レガシー客観指標チェック完了: {len(metrics)}指標")
            
        except Exception as e:
            logger.error(f"レガシー客観指標チェックエラー: {e}")
            metrics.append(QualityMetric(
                name="客観的指標システム",
                value=0.0,
                status="error",
                category="objective",
                notes=f"エラー: {str(e)}"
            ))
        
        return metrics
    
    def _create_unified_report(self, dataset_name: str, extraction_data: Dict,
                             evaluation_metrics: List[QualityMetric],
                             mask_metrics: List[QualityMetric],
                             objective_metrics: List[QualityMetric]) -> UnifiedQualityReport:
        """統合レポート作成（レガシー版）"""
        
        all_metrics = evaluation_metrics + mask_metrics + objective_metrics
        implemented_metrics = [m for m in all_metrics if m.status not in ["not_implemented", "error"]]
        passed_metrics = sum(1 for m in implemented_metrics if m.status == "passed")
        
        overall_score = passed_metrics / len(implemented_metrics) if implemented_metrics else 0.0
        
        if overall_score >= 0.8:
            status = "PASS"
        elif overall_score >= 0.5:
            status = "PARTIAL"
        else:
            status = "FAIL"
        
        priority_improvements = []
        for metric in implemented_metrics:
            if metric.status == "failed":
                priority_improvements.extend(metric.improvement_suggestions)
        
        priority_improvements = list(set(priority_improvements))
        
        return UnifiedQualityReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name=dataset_name,
            total_images=extraction_data.get("total_images", 0),
            evaluation_metrics=evaluation_metrics,
            mask_metrics=mask_metrics,
            objective_metrics=objective_metrics,
            overall_score=overall_score,
            passed_metrics=passed_metrics,
            total_metrics=len(implemented_metrics),
            status=status,
            priority_improvements=priority_improvements,
            technical_recommendations=[]
        )
    
    def save_report(self, report: UnifiedQualityReport, output_path: str) -> None:
        """レポート保存"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # バージョン情報を追加
            report_dict = asdict(report)
            report_dict["algorithm_version"] = self.VERSION
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"レガシー品質レポート保存完了: {output_path}")
            
        except Exception as e:
            logger.error(f"レガシーレポート保存エラー: {e}")
            raise


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="統合品質チェックシステム レガシー版")
    parser.add_argument("--results", "-r", required=True, help="抽出結果JSONファイルパス")
    parser.add_argument("--output", "-o", help="レポート出力パス（省略時は自動生成）")
    
    args = parser.parse_args()
    
    try:
        checker = LegacyUnifiedQualityChecker()
        report = checker.check_extraction_results(args.results)
        
        if args.output:
            output_path = args.output
        else:
            results_path = Path(args.results)
            output_path = results_path.parent / f"unified_quality_report_legacy_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        checker.save_report(report, str(output_path))
        
        print(f"📄 レガシー品質レポート: {output_path}")
        print(f"🔢 アルゴリズムバージョン: {checker.VERSION}")
        print(f"📊 A/B評価率: {[m.value for m in report.evaluation_metrics if 'A/B評価率' in m.name][0]:.1%}")
        print(f"🎯 総合スコア: {report.overall_score:.1%}")
        print(f"🏆 ステータス: {report.status}")
        
    except Exception as e:
        logger.error(f"レガシー品質チェック失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()