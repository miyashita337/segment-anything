"""
階層的品質評価システム
P1-006: 複数レベルでの品質評価と統合判定
"""

import numpy as np
import cv2

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """品質レベル定義"""

    PIXEL = "pixel"  # ピクセルレベル
    SEGMENT = "segment"  # セグメントレベル
    OBJECT = "object"  # オブジェクトレベル
    SCENE = "scene"  # シーンレベル
    DATASET = "dataset"  # データセットレベル


class QualityMetric(Enum):
    """品質評価メトリクス"""

    COMPLETENESS = "completeness"  # 完全性
    ACCURACY = "accuracy"  # 精度
    CONSISTENCY = "consistency"  # 一貫性
    CLARITY = "clarity"  # 明瞭性
    RELEVANCE = "relevance"  # 関連性


@dataclass
class QualityScore:
    """品質スコア"""

    metric: QualityMetric
    level: QualityLevel
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "metric": self.metric.value,
            "level": self.level.value,
            "score": self.score,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class HierarchicalQualityResult:
    """階層的品質評価結果"""

    item_id: str
    scores: List[QualityScore] = field(default_factory=list)
    overall_score: float = 0.0
    overall_grade: str = "F"
    level_summaries: Dict[str, Dict] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def add_score(self, score: QualityScore):
        """スコア追加"""
        self.scores.append(score)
        self._update_overall()

    def _update_overall(self):
        """総合スコア更新"""
        if not self.scores:
            return

        # 重み付き平均による総合スコア計算
        total_weight = 0.0
        weighted_sum = 0.0

        for score in self.scores:
            weight = self._get_weight(score.level, score.metric)
            weighted_sum += score.score * score.confidence * weight
            total_weight += score.confidence * weight

        if total_weight > 0:
            self.overall_score = weighted_sum / total_weight
            self.overall_grade = self._score_to_grade(self.overall_score)

    def _get_weight(self, level: QualityLevel, metric: QualityMetric) -> float:
        """レベル・メトリクス別重み取得"""
        level_weights = {
            QualityLevel.PIXEL: 0.1,
            QualityLevel.SEGMENT: 0.2,
            QualityLevel.OBJECT: 0.3,
            QualityLevel.SCENE: 0.25,
            QualityLevel.DATASET: 0.15,
        }

        metric_weights = {
            QualityMetric.COMPLETENESS: 0.25,
            QualityMetric.ACCURACY: 0.25,
            QualityMetric.CONSISTENCY: 0.2,
            QualityMetric.CLARITY: 0.15,
            QualityMetric.RELEVANCE: 0.15,
        }

        return level_weights.get(level, 0.2) * metric_weights.get(metric, 0.2)

    def _score_to_grade(self, score: float) -> str:
        """スコアをグレードに変換"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        elif score >= 0.5:
            return "E"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "item_id": self.item_id,
            "scores": [score.to_dict() for score in self.scores],
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "level_summaries": self.level_summaries,
            "recommendations": self.recommendations,
        }


class PixelLevelEvaluator:
    """ピクセルレベル評価"""

    def evaluate(self, image: np.ndarray, mask: np.ndarray) -> List[QualityScore]:
        """ピクセルレベル評価実行"""
        scores = []

        # 完全性: マスクカバレッジ
        coverage = self._evaluate_coverage(image, mask)
        scores.append(
            QualityScore(
                QualityMetric.COMPLETENESS,
                QualityLevel.PIXEL,
                coverage["score"],
                coverage["confidence"],
                coverage,
            )
        )

        # 精度: エッジの正確性
        edge_accuracy = self._evaluate_edge_accuracy(image, mask)
        scores.append(
            QualityScore(
                QualityMetric.ACCURACY,
                QualityLevel.PIXEL,
                edge_accuracy["score"],
                edge_accuracy["confidence"],
                edge_accuracy,
            )
        )

        # 明瞭性: ピクセル品質
        clarity = self._evaluate_pixel_clarity(image, mask)
        scores.append(
            QualityScore(
                QualityMetric.CLARITY,
                QualityLevel.PIXEL,
                clarity["score"],
                clarity["confidence"],
                clarity,
            )
        )

        return scores

    def _evaluate_coverage(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """マスクカバレッジ評価"""
        if mask.size == 0:
            return {"score": 0.0, "confidence": 1.0, "coverage_ratio": 0.0}

        total_pixels = mask.size
        covered_pixels = np.sum(mask > 0)
        coverage_ratio = covered_pixels / total_pixels

        # カバレッジスコア計算（適度なカバレッジを評価）
        if coverage_ratio < 0.05:
            score = coverage_ratio / 0.05 * 0.5  # 5%未満は低評価
        elif coverage_ratio > 0.8:
            score = max(0.0, 1.0 - (coverage_ratio - 0.8) / 0.2 * 0.3)  # 80%超は減点
        else:
            score = 0.5 + (coverage_ratio - 0.05) / 0.75 * 0.5  # 適正範囲

        return {
            "score": score,
            "confidence": 0.9,
            "coverage_ratio": coverage_ratio,
            "covered_pixels": int(covered_pixels),
            "total_pixels": int(total_pixels),
        }

    def _evaluate_edge_accuracy(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """エッジ精度評価"""
        if image.size == 0 or mask.size == 0:
            return {"score": 0.0, "confidence": 1.0}

        # エッジ検出
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        # マスクエッジ
        mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

        # エッジ一致度計算
        intersection = np.logical_and(edges > 0, mask_edges > 0)
        union = np.logical_or(edges > 0, mask_edges > 0)

        if np.sum(union) == 0:
            score = 0.5  # エッジがない場合は中程度
        else:
            iou = np.sum(intersection) / np.sum(union)
            score = iou

        return {
            "score": score,
            "confidence": 0.8,
            "edge_iou": float(iou) if "iou" in locals() else 0.0,
            "image_edges": int(np.sum(edges > 0)),
            "mask_edges": int(np.sum(mask_edges > 0)),
        }

    def _evaluate_pixel_clarity(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """ピクセル明瞭性評価"""
        if image.size == 0 or mask.size == 0:
            return {"score": 0.0, "confidence": 1.0}

        # マスク領域の画像品質評価
        masked_region = image[mask > 0]

        if masked_region.size == 0:
            return {"score": 0.0, "confidence": 1.0, "mean_brightness": 0.0}

        # 明度分析
        brightness = np.mean(masked_region)
        brightness_std = np.std(masked_region)

        # 明瞭性スコア計算
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # 中間明度が最適
        contrast_score = min(1.0, brightness_std / 64.0)  # 適度なコントラスト

        score = (brightness_score + contrast_score) / 2

        return {
            "score": score,
            "confidence": 0.7,
            "mean_brightness": float(brightness),
            "brightness_std": float(brightness_std),
            "brightness_score": float(brightness_score),
            "contrast_score": float(contrast_score),
        }


class ObjectLevelEvaluator:
    """オブジェクトレベル評価"""

    def evaluate(
        self, image: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple] = None
    ) -> List[QualityScore]:
        """オブジェクトレベル評価実行"""
        scores = []

        # 完全性: オブジェクト完全性
        completeness = self._evaluate_object_completeness(image, mask, bbox)
        scores.append(
            QualityScore(
                QualityMetric.COMPLETENESS,
                QualityLevel.OBJECT,
                completeness["score"],
                completeness["confidence"],
                completeness,
            )
        )

        # 関連性: オブジェクトの妥当性
        relevance = self._evaluate_object_relevance(image, mask)
        scores.append(
            QualityScore(
                QualityMetric.RELEVANCE,
                QualityLevel.OBJECT,
                relevance["score"],
                relevance["confidence"],
                relevance,
            )
        )

        return scores

    def _evaluate_object_completeness(
        self, image: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple]
    ) -> Dict[str, float]:
        """オブジェクト完全性評価"""
        if mask.size == 0:
            return {"score": 0.0, "confidence": 1.0}

        # 境界ボックス情報がある場合の評価
        if bbox is not None:
            x, y, w, h = bbox
            bbox_area = w * h
            mask_area = np.sum(mask > 0)

            # バウンディングボックスに対するマスクの充実度
            fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0.0

            # 適切な充実度（30-80%）を評価
            if 0.3 <= fill_ratio <= 0.8:
                score = 1.0
            elif fill_ratio < 0.3:
                score = fill_ratio / 0.3
            else:
                score = max(0.0, 1.0 - (fill_ratio - 0.8) / 0.2)
        else:
            # 境界ボックス情報がない場合は形状から推定
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                score = 0.0
            else:
                # 最大輪郭の凸包比で完全性評価
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)

                convexity = contour_area / hull_area if hull_area > 0 else 0.0
                score = convexity

        return {
            "score": score,
            "confidence": 0.8,
            "fill_ratio": fill_ratio if "fill_ratio" in locals() else 0.0,
            "convexity": convexity if "convexity" in locals() else 0.0,
        }

    def _evaluate_object_relevance(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """オブジェクト妥当性評価"""
        if image.size == 0 or mask.size == 0:
            return {"score": 0.0, "confidence": 1.0}

        # サイズ妥当性
        h, w = image.shape[:2]
        mask_area = np.sum(mask > 0)
        total_area = h * w
        size_ratio = mask_area / total_area

        # 適切なサイズ範囲（5-70%）を評価
        if 0.05 <= size_ratio <= 0.7:
            size_score = 1.0
        elif size_ratio < 0.05:
            size_score = size_ratio / 0.05
        else:
            size_score = max(0.0, 1.0 - (size_ratio - 0.7) / 0.3)

        # 位置妥当性（中央寄り）
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            center_x_norm = center_x / w
            center_y_norm = center_y / h

            # 中央からの距離
            center_distance = np.sqrt((center_x_norm - 0.5) ** 2 + (center_y_norm - 0.5) ** 2)
            position_score = max(0.0, 1.0 - center_distance * 2)
        else:
            position_score = 0.0

        score = (size_score + position_score) / 2

        return {
            "score": score,
            "confidence": 0.7,
            "size_ratio": float(size_ratio),
            "size_score": float(size_score),
            "position_score": float(position_score),
            "center_x": float(center_x) if "center_x" in locals() else 0.0,
            "center_y": float(center_y) if "center_y" in locals() else 0.0,
        }


class HierarchicalQualityEvaluator:
    """階層的品質評価器"""

    def __init__(self):
        self.pixel_evaluator = PixelLevelEvaluator()
        self.object_evaluator = ObjectLevelEvaluator()
        self.evaluation_history = []

    def evaluate(
        self,
        item_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None,
    ) -> HierarchicalQualityResult:
        """階層的品質評価実行"""
        logger.info(f"🔍 階層的品質評価開始: {item_id}")

        result = HierarchicalQualityResult(item_id=item_id)

        try:
            # ピクセルレベル評価
            pixel_scores = self.pixel_evaluator.evaluate(image, mask)
            for score in pixel_scores:
                result.add_score(score)

            # オブジェクトレベル評価
            object_scores = self.object_evaluator.evaluate(image, mask, bbox)
            for score in object_scores:
                result.add_score(score)

            # レベル別サマリー生成
            result.level_summaries = self._generate_level_summaries(result.scores)

            # 改善提案生成
            result.recommendations = self._generate_recommendations(result)

            # 履歴に追加
            self.evaluation_history.append(result)

            logger.info(
                f"✅ 階層的品質評価完了: {item_id} (スコア: {result.overall_score:.3f}, グレード: {result.overall_grade})"
            )

        except Exception as e:
            logger.error(f"❌ 階層的品質評価エラー: {item_id} - {e}")
            result.overall_score = 0.0
            result.overall_grade = "F"

        return result

    def _generate_level_summaries(self, scores: List[QualityScore]) -> Dict[str, Dict]:
        """レベル別サマリー生成"""
        summaries = {}

        # レベル別グループ化
        level_groups = defaultdict(list)
        for score in scores:
            level_groups[score.level.value].append(score)

        # 各レベルのサマリー計算
        for level, level_scores in level_groups.items():
            avg_score = np.mean([s.score for s in level_scores])
            avg_confidence = np.mean([s.confidence for s in level_scores])

            summaries[level] = {
                "average_score": float(avg_score),
                "average_confidence": float(avg_confidence),
                "metric_count": len(level_scores),
                "metrics": [s.metric.value for s in level_scores],
            }

        return summaries

    def _generate_recommendations(self, result: HierarchicalQualityResult) -> List[str]:
        """改善提案生成"""
        recommendations = []

        # 低スコア項目の改善提案
        low_scores = [s for s in result.scores if s.score < 0.6]

        for score in low_scores:
            if score.metric == QualityMetric.COMPLETENESS:
                if score.level == QualityLevel.PIXEL:
                    recommendations.append("マスクのカバレッジを改善してください")
                elif score.level == QualityLevel.OBJECT:
                    recommendations.append("オブジェクトの完全な輪郭抽出を確認してください")

            elif score.metric == QualityMetric.ACCURACY:
                recommendations.append("エッジの精度向上のため、セグメンテーション閾値を調整してください")

            elif score.metric == QualityMetric.CLARITY:
                recommendations.append("画像の明度・コントラストを改善してください")

            elif score.metric == QualityMetric.RELEVANCE:
                recommendations.append("オブジェクトのサイズと位置を確認してください")

        # 総合スコアに基づく提案
        if result.overall_score < 0.5:
            recommendations.append("品質が低いため、手動修正または再処理を推奨します")
        elif result.overall_score < 0.7:
            recommendations.append("部分的な修正により品質向上が期待できます")

        return recommendations

    def evaluate_batch(self, items: List[Dict]) -> List[HierarchicalQualityResult]:
        """バッチ評価"""
        results = []

        for i, item in enumerate(items):
            logger.info(f"📊 バッチ評価進捗: {i+1}/{len(items)}")

            result = self.evaluate(
                item["id"], item["image"], item["mask"], item.get("bbox"), item.get("metadata")
            )
            results.append(result)

        return results

    def get_dataset_summary(self) -> Dict[str, Any]:
        """データセット統計取得"""
        if not self.evaluation_history:
            return {}

        scores = [r.overall_score for r in self.evaluation_history]
        grades = [r.overall_grade for r in self.evaluation_history]

        grade_counts = {}
        for grade in ["A", "B", "C", "D", "E", "F"]:
            grade_counts[grade] = grades.count(grade)

        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "grade_distribution": grade_counts,
            "success_rate": grade_counts.get("A", 0) + grade_counts.get("B", 0),
        }

    def save_results(self, output_path: Path):
        """結果保存"""
        output_path.mkdir(parents=True, exist_ok=True)

        # 個別結果保存
        for result in self.evaluation_history:
            result_file = output_path / f"{result.item_id}_hierarchical_quality.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        # データセット統計保存
        summary = self.get_dataset_summary()
        summary_file = output_path / "dataset_quality_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"📄 階層的品質評価結果保存完了: {output_path}")


# 使いやすいインターフェース関数
def evaluate_hierarchical_quality(
    image: np.ndarray, mask: np.ndarray, item_id: str = "unknown", bbox: Optional[Tuple] = None
) -> HierarchicalQualityResult:
    """階層的品質評価のシンプルインターフェース"""
    evaluator = HierarchicalQualityEvaluator()
    return evaluator.evaluate(item_id, image, mask, bbox)
