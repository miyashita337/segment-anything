"""
評価説明可能性システム
P1-007: 品質評価の根拠と改善提案の詳細説明
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """説明タイプ定義"""

    VISUAL = "visual"  # ビジュアル説明
    TEXTUAL = "textual"  # テキスト説明
    NUMERICAL = "numerical"  # 数値説明
    COMPARATIVE = "comparative"  # 比較説明


@dataclass
class QualityFactor:
    """品質要因"""

    name: str
    value: float
    importance: float
    description: str
    improvement_suggestion: str = ""
    visual_indicators: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "value": self.value,
            "importance": self.importance,
            "description": self.description,
            "improvement_suggestion": self.improvement_suggestion,
            "visual_indicators": self.visual_indicators,
        }


@dataclass
class ExplanationResult:
    """説明結果"""

    item_id: str
    overall_score: float
    overall_grade: str
    factors: List[QualityFactor] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    visual_paths: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "item_id": self.item_id,
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "factors": [f.to_dict() for f in self.factors],
            "explanations": self.explanations,
            "visual_paths": self.visual_paths,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


class VisualExplainer:
    """ビジュアル説明生成器"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日本語フォント設定
        try:
            self.font_prop = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        except:
            self.font_prop = None
            logger.warning("日本語フォントが利用できません")

    def create_quality_breakdown_chart(self, item_id: str, factors: List[QualityFactor]) -> str:
        """品質分解チャート生成"""
        plt.figure(figsize=(12, 8))

        # データ準備
        names = [f.name for f in factors]
        values = [f.value for f in factors]
        importances = [f.importance for f in factors]

        # カラーマップ
        colors = plt.cm.RdYlGn(np.array(values))

        # 横棒グラフ
        y_pos = np.arange(len(names))
        bars = plt.barh(y_pos, values, color=colors, alpha=0.7)

        # 重要度の表示（バーの幅で表現）
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            height = 0.3 + (importance * 0.4)  # 0.3-0.7の範囲
            bar.set_height(height)

        # グラフ設定
        plt.xlabel("品質スコア", fontproperties=self.font_prop)
        plt.ylabel("品質要因", fontproperties=self.font_prop)
        plt.title(f"品質分解チャート: {item_id}", fontproperties=self.font_prop)
        plt.yticks(y_pos, names, fontproperties=self.font_prop)
        plt.xlim(0, 1)

        # スコア値をバーに表示
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontproperties=self.font_prop,
            )

        # 凡例
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="red",
                markersize=10,
                alpha=0.7,
                label="低品質 (0.0-0.3)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="yellow",
                markersize=10,
                alpha=0.7,
                label="中品質 (0.3-0.7)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="green",
                markersize=10,
                alpha=0.7,
                label="高品質 (0.7-1.0)",
            ),
        ]
        plt.legend(handles=legend_elements, loc="lower right", prop=self.font_prop)

        plt.tight_layout()

        # 保存
        output_path = self.output_dir / f"{item_id}_quality_breakdown.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 品質分解チャート生成: {output_path}")
        return str(output_path)

    def create_image_annotation(
        self, item_id: str, image: np.ndarray, mask: np.ndarray, factors: List[QualityFactor]
    ) -> str:
        """画像アノテーション生成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 元画像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("元画像", fontproperties=self.font_prop)
        axes[0, 0].axis("off")

        # マスク表示
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title("抽出マスク", fontproperties=self.font_prop)
        axes[0, 1].axis("off")

        # マスク適用画像
        masked_image = image.copy()
        if len(image.shape) == 3:
            masked_image[mask == 0] = [0, 0, 0]
        else:
            masked_image[mask == 0] = 0
        axes[1, 0].imshow(masked_image)
        axes[1, 0].set_title("マスク適用結果", fontproperties=self.font_prop)
        axes[1, 0].axis("off")

        # 品質要因オーバーレイ
        axes[1, 1].imshow(image)

        # エッジ表示
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 50, 150)

        # マスクエッジ
        mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

        # エッジオーバーレイ
        axes[1, 1].contour(edges, colors="red", linewidths=1, alpha=0.7)
        axes[1, 1].contour(mask_edges, colors="blue", linewidths=2, alpha=0.8)

        axes[1, 1].set_title("エッジ解析 (赤:画像エッジ, 青:マスクエッジ)", fontproperties=self.font_prop)
        axes[1, 1].axis("off")

        # 品質要因テキスト追加
        textstr = "\\n".join(
            [
                f"• {f.name}: {f.value:.3f} ({'高' if f.value > 0.7 else '中' if f.value > 0.4 else '低'}品質)"
                for f in factors[:5]  # 最大5個まで表示
            ]
        )

        # テキストボックスを右下に配置
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        fig.text(
            0.98,
            0.02,
            textstr,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
            fontproperties=self.font_prop,
        )

        plt.suptitle(f"画像品質解析: {item_id}", fontsize=16, fontproperties=self.font_prop)
        plt.tight_layout()

        # 保存
        output_path = self.output_dir / f"{item_id}_image_annotation.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"🖼️ 画像アノテーション生成: {output_path}")
        return str(output_path)

    def create_comparison_chart(self, results: List[ExplanationResult]) -> str:
        """比較チャート生成"""
        if len(results) < 2:
            return ""

        plt.figure(figsize=(14, 10))

        # データ準備
        item_ids = [r.item_id for r in results]
        scores = [r.overall_score for r in results]

        # 全要因の収集
        all_factor_names = set()
        for result in results:
            for factor in result.factors:
                all_factor_names.add(factor.name)

        all_factor_names = sorted(list(all_factor_names))

        # ヒートマップデータ作成
        heatmap_data = []
        for result in results:
            factor_dict = {f.name: f.value for f in result.factors}
            row = [factor_dict.get(name, 0.0) for name in all_factor_names]
            heatmap_data.append(row)

        # ヒートマップ描画
        im = plt.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # 軸設定
        plt.xticks(
            range(len(all_factor_names)),
            all_factor_names,
            rotation=45,
            ha="right",
            fontproperties=self.font_prop,
        )
        plt.yticks(range(len(item_ids)), item_ids, fontproperties=self.font_prop)

        # 値を表示
        for i in range(len(item_ids)):
            for j in range(len(all_factor_names)):
                text = plt.text(
                    j,
                    i,
                    f"{heatmap_data[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontproperties=self.font_prop,
                )

        plt.title("品質要因比較ヒートマップ", fontproperties=self.font_prop)
        plt.colorbar(im, label="品質スコア")
        plt.tight_layout()

        # 保存
        output_path = self.output_dir / "quality_comparison_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📈 比較チャート生成: {output_path}")
        return str(output_path)


class TextualExplainer:
    """テキスト説明生成器"""

    def __init__(self):
        self.quality_thresholds = {"excellent": 0.9, "good": 0.7, "fair": 0.5, "poor": 0.3}

    def explain_overall_quality(self, score: float, grade: str) -> str:
        """総合品質説明"""
        if score >= self.quality_thresholds["excellent"]:
            level_desc = "極めて高品質"
            detail = "全ての品質要因が優秀な水準を達成しています。"
        elif score >= self.quality_thresholds["good"]:
            level_desc = "高品質"
            detail = "大部分の品質要因が良好な水準にあります。"
        elif score >= self.quality_thresholds["fair"]:
            level_desc = "中程度品質"
            detail = "一部の品質要因に改善の余地があります。"
        elif score >= self.quality_thresholds["poor"]:
            level_desc = "低品質"
            detail = "複数の品質要因で大幅な改善が必要です。"
        else:
            level_desc = "極めて低品質"
            detail = "ほぼ全ての品質要因で抜本的な改善が必要です。"

        return f"総合品質は{level_desc}（スコア: {score:.3f}, グレード: {grade}）です。{detail}"

    def explain_factor_impact(self, factors: List[QualityFactor]) -> Dict[str, str]:
        """要因影響説明"""
        explanations = {}

        # 重要度でソート
        sorted_factors = sorted(factors, key=lambda f: f.importance, reverse=True)

        for i, factor in enumerate(sorted_factors[:5]):  # 上位5要因
            rank = i + 1
            impact_level = (
                "高" if factor.importance > 0.7 else "中" if factor.importance > 0.4 else "低"
            )
            quality_level = "高" if factor.value > 0.7 else "中" if factor.value > 0.4 else "低"

            explanation = f"""
第{rank}位の重要要因: {factor.name}
• 品質スコア: {factor.value:.3f} ({quality_level}品質)
• 重要度: {factor.importance:.3f} ({impact_level}影響)
• 説明: {factor.description}
"""

            if factor.improvement_suggestion:
                explanation += f"• 改善提案: {factor.improvement_suggestion}"

            explanations[factor.name] = explanation.strip()

        return explanations

    def generate_improvement_plan(self, factors: List[QualityFactor]) -> List[str]:
        """改善計画生成"""
        plan = []

        # 低品質要因を特定
        low_quality_factors = [f for f in factors if f.value < 0.5]

        if not low_quality_factors:
            plan.append("✅ 現在の品質は良好です。定期的な監視を継続してください。")
            return plan

        # 重要度順でソート
        low_quality_factors.sort(key=lambda f: f.importance, reverse=True)

        plan.append(f"🎯 優先改善項目（{len(low_quality_factors)}件）:")

        for i, factor in enumerate(low_quality_factors[:3], 1):  # 上位3件
            priority = "高" if factor.importance > 0.7 else "中" if factor.importance > 0.4 else "低"
            expected_impact = factor.importance * (1.0 - factor.value)  # 改善期待値

            plan.append(
                f"""
{i}. {factor.name} (優先度: {priority})
   現在スコア: {factor.value:.3f}
   期待改善効果: {expected_impact:.3f}
   具体的改善案: {factor.improvement_suggestion or '詳細分析が必要です'}
""".strip()
            )

        # 総合改善提案
        if len(low_quality_factors) > 3:
            plan.append(f"\n📋 その他の改善項目が{len(low_quality_factors) - 3}件あります。")

        return plan

    def create_confidence_explanation(self, confidence: float, factors: List[QualityFactor]) -> str:
        """信頼性説明"""
        if confidence >= 0.9:
            level = "極めて高い"
            detail = "評価結果の信頼性は非常に高く、改善提案に従うことを強く推奨します。"
        elif confidence >= 0.7:
            level = "高い"
            detail = "評価結果の信頼性は高く、改善提案は有効と考えられます。"
        elif confidence >= 0.5:
            level = "中程度"
            detail = "評価結果には一定の信頼性がありますが、追加の検証を推奨します。"
        else:
            level = "低い"
            detail = "評価結果の信頼性が低いため、手動での再確認が必要です。"

        # 信頼性に影響する要因
        high_conf_factors = [f.name for f in factors if getattr(f, "confidence", 0.8) > 0.8]
        low_conf_factors = [f.name for f in factors if getattr(f, "confidence", 0.8) < 0.5]

        detail += (
            f"\\n\\n信頼性の高い要因: {', '.join(high_conf_factors[:3]) if high_conf_factors else 'なし'}"
        )
        if low_conf_factors:
            detail += f"\\n不確実な要因: {', '.join(low_conf_factors[:2])} - これらの要因は手動確認を推奨"

        return f"評価の信頼性: {level}（{confidence:.3f}）\\n{detail}"


class ExplainableQualityEvaluator:
    """説明可能品質評価器"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("explanations")
        self.visual_explainer = VisualExplainer(self.output_dir / "visuals")
        self.textual_explainer = TextualExplainer()
        self.evaluation_history = []

    def evaluate_with_explanation(
        self,
        item_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        quality_scores: Dict[str, float],
        bbox: Optional[Tuple] = None,
    ) -> ExplanationResult:
        """説明付き品質評価"""
        logger.info(f"🔍 説明可能品質評価開始: {item_id}")

        result = ExplanationResult(item_id=item_id, overall_score=0.0, overall_grade="F")

        try:
            # 品質要因分析
            factors = self._analyze_quality_factors(image, mask, quality_scores, bbox)
            result.factors = factors

            # 総合スコア計算
            result.overall_score = self._calculate_weighted_score(factors)
            result.overall_grade = self._score_to_grade(result.overall_score)

            # 信頼性計算
            result.confidence = self._calculate_confidence(factors)

            # テキスト説明生成
            result.explanations = self._generate_text_explanations(result)

            # ビジュアル説明生成
            result.visual_paths = self._generate_visual_explanations(result, image, mask)

            # 改善提案生成
            result.recommendations = self._generate_detailed_recommendations(factors)

            # 履歴に追加
            self.evaluation_history.append(result)

            logger.info(
                f"✅ 説明可能品質評価完了: {item_id} (スコア: {result.overall_score:.3f}, 信頼性: {result.confidence:.3f})"
            )

        except Exception as e:
            logger.error(f"❌ 説明可能品質評価エラー: {item_id} - {e}")
            result.explanations["error"] = f"評価中にエラーが発生しました: {e}"

        return result

    def _analyze_quality_factors(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        quality_scores: Dict[str, float],
        bbox: Optional[Tuple],
    ) -> List[QualityFactor]:
        """品質要因分析"""
        factors = []

        # カバレッジ要因
        coverage_score = quality_scores.get("coverage", 0.0)
        factors.append(
            QualityFactor(
                name="マスクカバレッジ",
                value=coverage_score,
                importance=0.8,
                description=f"抽出マスクが対象物をどの程度カバーしているかを示します（{coverage_score:.3f}）",
                improvement_suggestion="マスクの境界を調整し、対象物の完全な輪郭を確保してください",
            )
        )

        # エッジ精度要因
        edge_score = quality_scores.get("edge_accuracy", 0.0)
        factors.append(
            QualityFactor(
                name="エッジ精度",
                value=edge_score,
                importance=0.7,
                description=f"マスクの境界線が実際のオブジェクトエッジとどの程度一致しているかを示します（{edge_score:.3f}）",
                improvement_suggestion="セグメンテーション閾値を調整し、より正確な境界抽出を行ってください",
            )
        )

        # 明瞭性要因
        clarity_score = quality_scores.get("clarity", 0.0)
        factors.append(
            QualityFactor(
                name="画像明瞭性",
                value=clarity_score,
                importance=0.6,
                description=f"抽出された領域の画像品質（明度・コントラスト）を示します（{clarity_score:.3f}）",
                improvement_suggestion="画像の明度調整やコントラスト向上を検討してください",
            )
        )

        # サイズ妥当性要因
        size_score = quality_scores.get("size_relevance", 0.0)
        factors.append(
            QualityFactor(
                name="サイズ妥当性",
                value=size_score,
                importance=0.5,
                description=f"抽出されたオブジェクトのサイズが適切かどうかを示します（{size_score:.3f}）",
                improvement_suggestion="検出感度を調整し、適切なサイズのオブジェクトを抽出してください",
            )
        )

        # 位置妥当性要因
        position_score = quality_scores.get("position_relevance", 0.0)
        factors.append(
            QualityFactor(
                name="位置妥当性",
                value=position_score,
                importance=0.4,
                description=f"抽出されたオブジェクトの位置が適切かどうかを示します（{position_score:.3f}）",
                improvement_suggestion="構図や画像のトリミングを調整してください",
            )
        )

        return factors

    def _calculate_weighted_score(self, factors: List[QualityFactor]) -> float:
        """重み付きスコア計算"""
        if not factors:
            return 0.0

        total_weight = sum(f.importance for f in factors)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(f.value * f.importance for f in factors)
        return weighted_sum / total_weight

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

    def _calculate_confidence(self, factors: List[QualityFactor]) -> float:
        """信頼性計算"""
        if not factors:
            return 0.0

        # 要因の一貫性から信頼性を計算
        values = [f.value for f in factors]
        mean_value = np.mean(values)
        std_value = np.std(values)

        # 標準偏差が小さいほど一貫性が高い
        consistency = max(0.0, 1.0 - std_value)

        # 重要度の高い要因の信頼性も考慮
        high_importance_factors = [f for f in factors if f.importance > 0.6]
        importance_reliability = len(high_importance_factors) / len(factors)

        return (consistency + importance_reliability) / 2

    def _generate_text_explanations(self, result: ExplanationResult) -> Dict[str, str]:
        """テキスト説明生成"""
        explanations = {}

        # 総合品質説明
        explanations["overall"] = self.textual_explainer.explain_overall_quality(
            result.overall_score, result.overall_grade
        )

        # 要因別説明
        factor_explanations = self.textual_explainer.explain_factor_impact(result.factors)
        explanations.update(factor_explanations)

        # 信頼性説明
        explanations["confidence"] = self.textual_explainer.create_confidence_explanation(
            result.confidence, result.factors
        )

        return explanations

    def _generate_visual_explanations(
        self, result: ExplanationResult, image: np.ndarray, mask: np.ndarray
    ) -> Dict[str, str]:
        """ビジュアル説明生成"""
        visual_paths = {}

        try:
            # 品質分解チャート
            breakdown_path = self.visual_explainer.create_quality_breakdown_chart(
                result.item_id, result.factors
            )
            visual_paths["breakdown_chart"] = breakdown_path

            # 画像アノテーション
            annotation_path = self.visual_explainer.create_image_annotation(
                result.item_id, image, mask, result.factors
            )
            visual_paths["image_annotation"] = annotation_path

        except Exception as e:
            logger.warning(f"ビジュアル説明生成エラー: {e}")

        return visual_paths

    def _generate_detailed_recommendations(self, factors: List[QualityFactor]) -> List[str]:
        """詳細改善提案生成"""
        return self.textual_explainer.generate_improvement_plan(factors)

    def create_batch_comparison(self, results: List[ExplanationResult]) -> str:
        """バッチ比較レポート作成"""
        try:
            comparison_path = self.visual_explainer.create_comparison_chart(results)
            return comparison_path
        except Exception as e:
            logger.error(f"バッチ比較レポート作成エラー: {e}")
            return ""

    def save_explanations(self, output_path: Path):
        """説明結果保存"""
        output_path.mkdir(parents=True, exist_ok=True)

        # 個別説明保存
        for result in self.evaluation_history:
            result_file = output_path / f"{result.item_id}_explanation.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        # バッチ比較レポート
        if len(self.evaluation_history) > 1:
            comparison_path = self.create_batch_comparison(self.evaluation_history)
            if comparison_path:
                logger.info(f"📈 バッチ比較レポート: {comparison_path}")

        # 統合レポート
        summary_file = output_path / "explanation_summary.json"
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "average_score": np.mean([r.overall_score for r in self.evaluation_history])
            if self.evaluation_history
            else 0.0,
            "average_confidence": np.mean([r.confidence for r in self.evaluation_history])
            if self.evaluation_history
            else 0.0,
            "grade_distribution": self._get_grade_distribution(),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"📄 説明可能品質評価結果保存完了: {output_path}")

    def _get_grade_distribution(self) -> Dict[str, int]:
        """グレード分布取得"""
        grades = [r.overall_grade for r in self.evaluation_history]
        distribution = {}
        for grade in ["A", "B", "C", "D", "E", "F"]:
            distribution[grade] = grades.count(grade)
        return distribution


# 使いやすいインターフェース関数
def explain_quality_evaluation(
    image: np.ndarray,
    mask: np.ndarray,
    quality_scores: Dict[str, float],
    item_id: str = "unknown",
    output_dir: Optional[Path] = None,
) -> ExplanationResult:
    """説明可能品質評価のシンプルインターフェース"""
    evaluator = ExplainableQualityEvaluator(output_dir)
    return evaluator.evaluate_with_explanation(item_id, image, mask, quality_scores)
