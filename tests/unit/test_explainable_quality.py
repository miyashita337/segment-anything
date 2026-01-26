"""
評価説明可能性システムのテスト
"""

import numpy as np
import cv2

import json
import pytest
from features.evaluation.explainable_quality import (
    ExplainableQualityEvaluator,
    ExplanationResult,
    ExplanationType,
    QualityFactor,
    TextualExplainer,
    VisualExplainer,
    explain_quality_evaluation,
)
from pathlib import Path
from unittest.mock import Mock, patch


class TestQualityFactor:
    """QualityFactorのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        factor = QualityFactor(
            name="テスト要因",
            value=0.85,
            importance=0.7,
            description="テスト説明",
            improvement_suggestion="改善提案",
        )

        assert factor.name == "テスト要因"
        assert factor.value == 0.85
        assert factor.importance == 0.7
        assert factor.description == "テスト説明"
        assert factor.improvement_suggestion == "改善提案"
        assert factor.visual_indicators == []

    def test_to_dict(self):
        """辞書変換のテスト"""
        factor = QualityFactor(
            name="テスト要因",
            value=0.75,
            importance=0.6,
            description="説明",
            visual_indicators=[{"type": "edge", "confidence": 0.8}],
        )

        result_dict = factor.to_dict()

        assert result_dict["name"] == "テスト要因"
        assert result_dict["value"] == 0.75
        assert result_dict["importance"] == 0.6
        assert len(result_dict["visual_indicators"]) == 1


class TestExplanationResult:
    """ExplanationResultのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        result = ExplanationResult(item_id="test_item", overall_score=0.8, overall_grade="B")

        assert result.item_id == "test_item"
        assert result.overall_score == 0.8
        assert result.overall_grade == "B"
        assert result.factors == []
        assert result.explanations == {}
        assert result.visual_paths == {}
        assert result.recommendations == []
        assert result.confidence == 0.0

    def test_to_dict(self):
        """辞書変換のテスト"""
        result = ExplanationResult(item_id="test_item", overall_score=0.7, overall_grade="C")

        factor = QualityFactor("要因1", 0.6, 0.8, "説明1")
        result.factors.append(factor)
        result.explanations["overall"] = "総合説明"
        result.confidence = 0.85

        result_dict = result.to_dict()

        assert result_dict["item_id"] == "test_item"
        assert result_dict["overall_score"] == 0.7
        assert result_dict["overall_grade"] == "C"
        assert len(result_dict["factors"]) == 1
        assert result_dict["explanations"]["overall"] == "総合説明"
        assert result_dict["confidence"] == 0.85


class TestTextualExplainer:
    """TextualExplainerのテスト"""

    @pytest.fixture
    def explainer(self):
        return TextualExplainer()

    def test_explain_overall_quality(self, explainer):
        """総合品質説明のテスト"""
        # 高品質
        explanation = explainer.explain_overall_quality(0.92, "A")
        assert "極めて高品質" in explanation
        assert "0.920" in explanation
        assert "A" in explanation

        # 中品質
        explanation = explainer.explain_overall_quality(0.65, "D")
        assert "中程度品質" in explanation

        # 低品質
        explanation = explainer.explain_overall_quality(0.25, "F")
        assert "極めて低品質" in explanation

    def test_explain_factor_impact(self, explainer):
        """要因影響説明のテスト"""
        factors = [
            QualityFactor("重要要因", 0.9, 0.8, "高重要度要因"),
            QualityFactor("普通要因", 0.5, 0.5, "中重要度要因"),
            QualityFactor("低重要要因", 0.3, 0.2, "低重要度要因"),
        ]

        explanations = explainer.explain_factor_impact(factors)

        assert "重要要因" in explanations
        assert "第1位の重要要因" in explanations["重要要因"]
        assert "0.900" in explanations["重要要因"]
        assert "高影響" in explanations["重要要因"]

    def test_generate_improvement_plan(self, explainer):
        """改善計画生成のテスト"""
        # 低品質要因あり
        factors = [
            QualityFactor("問題要因1", 0.3, 0.8, "説明1", "改善案1"),
            QualityFactor("問題要因2", 0.4, 0.6, "説明2", "改善案2"),
            QualityFactor("良好要因", 0.8, 0.5, "説明3"),
        ]

        plan = explainer.generate_improvement_plan(factors)

        assert len(plan) > 0
        assert "優先改善項目" in plan[0]
        assert "問題要因1" in "".join(plan)  # 重要度が高い方が優先

        # 全て良好
        good_factors = [
            QualityFactor("良好要因1", 0.8, 0.7, "説明"),
            QualityFactor("良好要因2", 0.9, 0.6, "説明"),
        ]

        plan = explainer.generate_improvement_plan(good_factors)
        assert "現在の品質は良好" in plan[0]

    def test_create_confidence_explanation(self, explainer):
        """信頼性説明のテスト"""
        factors = [QualityFactor("要因1", 0.8, 0.7, "説明1"), QualityFactor("要因2", 0.7, 0.6, "説明2")]

        # 高信頼性
        explanation = explainer.create_confidence_explanation(0.95, factors)
        assert "極めて高い" in explanation
        assert "0.950" in explanation

        # 低信頼性
        explanation = explainer.create_confidence_explanation(0.3, factors)
        assert "低い" in explanation
        assert "手動での再確認が必要" in explanation


class TestVisualExplainer:
    """VisualExplainerのテスト"""

    @pytest.fixture
    def explainer(self, tmp_path):
        return VisualExplainer(tmp_path)

    @pytest.fixture
    def sample_factors(self):
        return [
            QualityFactor("要因A", 0.9, 0.8, "高品質要因"),
            QualityFactor("要因B", 0.5, 0.6, "中品質要因"),
            QualityFactor("要因C", 0.2, 0.4, "低品質要因"),
        ]

    def test_create_quality_breakdown_chart(self, explainer, sample_factors):
        """品質分解チャート生成のテスト"""
        output_path = explainer.create_quality_breakdown_chart("test_item", sample_factors)

        assert Path(output_path).exists()
        assert "quality_breakdown.png" in output_path

    def test_create_image_annotation(self, explainer, sample_factors):
        """画像アノテーション生成のテスト"""
        # サンプル画像とマスク
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 1, -1)

        output_path = explainer.create_image_annotation("test_item", image, mask, sample_factors)

        assert Path(output_path).exists()
        assert "image_annotation.png" in output_path

    def test_create_comparison_chart(self, explainer):
        """比較チャート生成のテスト"""
        results = []
        for i in range(3):
            result = ExplanationResult(f"item_{i}", 0.5 + i * 0.2, "C")
            result.factors = [
                QualityFactor(f"要因{j}", np.random.rand(), 0.5, f"説明{j}") for j in range(3)
            ]
            results.append(result)

        output_path = explainer.create_comparison_chart(results)

        if output_path:  # 2件以上の場合のみ生成
            assert Path(output_path).exists()
            assert "comparison_heatmap.png" in output_path


class TestExplainableQualityEvaluator:
    """ExplainableQualityEvaluatorのテスト"""

    @pytest.fixture
    def evaluator(self, tmp_path):
        return ExplainableQualityEvaluator(tmp_path)

    @pytest.fixture
    def sample_data(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, 1, -1)
        quality_scores = {
            "coverage": 0.8,
            "edge_accuracy": 0.6,
            "clarity": 0.7,
            "size_relevance": 0.9,
            "position_relevance": 0.5,
        }
        return image, mask, quality_scores

    def test_evaluate_with_explanation(self, evaluator, sample_data):
        """説明付き評価のテスト"""
        image, mask, quality_scores = sample_data

        result = evaluator.evaluate_with_explanation("test_item", image, mask, quality_scores)

        assert isinstance(result, ExplanationResult)
        assert result.item_id == "test_item"
        assert len(result.factors) > 0
        assert result.overall_score > 0
        assert result.overall_grade in ["A", "B", "C", "D", "E", "F"]
        assert result.confidence > 0
        assert len(result.explanations) > 0
        assert "overall" in result.explanations

    def test_analyze_quality_factors(self, evaluator, sample_data):
        """品質要因分析のテスト"""
        image, mask, quality_scores = sample_data

        factors = evaluator._analyze_quality_factors(image, mask, quality_scores, None)

        assert len(factors) == 5  # 5つの要因
        assert all(isinstance(f, QualityFactor) for f in factors)

        # 要因名の確認
        factor_names = [f.name for f in factors]
        assert "マスクカバレッジ" in factor_names
        assert "エッジ精度" in factor_names
        assert "画像明瞭性" in factor_names

    def test_calculate_weighted_score(self, evaluator):
        """重み付きスコア計算のテスト"""
        factors = [
            QualityFactor("要因1", 0.8, 0.6, "説明1"),
            QualityFactor("要因2", 0.6, 0.3, "説明2"),
            QualityFactor("要因3", 0.9, 0.1, "説明3"),
        ]

        score = evaluator._calculate_weighted_score(factors)

        # 手動計算: (0.8*0.6 + 0.6*0.3 + 0.9*0.1) / (0.6 + 0.3 + 0.1) = 0.75
        assert abs(score - 0.75) < 0.01

    def test_calculate_confidence(self, evaluator):
        """信頼性計算のテスト"""
        # 一貫性の高い要因
        consistent_factors = [
            QualityFactor("要因1", 0.8, 0.7, "説明1"),
            QualityFactor("要因2", 0.75, 0.8, "説明2"),
            QualityFactor("要因3", 0.85, 0.6, "説明3"),
        ]

        confidence = evaluator._calculate_confidence(consistent_factors)
        assert confidence > 0.7  # 高い信頼性

        # 一貫性の低い要因
        inconsistent_factors = [
            QualityFactor("要因1", 0.9, 0.7, "説明1"),
            QualityFactor("要因2", 0.2, 0.8, "説明2"),
            QualityFactor("要因3", 0.5, 0.6, "説明3"),
        ]

        confidence = evaluator._calculate_confidence(inconsistent_factors)
        assert confidence < 0.7  # 低い信頼性

    def test_save_explanations(self, evaluator, sample_data, tmp_path):
        """説明結果保存のテスト"""
        image, mask, quality_scores = sample_data

        # 複数の評価を実行
        for i in range(3):
            evaluator.evaluate_with_explanation(f"item_{i}", image, mask, quality_scores)

        # 保存
        output_path = tmp_path / "explanations"
        evaluator.save_explanations(output_path)

        # ファイル確認
        assert (output_path / "item_0_explanation.json").exists()
        assert (output_path / "explanation_summary.json").exists()

        # サマリー内容確認
        with open(output_path / "explanation_summary.json", "r") as f:
            summary = json.load(f)

        assert summary["total_evaluations"] == 3
        assert "average_score" in summary
        assert "grade_distribution" in summary


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""

    def test_explain_quality_evaluation(self, tmp_path):
        """説明可能品質評価シンプルインターフェースのテスト"""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 15, 1, -1)

        quality_scores = {"coverage": 0.7, "edge_accuracy": 0.8, "clarity": 0.6}

        result = explain_quality_evaluation(image, mask, quality_scores, "test_simple", tmp_path)

        assert isinstance(result, ExplanationResult)
        assert result.item_id == "test_simple"
        assert len(result.factors) > 0
        assert len(result.explanations) > 0


class TestIntegration:
    """統合テスト"""

    def test_full_evaluation_pipeline(self, tmp_path):
        """完全評価パイプラインのテスト"""
        evaluator = ExplainableQualityEvaluator(tmp_path)

        # 複数のサンプルで評価
        samples = []
        for i in range(3):
            image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)

            # 異なる品質のマスクを作成
            if i == 0:  # 高品質
                cv2.circle(mask, (50, 50), 35, 1, -1)
                quality_scores = {
                    "coverage": 0.9,
                    "edge_accuracy": 0.8,
                    "clarity": 0.85,
                    "size_relevance": 0.9,
                    "position_relevance": 0.95,
                }
            elif i == 1:  # 中品質
                cv2.ellipse(mask, (50, 50), (25, 15), 0, 0, 360, 1, -1)
                quality_scores = {
                    "coverage": 0.6,
                    "edge_accuracy": 0.5,
                    "clarity": 0.7,
                    "size_relevance": 0.6,
                    "position_relevance": 0.7,
                }
            else:  # 低品質
                mask[80:90, 80:90] = 1
                quality_scores = {
                    "coverage": 0.2,
                    "edge_accuracy": 0.3,
                    "clarity": 0.4,
                    "size_relevance": 0.1,
                    "position_relevance": 0.2,
                }

            result = evaluator.evaluate_with_explanation(f"sample_{i}", image, mask, quality_scores)
            samples.append(result)

        # 結果検証
        assert len(samples) == 3

        # 品質順序の確認（高→中→低）
        assert samples[0].overall_score > samples[1].overall_score
        assert samples[1].overall_score > samples[2].overall_score

        # 保存
        evaluator.save_explanations(tmp_path / "results")

        # バッチ比較
        comparison_path = evaluator.create_batch_comparison(samples)
        if comparison_path:
            assert Path(comparison_path).exists()
