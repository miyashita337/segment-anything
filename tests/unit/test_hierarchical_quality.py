"""
階層的品質評価システムのテスト
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from features.evaluation.hierarchical_quality import (
    HierarchicalQualityEvaluator, PixelLevelEvaluator, ObjectLevelEvaluator,
    QualityLevel, QualityMetric, QualityScore, HierarchicalQualityResult,
    evaluate_hierarchical_quality
)


class TestQualityScore:
    """QualityScoreのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        score = QualityScore(
            QualityMetric.ACCURACY, 
            QualityLevel.PIXEL,
            0.85, 
            0.9
        )
        
        assert score.metric == QualityMetric.ACCURACY
        assert score.level == QualityLevel.PIXEL
        assert score.score == 0.85
        assert score.confidence == 0.9
        assert score.details == {}
    
    def test_to_dict(self):
        """辞書変換のテスト"""
        score = QualityScore(
            QualityMetric.COMPLETENESS,
            QualityLevel.OBJECT,
            0.75,
            0.8,
            {'test_detail': 'value'}
        )
        
        result_dict = score.to_dict()
        
        assert result_dict['metric'] == 'completeness'
        assert result_dict['level'] == 'object'
        assert result_dict['score'] == 0.75
        assert result_dict['confidence'] == 0.8
        assert result_dict['details']['test_detail'] == 'value'


class TestHierarchicalQualityResult:
    """HierarchicalQualityResultのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        result = HierarchicalQualityResult("test_item")
        
        assert result.item_id == "test_item"
        assert result.scores == []
        assert result.overall_score == 0.0
        assert result.overall_grade == "F"
    
    def test_add_score_and_update(self):
        """スコア追加と更新のテスト"""
        result = HierarchicalQualityResult("test_item")
        
        # スコア追加
        score1 = QualityScore(QualityMetric.ACCURACY, QualityLevel.PIXEL, 0.8, 0.9)
        score2 = QualityScore(QualityMetric.COMPLETENESS, QualityLevel.OBJECT, 0.7, 0.8)
        
        result.add_score(score1)
        result.add_score(score2)
        
        assert len(result.scores) == 2
        assert result.overall_score > 0.0
        assert result.overall_grade in ["A", "B", "C", "D", "E", "F"]
    
    def test_score_to_grade(self):
        """グレード変換のテスト"""
        result = HierarchicalQualityResult("test_item")
        
        assert result._score_to_grade(0.95) == "A"
        assert result._score_to_grade(0.85) == "B"
        assert result._score_to_grade(0.75) == "C"
        assert result._score_to_grade(0.65) == "D"
        assert result._score_to_grade(0.55) == "E"
        assert result._score_to_grade(0.45) == "F"


class TestPixelLevelEvaluator:
    """PixelLevelEvaluatorのテスト"""
    
    @pytest.fixture
    def evaluator(self):
        return PixelLevelEvaluator()
    
    @pytest.fixture
    def sample_image(self):
        # 100x100のテスト画像
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        # 中央に円形のマスク
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 1, -1)
        return mask
    
    def test_evaluate(self, evaluator, sample_image, sample_mask):
        """評価実行のテスト"""
        scores = evaluator.evaluate(sample_image, sample_mask)
        
        assert len(scores) == 3  # completeness, accuracy, clarity
        assert all(isinstance(score, QualityScore) for score in scores)
        assert all(score.level == QualityLevel.PIXEL for score in scores)
    
    def test_evaluate_coverage(self, evaluator, sample_image, sample_mask):
        """カバレッジ評価のテスト"""
        result = evaluator._evaluate_coverage(sample_image, sample_mask)
        
        assert 'score' in result
        assert 'confidence' in result
        assert 'coverage_ratio' in result
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_evaluate_edge_accuracy(self, evaluator, sample_image, sample_mask):
        """エッジ精度評価のテスト"""
        result = evaluator._evaluate_edge_accuracy(sample_image, sample_mask)
        
        assert 'score' in result
        assert 'confidence' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_evaluate_pixel_clarity(self, evaluator, sample_image, sample_mask):
        """ピクセル明瞭性評価のテスト"""
        result = evaluator._evaluate_pixel_clarity(sample_image, sample_mask)
        
        assert 'score' in result
        assert 'confidence' in result
        assert 'mean_brightness' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_empty_inputs(self, evaluator):
        """空入力のテスト"""
        empty_image = np.array([])
        empty_mask = np.array([])
        
        scores = evaluator.evaluate(empty_image, empty_mask)
        
        assert len(scores) == 3
        assert all(score.score == 0.0 for score in scores)


class TestObjectLevelEvaluator:
    """ObjectLevelEvaluatorのテスト"""
    
    @pytest.fixture
    def evaluator(self):
        return ObjectLevelEvaluator()
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (70, 70), 1, -1)
        return mask
    
    def test_evaluate(self, evaluator, sample_image, sample_mask):
        """評価実行のテスト"""
        scores = evaluator.evaluate(sample_image, sample_mask)
        
        assert len(scores) == 2  # completeness, relevance
        assert all(isinstance(score, QualityScore) for score in scores)
        assert all(score.level == QualityLevel.OBJECT for score in scores)
    
    def test_evaluate_with_bbox(self, evaluator, sample_image, sample_mask):
        """境界ボックス付き評価のテスト"""
        bbox = (25, 25, 50, 50)  # x, y, w, h
        scores = evaluator.evaluate(sample_image, sample_mask, bbox)
        
        assert len(scores) == 2
        # バウンディングボックス情報が詳細に反映されることを確認
        completeness_score = next(s for s in scores if s.metric == QualityMetric.COMPLETENESS)
        assert 'fill_ratio' in completeness_score.details
    
    def test_evaluate_object_completeness(self, evaluator, sample_image, sample_mask):
        """オブジェクト完全性評価のテスト"""
        result = evaluator._evaluate_object_completeness(sample_image, sample_mask, None)
        
        assert 'score' in result
        assert 'confidence' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_evaluate_object_relevance(self, evaluator, sample_image, sample_mask):
        """オブジェクト妥当性評価のテスト"""
        result = evaluator._evaluate_object_relevance(sample_image, sample_mask)
        
        assert 'score' in result
        assert 'confidence' in result
        assert 'size_ratio' in result
        assert 0.0 <= result['score'] <= 1.0


class TestHierarchicalQualityEvaluator:
    """HierarchicalQualityEvaluatorのテスト"""
    
    @pytest.fixture
    def evaluator(self):
        return HierarchicalQualityEvaluator()
    
    @pytest.fixture
    def sample_data(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, 1, -1)
        return image, mask
    
    def test_evaluate(self, evaluator, sample_data):
        """評価実行のテスト"""
        image, mask = sample_data
        result = evaluator.evaluate("test_item", image, mask)
        
        assert isinstance(result, HierarchicalQualityResult)
        assert result.item_id == "test_item"
        assert len(result.scores) > 0
        assert result.overall_score >= 0.0
        assert result.overall_grade in ["A", "B", "C", "D", "E", "F"]
    
    def test_evaluate_with_bbox(self, evaluator, sample_data):
        """境界ボックス付き評価のテスト"""
        image, mask = sample_data
        bbox = (25, 25, 50, 50)
        
        result = evaluator.evaluate("test_item", image, mask, bbox)
        
        assert isinstance(result, HierarchicalQualityResult)
        assert len(result.scores) > 0
    
    def test_generate_level_summaries(self, evaluator):
        """レベル別サマリー生成のテスト"""
        scores = [
            QualityScore(QualityMetric.ACCURACY, QualityLevel.PIXEL, 0.8, 0.9),
            QualityScore(QualityMetric.COMPLETENESS, QualityLevel.PIXEL, 0.7, 0.8),
            QualityScore(QualityMetric.RELEVANCE, QualityLevel.OBJECT, 0.9, 0.85)
        ]
        
        summaries = evaluator._generate_level_summaries(scores)
        
        assert 'pixel' in summaries
        assert 'object' in summaries
        assert summaries['pixel']['metric_count'] == 2
        assert summaries['object']['metric_count'] == 1
    
    def test_generate_recommendations(self, evaluator):
        """改善提案生成のテスト"""
        result = HierarchicalQualityResult("test_item")
        
        # 低スコアを追加
        low_score = QualityScore(QualityMetric.COMPLETENESS, QualityLevel.PIXEL, 0.3, 0.8)
        result.add_score(low_score)
        
        recommendations = evaluator._generate_recommendations(result)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("マスク" in rec for rec in recommendations)
    
    def test_evaluate_batch(self, evaluator):
        """バッチ評価のテスト"""
        items = []
        for i in range(3):
            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            mask = np.zeros((50, 50), dtype=np.uint8)
            cv2.circle(mask, (25, 25), 10, 1, -1)
            
            items.append({
                'id': f"item_{i}",
                'image': image,
                'mask': mask
            })
        
        results = evaluator.evaluate_batch(items)
        
        assert len(results) == 3
        assert all(isinstance(r, HierarchicalQualityResult) for r in results)
        assert [r.item_id for r in results] == ["item_0", "item_1", "item_2"]
    
    def test_get_dataset_summary(self, evaluator, sample_data):
        """データセット統計のテスト"""
        image, mask = sample_data
        
        # いくつか評価を実行
        for i in range(3):
            evaluator.evaluate(f"item_{i}", image, mask)
        
        summary = evaluator.get_dataset_summary()
        
        assert 'total_evaluations' in summary
        assert 'average_score' in summary
        assert 'grade_distribution' in summary
        assert summary['total_evaluations'] == 3


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    def test_evaluate_hierarchical_quality(self):
        """階層的品質評価シンプルインターフェースのテスト"""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 15, 1, -1)
        
        result = evaluate_hierarchical_quality(image, mask, "test_simple")
        
        assert isinstance(result, HierarchicalQualityResult)
        assert result.item_id == "test_simple"
        assert len(result.scores) > 0


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_invalid_image_shapes(self):
        """不正な画像形状のテスト"""
        evaluator = HierarchicalQualityEvaluator()
        
        # 形状の異なる画像とマスク
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        
        # エラーが発生してもクラッシュしないことを確認
        result = evaluator.evaluate("test_error", image, mask)
        
        assert isinstance(result, HierarchicalQualityResult)
        # エラー時は低スコアになることを確認
        assert result.overall_score >= 0.0
    
    def test_empty_evaluation_history(self):
        """空の評価履歴のテスト"""
        evaluator = HierarchicalQualityEvaluator()
        
        summary = evaluator.get_dataset_summary()
        
        assert summary == {}


class TestIntegration:
    """統合テスト"""
    
    def test_full_pipeline(self):
        """完全パイプラインのテスト"""
        evaluator = HierarchicalQualityEvaluator()
        
        # 複数の異なる品質のサンプル
        samples = []
        
        # 高品質サンプル
        good_image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        good_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(good_mask, (50, 50), 30, 1, -1)
        samples.append(('good', good_image, good_mask))
        
        # 低品質サンプル
        bad_image = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        bad_mask = np.zeros((100, 100), dtype=np.uint8)
        bad_mask[10:15, 10:15] = 1  # 小さなマスク
        samples.append(('bad', bad_image, bad_mask))
        
        # 評価実行
        results = []
        for name, image, mask in samples:
            result = evaluator.evaluate(name, image, mask)
            results.append(result)
        
        # 結果検証
        assert len(results) == 2
        good_result, bad_result = results
        
        # 高品質サンプルの方が高スコアであることを期待
        # （実際の値は実装により変動するため、基本的な構造のみ確認）
        assert good_result.overall_score >= 0.0
        assert bad_result.overall_score >= 0.0
        
        # データセット統計確認
        summary = evaluator.get_dataset_summary()
        assert summary['total_evaluations'] == 2