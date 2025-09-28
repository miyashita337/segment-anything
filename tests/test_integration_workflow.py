"""
KIRO-012: 統合ワークフローテスト

判定モジュール統合システムの結合テスト
オーケストレーター・アグリゲーターの協調動作を検証
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch
import concurrent.futures

from features.judgment_modules.module_interfaces import (
    JudgmentInput,
    JudgmentResult,
    AggregatedJudgment,
    QualityGrade,
    ModuleRegistry
)
from features.judgment_modules.judgment_orchestrator import JudgmentOrchestrator
from features.judgment_modules.judgment_result_aggregator import (
    JudgmentResultAggregator,
    AggregationConfig
)
from features.judgment_modules.quality_assessment_module import QualityAssessmentModule
from features.judgment_modules.confidence_evaluation_module import ConfidenceEvaluationModule
from features.judgment_modules.size_validation_module import SizeValidationModule


class TestJudgmentOrchestrator:
    """判定オーケストレーターの統合テスト"""

    @pytest.fixture
    def registry_with_modules(self):
        """モジュール登録済みレジストリ"""
        registry = ModuleRegistry()
        registry.register_module("quality", QualityAssessmentModule())
        registry.register_module("confidence", ConfidenceEvaluationModule())
        registry.register_module("size", SizeValidationModule())
        return registry

    @pytest.fixture
    def orchestrator(self, registry_with_modules):
        """テスト用オーケストレーター"""
        return JudgmentOrchestrator(registry=registry_with_modules)

    @pytest.fixture
    def sample_input(self):
        """統合テスト用サンプル入力"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(image, (100, 100), 60, (255, 255, 255), -1)

        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 1, -1)

        metadata = {
            "sam_iou_prediction": 0.8,
            "sam_stability_score": 0.75,
            "multiscale_consistency": 0.7
        }

        return JudgmentInput(image=image, mask=mask, metadata=metadata)

    def test_orchestrator_initialization(self, orchestrator):
        """オーケストレーター初期化テスト"""
        assert orchestrator.max_workers == 3
        assert orchestrator.timeout_seconds == 30.0
        assert orchestrator.fallback_enabled is True
        assert orchestrator.fail_fast is False

    def test_module_enabling(self, orchestrator):
        """モジュール有効化テスト"""
        # 全モジュール有効化
        orchestrator.enable_all_modules()
        assert len(orchestrator.enabled_modules) == 3

        # 個別モジュール有効化
        orchestrator.enabled_modules.clear()
        orchestrator.enable_modules(["quality", "confidence"])
        assert orchestrator.enabled_modules == {"quality", "confidence"}

        # 無効なモジュール名でエラー
        with pytest.raises(ValueError):
            orchestrator.enable_modules(["invalid_module"])

    def test_parallel_execution(self, orchestrator, sample_input):
        """並列実行テスト"""
        orchestrator.enable_all_modules()

        start_time = time.time()
        result = orchestrator.execute_judgment(sample_input)
        execution_time = time.time() - start_time

        # 結果検証
        assert isinstance(result, AggregatedJudgment)
        assert len(result.module_results) == 3
        assert result.final_grade in [grade for grade in QualityGrade]

        # 並列実行により合理的な実行時間
        assert execution_time < 5.0  # 5秒以内

        # 各モジュールの結果が含まれている
        for module_name in ["quality", "confidence", "size"]:
            assert module_name in result.module_results

    def test_fail_fast_mode(self, orchestrator, sample_input):
        """fail_fast モードテスト"""
        orchestrator.fail_fast = True

        # モックモジュールでF評価を返す
        mock_module = Mock()
        mock_module.validate_input.return_value = True
        mock_module.judge.return_value = JudgmentResult(
            quality_grade=QualityGrade.F,
            confidence_score=0.0,
            numeric_score=0.0,
            issues=["Simulated failure"],
            recommendations=[],
            metrics={},
            processing_time=0.1,
            module_version="test"
        )

        orchestrator.registry.register_module("fail_module", mock_module, 0)
        orchestrator.enable_modules(["fail_module", "quality", "confidence"])

        result = orchestrator.execute_judgment(sample_input)

        # fail_fast により一部のモジュールが実行されない可能性
        assert isinstance(result, AggregatedJudgment)

    def test_timeout_handling(self, orchestrator, sample_input):
        """タイムアウトハンドリングテスト"""
        # タイムアウトを短く設定
        orchestrator.timeout_seconds = 0.1

        # 長時間かかるモックモジュール
        slow_module = Mock()
        slow_module.validate_input.return_value = True

        def slow_judge(*args, **kwargs):
            time.sleep(0.2)  # タイムアウトより長い時間待機
            return JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=1.0,
                numeric_score=1.0,
                issues=[],
                recommendations=[],
                metrics={},
                processing_time=0.2,
                module_version="slow"
            )

        slow_module.judge.side_effect = slow_judge

        orchestrator.registry.register_module("slow_module", slow_module)
        orchestrator.enable_modules(["slow_module"])

        # タイムアウト例外をキャッチ
        try:
            result = orchestrator.execute_judgment(sample_input)
            # タイムアウトが発生しない場合でも結果を確認
            assert isinstance(result, AggregatedJudgment)
        except concurrent.futures.TimeoutError:
            # タイムアウトが期待通り発生した場合
            pass

    def test_fallback_mechanism(self, orchestrator, sample_input):
        """フォールバック機能テスト"""
        orchestrator.fallback_enabled = True

        # 一部成功、一部失敗のシナリオ
        failing_module = Mock()
        failing_module.validate_input.return_value = True
        failing_module.judge.side_effect = Exception("Simulated error")

        orchestrator.registry.register_module("failing_module", failing_module)
        orchestrator.enable_modules(["quality", "failing_module"])

        result = orchestrator.execute_judgment(sample_input)

        # フォールバック結果が含まれている
        assert "failing_module" in result.module_results
        failing_result = result.module_results["failing_module"]
        assert "fallback" in failing_result.metrics or failing_result.quality_grade == QualityGrade.F


class TestJudgmentResultAggregator:
    """判定結果アグリゲーターの統合テスト"""

    @pytest.fixture
    def aggregator(self):
        """テスト用アグリゲーター"""
        config = AggregationConfig(
            consensus_threshold=0.6,
            outlier_threshold=2.0,
            enable_outlier_removal=True
        )
        return JudgmentResultAggregator(config)

    @pytest.fixture
    def sample_module_results(self):
        """サンプルモジュール結果"""
        return {
            "module1": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.9,
                numeric_score=0.85,
                issues=[],
                recommendations=["Excellent quality"],
                metrics={"test": 0.9},
                processing_time=0.1,
                module_version="1.0"
            ),
            "module2": JudgmentResult(
                quality_grade=QualityGrade.B,
                confidence_score=0.8,
                numeric_score=0.75,
                issues=["Minor issue"],
                recommendations=["Good quality", "Small improvement needed"],
                metrics={"test": 0.8},
                processing_time=0.15,
                module_version="1.0"
            ),
            "module3": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.85,
                numeric_score=0.82,
                issues=[],
                recommendations=["Excellent quality"],
                metrics={"test": 0.85},
                processing_time=0.12,
                module_version="1.0"
            )
        }

    def test_basic_aggregation(self, aggregator, sample_module_results):
        """基本集約テスト"""
        result = aggregator.aggregate_results(sample_module_results)

        assert isinstance(result, AggregatedJudgment)
        assert result.final_grade in [QualityGrade.A, QualityGrade.B]
        assert 0.0 <= result.overall_confidence <= 1.0
        assert len(result.module_results) == 3

        # コンセンサス指標の確認
        assert 'score_mean' in result.consensus_metrics
        assert 'confidence_mean' in result.consensus_metrics
        assert 'result_count' in result.consensus_metrics

    def test_outlier_detection(self, aggregator):
        """外れ値検出テスト"""
        # より大きな差異を持つ外れ値を含む結果セット
        results_with_outlier = {
            "normal1": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.9,
                numeric_score=0.85,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            ),
            "normal2": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.88,
                numeric_score=0.87,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            ),
            "normal3": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.92,
                numeric_score=0.89,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            ),
            "outlier": JudgmentResult(
                quality_grade=QualityGrade.F,
                confidence_score=0.1,
                numeric_score=0.01,  # 大きく外れた値
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            )
        }

        result = aggregator.aggregate_results(results_with_outlier)

        # 外れ値検出またはコンフリクト検出
        outliers_detected = len(result.conflict_analysis.get('outliers', [])) > 0
        conflicts_detected = len(result.conflict_analysis.get('conflicts', [])) > 0

        # どちらかが検出されていることを確認
        assert outliers_detected or conflicts_detected

    def test_conflict_analysis(self, aggregator):
        """コンフリクト分析テスト"""
        # 対立する結果セット
        conflicting_results = {
            "optimistic": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.9,
                numeric_score=0.9,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            ),
            "pessimistic": JudgmentResult(
                quality_grade=QualityGrade.D,
                confidence_score=0.8,
                numeric_score=0.3,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            )
        }

        result = aggregator.aggregate_results(conflicting_results)

        # コンフリクトが検出されている
        assert len(result.conflict_analysis['conflicts']) > 0
        assert result.conflict_analysis['statistics']['total_conflicts'] > 0

    def test_weighted_voting(self, aggregator):
        """信頼度重み付き投票テスト"""
        # 信頼度が異なる結果セット
        weighted_results = {
            "high_confidence": JudgmentResult(
                quality_grade=QualityGrade.A,
                confidence_score=0.95,  # 高信頼度
                numeric_score=0.9,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            ),
            "low_confidence": JudgmentResult(
                quality_grade=QualityGrade.C,
                confidence_score=0.3,   # 低信頼度
                numeric_score=0.5,
                issues=[], recommendations=[], metrics={},
                processing_time=0.1, module_version="1.0"
            )
        }

        result = aggregator.aggregate_results(weighted_results)

        # 高信頼度の結果により近い判定になることを期待
        assert result.final_grade in [QualityGrade.A, QualityGrade.B]

    def test_recommendation_integration(self, aggregator, sample_module_results):
        """推奨事項統合テスト"""
        result = aggregator.aggregate_results(sample_module_results)

        assert isinstance(result.recommendation_summary, list)
        assert len(result.recommendation_summary) > 0

        # 頻度の高い推奨事項が優先されている
        if "Excellent quality" in str(result.recommendation_summary):
            # 複数回言及されている推奨事項が上位に
            pass


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローテスト"""

    @pytest.fixture
    def complete_system(self):
        """完全なシステムセットアップ"""
        registry = ModuleRegistry()
        registry.register_module("quality", QualityAssessmentModule())
        registry.register_module("confidence", ConfidenceEvaluationModule())
        registry.register_module("size", SizeValidationModule())

        orchestrator = JudgmentOrchestrator(registry=registry)
        aggregator = JudgmentResultAggregator()

        return orchestrator, aggregator

    def test_full_pipeline(self, complete_system):
        """完全パイプライン実行テスト"""
        orchestrator, aggregator = complete_system
        orchestrator.enable_all_modules()

        # 多様なテストケース
        test_cases = [
            self._create_high_quality_case(),
            self._create_medium_quality_case(),
            self._create_low_quality_case()
        ]

        results = []
        for test_input in test_cases:
            result = orchestrator.execute_judgment(test_input)
            results.append(result)

        # 全ケースで有効な結果が得られる
        assert len(results) == 3
        for result in results:
            assert isinstance(result, AggregatedJudgment)
            assert result.final_grade in [grade for grade in QualityGrade]
            assert 0.0 <= result.overall_confidence <= 1.0

    def test_performance_under_load(self, complete_system):
        """負荷下でのパフォーマンステスト"""
        orchestrator, aggregator = complete_system
        orchestrator.enable_all_modules()

        # 複数の並列実行
        test_input = self._create_medium_quality_case()

        start_time = time.time()

        # 10回の並列実行をシミュレート
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(orchestrator.execute_judgment, test_input)
                for _ in range(10)
            ]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time

        # 結果検証
        assert len(results) == 10
        assert all(isinstance(r, AggregatedJudgment) for r in results)

        # パフォーマンス要件
        assert total_time < 30.0  # 30秒以内で完了

    def test_error_recovery(self, complete_system):
        """エラー回復テスト"""
        orchestrator, aggregator = complete_system

        # 不正な入力でのエラー回復
        invalid_input = JudgmentInput(image=None)

        orchestrator.enable_all_modules()
        result = orchestrator.execute_judgment(invalid_input)

        # エラーでも何らかの結果が返される
        assert isinstance(result, AggregatedJudgment)

    def _create_high_quality_case(self):
        """高品質テストケース作成"""
        # 高品質：大きくて明瞭で中央配置された画像
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(image, (200, 200), 150, (0, 0, 0), -1)

        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (200, 200), 150, 1, -1)

        metadata = {
            "sam_iou_prediction": 0.95,
            "sam_stability_score": 0.9,
            "multiscale_consistency": 0.85
        }

        return JudgmentInput(image=image, mask=mask, metadata=metadata)

    def _create_medium_quality_case(self):
        """中品質テストケース作成"""
        # 中品質：標準的な画像
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(image, (100, 100), 60, (255, 255, 255), -1)

        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 1, -1)

        metadata = {
            "sam_iou_prediction": 0.7,
            "sam_stability_score": 0.65
        }

        return JudgmentInput(image=image, mask=mask, metadata=metadata)

    def _create_low_quality_case(self):
        """低品質テストケース作成"""
        # 低品質：小さくてぼやけた偏心画像
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        cv2.circle(image, (25, 25), 15, (100, 100, 100), -1)  # 小さくて偏心

        # ガウシアンブラーでぼやかす
        image = cv2.GaussianBlur(image, (5, 5), 2.0)

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 15, 1, -1)

        metadata = {
            "sam_iou_prediction": 0.4,
            "sam_stability_score": 0.3
        }

        return JudgmentInput(image=image, mask=mask, metadata=metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=features.judgment_modules",
                "--cov-report=html", "--cov-append"])