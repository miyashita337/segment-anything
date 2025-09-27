"""
KIRO-012: 判定処理モジュール単体テスト

各判定モジュールの単体テスト
90%以上のカバレッジを目標とした包括的テスト
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import tempfile
import os

from features.judgment_modules.module_interfaces import (
    JudgmentInput,
    JudgmentResult,
    QualityGrade,
    ModuleRegistry
)
from features.judgment_modules.quality_assessment_module import QualityAssessmentModule
from features.judgment_modules.confidence_evaluation_module import ConfidenceEvaluationModule
from features.judgment_modules.size_validation_module import SizeValidationModule
from features.judgment_modules.fullbody_detection_module import FullbodyDetectionModule
from features.judgment_modules.central_positioning_module import CentralPositioningModule


class TestModuleInterfaces:
    """共通インターフェースのテスト"""

    def test_quality_grade_enum(self):
        """QualityGrade enum の正常性テスト"""
        assert QualityGrade.A.value == "A"
        assert QualityGrade.B.value == "B"
        assert QualityGrade.C.value == "C"
        assert QualityGrade.D.value == "D"
        assert QualityGrade.E.value == "E"
        assert QualityGrade.F.value == "F"

    def test_judgment_input_creation(self):
        """JudgmentInput の作成テスト"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        metadata = {"test": "value"}

        input_data = JudgmentInput(
            image=image,
            mask=mask,
            metadata=metadata
        )

        assert np.array_equal(input_data.image, image)
        assert np.array_equal(input_data.mask, mask)
        assert input_data.metadata == metadata

    def test_judgment_result_creation(self):
        """JudgmentResult の作成テスト"""
        result = JudgmentResult(
            quality_grade=QualityGrade.A,
            confidence_score=0.9,
            numeric_score=0.85,
            issues=["test issue"],
            recommendations=["test recommendation"],
            metrics={"test": 0.5},
            processing_time=0.1,
            module_version="1.0.0"
        )

        assert result.quality_grade == QualityGrade.A
        assert result.confidence_score == 0.9
        assert result.numeric_score == 0.85
        assert result.issues == ["test issue"]
        assert result.recommendations == ["test recommendation"]
        assert result.metrics == {"test": 0.5}
        assert result.processing_time == 0.1
        assert result.module_version == "1.0.0"

    def test_module_registry(self):
        """ModuleRegistry のテスト"""
        registry = ModuleRegistry()
        mock_module = Mock()

        # 登録テスト
        registry.register_module("test_module", mock_module)
        assert registry.get_module("test_module") == mock_module

        # 実行順序テスト
        assert "test_module" in registry.get_execution_order()

        # 登録解除テスト
        assert registry.unregister_module("test_module") is True
        assert registry.get_module("test_module") is None


class TestQualityAssessmentModule:
    """品質評価モジュールのテスト"""

    @pytest.fixture
    def module(self):
        """テスト用モジュールインスタンス"""
        return QualityAssessmentModule()

    @pytest.fixture
    def sample_image(self):
        """サンプル画像作成"""
        # 白い背景に黒い円
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(image, (100, 100), 50, (0, 0, 0), -1)
        return image

    @pytest.fixture
    def sample_input(self, sample_image):
        """サンプル入力データ"""
        return JudgmentInput(image=sample_image)

    def test_module_initialization(self, module):
        """モジュール初期化テスト"""
        assert module.module_name == "QualityAssessment"
        assert module.version == "1.0.0"
        assert isinstance(module.get_thresholds(), dict)

    def test_input_validation(self, module):
        """入力検証テスト"""
        # 正常なケース
        valid_input = JudgmentInput(image=np.zeros((100, 100, 3), dtype=np.uint8))
        assert module.validate_input(valid_input) is True

        # 異常なケース
        invalid_input = JudgmentInput(image=None)
        assert module.validate_input(invalid_input) is False

        # 次元が不正なケース
        invalid_dim = JudgmentInput(image=np.zeros((100,), dtype=np.uint8))
        assert module.validate_input(invalid_dim) is False

    def test_threshold_operations(self, module):
        """閾値操作テスト"""
        original_thresholds = module.get_thresholds()

        # 閾値更新テスト
        new_thresholds = {"grade_a_threshold": 0.95}
        assert module.update_thresholds(new_thresholds) is True

        updated_thresholds = module.get_thresholds()
        assert updated_thresholds["grade_a_threshold"] == 0.95

        # 不正な閾値更新テスト
        invalid_thresholds = {"invalid_key": 0.5}
        assert module.update_thresholds(invalid_thresholds) is False

    def test_judge_execution(self, module, sample_input):
        """判定実行テスト"""
        result = module.judge(sample_input)

        # 結果の型チェック
        assert isinstance(result, JudgmentResult)
        assert isinstance(result.quality_grade, QualityGrade)
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.numeric_score <= 1.0
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.metrics, dict)

        # メトリクスの存在チェック
        expected_metrics = ['completeness', 'clarity', 'size_adequacy',
                          'shape_quality', 'detail_preservation', 'overall_score']
        for metric in expected_metrics:
            assert metric in result.metrics

    def test_error_handling(self, module):
        """エラーハンドリングテスト"""
        # 無効な入力でのエラーハンドリング
        invalid_input = JudgmentInput(image=None)
        result = module.judge(invalid_input)

        assert result.quality_grade == QualityGrade.F
        assert result.confidence_score == 0.0
        assert len(result.issues) > 0


class TestConfidenceEvaluationModule:
    """信頼度評価モジュールのテスト"""

    @pytest.fixture
    def module(self):
        return ConfidenceEvaluationModule()

    @pytest.fixture
    def sample_input_with_metadata(self):
        """メタデータ付きサンプル入力"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        metadata = {
            "sam_iou_prediction": 0.8,
            "sam_stability_score": 0.7,
            "multiscale_consistency": 0.6,
            "parameter_stability": 0.65,
            "noise_robustness": 0.7
        }
        return JudgmentInput(image=image, mask=mask, metadata=metadata)

    def test_sam_confidence_evaluation(self, module, sample_input_with_metadata):
        """SAM信頼度評価テスト"""
        result = module.judge(sample_input_with_metadata)

        assert isinstance(result, JudgmentResult)
        assert 'sam_confidence' in result.metrics
        assert 0.0 <= result.metrics['sam_confidence'] <= 1.0

    def test_mask_consistency_evaluation(self, module):
        """マスク一貫性評価テスト"""
        # 良質なマスクの場合
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 1, -1)  # 単一の連結成分

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        assert 'mask_consistency' in result.metrics
        assert result.metrics['mask_consistency'] > 0.5

    def test_no_mask_handling(self, module):
        """マスクなしケースのハンドリングテスト"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        input_data = JudgmentInput(image=image, mask=None)

        result = module.judge(input_data)
        assert isinstance(result, JudgmentResult)
        assert "No mask provided" in ' '.join(result.issues)


class TestSizeValidationModule:
    """サイズ検証モジュールのテスト"""

    @pytest.fixture
    def module(self):
        return SizeValidationModule()

    def test_character_dimensions_analysis(self, module):
        """キャラクター寸法分析テスト"""
        # テスト画像作成
        image = np.ones((200, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 100), dtype=np.uint8)
        cv2.rectangle(mask, (25, 25), (75, 175), 1, -1)  # 縦長の矩形

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        # アスペクト比のチェック
        assert 'character_aspect_ratio' in result.metrics
        assert result.metrics['character_aspect_ratio'] > 1.0  # 縦長

    def test_size_consistency_evaluation(self, module):
        """サイズ一貫性評価テスト"""
        # 極小キャラクター
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        small_mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(small_mask, (25, 25), 5, 1, -1)

        small_input = JudgmentInput(image=small_image, mask=small_mask)
        small_result = module.judge(small_input)

        # 小さすぎる警告があることを確認
        issues_text = ' '.join(small_result.issues)
        assert "too small" in issues_text.lower()

    def test_positioning_evaluation(self, module):
        """配置評価テスト"""
        # 中央配置
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 30, 1, -1)  # 中央の円

        center_input = JudgmentInput(image=image, mask=mask)
        center_result = module.judge(center_input)

        # 配置スコアが高いことを確認
        assert center_result.metrics['positioning_score'] > 0.7


class TestFullbodyDetectionModule:
    """全身検出モジュールのテスト"""

    @pytest.fixture
    def module(self):
        return FullbodyDetectionModule()

    def test_body_parts_analysis(self, module):
        """身体部位分析テスト"""
        # 縦長の全身像を模擬
        image = np.ones((300, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((300, 100), dtype=np.uint8)

        # 頭部、胴体、脚部を模擬
        cv2.circle(mask, (50, 30), 15, 1, -1)  # 頭部
        cv2.rectangle(mask, (35, 50), (65, 150), 1, -1)  # 胴体
        cv2.rectangle(mask, (40, 150), (60, 280), 1, -1)  # 脚部

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        # 身体部位スコアの確認
        assert 'head_region_score' in result.metrics
        assert 'torso_region_score' in result.metrics
        assert 'limbs_region_score' in result.metrics

    def test_aspect_ratio_validation(self, module):
        """アスペクト比検証テスト"""
        # 正常な縦横比の全身像
        image = np.ones((200, 100, 3), dtype=np.uint8) * 255
        mask = np.ones((200, 100), dtype=np.uint8)

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        assert 'aspect_ratio' in result.metrics
        assert result.metrics['aspect_ratio'] == 2.0


class TestCentralPositioningModule:
    """中央配置判定モジュールのテスト"""

    @pytest.fixture
    def module(self):
        return CentralPositioningModule()

    def test_perfect_center_positioning(self, module):
        """完璧な中央配置テスト"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 1, -1)  # 完璧に中央

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        assert result.metrics['center_alignment_score'] > 0.9

    def test_off_center_positioning(self, module):
        """中央から外れた配置テスト"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, -1)  # 左上に偏心

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        assert result.metrics['center_alignment_score'] < 0.8
        # 中央から外れている警告があることを確認
        issues_text = ' '.join(result.issues)
        assert "off-center" in issues_text.lower()

    def test_margin_balance_evaluation(self, module):
        """マージンバランス評価テスト"""
        # マージンが不均等な場合
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 50), (80, 150), 1, -1)  # 左寄り

        input_data = JudgmentInput(image=image, mask=mask)
        result = module.judge(input_data)

        # マージン不均衡の警告があることを確認
        issues_text = ' '.join(result.issues)
        assert ("margin" in issues_text.lower() or
                "imbalance" in issues_text.lower())


class TestIntegrationScenarios:
    """統合シナリオテスト"""

    def test_all_modules_with_same_input(self):
        """全モジュールで同一入力のテスト"""
        # 標準的なテスト画像
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 1, -1)

        metadata = {
            "sam_iou_prediction": 0.8,
            "sam_stability_score": 0.75
        }

        input_data = JudgmentInput(image=image, mask=mask, metadata=metadata)

        # 全モジュールで実行
        modules = [
            QualityAssessmentModule(),
            ConfidenceEvaluationModule(),
            SizeValidationModule(),
            FullbodyDetectionModule(),
            CentralPositioningModule()
        ]

        results = []
        for module in modules:
            result = module.judge(input_data)
            results.append(result)
            assert isinstance(result, JudgmentResult)

        # 全モジュールが何らかの結果を返すことを確認
        assert len(results) == 5
        assert all(r.processing_time >= 0 for r in results)

    def test_error_propagation(self):
        """エラー伝播テスト"""
        # 意図的に不正な入力
        invalid_input = JudgmentInput(image=np.array([]))

        module = QualityAssessmentModule()
        result = module.judge(invalid_input)

        assert result.quality_grade == QualityGrade.F
        assert len(result.issues) > 0
        assert result.confidence_score == 0.0

    def test_performance_characteristics(self):
        """パフォーマンス特性テスト"""
        # 大きな画像でのテスト
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        input_data = JudgmentInput(image=large_image)

        module = QualityAssessmentModule()
        result = module.judge(input_data)

        # 実行時間が合理的範囲内であることを確認
        assert result.processing_time < 10.0  # 10秒以内
        assert isinstance(result, JudgmentResult)


@pytest.fixture
def temp_config_file():
    """一時設定ファイル"""
    config_content = """
global:
  version: "1.0.0"

quality_assessment:
  enabled: true
  thresholds:
    grade_a_threshold: 0.85
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # クリーンアップ
    os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=features.judgment_modules", "--cov-report=html"])