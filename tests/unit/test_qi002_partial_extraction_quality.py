"""
QI-002: 部分抽出品質テスト

このテストモジュールは、キャラクターの一部のみが抽出される問題や
抽出品質の劣化を検出・改善する機能をテストします。
"""

import numpy as np
import cv2

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# テスト対象のモジュール（実装後にインポートされる予定）
try:
    from features.evaluation.detectors.partial_extraction_detector import PartialExtractionDetector
    from features.evaluation.utils.completeness_validator import CompletenessValidator
    from features.evaluation.utils.extraction_quality_analyzer import ExtractionQualityAnalyzer
except ImportError:
    # TDD: 実装前なので ImportError は予期される
    PartialExtractionDetector = None
    ExtractionQualityAnalyzer = None
    CompletenessValidator = None


class TestPartialExtractionDetector:
    """部分抽出検出器のテストクラス"""

    @pytest.fixture
    def detector(self):
        """PartialExtractionDetector インスタンスのフィクスチャ"""
        if PartialExtractionDetector is None:
            pytest.skip("PartialExtractionDetector not yet implemented")
        return PartialExtractionDetector()

    @pytest.fixture
    def sample_partial_extractions(self):
        """テスト用部分抽出画像データの生成"""
        # 完全抽出（理想的）
        complete_extraction = np.zeros((400, 400, 3), dtype=np.uint8)
        complete_extraction[50:350, 100:300] = 120  # 頭から足まで

        # 上半身のみ抽出（頭部・胴体のみ）
        upper_body_only = np.zeros((400, 400, 3), dtype=np.uint8)
        upper_body_only[50:200, 100:300] = 120  # 上半身のみ

        # 下半身のみ抽出（脚部のみ）
        lower_body_only = np.zeros((400, 400, 3), dtype=np.uint8)
        lower_body_only[200:350, 100:300] = 120  # 下半身のみ

        # 頭部切断（首から下のみ）
        headless_extraction = np.zeros((400, 400, 3), dtype=np.uint8)
        headless_extraction[100:350, 100:300] = 120  # 頭部なし

        # 手足切断（胴体のみ）
        torso_only = np.zeros((400, 400, 3), dtype=np.uint8)
        torso_only[100:250, 130:270] = 120  # 胴体のみ

        # 断片的抽出（複数の小さな領域）
        fragmented_extraction = np.zeros((400, 400, 3), dtype=np.uint8)
        fragmented_extraction[80:120, 100:140] = 100  # 頭部断片
        fragmented_extraction[160:200, 110:150] = 110  # 胴体断片
        fragmented_extraction[240:280, 105:145] = 120  # 脚部断片

        return {
            "complete": complete_extraction,
            "upper_body_only": upper_body_only,
            "lower_body_only": lower_body_only,
            "headless": headless_extraction,
            "torso_only": torso_only,
            "fragmented": fragmented_extraction,
        }

    def test_partial_extraction_detector_initialization(self, detector):
        """PartialExtractionDetector の初期化テスト"""
        assert detector is not None
        assert hasattr(detector, "detect_partial_extraction")
        assert hasattr(detector, "completeness_threshold")
        assert hasattr(detector, "fragmentation_threshold")

    def test_detect_complete_extraction(self, detector, sample_partial_extractions):
        """完全抽出の正確な検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["complete"])

        assert result.is_partial_extraction is False
        assert result.completeness_score > 0.8
        assert result.extraction_quality == "complete"
        assert result.confidence > 0.8

    def test_detect_upper_body_partial(self, detector, sample_partial_extractions):
        """上半身部分抽出の検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["upper_body_only"])

        assert result.is_partial_extraction is True
        assert result.extraction_type == "upper_body_only"
        assert result.completeness_score < 0.7
        assert "lower_body" in result.missing_parts

    def test_detect_lower_body_partial(self, detector, sample_partial_extractions):
        """下半身部分抽出の検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["lower_body_only"])

        assert result.is_partial_extraction is True
        assert result.extraction_type == "lower_body_only"
        assert result.completeness_score < 0.7
        assert "upper_body" in result.missing_parts or "head" in result.missing_parts

    def test_detect_headless_extraction(self, detector, sample_partial_extractions):
        """頭部切断抽出の検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["headless"])

        assert result.is_partial_extraction is True
        assert result.extraction_type == "headless"
        assert "head" in result.missing_parts
        assert result.severity == "high"  # 頭部欠損は重大

    def test_detect_torso_only_extraction(self, detector, sample_partial_extractions):
        """胴体のみ抽出の検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["torso_only"])

        assert result.is_partial_extraction is True
        assert result.extraction_type == "torso_only"
        assert result.completeness_score < 0.5
        assert len(result.missing_parts) >= 2  # 頭部・手足が欠損

    def test_detect_fragmented_extraction(self, detector, sample_partial_extractions):
        """断片的抽出の検出テスト"""
        result = detector.detect_partial_extraction(sample_partial_extractions["fragmented"])

        assert result.is_partial_extraction is True
        assert result.extraction_type == "fragmented"
        assert result.fragmentation_level > 0.5
        assert result.severity == "critical"  # 断片化は最も深刻

    def test_body_part_detection_accuracy(self, detector):
        """身体部位検出精度のテスト"""
        # 明確に定義された身体部位を持つテスト画像
        test_image = np.zeros((600, 400, 3), dtype=np.uint8)

        # 頭部（上部20%）
        test_image[50:150, 150:250] = 100
        # 胴体（中部40%）
        test_image[150:350, 130:270] = 110
        # 脚部（下部40%）
        test_image[350:550, 140:260] = 120

        result = detector.detect_partial_extraction(test_image)

        # 全身が揃っているので完全抽出と判定されるはず
        assert result.is_partial_extraction is False
        assert "head" in result.detected_parts
        assert "torso" in result.detected_parts
        assert "legs" in result.detected_parts

    def test_aspect_ratio_analysis(self, detector):
        """アスペクト比による部分抽出判定テスト"""
        # 異常に横長（手足切断の可能性）
        wide_image = np.zeros((200, 600, 3), dtype=np.uint8)
        wide_image[50:150, 100:500] = 100

        result_wide = detector.detect_partial_extraction(wide_image)
        assert result_wide.aspect_ratio_warning is True

        # 異常に縦長（頭部・足部切断の可能性）
        tall_image = np.zeros((800, 200, 3), dtype=np.uint8)
        tall_image[100:700, 50:150] = 100

        result_tall = detector.detect_partial_extraction(tall_image)
        # 縦長は比較的正常な場合もある
        assert hasattr(result_tall, "aspect_ratio_analysis")


class TestExtractionQualityAnalyzer:
    """抽出品質分析器のテストクラス"""

    @pytest.fixture
    def analyzer(self):
        """ExtractionQualityAnalyzer インスタンスのフィクスチャ"""
        if ExtractionQualityAnalyzer is None:
            pytest.skip("ExtractionQualityAnalyzer not yet implemented")
        return ExtractionQualityAnalyzer()

    def test_quality_analyzer_initialization(self, analyzer):
        """ExtractionQualityAnalyzer の初期化テスト"""
        assert analyzer is not None
        assert hasattr(analyzer, "analyze_extraction_quality")
        assert hasattr(analyzer, "calculate_completeness_metrics")

    def test_analyze_high_quality_extraction(self, analyzer):
        """高品質抽出の分析テスト"""
        # 高品質な抽出画像（明確な輪郭、適切なサイズ、完全性）
        high_quality_image = np.zeros((500, 400, 3), dtype=np.uint8)
        # グラデーション付きで自然な外観
        for y in range(100, 400):
            for x in range(100, 300):
                intensity = 100 + (y - 100) // 10
                high_quality_image[y, x] = min(255, intensity)

        result = analyzer.analyze_extraction_quality(high_quality_image)

        assert result.overall_quality_score > 0.8
        assert result.edge_quality_score > 0.7
        assert result.completeness_score > 0.8
        assert result.sharpness_score > 0.6

    def test_analyze_poor_quality_extraction(self, analyzer):
        """低品質抽出の分析テスト"""
        # 低品質な抽出画像（ぼやけた輪郭、小さなサイズ、不完全性）
        poor_quality_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # 小さく、ぼやけた領域
        poor_quality_image[120:180, 120:180] = 50  # 暗く小さな領域

        result = analyzer.analyze_extraction_quality(poor_quality_image)

        assert result.overall_quality_score < 0.5
        assert result.completeness_score < 0.6
        assert len(result.quality_issues) > 0

    def test_edge_quality_analysis(self, analyzer):
        """エッジ品質分析のテスト"""
        # 明確なエッジを持つ画像
        sharp_edges_image = np.zeros((400, 400, 3), dtype=np.uint8)
        sharp_edges_image[100:300, 100:300] = 255  # 明確な境界

        edge_result = analyzer.analyze_extraction_quality(sharp_edges_image)
        assert edge_result.edge_quality_score > 0.7

        # ぼやけたエッジを持つ画像
        blurry_image = np.zeros((400, 400, 3), dtype=np.uint8)
        # ガウシアンブラーでぼやけた領域を作成
        temp_region = np.ones((200, 200), dtype=np.uint8) * 200
        blurred_region = cv2.GaussianBlur(temp_region, (15, 15), 0)
        blurry_image[100:300, 100:300, 0] = blurred_region
        blurry_image[100:300, 100:300, 1] = blurred_region
        blurry_image[100:300, 100:300, 2] = blurred_region

        blur_result = analyzer.analyze_extraction_quality(blurry_image)
        assert blur_result.edge_quality_score < edge_result.edge_quality_score

    def test_size_adequacy_analysis(self, analyzer):
        """サイズ適正性分析のテスト"""
        # 適切なサイズの画像
        adequate_size_image = np.zeros((600, 400, 3), dtype=np.uint8)
        adequate_size_image[100:500, 50:350] = 150

        size_result = analyzer.analyze_extraction_quality(adequate_size_image)
        assert size_result.size_adequacy_score > 0.7

        # 小さすぎる画像
        tiny_image = np.zeros((400, 400, 3), dtype=np.uint8)
        tiny_image[180:220, 180:220] = 150  # 40x40の小さな領域

        tiny_result = analyzer.analyze_extraction_quality(tiny_image)
        assert tiny_result.size_adequacy_score < size_result.size_adequacy_score

    def test_calculate_completeness_metrics(self, analyzer):
        """完全性メトリクス計算のテスト"""
        # 完全なキャラクター（縦長、適切なアスペクト比）
        complete_char = np.zeros((600, 300, 3), dtype=np.uint8)
        complete_char[50:550, 75:225] = 120  # 2:1のアスペクト比

        completeness = analyzer.calculate_completeness_metrics(complete_char)

        assert completeness["aspect_ratio_score"] > 0.7
        assert completeness["vertical_coverage_score"] > 0.8
        assert completeness["horizontal_coverage_score"] > 0.6
        assert completeness["overall_completeness"] > 0.7


class TestCompletenessValidator:
    """完全性検証器のテストクラス"""

    @pytest.fixture
    def validator(self):
        """CompletenessValidator インスタンスのフィクスチャ"""
        if CompletenessValidator is None:
            pytest.skip("CompletenessValidator not yet implemented")
        return CompletenessValidator()

    def test_completeness_validator_initialization(self, validator):
        """CompletenessValidator の初期化テスト"""
        assert validator is not None
        assert hasattr(validator, "validate_completeness")
        assert hasattr(validator, "detect_missing_parts")

    def test_validate_complete_character(self, validator):
        """完全キャラクターの検証テスト"""
        # 完全なキャラクター（頭・胴・手・脚）
        complete_character = np.zeros((800, 400, 3), dtype=np.uint8)

        # 頭部
        complete_character[50:150, 150:250] = 100
        # 胴体
        complete_character[150:400, 120:280] = 110
        # 左腕
        complete_character[200:350, 80:120] = 105
        # 右腕
        complete_character[200:350, 280:320] = 105
        # 脚部
        complete_character[400:750, 140:260] = 120

        result = validator.validate_completeness(complete_character)

        assert result.is_complete is True
        assert result.completeness_percentage > 85
        assert len(result.missing_parts) == 0
        assert result.validation_confidence > 0.8

    def test_detect_missing_head(self, validator):
        """頭部欠損の検出テスト"""
        # 頭部なしキャラクター
        headless_character = np.zeros((700, 400, 3), dtype=np.uint8)

        # 胴体のみ（頭部なし）
        headless_character[100:350, 120:280] = 110
        # 脚部
        headless_character[350:650, 140:260] = 120

        result = validator.validate_completeness(headless_character)

        assert result.is_complete is False
        assert "head" in result.missing_parts
        assert result.completeness_percentage < 80

    def test_detect_missing_limbs(self, validator):
        """手足欠損の検出テスト"""
        # 手足なしキャラクター（胴体のみ）
        limbless_character = np.zeros((600, 300, 3), dtype=np.uint8)

        # 頭部
        limbless_character[50:150, 100:200] = 100
        # 胴体のみ（手足なし）
        limbless_character[150:450, 80:220] = 110

        result = validator.validate_completeness(limbless_character)

        assert result.is_complete is False
        assert "arms" in result.missing_parts or "limbs" in result.missing_parts
        assert "legs" in result.missing_parts or "limbs" in result.missing_parts
        assert result.completeness_percentage < 70

    def test_partial_limb_detection(self, validator):
        """部分的な手足の検出テスト"""
        # 一部の手足が欠けているキャラクター
        partial_limbs_character = np.zeros((700, 500, 3), dtype=np.uint8)

        # 頭部
        partial_limbs_character[50:150, 200:300] = 100
        # 胴体
        partial_limbs_character[150:400, 150:350] = 110
        # 左腕（部分的）
        partial_limbs_character[200:300, 100:150] = 105
        # 右腕なし
        # 脚部（部分的）
        partial_limbs_character[400:600, 180:320] = 120

        result = validator.validate_completeness(partial_limbs_character)

        # 部分的な欠損として検出されるべき
        assert result.is_complete is False
        assert result.completeness_percentage > 50  # 完全欠損よりは良い
        assert result.completeness_percentage < 90  # でも完全ではない

    def test_fragmentation_detection(self, validator):
        """断片化検出のテスト"""
        # 断片化したキャラクター（複数の分離した小領域）
        fragmented_character = np.zeros((600, 400, 3), dtype=np.uint8)

        # 複数の小さな断片
        fragmented_character[100:140, 180:220] = 90  # 断片1
        fragmented_character[200:240, 160:200] = 100  # 断片2
        fragmented_character[300:340, 190:230] = 110  # 断片3
        fragmented_character[450:490, 170:210] = 120  # 断片4

        result = validator.validate_completeness(fragmented_character)

        assert result.is_complete is False
        assert result.fragmentation_detected is True
        assert result.fragment_count >= 3
        assert result.completeness_percentage < 60

    def test_aspect_ratio_validation(self, validator):
        """アスペクト比による検証テスト"""
        # 異常なアスペクト比のキャラクター（横長すぎ）
        wide_character = np.zeros((300, 900, 3), dtype=np.uint8)
        wide_character[50:250, 100:800] = 100

        result_wide = validator.validate_completeness(wide_character)
        assert result_wide.aspect_ratio_warning is True
        assert result_wide.is_complete is False

        # 正常なアスペクト比
        normal_character = np.zeros((600, 300, 3), dtype=np.uint8)
        normal_character[50:550, 50:250] = 100

        result_normal = validator.validate_completeness(normal_character)
        assert hasattr(result_normal, "aspect_ratio_score")


class TestQI002PartialExtractionIntegration:
    """QI-002 部分抽出品質の統合テスト"""

    def test_qi002_partial_extraction_workflow(self):
        """QI-002 部分抽出検出の完全ワークフロー統合テスト"""
        if any(
            cls is None
            for cls in [PartialExtractionDetector, ExtractionQualityAnalyzer, CompletenessValidator]
        ):
            pytest.skip("Implementation not yet available")

        detector = PartialExtractionDetector()
        analyzer = ExtractionQualityAnalyzer()
        validator = CompletenessValidator()

        # QI-002で想定される問題パターンのテスト画像
        problematic_extractions = [
            # パターン1: 上半身のみ
            np.array([[[0] * 3] * 300] * 150 + [[[100] * 3] * 300] * 150),  # 上半身のみ
            # パターン2: 断片化
            np.zeros((400, 400, 3), dtype=np.uint8),  # 後で断片を追加
            # パターン3: 頭部切断
            np.array([[[0] * 3] * 300] * 100 + [[[110] * 3] * 300] * 300),  # 頭部なし
        ]

        # パターン2の断片を追加
        problematic_extractions[1][50:90, 50:90] = 80  # 断片1
        problematic_extractions[1][150:190, 150:190] = 90  # 断片2
        problematic_extractions[1][250:290, 100:140] = 100  # 断片3

        results = []
        for i, test_image in enumerate(problematic_extractions):
            # 3段階の統合解析
            detection_result = detector.detect_partial_extraction(test_image)
            quality_result = analyzer.analyze_extraction_quality(test_image)
            completeness_result = validator.validate_completeness(test_image)

            # 統合結果の評価
            integrated_result = {
                "pattern_id": i,
                "is_partial": detection_result.is_partial_extraction,
                "quality_score": quality_result.overall_quality_score,
                "completeness": completeness_result.completeness_percentage,
                "has_issues": detection_result.is_partial_extraction
                or quality_result.overall_quality_score < 0.6,
            }
            results.append(integrated_result)

        # 期待される結果の検証
        assert all(result["has_issues"] for result in results)  # 全て問題として検出
        assert results[0]["is_partial"] is True  # 上半身のみは部分抽出
        assert results[1]["completeness"] < 60  # 断片化は低完全性
        assert results[2]["quality_score"] < 0.7  # 頭部切断は低品質

    def test_qi002_improvement_validation(self):
        """QI-002 改善効果の検証テスト"""
        if PartialExtractionDetector is None:
            pytest.skip("Implementation not yet available")

        detector = PartialExtractionDetector()

        # 改善前後の画像比較
        before_improvement = np.zeros((400, 400, 3), dtype=np.uint8)
        before_improvement[200:400, 150:250] = 90  # 下半身のみ

        after_improvement = np.zeros((600, 400, 3), dtype=np.uint8)
        after_improvement[50:550, 100:300] = 110  # 全身

        result_before = detector.detect_partial_extraction(before_improvement)
        result_after = detector.detect_partial_extraction(after_improvement)

        # 改善効果の確認
        assert result_before.is_partial_extraction is True
        assert result_after.is_partial_extraction is False
        assert result_after.completeness_score > result_before.completeness_score

        improvement_ratio = result_after.completeness_score / result_before.completeness_score
        assert improvement_ratio > 1.5  # 50%以上の改善


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
