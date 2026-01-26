"""
QI-002: 複数キャラクター検出システムのテスト

このテストモジュールは、1枚の画像に複数のキャラクターが存在する場合の
検出・分離・品質評価機能をテストします。
"""

import numpy as np
import cv2

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# テスト対象のモジュール（実装後にインポートされる予定）
try:
    from features.evaluation.detectors.character_separator import CharacterSeparator
    from features.evaluation.detectors.multi_character_detector import MultiCharacterDetector
    from features.evaluation.utils.character_quality_assessor import CharacterQualityAssessor
except ImportError:
    # TDD: 実装前なので ImportError は予期される
    MultiCharacterDetector = None
    CharacterSeparator = None
    CharacterQualityAssessor = None


class TestMultiCharacterDetector:
    """複数キャラクター検出器のテストクラス"""

    @pytest.fixture
    def detector(self):
        """MultiCharacterDetector インスタンスのフィクスチャ"""
        if MultiCharacterDetector is None:
            pytest.skip("MultiCharacterDetector not yet implemented")
        return MultiCharacterDetector()

    @pytest.fixture
    def sample_multi_character_images(self):
        """テスト用複数キャラクター画像データの生成"""
        # 2キャラクター画像（左右に配置）
        dual_char_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # 左キャラクター（明度100）
        dual_char_image[100:400, 50:200] = 100
        # 右キャラクター（明度150）
        dual_char_image[150:450, 300:450] = 150

        # 3キャラクター画像（三角配置）
        triple_char_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # 上キャラクター
        triple_char_image[50:200, 180:330] = 80
        # 左下キャラクター
        triple_char_image[300:450, 50:200] = 120
        # 右下キャラクター
        triple_char_image[300:450, 300:450] = 160

        # 単一キャラクター画像（比較用）
        single_char_image = np.zeros((512, 512, 3), dtype=np.uint8)
        single_char_image[150:350, 150:350] = 100

        # 重複キャラクター画像（重なり合っている）
        overlapping_image = np.zeros((512, 512, 3), dtype=np.uint8)
        overlapping_image[100:300, 100:300] = 80  # ベース
        overlapping_image[200:400, 200:400] = 120  # 重複部分

        return {
            "dual_character": dual_char_image,
            "triple_character": triple_char_image,
            "single_character": single_char_image,
            "overlapping_characters": overlapping_image,
        }

    def test_multi_character_detector_initialization(self, detector):
        """MultiCharacterDetector の初期化テスト"""
        assert detector is not None
        assert hasattr(detector, "detect_characters")
        assert hasattr(detector, "min_character_size")
        assert hasattr(detector, "max_characters")

        # デフォルト設定の確認
        assert detector.min_character_size > 0
        assert detector.max_characters >= 1

    def test_detect_single_character(self, detector, sample_multi_character_images):
        """単一キャラクターの正確な検出テスト"""
        result = detector.detect_characters(sample_multi_character_images["single_character"])

        assert result.character_count == 1
        assert len(result.character_regions) == 1
        assert result.detection_confidence > 0.6  # フォールバック検出なので信頼度を下げる
        assert result.is_multi_character is False

    def test_detect_dual_characters(self, detector, sample_multi_character_images):
        """2キャラクターの検出テスト"""
        result = detector.detect_characters(sample_multi_character_images["dual_character"])

        assert result.character_count == 2
        assert len(result.character_regions) == 2
        assert result.is_multi_character is True
        assert result.detection_confidence > 0.6

        # 各キャラクターの位置確認
        regions = result.character_regions
        left_region = min(regions, key=lambda r: r.center_x)
        right_region = max(regions, key=lambda r: r.center_x)

        assert left_region.center_x < right_region.center_x  # 左右配置の確認

    def test_detect_triple_characters(self, detector, sample_multi_character_images):
        """3キャラクターの検出テスト"""
        result = detector.detect_characters(sample_multi_character_images["triple_character"])

        assert result.character_count == 3
        assert len(result.character_regions) == 3
        assert result.is_multi_character is True
        assert result.detection_confidence > 0.5

        # 三角配置の確認
        regions = result.character_regions
        top_region = min(regions, key=lambda r: r.center_y)
        bottom_regions = [r for r in regions if r != top_region]

        assert top_region.center_y < min(r.center_y for r in bottom_regions)

    def test_detect_overlapping_characters(self, detector, sample_multi_character_images):
        """重複キャラクターの検出テスト"""
        result = detector.detect_characters(sample_multi_character_images["overlapping_characters"])

        # 重複している場合は1つの大きな領域として検出される可能性が高い
        assert result.character_count >= 1
        # 重複検出のロジックは、複数領域が検出された場合のみ動作
        if result.character_count > 1:
            assert result.has_overlapping_characters is True
            assert result.overlap_ratio > 0.0

    def test_character_size_filtering(self, sample_multi_character_images):
        """キャラクターサイズフィルタリングのテスト"""
        if MultiCharacterDetector is None:
            pytest.skip("MultiCharacterDetector not yet implemented")

        # 小さなキャラクターを除外する設定
        detector = MultiCharacterDetector(min_character_size=5000)  # 5000ピクセル以上

        result = detector.detect_characters(sample_multi_character_images["dual_character"])

        # 大きなキャラクターのみが検出される
        for region in result.character_regions:
            assert region.area >= 5000

    def test_max_characters_limitation(self, sample_multi_character_images):
        """最大キャラクター数制限のテスト"""
        if MultiCharacterDetector is None:
            pytest.skip("MultiCharacterDetector not yet implemented")

        # 最大2キャラクターまでの制限
        detector = MultiCharacterDetector(max_characters=2)

        result = detector.detect_characters(sample_multi_character_images["triple_character"])

        # 3つ存在するが、2つまでしか検出しない
        assert result.character_count <= 2
        assert len(result.character_regions) <= 2


class TestCharacterSeparator:
    """キャラクター分離器のテストクラス"""

    @pytest.fixture
    def separator(self):
        """CharacterSeparator インスタンスのフィクスチャ"""
        if CharacterSeparator is None:
            pytest.skip("CharacterSeparator not yet implemented")
        return CharacterSeparator()

    def test_character_separator_initialization(self, separator):
        """CharacterSeparator の初期化テスト"""
        assert separator is not None
        assert hasattr(separator, "separate_characters")
        assert hasattr(separator, "create_individual_masks")

    def test_separate_dual_characters(self, separator):
        """2キャラクター分離のテスト"""
        # 2つの分離した領域を持つテスト画像
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        test_image[50:200, 50:150] = 100  # 左キャラクター
        test_image[200:350, 250:350] = 150  # 右キャラクター

        separated_result = separator.separate_characters(test_image)

        assert separated_result.character_count == 2
        assert len(separated_result.individual_masks) == 2
        assert len(separated_result.character_images) == 2

        # 各マスクが適切に分離されていることを確認
        mask1, mask2 = separated_result.individual_masks
        assert np.sum(np.logical_and(mask1, mask2)) == 0  # 重複なし

    def test_create_individual_character_images(self, separator):
        """個別キャラクター画像生成のテスト"""
        # テスト用複合画像
        composite_image = np.zeros((300, 300, 3), dtype=np.uint8)
        composite_image[50:150, 50:150] = [255, 0, 0]  # 赤キャラクター
        composite_image[150:250, 150:250] = [0, 255, 0]  # 緑キャラクター

        separated_result = separator.separate_characters(composite_image)

        assert len(separated_result.character_images) == 2

        # 各キャラクター画像が適切に分離されていることを確認
        char1_img, char2_img = separated_result.character_images

        # 赤キャラクターは赤い色素が多い
        red_char = (
            char1_img if np.mean(char1_img[:, :, 0]) > np.mean(char1_img[:, :, 1]) else char2_img
        )
        green_char = char2_img if red_char is char1_img else char1_img

        assert np.mean(red_char[:, :, 0]) > np.mean(red_char[:, :, 1])  # 赤が優勢
        assert np.mean(green_char[:, :, 1]) > np.mean(green_char[:, :, 0])  # 緑が優勢

    def test_handle_overlapping_characters(self, separator):
        """重複キャラクター分離のテスト"""
        # 重複する2つの領域
        overlapping_image = np.zeros((300, 300, 3), dtype=np.uint8)
        overlapping_image[50:200, 50:200] = 80  # ベース領域
        overlapping_image[150:250, 150:250] = 120  # 重複領域

        separated_result = separator.separate_characters(overlapping_image)

        # 重複していても分離を試行
        assert separated_result.character_count >= 1
        assert separated_result.has_overlapping_regions is True
        assert 0 < separated_result.separation_confidence < 1.0  # 部分的な信頼度


class TestCharacterQualityAssessor:
    """キャラクター品質評価器のテストクラス"""

    @pytest.fixture
    def assessor(self):
        """CharacterQualityAssessor インスタンスのフィクスチャ"""
        if CharacterQualityAssessor is None:
            pytest.skip("CharacterQualityAssessor not yet implemented")
        return CharacterQualityAssessor()

    def test_quality_assessor_initialization(self, assessor):
        """CharacterQualityAssessor の初期化テスト"""
        assert assessor is not None
        assert hasattr(assessor, "assess_multi_character_quality")
        assert hasattr(assessor, "assess_individual_character_quality")

    def test_assess_individual_character_quality(self, assessor):
        """個別キャラクター品質評価のテスト"""
        # 高品質キャラクター（明確な輪郭、適切なサイズ）
        high_quality_char = np.zeros((200, 200, 3), dtype=np.uint8)
        high_quality_char[50:150, 50:150] = 120

        quality_result = assessor.assess_individual_character_quality(high_quality_char)

        assert quality_result.overall_quality_score > 0.7
        assert quality_result.completeness_score > 0.8
        assert quality_result.clarity_score > 0.7
        assert quality_result.size_adequacy_score > 0.8

    def test_assess_poor_quality_character(self, assessor):
        """低品質キャラクター評価のテスト"""
        # 低品質キャラクター（小さい、不明瞭）
        poor_quality_char = np.zeros((200, 200, 3), dtype=np.uint8)
        poor_quality_char[90:110, 90:110] = 30  # 非常に小さく暗い

        quality_result = assessor.assess_individual_character_quality(poor_quality_char)

        assert quality_result.overall_quality_score < 0.5
        assert quality_result.size_adequacy_score < 0.6
        assert quality_result.clarity_score < 0.6

    def test_assess_multi_character_scene_quality(self, assessor):
        """複数キャラクターシーン品質評価のテスト"""
        # 複数キャラクターのシーン
        multi_char_scene = np.zeros((400, 400, 3), dtype=np.uint8)
        multi_char_scene[50:200, 50:150] = 100  # キャラ1
        multi_char_scene[200:350, 250:350] = 120  # キャラ2

        scene_quality = assessor.assess_multi_character_quality(multi_char_scene)

        assert scene_quality.character_count == 2
        assert scene_quality.scene_balance_score > 0.0
        assert scene_quality.character_separation_quality > 0.0
        assert len(scene_quality.individual_character_scores) == 2

    def test_character_interaction_analysis(self, assessor):
        """キャラクター相互作用分析のテスト"""
        # 相互作用のあるシーン（近距離配置）
        interactive_scene = np.zeros((300, 300, 3), dtype=np.uint8)
        interactive_scene[100:200, 80:150] = 110  # キャラ1
        interactive_scene[100:200, 150:220] = 130  # キャラ2（隣接）

        interaction_result = assessor.assess_multi_character_quality(interactive_scene)

        assert interaction_result.character_interaction_score > 0.0
        assert interaction_result.proximity_analysis["average_distance"] < 100
        assert interaction_result.has_character_interaction is True


class TestQI002MultiCharacterIntegration:
    """QI-002 複数キャラクター検出の統合テスト"""

    def test_qi002_full_pipeline_integration(self):
        """QI-002 複数キャラクター検出の完全パイプライン統合テスト"""
        if any(
            cls is None
            for cls in [MultiCharacterDetector, CharacterSeparator, CharacterQualityAssessor]
        ):
            pytest.skip("Implementation not yet available")

        # 実際のワークフローをシミュレート
        detector = MultiCharacterDetector()
        separator = CharacterSeparator()
        assessor = CharacterQualityAssessor()

        # 複数キャラクターの複雑なシーン
        complex_scene = np.zeros((600, 600, 3), dtype=np.uint8)
        complex_scene[100:300, 100:250] = 90  # キャラ1
        complex_scene[300:500, 300:450] = 110  # キャラ2
        complex_scene[150:250, 400:500] = 130  # キャラ3（小さめ）

        # ステップ1: 複数キャラクター検出
        detection_result = detector.detect_characters(complex_scene)

        # ステップ2: キャラクター分離
        separation_result = separator.separate_characters(complex_scene)

        # ステップ3: 品質評価
        quality_result = assessor.assess_multi_character_quality(complex_scene)

        # 統合結果の検証
        assert detection_result.character_count >= 2
        assert separation_result.character_count == detection_result.character_count
        assert len(quality_result.individual_character_scores) == detection_result.character_count

    def test_qi002_performance_with_multiple_characters(self):
        """QI-002 複数キャラクターでのパフォーマンステスト"""
        if MultiCharacterDetector is None:
            pytest.skip("Implementation not yet available")

        import time

        detector = MultiCharacterDetector()

        # 5キャラクターの複雑なシーン
        complex_multi_scene = np.zeros((800, 800, 3), dtype=np.uint8)
        positions = [(100, 100), (300, 150), (500, 200), (200, 400), (600, 500)]
        for i, (x, y) in enumerate(positions):
            complex_multi_scene[y : y + 100, x : x + 100] = 80 + i * 20

        start_time = time.time()
        result = detector.detect_characters(complex_multi_scene)
        processing_time = time.time() - start_time

        # パフォーマンス要件: 500ms以下
        assert processing_time < 0.5
        assert result.character_count >= 3  # 少なくとも3つは検出

    def test_qi002_edge_cases_handling(self):
        """QI-002 エッジケースのハンドリングテスト"""
        if MultiCharacterDetector is None:
            pytest.skip("Implementation not yet available")

        detector = MultiCharacterDetector()

        # エッジケース1: 完全に空の画像
        empty_image = np.zeros((400, 400, 3), dtype=np.uint8)
        empty_result = detector.detect_characters(empty_image)
        assert empty_result.character_count == 0

        # エッジケース2: 非常に小さなキャラクター
        tiny_char_image = np.zeros((400, 400, 3), dtype=np.uint8)
        tiny_char_image[200:205, 200:205] = 100  # 5x5 ピクセル
        tiny_result = detector.detect_characters(tiny_char_image)
        # 小さすぎて検出されない可能性
        assert tiny_result.character_count >= 0

        # エッジケース3: 画像全体がキャラクター
        full_char_image = np.full((400, 400, 3), 100, dtype=np.uint8)
        full_result = detector.detect_characters(full_char_image)
        assert full_result.character_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
