#!/usr/bin/env python3
"""
CI-INTEGRATION-001 統合テスト
GitHub Actions CI環境での動作確認テスト
"""

import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.ci_environment import (
    CIConfiguration,
    CIEnvironmentDetector,
    CIProvider,
    get_extraction_params,
    is_ci_environment,
)


class TestCIEnvironmentDetection:
    """CI環境検出テスト"""

    def test_local_environment_detection(self):
        """ローカル環境検出テスト"""
        with patch.dict(os.environ, {}, clear=True):
            config = CIEnvironmentDetector.detect_ci_environment()

            assert config.is_ci is False
            assert config.provider == CIProvider.LOCAL
            assert config.cpu_only is False
            assert config.yolo_model == "yolov8x.pt"  # ローカル環境では高品質モデル
            assert config.sam_model == "sam_vit_h_4b8939.pth"

    def test_github_actions_detection(self):
        """GitHub Actions環境検出テスト"""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "CI": "true"}):
            config = CIEnvironmentDetector.detect_ci_environment()

            assert config.is_ci is True
            assert config.provider == CIProvider.GITHUB_ACTIONS
            assert config.cpu_only is True
            assert config.lightweight_models is True
            assert config.yolo_model == "yolov8n.pt"  # CI環境では軽量モデル
            assert config.sam_model == "sam_vit_b_01ec64.pth"

    def test_manual_ci_flag(self):
        """手動CI環境フラグテスト"""
        with patch.dict(os.environ, {"CI_ENVIRONMENT": "true"}):
            config = CIEnvironmentDetector.detect_ci_environment()

            assert config.is_ci is True
            assert config.cpu_only is True
            assert config.memory_limit_disabled is True

    def test_custom_model_override(self):
        """カスタムモデル指定テスト"""
        with patch.dict(
            os.environ,
            {
                "CI_ENVIRONMENT": "true",
                "YOLO_MODEL": "custom_yolo.pt",
                "SAM_MODEL": "custom_sam.pth",
            },
        ):
            config = CIEnvironmentDetector.detect_ci_environment()

            assert config.yolo_model == "custom_yolo.pt"
            assert config.sam_model == "custom_sam.pth"

    def test_extraction_params_optimization(self):
        """抽出パラメータ最適化テスト"""
        # ローカル環境
        with patch.dict(os.environ, {}, clear=True):
            params = get_extraction_params()
            assert params["ci_mode"] is False
            assert params["max_masks"] == 10
            assert params["yolo_confidence"] == 0.07

        # CI環境
        with patch.dict(os.environ, {"CI_ENVIRONMENT": "true"}):
            params = get_extraction_params()
            assert params["ci_mode"] is True
            assert params["max_masks"] == 5  # 制限強化
            assert params["yolo_confidence"] == 0.1  # 少し緩め
            assert params["cpu_only"] is True


class TestModelWrapperIntegration:
    """モデルラッパー統合テスト"""

    @pytest.mark.skipif(
        not Path("features/extraction/models").exists(), reason="Model wrapper modules not found"
    )
    def test_yolo_wrapper_ci_optimization(self):
        """YOLOラッパーCI最適化テスト"""
        with patch.dict(os.environ, {"CI_ENVIRONMENT": "true"}):
            from features.extraction.models.yolo_wrapper import YOLOModelWrapper

            # デフォルト初期化
            wrapper = YOLOModelWrapper()

            assert wrapper.model_path == "yolov8n.pt"  # CI環境では軽量モデル
            assert wrapper.confidence_threshold == 0.1  # CI環境では緩い閾値
            assert wrapper.device == "cpu"  # CI環境ではCPU

    @pytest.mark.skipif(
        not Path("features/extraction/models").exists(), reason="Model wrapper modules not found"
    )
    def test_sam_wrapper_ci_optimization(self):
        """SAMラッパーCI最適化テスト"""
        with patch.dict(os.environ, {"CI_ENVIRONMENT": "true"}):
            from features.extraction.models.sam_wrapper import SAMModelWrapper

            # デフォルト初期化
            wrapper = SAMModelWrapper()

            assert wrapper.model_type == "vit_b"  # CI環境では軽量モデル
            assert "sam_vit_b_01ec64.pth" in wrapper.checkpoint_path  # CI環境では軽量チェックポイント
            assert wrapper.device == "cpu"  # CI環境ではCPU


class TestCIStatisticalValidation:
    """CI統計整合性検証テスト（QCC-FIX-001準拠）"""

    def test_wilson_confidence_interval_validation(self):
        """Wilson信頼区間計算検証"""
        import math

        # 正常ケース
        def wilson_confidence_interval(x, n, z=1.96):
            if n == 0:
                return 0.0, 0.0
            if x > n:
                raise ValueError(f"Success count ({x}) cannot exceed total count ({n})")

            p = x / n
            denominator = 1 + (z**2 / n)
            center = (p + (z**2 / (2 * n))) / denominator
            margin = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denominator

            return center - margin, center + margin

        # テストケース1: 通常ケース
        lower, upper = wilson_confidence_interval(400, 424)
        assert 0.9 <= lower <= 1.0
        assert 0.9 <= upper <= 1.0
        assert lower <= upper

        # テストケース2: 425/424矛盾検出
        with pytest.raises(
            ValueError, match="Success count \\(425\\) cannot exceed total count \\(424\\)"
        ):
            wilson_confidence_interval(425, 424)

        # テストケース3: ゼロケース
        lower, upper = wilson_confidence_interval(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_statistical_impossibility_detection(self):
        """統計的不可能値検出テスト"""
        test_cases = [
            (424, 424, True),  # 正常
            (425, 424, False),  # 不可能（QCC-FIX-001で修正対象）
            (100, 200, True),  # 正常
            (0, 100, True),  # 正常（全失敗）
            (50, 0, False),  # 不可能（総数ゼロで成功あり）
        ]

        for success, total, should_be_valid in test_cases:
            is_valid = success <= total and total >= 0
            assert is_valid == should_be_valid, f"Case ({success}, {total}) validation failed"


@pytest.mark.integration
class TestCIProcessingPipeline:
    """CI処理パイプライン統合テスト"""

    def test_ci_environment_flag_consistency(self):
        """CI環境フラグ整合性テスト"""
        # 複数のフラグが同時に設定された場合の動作確認
        with patch.dict(
            os.environ,
            {
                "CI": "true",
                "CI_ENVIRONMENT": "true",
                "GITHUB_ACTIONS": "true",
                "CPU_ONLY": "false",  # 矛盾する設定
            },
        ):
            config = CIEnvironmentDetector.detect_ci_environment()

            # CI環境が検出された場合、cpu_onlyは強制的にTrueになるべき
            assert config.is_ci is True
            assert config.cpu_only is True  # CIが優先

    def test_processing_timeout_settings(self):
        """処理タイムアウト設定テスト"""
        # CI環境
        with patch.dict(os.environ, {"CI_ENVIRONMENT": "true", "MAX_PROCESSING_TIME": "180"}):
            config = CIEnvironmentDetector.detect_ci_environment()
            assert config.max_processing_time == 180

        # ローカル環境
        with patch.dict(os.environ, {}, clear=True):
            config = CIEnvironmentDetector.detect_ci_environment()
            assert config.max_processing_time == 1800  # デフォルト30分


if __name__ == "__main__":
    # スタンドアロンテスト実行
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🧪 CI Integration Test Suite")
    print("=" * 40)

    # 基本的な環境検出テスト
    print("\n🔍 Environment Detection Test:")
    CIEnvironmentDetector.log_environment_info()

    # 統計検証テスト
    print("\n📊 Statistical Validation Test:")
    import math

    def test_wilson_confidence():
        test_cases = [(424, 424), (400, 424), (100, 200)]
        for x, n in test_cases:
            p = x / n
            z = 1.96
            denominator = 1 + (z**2 / n)
            center = (p + (z**2 / (2 * n))) / denominator
            margin = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denominator
            lower, upper = center - margin, center + margin
            print(f"   {x}/{n}: [{lower:.3%}, {upper:.3%}]")

    test_wilson_confidence()

    print("\n✅ CI Integration tests completed")

    # pytest実行
    print("\n🚀 Running pytest...")
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
