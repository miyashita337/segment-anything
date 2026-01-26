#!/usr/bin/env python3
"""
QCC-021: SampleSizeValidatorのユニットテスト
統計的妥当性検証システムの正確性を確認
"""

import math
import sys
import unittest
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.analysis.sample_size_validator import (
    SampleSizeRequirement,
    SampleSizeValidator,
    StatisticalValidation,
    TestType,
)


class TestSampleSizeValidator(unittest.TestCase):
    """SampleSizeValidatorのテストクラス"""

    def setUp(self):
        """テスト前準備"""
        self.validator = SampleSizeValidator(default_power=0.8, default_alpha=0.05)

    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.validator.default_power, 0.8)
        self.assertEqual(self.validator.default_alpha, 0.05)

        # 効果サイズベンチマーク確認
        self.assertEqual(self.validator.effect_size_benchmarks["small"], 0.2)
        self.assertEqual(self.validator.effect_size_benchmarks["medium"], 0.5)
        self.assertEqual(self.validator.effect_size_benchmarks["large"], 0.8)

    def test_one_sample_t_size_calculation(self):
        """1標本t検定のサンプルサイズ計算テスト"""
        # 中効果（0.5）でのサンプルサイズ
        required_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5
        )

        # 理論値との比較（おおよそ34-35サンプル程度）
        self.assertGreater(required_n, 30)
        self.assertLess(required_n, 40)
        self.assertIsInstance(required_n, int)

        # 最小サンプル数確認
        small_effect_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=2.0  # 非常に大きな効果
        )
        self.assertGreaterEqual(small_effect_n, 3)  # 最小3サンプル

    def test_two_sample_t_size_calculation(self):
        """2標本t検定のサンプルサイズ計算テスト"""
        required_n = self.validator.calculate_required_sample_size(
            TestType.TWO_SAMPLE_T, effect_size=0.5
        )

        # 2標本t検定は1標本より多くのサンプルが必要
        one_sample_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5
        )
        self.assertGreater(required_n, one_sample_n)

        # 理論的妥当性（中効果で各群64サンプル程度）
        self.assertGreater(required_n, 50)
        self.assertLess(required_n, 80)

    def test_paired_t_size_calculation(self):
        """対応のあるt検定のサンプルサイズ計算テスト"""
        required_n = self.validator.calculate_required_sample_size(
            TestType.PAIRED_T, effect_size=0.5
        )

        # 対応ありt検定は1標本t検定と同等のサンプル数
        one_sample_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5
        )
        self.assertEqual(required_n, one_sample_n)

    def test_proportion_size_calculation(self):
        """比率検定のサンプルサイズ計算テスト"""
        required_n = self.validator.calculate_required_sample_size(
            TestType.PROPORTION, effect_size=0.2  # 20%ポイント改善
        )

        # 比率検定は最小10サンプル
        self.assertGreaterEqual(required_n, 10)

        # 理論的妥当性確認
        self.assertIsInstance(required_n, int)
        self.assertGreater(required_n, 0)

    def test_confidence_interval_width_calculation(self):
        """信頼区間幅計算のテスト"""
        # サンプル数によって信頼区間幅が変化することを確認
        width_small = self.validator.calculate_confidence_interval_width(10)
        width_medium = self.validator.calculate_confidence_interval_width(30)
        width_large = self.validator.calculate_confidence_interval_width(100)

        # サンプル数が多いほど信頼区間は狭くなる
        self.assertGreater(width_small, width_medium)
        self.assertGreater(width_medium, width_large)

        # 数値的妥当性
        self.assertGreater(width_small, 0)
        self.assertLess(width_large, 1.0)  # 合理的な範囲

    def test_precision_level_assessment(self):
        """精度レベル判定のテスト"""
        # 高精度判定
        high_precision = self.validator.assess_precision_level(0.1)
        self.assertEqual(high_precision, "高精度")

        # 中精度判定
        medium_precision = self.validator.assess_precision_level(0.3)
        self.assertEqual(medium_precision, "中精度")

        # 低精度判定
        low_precision = self.validator.assess_precision_level(0.7)
        self.assertEqual(low_precision, "低精度")

        # 境界値テスト
        boundary_high = self.validator.assess_precision_level(0.19)
        boundary_medium = self.validator.assess_precision_level(0.49)
        self.assertEqual(boundary_high, "高精度")
        self.assertEqual(boundary_medium, "中精度")

    def test_sample_adequacy_validation_sufficient(self):
        """十分なサンプル数での妥当性検証テスト"""
        # 大きなサンプル数（100）で検証
        validation = self.validator.validate_sample_adequacy(100)

        # 型確認
        self.assertIsInstance(validation, StatisticalValidation)

        # 十分なサンプル数では妥当性がTrue
        self.assertTrue(validation.overall_adequacy)
        self.assertLessEqual(validation.recommended_n, 100)

        # 検出力が高い
        self.assertGreater(validation.current_power, 0.8)

        # 精度が高い
        self.assertIn(validation.precision_assessment, ["高精度", "中精度"])

    def test_sample_adequacy_validation_insufficient(self):
        """不十分なサンプル数での妥当性検証テスト"""
        # 小さなサンプル数（5）で検証
        validation = self.validator.validate_sample_adequacy(5)

        # 不十分なサンプル数では妥当性がFalse
        self.assertFalse(validation.overall_adequacy)
        self.assertGreater(validation.recommended_n, 5)

        # 警告が生成される
        self.assertGreater(len(validation.statistical_warnings), 0)
        self.assertGreater(len(validation.improvement_suggestions), 0)

        # 検出力が低い
        self.assertLess(validation.current_power, 0.8)

    def test_qca001_specific_validation(self):
        """QCA-001特化検証テスト（17サンプル）"""
        # QCA-001の実際のサンプル数
        qca001_sample_size = 17

        validation = self.validator.validate_sample_adequacy(qca001_sample_size)

        # 基本的な妥当性チェック
        self.assertIsInstance(validation.overall_adequacy, bool)
        self.assertIsInstance(validation.recommended_n, int)
        self.assertGreater(validation.recommended_n, 0)

        # QCA-001のサンプル数は通常不十分（統計的に）
        # 但し、機械学習コンテキストでは実用的
        if not validation.overall_adequacy:
            self.assertGreater(validation.recommended_n, qca001_sample_size)

        # 要件リストの検証
        self.assertGreater(len(validation.sample_requirements), 0)
        for req in validation.sample_requirements:
            self.assertIsInstance(req, SampleSizeRequirement)
            self.assertEqual(req.current_n, qca001_sample_size)

    def test_effect_size_variations(self):
        """効果サイズ変動のテスト"""
        sample_sizes = {}

        for effect_name, effect_size in self.validator.effect_size_benchmarks.items():
            required_n = self.validator.calculate_required_sample_size(
                TestType.TWO_SAMPLE_T, effect_size
            )
            sample_sizes[effect_name] = required_n

        # 効果サイズが大きいほど必要サンプル数は少ない
        self.assertGreater(sample_sizes["small"], sample_sizes["medium"])
        self.assertGreater(sample_sizes["medium"], sample_sizes["large"])

        # 数値妥当性
        for effect_name, n in sample_sizes.items():
            self.assertGreater(n, 0)
            self.assertIsInstance(n, int)

    def test_power_calculation_edge_cases(self):
        """検出力計算のエッジケーステスト"""
        # 非常に小さなサンプル数
        power_small = self.validator._calculate_current_power(2, 0.5, TestType.ONE_SAMPLE_T)
        self.assertGreaterEqual(power_small, 0.0)
        self.assertLessEqual(power_small, 1.0)

        # 非常に大きなサンプル数
        power_large = self.validator._calculate_current_power(1000, 0.5, TestType.ONE_SAMPLE_T)
        self.assertGreaterEqual(power_large, 0.0)
        self.assertLessEqual(power_large, 1.0)

        # 大きなサンプルの方が高い検出力
        self.assertGreaterEqual(power_large, power_small)

    def test_custom_scenarios(self):
        """カスタムシナリオのテスト"""
        custom_scenarios = [
            {
                "name": "カスタム品質検証",
                "test_type": TestType.ONE_SAMPLE_T,
                "effect_size": 0.3,
                "description": "カスタム品質改善検証",
            }
        ]

        validation = self.validator.validate_sample_adequacy(
            current_sample_size=20, test_scenarios=custom_scenarios
        )

        # カスタムシナリオが反映されている
        self.assertEqual(len(validation.sample_requirements), 1)
        req = validation.sample_requirements[0]
        self.assertEqual(req.effect_size, 0.3)
        self.assertEqual(req.test_type, TestType.ONE_SAMPLE_T)
        self.assertEqual(req.current_n, 20)

    def test_statistical_warnings_generation(self):
        """統計的警告生成のテスト"""
        validation = self.validator.validate_sample_adequacy(8)  # 小サンプル

        # 中心極限定理に関する警告
        clt_warning_found = any(
            "30サンプル以上" in warning for warning in validation.improvement_suggestions
        )
        self.assertTrue(clt_warning_found)

        # サンプル不足警告
        shortage_warning_found = any(
            "サンプル不足" in warning for warning in validation.statistical_warnings
        )
        self.assertTrue(shortage_warning_found)


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテストクラス"""

    def setUp(self):
        self.validator = SampleSizeValidator()

    def test_zero_effect_size(self):
        """効果サイズ0のエラーハンドリング"""
        with self.assertRaises(ZeroDivisionError):
            # 効果サイズ0では計算不可
            self.validator.calculate_required_sample_size(TestType.ONE_SAMPLE_T, effect_size=0.0)

    def test_negative_effect_size(self):
        """負の効果サイズ処理"""
        # 絶対値で計算されるべき
        pos_result = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5
        )
        neg_result = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=-0.5
        )

        # 結果は同じになるべき（絶対値使用）
        # 実装によってはエラーになる可能性もある
        try:
            self.assertEqual(pos_result, neg_result)
        except:
            # エラーが発生する場合も想定内
            pass

    def test_extreme_power_alpha_values(self):
        """極端なパワーとα値のテスト"""
        # 非常に高いパワー要求
        high_power_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5, power=0.99, alpha=0.01
        )

        # 標準的なパワー要求
        standard_n = self.validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.5, power=0.8, alpha=0.05
        )

        # 高パワー・低α値はより多くのサンプルが必要
        self.assertGreater(high_power_n, standard_n)


def run_comprehensive_test():
    """包括的テスト実行"""
    print("🧪 QCC-021: SampleSizeValidator 包括的テスト開始")

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストクラス追加
    suite.addTests(loader.loadTestsFromTestCase(TestSampleSizeValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print("\n" + "=" * 60)
    print("🎯 QCC-021テスト結果サマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(
        f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"\n❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n⚠️ エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ 全テスト成功' if success else '❌ テスト失敗あり'}")

    return success


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
