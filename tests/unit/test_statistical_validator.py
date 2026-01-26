"""
QCC-022: StatisticalValidatorの単体テスト
"""

import numpy as np

import sys
import unittest
from pathlib import Path
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.validation import StatisticalValidator
from tools.validation.statistical_validator import TTestResult


class TestStatisticalValidator(unittest.TestCase):
    """StatisticalValidatorクラスのテスト"""

    def setUp(self):
        """テストの前準備"""
        self.validator = StatisticalValidator(alpha=0.05)

        # テストデータ生成
        np.random.seed(42)
        self.group_a = np.random.normal(100, 15, 30)  # 平均100、標準偏差15
        self.group_b = np.random.normal(110, 15, 30)  # 平均110、標準偏差15（有意差あり）
        self.group_c = np.random.normal(100.5, 15, 30)  # 平均100.5、標準偏差15（有意差なし）

    def test_welch_t_test_significant(self):
        """有意差がある場合のt検定"""
        result = self.validator.welch_t_test(self.group_a, self.group_b)

        # 基本的なアサーション
        self.assertIsInstance(result, TTestResult)
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

        # 平均値の確認
        self.assertAlmostEqual(result.mean_a, np.mean(self.group_a), places=5)
        self.assertAlmostEqual(result.mean_b, np.mean(self.group_b), places=5)

        # 効果サイズが中程度以上
        self.assertGreater(abs(result.effect_size), 0.5)

    def test_welch_t_test_not_significant(self):
        """有意差がない場合のt検定"""
        result = self.validator.welch_t_test(self.group_a, self.group_c)

        self.assertFalse(result.is_significant)
        self.assertGreater(result.p_value, 0.05)

        # 効果サイズが小さい
        self.assertLess(abs(result.effect_size), 0.5)

    def test_confidence_interval(self):
        """信頼区間の計算テスト"""
        result = self.validator.welch_t_test(self.group_a, self.group_b)

        # 信頼区間が平均値の差を含むか
        mean_diff = result.mean_a - result.mean_b
        ci_lower, ci_upper = result.confidence_interval

        self.assertLess(ci_lower, mean_diff)
        self.assertGreater(ci_upper, mean_diff)

        # 有意差がある場合、信頼区間は0を含まない
        self.assertLess(ci_upper, 0)  # group_a < group_bなので負の値

    def test_cohens_d_calculation(self):
        """Cohen's dの計算テスト"""
        # 効果サイズが既知のデータ
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])  # 平均が2異なる

        d = self.validator.calculate_cohens_d(group1, group2)

        # Cohen's dは約1.26（大きい効果）のはず
        self.assertAlmostEqual(abs(d), 1.26, places=1)

    def test_small_sample_size(self):
        """小サンプルサイズのテスト"""
        small_a = np.array([1, 2, 3])
        small_b = np.array([4, 5, 6])

        result = self.validator.welch_t_test(small_a, small_b)

        self.assertIsNotNone(result)
        self.assertEqual(result.sample_size_a, 3)
        self.assertEqual(result.sample_size_b, 3)

    def test_edge_cases(self):
        """エッジケースのテスト"""
        # サンプルサイズが不足
        with self.assertRaises(ValueError):
            self.validator.welch_t_test([1], [2, 3])

        # NaNを含むデータ
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        clean_data = np.array([6, 7, 8, 9, 10])

        result = self.validator.welch_t_test(data_with_nan, clean_data)
        self.assertEqual(result.sample_size_a, 4)  # NaNが除外される

    def test_normality_check(self):
        """正規性検定のテスト"""
        # 正規分布データ
        normal_data = np.random.normal(0, 1, 100)
        p_value, is_normal = self.validator.validate_normality(normal_data)

        self.assertTrue(is_normal)
        self.assertGreater(p_value, 0.05)

        # 非正規分布データ（一様分布）
        uniform_data = np.random.uniform(0, 1, 100)
        p_value, is_normal = self.validator.validate_normality(uniform_data)

        # 一様分布は正規分布ではない可能性が高い
        self.assertIsNotNone(p_value)

    def test_equal_variance_check(self):
        """等分散性検定のテスト"""
        # 等分散のデータ
        equal_var_a = np.random.normal(0, 1, 30)
        equal_var_b = np.random.normal(0, 1, 30)

        p_value, is_equal = self.validator.validate_equal_variance(equal_var_a, equal_var_b)

        # 同じ分散なので等分散性があるはず
        self.assertGreater(p_value, 0.05)

        # 不等分散のデータ
        unequal_var_a = np.random.normal(0, 1, 30)
        unequal_var_b = np.random.normal(0, 5, 30)  # 分散が大きい

        p_value, is_equal = self.validator.validate_equal_variance(unequal_var_a, unequal_var_b)

        # 分散が異なるので等分散性がないはず
        self.assertLess(p_value, 0.05)

    def test_multiple_comparison_correction(self):
        """多重比較補正のテスト"""
        p_values = [0.01, 0.04, 0.03, 0.20]

        # Bonferroni補正
        corrected = self.validator.perform_multiple_comparison_correction(
            p_values, method="bonferroni"
        )

        # 補正後のp値は元の値のn倍
        self.assertEqual(corrected[0], 0.04)  # 0.01 * 4
        self.assertEqual(corrected[1], 0.16)  # 0.04 * 4

        # Holm補正
        corrected_holm = self.validator.perform_multiple_comparison_correction(
            p_values, method="holm"
        )

        # Holm補正は段階的
        self.assertIsNotNone(corrected_holm)
        self.assertEqual(len(corrected_holm), len(p_values))

    def test_multiple_group_comparison(self):
        """複数グループ比較のテスト"""
        groups = {"group_a": self.group_a, "group_b": self.group_b, "group_c": self.group_c}

        # ベースライン比較
        results = self.validator.compare_multiple_groups(groups, baseline="group_a")

        self.assertEqual(len(results), 2)  # group_a vs b, group_a vs c
        self.assertIn("group_a_vs_group_b", results)
        self.assertIn("group_a_vs_group_c", results)

        # 総当たり比較
        results_all = self.validator.compare_multiple_groups(groups)

        self.assertEqual(len(results_all), 3)  # 3C2 = 3通り

    def test_interpretation_functions(self):
        """解釈関数のテスト"""
        # p値の解釈
        self.assertIn("非常に強い", self.validator.interpret_p_value(0.0001))
        self.assertIn("強い", self.validator.interpret_p_value(0.005))
        self.assertIn("有意性あり", self.validator.interpret_p_value(0.04))
        self.assertIn("境界的", self.validator.interpret_p_value(0.08))
        self.assertIn("有意性なし", self.validator.interpret_p_value(0.15))

        # 効果サイズの解釈
        self.assertIn("効果なし", self.validator.interpret_effect_size(0.1))
        self.assertIn("小さい", self.validator.interpret_effect_size(0.3))
        self.assertIn("中程度", self.validator.interpret_effect_size(0.6))
        self.assertIn("大きい", self.validator.interpret_effect_size(0.9))

    def test_alternative_hypotheses(self):
        """片側検定のテスト"""
        # 両側検定（デフォルト）
        result_two = self.validator.welch_t_test(
            self.group_a, self.group_b, alternative="two-sided"
        )

        # 片側検定（less: group_a < group_b）
        result_less = self.validator.welch_t_test(self.group_a, self.group_b, alternative="less")

        # 片側検定（greater: group_a > group_b）
        result_greater = self.validator.welch_t_test(
            self.group_a, self.group_b, alternative="greater"
        )

        # 片側検定のp値は両側検定の半分（方向が正しい場合）
        self.assertLess(result_less.p_value, result_two.p_value)
        self.assertGreater(result_greater.p_value, result_two.p_value)


if __name__ == "__main__":
    unittest.main()
