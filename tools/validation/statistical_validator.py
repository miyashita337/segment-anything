"""
QCC-022: 統計的有意性検定システム（ウェルチのt検定）

設定A vs 設定Bの統計的差異を科学的に検定する独立評価システム。
scipy.stats.ttest_indを使用したウェルチのt検定（不等分散対応）を実装。
"""

import numpy as np

import warnings
from dataclasses import dataclass
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class TTestResult:
    """t検定結果を格納するデータクラス"""

    statistic: float
    p_value: float
    degrees_of_freedom: float
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    sample_size_a: int
    sample_size_b: int
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    interpretation: str


class StatisticalValidator:
    """
    統計的有意性検定を行うバリデータークラス

    主要機能:
    - ウェルチのt検定（不等分散対応）
    - 信頼区間計算
    - 効果サイズ計算（Cohen's d）
    - p値の解釈
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 有意水準（デフォルト: 0.05）
        """
        self.alpha = alpha

    def welch_t_test(
        self,
        group_a: Union[List[float], np.ndarray],
        group_b: Union[List[float], np.ndarray],
        alternative: str = "two-sided",
    ) -> TTestResult:
        """
        ウェルチのt検定を実行（不等分散を仮定）

        Args:
            group_a: グループAのデータ
            group_b: グループBのデータ
            alternative: 'two-sided', 'less', 'greater'のいずれか

        Returns:
            TTestResult: 検定結果
        """
        # NumPy配列に変換
        a = np.asarray(group_a)
        b = np.asarray(group_b)

        # NaNや無限大を除去
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        # サンプルサイズチェック
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"各グループには最低2つのサンプルが必要です。" f"グループA: {len(a)}個, グループB: {len(b)}個")

        # t検定実行（equal_var=Falseでウェルチの検定）
        statistic, p_value = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

        # 基本統計量計算
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        std_a = np.std(a, ddof=1)  # 不偏標準偏差
        std_b = np.std(b, ddof=1)
        n_a = len(a)
        n_b = len(b)

        # 自由度計算（ウェルチの公式）
        df = self._calculate_welch_df(std_a, std_b, n_a, n_b)

        # 信頼区間計算
        ci = self.calculate_confidence_interval(
            mean_a - mean_b, self._calculate_standard_error(std_a, std_b, n_a, n_b), df
        )

        # 効果サイズ計算（Cohen's d）
        effect_size = self.calculate_cohens_d(a, b)

        # 有意性判定
        is_significant = p_value < self.alpha

        # 結果の解釈
        interpretation = self.interpret_results(p_value, effect_size, is_significant)

        return TTestResult(
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            sample_size_a=n_a,
            sample_size_b=n_b,
            confidence_interval=ci,
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
        )

    def _calculate_welch_df(self, std_a: float, std_b: float, n_a: int, n_b: int) -> float:
        """
        ウェルチの自由度を計算

        Args:
            std_a: グループAの標準偏差
            std_b: グループBの標準偏差
            n_a: グループAのサンプルサイズ
            n_b: グループBのサンプルサイズ

        Returns:
            float: 自由度
        """
        var_a = std_a**2
        var_b = std_b**2

        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)

        return numerator / denominator

    def _calculate_standard_error(self, std_a: float, std_b: float, n_a: int, n_b: int) -> float:
        """
        標準誤差を計算

        Args:
            std_a: グループAの標準偏差
            std_b: グループBの標準偏差
            n_a: グループAのサンプルサイズ
            n_b: グループBのサンプルサイズ

        Returns:
            float: 標準誤差
        """
        return np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))

    def calculate_confidence_interval(
        self, mean_diff: float, se: float, df: float, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        信頼区間を計算

        Args:
            mean_diff: 平均値の差
            se: 標準誤差
            df: 自由度
            confidence: 信頼水準（デフォルト: 0.95）

        Returns:
            Tuple[float, float]: (下限, 上限)
        """
        # t分布の臨界値
        t_critical = stats.t.ppf((1 + confidence) / 2, df)

        # 信頼区間
        margin = t_critical * se
        lower = mean_diff - margin
        upper = mean_diff + margin

        return (lower, upper)

    def calculate_cohens_d(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """
        Cohen's d（効果サイズ）を計算

        Args:
            group_a: グループAのデータ
            group_b: グループBのデータ

        Returns:
            float: Cohen's d
        """
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)

        # プールされた標準偏差
        n_a = len(group_a)
        n_b = len(group_b)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)

        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std

    def interpret_p_value(self, p_value: float) -> str:
        """
        p値を解釈

        Args:
            p_value: p値

        Returns:
            str: 解釈文字列
        """
        if p_value < 0.001:
            return "非常に強い有意性 (p < 0.001)"
        elif p_value < 0.01:
            return "強い有意性 (p < 0.01)"
        elif p_value < 0.05:
            return "有意性あり (p < 0.05)"
        elif p_value < 0.10:
            return "境界的有意性 (p < 0.10)"
        else:
            return "有意性なし (p >= 0.10)"

    def interpret_effect_size(self, d: float) -> str:
        """
        効果サイズ（Cohen's d）を解釈

        Args:
            d: Cohen's d

        Returns:
            str: 解釈文字列
        """
        abs_d = abs(d)

        if abs_d < 0.2:
            return "効果なし"
        elif abs_d < 0.5:
            return "小さい効果"
        elif abs_d < 0.8:
            return "中程度の効果"
        else:
            return "大きい効果"

    def interpret_results(self, p_value: float, effect_size: float, is_significant: bool) -> str:
        """
        検定結果を総合的に解釈

        Args:
            p_value: p値
            effect_size: 効果サイズ
            is_significant: 有意性の有無

        Returns:
            str: 解釈文字列
        """
        p_interpretation = self.interpret_p_value(p_value)
        d_interpretation = self.interpret_effect_size(effect_size)

        if is_significant:
            if abs(effect_size) >= 0.5:
                result = f"統計的に有意かつ実用的に意味のある差が認められます。"
            else:
                result = f"統計的に有意ですが、実用的な効果は小さい可能性があります。"
        else:
            result = f"統計的に有意な差は認められません。"

        return f"{result} {p_interpretation}、{d_interpretation}（d = {effect_size:.3f}）"

    def validate_normality(self, data: Union[List[float], np.ndarray]) -> Tuple[float, bool]:
        """
        正規性検定（Shapiro-Wilk検定）

        Args:
            data: 検定対象データ

        Returns:
            Tuple[float, bool]: (p値, 正規性の有無)
        """
        if len(data) < 3:
            warnings.warn("サンプルサイズが小さすぎるため正規性検定をスキップします")
            return (1.0, True)

        statistic, p_value = stats.shapiro(data)
        is_normal = p_value > self.alpha

        return (p_value, is_normal)

    def validate_equal_variance(
        self, group_a: Union[List[float], np.ndarray], group_b: Union[List[float], np.ndarray]
    ) -> Tuple[float, bool]:
        """
        等分散性検定（Levene検定）

        Args:
            group_a: グループAのデータ
            group_b: グループBのデータ

        Returns:
            Tuple[float, bool]: (p値, 等分散性の有無)
        """
        statistic, p_value = stats.levene(group_a, group_b)
        is_equal_var = p_value > self.alpha

        return (p_value, is_equal_var)

    def perform_multiple_comparison_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> List[float]:
        """
        多重比較補正

        Args:
            p_values: p値のリスト
            method: 補正方法（'bonferroni', 'holm'）

        Returns:
            List[float]: 補正後のp値
        """
        n = len(p_values)

        if method == "bonferroni":
            # Bonferroni補正
            return [min(p * n, 1.0) for p in p_values]

        elif method == "holm":
            # Holm補正
            sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected = []

            for i, (idx, p) in enumerate(sorted_p):
                corrected_p = min(p * (n - i), 1.0)
                if i > 0:
                    corrected_p = max(corrected_p, corrected[i - 1][1])
                corrected.append((idx, corrected_p))

            # 元の順序に戻す
            return [p for _, p in sorted(corrected)]

        else:
            raise ValueError(f"未対応の補正方法: {method}")

    def compare_multiple_groups(
        self, groups: Dict[str, Union[List[float], np.ndarray]], baseline: Optional[str] = None
    ) -> Dict[str, TTestResult]:
        """
        複数グループの比較（ベースラインとの比較）

        Args:
            groups: グループ名をキーとするデータの辞書
            baseline: ベースラインとするグループ名（Noneの場合は総当たり）

        Returns:
            Dict[str, TTestResult]: 比較結果の辞書
        """
        results = {}

        if baseline:
            # ベースラインとの比較
            baseline_data = groups[baseline]
            for name, data in groups.items():
                if name != baseline:
                    key = f"{baseline}_vs_{name}"
                    results[key] = self.welch_t_test(baseline_data, data)
        else:
            # 総当たり比較
            group_names = list(groups.keys())
            for i, name_a in enumerate(group_names):
                for name_b in group_names[i + 1 :]:
                    key = f"{name_a}_vs_{name_b}"
                    results[key] = self.welch_t_test(groups[name_a], groups[name_b])

        return results
