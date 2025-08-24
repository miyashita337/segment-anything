#!/usr/bin/env python3
"""
QCC-021: サンプルサイズ妥当性検証システム
統計的有意性を保証するための最小サンプルサイズ計算と妥当性検証

統計学的背景:
- 効果サイズ（Cohen's d）に基づくパワー分析
- 信頼区間による精度推定
- 母集団推定のための必要サンプル数計算
"""

import numpy as np

import logging
import math
import scipy.stats as stats
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)


class TestType(Enum):
    """統計検定の種類"""
    ONE_SAMPLE_T = "one_sample_t"      # 1標本t検定
    TWO_SAMPLE_T = "two_sample_t"      # 2標本t検定
    PAIRED_T = "paired_t"              # 対応のあるt検定
    PROPORTION = "proportion"          # 比率検定
    CHI_SQUARE = "chi_square"          # χ²検定


@dataclass
class SampleSizeRequirement:
    """サンプルサイズ要件"""
    test_type: TestType
    effect_size: float          # 効果サイズ（Cohen's d）
    power: float                # 検出力（1-β）
    alpha: float                # 有意水準（α）
    required_n: int             # 必要サンプル数
    current_n: int              # 現在のサンプル数
    is_adequate: bool           # サンプルサイズが十分か
    confidence_width: float     # 信頼区間の幅
    precision_level: str        # 精度レベル（高・中・低）


@dataclass
class StatisticalValidation:
    """統計的妥当性検証結果"""
    sample_requirements: List[SampleSizeRequirement]
    overall_adequacy: bool      # 全体的サンプルサイズ妥当性
    recommended_n: int          # 推奨サンプル数
    current_power: float        # 現在の検出力
    precision_assessment: str   # 精度評価
    statistical_warnings: List[str]  # 統計的警告
    improvement_suggestions: List[str]  # 改善提案


class SampleSizeValidator:
    """
    サンプルサイズ妥当性検証システム
    
    統計学的原理：
    1. パワー分析による最小サンプル数計算
    2. 信頼区間による精度評価
    3. 効果サイズに基づく実用的有意性判定
    """
    
    def __init__(self, 
                 default_power: float = 0.80,
                 default_alpha: float = 0.05):
        """
        初期化
        
        Args:
            default_power: デフォルト検出力（80%）
            default_alpha: デフォルト有意水準（5%）
        """
        self.default_power = default_power
        self.default_alpha = default_alpha
        
        # 効果サイズの基準（Cohen's convention）
        self.effect_size_benchmarks = {
            'small': 0.2,     # 小効果
            'medium': 0.5,    # 中効果
            'large': 0.8      # 大効果
        }
        
        logger.info(f"SampleSizeValidator初期化: power={default_power}, alpha={default_alpha}")
    
    def calculate_required_sample_size(self,
                                     test_type: TestType,
                                     effect_size: float,
                                     power: float = None,
                                     alpha: float = None) -> int:
        """
        統計検定に必要な最小サンプル数を計算
        
        なぜこの計算が必要か:
        - 統計的有意性を適切な確率で検出するため
        - Type I error（偽陽性）とType II error（偽陰性）のバランス
        - 実用的な意義のある差を検出できる最小サンプル数
        
        Args:
            test_type: 統計検定の種類
            effect_size: 検出したい効果サイズ（Cohen's d）
            power: 検出力（デフォルト: 0.80）
            alpha: 有意水準（デフォルト: 0.05）
            
        Returns:
            必要最小サンプル数
        """
        power = power or self.default_power
        alpha = alpha or self.default_alpha
        
        if test_type == TestType.ONE_SAMPLE_T:
            return self._calculate_one_sample_t_size(effect_size, power, alpha)
        elif test_type == TestType.TWO_SAMPLE_T:
            return self._calculate_two_sample_t_size(effect_size, power, alpha)
        elif test_type == TestType.PAIRED_T:
            return self._calculate_paired_t_size(effect_size, power, alpha)
        elif test_type == TestType.PROPORTION:
            return self._calculate_proportion_size(effect_size, power, alpha)
        else:
            # デフォルトは2標本t検定
            return self._calculate_two_sample_t_size(effect_size, power, alpha)
    
    def _calculate_one_sample_t_size(self, effect_size: float, 
                                   power: float, alpha: float) -> int:
        """
        1標本t検定の必要サンプル数計算
        
        数学的背景:
        δ = |μ - μ₀| / σ (効果サイズ)
        n = ((z₁₋α/₂ + z₁₋β) / δ)²
        
        実用例: 品質スコアが基準値0.7と有意に違うかテスト
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)  # 両側検定
        z_beta = stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return max(3, math.ceil(n))  # 最小3サンプル
    
    def _calculate_two_sample_t_size(self, effect_size: float,
                                   power: float, alpha: float) -> int:
        """
        2標本t検定の必要サンプル数計算（各群）
        
        実用例: yado作者とkiri作者の品質差を検出
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # 各群のサンプル数
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return max(3, math.ceil(n_per_group))
    
    def _calculate_paired_t_size(self, effect_size: float,
                                power: float, alpha: float) -> int:
        """
        対応のあるt検定の必要サンプル数計算
        
        実用例: 改善前後の品質比較（同じ画像セットでの比較）
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return max(3, math.ceil(n))
    
    def _calculate_proportion_size(self, effect_size: float,
                                 power: float, alpha: float) -> int:
        """
        比率検定の必要サンプル数計算
        
        実用例: 成功率（A/B評価率）の改善を検出
        """
        # 比率検定の場合、effect_sizeは比率の差
        p1 = 0.5  # ベースライン比率（仮定）
        p2 = p1 + effect_size  # 改善後比率
        
        if p2 > 1.0 or p2 < 0.0:
            p2 = max(0.0, min(1.0, p2))
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p_pooled = (p1 + p2) / 2
        n = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
             
        return max(10, math.ceil(n))  # 比率検定は最小10サンプル
    
    def calculate_confidence_interval_width(self,
                                          sample_size: int,
                                          confidence_level: float = 0.95) -> float:
        """
        信頼区間の幅を計算
        
        なぜ重要か:
        - 推定精度の指標
        - 実用的な判断のための不確実性評価
        - より多いサンプル = より狭い信頼区間 = より高い精度
        """
        alpha = 1 - confidence_level
        z_value = stats.norm.ppf(1 - alpha/2)
        
        # 標準誤差（母分散未知の場合の近似）
        standard_error = 1 / math.sqrt(sample_size)
        
        # 信頼区間の半幅
        margin_of_error = z_value * standard_error
        
        # 全幅を返す
        return 2 * margin_of_error
    
    def assess_precision_level(self, confidence_width: float) -> str:
        """
        信頼区間幅から精度レベルを判定
        
        精度判定基準:
        - 幅 < 0.2: 高精度
        - 0.2 ≤ 幅 < 0.5: 中精度  
        - 幅 ≥ 0.5: 低精度
        """
        if confidence_width < 0.2:
            return "高精度"
        elif confidence_width < 0.5:
            return "中精度"
        else:
            return "低精度"
    
    def validate_sample_adequacy(self,
                                current_sample_size: int,
                                quality_data: List[float] = None,
                                test_scenarios: List[Dict] = None) -> StatisticalValidation:
        """
        現在のサンプルサイズの妥当性を包括的に検証
        
        修正の核心部分:
        複数の統計的観点から現在のサンプルサイズを評価し、
        具体的な改善提案を提供する
        
        Args:
            current_sample_size: 現在のサンプル数
            quality_data: 品質データ（あれば）
            test_scenarios: 検証シナリオ（あれば）
            
        Returns:
            包括的な統計的妥当性評価
        """
        requirements = []
        warnings = []
        suggestions = []
        
        # デフォルトテストシナリオ
        if not test_scenarios:
            test_scenarios = [
                {
                    'name': '品質改善検出（中効果）',
                    'test_type': TestType.ONE_SAMPLE_T,
                    'effect_size': self.effect_size_benchmarks['medium'],
                    'description': '品質スコア改善を統計的に検出'
                },
                {
                    'name': '作者間品質差（小効果）',
                    'test_type': TestType.TWO_SAMPLE_T,
                    'effect_size': self.effect_size_benchmarks['small'],
                    'description': '異なる作者の品質差を検出'
                },
                {
                    'name': '成功率改善（比率）',
                    'test_type': TestType.PROPORTION,
                    'effect_size': 0.2,  # 20%ポイントの改善
                    'description': 'A/B評価成功率の改善を検出'
                }
            ]
        
        # 各テストシナリオでサンプルサイズ要件を計算
        for scenario in test_scenarios:
            required_n = self.calculate_required_sample_size(
                scenario['test_type'],
                scenario['effect_size']
            )
            
            confidence_width = self.calculate_confidence_interval_width(
                current_sample_size
            )
            
            is_adequate = current_sample_size >= required_n
            precision = self.assess_precision_level(confidence_width)
            
            requirement = SampleSizeRequirement(
                test_type=scenario['test_type'],
                effect_size=scenario['effect_size'],
                power=self.default_power,
                alpha=self.default_alpha,
                required_n=required_n,
                current_n=current_sample_size,
                is_adequate=is_adequate,
                confidence_width=confidence_width,
                precision_level=precision
            )
            
            requirements.append(requirement)
            
            # 警告とアドバイス生成
            if not is_adequate:
                shortfall = required_n - current_sample_size
                warnings.append(
                    f"{scenario['name']}: {shortfall}サンプル不足 "
                    f"(必要{required_n}, 現在{current_sample_size})"
                )
                
                suggestions.append(
                    f"{scenario['description']}には最低{required_n}サンプル推奨"
                )
        
        # 全体評価
        overall_adequacy = all(req.is_adequate for req in requirements)
        recommended_n = max(req.required_n for req in requirements)
        
        # 現在の検出力計算（最も厳しい要件に基づく）
        if requirements:
            worst_case = max(requirements, key=lambda x: x.required_n)
            current_power = self._calculate_current_power(
                current_sample_size, 
                worst_case.effect_size,
                worst_case.test_type
            )
        else:
            current_power = 0.0
        
        # 精度評価（平均的な信頼区間幅から）
        avg_confidence_width = np.mean([req.confidence_width for req in requirements])
        precision_assessment = self.assess_precision_level(avg_confidence_width)
        
        # 追加提案
        if current_sample_size < 30:
            suggestions.append("中心極限定理適用には30サンプル以上推奨")
        
        if precision_assessment == "低精度":
            suggestions.append(f"推定精度向上には{recommended_n}サンプル以上推奨")
        
        return StatisticalValidation(
            sample_requirements=requirements,
            overall_adequacy=overall_adequacy,
            recommended_n=recommended_n,
            current_power=current_power,
            precision_assessment=precision_assessment,
            statistical_warnings=warnings,
            improvement_suggestions=suggestions
        )
    
    def _calculate_current_power(self, current_n: int, effect_size: float, 
                               test_type: TestType) -> float:
        """
        現在のサンプル数での検出力を計算
        
        検出力 = 真の効果があるときに、それを統計的有意として検出する確率
        """
        try:
            if test_type == TestType.ONE_SAMPLE_T:
                # 非心t分布を使用
                ncp = effect_size * math.sqrt(current_n)  # 非心度パラメータ
                critical_t = stats.t.ppf(1 - self.default_alpha/2, current_n - 1)
                power = 1 - stats.nct.cdf(critical_t, current_n - 1, ncp)
                return min(1.0, max(0.0, power))
            else:
                # 簡易近似（正確な計算は複雑）
                z_alpha = stats.norm.ppf(1 - self.default_alpha/2)
                z_beta = (effect_size * math.sqrt(current_n)) - z_alpha
                power = stats.norm.cdf(z_beta)
                return min(1.0, max(0.0, power))
        except:
            return 0.0


def main():
    """デモ・テスト実行"""
    validator = SampleSizeValidator()
    
    # QCA-001のサンプル数（17枚）での妥当性検証
    print("=== QCC-021: サンプルサイズ妥当性検証デモ ===")
    print(f"現在のQCA-001サンプル数: 17枚")
    
    validation = validator.validate_sample_adequacy(17)
    
    print(f"\n📊 統計的妥当性評価:")
    print(f"・全体的妥当性: {'✅ 適切' if validation.overall_adequacy else '❌ 不適切'}")
    print(f"・推奨サンプル数: {validation.recommended_n}")
    print(f"・現在の検出力: {validation.current_power:.3f}")
    print(f"・精度評価: {validation.precision_assessment}")
    
    if validation.statistical_warnings:
        print(f"\n⚠️ 統計的警告:")
        for warning in validation.statistical_warnings:
            print(f"  - {warning}")
    
    if validation.improvement_suggestions:
        print(f"\n💡 改善提案:")
        for suggestion in validation.improvement_suggestions:
            print(f"  - {suggestion}")
    
    print(f"\n📋 詳細要件:")
    for req in validation.sample_requirements:
        status = "✅" if req.is_adequate else "❌"
        print(f"  {status} {req.test_type.value}: {req.current_n}/{req.required_n} "
               f"(精度: {req.precision_level})")


if __name__ == "__main__":
    main()