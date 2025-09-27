"""
KIRO-012: 判定結果アグリゲーター

複数モジュールの判定結果を集約し、最終的な統合判定を生成
高度な統計分析・コンフリクト解決・信頼度計算を実行
"""

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .module_interfaces import (
    JudgmentResult,
    AggregatedJudgment,
    QualityGrade
)


@dataclass
class AggregationConfig:
    """集約設定"""
    consensus_threshold: float = 0.7        # コンセンサス必要閾値
    confidence_weight: float = 0.3          # 信頼度重み
    score_weight: float = 0.7               # スコア重み
    outlier_threshold: float = 2.0          # 外れ値検出閾値（標準偏差倍）
    min_agreement_ratio: float = 0.6        # 最小合意比率
    enable_weighted_voting: bool = True     # 信頼度重み付き投票
    enable_outlier_removal: bool = True     # 外れ値除去
    enable_confidence_boost: bool = True    # 高信頼度結果の重視


class JudgmentResultAggregator:
    """判定結果集約・最終判定クラス"""

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Args:
            config: 集約設定（None の場合はデフォルト設定）
        """
        self.config = config or AggregationConfig()

    def aggregate_results(self, module_results: Dict[str, JudgmentResult]) -> AggregatedJudgment:
        """
        複数モジュール結果の集約

        Args:
            module_results: モジュール別判定結果

        Returns:
            AggregatedJudgment: 統合判定結果
        """
        if not module_results:
            return self._create_empty_result("No module results to aggregate")

        # 基本統計計算
        basic_stats = self._calculate_basic_statistics(module_results)

        # 外れ値検出・除去
        if self.config.enable_outlier_removal:
            filtered_results, outliers = self._detect_and_remove_outliers(module_results)
        else:
            filtered_results = module_results
            outliers = {}

        # 最終グレード決定
        final_grade = self._determine_final_grade(filtered_results)

        # 全体信頼度計算
        overall_confidence = self._calculate_overall_confidence(filtered_results)

        # コンフリクト分析
        conflict_analysis = self._perform_conflict_analysis(module_results, outliers)

        # コンセンサス指標
        consensus_metrics = self._calculate_consensus_metrics(filtered_results, basic_stats)

        # 推奨事項統合
        recommendation_summary = self._integrate_recommendations(filtered_results)

        return AggregatedJudgment(
            final_grade=final_grade,
            overall_confidence=overall_confidence,
            module_results=module_results,
            consensus_metrics=consensus_metrics,
            conflict_analysis=conflict_analysis,
            recommendation_summary=recommendation_summary
        )

    def _calculate_basic_statistics(self, module_results: Dict[str, JudgmentResult]) -> Dict[str, float]:
        """基本統計の計算"""
        scores = [r.numeric_score for r in module_results.values()]
        confidences = [r.confidence_score for r in module_results.values()]

        if not scores:
            return {}

        return {
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_median': np.median(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_median': np.median(confidences),
            'result_count': len(scores)
        }

    def _detect_and_remove_outliers(self, module_results: Dict[str, JudgmentResult]) -> Tuple[Dict[str, JudgmentResult], Dict[str, JudgmentResult]]:
        """外れ値の検出と除去"""
        scores = np.array([r.numeric_score for r in module_results.values()])
        module_names = list(module_results.keys())

        if len(scores) <= 2:
            return module_results, {}

        # Z-score による外れ値検出
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if std_score == 0:
            return module_results, {}

        z_scores = np.abs((scores - mean_score) / std_score)
        outlier_mask = z_scores > self.config.outlier_threshold

        # 外れ値分離
        filtered_results = {}
        outliers = {}

        for i, (name, result) in enumerate(module_results.items()):
            if outlier_mask[i]:
                outliers[name] = result
            else:
                filtered_results[name] = result

        # 最低限の結果数を保証（50%以上は残す）
        if len(filtered_results) < len(module_results) * 0.5:
            return module_results, {}

        return filtered_results, outliers

    def _determine_final_grade(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """最終グレードの決定"""
        if not module_results:
            return QualityGrade.F

        # 複数の決定方法を組み合わせ
        methods = [
            self._grade_by_weighted_voting(module_results),
            self._grade_by_consensus_threshold(module_results),
            self._grade_by_average_score(module_results),
            self._grade_by_conservative_approach(module_results)
        ]

        # 最多決による最終決定
        grade_counter = Counter(methods)
        most_common_grade = grade_counter.most_common(1)[0][0]

        return most_common_grade

    def _grade_by_weighted_voting(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """信頼度重み付き投票"""
        if not self.config.enable_weighted_voting:
            return self._grade_by_simple_voting(module_results)

        grade_weights = {}
        total_weight = 0.0

        for result in module_results.values():
            grade = result.quality_grade
            weight = result.confidence_score

            if grade not in grade_weights:
                grade_weights[grade] = 0.0

            grade_weights[grade] += weight
            total_weight += weight

        if total_weight == 0:
            return QualityGrade.F

        # 正規化された重みで最多票を決定
        max_weight = 0.0
        winning_grade = QualityGrade.F

        for grade, weight in grade_weights.items():
            normalized_weight = weight / total_weight
            if normalized_weight > max_weight:
                max_weight = normalized_weight
                winning_grade = grade

        return winning_grade

    def _grade_by_simple_voting(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """単純多数決投票"""
        grades = [r.quality_grade for r in module_results.values()]
        grade_counter = Counter(grades)
        return grade_counter.most_common(1)[0][0]

    def _grade_by_consensus_threshold(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """コンセンサス閾値ベース"""
        grades = [r.quality_grade for r in module_results.values()]
        grade_counter = Counter(grades)

        total_votes = len(grades)
        required_consensus = int(total_votes * self.config.consensus_threshold)

        # コンセンサスが得られたグレードを返す
        for grade, count in grade_counter.most_common():
            if count >= required_consensus:
                return grade

        # コンセンサスが得られない場合は最多票
        return grade_counter.most_common(1)[0][0]

    def _grade_by_average_score(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """平均スコアベース"""
        scores = [r.numeric_score for r in module_results.values()]
        avg_score = np.mean(scores)

        # スコア閾値でグレード決定
        if avg_score >= 0.85:
            return QualityGrade.A
        elif avg_score >= 0.70:
            return QualityGrade.B
        elif avg_score >= 0.55:
            return QualityGrade.C
        elif avg_score >= 0.40:
            return QualityGrade.D
        else:
            return QualityGrade.F

    def _grade_by_conservative_approach(self, module_results: Dict[str, JudgmentResult]) -> QualityGrade:
        """保守的アプローチ（最低グレードを重視）"""
        grades = [r.quality_grade for r in module_results.values()]

        # グレード順序（悪い順）
        grade_order = [QualityGrade.F, QualityGrade.E, QualityGrade.D,
                      QualityGrade.C, QualityGrade.B, QualityGrade.A]

        # F評価が30%以上あれば F
        f_count = grades.count(QualityGrade.F)
        if f_count / len(grades) >= 0.3:
            return QualityGrade.F

        # それ以外は中央値ベース
        grade_indices = [grade_order.index(grade) for grade in grades]
        median_index = int(np.median(grade_indices))
        return grade_order[median_index]

    def _calculate_overall_confidence(self, module_results: Dict[str, JudgmentResult]) -> float:
        """全体信頼度の計算"""
        if not module_results:
            return 0.0

        confidences = [r.confidence_score for r in module_results.values()]
        scores = [r.numeric_score for r in module_results.values()]

        # 基本信頼度（平均）
        base_confidence = np.mean(confidences)

        # スコア一致度による調整
        score_variance = np.var(scores)
        consistency_bonus = max(0, 1.0 - score_variance * 2)

        # モジュール数による調整
        count_factor = min(1.0, len(module_results) / 3.0)

        # 高信頼度結果のブースト
        if self.config.enable_confidence_boost:
            high_confidence_count = sum(1 for c in confidences if c > 0.8)
            boost_factor = 1.0 + (high_confidence_count / len(confidences)) * 0.2
        else:
            boost_factor = 1.0

        overall_confidence = base_confidence * consistency_bonus * count_factor * boost_factor
        return min(1.0, max(0.0, overall_confidence))

    def _perform_conflict_analysis(self, original_results: Dict[str, JudgmentResult],
                                  outliers: Dict[str, JudgmentResult]) -> Dict[str, Any]:
        """コンフリクト分析"""
        if len(original_results) <= 1:
            return {'conflicts': [], 'outliers': [], 'analysis': 'Insufficient data for conflict analysis'}

        conflicts = []
        grade_order = [QualityGrade.A, QualityGrade.B, QualityGrade.C,
                      QualityGrade.D, QualityGrade.E, QualityGrade.F]

        # ペアワイズコンフリクト検出
        module_items = list(original_results.items())
        for i in range(len(module_items)):
            for j in range(i + 1, len(module_items)):
                name1, result1 = module_items[i]
                name2, result2 = module_items[j]

                # グレード差
                grade_diff = abs(grade_order.index(result1.quality_grade) -
                               grade_order.index(result2.quality_grade))

                # スコア差
                score_diff = abs(result1.numeric_score - result2.numeric_score)

                # 大きな不一致の検出
                if grade_diff >= 2 or score_diff >= 0.4:
                    conflict_severity = 'high' if grade_diff >= 3 or score_diff >= 0.6 else 'medium'

                    conflicts.append({
                        'modules': [name1, name2],
                        'grade_difference': grade_diff,
                        'score_difference': score_diff,
                        'severity': conflict_severity,
                        'details': {
                            'module1': {
                                'grade': result1.quality_grade.value,
                                'score': result1.numeric_score,
                                'confidence': result1.confidence_score
                            },
                            'module2': {
                                'grade': result2.quality_grade.value,
                                'score': result2.numeric_score,
                                'confidence': result2.confidence_score
                            }
                        }
                    })

        # 外れ値情報
        outlier_info = []
        for name, result in outliers.items():
            outlier_info.append({
                'module': name,
                'grade': result.quality_grade.value,
                'score': result.numeric_score,
                'confidence': result.confidence_score,
                'reason': 'Statistical outlier'
            })

        # コンフリクト統計
        conflict_stats = {
            'total_conflicts': len(conflicts),
            'high_severity_conflicts': len([c for c in conflicts if c['severity'] == 'high']),
            'outlier_count': len(outliers),
            'agreement_ratio': self._calculate_agreement_ratio(original_results)
        }

        return {
            'conflicts': conflicts,
            'outliers': outlier_info,
            'statistics': conflict_stats,
            'analysis': self._generate_conflict_analysis_text(conflicts, outliers)
        }

    def _calculate_agreement_ratio(self, module_results: Dict[str, JudgmentResult]) -> float:
        """合意比率の計算"""
        grades = [r.quality_grade for r in module_results.values()]
        if not grades:
            return 0.0

        most_common_grade = Counter(grades).most_common(1)[0][0]
        agreement_count = grades.count(most_common_grade)
        return agreement_count / len(grades)

    def _generate_conflict_analysis_text(self, conflicts: List[Dict],
                                       outliers: Dict[str, JudgmentResult]) -> str:
        """コンフリクト分析テキスト生成"""
        if not conflicts and not outliers:
            return "Strong consensus across all modules"

        analysis_parts = []

        if conflicts:
            high_conflicts = [c for c in conflicts if c['severity'] == 'high']
            if high_conflicts:
                analysis_parts.append(f"Detected {len(high_conflicts)} high-severity conflicts")
            else:
                analysis_parts.append(f"Detected {len(conflicts)} moderate conflicts")

        if outliers:
            analysis_parts.append(f"Identified {len(outliers)} statistical outliers")

        return "; ".join(analysis_parts)

    def _calculate_consensus_metrics(self, module_results: Dict[str, JudgmentResult],
                                   basic_stats: Dict[str, float]) -> Dict[str, float]:
        """コンセンサス指標の計算"""
        if not module_results:
            return {}

        # 基本指標
        metrics = basic_stats.copy()

        # グレード分布
        grades = [r.quality_grade for r in module_results.values()]
        grade_distribution = Counter(grades)
        metrics['grade_diversity'] = len(grade_distribution) / len(grades)

        # 最多グレードの比率
        most_common_count = grade_distribution.most_common(1)[0][1]
        metrics['majority_grade_ratio'] = most_common_count / len(grades)

        # スコア範囲
        scores = [r.numeric_score for r in module_results.values()]
        metrics['score_range'] = max(scores) - min(scores) if scores else 0

        # 一致度指標
        metrics['agreement_ratio'] = self._calculate_agreement_ratio(module_results)

        # 信頼度指標
        confidences = [r.confidence_score for r in module_results.values()]
        metrics['high_confidence_ratio'] = sum(1 for c in confidences if c > 0.8) / len(confidences)

        return metrics

    def _integrate_recommendations(self, module_results: Dict[str, JudgmentResult]) -> List[str]:
        """推奨事項の統合"""
        # 全推奨事項の収集
        all_recommendations = []
        recommendation_sources = {}

        for module_name, result in module_results.items():
            for rec in result.recommendations:
                all_recommendations.append(rec)
                if rec not in recommendation_sources:
                    recommendation_sources[rec] = []
                recommendation_sources[rec].append(module_name)

        # 頻度と信頼度による重み付け
        weighted_recommendations = []

        for rec, sources in recommendation_sources.items():
            frequency = len(sources)

            # ソースモジュールの平均信頼度
            avg_confidence = np.mean([
                module_results[source].confidence_score for source in sources
            ])

            # 重み計算
            weight = frequency * avg_confidence

            weighted_recommendations.append({
                'recommendation': rec,
                'weight': weight,
                'frequency': frequency,
                'sources': sources,
                'avg_confidence': avg_confidence
            })

        # 重みでソート
        weighted_recommendations.sort(key=lambda x: x['weight'], reverse=True)

        # 上位推奨事項を返す
        top_recommendations = []
        for item in weighted_recommendations[:8]:  # 最大8個
            if item['frequency'] >= 2 or item['avg_confidence'] > 0.8:
                top_recommendations.append(item['recommendation'])

        return top_recommendations

    def _create_empty_result(self, reason: str) -> AggregatedJudgment:
        """空結果の作成"""
        return AggregatedJudgment(
            final_grade=QualityGrade.F,
            overall_confidence=0.0,
            module_results={},
            consensus_metrics={'error': reason},
            conflict_analysis={'error': reason},
            recommendation_summary=[f"Aggregation failed: {reason}"]
        )