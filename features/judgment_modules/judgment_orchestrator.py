"""
KIRO-012: 判定オーケストレーター

各判定モジュールを統合制御し、協調的な判定処理を実行
エラーハンドリング・フォールバック機能・並列実行を管理
"""

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Optional, Set

from .module_interfaces import (
    JudgmentInput,
    JudgmentResult,
    AggregatedJudgment,
    QualityGrade,
    ModuleRegistry,
    default_registry
)


class JudgmentOrchestrator:
    """判定処理オーケストレーター"""

    def __init__(self, registry: Optional[ModuleRegistry] = None,
                 max_workers: int = 3, timeout_seconds: float = 30.0):
        """
        Args:
            registry: モジュールレジストリ（None の場合はデフォルトを使用）
            max_workers: 並列実行の最大ワーカー数
            timeout_seconds: モジュール実行のタイムアウト
        """
        self.registry = registry or default_registry
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.enabled_modules: Set[str] = set()
        self.fallback_enabled = True
        self.fail_fast = False

    def enable_modules(self, module_names: List[str]):
        """実行するモジュールを有効化"""
        available_modules = set(self.registry.get_all_modules().keys())
        for name in module_names:
            if name in available_modules:
                self.enabled_modules.add(name)
            else:
                raise ValueError(f"Module '{name}' not found in registry")

    def disable_module(self, module_name: str):
        """モジュールを無効化"""
        self.enabled_modules.discard(module_name)

    def enable_all_modules(self):
        """全モジュールを有効化"""
        self.enabled_modules = set(self.registry.get_all_modules().keys())

    def execute_judgment(self, input_data: JudgmentInput,
                        selected_modules: Optional[List[str]] = None) -> AggregatedJudgment:
        """
        判定処理の実行

        Args:
            input_data: 判定対象データ
            selected_modules: 実行するモジュール名リスト（None の場合は有効化済みを使用）

        Returns:
            AggregatedJudgment: 統合判定結果
        """
        start_time = time.time()

        # 実行対象モジュールの決定
        if selected_modules is not None:
            target_modules = selected_modules
        else:
            target_modules = list(self.enabled_modules)

        if not target_modules:
            return self._create_empty_result("No modules enabled")

        # モジュール実行
        module_results = self._execute_modules(input_data, target_modules)

        # 結果統合
        aggregated_result = self._aggregate_results(module_results)

        # 実行時間記録
        total_time = time.time() - start_time
        aggregated_result.consensus_metrics['total_execution_time'] = total_time

        return aggregated_result

    def _execute_modules(self, input_data: JudgmentInput,
                        target_modules: List[str]) -> Dict[str, JudgmentResult]:
        """モジュールの並列実行"""
        modules = self.registry.get_all_modules()
        execution_order = self._determine_execution_order(target_modules)

        module_results = {}
        failed_modules = []

        # 並列実行用のタスク作成
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_module = {}

            for module_name in execution_order:
                if module_name in modules:
                    module = modules[module_name]
                    future = executor.submit(self._execute_single_module,
                                           module, input_data, module_name)
                    future_to_module[future] = module_name

            # 結果収集
            for future in concurrent.futures.as_completed(future_to_module,
                                                         timeout=self.timeout_seconds):
                module_name = future_to_module[future]
                try:
                    result = future.result()
                    module_results[module_name] = result

                    # fail_fast モードでの早期終了
                    if self.fail_fast and result.quality_grade == QualityGrade.F:
                        # 残りのタスクをキャンセル
                        for remaining_future in future_to_module:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                except Exception as e:
                    failed_modules.append(module_name)
                    error_result = self._create_error_result(
                        module_name, f"Module execution failed: {str(e)}"
                    )
                    module_results[module_name] = error_result

        # フォールバック処理
        if failed_modules and self.fallback_enabled:
            self._handle_module_failures(failed_modules, input_data, module_results)

        return module_results

    def _execute_single_module(self, module, input_data: JudgmentInput,
                              module_name: str) -> JudgmentResult:
        """単一モジュールの実行"""
        try:
            # 入力データの検証
            if not module.validate_input(input_data):
                return self._create_error_result(
                    module_name, "Input validation failed"
                )

            # モジュール実行
            result = module.judge(input_data)

            # 結果検証
            if not self._validate_result(result):
                return self._create_error_result(
                    module_name, "Invalid result format"
                )

            return result

        except Exception as e:
            return self._create_error_result(
                module_name, f"Execution error: {str(e)}"
            )

    def _determine_execution_order(self, target_modules: List[str]) -> List[str]:
        """実行順序の決定"""
        registry_order = self.registry.get_execution_order()

        # レジストリの順序に従い、対象モジュールのみを抽出
        ordered_modules = []
        for module_name in registry_order:
            if module_name in target_modules:
                ordered_modules.append(module_name)

        # レジストリにない追加モジュールを末尾に追加
        for module_name in target_modules:
            if module_name not in ordered_modules:
                ordered_modules.append(module_name)

        return ordered_modules

    def _validate_result(self, result: JudgmentResult) -> bool:
        """結果の妥当性検証"""
        if not isinstance(result, JudgmentResult):
            return False

        # 必須フィールドの検証
        if not isinstance(result.quality_grade, QualityGrade):
            return False

        if not (0.0 <= result.confidence_score <= 1.0):
            return False

        if not (0.0 <= result.numeric_score <= 1.0):
            return False

        if not isinstance(result.issues, list):
            return False

        if not isinstance(result.recommendations, list):
            return False

        if not isinstance(result.metrics, dict):
            return False

        return True

    def _handle_module_failures(self, failed_modules: List[str],
                               input_data: JudgmentInput,
                               module_results: Dict[str, JudgmentResult]):
        """モジュール失敗時のフォールバック処理"""
        # 成功したモジュールの平均値でフォールバック
        successful_results = [r for r in module_results.values()
                            if r.quality_grade != QualityGrade.F]

        if successful_results:
            avg_score = sum(r.numeric_score for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence_score for r in successful_results) / len(successful_results)

            # 平均値に基づくフォールバック結果作成
            for module_name in failed_modules:
                fallback_result = self._create_fallback_result(
                    module_name, avg_score, avg_confidence
                )
                module_results[module_name] = fallback_result

    def _aggregate_results(self, module_results: Dict[str, JudgmentResult]) -> AggregatedJudgment:
        """結果の統合処理"""
        if not module_results:
            return self._create_empty_result("No valid results")

        # 基本統計の計算
        scores = [r.numeric_score for r in module_results.values()]
        confidences = [r.confidence_score for r in module_results.values()]

        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        # 最終グレードの決定（複数手法）
        final_grade = self._determine_final_grade(module_results, avg_score)

        # コンフリクト分析
        conflict_analysis = self._analyze_conflicts(module_results)

        # 推奨事項の統合
        recommendation_summary = self._summarize_recommendations(module_results)

        # コンセンサス指標
        consensus_metrics = {
            'average_score': avg_score,
            'average_confidence': avg_confidence,
            'score_variance': float(sum((s - avg_score)**2 for s in scores) / len(scores)) if len(scores) > 1 else 0.0,
            'confidence_variance': float(sum((c - avg_confidence)**2 for c in confidences) / len(confidences)) if len(confidences) > 1 else 0.0,
            'module_count': len(module_results),
            'successful_modules': len([r for r in module_results.values() if r.quality_grade != QualityGrade.F]),
            'consensus_strength': self._calculate_consensus_strength(module_results)
        }

        return AggregatedJudgment(
            final_grade=final_grade,
            overall_confidence=avg_confidence,
            module_results=module_results,
            consensus_metrics=consensus_metrics,
            conflict_analysis=conflict_analysis,
            recommendation_summary=recommendation_summary
        )

    def _determine_final_grade(self, module_results: Dict[str, JudgmentResult],
                              avg_score: float) -> QualityGrade:
        """最終グレードの決定"""
        grades = [r.quality_grade for r in module_results.values()]

        # 投票ベースの決定
        grade_counts = {}
        for grade in grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        # 最多票のグレード
        most_common_grade = max(grade_counts.items(), key=lambda x: x[1])[0]

        # 平均スコアベースの決定
        if avg_score >= 0.85:
            score_based_grade = QualityGrade.A
        elif avg_score >= 0.70:
            score_based_grade = QualityGrade.B
        elif avg_score >= 0.55:
            score_based_grade = QualityGrade.C
        elif avg_score >= 0.40:
            score_based_grade = QualityGrade.D
        else:
            score_based_grade = QualityGrade.F

        # コンサバティブな決定（より低いグレードを選択）
        grade_order = [QualityGrade.A, QualityGrade.B, QualityGrade.C,
                      QualityGrade.D, QualityGrade.E, QualityGrade.F]

        most_common_idx = grade_order.index(most_common_grade)
        score_based_idx = grade_order.index(score_based_grade)

        final_idx = max(most_common_idx, score_based_idx)
        return grade_order[final_idx]

    def _analyze_conflicts(self, module_results: Dict[str, JudgmentResult]) -> Dict[str, any]:
        """コンフリクト分析"""
        grades = [r.quality_grade for r in module_results.values()]
        scores = [r.numeric_score for r in module_results.values()]

        # グレードの多様性
        unique_grades = len(set(grades))
        grade_diversity = unique_grades / len(grades) if grades else 0

        # スコアの範囲
        score_range = max(scores) - min(scores) if scores else 0

        # 大きな不一致の検出
        major_conflicts = []
        grade_order = [QualityGrade.A, QualityGrade.B, QualityGrade.C,
                      QualityGrade.D, QualityGrade.E, QualityGrade.F]

        for i, (name1, result1) in enumerate(module_results.items()):
            for name2, result2 in list(module_results.items())[i+1:]:
                grade_diff = abs(grade_order.index(result1.quality_grade) -
                               grade_order.index(result2.quality_grade))
                if grade_diff >= 2:  # 2段階以上の差
                    major_conflicts.append({
                        'modules': [name1, name2],
                        'grades': [result1.quality_grade.value, result2.quality_grade.value],
                        'score_diff': abs(result1.numeric_score - result2.numeric_score)
                    })

        return {
            'grade_diversity': grade_diversity,
            'score_range': score_range,
            'major_conflicts': major_conflicts,
            'conflict_count': len(major_conflicts)
        }

    def _summarize_recommendations(self, module_results: Dict[str, JudgmentResult]) -> List[str]:
        """推奨事項の統合"""
        all_recommendations = []
        for result in module_results.values():
            all_recommendations.extend(result.recommendations)

        # 重複除去と頻度順ソート
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        # 2回以上言及された推奨事項を優先
        priority_recommendations = [rec for rec, count in recommendation_counts.items()
                                  if count >= 2]
        other_recommendations = [rec for rec, count in recommendation_counts.items()
                               if count == 1]

        # 頻度順にソート
        priority_recommendations.sort(key=lambda x: recommendation_counts[x], reverse=True)

        return priority_recommendations + other_recommendations[:5]  # 最大10個まで

    def _calculate_consensus_strength(self, module_results: Dict[str, JudgmentResult]) -> float:
        """コンセンサス強度の計算"""
        if len(module_results) <= 1:
            return 1.0

        grades = [r.quality_grade for r in module_results.values()]
        scores = [r.numeric_score for r in module_results.values()]

        # グレード一致度
        most_common_grade = max(set(grades), key=grades.count)
        grade_consensus = grades.count(most_common_grade) / len(grades)

        # スコア一致度（標準偏差の逆数）
        score_std = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
        score_consensus = 1.0 / (1.0 + score_std)

        return (grade_consensus + score_consensus) / 2.0

    def _create_error_result(self, module_name: str, error_message: str) -> JudgmentResult:
        """エラー結果の作成"""
        return JudgmentResult(
            quality_grade=QualityGrade.F,
            confidence_score=0.0,
            numeric_score=0.0,
            issues=[f"Module '{module_name}': {error_message}"],
            recommendations=["Check module configuration and retry"],
            metrics={},
            processing_time=0.0,
            module_version="error"
        )

    def _create_fallback_result(self, module_name: str, avg_score: float,
                               avg_confidence: float) -> JudgmentResult:
        """フォールバック結果の作成"""
        # 平均スコアに基づくグレード
        if avg_score >= 0.85:
            grade = QualityGrade.A
        elif avg_score >= 0.70:
            grade = QualityGrade.B
        elif avg_score >= 0.55:
            grade = QualityGrade.C
        elif avg_score >= 0.40:
            grade = QualityGrade.D
        else:
            grade = QualityGrade.F

        return JudgmentResult(
            quality_grade=grade,
            confidence_score=avg_confidence * 0.7,  # 信頼度を下げる
            numeric_score=avg_score,
            issues=[f"Module '{module_name}' failed - using fallback result"],
            recommendations=["Investigate module failure"],
            metrics={'fallback': True},
            processing_time=0.0,
            module_version="fallback"
        )

    def _create_empty_result(self, reason: str) -> AggregatedJudgment:
        """空結果の作成"""
        return AggregatedJudgment(
            final_grade=QualityGrade.F,
            overall_confidence=0.0,
            module_results={},
            consensus_metrics={'error': reason},
            conflict_analysis={'error': reason},
            recommendation_summary=[f"Failed to execute judgment: {reason}"]
        )