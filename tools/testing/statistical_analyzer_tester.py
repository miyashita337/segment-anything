#!/usr/bin/env python3
"""
Level 3: 統計分析ワークフローテスター

統計分析ワークフロー全体のテスト実行・検証を行う
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile

# テスト対象とモックをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tests.mocks.mock_google_sheets import (
    MockGoogleSheetsClient, MockStatisticalAnalyzer, MockTrackerEntry
)


class StatisticalAnalyzerTester:
    """統計分析ワークフローテスタークラス"""
    
    def __init__(self):
        """統計分析ワークフローテスター初期化"""
        self.sheets_client = MockGoogleSheetsClient()
        self.analyzer = MockStatisticalAnalyzer()
    
    def test_cohens_d_calculation(self, current_scores: List[float], baseline_scores: List[float]) -> Dict[str, Any]:
        """
        Cohen's d効果サイズ計算テスト
        
        Args:
            current_scores: 現在の品質スコア
            baseline_scores: ベースラインの品質スコア
            
        Returns:
            計算結果とテスト詳細
        """
        print("🔬 Cohen's d効果サイズ計算テスト実行")
        
        # Cohen's d計算
        cohens_d = self.analyzer.calculate_cohens_d(current_scores, baseline_scores)
        
        # 効果サイズの解釈
        if abs(cohens_d) >= 0.8:
            effect_interpretation = "大効果"
        elif abs(cohens_d) >= 0.5:
            effect_interpretation = "中効果"
        elif abs(cohens_d) >= 0.2:
            effect_interpretation = "小効果"
        else:
            effect_interpretation = "効果なし"
        
        # 結果統合
        result = {
            "test_type": "cohens_d_calculation",
            "input_data": {
                "current_scores": current_scores,
                "baseline_scores": baseline_scores,
                "current_mean": sum(current_scores) / len(current_scores) if current_scores else 0,
                "baseline_mean": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
            },
            "calculation_result": {
                "cohens_d": round(cohens_d, 3),
                "effect_interpretation": effect_interpretation,
                "effect_direction": "改善" if cohens_d > 0 else "悪化" if cohens_d < 0 else "変化なし"
            },
            "validation": {
                "is_valid": abs(cohens_d) >= 0 and abs(cohens_d) <= 5.0,  # 妥当な範囲
                "data_sufficient": len(current_scores) >= 3 and len(baseline_scores) >= 3
            }
        }
        
        return result
    
    def test_p_value_calculation(self, current_scores: List[float], baseline_scores: List[float]) -> Dict[str, Any]:
        """
        p値計算テスト（ウェルチのt検定）
        
        Args:
            current_scores: 現在の品質スコア
            baseline_scores: ベースラインの品質スコア
            
        Returns:
            p値計算結果とテスト詳細
        """
        print("🔬 p値計算テスト実行")
        
        # p値計算
        p_value = self.analyzer.calculate_p_value(current_scores, baseline_scores)
        
        # 統計的有意性判定
        significance = self.analyzer.determine_significance(p_value)
        
        # 信頼区間計算
        current_ci = self.analyzer.calculate_confidence_interval(current_scores)
        baseline_ci = self.analyzer.calculate_confidence_interval(baseline_scores)
        
        # 結果統合
        result = {
            "test_type": "p_value_calculation",
            "input_data": {
                "current_scores": current_scores,
                "baseline_scores": baseline_scores,
                "sample_sizes": {
                    "current": len(current_scores),
                    "baseline": len(baseline_scores)
                }
            },
            "calculation_result": {
                "p_value": round(p_value, 4),
                "significance": significance,
                "alpha_level": 0.05,
                "confidence_intervals": {
                    "current": {"lower": round(current_ci[0], 3), "upper": round(current_ci[1], 3)},
                    "baseline": {"lower": round(baseline_ci[0], 3), "upper": round(baseline_ci[1], 3)}
                }
            },
            "interpretation": {
                "is_significant": significance == "有意",
                "confidence_level": "95%",
                "null_hypothesis": "現在とベースラインに差はない"
            }
        }
        
        return result
    
    def test_complete_statistical_analysis(self, tracker_id: str, baseline_tracker_id: str) -> Dict[str, Any]:
        """
        完全統計分析テスト
        
        Args:
            tracker_id: 現在のトラッカーID
            baseline_tracker_id: ベースライントラッカーID
            
        Returns:
            完全統計分析結果
        """
        print(f"🔬 完全統計分析テスト実行: {tracker_id} vs {baseline_tracker_id}")
        
        # 模擬データ生成（実際のシステムでは実際のワークスペースから取得）
        current_scores = self.analyzer.generate_mock_quality_data(15, 0.75, 0.15)
        baseline_scores = self.analyzer.generate_mock_quality_data(12, 0.70, 0.12)
        
        # 統計計算実行
        cohens_d_result = self.test_cohens_d_calculation(current_scores, baseline_scores)
        p_value_result = self.test_p_value_calculation(current_scores, baseline_scores)
        
        # 改善率計算
        current_mean = sum(current_scores) / len(current_scores)
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        improvement_rate = ((current_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
        
        # 結果統合
        complete_result = {
            "test_type": "complete_statistical_analysis",
            "trackers": {
                "current": tracker_id,
                "baseline": baseline_tracker_id
            },
            "descriptive_statistics": {
                "current": {
                    "mean": round(current_mean, 3),
                    "n_samples": len(current_scores),
                    "min": round(min(current_scores), 3),
                    "max": round(max(current_scores), 3)
                },
                "baseline": {
                    "mean": round(baseline_mean, 3),
                    "n_samples": len(baseline_scores),
                    "min": round(min(baseline_scores), 3),
                    "max": round(max(baseline_scores), 3)
                }
            },
            "statistical_tests": {
                "cohens_d": cohens_d_result["calculation_result"],
                "welch_t_test": p_value_result["calculation_result"]
            },
            "improvement_analysis": {
                "improvement_rate": round(improvement_rate, 1),
                "improvement_direction": "向上" if improvement_rate > 0 else "低下" if improvement_rate < 0 else "変化なし",
                "practical_significance": self._determine_practical_significance(
                    abs(improvement_rate), abs(cohens_d_result["calculation_result"]["cohens_d"])
                )
            }
        }
        
        return complete_result
    
    def test_google_sheets_integration(self, tracker_id: str) -> Dict[str, Any]:
        """
        Google Sheets統合テスト
        
        Args:
            tracker_id: テスト対象トラッカーID
            
        Returns:
            Google Sheets連携テスト結果
        """
        print(f"🔬 Google Sheets連携テスト実行: {tracker_id}")
        
        # 1. データ取得テスト
        tracker_data = self.sheets_client.get_tracker_data(tracker_id)
        
        # 2. ベースライン候補検索テスト
        baseline_candidate = self.sheets_client.find_baseline_candidate(tracker_id)
        
        # 3. 統計データ更新テスト
        update_success = self.sheets_client.update_statistical_data(
            tracker_id=tracker_id,
            current_score=0.85,
            baseline_score=0.75,
            p_value=0.0234,
            effect_size=1.245,
            improvement_rate=13.3,
            significance="有意"
        )
        
        # 4. ステータス更新テスト
        status_update_success = self.sheets_client.update_tracker_status(tracker_id, "/release")
        
        # テスト結果統合
        result = {
            "test_type": "google_sheets_integration",
            "tracker_id": tracker_id,
            "operations": {
                "data_retrieval": {
                    "success": tracker_data is not None,
                    "data_found": tracker_data.tracker_id if tracker_data else None
                },
                "baseline_search": {
                    "success": baseline_candidate is not None,
                    "baseline_found": baseline_candidate.tracker_id if baseline_candidate else None
                },
                "statistical_update": {
                    "success": update_success,
                    "data_updated": update_success
                },
                "status_update": {
                    "success": status_update_success,
                    "status_changed": status_update_success
                }
            },
            "integration_summary": {
                "all_operations_successful": all([
                    tracker_data is not None,
                    baseline_candidate is not None,
                    update_success,
                    status_update_success
                ]),
                "api_response_time": "< 1s (モック)"
            }
        }
        
        return result
    
    def test_statistical_workflow_end_to_end(self, tracker_id: str) -> Dict[str, Any]:
        """
        統計ワークフローエンドツーエンドテスト
        
        Args:
            tracker_id: テスト対象トラッカーID
            
        Returns:
            エンドツーエンドテスト結果
        """
        print(f"🔬 統計ワークフローE2Eテスト実行: {tracker_id}")
        
        workflow_results = {}
        
        try:
            # Step 1: Google Sheetsからデータ取得
            print("  📊 Step 1: データ取得")
            sheets_test = self.test_google_sheets_integration(tracker_id)
            workflow_results["step1_data_retrieval"] = sheets_test
            
            # Step 2: ベースライン候補決定
            print("  📊 Step 2: ベースライン決定")
            baseline_candidate = self.sheets_client.find_baseline_candidate(tracker_id)
            if baseline_candidate:
                workflow_results["step2_baseline_selection"] = {
                    "success": True,
                    "baseline_tracker": baseline_candidate.tracker_id,
                    "baseline_score": baseline_candidate.current_score
                }
            else:
                workflow_results["step2_baseline_selection"] = {
                    "success": False,
                    "error": "ベースライン候補が見つかりません"
                }
                return workflow_results
            
            # Step 3: 統計分析実行
            print("  📊 Step 3: 統計分析実行")
            statistical_analysis = self.test_complete_statistical_analysis(
                tracker_id, baseline_candidate.tracker_id
            )
            workflow_results["step3_statistical_analysis"] = statistical_analysis
            
            # Step 4: 結果をGoogle Sheetsに更新
            print("  📊 Step 4: 結果更新")
            stats = statistical_analysis["statistical_tests"]
            improvement = statistical_analysis["improvement_analysis"]
            
            update_success = self.sheets_client.update_statistical_data(
                tracker_id=tracker_id,
                current_score=statistical_analysis["descriptive_statistics"]["current"]["mean"],
                baseline_score=statistical_analysis["descriptive_statistics"]["baseline"]["mean"],
                p_value=stats["welch_t_test"]["p_value"],
                effect_size=stats["cohens_d"]["cohens_d"],
                improvement_rate=improvement["improvement_rate"],
                significance=stats["welch_t_test"]["significance"]
            )
            
            workflow_results["step4_result_update"] = {
                "success": update_success,
                "updated_tracker": tracker_id
            }
            
            # Step 5: ワークフロー完了確認
            print("  📊 Step 5: 完了確認")
            final_data = self.sheets_client.get_tracker_data(tracker_id)
            workflow_results["step5_completion_check"] = {
                "success": final_data is not None and final_data.current_score is not None,
                "final_status": final_data.status if final_data else None,
                "statistical_data_complete": all([
                    final_data.current_score is not None,
                    final_data.baseline_score is not None,
                    final_data.p_value is not None,
                    final_data.effect_size is not None
                ]) if final_data else False
            }
            
            # 全体成功判定
            workflow_results["workflow_summary"] = {
                "overall_success": all([
                    workflow_results["step1_data_retrieval"]["integration_summary"]["all_operations_successful"],
                    workflow_results["step2_baseline_selection"]["success"],
                    workflow_results["step4_result_update"]["success"],
                    workflow_results["step5_completion_check"]["success"]
                ]),
                "execution_time": "模擬実行（< 5秒）",
                "tracker_processed": tracker_id
            }
            
        except Exception as e:
            workflow_results["error"] = {
                "message": str(e),
                "step": "統計ワークフロー実行中"
            }
            workflow_results["workflow_summary"] = {
                "overall_success": False,
                "error_occurred": True
            }
        
        return workflow_results
    
    def _determine_practical_significance(self, improvement_rate: float, effect_size: float) -> str:
        """
        実用的意義判定
        
        Args:
            improvement_rate: 改善率
            effect_size: 効果サイズ
            
        Returns:
            実用的意義判定結果
        """
        if improvement_rate >= 10.0 and effect_size >= 0.8:
            return "高い実用的意義"
        elif improvement_rate >= 5.0 and effect_size >= 0.5:
            return "中程度の実用的意義"
        elif improvement_rate >= 2.0 and effect_size >= 0.2:
            return "低い実用的意義"
        else:
            return "実用的意義なし"
    
    def generate_statistical_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        統計分析レポート生成
        
        Args:
            analysis_results: 分析結果
            
        Returns:
            統計レポート（Markdown形式）
        """
        if analysis_results["test_type"] != "complete_statistical_analysis":
            return "エラー: 完全統計分析結果が必要です"
        
        stats = analysis_results["statistical_tests"]
        desc = analysis_results["descriptive_statistics"]
        improvement = analysis_results["improvement_analysis"]
        trackers = analysis_results["trackers"]
        
        report = f"""# 統計分析レポート

## 分析対象
- **現在のトラッカー**: {trackers["current"]}
- **ベースライン**: {trackers["baseline"]}

## 記述統計
### 現在データ
- **平均品質スコア**: {desc["current"]["mean"]}
- **サンプル数**: {desc["current"]["n_samples"]}
- **範囲**: {desc["current"]["min"]} - {desc["current"]["max"]}

### ベースラインデータ  
- **平均品質スコア**: {desc["baseline"]["mean"]}
- **サンプル数**: {desc["baseline"]["n_samples"]}
- **範囲**: {desc["baseline"]["min"]} - {desc["baseline"]["max"]}

## 統計的検定結果
### ウェルチのt検定
- **p値**: {stats["welch_t_test"]["p_value"]}
- **統計的有意性**: {stats["welch_t_test"]["significance"]} (α = 0.05)
- **95%信頼区間（現在）**: [{stats["welch_t_test"]["confidence_intervals"]["current"]["lower"]}, {stats["welch_t_test"]["confidence_intervals"]["current"]["upper"]}]
- **95%信頼区間（ベースライン）**: [{stats["welch_t_test"]["confidence_intervals"]["baseline"]["lower"]}, {stats["welch_t_test"]["confidence_intervals"]["baseline"]["upper"]}]

### 効果サイズ分析
- **Cohen's d**: {stats["cohens_d"]["cohens_d"]}
- **効果サイズ判定**: {stats["cohens_d"]["effect_interpretation"]}
- **効果方向**: {stats["cohens_d"]["effect_direction"]}

## 改善分析
- **改善率**: {improvement["improvement_rate"]}%
- **改善方向**: {improvement["improvement_direction"]}
- **実用的意義**: {improvement["practical_significance"]}

## 結論
{"統計的に有意" if stats["welch_t_test"]["significance"] == "有意" else "統計的に非有意"}な{"改善" if improvement["improvement_rate"] > 0 else "変化"}が確認されました。
効果サイズは{stats["cohens_d"]["effect_interpretation"]}であり、{improvement["practical_significance"]}があると判定されます。
"""
        
        return report
    
    def save_test_results(self, test_results: Dict[str, Any], output_dir: str) -> None:
        """
        統計テスト結果保存
        
        Args:
            test_results: テスト結果
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # テスト結果JSON保存
        with open(output_path / "statistical_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        # 統計レポート保存
        if test_results.get("test_type") == "complete_statistical_analysis":
            report = self.generate_statistical_report(test_results)
            with open(output_path / "statistical_analysis_report.md", 'w', encoding='utf-8') as f:
                f.write(report)
        
        # Google Sheetsデータエクスポート
        sheets_export = self.sheets_client.export_to_json()
        with open(output_path / "mock_sheets_data.json", 'w', encoding='utf-8') as f:
            f.write(sheets_export)
        
        print(f"✅ 統計テスト結果保存完了: {output_dir}")


def main():
    """CLI実行用メイン関数"""
    if len(sys.argv) < 3:
        print("Usage: python statistical_analyzer_tester.py <test_type> <tracker_id> [baseline_tracker_id] [output_dir]")
        print("Test types: cohens_d, p_value, complete, sheets, workflow")
        sys.exit(1)
    
    test_type = sys.argv[1]
    tracker_id = sys.argv[2]
    baseline_tracker_id = sys.argv[3] if len(sys.argv) > 3 else "QUAL-001"
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "/tmp/statistical_test_output"
    
    # テスター初期化
    tester = StatisticalAnalyzerTester()
    
    try:
        if test_type == "cohens_d":
            # Cohen's dテスト
            current = tester.analyzer.generate_mock_quality_data(10, 0.8, 0.1)
            baseline = tester.analyzer.generate_mock_quality_data(10, 0.7, 0.1)
            result = tester.test_cohens_d_calculation(current, baseline)
        elif test_type == "p_value":
            # p値テスト
            current = tester.analyzer.generate_mock_quality_data(15, 0.75, 0.12)
            baseline = tester.analyzer.generate_mock_quality_data(12, 0.70, 0.10)
            result = tester.test_p_value_calculation(current, baseline)
        elif test_type == "complete":
            # 完全統計分析テスト
            result = tester.test_complete_statistical_analysis(tracker_id, baseline_tracker_id)
        elif test_type == "sheets":
            # Google Sheets連携テスト
            result = tester.test_google_sheets_integration(tracker_id)
        elif test_type == "workflow":
            # エンドツーエンドワークフローテスト
            result = tester.test_statistical_workflow_end_to_end(tracker_id)
        else:
            print(f"❌ 未サポートテストタイプ: {test_type}")
            sys.exit(1)
        
        # 結果保存
        tester.save_test_results(result, output_dir)
        
        print("✅ 統計分析ワークフローテスト完了")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()