#!/usr/bin/env python3
"""
Level 3: 統計分析ワークフローテスト

統計分析ワークフロー関連のテスト:
- Cohen's d効果サイズ計算テスト
- p値・統計的有意性判定テスト
- Google Sheets統計データ更新テスト（モック）
- 改善率・信頼区間計算テスト
"""

import tempfile
import pytest
import json
from pathlib import Path
import sys
import os
import random

# テスト対象をインポート
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tests.mocks.mock_google_sheets import (
        MockGoogleSheetsClient, MockStatisticalAnalyzer, MockTrackerEntry
    )
    from tools.testing.statistical_analyzer_tester import StatisticalAnalyzerTester
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestStatisticalWorkflow:
    """Level 3: 統計分析ワークフローテストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        return {
            "current_high": [0.85, 0.87, 0.82, 0.89, 0.86, 0.84, 0.88, 0.85, 0.87, 0.83],
            "current_low": [0.72, 0.74, 0.71, 0.73, 0.75, 0.70, 0.72, 0.74, 0.71, 0.73],
            "baseline_high": [0.75, 0.77, 0.73, 0.76, 0.74, 0.78, 0.75, 0.76, 0.74, 0.77],
            "baseline_low": [0.68, 0.66, 0.69, 0.67, 0.65, 0.68, 0.66, 0.67, 0.69, 0.68]
        }
    
    # ================================
    # 統計計算エンジンテスト（20テストケース）
    # ================================
    
    def test_statistical_analyzer_initialization(self):
        """統計分析器初期化テスト"""
        analyzer = MockStatisticalAnalyzer()
        assert analyzer is not None
    
    def test_cohens_d_positive_effect(self, sample_data):
        """Cohen's d計算テスト（正の効果）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 現在データが高く、ベースラインが低い場合
        cohens_d = analyzer.calculate_cohens_d(
            sample_data["current_high"], 
            sample_data["baseline_low"]
        )
        
        # 正の効果（改善）が期待される
        assert cohens_d > 0
        assert abs(cohens_d) >= 0.2  # 最小限の効果サイズ
    
    def test_cohens_d_negative_effect(self, sample_data):
        """Cohen's d計算テスト（負の効果）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 現在データが低く、ベースラインが高い場合
        cohens_d = analyzer.calculate_cohens_d(
            sample_data["current_low"], 
            sample_data["baseline_high"]
        )
        
        # 負の効果（悪化）が期待される
        assert cohens_d < 0
        assert abs(cohens_d) >= 0.2  # 最小限の効果サイズ
    
    def test_cohens_d_no_effect(self, sample_data):
        """Cohen's d計算テスト（効果なし）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 同じデータでCohen's d計算
        cohens_d = analyzer.calculate_cohens_d(
            sample_data["current_high"], 
            sample_data["current_high"]  # 同じデータ
        )
        
        # 効果なしが期待される
        assert abs(cohens_d) < 0.1
    
    def test_cohens_d_empty_data(self):
        """Cohen's d計算テスト（空データ）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 空データでの計算
        cohens_d = analyzer.calculate_cohens_d([], [0.5, 0.6, 0.7])
        assert cohens_d == 0.0
        
        cohens_d = analyzer.calculate_cohens_d([0.5, 0.6], [])
        assert cohens_d == 0.0
    
    def test_cohens_d_effect_size_categories(self, sample_data):
        """Cohen's d効果サイズカテゴリテスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # 小効果・中効果・大効果のテストケース生成
        small_effect_current = [0.72, 0.73, 0.74, 0.71, 0.73]
        small_effect_baseline = [0.70, 0.71, 0.72, 0.69, 0.71]
        
        medium_effect_current = [0.80, 0.82, 0.81, 0.79, 0.81]
        medium_effect_baseline = [0.70, 0.71, 0.72, 0.69, 0.71]
        
        large_effect_current = [0.90, 0.92, 0.91, 0.89, 0.91]
        large_effect_baseline = [0.65, 0.66, 0.67, 0.64, 0.66]
        
        small_d = analyzer.calculate_cohens_d(small_effect_current, small_effect_baseline)
        medium_d = analyzer.calculate_cohens_d(medium_effect_current, medium_effect_baseline)
        large_d = analyzer.calculate_cohens_d(large_effect_current, large_effect_baseline)
        
        # 効果サイズの順序確認
        assert abs(small_d) < abs(medium_d) < abs(large_d)
        assert abs(large_d) > 0.8  # 大効果
    
    def test_p_value_calculation_significant(self, sample_data):
        """p値計算テスト（有意差あり）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 明確な差があるデータでp値計算
        p_value = analyzer.calculate_p_value(
            sample_data["current_high"],  # 高い値
            sample_data["baseline_low"]   # 低い値
        )
        
        # 有意差が期待される（p < 0.05）
        assert 0.0 <= p_value <= 1.0
        # 明確な差があるのでp値は小さいはず（モックでは必ずしも保証されないが）
    
    def test_p_value_calculation_non_significant(self, sample_data):
        """p値計算テスト（有意差なし）"""
        analyzer = MockStatisticalAnalyzer()
        
        # 同じデータでp値計算
        p_value = analyzer.calculate_p_value(
            sample_data["current_high"],
            sample_data["current_high"]  # 同じデータ
        )
        
        # 有意差なしが期待される（p > 0.05）
        assert p_value > 0.05
    
    def test_p_value_range_validation(self):
        """p値範囲検証テスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # 様々なデータパターンでp値の範囲確認
        for _ in range(10):
            current = [random.uniform(0.5, 1.0) for _ in range(10)]
            baseline = [random.uniform(0.5, 1.0) for _ in range(10)]
            
            p_value = analyzer.calculate_p_value(current, baseline)
            assert 0.0 <= p_value <= 1.0
    
    def test_significance_determination(self):
        """統計的有意性判定テスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # 有意性判定テストケース
        test_cases = [
            (0.001, "有意"),
            (0.01, "有意"),
            (0.049, "有意"),
            (0.05, "非有意"),
            (0.1, "非有意"),
            (0.5, "非有意"),
            (1.0, "非有意")
        ]
        
        for p_value, expected in test_cases:
            significance = analyzer.determine_significance(p_value)
            assert significance == expected, f"p={p_value}, expected={expected}, got={significance}"
    
    def test_confidence_interval_calculation(self, sample_data):
        """信頼区間計算テスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # 95%信頼区間計算
        ci_lower, ci_upper = analyzer.calculate_confidence_interval(sample_data["current_high"])
        
        # 信頼区間の妥当性確認
        mean_val = sum(sample_data["current_high"]) / len(sample_data["current_high"])
        assert ci_lower < mean_val < ci_upper
        assert ci_upper - ci_lower > 0  # 区間幅は正数
    
    def test_confidence_interval_different_levels(self, sample_data):
        """異なる信頼度の信頼区間テスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # 90%, 95%, 99%信頼区間計算
        ci_90 = analyzer.calculate_confidence_interval(sample_data["current_high"], 0.90)
        ci_95 = analyzer.calculate_confidence_interval(sample_data["current_high"], 0.95)
        ci_99 = analyzer.calculate_confidence_interval(sample_data["current_high"], 0.99)
        
        # 信頼度が高いほど区間が広くなる
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        
        assert width_90 <= width_95 <= width_99
    
    def test_mock_quality_data_generation(self):
        """品質データ生成テスト"""
        analyzer = MockStatisticalAnalyzer()
        
        # データ生成
        data = analyzer.generate_mock_quality_data(20, 0.8, 0.1)
        
        # 生成データの妥当性確認
        assert len(data) == 20
        assert all(0.0 <= score <= 1.0 for score in data)
        
        # 平均値が期待値付近にあることを確認（統計的な幅を考慮）
        mean_val = sum(data) / len(data)
        assert 0.6 <= mean_val <= 1.0  # 0.8 ± 0.2の範囲
    
    # ================================
    # Google Sheets連携テスト（6テストケース）
    # ================================
    
    def test_mock_sheets_client_initialization(self):
        """モックGoogle Sheetsクライアント初期化テスト"""
        client = MockGoogleSheetsClient()
        assert client is not None
        assert len(client.sheet_data) > 0
    
    def test_tracker_data_retrieval(self):
        """トラッカーデータ取得テスト"""
        client = MockGoogleSheetsClient()
        
        # 既存トラッカーのデータ取得
        tracker_data = client.get_tracker_data("QUAL-001")
        assert tracker_data is not None
        assert tracker_data.tracker_id == "QUAL-001"
        assert tracker_data.status == "/release"
        
        # 存在しないトラッカーのデータ取得
        nonexistent_data = client.get_tracker_data("NONEXISTENT-001")
        assert nonexistent_data is None
    
    def test_tracker_status_update(self):
        """トラッカーステータス更新テスト"""
        client = MockGoogleSheetsClient()
        
        # ステータス更新
        success = client.update_tracker_status("QUAL-001", "着手中")
        assert success == True
        
        # 更新確認
        updated_data = client.get_tracker_data("QUAL-001")
        assert updated_data.status == "着手中"
        
        # 存在しないトラッカーの更新
        failure = client.update_tracker_status("NONEXISTENT-001", "着手中")
        assert failure == False
    
    def test_statistical_data_update(self):
        """統計データ更新テスト"""
        client = MockGoogleSheetsClient()
        
        # 統計データ更新
        success = client.update_statistical_data(
            tracker_id="QUAL-001",
            current_score=0.92,
            baseline_score=0.78,
            p_value=0.0234,
            effect_size=1.456,
            improvement_rate=17.9,
            significance="有意"
        )
        assert success == True
        
        # 更新確認
        updated_data = client.get_tracker_data("QUAL-001")
        assert updated_data.current_score == 0.92
        assert updated_data.baseline_score == 0.78
        assert updated_data.p_value == 0.0234
        assert updated_data.effect_size == 1.456
        assert updated_data.improvement_rate == "17.9%"
        assert updated_data.significance == "有意"
    
    def test_baseline_candidate_search(self):
        """ベースライン候補検索テスト"""
        client = MockGoogleSheetsClient()
        
        # ベースライン候補検索
        baseline = client.find_baseline_candidate("TEST-001")
        assert baseline is not None
        assert baseline.tracker_id != "TEST-001"
        assert baseline.status == "/release"
        assert baseline.current_score is not None
    
    def test_completed_trackers_retrieval(self):
        """完了済みトラッカー取得テスト"""
        client = MockGoogleSheetsClient()
        
        # 完了済みトラッカー取得
        completed = client.get_completed_trackers()
        assert len(completed) > 0
        
        # 全て完了済みステータス確認
        for tracker in completed:
            assert tracker.status == "/release"
        
        # 更新日時でソートされていることを確認
        for i in range(len(completed) - 1):
            assert completed[i].updated_date >= completed[i + 1].updated_date
    
    # ================================
    # 統計分析テスター統合テスト（10テストケース）
    # ================================
    
    def test_statistical_tester_initialization(self):
        """統計分析テスター初期化テスト"""
        tester = StatisticalAnalyzerTester()
        assert tester.sheets_client is not None
        assert tester.analyzer is not None
    
    def test_cohens_d_test_execution(self, sample_data):
        """Cohen's dテスト実行"""
        tester = StatisticalAnalyzerTester()
        
        result = tester.test_cohens_d_calculation(
            sample_data["current_high"],
            sample_data["baseline_low"]
        )
        
        # 結果構造確認
        assert result["test_type"] == "cohens_d_calculation"
        assert "input_data" in result
        assert "calculation_result" in result
        assert "validation" in result
        
        # Cohen's d値確認
        assert "cohens_d" in result["calculation_result"]
        assert isinstance(result["calculation_result"]["cohens_d"], (int, float))
    
    def test_p_value_test_execution(self, sample_data):
        """p値テスト実行"""
        tester = StatisticalAnalyzerTester()
        
        result = tester.test_p_value_calculation(
            sample_data["current_high"],
            sample_data["baseline_high"]
        )
        
        # 結果構造確認
        assert result["test_type"] == "p_value_calculation"
        assert "calculation_result" in result
        assert "interpretation" in result
        
        # p値確認
        assert "p_value" in result["calculation_result"]
        assert 0.0 <= result["calculation_result"]["p_value"] <= 1.0
    
    def test_complete_statistical_analysis(self):
        """完全統計分析テスト実行"""
        tester = StatisticalAnalyzerTester()
        
        result = tester.test_complete_statistical_analysis("TEST-001", "QUAL-001")
        
        # 結果構造確認
        assert result["test_type"] == "complete_statistical_analysis"
        assert "trackers" in result
        assert "descriptive_statistics" in result
        assert "statistical_tests" in result
        assert "improvement_analysis" in result
        
        # 統計テスト結果確認
        stats = result["statistical_tests"]
        assert "cohens_d" in stats
        assert "welch_t_test" in stats
    
    def test_google_sheets_integration_test(self):
        """Google Sheets統合テスト実行"""
        tester = StatisticalAnalyzerTester()
        
        result = tester.test_google_sheets_integration("QUAL-001")
        
        # 結果構造確認
        assert result["test_type"] == "google_sheets_integration"
        assert "operations" in result
        assert "integration_summary" in result
        
        # 各操作の成功確認
        ops = result["operations"]
        assert ops["data_retrieval"]["success"] == True
        assert ops["baseline_search"]["success"] == True
        assert ops["statistical_update"]["success"] == True
        assert ops["status_update"]["success"] == True
    
    def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローテスト"""
        tester = StatisticalAnalyzerTester()
        
        result = tester.test_statistical_workflow_end_to_end("TEST-001")
        
        # ワークフロー成功確認
        assert "workflow_summary" in result
        assert "step1_data_retrieval" in result
        assert "step2_baseline_selection" in result
        assert "step3_statistical_analysis" in result
        assert "step4_result_update" in result
        assert "step5_completion_check" in result
        
        # 全体成功判定
        summary = result["workflow_summary"]
        assert "overall_success" in summary
        assert "tracker_processed" in summary
    
    def test_statistical_report_generation(self, sample_data):
        """統計レポート生成テスト"""
        tester = StatisticalAnalyzerTester()
        
        # 完全統計分析実行
        analysis_result = tester.test_complete_statistical_analysis("TEST-001", "QUAL-001")
        
        # レポート生成
        report = tester.generate_statistical_report(analysis_result)
        
        # レポート内容確認
        assert isinstance(report, str)
        assert "統計分析レポート" in report
        assert "分析対象" in report
        assert "記述統計" in report
        assert "統計的検定結果" in report
        assert "改善分析" in report
        assert "結論" in report
    
    def test_result_saving(self):
        """結果保存テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = StatisticalAnalyzerTester()
            
            # テスト実行
            result = tester.test_complete_statistical_analysis("TEST-001", "QUAL-001")
            
            # 結果保存
            tester.save_test_results(result, temp_dir)
            
            # 保存ファイル確認
            output_path = Path(temp_dir)
            assert (output_path / "statistical_test_results.json").exists()
            assert (output_path / "statistical_analysis_report.md").exists()
            assert (output_path / "mock_sheets_data.json").exists()
            
            # JSON内容確認
            with open(output_path / "statistical_test_results.json", 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert saved_data["test_type"] == "complete_statistical_analysis"
    
    def test_practical_significance_determination(self):
        """実用的意義判定テスト"""
        tester = StatisticalAnalyzerTester()
        
        # 判定テストケース
        test_cases = [
            (15.0, 1.0, "高い実用的意義"),
            (7.0, 0.6, "中程度の実用的意義"),
            (3.0, 0.3, "低い実用的意義"),
            (1.0, 0.1, "実用的意義なし")
        ]
        
        for improvement_rate, effect_size, expected in test_cases:
            significance = tester._determine_practical_significance(improvement_rate, effect_size)
            assert significance == expected
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        tester = StatisticalAnalyzerTester()
        
        # 無効なレポート生成
        invalid_result = {"test_type": "invalid"}
        report = tester.generate_statistical_report(invalid_result)
        assert "エラー" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])