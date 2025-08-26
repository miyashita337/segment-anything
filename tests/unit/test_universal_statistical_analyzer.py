#!/usr/bin/env python3
"""
Universal Statistical Analyzer ユニットテスト
INCI-004対応：BASELINE_ID必須化テスト含む

テストカバレッジ:
1. BASELINE_ID必須バリデーション
2. Cohen's d計算精度
3. Glass's delta計算
4. 信頼区間計算
5. t検定実行
6. 実用的意義判定
7. Google Sheets統合（モック）
8. エラーハンドリング
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.universal_statistical_analyzer import (
    UniversalStatisticalAnalyzer,
    UniversalAnalysisResult
)
from tools.validation.statistical_validator import TTestResult


class TestUniversalStatisticalAnalyzer:
    """Universal Statistical Analyzerの総合テストスイート"""
    
    @pytest.fixture
    def analyzer(self):
        """テスト用アナライザーインスタンス"""
        with patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient'):
            with patch('tools.progress_tracker.universal_statistical_analyzer.get_default_config'):
                return UniversalStatisticalAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        return {
            'current': np.random.normal(0.75, 0.1, 50),
            'baseline': np.random.normal(0.65, 0.12, 50)
        }
    
    # ================== BASELINE_ID必須バリデーションテスト ==================
    
    def test_baseline_id_required_with_none(self, analyzer):
        """BASELINE_IDがNoneの場合のバリデーションテスト"""
        with pytest.raises(SystemExit) as exc_info:
            analyzer.validate_baseline_required(None)
        assert exc_info.value.code == 1
    
    def test_baseline_id_required_with_empty_string(self, analyzer):
        """BASELINE_IDが空文字の場合のバリデーションテスト"""
        with pytest.raises(SystemExit) as exc_info:
            analyzer.validate_baseline_required("")
        assert exc_info.value.code == 1
    
    def test_baseline_id_required_with_whitespace(self, analyzer):
        """BASELINE_IDが空白文字のみの場合のバリデーションテスト"""
        with pytest.raises(SystemExit) as exc_info:
            analyzer.validate_baseline_required("   ")
        assert exc_info.value.code == 1
    
    def test_baseline_id_valid(self, analyzer):
        """有効なBASELINE_IDの場合のバリデーションテスト"""
        # 有効なIDでは例外が発生しないことを確認
        try:
            analyzer.validate_baseline_required("QUAL-001")
            assert True  # 例外が発生しなければ成功
        except SystemExit:
            pytest.fail("Valid BASELINE_ID raised SystemExit")
    
    # ================== Cohen's d計算テスト ==================
    
    def test_cohens_d_calculation_accuracy(self, analyzer, sample_data):
        """Cohen's d計算精度のテスト"""
        result = analyzer.calculate_enhanced_cohens_d(
            sample_data['current'],
            sample_data['baseline'],
            "TEST-CURRENT",
            "TEST-BASELINE"
        )
        
        # 手動計算で期待値を確認
        mean_diff = np.mean(sample_data['current']) - np.mean(sample_data['baseline'])
        n1, n2 = len(sample_data['current']), len(sample_data['baseline'])
        s1, s2 = np.std(sample_data['current'], ddof=1), np.std(sample_data['baseline'], ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        expected_cohens_d = mean_diff / pooled_std
        
        assert abs(result['cohens_d'] - expected_cohens_d) < 0.001
        assert 'glass_delta' in result
        assert 'confidence_interval' in result
        assert 'practical_significance' in result
        assert 'interpretation_level' in result
        assert 'effect_magnitude' in result
        assert 'improvement_rate' in result
    
    def test_cohens_d_zero_variance(self, analyzer):
        """分散がゼロの場合のCohen's d計算テスト"""
        # 全て同じ値のデータ
        current = np.ones(10) * 0.8
        baseline = np.ones(10) * 0.7
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        
        # 分散がゼロでも計算が破綻しないことを確認
        assert result['cohens_d'] == 0.0 or np.isfinite(result['cohens_d'])
        assert result['glass_delta'] == 0.0 or np.isfinite(result['glass_delta'])
    
    def test_glass_delta_calculation(self, analyzer):
        """Glass's delta計算テスト（不等分散対応）"""
        # 分散が大きく異なるデータ
        current = np.random.normal(0.8, 0.05, 30)  # 小さい分散
        baseline = np.random.normal(0.7, 0.2, 50)   # 大きい分散
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        
        # Glass's deltaは元群の標準偏差を使用
        expected_glass = (np.mean(current) - np.mean(baseline)) / np.std(baseline, ddof=1)
        assert abs(result['glass_delta'] - expected_glass) < 0.1
    
    # ================== 信頼区間計算テスト ==================
    
    def test_confidence_interval_calculation(self, analyzer):
        """95%信頼区間計算のテスト"""
        current = np.random.normal(0.75, 0.1, 100)
        baseline = np.random.normal(0.65, 0.1, 100)
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        ci = result['confidence_interval']
        
        # 信頼区間の妥当性チェック
        assert len(ci) == 2
        assert ci[0] < result['cohens_d'] < ci[1]  # Cohen's dが信頼区間内
        assert ci[1] - ci[0] > 0  # 正の区間幅
    
    def test_confidence_interval_small_sample(self, analyzer):
        """小サンプルサイズでの信頼区間計算テスト"""
        current = np.array([0.8, 0.75, 0.82, 0.78, 0.79])
        baseline = np.array([0.65, 0.68, 0.62, 0.66, 0.64])
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        ci = result['confidence_interval']
        
        # 小サンプルでは信頼区間が広くなることを確認
        assert ci[1] - ci[0] > 0.5  # 幅が0.5より大きい
    
    # ================== 実用的意義判定テスト ==================
    
    def test_practical_significance_categories(self, analyzer):
        """実用的意義の全カテゴリ判定テスト"""
        test_cases = [
            (0.05, "実用的意義なし"),
            (0.3, "小さいが実用的意義あり"),
            (0.6, "中程度の実用的意義"),
            (1.0, "大きな実用的意義"),
            (1.5, "非常に大きな実用的意義")
        ]
        
        for cohens_d, expected_sig in test_cases:
            sig = analyzer._assess_practical_significance(cohens_d, 50, 50)
            assert sig == expected_sig
    
    def test_interpretation_level_bidirectional(self, analyzer):
        """解釈レベルの双方向（改善・劣化）テスト"""
        # 改善ケース
        assert "改善" in analyzer._get_interpretation_level(0.8)
        
        # 劣化ケース
        assert "劣化" in analyzer._get_interpretation_level(-0.8)
        
        # 変化なしケース
        assert "変化なし" in analyzer._get_interpretation_level(0.01)
    
    def test_effect_magnitude_categories(self, analyzer):
        """効果の大きさカテゴリ分類テスト"""
        test_cases = [
            (0.0, "効果なし"),
            (0.15, "微小効果"),
            (0.3, "小効果"),
            (0.6, "中効果"),
            (1.0, "大効果"),
            (1.5, "非常に大きい効果"),
            (2.5, "極大効果")
        ]
        
        for cohens_d, expected_category in test_cases:
            category = analyzer._categorize_effect_magnitude(cohens_d)
            # 実装は詳細な範囲情報付きカテゴリを返すので、キーワードが含まれることを確認
            assert expected_category in category, f"Expected '{expected_category}' in '{category}' for d={cohens_d}"
    
    # ================== Google Sheets統合テスト ==================
    
    @patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient')
    def test_google_sheets_update_success(self, mock_sheets_client, analyzer):
        """Google Sheets更新成功のテスト"""
        # モックの設定
        mock_client_instance = MagicMock()
        mock_sheets_client.return_value = mock_client_instance
        analyzer.sheets_client = mock_client_instance
        
        # テストデータ
        mock_client_instance.get_sheet_values.return_value = [
            ['トラッカーID', 'ステータス'],  # ヘッダー
            ['TEST-001', '完了']  # データ行
        ]
        
        analysis_result = {
            'descriptive_stats': {
                'current': {'mean': 0.75, 'std': 0.1, 'n': 50},
                'baseline': {'mean': 0.65, 'std': 0.12, 'n': 50}
            },
            't_test_result': Mock(p_value=0.001, is_significant=True),
            'cohens_d': 0.87,
            'improvement_rate': 15.4
        }
        
        # 実行
        success = analyzer.update_google_sheets_statistics('TEST-001', analysis_result)
        
        # 検証
        assert success == True
        assert mock_client_instance.update_sheet_values.called
        assert mock_client_instance.update_sheet_values.call_count == 6  # X-AC列の6項目
    
    @patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient')
    def test_google_sheets_tracker_not_found(self, mock_sheets_client, analyzer):
        """存在しないトラッカーIDでのGoogle Sheets更新テスト"""
        mock_client_instance = MagicMock()
        mock_sheets_client.return_value = mock_client_instance
        analyzer.sheets_client = mock_client_instance
        
        # 空のシートデータ
        mock_client_instance.get_sheet_values.return_value = [['トラッカーID', 'ステータス']]
        
        analysis_result = {
            'descriptive_stats': {
                'current': {'mean': 0.75, 'std': 0.1, 'n': 50},
                'baseline': {'mean': 0.65, 'std': 0.12, 'n': 50}
            },
            't_test_result': Mock(p_value=0.001, is_significant=True),
            'cohens_d': 0.87,
            'improvement_rate': 15.4
        }
        
        success = analyzer.update_google_sheets_statistics('NONEXISTENT-001', analysis_result)
        assert success == False
    
    # ================== 統合分析テスト ==================
    
    @patch('tools.progress_tracker.universal_statistical_analyzer.StatisticalQualityAnalyzer')
    def test_integrated_analysis_success(self, mock_sqa, analyzer):
        """統合分析の成功ケーステスト"""
        # モック設定
        mock_analyzer_instance = MagicMock()
        mock_sqa.return_value = mock_analyzer_instance
        analyzer.statistical_analyzer = mock_analyzer_instance
        
        # モックデータ（異なる品質スコアでCohen's d ≠ 0になるように設定）
        mock_current_metrics = Mock(quality_scores=[0.8, 0.85, 0.9, 0.82, 0.88])  # 高品質
        mock_baseline_metrics = Mock(quality_scores=[0.6, 0.65, 0.7, 0.62, 0.68])  # 低品質
        
        # load_extraction_resultsは呼び出し順序に応じて異なる値を返す
        mock_analyzer_instance.load_extraction_results.side_effect = [mock_current_metrics, mock_baseline_metrics]
        
        # Google Sheetsモック
        with patch.object(analyzer, 'update_google_sheets_statistics', return_value=True):
            result = analyzer.run_integrated_analysis(
                'TEST-CURRENT',
                'TEST-BASELINE',
                verbose=True
            )
        
        # 検証
        assert isinstance(result, UniversalAnalysisResult)
        assert result.success == True
        assert result.current_tracker == 'TEST-CURRENT'
        assert result.baseline_tracker == 'TEST-BASELINE'
        assert result.cohens_d != 0.0
        assert result.error_message is None
    
    def test_integrated_analysis_data_load_failure(self, analyzer):
        """データ読み込み失敗時の統合分析テスト"""
        # load_extraction_resultsがNoneを返すようにモック
        with patch.object(analyzer.statistical_analyzer, 
                         'load_extraction_results', 
                         return_value=None):
            result = analyzer.run_integrated_analysis(
                'INVALID-001',
                'INVALID-002',
                verbose=False
            )
        
        # エラー結果の検証
        assert result.success == False
        assert result.error_message is not None
        assert "データ読み込み失敗" in result.error_message
    
    # ================== エラーハンドリングテスト ==================
    
    def test_division_by_zero_handling(self, analyzer):
        """ゼロ除算エラーハンドリングテスト"""
        # 同一データ（分散ゼロ）
        identical_data = np.ones(10) * 0.5
        
        result = analyzer.calculate_enhanced_cohens_d(
            identical_data,
            identical_data
        )
        
        # エラーにならず、適切な値が返ることを確認
        assert result['cohens_d'] == 0.0
        assert result['improvement_rate'] == 0.0
    
    def test_empty_data_handling(self, analyzer):
        """空データハンドリングテスト"""
        with pytest.raises(Exception):
            analyzer.calculate_enhanced_cohens_d(
                np.array([]),
                np.array([])
            )
    
    def test_mismatched_data_types(self, analyzer):
        """異なるデータ型での処理テスト"""
        # リストとnumpy arrayの混在
        current = [0.7, 0.75, 0.8]
        baseline = np.array([0.6, 0.65, 0.62])
        
        # numpy変換されて処理されることを確認
        result = analyzer.calculate_enhanced_cohens_d(
            np.array(current),
            baseline
        )
        assert 'cohens_d' in result
    
    # ================== 改善率計算テスト ==================
    
    def test_improvement_rate_positive(self, analyzer):
        """正の改善率計算テスト"""
        current = np.array([0.8, 0.82, 0.79])
        baseline = np.array([0.7, 0.68, 0.71])
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        
        expected_rate = ((np.mean(current) - np.mean(baseline)) / np.mean(baseline)) * 100
        assert abs(result['improvement_rate'] - expected_rate) < 0.01
    
    def test_improvement_rate_negative(self, analyzer):
        """負の改善率（劣化）計算テスト"""
        current = np.array([0.6, 0.62, 0.59])
        baseline = np.array([0.7, 0.68, 0.71])
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        assert result['improvement_rate'] < 0
    
    def test_improvement_rate_zero_baseline(self, analyzer):
        """ベースラインがゼロの場合の改善率計算テスト"""
        current = np.array([0.5, 0.52, 0.49])
        baseline = np.array([0.0, 0.0, 0.0])
        
        result = analyzer.calculate_enhanced_cohens_d(current, baseline)
        # ゼロ除算でエラーにならないことを確認
        assert result['improvement_rate'] == 0.0


# ================== パラメトリックテスト ==================

@pytest.mark.parametrize("baseline_id,should_exit", [
    (None, True),
    ("", True),
    ("   ", True),
    ("VALID-001", False),
    ("  VALID-002  ", False),  # 前後空白があっても有効
])
def test_baseline_validation_parametric(baseline_id, should_exit):
    """BASELINE_ID検証のパラメトリックテスト"""
    with patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient'):
        with patch('tools.progress_tracker.universal_statistical_analyzer.get_default_config'):
            analyzer = UniversalStatisticalAnalyzer()
    
    if should_exit:
        with pytest.raises(SystemExit):
            analyzer.validate_baseline_required(baseline_id)
    else:
        try:
            analyzer.validate_baseline_required(baseline_id)
            assert True
        except SystemExit:
            pytest.fail(f"Valid baseline_id {baseline_id} raised SystemExit")


@pytest.mark.parametrize("cohens_d,expected_category", [
    (0.0, "効果なし"),
    (0.1, "微小効果"),
    (0.25, "小効果"),
    (0.55, "中効果"),
    (0.9, "大効果"),
    (1.3, "非常に大きい効果"),
    (2.1, "極大効果"),
])
def test_effect_magnitude_parametric(cohens_d, expected_category):
    """効果サイズカテゴリのパラメトリックテスト"""
    with patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient'):
        with patch('tools.progress_tracker.universal_statistical_analyzer.get_default_config'):
            analyzer = UniversalStatisticalAnalyzer()
    
    category = analyzer._categorize_effect_magnitude(cohens_d)
    assert expected_category in category


# ================== 統合テスト ==================

class TestIntegrationScenarios:
    """実使用シナリオの統合テスト"""
    
    def test_full_workflow_with_mocked_data(self):
        """完全なワークフローのモック統合テスト"""
        with patch('tools.progress_tracker.universal_statistical_analyzer.GoogleSheetsClient'):
            with patch('tools.progress_tracker.universal_statistical_analyzer.get_default_config'):
                with patch('tools.progress_tracker.universal_statistical_analyzer.StatisticalQualityAnalyzer'):
                    analyzer = UniversalStatisticalAnalyzer()
                    
                    # データモック
                    mock_metrics = Mock(quality_scores=list(np.random.normal(0.7, 0.1, 30)))
                    analyzer.statistical_analyzer.load_extraction_results = Mock(return_value=mock_metrics)
                    
                    # Google Sheetsモック
                    analyzer.sheets_client.get_sheet_values = Mock(return_value=[
                        ['トラッカーID'],
                        ['INCI-004']
                    ])
                    analyzer.sheets_client.update_sheet_values = Mock(return_value=True)
                    
                    # 実行
                    result = analyzer.run_integrated_analysis(
                        'INCI-004',
                        'QUAL-001',
                        verbose=True
                    )
                    
                    # 総合検証
                    assert result.success == True
                    assert result.current_tracker == 'INCI-004'
                    assert result.baseline_tracker == 'QUAL-001'
                    assert isinstance(result.cohens_d, float)
                    assert isinstance(result.improvement_rate, float)
                    assert result.analysis_timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])