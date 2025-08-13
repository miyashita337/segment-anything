"""
QCC-022: 統計的有意性検定システムの統合テスト
"""

import unittest
import numpy as np
import json
import tempfile
from pathlib import Path
import shutil
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.validation import StatisticalValidator, StatisticalReporter
from features.evaluation.statistical_quality_analyzer import (
    StatisticalQualityAnalyzer, QualityMetrics
)


class TestQCC022Integration(unittest.TestCase):
    """QCC-022統合テスト"""
    
    @classmethod
    def setUpClass(cls):
        """テストクラスの初期化"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.workspace_base = Path(cls.temp_dir) / "test_workspace"
        cls.workspace_base.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """テストクラスのクリーンアップ"""
        if Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """各テストの前準備"""
        self.validator = StatisticalValidator()
        self.reporter = StatisticalReporter(output_dir=self.temp_dir)
        self.analyzer = StatisticalQualityAnalyzer(
            workspace_base=str(self.workspace_base)
        )
        
        # テスト用トラッカーデータ作成
        self._create_test_tracker_data()
    
    def _create_test_tracker_data(self):
        """テスト用トラッカーデータを作成"""
        # ベースライントラッカー（低品質）
        baseline_dir = self.workspace_base / "TEST-BASELINE"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(42)
        baseline_scores = np.random.beta(2, 5, 30).tolist()  # 低めのスコア
        
        baseline_data = {
            "tracker_id": "TEST-BASELINE",
            "success_rate": 0.45,
            "total_images": 30,
            "quality_scores": baseline_scores,
            "results": [
                {
                    "filename": f"image_{i}.jpg",
                    "quality_score": score,
                    "extraction_time": np.random.gamma(2, 2)
                }
                for i, score in enumerate(baseline_scores)
            ]
        }
        
        with open(baseline_dir / "extraction_result.json", 'w') as f:
            json.dump(baseline_data, f)
        
        # 改善版トラッカー（高品質）
        improved_dir = self.workspace_base / "TEST-IMPROVED"
        improved_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(43)  # 異なるシードで確実に差を作る
        improved_scores = np.random.beta(6, 2, 30).tolist()  # 高めのスコア
        
        improved_data = {
            "tracker_id": "TEST-IMPROVED",
            "success_rate": 0.78,
            "total_images": 30,
            "quality_scores": improved_scores,
            "results": [
                {
                    "filename": f"image_{i}.jpg",
                    "quality_score": score,
                    "extraction_time": np.random.gamma(1.5, 2)
                }
                for i, score in enumerate(improved_scores)
            ]
        }
        
        with open(improved_dir / "extraction_result.json", 'w') as f:
            json.dump(improved_data, f)
    
    def test_end_to_end_comparison(self):
        """エンドツーエンドの比較テスト"""
        # トラッカー比較（改善版が後、ベースラインが前）
        result = self.analyzer.compare_trackers(
            "TEST-IMPROVED",
            "TEST-BASELINE", 
            metric='quality_score'
        )
        
        # 改善版が有意に良いはず
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)
        self.assertGreater(abs(result.effect_size), 0.5)  # 中程度以上の効果
        
        # 改善版の平均が高い（group_aが改善版）
        self.assertGreater(result.mean_a, result.mean_b)
    
    def test_improvement_analysis(self):
        """改善効果分析のテスト"""
        analysis = self.analyzer.analyze_improvement(
            "TEST-BASELINE",
            "TEST-IMPROVED"
        )
        
        # 分析結果の検証
        self.assertIn('quality_comparison', analysis)
        self.assertIn('success_rate_comparison', analysis)
        self.assertIn('sample_sizes', analysis)
        
        # 品質改善の確認
        quality_comp = analysis['quality_comparison']
        self.assertGreater(quality_comp['improved_mean'], quality_comp['baseline_mean'])
        self.assertTrue(quality_comp['is_significant'])
        
        # 成功率改善の確認
        success_comp = analysis['success_rate_comparison']
        self.assertGreater(success_comp['improved'], success_comp['baseline'])
    
    def test_report_generation(self):
        """レポート生成のテスト"""
        # t検定実行
        baseline_metrics = self.analyzer.load_extraction_results("TEST-BASELINE")
        improved_metrics = self.analyzer.load_extraction_results("TEST-IMPROVED")
        
        result = self.validator.welch_t_test(
            baseline_metrics.quality_scores,
            improved_metrics.quality_scores
        )
        
        # JSONレポート生成
        json_report = self.reporter.generate_json_report(
            result,
            test_name="Integration Test",
            metadata={"test": True}
        )
        
        self.assertIn('results', json_report)
        self.assertIn('statistical_test', json_report['results'])
        self.assertIn('confidence_interval', json_report['results'])
        self.assertIn('effect_size', json_report['results'])
        
        # JSONファイル保存
        json_path = self.reporter.save_json_report(
            result,
            filename="test_report.json"
        )
        
        self.assertTrue(json_path.exists())
        
        # HTMLレポート生成
        html_content = self.reporter.generate_html_report(
            result,
            test_name="統合テスト",
            group_a_name="ベースライン",
            group_b_name="改善版"
        )
        
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('統合テスト', html_content)
        # p値は非常に小さい場合 "< 0.0001" と表示される
        self.assertTrue(
            str(result.p_value)[:6] in html_content or "< 0.0001" in html_content
        )
        
        # HTMLファイル保存
        html_path = self.reporter.save_html_report(
            result,
            filename="test_report.html"
        )
        
        self.assertTrue(html_path.exists())
    
    def test_visualization(self):
        """可視化機能のテスト"""
        # データ読み込み
        baseline_metrics = self.analyzer.load_extraction_results("TEST-BASELINE")
        improved_metrics = self.analyzer.load_extraction_results("TEST-IMPROVED")
        
        result = self.validator.welch_t_test(
            baseline_metrics.quality_scores,
            improved_metrics.quality_scores
        )
        
        # プロット生成
        fig = self.reporter.plot_comparison(
            baseline_metrics.quality_scores,
            improved_metrics.quality_scores,
            result,
            group_a_name="ベースライン",
            group_b_name="改善版",
            title="品質比較"
        )
        
        self.assertIsNotNone(fig)
        
        # プロット保存
        plot_path = self.reporter.save_plot(
            baseline_metrics.quality_scores,
            improved_metrics.quality_scores,
            result,
            filename="test_plot.png"
        )
        
        self.assertTrue(plot_path.exists())
    
    def test_batch_comparison(self):
        """バッチ比較のテスト"""
        # 3つ目のトラッカー作成
        middle_dir = self.workspace_base / "TEST-MIDDLE"
        middle_dir.mkdir(parents=True, exist_ok=True)
        
        middle_scores = np.random.beta(3, 3, 25).tolist()
        
        middle_data = {
            "tracker_id": "TEST-MIDDLE",
            "success_rate": 0.60,
            "total_images": 25,
            "quality_scores": middle_scores
        }
        
        with open(middle_dir / "extraction_result.json", 'w') as f:
            json.dump(middle_data, f)
        
        # バッチ比較実行
        summary = self.analyzer.batch_compare(
            ["TEST-BASELINE", "TEST-IMPROVED", "TEST-MIDDLE"],
            baseline="TEST-BASELINE"
        )
        
        self.assertEqual(summary['n_trackers'], 3)
        self.assertEqual(summary['baseline'], "TEST-BASELINE")
        self.assertIn('comparisons', summary)
        self.assertIn('ranking', summary)
        
        # ランキング確認（改善版が1位のはず）
        ranking = summary['ranking']
        self.assertEqual(ranking[0]['tracker_id'], "TEST-IMPROVED")
    
    def test_statistical_dashboard_generation(self):
        """統計ダッシュボード生成のテスト"""
        dashboard_path = self.analyzer.generate_statistical_dashboard(
            "TEST-IMPROVED",
            comparison_trackers=["TEST-BASELINE"],
            output_dir=self.temp_dir
        )
        
        self.assertTrue(dashboard_path.exists())
        
        # HTMLコンテンツ確認
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('統計分析ダッシュボード', content)
        self.assertIn('TEST-IMPROVED', content)
        self.assertIn('基本統計量', content)
        self.assertIn('統計的比較分析', content)
    
    def test_multiple_comparison_correction(self):
        """多重比較補正のテスト"""
        # 複数の比較を実行
        groups = {
            'TEST-BASELINE': self.analyzer.load_extraction_results("TEST-BASELINE").quality_scores,
            'TEST-IMPROVED': self.analyzer.load_extraction_results("TEST-IMPROVED").quality_scores
        }
        
        # 総当たり比較
        results = self.validator.compare_multiple_groups(groups)
        
        # 多重比較レポート生成
        report = self.reporter.generate_multiple_comparison_report(
            results,
            correction_method='bonferroni'
        )
        
        self.assertIn('n_comparisons', report)
        self.assertIn('correction_method', report)
        self.assertIn('comparisons', report)
        
        # 補正後のp値確認
        for comp_name, comp_data in report['comparisons'].items():
            self.assertIn('original_p_value', comp_data)
            self.assertIn('corrected_p_value', comp_data)
            # 補正後のp値は元の値以上
            self.assertGreaterEqual(
                comp_data['corrected_p_value'],
                comp_data['original_p_value']
            )
    
    def test_edge_cases_handling(self):
        """エッジケース処理のテスト"""
        # 存在しないトラッカー
        with self.assertRaises(FileNotFoundError):
            self.analyzer.load_extraction_results("NON-EXISTENT")
        
        # 空のデータ
        empty_dir = self.workspace_base / "TEST-EMPTY"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        empty_data = {
            "tracker_id": "TEST-EMPTY",
            "success_rate": 0,
            "total_images": 0,
            "quality_scores": []
        }
        
        with open(empty_dir / "extraction_result.json", 'w') as f:
            json.dump(empty_data, f)
        
        # 空データの処理（エラーにならないこと）
        try:
            metrics = self.analyzer.load_extraction_results("TEST-EMPTY")
            self.assertEqual(metrics.sample_size, 0)
        except Exception as e:
            self.fail(f"空データ処理でエラー: {e}")
    
    def test_real_world_scenario(self):
        """実際の使用シナリオのテスト"""
        # 1. 複数回の実験結果を記録
        experiments = []
        for i in range(3):
            np.random.seed(i)
            scores = np.random.beta(3 + i, 5 - i*0.5, 20).tolist()
            experiments.append(scores)
        
        # 2. 各実験間で比較
        for i in range(len(experiments) - 1):
            result = self.validator.welch_t_test(
                experiments[i],
                experiments[i + 1]
            )
            
            # 結果の妥当性確認
            self.assertIsNotNone(result.p_value)
            self.assertIsNotNone(result.effect_size)
            self.assertIsNotNone(result.interpretation)
        
        # 3. 最終レポート生成
        final_result = self.validator.welch_t_test(
            experiments[0],  # 初期状態
            experiments[-1]  # 最終状態
        )
        
        # レポート保存
        report_path = self.reporter.save_json_report(
            final_result,
            filename="final_experiment_report.json",
            test_name="実験結果比較",
            metadata={
                "n_experiments": len(experiments),
                "sample_sizes": [len(exp) for exp in experiments]
            }
        )
        
        self.assertTrue(report_path.exists())


if __name__ == '__main__':
    unittest.main()