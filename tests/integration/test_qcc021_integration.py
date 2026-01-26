#!/usr/bin/env python3
"""
QCC-021統合テスト: SampleSizeValidatorとQCA-001の統合動作テスト
実際のワークスペースデータとの結合テスト
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.analysis.sample_size_validator import SampleSizeValidator, TestType
from tools.scripts.qcc021_qca001_validation import QCA001ValidationIntegrator


class TestQCC021Integration(unittest.TestCase):
    """QCC-021統合テストクラス"""

    def setUp(self):
        """テスト前準備"""
        self.validator = SampleSizeValidator()

        # テスト用一時ディレクトリ作成
        self.test_workspace = tempfile.mkdtemp()
        self.test_workspace_path = Path(self.test_workspace)

        # QCA-001モックワークスペース作成
        self.mock_qca001_workspace = self.test_workspace_path / "QCA-001"
        self.mock_extraction_dir = self.mock_qca001_workspace / "extraction"
        self.mock_extraction_dir.mkdir(parents=True)

        # モック画像ファイル作成（17枚のQCA-001サンプル）
        self.mock_image_files = []
        for i in range(17):
            mock_image = self.mock_extraction_dir / f"extracted_kana05_{i:04d}.jpg"
            mock_image.write_text("mock image data")  # ダミーデータ
            self.mock_image_files.append(mock_image)

    def tearDown(self):
        """テスト後クリーンアップ"""
        shutil.rmtree(self.test_workspace, ignore_errors=True)

    def test_qca001_integrator_initialization(self):
        """QCA001ValidationIntegratorの初期化テスト"""
        # モックパスでintegrator初期化
        with patch("tools.scripts.qcc021_qca001_validation.Path") as mock_path:
            mock_path.return_value = self.test_workspace_path
            integrator = QCA001ValidationIntegrator()

            self.assertIsInstance(integrator.validator, SampleSizeValidator)
            self.assertEqual(integrator.validator.default_power, 0.8)
            self.assertEqual(integrator.validator.default_alpha, 0.05)

    def test_qca001_sample_counting(self):
        """QCA-001サンプル数カウントテスト"""
        # integratorのworkspace_baseをテスト用に変更
        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        # QCA-001のワークスペース確認をテスト
        qca001_workspace = integrator.workspace_base / "QCA-001"
        self.assertTrue(qca001_workspace.exists())

        # 抽出ディレクトリ確認
        extraction_dir = qca001_workspace / "extraction"
        self.assertTrue(extraction_dir.exists())

        # 画像ファイル数確認
        image_files = list(extraction_dir.glob("*.jpg"))
        self.assertEqual(len(image_files), 17)

    def test_comprehensive_validation_flow(self):
        """包括的検証フローテスト"""
        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        try:
            # QCA-001検証実行
            validation_report = integrator.analyze_qca001_sample_adequacy()

            # レポート構造確認
            self.assertIn("qca001_sample_info", validation_report)
            self.assertIn("statistical_validation", validation_report)
            self.assertIn("detailed_requirements", validation_report)
            self.assertIn("warnings_and_suggestions", validation_report)
            self.assertIn("analysis_results", validation_report)

            # サンプル情報確認
            sample_info = validation_report["qca001_sample_info"]
            self.assertEqual(sample_info["current_sample_size"], 17)
            self.assertIn("workspace_path", sample_info)
            self.assertIn("image_files", sample_info)

            # 統計的検証確認
            stat_validation = validation_report["statistical_validation"]
            self.assertIn("overall_adequacy", stat_validation)
            self.assertIn("recommended_n", stat_validation)
            self.assertIn("current_power", stat_validation)
            self.assertIn("precision_assessment", stat_validation)

            # 数値妥当性確認
            self.assertIsInstance(stat_validation["overall_adequacy"], bool)
            self.assertIsInstance(stat_validation["recommended_n"], int)
            self.assertGreater(stat_validation["recommended_n"], 0)
            self.assertGreaterEqual(stat_validation["current_power"], 0.0)
            self.assertLessEqual(stat_validation["current_power"], 1.0)
            self.assertIn(stat_validation["precision_assessment"], ["高精度", "中精度", "低精度"])

        except FileNotFoundError as e:
            # ワークスペースが見つからない場合のエラーハンドリング確認
            self.assertIn("ワークスペースが見つかりません", str(e))

    def test_qca001_specific_scenarios(self):
        """QCA-001特化シナリオテスト"""
        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        # QCA-001特化シナリオの妥当性確認
        expected_scenarios = [
            "QCA-001作者別品質差検出",
            "QCA-001パラメータ最適化効果",
            "QCA-001成功率改善検証",
            "QCA-001品質スコア基準値比較",
        ]

        # 手動でQCA-001シナリオの要件計算
        validator = SampleSizeValidator()

        # 作者別品質差（小効果）
        author_diff_n = validator.calculate_required_sample_size(
            TestType.TWO_SAMPLE_T, effect_size=0.2
        )

        # パラメータ最適化（中効果）
        optimization_n = validator.calculate_required_sample_size(
            TestType.PAIRED_T, effect_size=0.5
        )

        # 成功率改善（比率）
        success_rate_n = validator.calculate_required_sample_size(
            TestType.PROPORTION, effect_size=0.25
        )

        # 品質スコア基準値（1標本）
        quality_score_n = validator.calculate_required_sample_size(
            TestType.ONE_SAMPLE_T, effect_size=0.3
        )

        # 理論値妥当性確認
        self.assertGreater(author_diff_n, 0)
        self.assertGreater(optimization_n, 0)
        self.assertGreater(success_rate_n, 0)
        self.assertGreater(quality_score_n, 0)

        # 効果サイズと必要サンプル数の関係確認
        # 小効果（作者差）は中効果（最適化）より多くのサンプルが必要
        self.assertGreater(author_diff_n, optimization_n)

    def test_report_generation_and_saving(self):
        """レポート生成・保存テスト"""
        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        # モックレポートデータ作成
        mock_report = {
            "qca001_sample_info": {
                "current_sample_size": 17,
                "workspace_path": str(self.mock_qca001_workspace),
                "image_files": [f.name for f in self.mock_image_files[:10]],
            },
            "statistical_validation": {
                "overall_adequacy": False,
                "recommended_n": 64,
                "current_power": 0.42,
                "precision_assessment": "中精度",
            },
            "detailed_requirements": [
                {
                    "scenario": "two_sample_t",
                    "current_n": 17,
                    "required_n": 64,
                    "is_adequate": False,
                    "confidence_width": 0.35,
                    "precision_level": "中精度",
                    "effect_size": 0.2,
                }
            ],
            "warnings_and_suggestions": {
                "statistical_warnings": ["サンプル不足警告"],
                "improvement_suggestions": ["追加サンプル推奨"],
                "qca001_specific_recommendations": ["QCA-001特化改善提案"],
            },
            "analysis_results": {
                "power_analysis": {"current_power": 0.42},
                "precision_analysis": {"precision_level": "中精度"},
                "effect_size_analysis": {},
                "overall_assessment": "テスト評価",
            },
        }

        # レポート保存テスト
        report_path = integrator.save_validation_report(mock_report, "QCC-021")

        # ファイル生成確認
        self.assertTrue(Path(report_path).exists())

        # JSONファイル内容確認
        with open(report_path, "r", encoding="utf-8") as f:
            saved_report = json.load(f)

        self.assertEqual(saved_report["qca001_sample_info"]["current_sample_size"], 17)
        self.assertEqual(saved_report["statistical_validation"]["recommended_n"], 64)

        # テキストサマリー確認
        txt_path = Path(report_path).parent / "qca001_sample_validation_summary.txt"
        self.assertTrue(txt_path.exists())

        # テキスト内容確認
        txt_content = txt_path.read_text(encoding="utf-8")
        self.assertIn("QCC-021", txt_content)
        self.assertIn("サンプル数: 17枚", txt_content)
        self.assertIn("推奨サンプル数: 64", txt_content)

    def test_error_handling_missing_workspace(self):
        """ワークスペース不存在時のエラーハンドリングテスト"""
        integrator = QCA001ValidationIntegrator()
        # 存在しないパスを設定
        integrator.workspace_base = Path("/nonexistent/path")

        with self.assertRaises(FileNotFoundError) as context:
            integrator.analyze_qca001_sample_adequacy()

        self.assertIn("ワークスペースが見つかりません", str(context.exception))

    def test_error_handling_missing_extraction_dir(self):
        """抽出ディレクトリ不存在時のエラーハンドリング"""
        # 抽出ディレクトリを削除
        shutil.rmtree(self.mock_extraction_dir)

        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        with self.assertRaises(FileNotFoundError) as context:
            integrator.analyze_qca001_sample_adequacy()

        self.assertIn("抽出ディレクトリが見つかりません", str(context.exception))

    def test_different_sample_sizes(self):
        """異なるサンプル数での動作テスト"""
        # 追加画像作成（合計30枚）
        for i in range(17, 30):
            mock_image = self.mock_extraction_dir / f"extracted_extra_{i:04d}.jpg"
            mock_image.write_text("mock extra image data")

        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        validation_report = integrator.analyze_qca001_sample_adequacy()

        # サンプル数が30に更新されていることを確認
        self.assertEqual(validation_report["qca001_sample_info"]["current_sample_size"], 30)

        # 統計的妥当性が向上している可能性
        stat_validation = validation_report["statistical_validation"]
        self.assertGreaterEqual(stat_validation["current_power"], 0.0)

    def test_power_precision_relationship(self):
        """検出力と精度の関係性テスト"""
        validator = SampleSizeValidator()

        sample_sizes = [10, 30, 50, 100]
        powers = []
        precisions = []

        for n in sample_sizes:
            # 中効果での検出力計算
            power = validator._calculate_current_power(n, 0.5, TestType.ONE_SAMPLE_T)
            powers.append(power)

            # 信頼区間幅計算
            width = validator.calculate_confidence_interval_width(n)
            precision = validator.assess_precision_level(width)
            precisions.append(width)

        # サンプル数増加に伴う改善確認
        # 検出力は単調増加
        for i in range(1, len(powers)):
            self.assertGreaterEqual(powers[i], powers[i - 1])

        # 信頼区間幅は単調減少
        for i in range(1, len(precisions)):
            self.assertLessEqual(precisions[i], precisions[i - 1])

    def test_integration_with_existing_systems(self):
        """既存システムとの統合テスト"""
        # Google Sheets連携の想定テスト
        # （実際のAPI呼び出しは行わない）

        integrator = QCA001ValidationIntegrator()
        integrator.workspace_base = self.test_workspace_path

        # モックレポート生成
        validation_report = integrator.analyze_qca001_sample_adequacy()

        # ダッシュボード連携用データ形式確認
        self.assertIn("qca001_sample_info", validation_report)

        # 進捗トラッカー連携用データ確認
        analysis_results = validation_report["analysis_results"]
        self.assertIn("overall_assessment", analysis_results)

        # 統計データのJSON serializable確認
        try:
            json.dumps(validation_report, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            self.fail(f"レポートデータがJSON serializable ではありません: {e}")


def run_integration_tests():
    """統合テスト実行"""
    print("🔧 QCC-021統合テスト開始")

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQCC021Integration)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print("\n" + "=" * 60)
    print("🔧 QCC-021統合テスト結果")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")

    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    )
    print(f"成功率: {success_rate:.1f}%")

    if result.failures:
        print(f"\n❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n⚠️ エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ 統合テスト全成功' if success else '❌ 統合テスト失敗あり'}")

    return success


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
