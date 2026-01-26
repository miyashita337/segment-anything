#!/usr/bin/env python3
"""
統合統計分析システム - INCI-004対応
BASELINE_ID必須化、既存システム統合改良版

機能:
- Cohen's d計算（信頼区間・実用的意義判定含む）
- Welch t検定実行
- Google Sheets統計列自動更新
- BASELINE_ID必須バリデーション

使用方法:
python tools/progress_tracker/universal_statistical_analyzer.py --current TRACKER_ID --baseline BASELINE_ID
"""

import numpy as np

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.evaluation.statistical_quality_analyzer import StatisticalQualityAnalyzer
from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.validation.statistical_validator import StatisticalValidator, TTestResult


@dataclass
class UniversalAnalysisResult:
    """統合分析結果を格納するデータクラス"""

    success: bool
    current_tracker: str
    baseline_tracker: str
    cohens_d: float
    glass_delta: float
    confidence_interval: Tuple[float, float]
    t_test_result: TTestResult
    practical_significance: str
    interpretation_level: str
    improvement_rate: float
    effect_magnitude: str
    analysis_timestamp: str
    error_message: Optional[str] = None


class UniversalStatisticalAnalyzer:
    """統合統計分析システム - INCI-004対応"""

    def __init__(self):
        """統合統計分析システム初期化"""
        # 既存システム連携
        self.statistical_analyzer = StatisticalQualityAnalyzer()
        self.validator = StatisticalValidator()

        # Google Sheetsクライアント
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)

        print("🔬 統合統計分析システム初期化完了 (INCI-004)")

    def validate_baseline_required(self, baseline_id: str) -> None:
        """BASELINE_ID必須バリデーション - INCI-004要件"""
        if not baseline_id or baseline_id.strip() == "":
            error_msg = """❌ エラー: BASELINE_IDの指定が必要です

🔧 対処方法:
   1. --baseline オプションでベーストラッカーIDを指定してください
   2. 使用方法: python universal_statistical_analyzer.py --current TRACKER_ID --baseline BASELINE_ID
   3. 例: python universal_statistical_analyzer.py --current INCI-004 --baseline QUAL-001

⚠️ 注意: INCI-004により、統計分析時のBASELINE_ID指定が必須化されました"""

            print(error_msg)
            sys.exit(1)

    def calculate_enhanced_cohens_d(
        self,
        group_current: np.ndarray,
        group_baseline: np.ndarray,
        current_name: str = "改善版",
        baseline_name: str = "ベースライン",
    ) -> Dict:
        """
        強化版Cohen's d計算（qcc023_enhanced_cohens_d.py統合）

        Args:
            group_current: 改善版データ
            group_baseline: ベースライン版データ
            current_name: 改善版の名前
            baseline_name: ベースライン版の名前

        Returns:
            Dict: Cohen's d分析結果
        """
        # 基本統計量
        mean_current = np.mean(group_current)
        mean_baseline = np.mean(group_baseline)
        std_current = np.std(group_current, ddof=1)
        std_baseline = np.std(group_baseline, ddof=1)
        n_current = len(group_current)
        n_baseline = len(group_baseline)

        # 1. 標準Cohen's d (プールした標準偏差)
        pooled_std = np.sqrt(
            ((n_current - 1) * std_current**2 + (n_baseline - 1) * std_baseline**2)
            / (n_current + n_baseline - 2)
        )
        cohens_d = (mean_current - mean_baseline) / pooled_std if pooled_std > 0 else 0.0

        # 2. Glass's delta (不等分散対応)
        glass_delta = (mean_current - mean_baseline) / std_baseline if std_baseline > 0 else 0.0

        # 3. 信頼区間計算 (Hedges' g補正込み)
        ci = self._calculate_effect_size_confidence_interval(cohens_d, n_current, n_baseline)

        # 4. 実用的意義判定
        practical_significance = self._assess_practical_significance(
            cohens_d, n_current, n_baseline
        )

        # 5. 解釈レベル（初心者向け詳細）
        interpretation_level = self._get_interpretation_level(cohens_d)

        # 6. 効果の大きさカテゴリ（詳細版）
        effect_magnitude = self._categorize_effect_magnitude(cohens_d)

        # 7. 改善率計算
        improvement_rate = (
            ((mean_current - mean_baseline) / mean_baseline * 100) if mean_baseline > 0 else 0.0
        )

        return {
            "cohens_d": cohens_d,
            "glass_delta": glass_delta,
            "confidence_interval": ci,
            "practical_significance": practical_significance,
            "interpretation_level": interpretation_level,
            "effect_magnitude": effect_magnitude,
            "improvement_rate": improvement_rate,
            "descriptive_stats": {
                "current": {"mean": float(mean_current), "std": float(std_current), "n": n_current},
                "baseline": {
                    "mean": float(mean_baseline),
                    "std": float(std_baseline),
                    "n": n_baseline,
                },
            },
        }

    def _calculate_effect_size_confidence_interval(
        self, d: float, n1: int, n2: int, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """エフェクトサイズの信頼区間計算"""
        # Hedges' g補正
        j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        hedges_g = d * j

        # 標準誤差計算
        se = np.sqrt((n1 + n2) / (n1 * n2) + (hedges_g**2) / (2 * (n1 + n2)))

        # t分布の臨界値（自由度: n1 + n2 - 2）
        from scipy import stats

        df = n1 + n2 - 2
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)

        # 信頼区間
        margin = t_critical * se
        lower = hedges_g - margin
        upper = hedges_g + margin

        return (lower, upper)

    def _assess_practical_significance(self, d: float, n1: int, n2: int) -> str:
        """実用的意義の判定"""
        abs_d = abs(d)

        # Cohen (1988) + Ferguson (2009) の拡張基準
        if abs_d < 0.2:
            return "実用的意義なし"
        elif abs_d < 0.5:
            return "小さいが実用的意義あり"
        elif abs_d < 0.8:
            return "中程度の実用的意義"
        elif abs_d < 1.2:
            return "大きな実用的意義"
        else:
            return "非常に大きな実用的意義"

    def _get_interpretation_level(self, d: float) -> str:
        """解釈レベル（詳細版）"""
        abs_d = abs(d)
        direction = "改善" if d > 0 else "劣化" if d < 0 else "変化なし"

        if abs_d < 0.2:
            return f"ほぼ変化なし（{direction}）"
        elif abs_d < 0.5:
            return f"小さな{direction}"
        elif abs_d < 0.8:
            return f"中程度の{direction}"
        elif abs_d < 1.2:
            return f"大きな{direction}"
        else:
            return f"非常に大きな{direction}"

    def _categorize_effect_magnitude(self, d: float) -> str:
        """効果の大きさカテゴリ（詳細版）"""
        abs_d = abs(d)

        # Cohen (1988) + 現代的基準の統合
        if abs_d == 0.0:
            return "効果なし (|d| = 0)"
        elif abs_d < 0.2:
            return "微小効果 (0 < |d| < 0.2)"
        elif abs_d < 0.5:
            return "小効果 (0.2 ≤ |d| < 0.5)"
        elif abs_d < 0.8:
            return "中効果 (0.5 ≤ |d| < 0.8)"
        elif abs_d < 1.2:
            return "大効果 (0.8 ≤ |d| < 1.2)"
        elif abs_d < 2.0:
            return "非常に大きい効果 (1.2 ≤ |d| < 2.0)"
        else:
            return "極大効果 (|d| ≥ 2.0)"  # 最後のカテゴリ

    def update_google_sheets_statistics(self, tracker_id: str, analysis_result: Dict) -> bool:
        """Google Sheets統計列自動更新 - N-S列対応版"""
        try:
            # トラッカー行を検索
            all_values = self.sheets_client.get_sheet_values("A:S")
            if not all_values:
                print("❌ Google Sheetsデータ取得失敗")
                return False

            tracker_row = None
            for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップ
                if row and len(row) > 0 and row[0] == tracker_id:
                    tracker_row = i
                    break

            if not tracker_row:
                print(f"❌ トラッカー {tracker_id} がGoogle Sheetsに見つかりません")
                return False

            # 統計データ更新（N-S列：14-19番目）
            current_score = analysis_result["descriptive_stats"]["current"]["mean"]
            baseline_score = analysis_result["descriptive_stats"]["baseline"]["mean"]
            p_value = analysis_result["t_test_result"].p_value
            cohens_d = analysis_result["cohens_d"]
            improvement_rate = analysis_result["improvement_rate"]
            significance = "有意" if analysis_result["t_test_result"].is_significant else "非有意"

            # N-S列を一括更新（正しい列マッピング）
            self.sheets_client.update_sheet_values(f"N{tracker_row}", [[f"{current_score:.3f}"]])
            self.sheets_client.update_sheet_values(f"O{tracker_row}", [[f"{baseline_score:.3f}"]])
            self.sheets_client.update_sheet_values(f"P{tracker_row}", [[f"{p_value:.4f}"]])
            self.sheets_client.update_sheet_values(f"Q{tracker_row}", [[f"{cohens_d:.3f}"]])
            self.sheets_client.update_sheet_values(
                f"R{tracker_row}", [[f"{improvement_rate:+.1f}%"]]
            )
            self.sheets_client.update_sheet_values(f"S{tracker_row}", [[significance]])

            print(f"✅ Google Sheets統計列更新完了: {tracker_id} (行{tracker_row})")
            print(f"   N列(Current): {current_score:.3f}")
            print(f"   O列(BaseLine): {baseline_score:.3f}")
            print(f"   P列(p値): {p_value:.4f}")
            print(f"   Q列(効果サイズ): {cohens_d:.3f}")
            print(f"   R列(改善率): {improvement_rate:+.1f}%")
            print(f"   S列(統計的有意性): {significance}")
            return True

        except Exception as e:
            print(f"❌ Google Sheets更新エラー: {e}")
            return False

    def run_integrated_analysis(
        self, current_tracker: str, baseline_tracker: str, verbose: bool = False
    ) -> UniversalAnalysisResult:
        """統合統計分析実行 - INCI-004メイン機能"""

        print(f"🔬 統合統計分析開始 (INCI-004)")
        print(f"   比較対象: {current_tracker} vs {baseline_tracker} (ベースライン)")

        try:
            # 1. BASELINE_ID必須バリデーション
            self.validate_baseline_required(baseline_tracker)

            # 2. データ読み込み
            if verbose:
                print("📊 データ読み込み中...")

            current_metrics = self.statistical_analyzer.load_extraction_results(current_tracker)
            baseline_metrics = self.statistical_analyzer.load_extraction_results(baseline_tracker)

            if not current_metrics or not baseline_metrics:
                raise ValueError("データ読み込み失敗: トラッカーIDを確認してください")

            # 品質スコアデータ取得
            current_data = np.array(current_metrics.quality_scores)
            baseline_data = np.array(baseline_metrics.quality_scores)

            if verbose:
                print(f"📊 データ確認:")
                print(
                    f"   {current_tracker}: {len(current_data)}サンプル, 平均={np.mean(current_data):.4f}"
                )
                print(
                    f"   {baseline_tracker}: {len(baseline_data)}サンプル, 平均={np.mean(baseline_data):.4f}"
                )

            # 3. 統合統計分析実行
            # 3-1. Cohen's d計算
            cohens_analysis = self.calculate_enhanced_cohens_d(
                current_data, baseline_data, current_tracker, baseline_tracker
            )

            # 3-2. t検定実行
            t_test_result = self.validator.welch_t_test(baseline_data, current_data)

            # 4. 結果統合
            analysis_result = UniversalAnalysisResult(
                success=True,
                current_tracker=current_tracker,
                baseline_tracker=baseline_tracker,
                cohens_d=cohens_analysis["cohens_d"],
                glass_delta=cohens_analysis["glass_delta"],
                confidence_interval=cohens_analysis["confidence_interval"],
                t_test_result=t_test_result,
                practical_significance=cohens_analysis["practical_significance"],
                interpretation_level=cohens_analysis["interpretation_level"],
                improvement_rate=cohens_analysis["improvement_rate"],
                effect_magnitude=cohens_analysis["effect_magnitude"],
                analysis_timestamp=datetime.now().isoformat(),
            )

            # 5. 結果表示
            self._display_results(analysis_result, cohens_analysis, verbose)

            # 6. Google Sheets統計列自動更新
            if verbose:
                print("📝 Google Sheets統計列更新中...")

            update_data = {
                "descriptive_stats": cohens_analysis["descriptive_stats"],
                "t_test_result": t_test_result,
                "cohens_d": cohens_analysis["cohens_d"],
                "improvement_rate": cohens_analysis["improvement_rate"],
            }

            sheets_updated = self.update_google_sheets_statistics(current_tracker, update_data)
            if sheets_updated and verbose:
                print("✅ Google Sheets更新完了")

            return analysis_result

        except Exception as e:
            error_result = UniversalAnalysisResult(
                success=False,
                current_tracker=current_tracker,
                baseline_tracker=baseline_tracker,
                cohens_d=0.0,
                glass_delta=0.0,
                confidence_interval=(0.0, 0.0),
                t_test_result=None,
                practical_significance="エラー",
                interpretation_level="分析失敗",
                improvement_rate=0.0,
                effect_magnitude="エラー",
                analysis_timestamp=datetime.now().isoformat(),
                error_message=str(e),
            )

            print(f"❌ 統合統計分析失敗: {str(e)}")
            return error_result

    def _display_results(
        self, result: UniversalAnalysisResult, cohens_analysis: Dict, verbose: bool
    ) -> None:
        """分析結果表示"""
        print(f"\n📈 統合統計分析結果 (INCI-004):")
        print(f"   Cohen's d: {result.cohens_d:.4f}")
        print(f"   Glass's Δ: {result.glass_delta:.4f}")
        print(
            f"   95%信頼区間: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]"
        )
        print(f"   p値: {result.t_test_result.p_value:.4f}")
        print(f"   改善率: {result.improvement_rate:+.1f}%")
        print(f"   統計的有意性: {'有意' if result.t_test_result.is_significant else '非有意'}")
        print(f"   実用的意義: {result.practical_significance}")
        print(f"   解釈レベル: {result.interpretation_level}")
        print(f"   効果の大きさ: {result.effect_magnitude}")

        if verbose:
            print(f"\n📊 記述統計量:")
            current_stats = cohens_analysis["descriptive_stats"]["current"]
            baseline_stats = cohens_analysis["descriptive_stats"]["baseline"]
            print(
                f"   {result.current_tracker}: 平均={current_stats['mean']:.4f}, SD={current_stats['std']:.4f}, N={current_stats['n']}"
            )
            print(
                f"   {result.baseline_tracker}: 平均={baseline_stats['mean']:.4f}, SD={baseline_stats['std']:.4f}, N={baseline_stats['n']}"
            )


def main():
    """メイン実行関数 - INCI-004統合統計分析システム"""
    parser = argparse.ArgumentParser(
        description="統合統計分析システム (INCI-004対応) - BASELINE_ID必須化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python universal_statistical_analyzer.py --current INCI-004 --baseline QUAL-001
  python universal_statistical_analyzer.py --current TRACKER_ID --baseline BASELINE_ID --verbose

注意:
  INCI-004により、BASELINE_IDの指定が必須化されました。
  --baseline オプションを省略するとエラーで終了します。
        """,
    )

    parser.add_argument("--current", required=True, help="現在のトラッカーID（必須）")
    parser.add_argument("--baseline", required=True, help="ベースライントラッカーID（必須）- INCI-004により必須化")
    parser.add_argument("--verbose", action="store_true", help="詳細出力モード")

    args = parser.parse_args()

    # 統合統計分析実行
    analyzer = UniversalStatisticalAnalyzer()
    result = analyzer.run_integrated_analysis(args.current, args.baseline, args.verbose)

    # 結果に応じて終了コード設定
    if result.success:
        print(f"\n✅ 統合統計分析完了 (INCI-004): {args.current} vs {args.baseline}")
        sys.exit(0)
    else:
        print(f"\n❌ 統合統計分析失敗 (INCI-004): {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
