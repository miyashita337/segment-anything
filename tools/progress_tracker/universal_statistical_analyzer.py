#!/usr/bin/env python3
"""
汎用統計分析システム - Universal Statistical Analyzer

BASELINE-RECALC-001から始まり、今後全トラッカーで使用可能な統計分析システム

使用方法:
    python tools/progress_tracker/universal_statistical_analyzer.py \
        --current BASELINE-RECALC-001 \
        --auto-baseline  # 最新/releaseトラッカーを自動選択

機能:
    - QCA-001作者名動的決定（AuthorParameterAdapter流用）
    - 最新/releaseトラッカー自動選択（BaselineDetector流用）
    - フルセット統計分析（ウェルチt検定+Cohen's d+信頼区間+実用的意義判定）
    - Google Sheets N-S列自動更新
    - ダッシュボード生成（統計結果+改善推移グラフ+画像ギャラリー）
    - 詳細レポート生成（.md）
    - エラー処理・フォールバック機能

Created for: BASELINE-RECALC-001 汎用統計分析システム実装
Author: Claude Code Integration System
"""

import argparse
import json
import logging
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

# 既存実装流用
from features.adaptation.author_parameter_adapter import AuthorParameterAdapter
from tools.progress_tracker.baseline_detector import BaselineDetector
from features.evaluation.statistical_quality_analyzer import StatisticalQualityAnalyzer
from tools.validation.statistical_validator import StatisticalValidator
from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config
from tools.core.unified_quality_checker import UnifiedQualityChecker
from tools.progress_tracker.universal_dashboard_generator import UniversalDashboardGenerator

logger = logging.getLogger(__name__)


@dataclass
class UniversalStatisticalResult:
    """汎用統計分析結果"""
    success: bool
    current_tracker: str
    baseline_tracker: Optional[str]
    author_name: str
    
    # 統計結果
    current_score: float
    baseline_score: float
    p_value: float
    cohens_d: float
    improvement_rate: float
    is_significant: bool
    
    # 拡張統計
    confidence_interval: Tuple[float, float]
    practical_significance: str
    interpretation: str
    sample_adequacy: str
    
    # メタデータ
    current_sample_size: int
    baseline_sample_size: int
    analysis_timestamp: str
    error_message: Optional[str] = None


class UniversalStatisticalAnalyzer:
    """汎用統計分析システム"""
    
    def __init__(self):
        """初期化"""
        # 既存システム連携
        self.author_adapter = AuthorParameterAdapter()
        self.baseline_detector = BaselineDetector()
        self.statistical_analyzer = StatisticalQualityAnalyzer()
        self.validator = StatisticalValidator()
        self.quality_checker = UnifiedQualityChecker()
        
        # Google Sheets連携
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)
        
        # ダッシュボード生成器
        self.dashboard_generator = UniversalDashboardGenerator()
        
        logger.info("🔬 汎用統計分析システム初期化完了")
    
    def detect_author_from_tracker(self, tracker_id: str) -> str:
        """
        トラッカーIDから作者名を動的検出（QCA-001流用）
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            str: 作者名（yado, kiri, zundamon等）
        """
        try:
            # トラッカーワークスペースから任意の画像パスを生成
            workspace_base = "/mnt/c/AItools/lora/train"
            
            # 一般的なパスパターンを試行
            path_patterns = [
                f"{workspace_base}/yado/tracker-workspace/{tracker_id}",
                f"{workspace_base}/kiri/tracker-workspace/{tracker_id}",
                f"{workspace_base}/zundamon/tracker-workspace/{tracker_id}",
            ]
            
            for path_pattern in path_patterns:
                if Path(path_pattern).exists():
                    # AuthorParameterAdapterの作者検出を使用
                    detected_author = AuthorParameterAdapter.detect_author_from_path(path_pattern)
                    if detected_author:
                        logger.info(f"🔍 作者検出成功: {tracker_id} → {detected_author}")
                        return detected_author
            
            # フォールバック: ディレクトリ構造から直接検出
            for author in ['yado', 'kiri', 'zundamon']:
                author_workspace = f"{workspace_base}/{author}/tracker-workspace/{tracker_id}"
                if Path(author_workspace).exists():
                    logger.info(f"🔍 フォールバック作者検出: {tracker_id} → {author}")
                    return author
            
            # デフォルト
            logger.warning(f"⚠️ 作者検出失敗、デフォルト使用: {tracker_id} → yado")
            return "yado"
            
        except Exception as e:
            logger.error(f"❌ 作者検出エラー {tracker_id}: {e}")
            return "yado"
    
    def select_baseline_tracker(self, current_tracker: str, auto_baseline: bool = True) -> Optional[str]:
        """
        ベースライントラッカー選択（BaselineDetector流用）
        
        Args:
            current_tracker: 現在のトラッカーID
            auto_baseline: 自動選択フラグ
            
        Returns:
            Optional[str]: ベースライントラッカーID
        """
        if not auto_baseline:
            return None
        
        try:
            # 最新完了トラッカーを取得
            trackers = self.baseline_detector.get_all_trackers_by_update_date()
            
            if not trackers:
                logger.error("❌ トラッカーリスト取得失敗")
                return None
            
            # /release完了済みトラッカーのみフィルタ
            completed_trackers = [t for t in trackers if t['status'] == '/release']
            
            if not completed_trackers:
                logger.warning("⚠️ 完了済みトラッカーが見つかりません")
                return None
            
            # 自分自身を除外
            other_trackers = [t for t in completed_trackers if t['tracker_id'] != current_tracker]
            
            if not other_trackers:
                logger.warning(f"⚠️ {current_tracker}以外の完了済みトラッカーが見つかりません")
                return None
            
            # 最新完了トラッカー（更新日付順で最後）
            latest_tracker = other_trackers[-1]['tracker_id']
            
            logger.info(f"✅ 自動ベースライン選択: {current_tracker} vs {latest_tracker} (最新完了)")
            return latest_tracker
            
        except Exception as e:
            logger.error(f"❌ ベースライン選択エラー: {e}")
            return None
    
    def ensure_extraction_result_json(self, tracker_id: str, author_name: str) -> bool:
        """
        extraction_result.json存在確認・必要時再作成
        
        Args:
            tracker_id: トラッカーID
            author_name: 作者名
            
        Returns:
            bool: 成功可否
        """
        try:
            workspace_dir = Path(f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace/{tracker_id}")
            json_path = workspace_dir / "extraction_result.json"
            
            # 既存ファイル確認
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # データ完整性確認
                if (data.get('generation_method') == 'opencv_analysis' and 
                    'results' in data and len(data['results']) > 0):
                    logger.info(f"✅ {tracker_id}: extraction_result.json確認済み")
                    return True
            
            logger.warning(f"⚠️ {tracker_id}: extraction_result.json不完全、再作成実行")
            
            # 抽出画像ディレクトリ確認
            extraction_dir = workspace_dir / "extraction"
            if not extraction_dir.exists() or not any(extraction_dir.iterdir()):
                logger.error(f"❌ {tracker_id}: 抽出画像が存在しません")
                return False
            
            # UnifiedQualityCheckerで再作成
            self.quality_checker.run_quality_check(str(extraction_dir), str(workspace_dir))
            
            # 再作成確認
            if json_path.exists():
                logger.info(f"✅ {tracker_id}: extraction_result.json再作成完了")
                return True
            else:
                logger.error(f"❌ {tracker_id}: extraction_result.json再作成失敗")
                return False
                
        except Exception as e:
            logger.error(f"❌ {tracker_id} extraction_result.json処理エラー: {e}")
            return False
    
    def calculate_enhanced_statistics(
        self, 
        current_data: np.ndarray, 
        baseline_data: np.ndarray,
        current_tracker: str,
        baseline_tracker: str
    ) -> Dict[str, Any]:
        """
        強化統計分析（フルセット）
        
        Args:
            current_data: 現在のデータ
            baseline_data: ベースラインデータ
            current_tracker: 現在のトラッカーID
            baseline_tracker: ベースライントラッカーID
            
        Returns:
            Dict: 統計分析結果
        """
        try:
            # 基本統計量
            current_mean = np.mean(current_data)
            baseline_mean = np.mean(baseline_data)
            current_std = np.std(current_data, ddof=1)
            baseline_std = np.std(baseline_data, ddof=1)
            n_current = len(current_data)
            n_baseline = len(baseline_data)
            
            # 1. ウェルチのt検定（不等分散対応）
            t_test_result = self.validator.welch_t_test(baseline_data, current_data)
            
            # 2. Cohen's d（効果サイズ）
            pooled_std = np.sqrt(
                ((n_current - 1) * current_std**2 + (n_baseline - 1) * baseline_std**2) 
                / (n_current + n_baseline - 2)
            )
            cohens_d = (current_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0.0
            
            # 3. 95%信頼区間（Hedges' g補正込み）
            j = 1 - (3 / (4 * (n_current + n_baseline - 2) - 1))
            hedges_g = cohens_d * j
            se = np.sqrt((n_current + n_baseline) / (n_current * n_baseline) + (hedges_g**2) / (2 * (n_current + n_baseline)))
            df = n_current + n_baseline - 2
            t_critical = stats.t.ppf(0.975, df)
            margin = t_critical * se
            confidence_interval = (hedges_g - margin, hedges_g + margin)
            
            # 4. 実用的意義判定
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                practical_significance = "実用的意義なし"
            elif abs_d < 0.5:
                practical_significance = "小さいが実用的意義あり"
            elif abs_d < 0.8:
                practical_significance = "中程度の実用的意義"
            elif abs_d < 1.2:
                practical_significance = "大きな実用的意義"
            else:
                practical_significance = "非常に大きな実用的意義"
            
            # 5. 解釈レベル
            direction = "改善" if cohens_d > 0 else "劣化" if cohens_d < 0 else "変化なし"
            if abs_d < 0.2:
                interpretation = f"ほぼ変化なし（{direction}）"
            elif abs_d < 0.5:
                interpretation = f"小さな{direction}"
            elif abs_d < 0.8:
                interpretation = f"中程度の{direction}"
            elif abs_d < 1.2:
                interpretation = f"大きな{direction}"
            else:
                interpretation = f"非常に大きな{direction}"
            
            # 6. サンプルサイズ妥当性
            total_n = n_current + n_baseline
            if abs_d >= 0.8:
                required_n = 20
            elif abs_d >= 0.5:
                required_n = 50
            elif abs_d >= 0.2:
                required_n = 200
            else:
                required_n = 1000
            
            if total_n >= required_n:
                sample_adequacy = "サンプルサイズ十分"
            elif total_n >= required_n * 0.7:
                sample_adequacy = "サンプルサイズやや不足"
            else:
                sample_adequacy = "サンプルサイズ大幅不足"
            
            # 7. 改善率
            improvement_rate = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            
            return {
                'current_mean': current_mean,
                'baseline_mean': baseline_mean,
                'current_std': current_std,
                'baseline_std': baseline_std,
                'current_sample_size': n_current,
                'baseline_sample_size': n_baseline,
                'p_value': t_test_result.p_value,
                'cohens_d': cohens_d,
                'confidence_interval': confidence_interval,
                'practical_significance': practical_significance,
                'interpretation': interpretation,
                'sample_adequacy': sample_adequacy,
                'improvement_rate': improvement_rate,
                'is_significant': t_test_result.is_significant,
                't_statistic': t_test_result.statistic,
                'degrees_of_freedom': t_test_result.degrees_of_freedom
            }
            
        except Exception as e:
            logger.error(f"❌ 統計分析エラー: {e}")
            raise
    
    def update_google_sheets(self, tracker_id: str, baseline_tracker: str, stats_result: Dict[str, Any]) -> bool:
        """
        Google Sheets N-S列更新
        
        Args:
            tracker_id: トラッカーID
            baseline_tracker: ベースライントラッカーID
            stats_result: 統計分析結果
            
        Returns:
            bool: 更新成功可否
        """
        try:
            # トラッカー行検索
            all_values = self.sheets_client.get_sheet_values('A:S')
            tracker_row = None
            
            for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップ
                if row and len(row) > 0 and row[0] == tracker_id:
                    tracker_row = i
                    break
            
            if not tracker_row:
                logger.error(f"❌ {tracker_id}: Google Sheetsに未発見")
                return False
            
            # N-S列更新（N:Current, O:BaseLine, P:p値, Q:Cohen's d, R:改善率, S:統計的有意性）
            updates = [
                (f'N{tracker_row}', [[f"{stats_result['current_mean']:.3f}"]]),     # Current
                (f'O{tracker_row}', [[baseline_tracker]]),                          # BaseLine (トラッカーID)  
                (f'P{tracker_row}', [[f"{stats_result['p_value']:.4f}"]]),          # p値
                (f'Q{tracker_row}', [[f"{stats_result['cohens_d']:.3f}"]]),         # Cohen's d
                (f'R{tracker_row}', [[f"{stats_result['improvement_rate']:.1f}%"]]), # 改善率
                (f'S{tracker_row}', [["有意" if stats_result['is_significant'] else "非有意"]]) # 統計的有意性
            ]
            
            for range_name, values in updates:
                self.sheets_client.update_sheet_values(range_name, values)
            
            logger.info(f"✅ {tracker_id}: Google Sheets N-S列更新完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ {tracker_id} Google Sheets更新エラー: {e}")
            return False
    
    def run_universal_analysis(
        self, 
        current_tracker: str, 
        baseline_tracker: Optional[str] = None,
        auto_baseline: bool = True
    ) -> UniversalStatisticalResult:
        """
        汎用統計分析実行
        
        Args:
            current_tracker: 現在のトラッカーID
            baseline_tracker: ベースライントラッカーID（Noneで自動選択）
            auto_baseline: 自動ベースライン選択フラグ
            
        Returns:
            UniversalStatisticalResult: 統計分析結果
        """
        logger.info(f"🔬 汎用統計分析開始: {current_tracker}")
        
        try:
            # 1. 作者名検出
            author_name = self.detect_author_from_tracker(current_tracker)
            
            # 2. ベースライン選択
            if not baseline_tracker and auto_baseline:
                baseline_tracker = self.select_baseline_tracker(current_tracker, auto_baseline)
            
            if not baseline_tracker:
                return UniversalStatisticalResult(
                    success=False,
                    current_tracker=current_tracker,
                    baseline_tracker=None,
                    author_name=author_name,
                    current_score=0.0, baseline_score=0.0, p_value=0.0, cohens_d=0.0,
                    improvement_rate=0.0, is_significant=False,
                    confidence_interval=(0.0, 0.0), practical_significance="",
                    interpretation="", sample_adequacy="",
                    current_sample_size=0, baseline_sample_size=0,
                    analysis_timestamp=datetime.now().isoformat(),
                    error_message="ベースライントラッカー選択失敗"
                )
            
            # 3. extraction_result.json確認・再作成
            if not self.ensure_extraction_result_json(current_tracker, author_name):
                return UniversalStatisticalResult(
                    success=False,
                    current_tracker=current_tracker,
                    baseline_tracker=baseline_tracker,
                    author_name=author_name,
                    current_score=0.0, baseline_score=0.0, p_value=0.0, cohens_d=0.0,
                    improvement_rate=0.0, is_significant=False,
                    confidence_interval=(0.0, 0.0), practical_significance="",
                    interpretation="", sample_adequacy="",
                    current_sample_size=0, baseline_sample_size=0,
                    analysis_timestamp=datetime.now().isoformat(),
                    error_message=f"{current_tracker}: extraction_result.json作成失敗"
                )
            
            baseline_author = self.detect_author_from_tracker(baseline_tracker)
            if not self.ensure_extraction_result_json(baseline_tracker, baseline_author):
                return UniversalStatisticalResult(
                    success=False,
                    current_tracker=current_tracker,
                    baseline_tracker=baseline_tracker,
                    author_name=author_name,
                    current_score=0.0, baseline_score=0.0, p_value=0.0, cohens_d=0.0,
                    improvement_rate=0.0, is_significant=False,
                    confidence_interval=(0.0, 0.0), practical_significance="",
                    interpretation="", sample_adequacy="",
                    current_sample_size=0, baseline_sample_size=0,
                    analysis_timestamp=datetime.now().isoformat(),
                    error_message=f"{baseline_tracker}: extraction_result.json作成失敗"
                )
            
            # 4. データ読み込み
            current_metrics = self.statistical_analyzer.load_extraction_results(current_tracker)
            baseline_metrics = self.statistical_analyzer.load_extraction_results(baseline_tracker)
            
            current_data = np.array(current_metrics.quality_scores)
            baseline_data = np.array(baseline_metrics.quality_scores)
            
            logger.info(f"📊 データ確認: {current_tracker}={len(current_data)}サンプル, {baseline_tracker}={len(baseline_data)}サンプル")
            
            # 5. 統計分析実行
            stats_result = self.calculate_enhanced_statistics(
                current_data, baseline_data, current_tracker, baseline_tracker
            )
            
            # 6. Google Sheets更新
            self.update_google_sheets(current_tracker, baseline_tracker, stats_result)
            
            # 7. 結果構築
            result = UniversalStatisticalResult(
                success=True,
                current_tracker=current_tracker,
                baseline_tracker=baseline_tracker,
                author_name=author_name,
                current_score=stats_result['current_mean'],
                baseline_score=stats_result['baseline_mean'],
                p_value=stats_result['p_value'],
                cohens_d=stats_result['cohens_d'],
                improvement_rate=stats_result['improvement_rate'],
                is_significant=stats_result['is_significant'],
                confidence_interval=stats_result['confidence_interval'],
                practical_significance=stats_result['practical_significance'],
                interpretation=stats_result['interpretation'],
                sample_adequacy=stats_result['sample_adequacy'],
                current_sample_size=stats_result['current_sample_size'],
                baseline_sample_size=stats_result['baseline_sample_size'],
                analysis_timestamp=datetime.now().isoformat()
            )
            
            # 8. ダッシュボード・レポート生成
            try:
                workspace_dir = Path(f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace/{current_tracker}")
                if workspace_dir.exists():
                    generated_files = self.dashboard_generator.generate_complete_dashboard(
                        result, workspace_dir, save_html=True, save_markdown=True
                    )
                    logger.info(f"✅ ダッシュボード生成完了: {len(generated_files)}ファイル")
                    
                    # 生成ファイル情報をログ出力
                    for file_type, file_path in generated_files.items():
                        logger.info(f"   {file_type}: {file_path}")
                else:
                    logger.warning(f"⚠️ ワークスペースディレクトリ未発見: {workspace_dir}")
            except Exception as e:
                logger.error(f"❌ ダッシュボード生成エラー: {e}")
                # ダッシュボード生成失敗でも統計分析結果は返す
            
            logger.info(f"✅ 汎用統計分析完了: {current_tracker} vs {baseline_tracker}")
            logger.info(f"   改善率: {result.improvement_rate:.1f}%, Cohen's d: {result.cohens_d:.3f}")
            logger.info(f"   統計的有意性: {'有意' if result.is_significant else '非有意'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 汎用統計分析エラー: {e}")
            return UniversalStatisticalResult(
                success=False,
                current_tracker=current_tracker,
                baseline_tracker=baseline_tracker or "",
                author_name=author_name if 'author_name' in locals() else "unknown",
                current_score=0.0, baseline_score=0.0, p_value=0.0, cohens_d=0.0,
                improvement_rate=0.0, is_significant=False,
                confidence_interval=(0.0, 0.0), practical_significance="",
                interpretation="", sample_adequacy="",
                current_sample_size=0, baseline_sample_size=0,
                analysis_timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/tmp/universal_statistical_analyzer.log')
        ]
    )


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='汎用統計分析システム')
    parser.add_argument('--current', required=True, help='現在のトラッカーID')
    parser.add_argument('--baseline', help='ベースライントラッカーID（省略時は自動選択）')
    parser.add_argument('--auto-baseline', action='store_true', default=True, 
                       help='自動ベースライン選択（デフォルト: True）')
    parser.add_argument('--no-google-sheets', action='store_true', 
                       help='Google Sheets更新をスキップ')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 汎用統計分析実行
        analyzer = UniversalStatisticalAnalyzer()
        result = analyzer.run_universal_analysis(
            current_tracker=args.current,
            baseline_tracker=args.baseline,
            auto_baseline=args.auto_baseline
        )
        
        # 結果表示
        if result.success:
            print(f"✅ 汎用統計分析完了: {result.current_tracker} vs {result.baseline_tracker}")
            print(f"📊 統計結果:")
            print(f"   Current Score: {result.current_score:.3f}")
            print(f"   Baseline Score: {result.baseline_score:.3f}")
            print(f"   改善率: {result.improvement_rate:.1f}%")
            print(f"   p値: {result.p_value:.4f}")
            print(f"   Cohen's d: {result.cohens_d:.3f}")
            print(f"   統計的有意性: {'有意' if result.is_significant else '非有意'}")
            print(f"   実用的意義: {result.practical_significance}")
            print(f"   解釈: {result.interpretation}")
            print(f"   サンプルサイズ評価: {result.sample_adequacy}")
            
            # JSON保存
            output_file = f"/tmp/{result.current_tracker}_statistical_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': bool(result.success),
                    'current_tracker': str(result.current_tracker),
                    'baseline_tracker': str(result.baseline_tracker),
                    'author_name': str(result.author_name),
                    'current_score': float(result.current_score),
                    'baseline_score': float(result.baseline_score),
                    'p_value': float(result.p_value),
                    'cohens_d': float(result.cohens_d),
                    'improvement_rate': float(result.improvement_rate),
                    'is_significant': bool(result.is_significant),
                    'confidence_interval': [float(result.confidence_interval[0]), float(result.confidence_interval[1])],
                    'practical_significance': str(result.practical_significance),
                    'interpretation': str(result.interpretation),
                    'sample_adequacy': str(result.sample_adequacy),
                    'current_sample_size': int(result.current_sample_size),
                    'baseline_sample_size': int(result.baseline_sample_size),
                    'analysis_timestamp': str(result.analysis_timestamp)
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 結果保存: {output_file}")
            
        else:
            print(f"❌ 汎用統計分析失敗: {result.error_message}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ システムエラー: {e}")
        print(f"❌ システムエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()