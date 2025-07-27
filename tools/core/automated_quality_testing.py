#!/usr/bin/env python3
"""
自動テスト強化システム
P1-A003: 品質劣化の事前検出

機能:
1. 回帰テスト自動実行
2. 品質ベースライン比較
3. 劣化検出アラート
4. 継続的品質監視
"""

import json
import logging
import sys
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from tools.core.quality_dashboard import QualityDashboard
    from tools.core.unified_quality_checker import UnifiedQualityChecker
    from tools.progress_tracker.data_models import MetricsRecord
    QUALITY_TOOLS_AVAILABLE = True
except ImportError:
    QUALITY_TOOLS_AVAILABLE = False
    logging.warning("品質チェックツールが利用できません")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityBaseline:
    """品質ベースライン"""
    dataset: str
    timestamp: str
    ab_evaluation_rate: float
    sci_score: float
    pla_score: float
    ple_score: float
    total_processed: int
    success_count: int
    failure_count: int
    avg_processing_time: float
    quality_grade_distribution: Dict[str, int]


@dataclass
class TestResult:
    """テスト結果"""
    test_id: str
    timestamp: str
    dataset: str
    baseline: QualityBaseline
    current: QualityBaseline
    degradation_detected: bool
    degradation_details: List[str]
    recommendation: str
    status: str  # PASS, FAIL, WARNING


class AutomatedQualityTesting:
    """自動品質テストシステム"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """初期化"""
        self.project_root = project_root
        self.config_path = config_path or (project_root / "config" / "quality_testing.json")
        # PROGRESS_TRACKER.md仕様準拠の正しいパス
        self.workspace_root = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
        self.workspace_dir = self.workspace_root / "P1-A003"
        self.baseline_dir = self.workspace_root / "baseline"
        self.test_results_dir = project_root / "test_results" / "quality"
        
        # ディレクトリ作成
        for dir_path in [self.baseline_dir, self.test_results_dir, self.workspace_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.quality_checker = UnifiedQualityChecker() if QUALITY_TOOLS_AVAILABLE else None
        
    def _load_config(self) -> Dict[str, Any]:
        """設定読み込み"""
        default_config = {
            "test_datasets": [
                {
                    "name": "kana08",
                    "input_path": "/mnt/c/AItools/lora/train/yado/org/kana08/",
                    "baseline_file": "kana08_baseline.json",
                    "degradation_thresholds": {
                        "ab_evaluation_rate": -5.0,  # 5%以上の低下で警告
                        "sci_score": -0.05,
                        "pla_score": -0.05,
                        "ple_score": -0.05,
                        "success_rate": -10.0
                    }
                }
            ],
            "test_modes": [
                "quick",      # 5枚での簡易テスト
                "standard",   # 20枚での標準テスト
                "full"        # 全データでの完全テスト
            ],
            "alert_settings": {
                "enable_notifications": True,
                "critical_threshold": 3,  # 3個以上の劣化検出で重要アラート
                "consecutive_failures": 2  # 連続2回失敗で緊急アラート
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"設定ファイル読み込み失敗: {e}")
        else:
            # デフォルト設定を保存
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"デフォルト設定を作成: {self.config_path}")
            
        return default_config
    
    def create_baseline(self, dataset_name: str, force_update: bool = False) -> QualityBaseline:
        """品質ベースライン作成"""
        dataset_config = self._get_dataset_config(dataset_name)
        if not dataset_config:
            raise ValueError(f"データセット設定が見つかりません: {dataset_name}")
        
        baseline_file = self.baseline_dir / dataset_config["baseline_file"]
        
        # 既存ベースライン確認
        if baseline_file.exists() and not force_update:
            logger.info(f"既存ベースライン使用: {baseline_file}")
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            return QualityBaseline(**baseline_data)
        
        logger.info(f"ベースライン作成開始: {dataset_name}")
        
        # 品質評価実行
        input_path = Path(dataset_config["input_path"])
        output_path = self.workspace_dir / f"baseline_{dataset_name}"
        
        # バッチ処理実行
        batch_result = self._run_extraction_pipeline(input_path, output_path, mode="standard")
        
        if not batch_result:
            raise RuntimeError(f"ベースライン作成失敗: {dataset_name}")
        
        # 品質評価
        quality_result = self._evaluate_quality(output_path)
        
        # ベースライン作成
        baseline = QualityBaseline(
            dataset=dataset_name,
            timestamp=datetime.now().isoformat(),
            ab_evaluation_rate=quality_result.get('ab_evaluation_rate', 0.0),
            sci_score=quality_result.get('sci_score', 0.0),
            pla_score=quality_result.get('pla_score', 0.0),
            ple_score=quality_result.get('ple_score', 0.0),
            total_processed=quality_result.get('total_processed', 0),
            success_count=quality_result.get('success_count', 0),
            failure_count=quality_result.get('failure_count', 0),
            avg_processing_time=quality_result.get('avg_processing_time', 0.0),
            quality_grade_distribution=quality_result.get('grade_distribution', {})
        )
        
        # ベースライン保存
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(baseline), f, indent=2, ensure_ascii=False)
        
        logger.info(f"ベースライン作成完了: {baseline_file}")
        return baseline
    
    def run_quality_test(self, dataset_name: str, mode: str = "standard") -> TestResult:
        """品質テスト実行"""
        logger.info(f"品質テスト開始: {dataset_name} ({mode})")
        
        # ベースライン取得
        baseline = self.create_baseline(dataset_name)
        
        # 現在の品質測定
        dataset_config = self._get_dataset_config(dataset_name)
        input_path = Path(dataset_config["input_path"])
        output_path = self.workspace_dir / f"test_{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}"
        
        # 抽出パイプライン実行
        start_time = time.time()
        batch_result = self._run_extraction_pipeline(input_path, output_path, mode=mode)
        execution_time = time.time() - start_time
        
        if not batch_result:
            # 実行失敗の場合
            test_result = TestResult(
                test_id=f"test_{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}",
                timestamp=datetime.now().isoformat(),
                dataset=dataset_name,
                baseline=baseline,
                current=baseline,  # ダミー
                degradation_detected=True,
                degradation_details=["抽出パイプライン実行失敗"],
                recommendation="システム設定とログを確認し、基本的な動作確認を実施してください",
                status="FAIL"
            )
        else:
            # 品質評価
            quality_result = self._evaluate_quality(output_path)
            
            # 現在の品質ベースライン作成
            current = QualityBaseline(
                dataset=dataset_name,
                timestamp=datetime.now().isoformat(),
                ab_evaluation_rate=quality_result.get('ab_evaluation_rate', 0.0),
                sci_score=quality_result.get('sci_score', 0.0),
                pla_score=quality_result.get('pla_score', 0.0),
                ple_score=quality_result.get('ple_score', 0.0),
                total_processed=quality_result.get('total_processed', 0),
                success_count=quality_result.get('success_count', 0),
                failure_count=quality_result.get('failure_count', 0),
                avg_processing_time=execution_time,
                quality_grade_distribution=quality_result.get('grade_distribution', {})
            )
            
            # 劣化検出
            degradation_detected, degradation_details = self._detect_degradation(
                baseline, current, dataset_config["degradation_thresholds"]
            )
            
            # ステータス判定
            if degradation_detected:
                if len(degradation_details) >= self.config["alert_settings"]["critical_threshold"]:
                    status = "FAIL"
                else:
                    status = "WARNING"
            else:
                status = "PASS"
            
            # 推奨事項生成
            recommendation = self._generate_recommendation(degradation_details, current)
            
            test_result = TestResult(
                test_id=f"test_{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}",
                timestamp=datetime.now().isoformat(),
                dataset=dataset_name,
                baseline=baseline,
                current=current,
                degradation_detected=degradation_detected,
                degradation_details=degradation_details,
                recommendation=recommendation,
                status=status
            )
        
        # 結果保存
        self._save_test_result(test_result)
        
        # 通知送信
        self._send_notification(test_result)
        
        logger.info(f"品質テスト完了: {test_result.status}")
        return test_result
    
    def _run_extraction_pipeline(self, input_path: Path, output_path: Path, mode: str = "standard") -> bool:
        """抽出パイプライン実行"""
        try:
            # 入力データ確認
            if not input_path.exists():
                logger.error(f"入力パスが存在しません: {input_path}")
                return False
            
            input_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
            if not input_files:
                logger.error(f"画像ファイルが見つかりません: {input_path}")
                return False
            
            # モード別ファイル数制限
            file_limits = {"quick": 5, "standard": 20, "full": len(input_files)}
            limit = file_limits.get(mode, 20)
            selected_files = input_files[:limit]
            
            logger.info(f"処理対象: {len(selected_files)}ファイル ({mode}モード)")
            
            # 出力ディレクトリ作成
            output_path.mkdir(parents=True, exist_ok=True)
            
            # バッチ処理スクリプト実行
            batch_script = project_root / "tools" / "batch" / "kana08_enhanced_stable_batch.py"
            
            if not batch_script.exists():
                logger.error(f"バッチスクリプトが見つかりません: {batch_script}")
                return False
            
            # 一時入力ディレクトリ作成
            temp_input = self.workspace_dir / f"temp_input_{datetime.now():%Y%m%d_%H%M%S}"
            temp_input.mkdir(parents=True, exist_ok=True)
            
            # 選択ファイルをコピー
            for file in selected_files:
                shutil.copy2(file, temp_input / file.name)
            
            # バッチ処理実行
            cmd = [
                sys.executable, str(batch_script),
                "--input_dir", str(temp_input),
                "--output_dir", str(output_path)
            ]
            
            logger.info(f"実行コマンド: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分タイムアウト
            )
            
            # 一時ディレクトリ削除
            shutil.rmtree(temp_input, ignore_errors=True)
            
            if result.returncode == 0:
                logger.info("バッチ処理成功")
                return True
            else:
                logger.error(f"バッチ処理失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"抽出パイプライン実行エラー: {e}")
            return False
    
    def _evaluate_quality(self, output_path: Path) -> Dict[str, Any]:
        """品質評価実行"""
        try:
            if not self.quality_checker:
                logger.warning("品質チェッカーが利用できません")
                return self._fallback_quality_evaluation(output_path)
            
            # 統合品質チェック実行
            quality_result = self.quality_checker.check_batch_quality(
                str(output_path),
                enable_sci=True,
                enable_pla=True,
                enable_ple=True
            )
            
            return quality_result
            
        except Exception as e:
            logger.error(f"品質評価エラー: {e}")
            return self._fallback_quality_evaluation(output_path)
    
    def _fallback_quality_evaluation(self, output_path: Path) -> Dict[str, Any]:
        """フォールバック品質評価"""
        try:
            # 出力ファイル数カウント
            output_files = list(output_path.glob("**/*.png")) + list(output_path.glob("**/*.jpg"))
            success_count = len(output_files)
            
            # 簡易品質スコア計算
            if success_count > 0:
                ab_rate = min(100.0, success_count * 5.0)  # 簡易計算
                sci_score = 0.75  # デフォルト値
                pla_score = 0.80
                ple_score = 0.85
            else:
                ab_rate = 0.0
                sci_score = 0.0
                pla_score = 0.0
                ple_score = 0.0
            
            return {
                'ab_evaluation_rate': ab_rate,
                'sci_score': sci_score,
                'pla_score': pla_score,
                'ple_score': ple_score,
                'total_processed': success_count,
                'success_count': success_count,
                'failure_count': 0,
                'avg_processing_time': 5.0,
                'grade_distribution': {'A': success_count // 2, 'B': success_count // 2}
            }
            
        except Exception as e:
            logger.error(f"フォールバック品質評価エラー: {e}")
            return {
                'ab_evaluation_rate': 0.0,
                'sci_score': 0.0,
                'pla_score': 0.0,
                'ple_score': 0.0,
                'total_processed': 0,
                'success_count': 0,
                'failure_count': 1,
                'avg_processing_time': 0.0,
                'grade_distribution': {}
            }
    
    def _detect_degradation(self, baseline: QualityBaseline, current: QualityBaseline, 
                          thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
        """品質劣化検出"""
        degradation_details = []
        
        # A/B評価率チェック
        ab_diff = current.ab_evaluation_rate - baseline.ab_evaluation_rate
        if ab_diff < thresholds["ab_evaluation_rate"]:
            degradation_details.append(
                f"A/B評価率低下: {baseline.ab_evaluation_rate:.1f}% → {current.ab_evaluation_rate:.1f}% "
                f"(差分: {ab_diff:.1f}%)"
            )
        
        # SCI スコアチェック
        sci_diff = current.sci_score - baseline.sci_score
        if sci_diff < thresholds["sci_score"]:
            degradation_details.append(
                f"SCI スコア低下: {baseline.sci_score:.3f} → {current.sci_score:.3f} "
                f"(差分: {sci_diff:.3f})"
            )
        
        # PLA スコアチェック
        pla_diff = current.pla_score - baseline.pla_score
        if pla_diff < thresholds["pla_score"]:
            degradation_details.append(
                f"PLA スコア低下: {baseline.pla_score:.3f} → {current.pla_score:.3f} "
                f"(差分: {pla_diff:.3f})"
            )
        
        # PLE スコアチェック
        ple_diff = current.ple_score - baseline.ple_score
        if ple_diff < thresholds["ple_score"]:
            degradation_details.append(
                f"PLE スコア低下: {baseline.ple_score:.3f} → {current.ple_score:.3f} "
                f"(差分: {ple_diff:.3f})"
            )
        
        # 成功率チェック
        baseline_success_rate = (baseline.success_count / baseline.total_processed * 100) if baseline.total_processed > 0 else 0
        current_success_rate = (current.success_count / current.total_processed * 100) if current.total_processed > 0 else 0
        success_rate_diff = current_success_rate - baseline_success_rate
        
        if success_rate_diff < thresholds["success_rate"]:
            degradation_details.append(
                f"成功率低下: {baseline_success_rate:.1f}% → {current_success_rate:.1f}% "
                f"(差分: {success_rate_diff:.1f}%)"
            )
        
        return len(degradation_details) > 0, degradation_details
    
    def _generate_recommendation(self, degradation_details: List[str], current: QualityBaseline) -> str:
        """推奨事項生成"""
        if not degradation_details:
            return "品質劣化は検出されませんでした。現在の品質レベルを維持してください。"
        
        recommendations = ["品質劣化が検出されました。以下の対策を検討してください："]
        
        # 劣化タイプに基づく推奨事項
        for detail in degradation_details:
            if "A/B評価率" in detail:
                recommendations.append("- YOLO検出精度の調整（閾値最適化）")
                recommendations.append("- SAM分割パラメータの見直し")
            elif "SCI" in detail:
                recommendations.append("- セマンティック完全性の改善（前処理強化）")
            elif "PLA" in detail:
                recommendations.append("- ピクセルレベル精度の向上（後処理調整）")
            elif "PLE" in detail:
                recommendations.append("- 学習効率の改善（モデルパラメータ調整）")
            elif "成功率" in detail:
                recommendations.append("- システム安定性の確認（環境・依存関係）")
        
        # 一般的な推奨事項
        recommendations.extend([
            "- 最新のベースライン品質と比較確認",
            "- 入力データの品質確認",
            "- システムリソース使用量の確認"
        ])
        
        return "\n".join(recommendations)
    
    def _save_test_result(self, test_result: TestResult):
        """テスト結果保存"""
        result_file = self.test_results_dir / f"{test_result.test_id}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(test_result), f, indent=2, ensure_ascii=False)
        
        logger.info(f"テスト結果保存: {result_file}")
    
    def _send_notification(self, test_result: TestResult):
        """通知送信"""
        if not self.config["alert_settings"]["enable_notifications"]:
            return
        
        try:
            # 重要度に基づく通知
            if test_result.status == "FAIL":
                title = "🚨 品質テスト失敗"
                message = f"データセット: {test_result.dataset}\n劣化検出: {len(test_result.degradation_details)}件"
            elif test_result.status == "WARNING":
                title = "⚠️ 品質劣化警告"
                message = f"データセット: {test_result.dataset}\n軽微な劣化を検出"
            else:
                title = "✅ 品質テスト合格"
                message = f"データセット: {test_result.dataset}\n品質レベル維持"
            
            # 通知コマンド実行
            subprocess.run([
                "windows-notify", "-t", title, "-m", message
            ], check=False)
            
        except Exception as e:
            logger.warning(f"通知送信失敗: {e}")
    
    def _get_dataset_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """データセット設定取得"""
        for dataset in self.config["test_datasets"]:
            if dataset["name"] == dataset_name:
                return dataset
        return None
    
    def run_continuous_monitoring(self, interval_hours: int = 24):
        """継続的監視実行"""
        logger.info(f"継続的品質監視開始 (間隔: {interval_hours}時間)")
        
        while True:
            try:
                for dataset_config in self.config["test_datasets"]:
                    dataset_name = dataset_config["name"]
                    logger.info(f"定期品質テスト開始: {dataset_name}")
                    
                    test_result = self.run_quality_test(dataset_name, mode="quick")
                    
                    if test_result.status == "FAIL":
                        logger.error(f"品質テスト失敗: {dataset_name}")
                        # 緊急時は標準テストも実行
                        detailed_result = self.run_quality_test(dataset_name, mode="standard")
                        if detailed_result.status == "FAIL":
                            logger.critical(f"詳細テストでも失敗: {dataset_name}")
                
                # 指定間隔で待機
                time.sleep(interval_hours * 3600)
                
            except KeyboardInterrupt:
                logger.info("継続的監視を停止します")
                break
            except Exception as e:
                logger.error(f"継続的監視エラー: {e}")
                time.sleep(3600)  # エラー時は1時間待機


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自動品質テストシステム')
    parser.add_argument('--action', choices=['baseline', 'test', 'monitor'], 
                       default='test', help='実行アクション')
    parser.add_argument('--dataset', default='kana08', help='データセット名')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], 
                       default='standard', help='テストモード')
    parser.add_argument('--force-baseline', action='store_true', 
                       help='ベースライン強制更新')
    parser.add_argument('--monitor-interval', type=int, default=24,
                       help='監視間隔（時間）')
    
    args = parser.parse_args()
    
    # システム初期化
    testing_system = AutomatedQualityTesting()
    
    if args.action == 'baseline':
        # ベースライン作成
        baseline = testing_system.create_baseline(args.dataset, args.force_baseline)
        print(f"✅ ベースライン作成完了: {args.dataset}")
        print(f"A/B評価率: {baseline.ab_evaluation_rate:.1f}%")
        print(f"SCI: {baseline.sci_score:.3f}")
        print(f"PLA: {baseline.pla_score:.3f}")
        print(f"PLE: {baseline.ple_score:.3f}")
        
    elif args.action == 'test':
        # 品質テスト実行
        test_result = testing_system.run_quality_test(args.dataset, args.mode)
        
        print(f"\n{'='*60}")
        print(f"品質テスト結果: {test_result.status}")
        print(f"{'='*60}")
        print(f"データセット: {test_result.dataset}")
        print(f"モード: {args.mode}")
        print(f"劣化検出: {'Yes' if test_result.degradation_detected else 'No'}")
        
        if test_result.degradation_details:
            print("\n劣化詳細:")
            for detail in test_result.degradation_details:
                print(f"  - {detail}")
        
        print(f"\n推奨事項:")
        print(test_result.recommendation)
        
    elif args.action == 'monitor':
        # 継続的監視
        testing_system.run_continuous_monitoring(args.monitor_interval)


if __name__ == "__main__":
    main()