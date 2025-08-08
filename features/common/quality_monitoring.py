#!/usr/bin/env python3
"""
P1-B001: 統合品質チェック自動実行システム
継続的品質監視: 24時間365日の自動監視
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from tools.core.automated_quality_testing import AutomatedQualityTester
    from tools.core.quality_dashboard import QualityDashboard
    from tools.core.unified_quality_checker import UnifiedQualityChecker
    QUALITY_TOOLS_AVAILABLE = True
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    QUALITY_TOOLS_AVAILABLE = False
    DASHBOARD_AVAILABLE = False
    try:
        from tools.core.quality_dashboard import QualityDashboard
        DASHBOARD_AVAILABLE = True
    except ImportError:
        DASHBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class QualityMonitoringConfig:
    """品質監視設定"""
    enabled: bool = True
    auto_check_after_extraction: bool = True
    quality_threshold: float = 0.7
    alert_on_degradation: bool = True
    degradation_threshold: float = 0.1
    dashboard_generation: bool = True
    notification_enabled: bool = True
    baseline_update_frequency: int = 10  # 10回の実行ごとにベースライン更新
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityMonitoringConfig':
        return cls(**data)

@dataclass
class QualityResult:
    """品質チェック結果"""
    timestamp: str
    success_rate: float
    avg_quality_score: float
    total_processed: int
    quality_grades: Dict[str, int]
    degradation_detected: bool = False
    baseline_comparison: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class IntegratedQualityMonitor:
    """統合品質監視システム"""
    
    def __init__(self, workspace_path: Path, config: Optional[QualityMonitoringConfig] = None):
        self.workspace_path = workspace_path
        self.config = config or QualityMonitoringConfig()
        self.quality_history_file = workspace_path / "quality_history.json"
        self.baseline_file = workspace_path / "quality_baseline.json"
        
        # 品質チェックツール初期化
        if QUALITY_TOOLS_AVAILABLE:
            self.quality_checker = UnifiedQualityChecker()
            self.quality_tester = None
            try:
                self.quality_tester = AutomatedQualityTester()
            except Exception as e:
                logger.warning(f"AutomatedQualityTester初期化失敗: {e}")
        else:
            logger.warning("品質チェックツールが利用できません")
            self.quality_checker = None
            self.quality_tester = None
    
    def run_quality_check(self, extraction_results_dir: Path) -> QualityResult:
        """抽出結果の品質チェック実行"""
        logger.info("🔍 統合品質チェック開始")
        
        if not extraction_results_dir.exists():
            logger.error(f"抽出結果ディレクトリが存在しません: {extraction_results_dir}")
            return self._create_error_result()
        
        # 抽出画像の品質評価
        image_files = list(extraction_results_dir.glob("*.jpg")) + list(extraction_results_dir.glob("*.png"))
        
        if not image_files:
            logger.warning("評価対象の画像が見つかりません")
            return self._create_empty_result()
        
        total_processed = len(image_files)
        quality_scores = []
        quality_grades = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        
        for image_file in image_files:
            try:
                # 統合品質チェッカーによる評価
                if self.quality_checker:
                    quality_result = self.quality_checker.evaluate_extracted_image(image_file)
                    quality_scores.append(quality_result.get('overall_score', 0.0))
                    grade = quality_result.get('grade', 'F')
                    quality_grades[grade] = quality_grades.get(grade, 0) + 1
                else:
                    # フォールバック: 簡易品質評価
                    score = self._simple_quality_check(image_file)
                    quality_scores.append(score)
                    grade = self._score_to_grade(score)
                    quality_grades[grade] += 1
                    
            except Exception as e:
                logger.error(f"画像品質評価エラー {image_file}: {e}")
                quality_scores.append(0.0)
                quality_grades["F"] += 1
        
        # 結果集計
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        success_rate = (quality_grades["A"] + quality_grades["B"]) / total_processed if total_processed > 0 else 0.0
        
        result = QualityResult(
            timestamp=datetime.now().isoformat(),
            success_rate=success_rate,
            avg_quality_score=avg_quality_score,
            total_processed=total_processed,
            quality_grades=quality_grades
        )
        
        # 劣化検出
        if self.config.alert_on_degradation:
            result.degradation_detected, result.baseline_comparison = self._check_degradation(result)
        
        # 履歴保存
        self._save_quality_history(result)
        
        # ベースライン更新
        self._update_baseline_if_needed(result)
        
        logger.info(f"✅ 品質チェック完了: 成功率 {success_rate:.1%}, 平均スコア {avg_quality_score:.3f}")
        
        return result
    
    def _simple_quality_check(self, image_file: Path) -> float:
        """簡易品質チェック（フォールバック）"""
        try:
            import numpy as np

            from PIL import Image
            
            with Image.open(image_file) as img:
                # 画像サイズチェック
                width, height = img.size
                if width < 50 or height < 50:
                    return 0.1
                
                # アスペクト比チェック
                aspect_ratio = width / height
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    return 0.3
                
                # 基本品質スコア
                base_score = 0.6
                
                # サイズボーナス
                if width > 200 and height > 200:
                    base_score += 0.2
                
                # アスペクト比ボーナス（人物らしい比率）
                if 1.5 <= aspect_ratio <= 2.5:
                    base_score += 0.1
                
                return min(base_score, 1.0)
                
        except Exception as e:
            logger.error(f"簡易品質チェックエラー: {e}")
            return 0.0
    
    def _score_to_grade(self, score: float) -> str:
        """スコアを品質グレードに変換"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        elif score >= 0.5:
            return "E"
        else:
            return "F"
    
    def _check_degradation(self, current: QualityResult) -> Tuple[bool, Optional[Dict[str, float]]]:
        """品質劣化検出"""
        if not self.baseline_file.exists():
            return False, None
        
        try:
            with open(self.baseline_file, 'r') as f:
                baseline = json.load(f)
            
            current_success_rate = current.success_rate
            current_avg_score = current.avg_quality_score
            
            baseline_success_rate = baseline.get('success_rate', 0.0)
            baseline_avg_score = baseline.get('avg_quality_score', 0.0)
            
            success_rate_drop = baseline_success_rate - current_success_rate
            avg_score_drop = baseline_avg_score - current_avg_score
            
            degradation_detected = (
                success_rate_drop > self.config.degradation_threshold or
                avg_score_drop > self.config.degradation_threshold
            )
            
            comparison = {
                'baseline_success_rate': baseline_success_rate,
                'current_success_rate': current_success_rate,
                'success_rate_change': -success_rate_drop,
                'baseline_avg_score': baseline_avg_score,
                'current_avg_score': current_avg_score,
                'avg_score_change': -avg_score_drop
            }
            
            if degradation_detected:
                logger.warning(f"⚠️ 品質劣化検出: 成功率変化 {-success_rate_drop:+.1%}, スコア変化 {-avg_score_drop:+.3f}")
            
            return degradation_detected, comparison
            
        except Exception as e:
            logger.error(f"劣化検出エラー: {e}")
            return False, None
    
    def _save_quality_history(self, result: QualityResult):
        """品質履歴保存"""
        try:
            history = []
            if self.quality_history_file.exists():
                with open(self.quality_history_file, 'r') as f:
                    history = json.load(f)
            
            history.append(result.to_dict())
            
            # 履歴サイズ制限（最新100件）
            if len(history) > 100:
                history = history[-100:]
            
            with open(self.quality_history_file, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"品質履歴保存エラー: {e}")
    
    def _update_baseline_if_needed(self, result: QualityResult):
        """ベースライン更新"""
        try:
            history = []
            if self.quality_history_file.exists():
                with open(self.quality_history_file, 'r') as f:
                    history = json.load(f)
            
            # ベースライン更新頻度チェック
            if len(history) % self.config.baseline_update_frequency == 0:
                # 最近の結果からベースライン計算
                recent_results = history[-self.config.baseline_update_frequency:]
                
                avg_success_rate = sum(r['success_rate'] for r in recent_results) / len(recent_results)
                avg_quality_score = sum(r['avg_quality_score'] for r in recent_results) / len(recent_results)
                
                baseline = {
                    'success_rate': avg_success_rate,
                    'avg_quality_score': avg_quality_score,
                    'updated_at': datetime.now().isoformat(),
                    'sample_size': len(recent_results)
                }
                
                with open(self.baseline_file, 'w') as f:
                    json.dump(baseline, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📊 品質ベースライン更新: 成功率 {avg_success_rate:.1%}, 平均スコア {avg_quality_score:.3f}")
                
        except Exception as e:
            logger.error(f"ベースライン更新エラー: {e}")
    
    def _create_error_result(self) -> QualityResult:
        """エラー時の結果作成"""
        return QualityResult(
            timestamp=datetime.now().isoformat(),
            success_rate=0.0,
            avg_quality_score=0.0,
            total_processed=0,
            quality_grades={"F": 1}
        )
    
    def _create_empty_result(self) -> QualityResult:
        """空の結果作成"""
        return QualityResult(
            timestamp=datetime.now().isoformat(),
            success_rate=0.0,
            avg_quality_score=0.0,
            total_processed=0,
            quality_grades={}
        )
    
    def generate_quality_dashboard(self, output_dir: Path) -> Optional[Path]:
        """品質ダッシュボード生成"""
        if not self.config.dashboard_generation:
            return None
        
        try:
            if not DASHBOARD_AVAILABLE:
                logger.warning("QualityDashboardが利用できません")
                return None
                
            dashboard_generator = QualityDashboard()
            
            # 品質履歴データ読み込み
            history = []
            if self.quality_history_file.exists():
                with open(self.quality_history_file, 'r') as f:
                    history = json.load(f)
            
            if not history:
                logger.warning("品質履歴データが存在しません")
                return None
            
            # ダッシュボード生成
            dashboard_path = output_dir / "quality_monitoring_dashboard.html"
            dashboard_generator.create_monitoring_dashboard(history, dashboard_path)
            
            logger.info(f"📊 品質監視ダッシュボード生成: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            logger.error(f"ダッシュボード生成エラー: {e}")
            return None


def create_quality_monitor(workspace_path: Path, config: Optional[QualityMonitoringConfig] = None) -> IntegratedQualityMonitor:
    """品質監視システム作成"""
    return IntegratedQualityMonitor(workspace_path, config)


# P1-B001: extract_character.py用フック関数
def run_integrated_quality_check(workspace_path: Path, tracker_id: Optional[str] = None) -> Optional[QualityResult]:
    """
    統合品質チェック実行（extract_character.py用）
    
    Args:
        workspace_path: ワークスペースパス
        tracker_id: トラッカーID（通知用）
        
    Returns:
        品質チェック結果
    """
    try:
        config = QualityMonitoringConfig()
        monitor = create_quality_monitor(workspace_path, config)
        
        # 抽出結果ディレクトリ確認
        extraction_dir = workspace_path / "extraction"
        if not extraction_dir.exists():
            # 直接workspace内の画像ファイルをチェック
            extraction_dir = workspace_path
        
        # 品質チェック実行
        result = monitor.run_quality_check(extraction_dir)
        
        # ダッシュボード生成
        dashboard_dir = workspace_path / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        
        dashboard_path = monitor.generate_quality_dashboard(dashboard_dir)
        
        if result.degradation_detected and config.notification_enabled:
            logger.warning(f"🚨 品質劣化検出 ({tracker_id}): 即座に対応が必要です")
        
        return result
        
    except Exception as e:
        logger.error(f"統合品質チェックエラー: {e}")
        return None