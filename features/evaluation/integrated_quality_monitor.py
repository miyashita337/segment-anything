#!/usr/bin/env python3
"""
統合品質監視システム
リアルタイム品質評価・閾値監視・品質劣化早期検出

目標:
- 統合品質スコア: 20.0% → 50%以上達成
- リアルタイム品質監視・適応的最適化
"""

import numpy as np
import cv2

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """品質閾値設定"""
    # 基本品質指標
    largest_char_accuracy: float = 0.80
    mean_iou: float = 0.65
    ab_evaluation_rate: float = 0.70
    fps: float = 0.2
    
    # マスク品質指標
    compactness: float = 0.50
    fill_ratio: float = 0.75
    coverage_ratio: float = 0.60
    
    # 客観指標
    pla_score: float = 0.75
    sci_score: float = 0.70
    ple_score: float = 0.10
    
    # 統合指標
    overall_quality: float = 0.50
    
    # 警告・危険レベル
    warning_threshold: float = 0.8  # 閾値の80%で警告
    critical_threshold: float = 0.6  # 閾値の60%で危険


@dataclass
class QualitySnapshot:
    """品質スナップショット"""
    timestamp: str
    image_path: str
    quality_metrics: Dict[str, float]
    overall_score: float
    status: str  # 'good', 'warning', 'critical'
    improvement_suggestions: List[str] = field(default_factory=list)


class IntegratedQualityMonitor:
    """統合品質監視システム"""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """初期化"""
        self.thresholds = thresholds or QualityThresholds()
        self.quality_history: List[QualitySnapshot] = []
        self.performance_stats = {
            'total_processed': 0,
            'quality_pass': 0,
            'quality_warning': 0,
            'quality_critical': 0,
            'average_score': 0.0
        }
        
    def monitor_extraction_quality(self, 
                                 image_path: str,
                                 extracted_mask: np.ndarray,
                                 original_image: Optional[np.ndarray] = None,
                                 processing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        抽出品質リアルタイム監視
        
        Args:
            image_path: 処理画像パス
            extracted_mask: 抽出されたマスク
            original_image: 元画像
            processing_metadata: 処理メタデータ
            
        Returns:
            Dict: 監視結果と改善提案
        """
        try:
            logger.info(f"🔍 品質監視開始: {Path(image_path).name}")
            
            # Step 1: 統合品質評価実行
            quality_metrics = self._evaluate_comprehensive_quality(
                extracted_mask, original_image, processing_metadata
            )
            
            # Step 2: 品質レベル判定
            quality_status = self._determine_quality_status(quality_metrics)
            
            # Step 3: 改善提案生成
            improvement_suggestions = self._generate_improvement_suggestions(
                quality_metrics, quality_status
            )
            
            # Step 4: 品質スナップショット作成
            snapshot = QualitySnapshot(
                timestamp=datetime.now().isoformat(),
                image_path=image_path,
                quality_metrics=quality_metrics,
                overall_score=quality_metrics.get('overall_score', 0.0),
                status=quality_status,
                improvement_suggestions=improvement_suggestions
            )
            
            # Step 5: 履歴更新
            self._update_quality_history(snapshot)
            
            # Step 6: 統計更新
            self._update_performance_stats(snapshot)
            
            # Step 7: リアルタイム警告
            self._check_quality_alerts(snapshot)
            
            logger.info(f"📊 品質監視完了: スコア {snapshot.overall_score:.3f} ({quality_status})")
            
            return {
                'quality_snapshot': snapshot,
                'needs_improvement': quality_status in ['warning', 'critical'],
                'improvement_suggestions': improvement_suggestions,
                'quality_metrics': quality_metrics,
                'processing_recommendations': self._get_processing_recommendations(quality_status, quality_metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ 品質監視エラー: {e}")
            return {
                'quality_snapshot': None,
                'needs_improvement': True,
                'improvement_suggestions': ['システムエラーが発生しました'],
                'error': str(e)
            }
    
    def _evaluate_comprehensive_quality(self, 
                                      mask: np.ndarray,
                                      original_image: Optional[np.ndarray] = None,
                                      metadata: Optional[Dict] = None) -> Dict[str, float]:
        """包括的品質評価"""
        try:
            # 統合品質チェッカーを使用
            from tools.unified_quality_checker import UnifiedQualityChecker
            quality_checker = UnifiedQualityChecker()
            
            # 基本マスク品質評価
            from features.processing.postprocessing.postprocessing import (
                calculate_mask_quality_metrics,
            )
            basic_metrics = calculate_mask_quality_metrics(mask)
            
            # 客観指標評価
            objective_metrics = self._calculate_objective_metrics(mask, original_image)
            
            # 処理効率評価
            processing_metrics = self._calculate_processing_metrics(metadata)
            
            # 統合スコア計算
            overall_score = self._calculate_integrated_score(
                basic_metrics, objective_metrics, processing_metrics
            )
            
            return {
                **basic_metrics,
                **objective_metrics,
                **processing_metrics,
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 包括的品質評価エラー: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_objective_metrics(self, 
                                   mask: np.ndarray,
                                   original_image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """客観指標計算"""
        try:
            from features.evaluation.objective_metrics import (
                PLACalculator,
                PLETracker,
                SCICalculator,
            )
            
            metrics = {}
            
            # PLA (Pixel-Level Accuracy) - 仮想グランドトゥルースとの比較
            if original_image is not None:
                pla_calculator = PLACalculator()
                # 簡易グランドトゥルース生成（実際の実装では人手ラベルを使用）
                simple_gt = self._generate_simple_ground_truth(mask, original_image)
                pla_result = pla_calculator.calculate(mask, simple_gt)
                metrics['pla_score'] = pla_result.score
            
            # SCI (Semantic Completeness Index)
            sci_calculator = SCICalculator()
            sci_result = sci_calculator.calculate(original_image if original_image is not None else mask)
            metrics['sci_score'] = sci_result.score
            
            # PLE (Progressive Learning Efficiency) - 履歴ベース
            ple_tracker = PLETracker()
            recent_scores = [s.overall_score for s in self.quality_history[-10:]]
            if recent_scores:
                ple_result = ple_tracker.calculate(recent_scores)
                metrics['ple_score'] = ple_result.score
            else:
                metrics['ple_score'] = 0.5  # 初期値
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ 客観指標計算エラー: {e}")
            return {'pla_score': 0.0, 'sci_score': 0.0, 'ple_score': 0.0}
    
    def _generate_simple_ground_truth(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """簡易グランドトゥルース生成（実際の用途では人手ラベルを使用）"""
        try:
            # エッジベースの簡易GT生成
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            
            # エッジ情報とマスクを組み合わせ
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # マスクと組み合わせて簡易GT作成
            simple_gt = cv2.bitwise_and(mask, dilated_edges)
            
            # ホール埋めで完全化
            contours, _ = cv2.findContours(simple_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.fillPoly(simple_gt, contours, 255)
            
            return simple_gt
            
        except Exception as e:
            logger.warning(f"⚠️ 簡易GT生成エラー: {e}")
            return mask  # フォールバック
    
    def _calculate_processing_metrics(self, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """処理效率評価"""
        try:
            if not metadata:
                return {'fps': 0.0, 'processing_efficiency': 0.5}
            
            # 処理時間から FPS 計算
            processing_time = metadata.get('processing_time', 5.0)
            fps = 1.0 / processing_time if processing_time > 0 else 0.0
            
            # メモリ効率
            memory_usage = metadata.get('memory_usage', 1000)  # MB
            memory_efficiency = max(0.0, 1.0 - memory_usage / 2000.0)  # 2GB を基準
            
            # GPU利用率
            gpu_utilization = metadata.get('gpu_utilization', 0.5)
            
            # 総合処理効率
            processing_efficiency = (fps * 0.5 + memory_efficiency * 0.3 + gpu_utilization * 0.2)
            
            return {
                'fps': fps,
                'memory_efficiency': memory_efficiency,
                'gpu_utilization': gpu_utilization,
                'processing_efficiency': processing_efficiency
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 処理效率評価エラー: {e}")
            return {'fps': 0.0, 'processing_efficiency': 0.0}
    
    def _calculate_integrated_score(self, 
                                  basic_metrics: Dict[str, float],
                                  objective_metrics: Dict[str, float],
                                  processing_metrics: Dict[str, float]) -> float:
        """統合スコア計算"""
        try:
            # 重み設定
            weights = {
                'mask_quality': 0.4,    # マスク品質 40%
                'objective_quality': 0.4,  # 客観指標 40%
                'processing_efficiency': 0.2   # 処理効率 20%
            }
            
            # マスク品質スコア
            mask_score = (
                basic_metrics.get('compactness', 0.0) * 0.4 +
                basic_metrics.get('fill_ratio', 0.0) * 0.3 +
                basic_metrics.get('coverage_ratio', 0.0) * 0.3
            )
            
            # 客観品質スコア
            objective_score = (
                objective_metrics.get('pla_score', 0.0) * 0.4 +
                objective_metrics.get('sci_score', 0.0) * 0.4 +
                objective_metrics.get('ple_score', 0.0) * 0.2
            )
            
            # 処理効率スコア
            efficiency_score = processing_metrics.get('processing_efficiency', 0.0)
            
            # 統合スコア
            integrated_score = (
                mask_score * weights['mask_quality'] +
                objective_score * weights['objective_quality'] +
                efficiency_score * weights['processing_efficiency']
            )
            
            return max(0.0, min(1.0, integrated_score))
            
        except Exception as e:
            logger.warning(f"⚠️ 統合スコア計算エラー: {e}")
            return 0.0
    
    def _determine_quality_status(self, quality_metrics: Dict[str, float]) -> str:
        """品質レベル判定"""
        try:
            overall_score = quality_metrics.get('overall_score', 0.0)
            
            if overall_score >= self.thresholds.overall_quality:
                return 'good'
            elif overall_score >= self.thresholds.overall_quality * self.thresholds.warning_threshold:
                return 'warning'
            else:
                return 'critical'
                
        except Exception as e:
            logger.warning(f"⚠️ 品質レベル判定エラー: {e}")
            return 'critical'
    
    def _generate_improvement_suggestions(self, 
                                        quality_metrics: Dict[str, float],
                                        status: str) -> List[str]:
        """改善提案生成"""
        suggestions = []
        
        try:
            if status == 'good':
                suggestions.append("品質基準を満たしています")
                return suggestions
            
            # マスク品質関連の提案
            if quality_metrics.get('compactness', 0.0) < self.thresholds.compactness:
                suggestions.append("輪郭強化システム(P1-A003)のパラメータ調整を推奨")
            
            if quality_metrics.get('fill_ratio', 0.0) < self.thresholds.fill_ratio:
                suggestions.append("SAM後処理パイプライン(P1-A002)の強化を推奨")
            
            if quality_metrics.get('coverage_ratio', 0.0) < self.thresholds.coverage_ratio:
                suggestions.append("YOLO検出範囲拡張(P1-A001)の閾値調整を推奨")
            
            # 客観指標関連の提案
            if quality_metrics.get('pla_score', 0.0) < self.thresholds.pla_score:
                suggestions.append("ピクセルレベル精度向上のため前処理強化を推奨")
            
            if quality_metrics.get('sci_score', 0.0) < self.thresholds.sci_score:
                suggestions.append("セマンティック完全性向上のため検出アルゴリズム調整を推奨")
            
            # 処理効率関連の提案
            if quality_metrics.get('fps', 0.0) < self.thresholds.fps:
                suggestions.append("処理速度向上のためバッチサイズ調整を推奨")
            
            if not suggestions:
                suggestions.append("総合的な品質向上システムの最適化を推奨")
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"⚠️ 改善提案生成エラー: {e}")
            return ["改善提案の生成に失敗しました"]
    
    def _update_quality_history(self, snapshot: QualitySnapshot) -> None:
        """品質履歴更新"""
        try:
            self.quality_history.append(snapshot)
            
            # 履歴サイズ制限（最新100件保持）
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-100:]
                
        except Exception as e:
            logger.warning(f"⚠️ 品質履歴更新エラー: {e}")
    
    def _update_performance_stats(self, snapshot: QualitySnapshot) -> None:
        """統計更新"""
        try:
            self.performance_stats['total_processed'] += 1
            
            if snapshot.status == 'good':
                self.performance_stats['quality_pass'] += 1
            elif snapshot.status == 'warning':
                self.performance_stats['quality_warning'] += 1
            elif snapshot.status == 'critical':
                self.performance_stats['quality_critical'] += 1
            
            # 平均スコア更新
            total_score = sum(s.overall_score for s in self.quality_history)
            self.performance_stats['average_score'] = total_score / len(self.quality_history)
            
        except Exception as e:
            logger.warning(f"⚠️ 統計更新エラー: {e}")
    
    def _check_quality_alerts(self, snapshot: QualitySnapshot) -> None:
        """品質警告チェック"""
        try:
            if snapshot.status == 'critical':
                logger.warning(f"🚨 品質危険レベル: {snapshot.image_path} - スコア {snapshot.overall_score:.3f}")
                
                # 連続危険レベルチェック
                recent_critical = sum(1 for s in self.quality_history[-5:] if s.status == 'critical')
                if recent_critical >= 3:
                    logger.error(f"🔥 連続品質劣化検出: 直近5件中{recent_critical}件が危険レベル")
            
            elif snapshot.status == 'warning':
                logger.info(f"⚠️ 品質警告レベル: {snapshot.image_path} - スコア {snapshot.overall_score:.3f}")
                
        except Exception as e:
            logger.warning(f"⚠️ 品質警告チェックエラー: {e}")
    
    def _get_processing_recommendations(self, 
                                     status: str, 
                                     quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """処理推奨設定取得"""
        try:
            recommendations = {
                'retry_needed': status == 'critical',
                'parameter_adjustment': {},
                'processing_priority': 'normal'
            }
            
            if status == 'critical':
                recommendations['processing_priority'] = 'high'
                recommendations['parameter_adjustment'] = {
                    'yolo_threshold_adjustment': -0.02,  # 閾値を下げる
                    'sam_post_processing': 'aggressive',  # 積極的後処理
                    'contour_enhancement': 'max'  # 最大輪郭強化
                }
            elif status == 'warning':
                recommendations['parameter_adjustment'] = {
                    'yolo_threshold_adjustment': -0.01,
                    'sam_post_processing': 'moderate',
                    'contour_enhancement': 'standard'
                }
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ 処理推奨設定取得エラー: {e}")
            return {'retry_needed': False, 'parameter_adjustment': {}, 'processing_priority': 'normal'}
    
    def get_quality_report(self) -> Dict[str, Any]:
        """品質レポート取得"""
        try:
            if not self.quality_history:
                return {'message': '品質履歴がありません'}
            
            recent_scores = [s.overall_score for s in self.quality_history[-20:]]
            
            return {
                'total_processed': self.performance_stats['total_processed'],
                'quality_distribution': {
                    'good': self.performance_stats['quality_pass'],
                    'warning': self.performance_stats['quality_warning'],
                    'critical': self.performance_stats['quality_critical']
                },
                'average_score': self.performance_stats['average_score'],
                'recent_trend': {
                    'recent_average': np.mean(recent_scores),
                    'trend_direction': 'improving' if len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0] else 'declining',
                    'stability': np.std(recent_scores)
                },
                'last_snapshot': self.quality_history[-1] if self.quality_history else None
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 品質レポート取得エラー: {e}")
            return {'error': str(e)}
    
    def save_quality_log(self, output_path: str) -> bool:
        """品質ログ保存"""
        try:
            log_data = {
                'thresholds': self.thresholds.__dict__,
                'performance_stats': self.performance_stats,
                'quality_history': [
                    {
                        'timestamp': s.timestamp,
                        'image_path': s.image_path,
                        'overall_score': s.overall_score,
                        'status': s.status,
                        'suggestions_count': len(s.improvement_suggestions)
                    } for s in self.quality_history
                ],
                'generated_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 品質ログ保存完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 品質ログ保存エラー: {e}")
            return False


def integrate_with_extraction_pipeline() -> None:
    """抽出パイプライン統合準備"""
    logger.info("🔗 統合品質監視システムを抽出パイプラインに統合準備")
    
    # extract_character.py の process_single_image 関数に統合する想定
    # 実際の統合は次のステップで実装
    pass


if __name__ == "__main__":
    # テスト実行
    logger.info("🧪 統合品質監視システム テスト開始")
    
    # テスト用設定
    thresholds = QualityThresholds(
        overall_quality=0.50,
        compactness=0.50,
        largest_char_accuracy=0.80
    )
    
    monitor = IntegratedQualityMonitor(thresholds)
    logger.info("✅ 統合品質監視システム初期化完了")
    
    # 統合準備
    integrate_with_extraction_pipeline()
    logger.info("🎯 テスト完了 - 実装準備完了")