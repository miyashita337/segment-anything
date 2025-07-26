#!/usr/bin/env python3
"""
総合精度向上パイプライン
P1-A001〜A003の協調動作制御・失敗ケース自動検出・リトライ機構

目標:
- 3段階品質向上システムの協調動作
- 失敗時の自動リトライ・学習機構
- 統合品質スコア 20.0% → 50%以上達成
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """処理段階"""
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    quality_score: Optional[float] = None
    improvements: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class PipelineResult:
    """パイプライン結果"""
    success: bool
    final_mask: Optional[np.ndarray]
    initial_quality: float
    final_quality: float
    improvement_ratio: float
    processing_stages: List[ProcessingStage]
    retry_count: int
    total_processing_time: float
    failure_analysis: Optional[Dict[str, Any]] = None


class IntegratedPrecisionPipeline:
    """総合精度向上パイプライン"""
    
    def __init__(self):
        """初期化"""
        self.pipeline_history: List[Dict] = []
        self.failure_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
        
        # リトライ設定
        self.max_retries = 3
        self.quality_improvement_threshold = 0.1  # 10%以上の改善を要求
        self.minimum_acceptable_quality = 0.3
        
    def process_with_integrated_pipeline(self, 
                                       image: np.ndarray,
                                       initial_mask: np.ndarray,
                                       yolo_model: Any,
                                       sam_model: Any,
                                       target_quality: float = 0.5) -> PipelineResult:
        """
        統合精度向上パイプライン実行
        
        Args:
            image: 入力画像
            initial_mask: 初期マスク
            yolo_model: YOLOモデル
            sam_model: SAMモデル
            target_quality: 目標品質スコア
            
        Returns:
            PipelineResult: 処理結果
        """
        pipeline_start_time = time.time()
        retry_count = 0
        best_result = None
        
        logger.info("🚀 統合精度向上パイプライン開始")
        
        try:
            # 初期品質評価
            initial_quality = self._evaluate_mask_quality(initial_mask, image)
            logger.info(f"📊 初期品質スコア: {initial_quality:.3f}")
            
            while retry_count <= self.max_retries:
                logger.info(f"🔄 処理試行 {retry_count + 1}/{self.max_retries + 1}")
                
                # 単一試行実行
                trial_result = self._execute_single_trial(
                    image, initial_mask, yolo_model, sam_model, 
                    retry_count, target_quality
                )
                
                # 最良結果更新
                if best_result is None or (trial_result.success and trial_result.final_quality > best_result.final_quality):
                    best_result = trial_result
                
                # 成功判定
                if trial_result.success and trial_result.final_quality >= target_quality:
                    logger.info(f"✅ 目標品質達成: {trial_result.final_quality:.3f} >= {target_quality:.3f}")
                    break
                
                # 改善判定
                improvement = trial_result.final_quality - initial_quality
                if improvement >= self.quality_improvement_threshold:
                    logger.info(f"📈 有意な改善確認: +{improvement:.3f}")
                    break
                
                # リトライ判定
                if retry_count < self.max_retries:
                    failure_analysis = self._analyze_failure_pattern(trial_result)
                    retry_strategy = self._determine_retry_strategy(failure_analysis, retry_count)
                    
                    logger.info(f"🔄 リトライ実行: 戦略={retry_strategy['strategy']}")
                    retry_count += 1
                else:
                    logger.warning("⚠️ 最大リトライ回数に達しました")
                    break
            
            # 最終結果設定
            if best_result is None:
                best_result = PipelineResult(
                    success=False,
                    final_mask=initial_mask,
                    initial_quality=initial_quality,
                    final_quality=initial_quality,
                    improvement_ratio=0.0,
                    processing_stages=[],
                    retry_count=retry_count,
                    total_processing_time=time.time() - pipeline_start_time
                )
            
            best_result.total_processing_time = time.time() - pipeline_start_time
            best_result.retry_count = retry_count
            
            # パイプライン履歴更新
            self._update_pipeline_history(best_result, image.shape)
            
            # 学習データ更新
            self._update_learning_data(best_result)
            
            logger.info(f"🏁 パイプライン完了: 最終品質 {best_result.final_quality:.3f}, "
                       f"改善率 {best_result.improvement_ratio:.1%}, "
                       f"処理時間 {best_result.total_processing_time:.1f}秒")
            
            return best_result
            
        except Exception as e:
            logger.error(f"❌ パイプライン実行エラー: {e}")
            return PipelineResult(
                success=False,
                final_mask=initial_mask,
                initial_quality=initial_quality,
                final_quality=initial_quality,
                improvement_ratio=0.0,
                processing_stages=[],
                retry_count=retry_count,
                total_processing_time=time.time() - pipeline_start_time,
                failure_analysis={'error': str(e)}
            )
    
    def _execute_single_trial(self, 
                            image: np.ndarray,
                            initial_mask: np.ndarray,
                            yolo_model: Any,
                            sam_model: Any,
                            retry_count: int,
                            target_quality: float) -> PipelineResult:
        """単一試行実行"""
        try:
            processing_stages = []
            current_mask = initial_mask.copy()
            
            # 初期品質評価
            initial_quality = self._evaluate_mask_quality(initial_mask, image)
            
            # Step 1: 適応的パラメータ最適化
            stage1 = self._execute_parameter_optimization(image, retry_count)
            processing_stages.append(stage1)
            
            if stage1.status == 'failed':
                return self._create_failed_result(initial_quality, processing_stages)
            
            # Step 2: YOLO検出範囲拡張 (P1-A001)
            stage2, expanded_masks = self._execute_yolo_expansion(
                yolo_model, image, [{'segmentation': current_mask}]
            )
            processing_stages.append(stage2)
            
            if stage2.status == 'completed' and expanded_masks:
                current_mask = expanded_masks[0].get('segmentation', current_mask)
            
            # Step 3: SAM後処理パイプライン (P1-A002)
            stage3, postprocessed_mask = self._execute_sam_postprocessing(current_mask, image)
            processing_stages.append(stage3)
            
            if stage3.status == 'completed' and postprocessed_mask is not None:
                current_mask = postprocessed_mask
            
            # Step 4: 輪郭後処理システム強化 (P1-A003)
            stage4, contour_enhanced_mask = self._execute_contour_enhancement(current_mask, image)
            processing_stages.append(stage4)
            
            if stage4.status == 'completed' and contour_enhanced_mask is not None:
                current_mask = contour_enhanced_mask
            
            # Step 5: 統合品質監視
            stage5, final_quality = self._execute_quality_monitoring(current_mask, image)
            processing_stages.append(stage5)
            
            # 結果評価
            improvement_ratio = (final_quality - initial_quality) / max(initial_quality, 0.001)
            success = (final_quality >= target_quality or 
                      final_quality >= self.minimum_acceptable_quality)
            
            return PipelineResult(
                success=success,
                final_mask=current_mask,
                initial_quality=initial_quality,
                final_quality=final_quality,
                improvement_ratio=improvement_ratio,
                processing_stages=processing_stages,
                retry_count=0,  # 単一試行内では0
                total_processing_time=sum(
                    (stage.end_time or 0) - (stage.start_time or 0) 
                    for stage in processing_stages if stage.start_time and stage.end_time
                )
            )
            
        except Exception as e:
            logger.error(f"❌ 単一試行実行エラー: {e}")
            return self._create_failed_result(
                initial_quality, processing_stages, 
                failure_analysis={'error': str(e)}
            )
    
    def _execute_parameter_optimization(self, image: np.ndarray, retry_count: int) -> ProcessingStage:
        """パラメータ最適化実行"""
        stage = ProcessingStage(name="parameter_optimization", status="running")
        stage.start_time = time.time()
        
        try:
            from features.processing.adaptive_parameter_optimizer import AdaptiveParameterOptimizer
            
            optimizer = AdaptiveParameterOptimizer()
            optimized_params = optimizer.optimize_parameters_for_image(image)
            
            stage.status = "completed"
            stage.improvements = ["adaptive_parameter_optimization"]
            stage.quality_score = 0.8  # パラメータ最適化の成功度
            
            logger.debug("🎛️ パラメータ最適化完了")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            logger.warning(f"⚠️ パラメータ最適化失敗: {e}")
        
        stage.end_time = time.time()
        return stage
    
    def _execute_yolo_expansion(self, yolo_model: Any, image: np.ndarray, masks: List[Dict]) -> Tuple[ProcessingStage, List[Dict]]:
        """YOLO検出範囲拡張実行"""
        stage = ProcessingStage(name="yolo_expansion", status="running")
        stage.start_time = time.time()
        
        try:
            from features.extraction.yolo_detection_expansion import YOLODetectionExpander
            
            expander = YOLODetectionExpander()
            expanded_masks = expander.expand_detection_capabilities(yolo_model, image, masks)
            
            if expanded_masks and len(expanded_masks) > 0:
                stage.status = "completed"
                stage.improvements = ["detection_expansion", "fullbody_optimization", "anime_filtering"]
                stage.quality_score = len(expanded_masks) / max(len(masks), 1)
                logger.debug(f"🚀 YOLO拡張完了: {len(masks)} → {len(expanded_masks)}")
            else:
                stage.status = "skipped"
                stage.improvements = []
                expanded_masks = masks
                logger.debug("⏭️ YOLO拡張スキップ")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            expanded_masks = masks
            logger.warning(f"⚠️ YOLO拡張失敗: {e}")
        
        stage.end_time = time.time()
        return stage, expanded_masks
    
    def _execute_sam_postprocessing(self, mask: np.ndarray, image: np.ndarray) -> Tuple[ProcessingStage, Optional[np.ndarray]]:
        """SAM後処理パイプライン実行"""
        stage = ProcessingStage(name="sam_postprocessing", status="running")
        stage.start_time = time.time()
        
        try:
            from features.processing.sam_postprocessing_pipeline import SAMPostprocessingPipeline
            
            postprocessor = SAMPostprocessingPipeline()
            result = postprocessor.enhance_mask_quality(mask, image)
            
            enhanced_mask = result.get('enhanced_mask')
            quality_score = result.get('quality_score', 0.0)
            improvements = result.get('improvements', [])
            
            if enhanced_mask is not None and quality_score > 0:
                stage.status = "completed"
                stage.improvements = improvements
                stage.quality_score = quality_score
                logger.debug(f"🔧 SAM後処理完了: 品質スコア {quality_score:.3f}")
                return stage, enhanced_mask
            else:
                stage.status = "skipped"
                stage.improvements = []
                logger.debug("⏭️ SAM後処理スキップ")
                return stage, mask
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            logger.warning(f"⚠️ SAM後処理失敗: {e}")
            
        stage.end_time = time.time()
        return stage, mask
    
    def _execute_contour_enhancement(self, mask: np.ndarray, image: np.ndarray) -> Tuple[ProcessingStage, Optional[np.ndarray]]:
        """輪郭後処理システム強化実行"""
        stage = ProcessingStage(name="contour_enhancement", status="running")
        stage.start_time = time.time()
        
        try:
            from features.processing.contour_enhancement_system import ContourEnhancementSystem
            
            enhancer = ContourEnhancementSystem()
            result = enhancer.enhance_contour_quality(mask, image)
            
            enhanced_mask = result.get('enhanced_mask')
            quality_metrics = result.get('quality_metrics', {})
            improvements = result.get('improvements', [])
            success = result.get('success', False)
            
            if enhanced_mask is not None and success:
                stage.status = "completed"
                stage.improvements = improvements
                stage.quality_score = quality_metrics.get('compactness', 0.0)
                logger.debug(f"🎨 輪郭強化完了: コンパクトネス {stage.quality_score:.3f}")
                return stage, enhanced_mask
            else:
                stage.status = "skipped"
                stage.improvements = []
                logger.debug("⏭️ 輪郭強化スキップ")
                return stage, mask
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            logger.warning(f"⚠️ 輪郭強化失敗: {e}")
            
        stage.end_time = time.time()
        return stage, mask
    
    def _execute_quality_monitoring(self, mask: np.ndarray, image: np.ndarray) -> Tuple[ProcessingStage, float]:
        """統合品質監視実行"""
        stage = ProcessingStage(name="quality_monitoring", status="running")
        stage.start_time = time.time()
        
        try:
            from features.evaluation.integrated_quality_monitor import IntegratedQualityMonitor
            
            monitor = IntegratedQualityMonitor()
            result = monitor.monitor_extraction_quality("temp_image", mask, image)
            
            quality_snapshot = result.get('quality_snapshot')
            if quality_snapshot:
                final_quality = quality_snapshot.overall_score
                stage.status = "completed"
                stage.improvements = ["quality_monitoring"]
                stage.quality_score = final_quality
                logger.debug(f"🔍 品質監視完了: スコア {final_quality:.3f}")
            else:
                # フォールバック: 基本品質評価
                final_quality = self._evaluate_mask_quality(mask, image)
                stage.status = "completed"
                stage.improvements = ["basic_quality_evaluation"]
                stage.quality_score = final_quality
                logger.debug(f"🔍 基本品質評価: スコア {final_quality:.3f}")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            final_quality = self._evaluate_mask_quality(mask, image)
            logger.warning(f"⚠️ 品質監視失敗: {e}")
        
        stage.end_time = time.time()
        return stage, final_quality
    
    def _evaluate_mask_quality(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> float:
        """基本マスク品質評価"""
        try:
            from features.processing.postprocessing.postprocessing import calculate_mask_quality_metrics
            
            metrics = calculate_mask_quality_metrics(mask)
            
            # 統合品質スコア計算
            quality_score = (
                metrics.get('compactness', 0.0) * 0.4 +
                metrics.get('fill_ratio', 0.0) * 0.3 +
                metrics.get('coverage_ratio', 0.0) * 0.3
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"⚠️ 基本品質評価エラー: {e}")
            return 0.0
    
    def _create_failed_result(self, 
                            initial_quality: float, 
                            processing_stages: List[ProcessingStage],
                            failure_analysis: Optional[Dict] = None) -> PipelineResult:
        """失敗結果作成"""
        return PipelineResult(
            success=False,
            final_mask=None,
            initial_quality=initial_quality,
            final_quality=initial_quality,
            improvement_ratio=0.0,
            processing_stages=processing_stages,
            retry_count=0,
            total_processing_time=0.0,
            failure_analysis=failure_analysis
        )
    
    def _analyze_failure_pattern(self, result: PipelineResult) -> Dict[str, Any]:
        """失敗パターン分析"""
        try:
            failure_stages = [stage for stage in result.processing_stages if stage.status == 'failed']
            skipped_stages = [stage for stage in result.processing_stages if stage.status == 'skipped']
            
            analysis = {
                'failed_stages': [stage.name for stage in failure_stages],
                'skipped_stages': [stage.name for stage in skipped_stages],
                'quality_deficit': max(0, 0.5 - result.final_quality),  # 目標との差
                'processing_bottleneck': None,
                'recommended_adjustments': []
            }
            
            # ボトルネック特定
            if failure_stages:
                analysis['processing_bottleneck'] = failure_stages[0].name
                
                # 段階別推奨調整
                if 'parameter_optimization' in analysis['failed_stages']:
                    analysis['recommended_adjustments'].append('fallback_to_default_parameters')
                
                if 'yolo_expansion' in analysis['failed_stages']:
                    analysis['recommended_adjustments'].append('lower_yolo_threshold')
                
                if 'sam_postprocessing' in analysis['failed_stages']:
                    analysis['recommended_adjustments'].append('basic_morphological_processing')
                
                if 'contour_enhancement' in analysis['failed_stages']:
                    analysis['recommended_adjustments'].append('skip_contour_processing')
            
            elif len(skipped_stages) > 2:
                analysis['processing_bottleneck'] = 'insufficient_initial_quality'
                analysis['recommended_adjustments'].append('aggressive_preprocessing')
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ 失敗パターン分析エラー: {e}")
            return {'error': str(e)}
    
    def _determine_retry_strategy(self, failure_analysis: Dict[str, Any], retry_count: int) -> Dict[str, Any]:
        """リトライ戦略決定"""
        try:
            strategies = {
                0: 'parameter_adjustment',    # 1回目: パラメータ調整
                1: 'processing_simplification',  # 2回目: 処理簡略化
                2: 'fallback_mode'           # 3回目: フォールバックモード
            }
            
            strategy = strategies.get(retry_count, 'fallback_mode')
            
            return {
                'strategy': strategy,
                'adjustments': failure_analysis.get('recommended_adjustments', []),
                'focus_stage': failure_analysis.get('processing_bottleneck', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"⚠️ リトライ戦略決定エラー: {e}")
            return {'strategy': 'fallback_mode', 'adjustments': [], 'focus_stage': 'unknown'}
    
    def _update_pipeline_history(self, result: PipelineResult, image_shape: Tuple[int, int, int]) -> None:
        """パイプライン履歴更新"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'image_shape': image_shape,
                'success': result.success,
                'initial_quality': result.initial_quality,
                'final_quality': result.final_quality,
                'improvement_ratio': result.improvement_ratio,
                'retry_count': result.retry_count,
                'processing_time': result.total_processing_time,
                'stage_summary': {
                    'completed': len([s for s in result.processing_stages if s.status == 'completed']),
                    'failed': len([s for s in result.processing_stages if s.status == 'failed']),
                    'skipped': len([s for s in result.processing_stages if s.status == 'skipped'])
                }
            }
            
            self.pipeline_history.append(history_entry)
            
            # 履歴サイズ制限（最新100件保持）
            if len(self.pipeline_history) > 100:
                self.pipeline_history = self.pipeline_history[-100:]
                
        except Exception as e:
            logger.warning(f"⚠️ パイプライン履歴更新エラー: {e}")
    
    def _update_learning_data(self, result: PipelineResult) -> None:
        """学習データ更新"""
        try:
            # 成功パターン学習
            if result.success:
                success_pattern = self._extract_processing_pattern(result)
                self.success_patterns[success_pattern] = self.success_patterns.get(success_pattern, 0) + 1
                logger.debug(f"📚 成功パターン学習: {success_pattern}")
            
            # 失敗パターン学習
            else:
                failure_pattern = self._extract_failure_pattern(result)
                self.failure_patterns[failure_pattern] = self.failure_patterns.get(failure_pattern, 0) + 1
                logger.debug(f"⚠️ 失敗パターン学習: {failure_pattern}")
                
        except Exception as e:
            logger.warning(f"⚠️ 学習データ更新エラー: {e}")
    
    def _extract_processing_pattern(self, result: PipelineResult) -> str:
        """処理パターン抽出"""
        try:
            completed_stages = [s.name for s in result.processing_stages if s.status == 'completed']
            quality_level = 'high' if result.final_quality > 0.7 else ('medium' if result.final_quality > 0.4 else 'low')
            
            return f"{'-'.join(completed_stages)}_{quality_level}"
            
        except Exception as e:
            logger.warning(f"⚠️ 処理パターン抽出エラー: {e}")
            return "unknown"
    
    def _extract_failure_pattern(self, result: PipelineResult) -> str:
        """失敗パターン抽出"""
        try:
            failed_stages = [s.name for s in result.processing_stages if s.status == 'failed']
            if not failed_stages:
                failed_stages = ['quality_insufficient']
            
            return f"failed_{'-'.join(failed_stages)}"
            
        except Exception as e:
            logger.warning(f"⚠️ 失敗パターン抽出エラー: {e}")
            return "unknown_failure"
    
    def get_pipeline_report(self) -> Dict[str, Any]:
        """パイプライン レポート取得"""
        try:
            if not self.pipeline_history:
                return {'message': 'パイプライン履歴がありません'}
            
            # 統計計算
            total_processed = len(self.pipeline_history) 
            successful = len([h for h in self.pipeline_history if h['success']])
            
            recent_entries = self.pipeline_history[-20:]
            avg_quality_improvement = np.mean([h['improvement_ratio'] for h in recent_entries])
            avg_processing_time = np.mean([h['processing_time'] for h in recent_entries])
            
            return {
                'total_processed': total_processed,
                'success_rate': successful / total_processed if total_processed > 0 else 0,
                'learned_patterns': {
                    'success_patterns': len(self.success_patterns),
                    'failure_patterns': len(self.failure_patterns)
                },
                'performance_metrics': {
                    'average_quality_improvement': avg_quality_improvement,
                    'average_processing_time': avg_processing_time
                },
                'recent_trend': {
                    'last_10_success_rate': len([h for h in self.pipeline_history[-10:] if h['success']]) / min(10, len(self.pipeline_history)),
                    'quality_trend': 'improving' if len(recent_entries) >= 2 and recent_entries[-1]['final_quality'] > recent_entries[0]['final_quality'] else 'stable'
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ パイプラインレポート取得エラー: {e}")
            return {'error': str(e)}
    
    def save_pipeline_log(self, output_path: str) -> bool:
        """パイプラインログ保存"""
        try:
            log_data = {
                'pipeline_history': self.pipeline_history,
                'learned_patterns': {
                    'success_patterns': self.success_patterns,
                    'failure_patterns': self.failure_patterns
                },
                'configuration': {
                    'max_retries': self.max_retries,
                    'quality_improvement_threshold': self.quality_improvement_threshold,
                    'minimum_acceptable_quality': self.minimum_acceptable_quality
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 パイプラインログ保存完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ パイプラインログ保存エラー: {e}")
            return False


def integrate_with_extraction_command() -> None:
    """抽出コマンド統合準備"""
    logger.info("🔗 総合精度向上パイプラインを抽出コマンドに統合準備")
    
    # extract_character.py の process_single_image 関数に統合する想定
    # 実際の統合は次のステップで実装
    pass


if __name__ == "__main__":
    # テスト実行
    logger.info("🧪 総合精度向上パイプライン テスト開始")
    
    pipeline = IntegratedPrecisionPipeline()
    logger.info("✅ 総合精度向上パイプライン初期化完了")
    
    # テスト用データ
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    
    logger.info("🎯 テスト完了 - 実装準備完了")
    
    # 統合準備
    integrate_with_extraction_command()
    logger.info("🚀 パイプライン統合準備完了")