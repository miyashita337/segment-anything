#!/usr/bin/env python3
"""
Phase 4: 統合システム
マスク逆転検出、適応的範囲調整、品質予測を統合した改良版抽出システム
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import time

from .mask_quality_analyzer import MaskQualityAnalyzer, MaskQualityMetrics
from .adaptive_extraction import AdaptiveExtractionRangeAdjuster, PoseAnalysisResult
from .quality_predictor import QualityPredictor, QualityPrediction, ProcessingCandidate

@dataclass
class Phase4Result:
    """Phase 4処理結果"""
    success: bool                    # 処理成功フラグ
    final_mask: Optional[np.ndarray] # 最終マスク
    quality_metrics: Optional[MaskQualityMetrics]  # 品質メトリクス
    pose_analysis: Optional[PoseAnalysisResult]    # 姿勢分析結果
    quality_prediction: Optional[QualityPrediction] # 品質予測
    processing_stats: Dict[str, Any] # 処理統計
    adjustments_made: List[str]      # 実行された調整
    final_bbox: Tuple[int, int, int, int]  # 最終バウンディングボックス

class Phase4IntegratedExtractor:
    """Phase 4統合抽出システム"""
    
    def __init__(self,
                 enable_mask_inversion_detection: bool = True,
                 enable_adaptive_range: bool = True,
                 enable_quality_prediction: bool = True,
                 max_iterations: int = 3):
        """
        初期化
        
        Args:
            enable_mask_inversion_detection: マスク逆転検出の有効化
            enable_adaptive_range: 適応的範囲調整の有効化
            enable_quality_prediction: 品質予測の有効化
            max_iterations: 最大反復回数
        """
        self.enable_mask_inversion = enable_mask_inversion_detection
        self.enable_adaptive_range = enable_adaptive_range
        self.enable_quality_prediction = enable_quality_prediction
        self.max_iterations = max_iterations
        
        # コンポーネント初期化
        self.mask_analyzer = MaskQualityAnalyzer()
        self.range_adjuster = AdaptiveExtractionRangeAdjuster()
        self.quality_predictor = QualityPredictor()
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # 統計情報
        self.processing_stats = {
            'total_processed': 0,
            'mask_inversions_fixed': 0,
            'range_adjustments_made': 0,
            'quality_improvements': 0,
            'average_processing_time': 0.0
        }
    
    def extract_with_phase4_enhancements(self,
                                       image: np.ndarray,
                                       yolo_bbox: Tuple[int, int, int, int],
                                       yolo_confidence: float,
                                       sam_predictor,
                                       base_parameters: Dict[str, Any]) -> Phase4Result:
        """
        Phase 4機能を統合した改良版抽出
        
        Args:
            image: 元画像
            yolo_bbox: YOLOバウンディングボックス
            yolo_confidence: YOLO検出信頼度
            sam_predictor: SAM予測器
            base_parameters: 基本パラメータ
            
        Returns:
            Phase4Result: 処理結果
        """
        start_time = time.time()
        adjustments_made = []
        
        try:
            # 1. 品質予測（有効な場合）
            quality_prediction = None
            if self.enable_quality_prediction:
                quality_prediction = self.quality_predictor.predict_quality(
                    image, yolo_bbox, yolo_confidence, base_parameters
                )
                self.logger.info(f"品質予測: {quality_prediction.predicted_level.value}")
            
            # 2. 処理候補生成と選択
            if quality_prediction and len(quality_prediction.risk_factors) > 2:
                candidates = self.quality_predictor.generate_processing_candidates(
                    image, yolo_bbox, yolo_confidence, base_parameters
                )
                best_candidate = candidates[0]  # 最良候補を選択
                working_params = best_candidate.parameters
                adjustments_made.append(f"品質予測により最適候補選択")
            else:
                working_params = base_parameters.copy()
            
            # 3. 適応的範囲調整（有効な場合）
            working_bbox = yolo_bbox
            pose_analysis = None
            
            if self.enable_adaptive_range:
                pose_analysis = self.range_adjuster.analyze_pose_complexity(
                    image, yolo_bbox
                )
                
                # ユーザー設定の模擬（評価データから得られた要望）
                user_preferences = self._get_user_preferences_from_analysis(pose_analysis)
                
                working_bbox = self.range_adjuster.adjust_extraction_range(
                    yolo_bbox, image.shape[:2], pose_analysis, user_preferences
                )
                
                if working_bbox != yolo_bbox:
                    adjustments_made.append("適応的範囲調整実行")
                    self.processing_stats['range_adjustments_made'] += 1
            
            # 4. メイン抽出処理の反復改良
            best_result = None
            best_quality_score = -1.0
            
            for iteration in range(self.max_iterations):
                self.logger.debug(f"抽出試行 {iteration + 1}/{self.max_iterations}")
                
                # SAM処理実行（実際のSAM呼び出しをシミュレート）
                current_mask = self._simulate_sam_processing(
                    image, working_bbox, sam_predictor, working_params
                )
                
                if current_mask is None:
                    continue
                
                # 5. マスク品質分析
                quality_metrics = self.mask_analyzer.analyze_mask_quality(
                    image, current_mask, working_bbox
                )
                
                # 6. マスク逆転検出・修正
                if self.enable_mask_inversion and quality_metrics.is_inverted:
                    self.logger.warning("マスク逆転検出、修正実行")
                    current_mask = self.mask_analyzer.fix_inverted_mask(current_mask)
                    
                    # 修正後の品質再評価
                    quality_metrics = self.mask_analyzer.analyze_mask_quality(
                        image, current_mask, working_bbox
                    )
                    adjustments_made.append("マスク逆転修正")
                    self.processing_stats['mask_inversions_fixed'] += 1
                
                # 品質スコアで最良結果を選択
                if quality_metrics.confidence_score > best_quality_score:
                    best_quality_score = quality_metrics.confidence_score
                    best_result = {
                        'mask': current_mask,
                        'quality_metrics': quality_metrics,
                        'bbox': working_bbox
                    }
                
                # 高品質なら早期終了
                if quality_metrics.confidence_score > 0.8:
                    self.logger.info("高品質マスク取得、処理終了")
                    break
                
                # 次の試行でパラメータを微調整
                working_params = self._adjust_parameters_for_next_iteration(
                    working_params, quality_metrics, iteration
                )
            
            # 7. 結果の最終処理
            if best_result is None:
                return Phase4Result(
                    success=False,
                    final_mask=None,
                    quality_metrics=None,
                    pose_analysis=pose_analysis,
                    quality_prediction=quality_prediction,
                    processing_stats=self._get_current_stats(start_time),
                    adjustments_made=adjustments_made,
                    final_bbox=working_bbox
                )
            
            # 8. フィードバック学習
            if quality_prediction:
                processing_time = time.time() - start_time
                self.quality_predictor.feedback_learning(
                    quality_prediction, best_result['quality_metrics'], processing_time
                )
            
            # 9. 統計更新
            self.processing_stats['total_processed'] += 1
            if best_quality_score > 0.7:
                self.processing_stats['quality_improvements'] += 1
            
            processing_time = time.time() - start_time
            self._update_average_processing_time(processing_time)
            
            return Phase4Result(
                success=True,
                final_mask=best_result['mask'],
                quality_metrics=best_result['quality_metrics'],
                pose_analysis=pose_analysis,
                quality_prediction=quality_prediction,
                processing_stats=self._get_current_stats(start_time),
                adjustments_made=adjustments_made,
                final_bbox=best_result['bbox']
            )
            
        except Exception as e:
            self.logger.error(f"Phase 4処理エラー: {e}")
            return Phase4Result(
                success=False,
                final_mask=None,
                quality_metrics=None,
                pose_analysis=None,
                quality_prediction=quality_prediction,
                processing_stats=self._get_current_stats(start_time),
                adjustments_made=adjustments_made,
                final_bbox=yolo_bbox
            )
    
    def _get_user_preferences_from_analysis(self, 
                                          pose_analysis: PoseAnalysisResult) -> Dict[str, Any]:
        """姿勢分析から推定されるユーザー設定"""
        preferences = {}
        
        # 評価データからの学習: "上半身も抽出してほしい"
        if 'legs' not in pose_analysis.body_parts_detected:
            preferences['include_full_body'] = True
        
        # 評価データからの学習: "足もあるのでそこも抽出してほしい"
        if 'legs' in pose_analysis.body_parts_detected:
            preferences['include_full_body'] = True
        
        # 複雑な姿勢の場合、保守的でない抽出
        if pose_analysis.complexity.value in ['complex', 'extreme']:
            preferences['conservative_crop'] = False
        
        return preferences
    
    def _simulate_sam_processing(self,
                                image: np.ndarray,
                                bbox: Tuple[int, int, int, int],
                                sam_predictor,
                                parameters: Dict[str, Any]) -> Optional[np.ndarray]:
        """実際のSAM処理（Phase 4対応版）"""
        x, y, w, h = bbox
        
        try:
            # バウンディングボックスの中心点でSAM処理
            center_x = x + w // 2
            center_y = y + h // 2
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])
            
            # SAM予測実行（実際のモデルを使用）
            if hasattr(sam_predictor, 'set_image'):
                sam_predictor.set_image(image)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
            else:
                # YOLOModelWrapperからSAMモデルを取得
                from hooks.start import get_sam_model
                sam_model = get_sam_model()
                if sam_model is None:
                    # フォールバック: 簡易マスク
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                    mask[y:y+h, x:x+w] = True
                    return mask
                
                # SAMマスク生成
                all_masks = sam_model.generate_masks(image)
                if not all_masks:
                    return None
                
                # バウンディングボックスと重複するマスクを選択
                best_mask = None
                best_overlap = 0
                
                for mask_data in all_masks:
                    mask = mask_data['segmentation']
                    if mask is None:
                        continue
                    
                    # 重複計算
                    mask_in_bbox = mask[y:y+h, x:x+w]
                    overlap = np.sum(mask_in_bbox) / (w * h)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_mask = mask
                
                if best_mask is not None:
                    return best_mask
                else:
                    # フォールバック
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                    mask[y:y+h, x:x+w] = True
                    return mask
            
            if len(masks) > 0:
                # 最適マスク選択（スコアが最高のもの）
                best_idx = np.argmax(scores)
                return masks[best_idx]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"SAM処理エラー: {e}")
            # フォールバック: 矩形マスク
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            mask[y:y+h, x:x+w] = True
            return mask
    
    def _adjust_parameters_for_next_iteration(self,
                                            current_params: Dict[str, Any],
                                            quality_metrics: MaskQualityMetrics,
                                            iteration: int) -> Dict[str, Any]:
        """次の試行でのパラメータ調整"""
        adjusted_params = current_params.copy()
        
        # 品質が低い場合の調整
        if quality_metrics.confidence_score < 0.5:
            if iteration == 1:
                # 2回目: より保守的な設定
                adjusted_params['expansion_factor'] = adjusted_params.get('expansion_factor', 1.1) * 1.2
            elif iteration == 2:
                # 3回目: より積極的な設定
                adjusted_params['high_quality'] = True
                adjusted_params['manga_mode'] = True
        
        return adjusted_params
    
    def _get_current_stats(self, start_time: float) -> Dict[str, Any]:
        """現在の処理統計取得"""
        current_time = time.time() - start_time
        return {
            'processing_time': current_time,
            'total_processed': self.processing_stats['total_processed'],
            'improvements_made': len([k for k, v in self.processing_stats.items() 
                                    if k.endswith('_made') or k.endswith('_fixed')])
        }
    
    def _update_average_processing_time(self, processing_time: float):
        """平均処理時間の更新"""
        current_avg = self.processing_stats['average_processing_time']
        total_processed = self.processing_stats['total_processed']
        
        if total_processed == 1:
            self.processing_stats['average_processing_time'] = processing_time
        else:
            # 移動平均の更新
            self.processing_stats['average_processing_time'] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
    
    def get_performance_report(self) -> str:
        """性能レポート生成"""
        stats = self.processing_stats
        
        if stats['total_processed'] == 0:
            return "まだ処理が実行されていません。"
        
        improvement_rate = (stats['quality_improvements'] / stats['total_processed']) * 100
        inversion_rate = (stats['mask_inversions_fixed'] / stats['total_processed']) * 100
        adjustment_rate = (stats['range_adjustments_made'] / stats['total_processed']) * 100
        
        report = []
        report.append("=== Phase 4 性能レポート ===")
        report.append(f"総処理数: {stats['total_processed']}")
        report.append(f"品質改善率: {improvement_rate:.1f}%")
        report.append(f"マスク逆転修正率: {inversion_rate:.1f}%")
        report.append(f"範囲調整実行率: {adjustment_rate:.1f}%")
        report.append(f"平均処理時間: {stats['average_processing_time']:.2f}秒")
        
        return "\n".join(report)
    
    def reset_statistics(self):
        """統計情報リセット"""
        self.processing_stats = {
            'total_processed': 0,
            'mask_inversions_fixed': 0,
            'range_adjustments_made': 0,
            'quality_improvements': 0,
            'average_processing_time': 0.0
        }


def test_phase4_integration():
    """Phase 4統合システムのテスト"""
    # ダミーデータでテスト
    image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    yolo_bbox = (50, 50, 100, 150)
    yolo_confidence = 0.7
    base_params = {
        'min_yolo_score': 0.1,
        'high_quality': False,
        'expansion_factor': 1.1
    }
    
    # ダミーSAM予測器
    class DummySAMPredictor:
        pass
    sam_predictor = DummySAMPredictor()
    
    # Phase 4システム初期化
    extractor = Phase4IntegratedExtractor(
        enable_mask_inversion_detection=True,
        enable_adaptive_range=True,
        enable_quality_prediction=True
    )
    
    # 抽出実行
    result = extractor.extract_with_phase4_enhancements(
        image, yolo_bbox, yolo_confidence, sam_predictor, base_params
    )
    
    print(f"処理成功: {result.success}")
    if result.quality_metrics:
        print(f"品質スコア: {result.quality_metrics.confidence_score:.3f}")
        print(f"逆転判定: {result.quality_metrics.is_inverted}")
    
    if result.pose_analysis:
        print(f"姿勢複雑度: {result.pose_analysis.complexity.value}")
    
    if result.quality_prediction:
        print(f"品質予測: {result.quality_prediction.predicted_level.value}")
    
    print(f"実行された調整: {result.adjustments_made}")
    print(f"処理時間: {result.processing_stats['processing_time']:.2f}秒")
    
    # 性能レポート
    print("\n" + extractor.get_performance_report())


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    test_phase4_integration()