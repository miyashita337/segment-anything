#!/usr/bin/env python3
"""
Phase 4: 品質予測・フィードバックループシステム
抽出前の品質予測と動的パラメータ調整機能
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from .mask_quality_analyzer import MaskQualityAnalyzer, MaskQualityMetrics
from .adaptive_extraction import AdaptiveExtractionRangeAdjuster, PoseAnalysisResult

class QualityLevel(Enum):
    """品質レベル"""
    EXCELLENT = "excellent"    # A評価相当
    GOOD = "good"             # B評価相当  
    FAIR = "fair"             # C評価相当
    POOR = "poor"             # D評価相当
    BAD = "bad"               # E評価相当
    FAILED = "failed"         # F評価相当

@dataclass
class QualityPrediction:
    """品質予測結果"""
    predicted_level: QualityLevel    # 予測品質レベル
    confidence: float               # 予測信頼度
    risk_factors: List[str]         # リスク要因
    recommended_actions: List[str]   # 推奨アクション
    parameter_adjustments: Dict[str, Any]  # パラメータ調整案

@dataclass
class ProcessingCandidate:
    """処理候補"""
    parameters: Dict[str, Any]      # 処理パラメータ
    predicted_quality: QualityLevel # 予測品質
    confidence: float              # 信頼度
    processing_cost: float         # 処理コスト

class QualityPredictor:
    """品質予測・フィードバックループクラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self.mask_analyzer = MaskQualityAnalyzer()
        self.range_adjuster = AdaptiveExtractionRangeAdjuster()
        
        # 過去の処理結果から学習したパターン（実際には機械学習モデルを使用）
        self.learned_patterns = self._initialize_learned_patterns()
    
    def _initialize_learned_patterns(self) -> Dict[str, Any]:
        """学習済みパターンの初期化"""
        return {
            # 失敗パターン
            'mask_inversion_indicators': {
                'low_complexity_ratio': 0.7,      # 複雑度比が低い
                'high_background_edges': 0.15,    # 背景エッジが多い
                'low_saturation': 0.3              # 彩度が低い
            },
            
            # 範囲不適切パターン
            'range_issues': {
                'extreme_aspect_ratio': [0.3, 3.0],  # 極端なアスペクト比
                'small_yolo_confidence': 0.3,         # YOLO信頼度が低い
                'edge_clipping': 0.8                  # エッジの切れ具合
            },
            
            # 成功パターン
            'success_indicators': {
                'ideal_complexity_ratio': [1.2, 2.0],  # 理想的複雑度比
                'good_edge_consistency': 0.6,          # 良いエッジ一貫性
                'balanced_aspect_ratio': [0.7, 1.5]    # バランスの良いアスペクト比
            }
        }
    
    def predict_quality(self,
                       image: np.ndarray,
                       yolo_bbox: Tuple[int, int, int, int],
                       yolo_confidence: float,
                       current_parameters: Dict[str, Any]) -> QualityPrediction:
        """
        抽出前の品質予測
        
        Args:
            image: 元画像
            yolo_bbox: YOLOバウンディングボックス
            yolo_confidence: YOLO検出信頼度
            current_parameters: 現在の処理パラメータ
            
        Returns:
            QualityPrediction: 品質予測結果
        """
        x, y, w, h = yolo_bbox
        roi = image[y:y+h, x:x+w]
        
        # 1. 基本的な画像特徴分析
        image_features = self._analyze_image_features(roi, yolo_bbox, image.shape[:2])
        
        # 2. リスク要因の特定
        risk_factors = self._identify_risk_factors(image_features, yolo_confidence, current_parameters)
        
        # 3. 品質レベル予測
        predicted_level = self._predict_quality_level(image_features, risk_factors, yolo_confidence)
        
        # 4. 予測信頼度計算
        confidence = self._calculate_prediction_confidence(image_features, risk_factors)
        
        # 5. 推奨アクション生成
        recommended_actions = self._generate_recommendations(risk_factors, predicted_level)
        
        # 6. パラメータ調整案生成
        parameter_adjustments = self._suggest_parameter_adjustments(risk_factors, current_parameters)
        
        prediction = QualityPrediction(
            predicted_level=predicted_level,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_actions=recommended_actions,
            parameter_adjustments=parameter_adjustments
        )
        
        self.logger.info(f"品質予測: {predicted_level.value} (信頼度={confidence:.3f})")
        return prediction
    
    def _analyze_image_features(self,
                              roi: np.ndarray,
                              bbox: Tuple[int, int, int, int],
                              image_shape: Tuple[int, int]) -> Dict[str, float]:
        """画像特徴分析"""
        x, y, w, h = bbox
        img_h, img_w = image_shape
        
        features = {}
        
        # 基本的な幾何学特徴
        features['aspect_ratio'] = w / h if h > 0 else 1.0
        features['relative_size'] = (w * h) / (img_w * img_h)
        features['position_x'] = x / img_w
        features['position_y'] = y / img_h
        
        # 色彩特徴
        features['brightness'] = np.mean(roi)
        features['contrast'] = np.std(roi)
        
        # HSV特徴
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        features['saturation'] = np.mean(hsv[:, :, 1])
        features['hue_variance'] = np.std(hsv[:, :, 0])
        
        # エッジ特徴
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (w * h)
        
        # テクスチャ特徴
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        
        # 境界近接度（画像端に近いかどうか）
        border_distance = min(x, y, img_w - (x + w), img_h - (y + h))
        features['border_proximity'] = 1.0 - (border_distance / min(img_w, img_h))
        
        return features
    
    def _identify_risk_factors(self,
                             features: Dict[str, float],
                             yolo_confidence: float,
                             parameters: Dict[str, Any]) -> List[str]:
        """リスク要因の特定"""
        risk_factors = []
        patterns = self.learned_patterns
        
        # YOLO信頼度チェック
        if yolo_confidence < 0.3:
            risk_factors.append("低YOLO信頼度")
        
        # アスペクト比チェック
        aspect_ratio = features['aspect_ratio']
        extreme_ratios = patterns['range_issues']['extreme_aspect_ratio']
        if aspect_ratio < extreme_ratios[0] or aspect_ratio > extreme_ratios[1]:
            risk_factors.append("極端なアスペクト比")
        
        # 彩度チェック（マスク逆転の指標）
        if features['saturation'] < patterns['mask_inversion_indicators']['low_saturation'] * 255:
            risk_factors.append("低彩度（逆転リスク）")
        
        # エッジ密度チェック
        if features['edge_density'] > patterns['mask_inversion_indicators']['high_background_edges']:
            risk_factors.append("高エッジ密度（複雑背景）")
        
        # 境界近接チェック
        if features['border_proximity'] > 0.8:
            risk_factors.append("画像境界近接")
        
        # 小さすぎる検出領域
        if features['relative_size'] < 0.01:
            risk_factors.append("極小検出領域")
        
        # コントラスト不足
        if features['contrast'] < 30:
            risk_factors.append("低コントラスト")
        
        # 暗すぎる画像
        if features['brightness'] < 50:
            risk_factors.append("低輝度")
        
        return risk_factors
    
    def _predict_quality_level(self,
                             features: Dict[str, float],
                             risk_factors: List[str],
                             yolo_confidence: float) -> QualityLevel:
        """品質レベル予測"""
        
        # リスクスコア計算
        risk_score = len(risk_factors) / 8.0  # 最大8種類のリスク
        
        # 成功指標チェック
        success_score = 0.0
        patterns = self.learned_patterns['success_indicators']
        
        # 複雑度比の推定（実際のマスクがないため簡易推定）
        estimated_complexity = features['texture_variance'] / features['contrast'] if features['contrast'] > 0 else 1.0
        if patterns['ideal_complexity_ratio'][0] <= estimated_complexity <= patterns['ideal_complexity_ratio'][1]:
            success_score += 0.3
        
        # アスペクト比
        aspect_ratio = features['aspect_ratio']
        if patterns['balanced_aspect_ratio'][0] <= aspect_ratio <= patterns['balanced_aspect_ratio'][1]:
            success_score += 0.3
        
        # YOLO信頼度
        success_score += min(0.4, yolo_confidence)
        
        # 総合スコア
        quality_score = success_score - risk_score
        
        # 品質レベル判定
        if quality_score >= 0.8:
            return QualityLevel.EXCELLENT
        elif quality_score >= 0.6:
            return QualityLevel.GOOD
        elif quality_score >= 0.4:
            return QualityLevel.FAIR
        elif quality_score >= 0.2:
            return QualityLevel.POOR
        elif quality_score >= 0.0:
            return QualityLevel.BAD
        else:
            return QualityLevel.FAILED
    
    def _calculate_prediction_confidence(self,
                                       features: Dict[str, float],
                                       risk_factors: List[str]) -> float:
        """予測信頼度計算"""
        base_confidence = 0.7
        
        # 特徴の明確さによる調整
        if features['contrast'] > 50:
            base_confidence += 0.1
        if features['edge_density'] > 0.05:
            base_confidence += 0.1
        if features['relative_size'] > 0.1:
            base_confidence += 0.1
        
        # リスク要因による信頼度低下
        confidence_penalty = len(risk_factors) * 0.05
        
        final_confidence = max(0.3, min(1.0, base_confidence - confidence_penalty))
        return final_confidence
    
    def _generate_recommendations(self,
                                risk_factors: List[str],
                                predicted_level: QualityLevel) -> List[str]:
        """推奨アクション生成"""
        recommendations = []
        
        if "低YOLO信頼度" in risk_factors:
            recommendations.append("YOLO閾値を下げる")
            recommendations.append("複数YOLO候補を試行")
        
        if "極端なアスペクト比" in risk_factors:
            recommendations.append("適応的範囲調整を使用")
            recommendations.append("手動境界調整を検討")
        
        if "低彩度（逆転リスク）" in risk_factors:
            recommendations.append("マスク逆転検出を有効化")
            recommendations.append("複数マスク候補を生成")
        
        if "高エッジ密度（複雑背景）" in risk_factors:
            recommendations.append("エフェクト線除去を適用")
            recommendations.append("SAM高品質モードを使用")
        
        if "画像境界近接" in risk_factors:
            recommendations.append("拡張係数を増加")
            recommendations.append("元画像の確認")
        
        if predicted_level in [QualityLevel.POOR, QualityLevel.BAD, QualityLevel.FAILED]:
            recommendations.append("Phase 3インタラクティブ機能を使用")
            recommendations.append("手動シードポイント指定を検討")
        
        return recommendations
    
    def _suggest_parameter_adjustments(self,
                                     risk_factors: List[str],
                                     current_params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータ調整案生成"""
        adjustments = {}
        
        if "低YOLO信頼度" in risk_factors:
            adjustments['min_yolo_score'] = max(0.02, current_params.get('min_yolo_score', 0.1) * 0.5)
        
        if "低彩度（逆転リスク）" in risk_factors:
            adjustments['enable_mask_inversion_detection'] = True
            adjustments['high_quality'] = True
        
        if "高エッジ密度（複雑背景）" in risk_factors:
            adjustments['effect_removal'] = True
            adjustments['manga_mode'] = True
        
        if "極端なアスペクト比" in risk_factors:
            adjustments['adaptive_range'] = True
            adjustments['expansion_factor'] = 1.5
        
        if "画像境界近接" in risk_factors:
            adjustments['expansion_factor'] = max(1.3, current_params.get('expansion_factor', 1.1))
        
        return adjustments
    
    def generate_processing_candidates(self,
                                     image: np.ndarray,
                                     yolo_bbox: Tuple[int, int, int, int],
                                     yolo_confidence: float,
                                     base_parameters: Dict[str, Any]) -> List[ProcessingCandidate]:
        """複数の処理候補を生成"""
        candidates = []
        
        # 1. 基本候補
        base_prediction = self.predict_quality(image, yolo_bbox, yolo_confidence, base_parameters)
        candidates.append(ProcessingCandidate(
            parameters=base_parameters.copy(),
            predicted_quality=base_prediction.predicted_level,
            confidence=base_prediction.confidence,
            processing_cost=1.0
        ))
        
        # 2. 推奨調整候補
        if base_prediction.parameter_adjustments:
            adjusted_params = base_parameters.copy()
            adjusted_params.update(base_prediction.parameter_adjustments)
            
            adjusted_prediction = self.predict_quality(image, yolo_bbox, yolo_confidence, adjusted_params)
            candidates.append(ProcessingCandidate(
                parameters=adjusted_params,
                predicted_quality=adjusted_prediction.predicted_level,
                confidence=adjusted_prediction.confidence,
                processing_cost=1.5  # 高機能のためコスト増
            ))
        
        # 3. 高品質候補（リスクが高い場合）
        if len(base_prediction.risk_factors) > 3:
            high_quality_params = base_parameters.copy()
            high_quality_params.update({
                'high_quality': True,
                'enable_mask_inversion_detection': True,
                'adaptive_range': True,
                'manga_mode': True,
                'effect_removal': True
            })
            
            hq_prediction = self.predict_quality(image, yolo_bbox, yolo_confidence, high_quality_params)
            candidates.append(ProcessingCandidate(
                parameters=high_quality_params,
                predicted_quality=hq_prediction.predicted_level,
                confidence=hq_prediction.confidence,
                processing_cost=2.0  # 最高コスト
            ))
        
        # 品質スコアでソート
        candidates.sort(key=lambda c: (c.predicted_quality.value, c.confidence), reverse=True)
        return candidates
    
    def feedback_learning(self,
                         prediction: QualityPrediction,
                         actual_metrics: MaskQualityMetrics,
                         processing_time: float):
        """フィードバック学習（簡易版）"""
        # 実際の実装では機械学習モデルを更新
        # ここでは簡易的な統計更新
        
        prediction_accuracy = self._calculate_prediction_accuracy(prediction, actual_metrics)
        
        self.logger.info(f"予測精度: {prediction_accuracy:.3f}, 処理時間: {processing_time:.2f}s")
        
        # 学習パターンの更新（簡易版）
        if prediction_accuracy < 0.7:
            self.logger.warning("予測精度が低下、パターン見直しが必要")
    
    def _calculate_prediction_accuracy(self,
                                     prediction: QualityPrediction,
                                     actual: MaskQualityMetrics) -> float:
        """予測精度計算"""
        # 簡易的な精度計算
        predicted_score = {
            QualityLevel.EXCELLENT: 1.0,
            QualityLevel.GOOD: 0.8,
            QualityLevel.FAIR: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.BAD: 0.2,
            QualityLevel.FAILED: 0.0
        }[prediction.predicted_level]
        
        actual_score = actual.confidence_score
        
        accuracy = 1.0 - abs(predicted_score - actual_score)
        return max(0.0, accuracy)


def test_quality_predictor():
    """品質予測システムのテスト"""
    # ダミーデータでテスト
    image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    yolo_bbox = (50, 50, 100, 150)
    yolo_confidence = 0.7
    base_params = {'min_yolo_score': 0.1, 'high_quality': False}
    
    predictor = QualityPredictor()
    
    # 品質予測
    prediction = predictor.predict_quality(image, yolo_bbox, yolo_confidence, base_params)
    
    print(f"予測品質: {prediction.predicted_level.value}")
    print(f"信頼度: {prediction.confidence:.3f}")
    print(f"リスク要因: {prediction.risk_factors}")
    print(f"推奨アクション: {prediction.recommended_actions}")
    print(f"パラメータ調整: {prediction.parameter_adjustments}")
    
    # 処理候補生成
    candidates = predictor.generate_processing_candidates(image, yolo_bbox, yolo_confidence, base_params)
    
    print("\n処理候補:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. 品質={candidate.predicted_quality.value}, 信頼度={candidate.confidence:.3f}, コスト={candidate.processing_cost:.1f}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    test_quality_predictor()