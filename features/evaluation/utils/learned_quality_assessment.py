#!/usr/bin/env python3
"""
学習した品質評価システム
137レコードの人間評価データに基づく適応的品質予測・手法選択

統合されるsegment-anythingパイプラインでの使用:
- 画像特性の自動分析
- 最適手法の選択
- 品質スコアの予測
- リアルタイム学習更新
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# 基本的な画像処理（SAMプロジェクト内での利用を想定）
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Image analysis will be limited.")


@dataclass
class QualityPrediction:
    """品質予測結果"""
    predicted_quality: float
    confidence: float
    recommended_method: str
    fallback_method: str
    reasoning: str
    image_characteristics: Dict[str, Any]


@dataclass
class ImageCharacteristics:
    """画像特性の分析結果"""
    has_complex_pose: bool
    has_full_body: bool
    has_multiple_characters: bool
    is_face_focus: bool
    aspect_ratio: float
    estimated_difficulty: float
    manga_style_score: float
    has_screentone_issues: bool = False
    has_mosaic_issues: bool = False
    has_boundary_complexity: bool = False


class LearnedQualityAssessment:
    """学習した品質評価システム"""
    
    def __init__(self, 
                 analysis_data_path: str = "/mnt/c/AItools/image_evaluation_system/analysis/quality_analysis_report.json",
                 recommendations_path: str = "/mnt/c/AItools/image_evaluation_system/analysis/method_recommendations.json"):
        self.analysis_data_path = analysis_data_path
        self.recommendations_path = recommendations_path
        
        # 学習データ
        self.method_stats = {}
        self.method_recommendations = {}
        self.quality_baseline = {}
        
        # 設定
        self.confidence_threshold = 0.7
        self.fallback_threshold = 0.5
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._load_learned_data()
    
    def _load_learned_data(self):
        """学習済みデータの読み込み"""
        try:
            # 品質分析データ読み込み
            if os.path.exists(self.analysis_data_path):
                with open(self.analysis_data_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                self.method_stats = analysis_data['method_performance']['method_performance']
                self.problem_rates = analysis_data['method_performance']['method_problem_rates']
                
                self.logger.info(f"✅ 品質分析データを読み込み: {len(self.method_stats)}手法")
            
            # 推奨データ読み込み
            if os.path.exists(self.recommendations_path):
                with open(self.recommendations_path, 'r', encoding='utf-8') as f:
                    self.method_recommendations = json.load(f)
                
                self.logger.info(f"✅ 推奨データを読み込み: {len(self.method_recommendations['method_strengths'])}手法")
            
            # ベースライン品質設定
            self._setup_quality_baseline()
            
        except Exception as e:
            self.logger.error(f"❌ 学習データ読み込み失敗: {e}")
            self._setup_fallback_data()
    
    def _setup_quality_baseline(self):
        """品質ベースラインの設定"""
        if self.method_stats:
            # 実際の学習データから設定
            self.quality_baseline = {
                'excellent': 4.0,
                'good': 3.0,
                'acceptable': 2.0,
                'poor': 1.0,
                'failed': 0.0
            }
            
            # 手法別期待品質
            self.method_expected_quality = {
                method: stats['mean'] 
                for method, stats in self.method_stats.items()
            }
        else:
            self._setup_fallback_data()
    
    def _setup_fallback_data(self):
        """フォールバック用の基本データ設定（最新の評価データに基づく）"""
        self.logger.warning("フォールバックデータを使用します（最新評価データベース）")
        
        # 2025-07-16更新: 281レコードの実際の評価データに基づく
        self.method_expected_quality = {
            'size_priority': 2.05,        # 最高評価手法
            'clean_version': 2.00,        # 2番目
            'balanced': 1.96,             # 安定した標準手法
            'v043_improved': 1.96,        # 改良版
            'reference_standard': 1.71,   # 基準版
            'confidence_priority': 1.28,  # 実際は期待より低評価
            'fullbody_priority': 2.0,     # 推定値
            'central_priority': 1.8       # 推定値
        }
        
        self.quality_baseline = {
            'excellent': 4.0,
            'good': 3.0,
            'acceptable': 2.0,
            'poor': 1.0,
            'failed': 0.0
        }
    
    def analyze_image_characteristics(self, image_path: str, 
                                    yolo_detections: Optional[List] = None,
                                    mask_candidates: Optional[List] = None) -> ImageCharacteristics:
        """画像特性の分析"""
        if not CV2_AVAILABLE:
            # OpenCVが利用できない場合のフォールバック
            return ImageCharacteristics(
                has_complex_pose=False,
                has_full_body=True,
                has_multiple_characters=False,
                is_face_focus=False,
                aspect_ratio=1.0,
                estimated_difficulty=0.5,
                manga_style_score=0.8,
                has_screentone_issues=False,
                has_mosaic_issues=False,
                has_boundary_complexity=False
            )
        
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"画像の読み込みに失敗: {image_path}")
            
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # YOLO検出結果からの特性分析
            has_multiple_characters = False
            has_full_body = False
            has_complex_pose = False
            
            if yolo_detections:
                # 複数キャラクター判定
                high_confidence_detections = [d for d in yolo_detections if d.get('confidence', 0) > 0.7]
                has_multiple_characters = len(high_confidence_detections) > 1
                
                # 全身判定（検出ボックスの高さから推定）
                for detection in high_confidence_detections:
                    bbox_height = detection.get('bbox', [0, 0, 0, 100])[3]
                    if bbox_height / height > 0.6:  # 画像の60%以上の高さ
                        has_full_body = True
                        break
                
                # 複雑姿勢判定（アスペクト比と検出スコアから推定）
                for detection in high_confidence_detections:
                    bbox = detection.get('bbox', [0, 0, 100, 100])
                    bbox_aspect = bbox[2] / bbox[3]  # width/height
                    confidence = detection.get('confidence', 0)
                    
                    # 横長のボックスかつ中程度の信頼度 = 複雑姿勢の可能性
                    if bbox_aspect > 1.5 and 0.3 < confidence < 0.8:
                        has_complex_pose = True
                        break
            
            # 顔フォーカス判定（上半分の領域重視）
            is_face_focus = aspect_ratio > 0.7 and aspect_ratio < 1.3  # 正方形に近い
            
            # 困難度推定
            difficulty_factors = 0
            if has_complex_pose:
                difficulty_factors += 0.3
            if has_multiple_characters:
                difficulty_factors += 0.2
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:  # 極端なアスペクト比
                difficulty_factors += 0.2
            
            estimated_difficulty = min(difficulty_factors, 1.0)
            
            # 漫画スタイル推定（カラー判定から）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_variance = np.var(image)
            gray_variance = np.var(gray)
            
            # 色の分散が小さい = モノクロ/漫画風
            manga_style_score = 1.0 - min(color_variance / 10000, 1.0)
            
            # スクリーントーン境界問題検出（モザイク処理を無効化）
            has_screentone = self._detect_screentone_patterns(gray)
            has_mosaic_patterns = False  # モザイク処理無効化
            
            # 境界複雑度の計算（スクリーントーンのみ）
            boundary_complexity = has_screentone
            if boundary_complexity:
                estimated_difficulty = min(estimated_difficulty + 0.3, 1.0)
            
            return ImageCharacteristics(
                has_complex_pose=has_complex_pose,
                has_full_body=has_full_body,
                has_multiple_characters=has_multiple_characters,
                is_face_focus=is_face_focus,
                aspect_ratio=aspect_ratio,
                estimated_difficulty=estimated_difficulty,
                manga_style_score=manga_style_score,
                has_screentone_issues=has_screentone,
                has_mosaic_issues=has_mosaic_patterns,
                has_boundary_complexity=boundary_complexity
            )
            
        except Exception as e:
            self.logger.error(f"画像特性分析エラー: {e}")
            # エラー時のフォールバック
            return ImageCharacteristics(
                has_complex_pose=True,  # 安全側に倒す
                has_full_body=True,
                has_multiple_characters=False,
                is_face_focus=False,
                aspect_ratio=1.0,
                estimated_difficulty=0.7,
                manga_style_score=0.8,
                has_screentone_issues=True,  # 安全側に倒す
                has_mosaic_issues=False,
                has_boundary_complexity=True
            )
    
    def _detect_screentone_patterns(self, gray_image: np.ndarray) -> bool:
        """スクリーントーンパターンの検出"""
        if not CV2_AVAILABLE:
            return False
            
        try:
            # スクリーントーンは規則的な点パターンを持つ
            # FFTを使用して周期的パターンを検出
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # 高周波成分の強度を計算
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            high_freq_region = magnitude_spectrum[center_h-50:center_h+50, center_w-50:center_w+50]
            high_freq_intensity = np.mean(high_freq_region)
            
            # 閾値以上であればスクリーントーンと判定
            return high_freq_intensity > 12.0  # 経験的閾値
            
        except Exception:
            return False
    
    def _detect_mosaic_patterns(self, image: np.ndarray) -> bool:
        """モザイクパターンの検出"""
        if not CV2_AVAILABLE:
            return False
            
        try:
            # モザイクは矩形ブロックの境界線が特徴的
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # エッジ検出
            edges = cv2.Canny(gray, 50, 150)
            
            # 水平・垂直線の検出
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # 格子パターンの検出
            grid_pattern = cv2.bitwise_or(horizontal_lines, vertical_lines)
            grid_ratio = np.sum(grid_pattern > 0) / grid_pattern.size
            
            # 閾値以上であればモザイクと判定
            return grid_ratio > 0.05  # 経験的閾値
            
        except Exception:
            return False
    
    def predict_quality_and_method(self, image_characteristics: ImageCharacteristics,
                                 context: Optional[Dict] = None) -> QualityPrediction:
        """画像特性に基づく品質予測と手法選択"""
        
        # 基本スコア計算（学習データベース）
        method_scores = {}
        
        for method, expected_quality in self.method_expected_quality.items():
            score = expected_quality
            
            # 画像特性による調整（最新データに基づく）
            if image_characteristics.has_complex_pose:
                if method == 'size_priority':
                    score += 0.4  # size_priorityが複雑姿勢に最も強い
                elif method == 'balanced':
                    score -= 0.1  # balancedは安定だが複雑姿勢では若干劣る
                elif method == 'confidence_priority':
                    score -= 0.2  # 実際は複雑姿勢に弱いことが判明
            
            if image_characteristics.has_full_body:
                if method == 'size_priority':
                    score += 0.2  # 全身検出に強い
                elif method == 'fullbody_priority':
                    score += 0.4  # 全身特化
            
            if image_characteristics.has_multiple_characters:
                if method == 'balanced':
                    score += 0.1  # balancedは複数キャラクター処理が安定
                elif method == 'size_priority':
                    score += 0.05  # size_priorityも複数キャラクターに対応可能
                elif method == 'confidence_priority':
                    score -= 0.15  # confidence_priorityは複数キャラクター時に問題が多い
            
            if image_characteristics.is_face_focus:
                if method == 'size_priority':
                    score += 0.1  # size_priorityは顔も適切に抽出
                elif method == 'balanced':
                    score += 0.05  # balancedも顔重視では安定
                elif method == 'confidence_priority':
                    score -= 0.1  # confidence_priorityは顔フォーカスでも期待以下
            
            # 困難度による調整
            difficulty_penalty = image_characteristics.estimated_difficulty * 0.3
            if method == 'size_priority':
                # size_priorityは困難度の影響を受けにくい（最も堅牢）
                difficulty_penalty *= 0.5
            elif method == 'balanced':
                # balancedは中程度の困難度耐性
                difficulty_penalty *= 0.8
            elif method == 'confidence_priority':
                # confidence_priorityは困難度の影響を受けやすい
                difficulty_penalty *= 1.2
            
            # スクリーントーン・モザイク境界問題による調整
            if image_characteristics.has_boundary_complexity:
                if method == 'size_priority':
                    score -= 0.1  # size_priorityでも境界問題は難しい
                elif method == 'balanced':
                    score -= 0.15  # balancedは境界問題に弱い
                elif method == 'confidence_priority':
                    score -= 0.25  # confidence_priorityは境界問題で大幅劣化
            
            score -= difficulty_penalty
            method_scores[method] = max(score, 0.1)  # 最低スコア保証
        
        # 最適手法の選択
        best_method = max(method_scores.items(), key=lambda x: x[1])
        fallback_method = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)[1]
        
        # 信頼度計算
        score_gap = best_method[1] - fallback_method[1]
        confidence = min(score_gap / 2.0 + 0.5, 1.0)
        
        # 推奨理由生成
        reasoning_parts = [f"期待品質: {best_method[1]:.2f}"]
        
        if image_characteristics.has_complex_pose:
            reasoning_parts.append("複雑姿勢対応")
        if image_characteristics.has_full_body:
            reasoning_parts.append("全身検出")
        if image_characteristics.has_multiple_characters:
            reasoning_parts.append("複数キャラクター処理")
        if image_characteristics.estimated_difficulty > 0.6:
            reasoning_parts.append(f"高難易度({image_characteristics.estimated_difficulty:.1f})")
        if image_characteristics.has_boundary_complexity:
            boundary_issues = []
            if image_characteristics.has_screentone_issues:
                boundary_issues.append("スクリーントーン")
            if image_characteristics.has_mosaic_issues:
                boundary_issues.append("モザイク")
            if boundary_issues:
                reasoning_parts.append(f"境界問題({'+'.join(boundary_issues)})")
        
        reasoning = "; ".join(reasoning_parts)
        
        return QualityPrediction(
            predicted_quality=best_method[1],
            confidence=confidence,
            recommended_method=best_method[0],
            fallback_method=fallback_method[0],
            reasoning=reasoning,
            image_characteristics=image_characteristics.__dict__
        )
    
    def should_use_adaptive_learning(self, prediction: QualityPrediction) -> bool:
        """適応学習を使用すべきかの判定"""
        # 高信頼度で良好な予測品質の場合は学習モードを推奨
        return (prediction.confidence >= self.confidence_threshold and 
                prediction.predicted_quality >= self.quality_baseline['good'])
    
    def get_method_parameters(self, method: str, image_characteristics: ImageCharacteristics) -> Dict[str, Any]:
        """手法別の最適化パラメータ取得（実用的な閾値に修正）"""
        # 実用的な低閾値を使用（kaname05検証結果に基づく）
        base_params = {
            'score_threshold': 0.005,  # 0.02 → 0.005 に修正（kaname05で動作確認済み）
            'multi_character_criteria': method,
            'anime_yolo': True
        }
        
        # 画像特性による調整（より実用的に）
        if image_characteristics.has_complex_pose:
            base_params['score_threshold'] = 0.003  # 複雑姿勢では更に低く
        
        if image_characteristics.has_multiple_characters:
            base_params['score_threshold'] = 0.01  # 複数キャラクターでは若干上げる
        
        if image_characteristics.estimated_difficulty > 0.7:
            base_params['score_threshold'] = 0.002  # 高難度では最低レベル
        
        # 境界問題がある場合はさらに感度を上げる
        if image_characteristics.has_boundary_complexity:
            base_params['score_threshold'] = min(base_params['score_threshold'], 0.005)
        
        return base_params
    
    def log_prediction_result(self, image_path: str, prediction: QualityPrediction, 
                            actual_quality: Optional[float] = None):
        """予測結果のログ記録（将来の学習更新用）"""
        # NumPy boolean型をPython boolean型に変換
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            return obj
        
        log_entry = {
            'timestamp': str(pd.Timestamp.now() if 'pd' in globals() else 'timestamp'),
            'image_path': image_path,
            'predicted_quality': prediction.predicted_quality,
            'predicted_method': prediction.recommended_method,
            'confidence': prediction.confidence,
            'reasoning': prediction.reasoning,
            'image_characteristics': convert_numpy_types(prediction.image_characteristics)
        }
        
        if actual_quality is not None:
            log_entry['actual_quality'] = actual_quality
            log_entry['prediction_error'] = abs(prediction.predicted_quality - actual_quality)
        
        # ログファイルに追記（JSONL形式）
        log_path = Path(__file__).parent.parent / "logs" / "quality_predictions.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"ログ記録エラー: {e}")
    
    def update_learning_data(self, feedback_data: List[Dict]):
        """フィードバックデータによる学習更新（将来実装）"""
        # 現在は基本実装のみ
        # 実際の運用では、予測精度と実際の結果の差分から
        # method_expected_qualityなどを動的に更新
        self.logger.info(f"学習データ更新: {len(feedback_data)}件のフィードバック")


def create_quality_assessor() -> LearnedQualityAssessment:
    """品質評価システムのファクトリ関数"""
    return LearnedQualityAssessment()


# segment-anythingパイプライン統合用のメイン関数
def assess_image_quality(image_path: str, 
                        yolo_detections: Optional[List] = None,
                        context: Optional[Dict] = None) -> QualityPrediction:
    """
    画像の品質評価と最適手法選択のメイン関数
    
    Args:
        image_path: 対象画像パス
        yolo_detections: YOLO検出結果（オプション）
        context: 追加コンテキスト情報（オプション）
    
    Returns:
        QualityPrediction: 品質予測結果
    """
    assessor = create_quality_assessor()
    characteristics = assessor.analyze_image_characteristics(image_path, yolo_detections)
    prediction = assessor.predict_quality_and_method(characteristics, context)
    
    # ログ記録
    assessor.log_prediction_result(image_path, prediction)
    
    return prediction


if __name__ == "__main__":
    # テスト実行
    test_image = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname05/kaname05_0001.jpg"
    
    if os.path.exists(test_image):
        print("🧪 品質評価システムテスト実行中...")
        
        prediction = assess_image_quality(test_image)
        
        print(f"📊 予測結果:")
        print(f"   推奨手法: {prediction.recommended_method}")
        print(f"   予測品質: {prediction.predicted_quality:.3f}")
        print(f"   信頼度: {prediction.confidence:.3f}")
        print(f"   フォールバック: {prediction.fallback_method}")
        print(f"   理由: {prediction.reasoning}")
        print(f"   画像特性: {prediction.image_characteristics}")
        
        print("✅ テスト完了")
    else:
        print("❌ テスト画像が見つかりません")