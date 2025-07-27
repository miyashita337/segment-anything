#!/usr/bin/env python3
"""
適応的パラメータ調整システム
画像特性に応じたリアルタイム最適化・動的パラメータ調整

目標:
- 画像特性3軸（明度・コントラスト・複雑度）での適応制御
- YOLO閾値・SAM設定・輪郭パラメータの自動最適化
"""

import numpy as np
import cv2

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageCharacteristics:
    """画像特性"""
    brightness: float  # 明度 (0-255)
    contrast: float    # コントラスト (標準偏差)
    complexity: float  # 複雑度 (エッジ密度)
    saturation: float  # 彩度平均
    noise_level: float # ノイズレベル
    character_density: float  # キャラクター密度推定


@dataclass
class OptimizationParameters:
    """最適化パラメータ"""
    # YOLO設定
    yolo_threshold: float = 0.07
    yolo_nms_threshold: float = 0.5
    yolo_confidence_boost: float = 1.0
    
    # SAM設定
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.86
    sam_stability_score_thresh: float = 0.92
    sam_crop_n_layers: int = 1
    
    # 輪郭設定
    contour_smoothing_epsilon: float = 0.02
    contour_gaussian_sigma: float = 1.0
    contour_morphological_kernel: int = 3
    
    # 後処理設定
    noise_removal_threshold: int = 100
    edge_smoothing_iterations: int = 2
    quality_enhancement_level: str = 'moderate'  # 'conservative', 'moderate', 'aggressive'


class AdaptiveParameterOptimizer:
    """適応的パラメータ調整システム"""
    
    def __init__(self):
        """初期化"""
        self.optimization_history: List[Dict] = []
        self.learned_patterns: Dict[str, Dict] = {}
        
    def optimize_parameters_for_image(self, 
                                    image: np.ndarray,
                                    initial_params: Optional[OptimizationParameters] = None) -> OptimizationParameters:
        """
        画像特性に基づくパラメータ最適化
        
        Args:
            image: 入力画像
            initial_params: 初期パラメータ（未指定時はデフォルト使用）
            
        Returns:
            OptimizationParameters: 最適化されたパラメータ
        """
        try:
            logger.info("🎛️ 適応的パラメータ最適化開始")
            
            # Step 1: 画像特性分析
            characteristics = self._analyze_image_characteristics(image)
            logger.info(f"📊 画像特性: 明度={characteristics.brightness:.1f}, "
                       f"コントラスト={characteristics.contrast:.1f}, "
                       f"複雑度={characteristics.complexity:.3f}")
            
            # Step 2: 基本パラメータ設定
            base_params = initial_params or OptimizationParameters()
            
            # Step 3: 明度ベース調整
            brightness_adjusted = self._adjust_for_brightness(base_params, characteristics)
            
            # Step 4: コントラストベース調整
            contrast_adjusted = self._adjust_for_contrast(brightness_adjusted, characteristics)
            
            # Step 5: 複雑度ベース調整
            complexity_adjusted = self._adjust_for_complexity(contrast_adjusted, characteristics)
            
            # Step 6: アニメ特性ベース調整
            anime_adjusted = self._adjust_for_anime_characteristics(complexity_adjusted, characteristics)
            
            # Step 7: 学習パターン適用
            learned_adjusted = self._apply_learned_patterns(anime_adjusted, characteristics)
            
            # Step 8: 最適化履歴更新
            self._update_optimization_history(characteristics, learned_adjusted)
            
            logger.info("✅ 適応的パラメータ最適化完了")
            
            return learned_adjusted
            
        except Exception as e:
            logger.error(f"❌ パラメータ最適化エラー: {e}")
            return initial_params or OptimizationParameters()
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> ImageCharacteristics:
        """画像特性分析"""
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                gray = image
                hsv = None
            
            # 1. 明度分析
            brightness = np.mean(gray)
            
            # 2. コントラスト分析（標準偏差）
            contrast = np.std(gray)
            
            # 3. 複雑度分析（エッジ密度）
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = gray.shape[0] * gray.shape[1]
            complexity = edge_pixels / total_pixels
            
            # 4. 彩度分析
            if hsv is not None:
                saturation = np.mean(hsv[:, :, 1])
            else:
                saturation = 128  # グレースケールの場合は中間値
            
            # 5. ノイズレベル分析
            noise_level = self._estimate_noise_level(gray)
            
            # 6. キャラクター密度推定
            character_density = self._estimate_character_density(gray)
            
            return ImageCharacteristics(
                brightness=brightness,
                contrast=contrast,
                complexity=complexity,
                saturation=saturation,
                noise_level=noise_level,
                character_density=character_density
            )
            
        except Exception as e:
            logger.warning(f"⚠️ 画像特性分析エラー: {e}")
            # フォールバック値
            return ImageCharacteristics(
                brightness=128, contrast=50, complexity=0.1,
                saturation=128, noise_level=0.1, character_density=0.2
            )
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """ノイズレベル推定"""
        try:
            # Laplacianによるノイズ推定
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = laplacian.var()
            
            # 0-1に正規化
            normalized_noise = min(1.0, noise_variance / 1000.0)
            
            return normalized_noise
            
        except Exception as e:
            logger.warning(f"⚠️ ノイズレベル推定エラー: {e}")
            return 0.1
    
    def _estimate_character_density(self, gray: np.ndarray) -> float:
        """キャラクター密度推定"""
        try:
            # 輪郭検出によるキャラクター候補推定
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # キャラクターサイズ範囲の輪郭をカウント
            character_contours = 0
            total_area = gray.shape[0] * gray.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                area_ratio = area / total_area
                
                # キャラクターらしいサイズ範囲
                if 0.01 <= area_ratio <= 0.6:
                    character_contours += 1
            
            # 密度計算
            density = min(1.0, character_contours / 5.0)  # 最大5キャラクターを想定
            
            return density
            
        except Exception as e:
            logger.warning(f"⚠️ キャラクター密度推定エラー: {e}")
            return 0.2
    
    def _adjust_for_brightness(self, 
                             params: OptimizationParameters, 
                             characteristics: ImageCharacteristics) -> OptimizationParameters:
        """明度ベース調整"""
        try:
            adjusted = OptimizationParameters(**params.__dict__)
            brightness = characteristics.brightness
            
            # 暗い画像 (0-80)
            if brightness < 80:
                logger.debug("🌙 暗い画像検出 - 検出感度向上")
                adjusted.yolo_threshold *= 0.7  # 閾値を下げて検出感度向上
                adjusted.sam_pred_iou_thresh *= 0.95  # IoU閾値を下げる
                adjusted.contour_smoothing_epsilon *= 0.8  # より細かい輪郭抽出
                adjusted.quality_enhancement_level = 'aggressive'
                
            # 明るい画像 (180-255)
            elif brightness > 180:
                logger.debug("☀️ 明るい画像検出 - ノイズ対策強化")
                adjusted.yolo_threshold *= 1.2  # 閾値を上げてノイズ除去
                adjusted.noise_removal_threshold = int(adjusted.noise_removal_threshold * 1.5)
                adjusted.edge_smoothing_iterations += 1
                adjusted.quality_enhancement_level = 'conservative'
                
            # 標準明度 (80-180)
            else:
                logger.debug("🌤️ 標準明度 - バランス調整")
                adjusted.quality_enhancement_level = 'moderate'
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"⚠️ 明度ベース調整エラー: {e}")
            return params
    
    def _adjust_for_contrast(self, 
                           params: OptimizationParameters, 
                           characteristics: ImageCharacteristics) -> OptimizationParameters:
        """コントラストベース調整"""
        try:
            adjusted = OptimizationParameters(**params.__dict__)
            contrast = characteristics.contrast
            
            # 低コントラスト (0-30)
            if contrast < 30:
                logger.debug("📉 低コントラスト検出 - エッジ強化")
                adjusted.sam_stability_score_thresh *= 0.9  # 安定性閾値を下げる
                adjusted.contour_gaussian_sigma *= 0.7  # ガウシアン平滑化を弱める
                adjusted.edge_smoothing_iterations = max(1, adjusted.edge_smoothing_iterations - 1)
                
            # 高コントラスト (80-255)
            elif contrast > 80:
                logger.debug("📈 高コントラスト検出 - スムージング強化")
                adjusted.contour_gaussian_sigma *= 1.3  # ガウシアン平滑化を強める
                adjusted.contour_morphological_kernel += 2  # モルフォロジカル処理を強化
                adjusted.edge_smoothing_iterations += 1
                
            # 標準コントラスト (30-80)
            else:
                logger.debug("📊 標準コントラスト - バランス維持")
                # デフォルト設定を維持
                pass
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"⚠️ コントラストベース調整エラー: {e}")
            return params
    
    def _adjust_for_complexity(self, 
                             params: OptimizationParameters, 
                             characteristics: ImageCharacteristics) -> OptimizationParameters:
        """複雑度ベース調整"""
        try:
            adjusted = OptimizationParameters(**params.__dict__)
            complexity = characteristics.complexity
            
            # 低複雑度 (0-0.05) - シンプルな画像
            if complexity < 0.05:
                logger.debug("🔹 シンプル画像 - 精度優先")
                adjusted.sam_points_per_side = min(64, adjusted.sam_points_per_side * 2)  # より多くのポイント
                adjusted.sam_crop_n_layers += 1  # より細かいクロップ
                adjusted.contour_smoothing_epsilon *= 0.5  # より精密な輪郭抽出
                
            # 高複雑度 (0.15-1.0) - 複雑な画像
            elif complexity > 0.15:
                logger.debug("🔸 複雑画像 - 効率優先")
                adjusted.sam_points_per_side = max(16, adjusted.sam_points_per_side // 2)  # ポイント数削減
                adjusted.noise_removal_threshold = int(adjusted.noise_removal_threshold * 1.5)  # ノイズ除去強化
                adjusted.contour_smoothing_epsilon *= 1.5  # スムージング強化
                
            # 標準複雑度 (0.05-0.15)
            else:
                logger.debug("🔶 標準複雑度 - バランス調整")
                # デフォルト設定を維持
                pass
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"⚠️ 複雑度ベース調整エラー: {e}")
            return params
    
    def _adjust_for_anime_characteristics(self, 
                                        params: OptimizationParameters, 
                                        characteristics: ImageCharacteristics) -> OptimizationParameters:
        """アニメ特性ベース調整"""
        try:
            adjusted = OptimizationParameters(**params.__dict__)
            saturation = characteristics.saturation
            
            # 高彩度 (180-255) - 典型的なアニメ
            if saturation > 180:
                logger.debug("🎨 高彩度アニメ - アニメ特化調整")
                adjusted.yolo_threshold *= 0.85  # アニメキャラクター検出感度向上
                adjusted.yolo_confidence_boost = 1.2  # 信頼度ブースト
                adjusted.contour_smoothing_epsilon *= 0.7  # アニメの明確な輪郭に対応
                
            # 低彩度 (0-100) - モノクロ・セピア調
            elif saturation < 100:
                logger.debug("🖤 低彩度画像 - モノクロ対応調整")
                adjusted.sam_stability_score_thresh *= 0.95  # 安定性要求を緩和
                adjusted.contour_gaussian_sigma *= 1.2  # スムージング強化
                
            # 中彩度 (100-180) - 標準的なアニメ
            else:
                logger.debug("🎭 標準彩度アニメ - 標準調整")
                adjusted.yolo_confidence_boost = 1.1  # 軽微なブースト
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"⚠️ アニメ特性ベース調整エラー: {e}")
            return params
    
    def _apply_learned_patterns(self, 
                              params: OptimizationParameters, 
                              characteristics: ImageCharacteristics) -> OptimizationParameters:
        """学習パターン適用"""
        try:
            # 特性に基づくパターンキー生成
            pattern_key = self._generate_pattern_key(characteristics)
            
            if pattern_key in self.learned_patterns:
                learned_adjustments = self.learned_patterns[pattern_key]
                adjusted = OptimizationParameters(**params.__dict__)
                
                logger.debug(f"🧠 学習パターン適用: {pattern_key}")
                
                # 学習済み調整の適用
                for param_name, adjustment_factor in learned_adjustments.items():
                    if hasattr(adjusted, param_name):
                        current_value = getattr(adjusted, param_name)
                        if isinstance(current_value, (int, float)):
                            new_value = current_value * adjustment_factor
                            setattr(adjusted, param_name, new_value)
                
                return adjusted
            else:
                logger.debug("💡 新規パターン - 学習対象として記録")
                return params
                
        except Exception as e:
            logger.warning(f"⚠️ 学習パターン適用エラー: {e}")
            return params
    
    def _generate_pattern_key(self, characteristics: ImageCharacteristics) -> str:
        """特性に基づくパターンキー生成"""
        try:
            # 特性を離散化してパターンキー作成
            brightness_level = 'dark' if characteristics.brightness < 80 else ('bright' if characteristics.brightness > 180 else 'normal')
            contrast_level = 'low' if characteristics.contrast < 30 else ('high' if characteristics.contrast > 80 else 'normal')
            complexity_level = 'simple' if characteristics.complexity < 0.05 else ('complex' if characteristics.complexity > 0.15 else 'normal')
            
            return f"{brightness_level}_{contrast_level}_{complexity_level}"
            
        except Exception as e:
            logger.warning(f"⚠️ パターンキー生成エラー: {e}")
            return "unknown"
    
    def _update_optimization_history(self, 
                                   characteristics: ImageCharacteristics, 
                                   optimized_params: OptimizationParameters) -> None:
        """最適化履歴更新"""
        try:
            history_entry = {
                'timestamp': time.time(),
                'characteristics': characteristics.__dict__,
                'optimized_params': optimized_params.__dict__,
                'pattern_key': self._generate_pattern_key(characteristics)
            }
            
            self.optimization_history.append(history_entry)
            
            # 履歴サイズ制限（最新50件保持）
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]
                
        except Exception as e:
            logger.warning(f"⚠️ 最適化履歴更新エラー: {e}")
    
    def learn_from_quality_feedback(self, 
                                  characteristics: ImageCharacteristics,
                                  quality_score: float,
                                  used_params: OptimizationParameters) -> None:
        """品質フィードバックからの学習"""
        try:
            pattern_key = self._generate_pattern_key(characteristics)
            
            # 高品質結果の学習 (スコア > 0.7)
            if quality_score > 0.7:
                if pattern_key not in self.learned_patterns:
                    self.learned_patterns[pattern_key] = {}
                
                # 成功パラメータの記録・強化
                base_params = OptimizationParameters()
                for param_name in ['yolo_threshold', 'sam_pred_iou_thresh', 'contour_smoothing_epsilon']:
                    if hasattr(used_params, param_name) and hasattr(base_params, param_name):
                        used_value = getattr(used_params, param_name)
                        base_value = getattr(base_params, param_name)
                        
                        if base_value != 0:
                            adjustment_factor = used_value / base_value
                            
                            # 既存学習の重み付き更新
                            if param_name in self.learned_patterns[pattern_key]:
                                current_factor = self.learned_patterns[pattern_key][param_name]
                                # 0.7:0.3の重みで更新
                                new_factor = current_factor * 0.7 + adjustment_factor * 0.3
                            else:
                                new_factor = adjustment_factor
                            
                            self.learned_patterns[pattern_key][param_name] = new_factor
                
                logger.info(f"📚 学習更新: {pattern_key} (品質スコア: {quality_score:.3f})")
                
        except Exception as e:
            logger.warning(f"⚠️ 品質フィードバック学習エラー: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポート取得"""
        try:
            if not self.optimization_history:
                return {'message': '最適化履歴がありません'}
            
            # パターン分布分析
            pattern_distribution = {}
            for entry in self.optimization_history:
                pattern = entry['pattern_key']
                pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1
            
            # 最新特性統計
            recent_entries = self.optimization_history[-10:]
            if recent_entries:
                avg_brightness = np.mean([e['characteristics']['brightness'] for e in recent_entries])
                avg_contrast = np.mean([e['characteristics']['contrast'] for e in recent_entries])
                avg_complexity = np.mean([e['characteristics']['complexity'] for e in recent_entries])
            else:
                avg_brightness = avg_contrast = avg_complexity = 0
            
            return {
                'total_optimizations': len(self.optimization_history),
                'learned_patterns': len(self.learned_patterns),
                'pattern_distribution': pattern_distribution,
                'recent_averages': {
                    'brightness': avg_brightness,
                    'contrast': avg_contrast,
                    'complexity': avg_complexity
                },
                'learned_pattern_keys': list(self.learned_patterns.keys())
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 最適化レポート取得エラー: {e}")
            return {'error': str(e)}
    
    def save_learned_patterns(self, output_path: str) -> bool:
        """学習パターン保存"""
        try:
            save_data = {
                'learned_patterns': self.learned_patterns,
                'optimization_history_summary': {
                    'total_optimizations': len(self.optimization_history),
                    'pattern_distribution': {}
                },
                'saved_at': time.time()
            }
            
            # パターン分布計算
            for entry in self.optimization_history:
                pattern = entry['pattern_key']
                save_data['optimization_history_summary']['pattern_distribution'][pattern] = \
                    save_data['optimization_history_summary']['pattern_distribution'].get(pattern, 0) + 1
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 学習パターン保存完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 学習パターン保存エラー: {e}")
            return False
    
    def load_learned_patterns(self, input_path: str) -> bool:
        """学習パターン読み込み"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            self.learned_patterns = load_data.get('learned_patterns', {})
            
            logger.info(f"📂 学習パターン読み込み完了: {len(self.learned_patterns)}パターン")
            return True
            
        except Exception as e:
            logger.error(f"❌ 学習パターン読み込みエラー: {e}")
            return False


def integrate_with_processing_systems() -> None:
    """処理システム統合準備"""
    logger.info("🔗 適応的パラメータ調整システムを処理システムに統合準備")
    
    # YOLO, SAM, 輪郭システムとの統合準備
    # 実際の統合は次のステップで実装
    pass


if __name__ == "__main__":
    # テスト実行
    logger.info("🧪 適応的パラメータ調整システム テスト開始")
    
    optimizer = AdaptiveParameterOptimizer()
    logger.info("✅ 適応的パラメータ調整システム初期化完了")
    
    # テスト用画像特性
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    optimized_params = optimizer.optimize_parameters_for_image(test_image)
    
    logger.info(f"🎛️ 最適化完了: YOLO閾値 {optimized_params.yolo_threshold:.3f}")
    
    # 統合準備
    integrate_with_processing_systems()
    logger.info("🎯 テスト完了 - 実装準備完了")