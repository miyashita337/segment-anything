#!/usr/bin/env python3
"""
P1-018: 滑らかさ評価指標の実装
境界線の滑らかさを多角的に定量評価するシステム

Features:
- Curvature-based smoothness analysis
- Frequency domain analysis
- Local variation assessment
- Multi-scale smoothness evaluation
- A-F grading system
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime

# フォールバック実装用
HAS_SCIPY = True
HAS_SKLEARN = True

try:
    from scipy import ndimage, signal
    from scipy.interpolate import interp1d
    from scipy.stats import variation
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    HAS_SKLEARN = False


class SmoothnessMetrics:
    """境界線滑らかさ評価システム"""
    
    def __init__(self):
        """初期化"""
        self.name = "SmoothnessMetrics"
        self.version = "1.0.0"
        
        # 滑らかさ評価パラメータ
        self.smoothness_params = {
            'curvature_window': 5,
            'frequency_cutoff': 0.1,
            'variation_threshold': 0.3,
            'gradient_smoothing': 2,
            'multi_scale_levels': 3
        }
        
        # グレーディング基準
        self.grading_thresholds = {
            'A': 0.85,  # Excellent smoothness
            'B': 0.70,  # Good smoothness
            'C': 0.55,  # Acceptable smoothness
            'D': 0.40,  # Poor smoothness
            'F': 0.00   # Very poor smoothness
        }
    
    def analyze_boundary_smoothness(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        境界線の包括的滑らかさ分析
        
        Args:
            mask: バイナリマスク画像
            
        Returns:
            Dict: 滑らかさ分析結果
        """
        if mask is None or mask.size == 0:
            return self._generate_error_result("Empty or invalid mask")
        
        try:
            # 境界線抽出
            contours = self._extract_contours(mask)
            if not contours:
                return self._generate_error_result("No contours found")
            
            # メイン境界線選択（最大面積）
            main_contour = max(contours, key=cv2.contourArea)
            
            if len(main_contour) < 10:
                return self._generate_error_result("Contour too small for analysis")
            
            # 各種滑らかさ指標計算
            curvature_metrics = self._analyze_curvature_smoothness(main_contour)
            frequency_metrics = self._analyze_frequency_smoothness(main_contour)
            variation_metrics = self._analyze_local_variation(main_contour)
            gradient_metrics = self._analyze_gradient_smoothness(main_contour)
            multiscale_metrics = self._analyze_multiscale_smoothness(main_contour)
            
            # 総合評価計算
            overall_assessment = self._calculate_overall_smoothness(
                curvature_metrics, frequency_metrics, variation_metrics,
                gradient_metrics, multiscale_metrics
            )
            
            return {
                'analysis_type': 'boundary_smoothness',
                'contour_info': {
                    'point_count': len(main_contour),
                    'area': float(cv2.contourArea(main_contour)),
                    'perimeter': float(cv2.arcLength(main_contour, True))
                },
                'curvature_analysis': curvature_metrics,
                'frequency_analysis': frequency_metrics,
                'variation_analysis': variation_metrics,
                'gradient_analysis': gradient_metrics,
                'multiscale_analysis': multiscale_metrics,
                'overall_assessment': overall_assessment,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': self.version
                }
            }
            
        except Exception as e:
            return self._generate_error_result(f"Analysis failed: {str(e)}")
    
    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """境界線抽出"""
        # マスクを8bit unsignedに変換
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # 境界線検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 最小サイズフィルタリング
        min_contour_length = 20
        valid_contours = [c for c in contours if len(c) >= min_contour_length]
        
        return valid_contours
    
    def _analyze_curvature_smoothness(self, contour: np.ndarray) -> Dict[str, Any]:
        """曲率ベース滑らかさ分析"""
        try:
            # 境界点座標抽出
            points = contour.reshape(-1, 2).astype(np.float32)
            
            # 曲率計算
            curvatures = self._calculate_curvature(points)
            
            if len(curvatures) == 0:
                return {'error': 'Failed to calculate curvatures'}
            
            # 曲率統計
            curvature_stats = {
                'mean_curvature': float(np.mean(np.abs(curvatures))),
                'std_curvature': float(np.std(curvatures)),
                'max_curvature': float(np.max(np.abs(curvatures))),
                'curvature_variation': float(np.var(curvatures))
            }
            
            # 急激な曲率変化の検出
            curvature_changes = np.abs(np.diff(curvatures))
            sharp_changes = np.sum(curvature_changes > np.percentile(curvature_changes, 90))
            
            # 滑らかさスコア計算（曲率変化の少なさベース）
            curvature_score = self._calculate_curvature_smoothness_score(curvatures)
            
            return {
                'curvature_statistics': curvature_stats,
                'sharp_change_count': int(sharp_changes),
                'curvature_smoothness_score': curvature_score,
                'curvature_grade': self._score_to_grade(curvature_score)
            }
            
        except Exception as e:
            return {'error': f'Curvature analysis failed: {str(e)}'}
    
    def _calculate_curvature(self, points: np.ndarray) -> np.ndarray:
        """曲率計算（3点法）"""
        if len(points) < 3:
            return np.array([])
        
        # 境界を循環として扱う
        extended_points = np.vstack([points[-1:], points, points[:1]])
        curvatures = []
        
        for i in range(1, len(extended_points) - 1):
            p1, p2, p3 = extended_points[i-1], extended_points[i], extended_points[i+1]
            
            # ベクトル計算
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 外積による曲率計算
            cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                curvature = cross_prod / (v1_norm * v2_norm)
                curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _calculate_curvature_smoothness_score(self, curvatures: np.ndarray) -> float:
        """曲率ベース滑らかさスコア計算"""
        if len(curvatures) == 0:
            return 0.0
        
        # 曲率変化の分散（小さいほど滑らか）
        curvature_variance = np.var(curvatures)
        
        # 急激な変化の頻度
        changes = np.abs(np.diff(curvatures))
        high_changes = np.sum(changes > np.percentile(changes, 75))
        change_ratio = high_changes / len(changes) if len(changes) > 0 else 1.0
        
        # スコア計算（0-1範囲、1が最も滑らか）
        variance_score = 1.0 / (1.0 + curvature_variance * 10)
        change_score = 1.0 - min(change_ratio, 1.0)
        
        return (variance_score + change_score) / 2.0
    
    def _analyze_frequency_smoothness(self, contour: np.ndarray) -> Dict[str, Any]:
        """周波数ドメイン滑らかさ分析"""
        try:
            points = contour.reshape(-1, 2)
            
            # X, Y座標の周波数解析
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # FFT解析（scipy利用可能時）
            if HAS_SCIPY:
                x_fft = np.fft.fft(x_coords)
                y_fft = np.fft.fft(y_coords)
                
                # 高周波成分の強度
                freqs = np.fft.fftfreq(len(x_coords))
                high_freq_mask = np.abs(freqs) > self.smoothness_params['frequency_cutoff']
                
                x_high_freq_power = np.sum(np.abs(x_fft[high_freq_mask])**2)
                y_high_freq_power = np.sum(np.abs(y_fft[high_freq_mask])**2)
                total_power = np.sum(np.abs(x_fft)**2) + np.sum(np.abs(y_fft)**2)
                
                high_freq_ratio = (x_high_freq_power + y_high_freq_power) / total_power if total_power > 0 else 0
                
                # 周波数ベース滑らかさスコア（高周波成分が少ないほど滑らか）
                frequency_score = 1.0 - min(high_freq_ratio * 2, 1.0)
                
            else:
                # フォールバック：差分ベース高周波推定
                x_diff = np.abs(np.diff(x_coords))
                y_diff = np.abs(np.diff(y_coords))
                
                high_freq_estimate = np.mean(x_diff) + np.mean(y_diff)
                frequency_score = 1.0 / (1.0 + high_freq_estimate * 0.1)
            
            return {
                'frequency_smoothness_score': frequency_score,
                'frequency_grade': self._score_to_grade(frequency_score),
                'analysis_method': 'scipy_fft' if HAS_SCIPY else 'fallback_diff'
            }
            
        except Exception as e:
            return {'error': f'Frequency analysis failed: {str(e)}'}
    
    def _analyze_local_variation(self, contour: np.ndarray) -> Dict[str, Any]:
        """局所変動分析"""
        try:
            points = contour.reshape(-1, 2).astype(np.float32)
            
            # 隣接点間距離の変動
            distances = []
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                dist = np.linalg.norm(p2 - p1)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # 変動係数計算
            if HAS_SCIPY:
                variation_coeff = variation(distances) if np.mean(distances) > 0 else 0
            else:
                # フォールバック実装
                mean_dist = np.mean(distances)
                variation_coeff = np.std(distances) / mean_dist if mean_dist > 0 else 0
            
            # 局所角度変化
            angle_changes = self._calculate_angle_changes(points)
            angle_variation = np.std(angle_changes) if len(angle_changes) > 0 else 0
            
            # 局所変動スコア
            distance_score = 1.0 / (1.0 + variation_coeff * 5)
            angle_score = 1.0 / (1.0 + angle_variation * 2)
            variation_score = (distance_score + angle_score) / 2.0
            
            return {
                'distance_variation_coefficient': float(variation_coeff),
                'angle_variation': float(angle_variation),
                'local_variation_score': variation_score,
                'variation_grade': self._score_to_grade(variation_score)
            }
            
        except Exception as e:
            return {'error': f'Local variation analysis failed: {str(e)}'}
    
    def _calculate_angle_changes(self, points: np.ndarray) -> np.ndarray:
        """隣接ベクトル間角度変化計算"""
        if len(points) < 3:
            return np.array([])
        
        angle_changes = []
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            p3 = points[(i + 2) % len(points)]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 角度計算
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)
                angle_changes.append(angle_change)
        
        return np.array(angle_changes)
    
    def _analyze_gradient_smoothness(self, contour: np.ndarray) -> Dict[str, Any]:
        """勾配ベース滑らかさ分析"""
        try:
            points = contour.reshape(-1, 2).astype(np.float32)
            
            # X, Y方向勾配計算
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # 勾配計算（中央差分法）
            x_grad = np.gradient(x_coords)
            y_grad = np.gradient(y_coords)
            
            # 勾配の大きさと方向
            grad_magnitude = np.sqrt(x_grad**2 + y_grad**2)
            grad_direction = np.arctan2(y_grad, x_grad)
            
            # 勾配の滑らかさ指標
            grad_smoothness = {
                'magnitude_variation': float(np.std(grad_magnitude)),
                'direction_variation': float(np.std(np.diff(grad_direction))),
                'magnitude_range': float(np.ptp(grad_magnitude))
            }
            
            # 勾配ベーススコア
            mag_score = 1.0 / (1.0 + grad_smoothness['magnitude_variation'] * 0.1)
            dir_score = 1.0 / (1.0 + grad_smoothness['direction_variation'] * 2)
            gradient_score = (mag_score + dir_score) / 2.0
            
            return {
                'gradient_statistics': grad_smoothness,
                'gradient_smoothness_score': gradient_score,
                'gradient_grade': self._score_to_grade(gradient_score)
            }
            
        except Exception as e:
            return {'error': f'Gradient analysis failed: {str(e)}'}
    
    def _analyze_multiscale_smoothness(self, contour: np.ndarray) -> Dict[str, Any]:
        """マルチスケール滑らかさ分析"""
        try:
            points = contour.reshape(-1, 2)
            
            scale_results = {}
            scale_scores = []
            
            # 複数スケールで解析
            for level in range(self.smoothness_params['multi_scale_levels']):
                scale_factor = 2 ** level
                
                # サブサンプリング
                if scale_factor < len(points):
                    sampled_indices = np.arange(0, len(points), scale_factor)
                    sampled_points = points[sampled_indices]
                    
                    # このスケールでの滑らかさ評価
                    if len(sampled_points) >= 3:
                        scale_curvatures = self._calculate_curvature(sampled_points.astype(np.float32))
                        scale_score = self._calculate_curvature_smoothness_score(scale_curvatures)
                        
                        scale_results[f'scale_{level}'] = {
                            'scale_factor': scale_factor,
                            'point_count': len(sampled_points),
                            'smoothness_score': scale_score
                        }
                        
                        scale_scores.append(scale_score)
            
            # マルチスケール総合評価
            if scale_scores:
                multiscale_score = np.mean(scale_scores)
                scale_consistency = 1.0 - np.std(scale_scores)  # スケール間の一貫性
            else:
                multiscale_score = 0.0
                scale_consistency = 0.0
            
            return {
                'scale_analyses': scale_results,
                'multiscale_smoothness_score': multiscale_score,
                'scale_consistency': scale_consistency,
                'multiscale_grade': self._score_to_grade(multiscale_score)
            }
            
        except Exception as e:
            return {'error': f'Multiscale analysis failed: {str(e)}'}
    
    def _calculate_overall_smoothness(self, curvature_metrics: Dict, frequency_metrics: Dict,
                                    variation_metrics: Dict, gradient_metrics: Dict,
                                    multiscale_metrics: Dict) -> Dict[str, Any]:
        """総合滑らかさ評価計算"""
        # 各メトリクスからスコア抽出
        scores = {}
        weights = {}
        
        if 'curvature_smoothness_score' in curvature_metrics:
            scores['curvature'] = curvature_metrics['curvature_smoothness_score']
            weights['curvature'] = 0.3
        
        if 'frequency_smoothness_score' in frequency_metrics:
            scores['frequency'] = frequency_metrics['frequency_smoothness_score']
            weights['frequency'] = 0.2
        
        if 'local_variation_score' in variation_metrics:
            scores['variation'] = variation_metrics['local_variation_score']
            weights['variation'] = 0.2
        
        if 'gradient_smoothness_score' in gradient_metrics:
            scores['gradient'] = gradient_metrics['gradient_smoothness_score']
            weights['gradient'] = 0.15
        
        if 'multiscale_smoothness_score' in multiscale_metrics:
            scores['multiscale'] = multiscale_metrics['multiscale_smoothness_score']
            weights['multiscale'] = 0.15
        
        # 重み付き平均計算
        if scores:
            total_weight = sum(weights.values())
            weighted_sum = sum(scores[key] * weights[key] for key in scores)
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = 0.0
        
        # 信頼性評価
        available_metrics = len(scores)
        confidence = min(available_metrics / 5.0, 1.0)  # 5メトリクス全て利用可能で信頼性100%
        
        return {
            'overall_smoothness_score': overall_score,
            'smoothness_grade': self._score_to_grade(overall_score),
            'individual_scores': scores,
            'confidence': confidence,
            'available_metrics': available_metrics,
            'assessment': self._generate_smoothness_assessment(overall_score, scores)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """スコアからグレードへの変換"""
        for grade, threshold in self.grading_thresholds.items():
            if score >= threshold:
                return grade
        return 'F'
    
    def _generate_smoothness_assessment(self, overall_score: float, scores: Dict) -> str:
        """滑らかさ評価コメント生成"""
        if overall_score >= 0.85:
            return "excellent_smoothness"
        elif overall_score >= 0.70:
            return "good_smoothness"
        elif overall_score >= 0.55:
            return "acceptable_smoothness"
        elif overall_score >= 0.40:
            return "poor_smoothness"
        else:
            return "very_poor_smoothness"
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        return {
            'error': error_message,
            'overall_assessment': {
                'overall_smoothness_score': 0.0,
                'smoothness_grade': 'F',
                'assessment': 'analysis_failed'
            }
        }


def main():
    """テスト実行"""
    print("🚀 P1-018: 滑らかさ評価指標システム テスト開始")
    
    # テスト用円形マスク作成
    test_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_mask, (100, 100), 80, 255, -1)
    
    # 滑らかさ分析実行
    analyzer = SmoothnessMetrics()
    result = analyzer.analyze_boundary_smoothness(test_mask)
    
    print("\n📊 滑らかさ分析結果:")
    if 'error' not in result:
        overall = result.get('overall_assessment', {})
        print(f"  総合スコア: {overall.get('overall_smoothness_score', 0):.3f}")
        print(f"  滑らかさグレード: {overall.get('smoothness_grade', 'N/A')}")
        print(f"  信頼性: {overall.get('confidence', 0):.3f}")
        print(f"  評価: {overall.get('assessment', 'N/A')}")
        
        # 個別メトリクス表示
        individual_scores = overall.get('individual_scores', {})
        if individual_scores:
            print(f"\n🔍 個別メトリクス:")
            for metric, score in individual_scores.items():
                print(f"    {metric}: {score:.3f}")
    else:
        print(f"  ❌ エラー: {result['error']}")
    
    print(f"\n✅ [P1-018] 滑らかさ評価指標システム完了")


if __name__ == "__main__":
    main()