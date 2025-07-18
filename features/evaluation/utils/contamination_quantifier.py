#!/usr/bin/env python3
"""
P1-022: 混入率定量化システム
背景・前景分離における混入率を詳細に定量化するシステム

Features:
- Multi-modal contamination detection
- Pixel-level purity analysis
- Color distance contamination assessment
- Texture-based contamination detection
- Spatial contamination pattern analysis
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
    from scipy import ndimage, stats
    from scipy.spatial.distance import cdist
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
except ImportError:
    HAS_SKLEARN = False


class ContaminationQuantifier:
    """混入率定量化システム"""
    
    def __init__(self):
        """初期化"""
        self.name = "ContaminationQuantifier"
        self.version = "1.0.0"
        
        # 混入検出パラメータ
        self.detection_params = {
            'color_similarity_threshold': 30,  # 色類似度閾値
            'texture_window_size': 8,          # テクスチャ解析ウィンドウサイズ
            'contamination_cluster_min': 5,    # 混入クラスタ最小サイズ
            'purity_threshold': 0.8,           # 純度閾値
            'spatial_coherence_radius': 5,     # 空間一貫性半径
            'edge_contamination_width': 3      # エッジ混入検出幅
        }
        
        # グレーディング基準（低い方が良い）
        self.contamination_grades = {
            'A': (0.0, 0.05),   # Excellent purity
            'B': (0.05, 0.15),  # Good purity
            'C': (0.15, 0.30),  # Acceptable purity
            'D': (0.30, 0.50),  # Poor purity
            'F': (0.50, 1.0)    # Very poor purity
        }
    
    def quantify_contamination(self, original_image: np.ndarray, 
                             mask: np.ndarray) -> Dict[str, Any]:
        """
        包括的混入率定量化分析
        
        Args:
            original_image: 元画像
            mask: 前景マスク
            
        Returns:
            Dict: 混入率分析結果
        """
        if original_image is None or mask is None:
            return self._generate_error_result("Invalid input images")
        
        if original_image.size == 0 or mask.size == 0:
            return self._generate_error_result("Empty input images")
        
        try:
            # 画像サイズ調整
            original_image, mask = self._align_image_sizes(original_image, mask)
            
            # 前景・背景領域抽出
            foreground_pixels, background_pixels = self._extract_regions(original_image, mask)
            
            # 色ベース混入分析
            color_contamination = self._analyze_color_contamination(
                original_image, mask, foreground_pixels, background_pixels
            )
            
            # テクスチャベース混入分析
            texture_contamination = self._analyze_texture_contamination(
                original_image, mask
            )
            
            # 空間的混入パターン分析
            spatial_contamination = self._analyze_spatial_contamination(
                original_image, mask
            )
            
            # エッジ混入分析
            edge_contamination = self._analyze_edge_contamination(
                original_image, mask
            )
            
            # ピクセルレベル純度分析
            purity_analysis = self._analyze_pixel_purity(
                original_image, mask, foreground_pixels, background_pixels
            )
            
            # 総合混入評価
            overall_assessment = self._calculate_overall_contamination(
                color_contamination, texture_contamination, spatial_contamination,
                edge_contamination, purity_analysis
            )
            
            return {
                'analysis_type': 'contamination_quantification',
                'image_info': {
                    'original_shape': original_image.shape,
                    'mask_shape': mask.shape,
                    'foreground_pixels': len(foreground_pixels) if foreground_pixels is not None else 0,
                    'background_pixels': len(background_pixels) if background_pixels is not None else 0
                },
                'color_contamination': color_contamination,
                'texture_contamination': texture_contamination,
                'spatial_contamination': spatial_contamination,
                'edge_contamination': edge_contamination,
                'purity_analysis': purity_analysis,
                'overall_assessment': overall_assessment,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': self.version
                }
            }
            
        except Exception as e:
            return self._generate_error_result(f"Contamination analysis failed: {str(e)}")
    
    def _align_image_sizes(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """画像サイズ調整"""
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # マスクをバイナリに正規化
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        
        return image, mask
    
    def _extract_regions(self, image: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """前景・背景領域ピクセル抽出"""
        try:
            mask_binary = mask > 0
            
            if len(image.shape) == 3:
                # カラー画像
                foreground_pixels = image[mask_binary]
                background_pixels = image[~mask_binary]
            else:
                # グレースケール画像
                foreground_pixels = image[mask_binary]
                background_pixels = image[~mask_binary]
            
            return foreground_pixels, background_pixels
            
        except Exception:
            return None, None
    
    def _analyze_color_contamination(self, image: np.ndarray, mask: np.ndarray,
                                   foreground_pixels: np.ndarray, 
                                   background_pixels: np.ndarray) -> Dict[str, Any]:
        """色ベース混入分析"""
        try:
            if foreground_pixels is None or background_pixels is None:
                return {'error': 'Invalid pixel data for color analysis'}
            
            if len(foreground_pixels) == 0 or len(background_pixels) == 0:
                return {'error': 'Empty foreground or background regions'}
            
            # 色統計計算
            if len(image.shape) == 3:
                fg_mean = np.mean(foreground_pixels, axis=0)
                bg_mean = np.mean(background_pixels, axis=0)
                fg_std = np.std(foreground_pixels, axis=0)
                bg_std = np.std(background_pixels, axis=0)
            else:
                fg_mean = np.mean(foreground_pixels)
                bg_mean = np.mean(background_pixels)
                fg_std = np.std(foreground_pixels)
                bg_std = np.std(background_pixels)
            
            # 色距離計算
            if len(image.shape) == 3:
                color_distance = np.linalg.norm(fg_mean - bg_mean)
            else:
                color_distance = abs(fg_mean - bg_mean)
            
            # 前景内の色変動（背景色混入指標）
            color_contamination_analysis = self._detect_color_contamination_in_foreground(
                image, mask, bg_mean
            )
            
            # 色ベース混入スコア
            color_separation_score = min(color_distance / 100, 1.0)  # 正規化
            contamination_score = 1.0 - color_separation_score
            
            return {
                'foreground_color_stats': {
                    'mean': fg_mean.tolist() if hasattr(fg_mean, 'tolist') else float(fg_mean),
                    'std': fg_std.tolist() if hasattr(fg_std, 'tolist') else float(fg_std)
                },
                'background_color_stats': {
                    'mean': bg_mean.tolist() if hasattr(bg_mean, 'tolist') else float(bg_mean),
                    'std': bg_std.tolist() if hasattr(bg_std, 'tolist') else float(bg_std)
                },
                'color_distance': float(color_distance),
                'contamination_analysis': color_contamination_analysis,
                'color_contamination_score': contamination_score,
                'color_grade': self._score_to_grade(contamination_score)
            }
            
        except Exception as e:
            return {'error': f'Color contamination analysis failed: {str(e)}'}
    
    def _detect_color_contamination_in_foreground(self, image: np.ndarray, 
                                                mask: np.ndarray, bg_color) -> Dict[str, Any]:
        """前景内の背景色混入検出"""
        try:
            mask_binary = mask > 0
            threshold = self.detection_params['color_similarity_threshold']
            
            if len(image.shape) == 3:
                # カラー画像での色距離計算
                color_diff = np.linalg.norm(image - bg_color, axis=2)
                contaminated_pixels = mask_binary & (color_diff < threshold)
            else:
                # グレースケール画像
                color_diff = np.abs(image - bg_color)
                contaminated_pixels = mask_binary & (color_diff < threshold)
            
            contamination_count = np.sum(contaminated_pixels)
            foreground_count = np.sum(mask_binary)
            
            contamination_ratio = contamination_count / foreground_count if foreground_count > 0 else 0
            
            return {
                'contaminated_pixel_count': int(contamination_count),
                'foreground_pixel_count': int(foreground_count),
                'contamination_ratio': float(contamination_ratio),
                'severity': self._assess_contamination_severity(contamination_ratio)
            }
            
        except Exception as e:
            return {'error': f'Color contamination detection failed: {str(e)}'}
    
    def _analyze_texture_contamination(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """テクスチャベース混入分析"""
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # テクスチャ特徴抽出（LBP風の簡易実装）
            texture_features = self._extract_texture_features(gray)
            
            # 前景・背景のテクスチャ統計
            mask_binary = mask > 0
            fg_texture = texture_features[mask_binary]
            bg_texture = texture_features[~mask_binary]
            
            if len(fg_texture) == 0 or len(bg_texture) == 0:
                return {'error': 'Insufficient texture data'}
            
            # テクスチャ類似度分析
            fg_texture_mean = np.mean(fg_texture)
            bg_texture_mean = np.mean(bg_texture)
            texture_distance = abs(fg_texture_mean - bg_texture_mean)
            
            # 前景内のテクスチャ一貫性
            fg_texture_std = np.std(fg_texture)
            texture_coherence = 1.0 / (1.0 + fg_texture_std * 0.1)
            
            # テクスチャベース混入スコア
            texture_contamination_score = 1.0 - min(texture_distance / 50, 1.0)
            
            return {
                'foreground_texture_stats': {
                    'mean': float(fg_texture_mean),
                    'std': float(fg_texture_std)
                },
                'background_texture_stats': {
                    'mean': float(bg_texture_mean),
                    'std': float(np.std(bg_texture))
                },
                'texture_distance': float(texture_distance),
                'texture_coherence': float(texture_coherence),
                'texture_contamination_score': texture_contamination_score,
                'texture_grade': self._score_to_grade(texture_contamination_score)
            }
            
        except Exception as e:
            return {'error': f'Texture contamination analysis failed: {str(e)}'}
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """簡易テクスチャ特徴抽出"""
        # ソーベルフィルタによるエッジ強度
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 局所標準偏差（テクスチャ指標）
        kernel = np.ones((3, 3), np.uint8)
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel/9)
        local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel/9)
        local_std = np.sqrt(local_variance)
        
        # 統合テクスチャ特徴
        texture_features = edge_magnitude + local_std
        
        return texture_features
    
    def _analyze_spatial_contamination(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """空間的混入パターン分析"""
        try:
            mask_binary = mask > 0
            
            # 連結成分分析
            if HAS_SCIPY:
                labeled_mask, num_components = ndimage.label(mask_binary)
                component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            else:
                # フォールバック
                contours, _ = cv2.findContours(mask_binary.astype(np.uint8) * 255, 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_components = len(contours)
                component_sizes = [cv2.contourArea(c) for c in contours] if contours else []
            
            # 空間的一貫性分析
            spatial_coherence = self._calculate_spatial_coherence(mask_binary)
            
            # 断片化分析
            fragmentation_analysis = self._analyze_fragmentation(component_sizes)
            
            # 空間的混入スコア
            spatial_contamination_score = 1.0 - spatial_coherence
            
            return {
                'component_analysis': {
                    'num_components': num_components,
                    'component_sizes': component_sizes,
                    'fragmentation_score': fragmentation_analysis['fragmentation_score']
                },
                'spatial_coherence': spatial_coherence,
                'spatial_contamination_score': spatial_contamination_score,
                'spatial_grade': self._score_to_grade(spatial_contamination_score)
            }
            
        except Exception as e:
            return {'error': f'Spatial contamination analysis failed: {str(e)}'}
    
    def _calculate_spatial_coherence(self, mask: np.ndarray) -> float:
        """空間的一貫性計算"""
        try:
            # 膨張・収縮による孤立点除去効果の測定
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            original_area = np.sum(mask)
            opened_area = np.sum(opened)
            
            # 一貫性スコア（形態学的処理による変化の少なさ）
            if original_area > 0:
                coherence = opened_area / original_area
            else:
                coherence = 0.0
            
            return min(coherence, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_fragmentation(self, component_sizes: List[int]) -> Dict[str, Any]:
        """断片化分析"""
        if not component_sizes:
            return {'fragmentation_score': 1.0, 'main_component_ratio': 0.0}
        
        total_area = sum(component_sizes)
        main_component_area = max(component_sizes)
        
        main_component_ratio = main_component_area / total_area if total_area > 0 else 0
        
        # 断片化スコア（主成分比率が低いほど断片化）
        fragmentation_score = 1.0 - main_component_ratio
        
        return {
            'fragmentation_score': fragmentation_score,
            'main_component_ratio': main_component_ratio,
            'total_components': len(component_sizes)
        }
    
    def _analyze_edge_contamination(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """エッジ混入分析"""
        try:
            # マスクエッジ抽出
            mask_binary = mask > 0
            kernel = np.ones((3, 3), np.uint8)
            
            # エッジ検出
            eroded = cv2.erode(mask_binary.astype(np.uint8), kernel, iterations=1)
            edge_mask = mask_binary & ~eroded.astype(bool)
            
            # エッジ拡張領域
            width = self.detection_params['edge_contamination_width']
            dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=width)
            edge_region = dilated.astype(bool)
            
            # エッジ領域の色分析
            if len(image.shape) == 3:
                edge_pixels = image[edge_region]
                if len(edge_pixels) > 0:
                    edge_color_var = np.var(edge_pixels.reshape(-1, 3), axis=0).mean()
                else:
                    edge_color_var = 0
            else:
                edge_pixels = image[edge_region]
                edge_color_var = np.var(edge_pixels) if len(edge_pixels) > 0 else 0
            
            # エッジ混入スコア（色変動が大きいほど混入の可能性）
            edge_contamination_score = min(edge_color_var / 1000, 1.0)
            
            return {
                'edge_pixel_count': int(np.sum(edge_mask)),
                'edge_region_pixel_count': int(np.sum(edge_region)),
                'edge_color_variance': float(edge_color_var),
                'edge_contamination_score': edge_contamination_score,
                'edge_grade': self._score_to_grade(edge_contamination_score)
            }
            
        except Exception as e:
            return {'error': f'Edge contamination analysis failed: {str(e)}'}
    
    def _analyze_pixel_purity(self, image: np.ndarray, mask: np.ndarray,
                            foreground_pixels: np.ndarray, 
                            background_pixels: np.ndarray) -> Dict[str, Any]:
        """ピクセルレベル純度分析"""
        try:
            if foreground_pixels is None or background_pixels is None:
                return {'error': 'Invalid pixel data for purity analysis'}
            
            # クラスタリングベース純度分析（sklearn利用可能時）
            if HAS_SKLEARN and len(foreground_pixels) > 10:
                purity_analysis = self._clustering_based_purity_analysis(
                    foreground_pixels, background_pixels
                )
            else:
                # フォールバック：統計ベース分析
                purity_analysis = self._statistical_purity_analysis(
                    foreground_pixels, background_pixels
                )
            
            return purity_analysis
            
        except Exception as e:
            return {'error': f'Pixel purity analysis failed: {str(e)}'}
    
    def _clustering_based_purity_analysis(self, fg_pixels: np.ndarray, 
                                        bg_pixels: np.ndarray) -> Dict[str, Any]:
        """クラスタリングベース純度分析"""
        try:
            # データ準備
            if len(fg_pixels.shape) == 1:
                fg_pixels = fg_pixels.reshape(-1, 1)
            if len(bg_pixels.shape) == 1:
                bg_pixels = bg_pixels.reshape(-1, 1)
            
            # サンプリング（計算効率のため）
            max_samples = 1000
            if len(fg_pixels) > max_samples:
                fg_indices = np.random.choice(len(fg_pixels), max_samples, replace=False)
                fg_sample = fg_pixels[fg_indices]
            else:
                fg_sample = fg_pixels
            
            if len(bg_pixels) > max_samples:
                bg_indices = np.random.choice(len(bg_pixels), max_samples, replace=False)
                bg_sample = bg_pixels[bg_indices]
            else:
                bg_sample = bg_pixels
            
            # K-meansクラスタリング
            n_clusters = min(3, len(fg_sample) // 10) if len(fg_sample) > 20 else 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            fg_clusters = kmeans.fit_predict(fg_sample)
            
            # シルエットスコア（クラスタ品質）
            if len(set(fg_clusters)) > 1:
                silhouette = silhouette_score(fg_sample, fg_clusters)
            else:
                silhouette = 0.0
            
            # 純度スコア（クラスタの一貫性）
            purity_score = max(silhouette, 0.0)
            contamination_score = 1.0 - purity_score
            
            return {
                'clustering_analysis': {
                    'n_clusters': n_clusters,
                    'silhouette_score': float(silhouette),
                    'cluster_distribution': [int(np.sum(fg_clusters == i)) for i in range(n_clusters)]
                },
                'purity_score': purity_score,
                'pixel_contamination_score': contamination_score,
                'purity_grade': self._score_to_grade(contamination_score),
                'analysis_method': 'clustering'
            }
            
        except Exception as e:
            return {'error': f'Clustering-based purity analysis failed: {str(e)}'}
    
    def _statistical_purity_analysis(self, fg_pixels: np.ndarray, 
                                   bg_pixels: np.ndarray) -> Dict[str, Any]:
        """統計ベース純度分析（フォールバック）"""
        try:
            # 統計的純度評価
            if len(fg_pixels.shape) > 1:
                fg_mean = np.mean(fg_pixels, axis=0)
                fg_std = np.std(fg_pixels, axis=0)
                fg_variance = np.mean(fg_std)
            else:
                fg_mean = np.mean(fg_pixels)
                fg_variance = np.std(fg_pixels)
            
            # 前景内の変動（小さいほど純粋）
            purity_score = 1.0 / (1.0 + fg_variance * 0.01)
            contamination_score = 1.0 - purity_score
            
            return {
                'statistical_analysis': {
                    'foreground_variance': float(fg_variance),
                    'foreground_mean': fg_mean.tolist() if hasattr(fg_mean, 'tolist') else float(fg_mean)
                },
                'purity_score': purity_score,
                'pixel_contamination_score': contamination_score,
                'purity_grade': self._score_to_grade(contamination_score),
                'analysis_method': 'statistical'
            }
            
        except Exception as e:
            return {'error': f'Statistical purity analysis failed: {str(e)}'}
    
    def _calculate_overall_contamination(self, color_analysis: Dict, texture_analysis: Dict,
                                       spatial_analysis: Dict, edge_analysis: Dict,
                                       purity_analysis: Dict) -> Dict[str, Any]:
        """総合混入評価計算"""
        scores = []
        weights = []
        
        # 色混入スコア
        if 'color_contamination_score' in color_analysis:
            scores.append(color_analysis['color_contamination_score'])
            weights.append(0.3)
        
        # テクスチャ混入スコア
        if 'texture_contamination_score' in texture_analysis:
            scores.append(texture_analysis['texture_contamination_score'])
            weights.append(0.2)
        
        # 空間混入スコア
        if 'spatial_contamination_score' in spatial_analysis:
            scores.append(spatial_analysis['spatial_contamination_score'])
            weights.append(0.2)
        
        # エッジ混入スコア
        if 'edge_contamination_score' in edge_analysis:
            scores.append(edge_analysis['edge_contamination_score'])
            weights.append(0.15)
        
        # ピクセル純度スコア
        if 'pixel_contamination_score' in purity_analysis:
            scores.append(purity_analysis['pixel_contamination_score'])
            weights.append(0.15)
        
        # 重み付き平均
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 1.0  # エラー時は最悪スコア
        
        # 信頼性評価
        available_metrics = len(scores)
        confidence = min(available_metrics / 5.0, 1.0)
        
        return {
            'overall_contamination_score': overall_score,
            'contamination_grade': self._score_to_grade(overall_score),
            'individual_scores': dict(zip(['color', 'texture', 'spatial', 'edge', 'purity'], scores)) if scores else {},
            'confidence': confidence,
            'available_metrics': available_metrics,
            'assessment': self._generate_contamination_assessment(overall_score)
        }
    
    def _assess_contamination_severity(self, ratio: float) -> str:
        """混入重要度評価"""
        if ratio <= 0.05:
            return "negligible"
        elif ratio <= 0.15:
            return "minor"
        elif ratio <= 0.30:
            return "moderate"
        elif ratio <= 0.50:
            return "significant"
        else:
            return "severe"
    
    def _score_to_grade(self, score: float) -> str:
        """スコアからグレードへの変換"""
        for grade, (min_val, max_val) in self.contamination_grades.items():
            if min_val <= score < max_val:
                return grade
        return 'F'
    
    def _generate_contamination_assessment(self, score: float) -> str:
        """混入評価コメント生成"""
        if score <= 0.05:
            return "excellent_purity"
        elif score <= 0.15:
            return "good_purity"
        elif score <= 0.30:
            return "acceptable_purity"
        elif score <= 0.50:
            return "poor_purity"
        else:
            return "very_poor_purity"
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        return {
            'error': error_message,
            'overall_assessment': {
                'overall_contamination_score': 1.0,
                'contamination_grade': 'F',
                'assessment': 'analysis_failed'
            }
        }


def main():
    """テスト実行"""
    print("🚀 P1-022: 混入率定量化システム テスト開始")
    
    # テスト用画像・マスク作成
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[:, :] = [135, 206, 235]  # 背景（青空）
    cv2.circle(test_image, (100, 100), 70, [255, 220, 177], -1)  # 前景（肌色）
    
    test_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_mask, (100, 100), 70, 255, -1)
    
    # 混入率分析実行
    quantifier = ContaminationQuantifier()
    result = quantifier.quantify_contamination(test_image, test_mask)
    
    print("\n📊 混入率分析結果:")
    if 'error' not in result:
        overall = result.get('overall_assessment', {})
        print(f"  総合混入スコア: {overall.get('overall_contamination_score', 0):.3f}")
        print(f"  混入グレード: {overall.get('contamination_grade', 'N/A')}")
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
    
    print(f"\n✅ [P1-022] 混入率定量化システム完了")


if __name__ == "__main__":
    main()