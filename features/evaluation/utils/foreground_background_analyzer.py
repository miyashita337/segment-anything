#!/usr/bin/env python3
"""
Foreground Background Analyzer - P1-021
背景・前景分離精度測定システム

背景混入を定量化し、抽出品質を向上
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass


@dataclass
class ColorCluster:
    """色クラスターの定義"""
    center: Tuple[int, int, int]  # RGB中心値
    pixel_count: int
    percentage: float
    variance: float


class ForegroundBackgroundAnalyzer:
    """
    背景・前景分離精度測定システム
    
    色彩分析による背景/前景判定と混入率の定量化
    """
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        
        # 分析パラメータ
        self.analysis_params = {
            'color_clusters': 8,           # K-meansクラスタ数
            'edge_threshold': 50,          # エッジ検出閾値
            'texture_window': 9,           # テクスチャ解析ウィンドウサイズ
            'uniformity_threshold': 30,    # 均一性閾値
            'contamination_threshold': 0.1  # 混入判定閾値
        }
    
    def analyze_separation_quality(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """分離品質の解析"""
        if image is None or mask is None:
            return {'error': '画像またはマスクが無効です'}
        
        # 画像形式の正規化
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return {'error': '画像は3チャンネルのカラー画像である必要があります'}
        
        # マスクの正規化
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 前景・背景領域の抽出
        foreground_analysis = self._analyze_foreground_region(image_rgb, binary_mask)
        background_analysis = self._analyze_background_region(image_rgb, binary_mask)
        
        # 境界分析
        boundary_analysis = self._analyze_boundary_region(image_rgb, binary_mask)
        
        # 混入率の計算
        contamination_analysis = self._calculate_contamination_rates(
            foreground_analysis, background_analysis, boundary_analysis
        )
        
        # 分離品質スコアの計算
        separation_score = self._calculate_separation_score(
            foreground_analysis, background_analysis, contamination_analysis
        )
        
        return {
            'foreground_analysis': foreground_analysis,
            'background_analysis': background_analysis,
            'boundary_analysis': boundary_analysis,
            'contamination_analysis': contamination_analysis,
            'separation_score': separation_score,
            'overall_assessment': self._generate_separation_assessment(separation_score, contamination_analysis)
        }
    
    def _analyze_foreground_region(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """前景領域の解析"""
        # 前景ピクセルの抽出
        foreground_pixels = image[mask > 0]
        
        if len(foreground_pixels) == 0:
            return {'error': '前景領域が見つかりません'}
        
        # 色彩統計
        color_stats = self._calculate_color_statistics(foreground_pixels)
        
        # 色クラスタリング
        color_clusters = self._perform_color_clustering(foreground_pixels)
        
        # テクスチャ解析
        texture_analysis = self._analyze_texture(image, mask, is_foreground=True)
        
        # エッジ密度
        edge_density = self._calculate_edge_density(image, mask, is_foreground=True)
        
        return {
            'pixel_count': len(foreground_pixels),
            'color_statistics': color_stats,
            'color_clusters': [self._cluster_to_dict(cluster) for cluster in color_clusters],
            'texture_analysis': texture_analysis,
            'edge_density': edge_density,
            'uniformity_score': self._calculate_uniformity_score(foreground_pixels)
        }
    
    def _analyze_background_region(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """背景領域の解析"""
        # 背景ピクセルの抽出
        background_pixels = image[mask == 0]
        
        if len(background_pixels) == 0:
            return {'error': '背景領域が見つかりません'}
        
        # 色彩統計
        color_stats = self._calculate_color_statistics(background_pixels)
        
        # 色クラスタリング
        color_clusters = self._perform_color_clustering(background_pixels)
        
        # テクスチャ解析
        texture_analysis = self._analyze_texture(image, mask, is_foreground=False)
        
        # エッジ密度
        edge_density = self._calculate_edge_density(image, mask, is_foreground=False)
        
        return {
            'pixel_count': len(background_pixels),
            'color_statistics': color_stats,
            'color_clusters': [self._cluster_to_dict(cluster) for cluster in color_clusters],
            'texture_analysis': texture_analysis,
            'edge_density': edge_density,
            'uniformity_score': self._calculate_uniformity_score(background_pixels)
        }
    
    def _analyze_boundary_region(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """境界領域の解析"""
        # 境界の抽出 (マスクのエッジ近傍)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = dilated - eroded
        
        boundary_pixels = image[boundary > 0]
        
        if len(boundary_pixels) == 0:
            return {'error': '境界領域が見つかりません'}
        
        # 境界の色彩解析
        color_stats = self._calculate_color_statistics(boundary_pixels)
        
        # 境界の急激な変化を解析
        gradient_analysis = self._analyze_color_gradients(image, boundary)
        
        return {
            'pixel_count': len(boundary_pixels),
            'color_statistics': color_stats,
            'gradient_analysis': gradient_analysis,
            'boundary_sharpness': self._calculate_boundary_sharpness(image, mask)
        }
    
    def _calculate_color_statistics(self, pixels: np.ndarray) -> Dict[str, Any]:
        """色彩統計の計算"""
        if len(pixels) == 0:
            return {}
        
        # RGB各チャンネルの統計
        r_stats = {
            'mean': float(np.mean(pixels[:, 0])),
            'std': float(np.std(pixels[:, 0])),
            'min': int(np.min(pixels[:, 0])),
            'max': int(np.max(pixels[:, 0]))
        }
        
        g_stats = {
            'mean': float(np.mean(pixels[:, 1])),
            'std': float(np.std(pixels[:, 1])),
            'min': int(np.min(pixels[:, 1])),
            'max': int(np.max(pixels[:, 1]))
        }
        
        b_stats = {
            'mean': float(np.mean(pixels[:, 2])),
            'std': float(np.std(pixels[:, 2])),
            'min': int(np.min(pixels[:, 2])),
            'max': int(np.max(pixels[:, 2]))
        }
        
        # 全体の明度・彩度
        hsv_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        brightness = float(np.mean(hsv_pixels[:, 2]))
        saturation = float(np.mean(hsv_pixels[:, 1]))
        
        return {
            'red': r_stats,
            'green': g_stats,
            'blue': b_stats,
            'brightness': brightness,
            'saturation': saturation,
            'color_variance': float(np.var(pixels.reshape(-1)))
        }
    
    def _perform_color_clustering(self, pixels: np.ndarray) -> List[ColorCluster]:
        """色クラスタリングの実行"""
        if len(pixels) < self.analysis_params['color_clusters']:
            # データが少ない場合は単純化
            unique_colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
            clusters = []
            total_pixels = len(pixels)
            
            for color, count in zip(unique_colors, counts):
                cluster = ColorCluster(
                    center=tuple(color.astype(int)),
                    pixel_count=int(count),
                    percentage=float(count / total_pixels * 100),
                    variance=0.0
                )
                clusters.append(cluster)
            
            return clusters[:self.analysis_params['color_clusters']]
        
        # K-meansクラスタリング
        try:
            data = pixels.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(
                data, self.analysis_params['color_clusters'], None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            clusters = []
            total_pixels = len(pixels)
            
            for i, center in enumerate(centers):
                cluster_pixels = data[labels.flatten() == i]
                pixel_count = len(cluster_pixels)
                
                if pixel_count > 0:
                    variance = float(np.var(cluster_pixels))
                    cluster = ColorCluster(
                        center=tuple(center.astype(int)),
                        pixel_count=pixel_count,
                        percentage=float(pixel_count / total_pixels * 100),
                        variance=variance
                    )
                    clusters.append(cluster)
            
            # ピクセル数で降順ソート
            clusters.sort(key=lambda x: x.pixel_count, reverse=True)
            return clusters
            
        except Exception:
            # フォールバック: シンプルな色分析
            mean_color = np.mean(pixels, axis=0)
            cluster = ColorCluster(
                center=tuple(mean_color.astype(int)),
                pixel_count=len(pixels),
                percentage=100.0,
                variance=float(np.var(pixels))
            )
            return [cluster]
    
    def _analyze_texture(self, image: np.ndarray, mask: np.ndarray, is_foreground: bool) -> Dict[str, Any]:
        """テクスチャ解析"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 対象領域のマスク
        region_mask = mask if is_foreground else (255 - mask)
        
        # Local Binary Pattern近似
        lbp_variance = self._calculate_lbp_variance(gray, region_mask)
        
        # エッジ方向性
        edge_orientation = self._calculate_edge_orientation(gray, region_mask)
        
        return {
            'lbp_variance': lbp_variance,
            'edge_orientation_variance': edge_orientation,
            'texture_complexity': self._classify_texture_complexity(lbp_variance)
        }
    
    def _calculate_lbp_variance(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Local Binary Pattern分散の計算"""
        # 簡易版LBP
        h, w = gray.shape
        lbp_values = []
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if mask[y, x] > 0:
                    center = gray[y, x]
                    neighbors = [
                        gray[y-1, x-1], gray[y-1, x], gray[y-1, x+1],
                        gray[y, x+1], gray[y+1, x+1], gray[y+1, x],
                        gray[y+1, x-1], gray[y, x-1]
                    ]
                    
                    lbp_code = 0
                    for i, neighbor in enumerate(neighbors):
                        if neighbor > center:
                            lbp_code += 2**i
                    
                    lbp_values.append(lbp_code)
        
        return float(np.var(lbp_values)) if lbp_values else 0.0
    
    def _calculate_edge_orientation(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """エッジ方向性の分散計算"""
        # Sobelフィルタでエッジ検出
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # エッジの角度計算
        angles = np.arctan2(sobel_y, sobel_x)
        
        # マスク領域のエッジ角度のみ抽出
        masked_angles = angles[mask > 0]
        
        return float(np.var(masked_angles)) if len(masked_angles) > 0 else 0.0
    
    def _classify_texture_complexity(self, lbp_variance: float) -> str:
        """テクスチャ複雑度の分類"""
        if lbp_variance < 100:
            return 'smooth'
        elif lbp_variance < 500:
            return 'moderate'
        else:
            return 'complex'
    
    def _calculate_edge_density(self, image: np.ndarray, mask: np.ndarray, is_foreground: bool) -> float:
        """エッジ密度の計算"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        region_mask = mask if is_foreground else (255 - mask)
        region_edges = edges & region_mask
        
        total_region_pixels = np.sum(region_mask > 0)
        edge_pixels = np.sum(region_edges > 0)
        
        return float(edge_pixels / total_region_pixels) if total_region_pixels > 0 else 0.0
    
    def _calculate_uniformity_score(self, pixels: np.ndarray) -> float:
        """均一性スコアの計算"""
        if len(pixels) == 0:
            return 0.0
        
        # 色の分散を基にした均一性
        color_variance = np.var(pixels.reshape(-1))
        
        # 0-1スケールに正規化 (分散が小さいほど均一性が高い)
        uniformity = 1.0 / (1.0 + color_variance / 1000.0)
        
        return float(uniformity)
    
    def _analyze_color_gradients(self, image: np.ndarray, boundary: np.ndarray) -> Dict[str, float]:
        """色グラデーションの解析"""
        # 境界での色変化の急激さを測定
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 境界領域での平均グラデーション
        boundary_gradients = gradient_magnitude[boundary > 0]
        
        if len(boundary_gradients) > 0:
            mean_gradient = float(np.mean(boundary_gradients))
            max_gradient = float(np.max(boundary_gradients))
            std_gradient = float(np.std(boundary_gradients))
        else:
            mean_gradient = max_gradient = std_gradient = 0.0
        
        return {
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'std_gradient': std_gradient
        }
    
    def _calculate_boundary_sharpness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """境界の鋭さ計算"""
        # マスクエッジの抽出
        edges = cv2.Canny(mask, 50, 150)
        
        # エッジ周辺での色変化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        
        edge_gradients = np.abs(gradient[edges > 0])
        
        return float(np.mean(edge_gradients)) if len(edge_gradients) > 0 else 0.0
    
    def _calculate_contamination_rates(self, fg_analysis: Dict, bg_analysis: Dict, boundary_analysis: Dict) -> Dict[str, Any]:
        """混入率の計算"""
        contamination = {}
        
        # 色相似性による混入検出
        fg_clusters = fg_analysis.get('color_clusters', [])
        bg_clusters = bg_analysis.get('color_clusters', [])
        
        if fg_clusters and bg_clusters:
            color_similarity = self._calculate_color_similarity(fg_clusters, bg_clusters)
            contamination['color_similarity'] = color_similarity
            contamination['high_similarity_risk'] = color_similarity > 0.7
        else:
            contamination['color_similarity'] = 0.0
            contamination['high_similarity_risk'] = False
        
        # テクスチャ類似性
        fg_texture = fg_analysis.get('texture_analysis', {})
        bg_texture = bg_analysis.get('texture_analysis', {})
        
        texture_similarity = self._calculate_texture_similarity(fg_texture, bg_texture)
        contamination['texture_similarity'] = texture_similarity
        contamination['texture_confusion_risk'] = texture_similarity > 0.8
        
        # 境界の曖昧さ
        boundary_sharpness = boundary_analysis.get('boundary_sharpness', 0)
        contamination['boundary_ambiguity'] = 1.0 - min(1.0, boundary_sharpness / 100.0)
        contamination['blurry_boundary_risk'] = contamination['boundary_ambiguity'] > 0.5
        
        # 総合混入リスク
        overall_risk = (
            color_similarity * 0.4 +
            texture_similarity * 0.3 +
            contamination['boundary_ambiguity'] * 0.3
        )
        contamination['overall_contamination_risk'] = float(overall_risk)
        contamination['contamination_level'] = self._classify_contamination_level(overall_risk)
        
        return contamination
    
    def _calculate_color_similarity(self, fg_clusters: List[Dict], bg_clusters: List[Dict]) -> float:
        """色類似性の計算"""
        if not fg_clusters or not bg_clusters:
            return 0.0
        
        max_similarity = 0.0
        
        for fg_cluster in fg_clusters[:3]:  # 上位3クラスタ
            fg_color = np.array(fg_cluster['center'])
            
            for bg_cluster in bg_clusters[:3]:
                bg_color = np.array(bg_cluster['center'])
                
                # ユークリッド距離ベースの類似性
                distance = np.linalg.norm(fg_color - bg_color)
                similarity = 1.0 - min(1.0, distance / (255.0 * np.sqrt(3)))
                
                max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    
    def _calculate_texture_similarity(self, fg_texture: Dict, bg_texture: Dict) -> float:
        """テクスチャ類似性の計算"""
        if not fg_texture or not bg_texture:
            return 0.0
        
        fg_complexity = fg_texture.get('texture_complexity', 'smooth')
        bg_complexity = bg_texture.get('texture_complexity', 'smooth')
        
        # 複雑度の類似性
        complexity_similarity = 1.0 if fg_complexity == bg_complexity else 0.5
        
        # LBP分散の類似性
        fg_lbp = fg_texture.get('lbp_variance', 0)
        bg_lbp = bg_texture.get('lbp_variance', 0)
        
        if fg_lbp + bg_lbp > 0:
            lbp_similarity = 1.0 - abs(fg_lbp - bg_lbp) / (fg_lbp + bg_lbp)
        else:
            lbp_similarity = 1.0
        
        return float((complexity_similarity + lbp_similarity) / 2.0)
    
    def _classify_contamination_level(self, risk_score: float) -> str:
        """混入レベルの分類"""
        if risk_score >= 0.8:
            return 'severe'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'moderate'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _calculate_separation_score(self, fg_analysis: Dict, bg_analysis: Dict, contamination_analysis: Dict) -> Dict[str, Any]:
        """分離品質スコアの計算"""
        # 前景・背景の区別度
        fg_uniformity = fg_analysis.get('uniformity_score', 0)
        bg_uniformity = bg_analysis.get('uniformity_score', 0)
        
        # 混入リスクの逆数
        contamination_risk = contamination_analysis.get('overall_contamination_risk', 1.0)
        separation_quality = 1.0 - contamination_risk
        
        # 境界の明確さ
        boundary_clarity = 1.0 - contamination_analysis.get('boundary_ambiguity', 1.0)
        
        # 重み付き総合スコア
        overall_score = (
            fg_uniformity * 0.2 +
            bg_uniformity * 0.2 +
            separation_quality * 0.4 +
            boundary_clarity * 0.2
        )
        
        return {
            'overall_score': float(overall_score),
            'separation_quality': float(separation_quality),
            'boundary_clarity': float(boundary_clarity),
            'foreground_uniformity': float(fg_uniformity),
            'background_uniformity': float(bg_uniformity),
            'quality_grade': self._grade_separation_quality(overall_score)
        }
    
    def _grade_separation_quality(self, score: float) -> str:
        """分離品質スコアをグレードに変換"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        elif score >= 0.5:
            return 'E'
        else:
            return 'F'
    
    def _generate_separation_assessment(self, separation_score: Dict, contamination_analysis: Dict) -> Dict[str, Any]:
        """分離評価の総合アセスメント"""
        overall_score = separation_score['overall_score']
        contamination_level = contamination_analysis['contamination_level']
        
        # 改善推奨事項
        recommendations = []
        
        if contamination_analysis['high_similarity_risk']:
            recommendations.append('improve_color_contrast')
        
        if contamination_analysis['texture_confusion_risk']:
            recommendations.append('enhance_texture_differentiation')
        
        if contamination_analysis['blurry_boundary_risk']:
            recommendations.append('sharpen_segmentation_boundary')
        
        if overall_score < 0.7:
            recommendations.append('review_segmentation_parameters')
        
        # 主要問題の特定
        primary_issues = []
        
        if contamination_level in ['severe', 'high']:
            primary_issues.append('high_contamination_risk')
        
        if separation_score['boundary_clarity'] < 0.6:
            primary_issues.append('unclear_boundaries')
        
        if separation_score['foreground_uniformity'] < 0.5:
            primary_issues.append('inconsistent_foreground')
        
        return {
            'overall_assessment': 'good' if overall_score >= 0.7 else 'needs_improvement',
            'contamination_level': contamination_level,
            'primary_issues': primary_issues,
            'recommendations': recommendations,
            'extraction_reliability': self._assess_extraction_reliability(overall_score, contamination_level)
        }
    
    def _assess_extraction_reliability(self, score: float, contamination_level: str) -> str:
        """抽出信頼性の評価"""
        if score >= 0.8 and contamination_level in ['minimal', 'low']:
            return 'high'
        elif score >= 0.6 and contamination_level in ['minimal', 'low', 'moderate']:
            return 'medium'
        else:
            return 'low'
    
    def _cluster_to_dict(self, cluster: ColorCluster) -> Dict[str, Any]:
        """ColorClusterを辞書に変換"""
        return {
            'center': cluster.center,
            'pixel_count': cluster.pixel_count,
            'percentage': cluster.percentage,
            'variance': cluster.variance
        }
    
    def analyze_extracted_image(self, original_image_path: str, extracted_image_path: str) -> Dict[str, Any]:
        """抽出画像の解析"""
        try:
            # 元画像読み込み
            original = cv2.imread(original_image_path)
            if original is None:
                return {'error': f'元画像の読み込みに失敗: {original_image_path}'}
            
            # 抽出画像読み込み
            extracted = cv2.imread(extracted_image_path)
            if extracted is None:
                return {'error': f'抽出画像の読み込みに失敗: {extracted_image_path}'}
            
            # アルファチャンネルからマスクを作成
            if extracted.shape[2] == 4:  # RGBA
                mask = extracted[:, :, 3]
            else:
                # RGB画像の場合、黒背景を除外してマスクを作成
                gray_extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_extracted, 10, 255, cv2.THRESH_BINARY)
            
            # 元画像をリサイズ（必要に応じて）
            if original.shape[:2] != extracted.shape[:2]:
                original = cv2.resize(original, (extracted.shape[1], extracted.shape[0]))
            
            # 分離品質解析
            analysis_result = self.analyze_separation_quality(original, mask)
            analysis_result['original_image_path'] = original_image_path
            analysis_result['extracted_image_path'] = extracted_image_path
            analysis_result['timestamp'] = datetime.now().isoformat()
            
            return analysis_result
            
        except Exception as e:
            return {'error': f'解析エラー: {str(e)}'}
    
    def print_analysis_summary(self, analysis_result: Dict[str, Any]):
        """解析結果のサマリー出力"""
        if 'error' in analysis_result:
            print(f"❌ {analysis_result['error']}")
            return
        
        print("\n" + "="*50)
        print("🎨 背景・前景分離品質分析結果")
        print("="*50)
        
        # 分離スコア
        separation_score = analysis_result.get('separation_score', {})
        print(f"📊 分離品質:")
        print(f"  総合スコア: {separation_score.get('overall_score', 0):.3f}")
        print(f"  品質グレード: {separation_score.get('quality_grade', 'unknown')}")
        print(f"  境界明確度: {separation_score.get('boundary_clarity', 0):.3f}")
        
        # 混入分析
        contamination = analysis_result.get('contamination_analysis', {})
        print(f"\n⚠️ 混入リスク:")
        print(f"  混入レベル: {contamination.get('contamination_level', 'unknown')}")
        print(f"  総合リスク: {contamination.get('overall_contamination_risk', 0):.3f}")
        print(f"  色類似性: {contamination.get('color_similarity', 0):.3f}")
        
        # 総合評価
        assessment = analysis_result.get('overall_assessment', {})
        print(f"\n🎯 総合評価:")
        print(f"  評価: {assessment.get('overall_assessment', 'unknown')}")
        print(f"  抽出信頼性: {assessment.get('extraction_reliability', 'unknown')}")
        
        issues = assessment.get('primary_issues', [])
        if issues:
            print(f"  主要問題: {', '.join(issues)}")
        
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"  推奨改善: {', '.join(recommendations)}")


def main():
    """メイン実行関数"""
    print("🚀 背景・前景分離精度測定システム開始")
    
    # テスト用のダミー画像・マスク作成
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 背景 (青)
    test_image[:, :] = [100, 150, 200]
    
    # 前景 (赤い円)
    cv2.circle(test_image, (50, 50), 30, [200, 100, 100], -1)
    
    # マスク作成
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_mask, (50, 50), 30, 255, -1)
    
    # 分析器初期化
    analyzer = ForegroundBackgroundAnalyzer()
    
    # テスト実行
    print("📊 テスト画像で背景・前景分離分析中...")
    analysis_result = analyzer.analyze_separation_quality(test_image, test_mask)
    
    # 結果出力
    analyzer.print_analysis_summary(analysis_result)
    
    print(f"\n✅ [P1-021] 背景・前景分離精度測定完了")


if __name__ == "__main__":
    main()