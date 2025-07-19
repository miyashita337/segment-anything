#!/usr/bin/env python3
"""
P1-006: 高度なベタ塗り領域処理システム

ベタ塗り領域の検出精度を向上し、キャラクター抽出の品質を改善
"""

import numpy as np
import cv2

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# オプションインポート
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using fallback clustering")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class SolidFillRegion:
    """ベタ塗り領域の情報"""
    mask: np.ndarray
    color: Tuple[int, int, int]
    area: int
    uniformity: float
    region_type: str  # 'character', 'background', 'effect'
    boundary_quality: float
    connected_components: int


@dataclass
class SolidFillAnalysis:
    """ベタ塗り分析結果"""
    regions: List[SolidFillRegion]
    total_solid_area: int
    solid_fill_ratio: float
    dominant_colors: List[Tuple[int, int, int]]
    has_large_solid_areas: bool
    processing_recommendations: List[str]


class ColorUniformityAnalyzer:
    """色の均一性分析器"""
    
    def __init__(self):
        self.uniformity_threshold = 0.95
        self.min_region_size = 100  # 最小領域サイズ（ピクセル）
        
    def analyze_color_uniformity(self, image: np.ndarray) -> Dict[str, Any]:
        """画像の色均一性を分析"""
        # 複数の色空間で分析
        uniformity_rgb = self._analyze_uniformity_rgb(image)
        uniformity_hsv = self._analyze_uniformity_hsv(image)
        uniformity_lab = self._analyze_uniformity_lab(image)
        
        # 総合的な均一性評価
        overall_uniformity = {
            'rgb': uniformity_rgb,
            'hsv': uniformity_hsv,
            'lab': uniformity_lab,
            'combined_score': (uniformity_rgb['score'] + 
                             uniformity_hsv['score'] + 
                             uniformity_lab['score']) / 3.0
        }
        
        return overall_uniformity
    
    def _analyze_uniformity_rgb(self, image: np.ndarray) -> Dict[str, Any]:
        """RGB色空間での均一性分析"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 各チャンネルの統計量計算
        mean_colors = np.mean(image, axis=(0, 1))
        std_colors = np.std(image, axis=(0, 1))
        
        # 変動係数（CV）による均一性評価
        cv_values = std_colors / (mean_colors + 1e-6)
        uniformity_score = 1.0 - np.mean(cv_values)
        
        return {
            'score': max(0, uniformity_score),
            'mean': mean_colors.tolist(),
            'std': std_colors.tolist(),
            'cv': cv_values.tolist()
        }
    
    def _analyze_uniformity_hsv(self, image: np.ndarray) -> Dict[str, Any]:
        """HSV色空間での均一性分析"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 彩度と明度の均一性を重視
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        sat_uniformity = 1.0 - (np.std(saturation) / (np.mean(saturation) + 1e-6))
        val_uniformity = 1.0 - (np.std(value) / (np.mean(value) + 1e-6))
        
        # 色相は循環的なので特別な処理
        hue = hsv[:, :, 0]
        hue_uniformity = self._compute_circular_uniformity(hue)
        
        overall_score = (sat_uniformity + val_uniformity + hue_uniformity * 0.5) / 2.5
        
        return {
            'score': max(0, overall_score),
            'hue_uniformity': hue_uniformity,
            'saturation_uniformity': sat_uniformity,
            'value_uniformity': val_uniformity
        }
    
    def _analyze_uniformity_lab(self, image: np.ndarray) -> Dict[str, Any]:
        """LAB色空間での均一性分析"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L（明度）とa,b（色度）の均一性
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        l_uniformity = 1.0 - (np.std(l_channel) / (np.mean(l_channel) + 1e-6))
        a_uniformity = 1.0 - (np.std(a_channel) / 128.0)  # a,bは-128から127
        b_uniformity = 1.0 - (np.std(b_channel) / 128.0)
        
        overall_score = (l_uniformity * 0.5 + a_uniformity * 0.25 + b_uniformity * 0.25)
        
        return {
            'score': max(0, overall_score),
            'l_uniformity': l_uniformity,
            'a_uniformity': a_uniformity,
            'b_uniformity': b_uniformity
        }
    
    def _compute_circular_uniformity(self, hue: np.ndarray) -> float:
        """循環的な値（色相）の均一性計算"""
        # 色相を複素数として扱う
        hue_rad = hue * np.pi / 90.0  # OpenCVの色相は0-179
        complex_hue = np.exp(1j * hue_rad)
        
        # 平均方向
        mean_complex = np.mean(complex_hue)
        mean_angle = np.angle(mean_complex)
        
        # 角度差の計算
        angle_diffs = np.angle(complex_hue * np.exp(-1j * mean_angle))
        circular_std = np.sqrt(np.mean(angle_diffs ** 2))
        
        # 均一性スコア（標準偏差が小さいほど高い）
        uniformity = np.exp(-circular_std)
        
        return uniformity


class SolidFillDetector:
    """ベタ塗り領域検出器"""
    
    def __init__(self):
        self.uniformity_analyzer = ColorUniformityAnalyzer()
        self.min_area = 100
        self.uniformity_threshold = 0.85
        
    def detect_solid_fill_regions(self, image: np.ndarray) -> List[SolidFillRegion]:
        """ベタ塗り領域を検出"""
        regions = []
        
        # 1. 色によるクラスタリング
        clustered_regions = self._cluster_by_color(image)
        
        # 2. 各クラスタの均一性評価
        for cluster_mask, cluster_color in clustered_regions:
            # 領域の均一性分析
            masked_region = cv2.bitwise_and(image, image, mask=cluster_mask)
            uniformity = self.uniformity_analyzer.analyze_color_uniformity(masked_region)
            
            if uniformity['combined_score'] > self.uniformity_threshold:
                # 連結成分分析
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    cluster_mask, connectivity=8
                )
                
                # 大きな連結成分のみを対象
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    
                    if area >= self.min_area:
                        component_mask = (labels == label_id).astype(np.uint8) * 255
                        
                        # 領域タイプの判定
                        region_type = self._classify_region_type(
                            component_mask, image, cluster_color
                        )
                        
                        # 境界品質の評価
                        boundary_quality = self._evaluate_boundary_quality(component_mask)
                        
                        region = SolidFillRegion(
                            mask=component_mask,
                            color=cluster_color,
                            area=area,
                            uniformity=uniformity['combined_score'],
                            region_type=region_type,
                            boundary_quality=boundary_quality,
                            connected_components=1
                        )
                        
                        regions.append(region)
        
        # 3. 領域の統合処理
        merged_regions = self._merge_similar_regions(regions)
        
        return merged_regions
    
    def _cluster_by_color(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """色によるクラスタリング"""
        h, w = image.shape[:2]
        
        # 画像を1次元配列に変換
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)
            pixels = np.repeat(pixels, 3, axis=1)
        
        if SKLEARN_AVAILABLE:
            # DBSCAN クラスタリング
            clustering = DBSCAN(eps=10, min_samples=50).fit(pixels)
            labels = clustering.labels_
        else:
            # フォールバック：色の範囲によるグルーピング
            labels = self._fallback_color_clustering(pixels)
        
        # 各クラスタのマスクと代表色を抽出
        clustered_regions = []
        unique_labels = set(labels) - {-1}  # -1はノイズ
        
        for label in unique_labels:
            mask = (labels == label).reshape(h, w).astype(np.uint8) * 255
            cluster_pixels = pixels[labels == label]
            mean_color = tuple(map(int, np.mean(cluster_pixels, axis=0)))
            
            clustered_regions.append((mask, mean_color))
        
        return clustered_regions
    
    def _fallback_color_clustering(self, pixels: np.ndarray) -> np.ndarray:
        """scikit-learn非対応時のフォールバッククラスタリング（高速化版）"""
        # 大きな画像では処理を間引く
        max_samples = 10000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sampled_pixels = pixels[indices]
        else:
            indices = np.arange(len(pixels))
            sampled_pixels = pixels
        
        labels = np.full(len(pixels), -1, dtype=int)
        current_label = 0
        threshold = 20  # 閾値を緩めに
        
        # サンプル処理
        sampled_labels = np.full(len(sampled_pixels), -1, dtype=int)
        cluster_centers = []
        
        # より効率的なクラスタリング
        for i, pixel in enumerate(sampled_pixels):
            if sampled_labels[i] != -1:
                continue
            
            cluster_centers.append(pixel)
            
            # ベクトル化された距離計算
            distances = np.linalg.norm(
                sampled_pixels.astype(float) - pixel.astype(float), axis=1
            )
            close_pixels = distances < threshold
            sampled_labels[close_pixels] = current_label
            
            current_label += 1
            
            # クラスタ数制限
            if current_label > 10:  # より少なく
                break
        
        # 全ピクセルにラベルを割り当て
        if len(pixels) > max_samples:
            # 各ピクセルを最も近いクラスタ中心に割り当て
            for i, pixel in enumerate(pixels):
                if len(cluster_centers) == 0:
                    labels[i] = 0
                    continue
                    
                distances = [np.linalg.norm(pixel.astype(float) - center.astype(float)) 
                           for center in cluster_centers]
                labels[i] = np.argmin(distances)
        else:
            labels = sampled_labels
        
        return labels
    
    def _classify_region_type(self, mask: np.ndarray, image: np.ndarray, 
                            color: Tuple[int, int, int]) -> str:
        """領域タイプの分類"""
        # 位置情報による分類
        h, w = mask.shape
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) == 0:
            return 'unknown'
        
        # 重心位置
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        # 画像の端に近い場合は背景の可能性が高い
        edge_threshold = 0.1
        if (centroid_y < h * edge_threshold or centroid_y > h * (1 - edge_threshold) or
            centroid_x < w * edge_threshold or centroid_x > w * (1 - edge_threshold)):
            return 'background'
        
        # 色による分類
        # 黒に近い色は髪や影の可能性
        if np.mean(color) < 50:
            # 形状を考慮
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                aspect_ratio = self._compute_aspect_ratio(contour)
                
                # 縦長の黒い領域は髪の可能性
                if aspect_ratio > 1.5:
                    return 'character'
        
        # デフォルトはキャラクター
        return 'character'
    
    def _evaluate_boundary_quality(self, mask: np.ndarray) -> float:
        """境界品質の評価"""
        # エッジの滑らかさを評価
        edges = cv2.Canny(mask, 50, 150)
        
        # 境界の連続性
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 最大輪郭の周囲長と面積の比
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        area = cv2.contourArea(max_contour)
        
        if area == 0:
            return 0.0
        
        # 円形度（円に近いほど1.0）
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # エッジの断片化度
        edge_fragments = self._count_edge_fragments(edges)
        fragmentation = 1.0 / (1.0 + edge_fragments / 100.0)
        
        # 総合的な品質スコア
        quality = (circularity * 0.3 + fragmentation * 0.7)
        
        return min(1.0, quality)
    
    def _compute_aspect_ratio(self, contour: np.ndarray) -> float:
        """輪郭のアスペクト比計算"""
        x, y, w, h = cv2.boundingRect(contour)
        return h / (w + 1e-6)
    
    def _count_edge_fragments(self, edges: np.ndarray) -> int:
        """エッジの断片数をカウント"""
        # 連結成分の数を数える
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        return num_labels - 1  # 背景を除く
    
    def _merge_similar_regions(self, regions: List[SolidFillRegion]) -> List[SolidFillRegion]:
        """類似領域の統合"""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            # 統合候補を探す
            merge_candidates = [region1]
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # 色の類似性チェック
                color_diff = np.linalg.norm(
                    np.array(region1.color) - np.array(region2.color)
                )
                
                if color_diff < 20:  # 色が似ている
                    # 空間的な近接性チェック
                    if self._are_regions_adjacent(region1.mask, region2.mask):
                        merge_candidates.append(region2)
                        used.add(j)
            
            # 統合実行
            if len(merge_candidates) > 1:
                merged_region = self._merge_regions(merge_candidates)
                merged.append(merged_region)
            else:
                merged.append(region1)
        
        return merged
    
    def _are_regions_adjacent(self, mask1: np.ndarray, mask2: np.ndarray, 
                            distance_threshold: int = 5) -> bool:
        """2つの領域が隣接しているかチェック"""
        # 膨張処理で距離を評価
        kernel = np.ones((distance_threshold, distance_threshold), np.uint8)
        dilated_mask1 = cv2.dilate(mask1, kernel, iterations=1)
        
        # 重なりがあれば隣接と判定
        overlap = cv2.bitwise_and(dilated_mask1, mask2)
        return np.any(overlap > 0)
    
    def _merge_regions(self, regions: List[SolidFillRegion]) -> SolidFillRegion:
        """複数の領域を統合"""
        # マスクの統合
        merged_mask = np.zeros_like(regions[0].mask)
        for region in regions:
            merged_mask = cv2.bitwise_or(merged_mask, region.mask)
        
        # 色の平均
        colors = [region.color for region in regions]
        mean_color = tuple(map(int, np.mean(colors, axis=0)))
        
        # その他のプロパティ
        total_area = sum(region.area for region in regions)
        mean_uniformity = np.mean([region.uniformity for region in regions])
        mean_boundary_quality = np.mean([region.boundary_quality for region in regions])
        
        # 連結成分数の再計算
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
        
        return SolidFillRegion(
            mask=merged_mask,
            color=mean_color,
            area=total_area,
            uniformity=mean_uniformity,
            region_type=regions[0].region_type,  # 最初の領域のタイプを採用
            boundary_quality=mean_boundary_quality,
            connected_components=num_labels - 1
        )


class AdaptiveSolidFillProcessor:
    """適応的ベタ塗り処理器"""
    
    def __init__(self):
        self.edge_preservation_factor = 0.8
        self.smoothing_iterations = 2
        
    def process_solid_fill_regions(self, image: np.ndarray, 
                                 regions: List[SolidFillRegion]) -> Dict[str, Any]:
        """ベタ塗り領域の適応的処理"""
        processed_image = image.copy()
        processing_masks = {}
        
        for i, region in enumerate(regions):
            # 領域タイプに応じた処理
            if region.region_type == 'character':
                processed_region = self._process_character_solid(
                    image, region.mask, region.color
                )
            elif region.region_type == 'background':
                processed_region = self._process_background_solid(
                    image, region.mask, region.color
                )
            else:
                processed_region = self._process_effect_solid(
                    image, region.mask, region.color
                )
            
            # 処理結果の適用
            mask_3ch = cv2.cvtColor(region.mask, cv2.COLOR_GRAY2BGR)
            mask_norm = mask_3ch.astype(np.float32) / 255.0
            
            processed_image = processed_image.astype(np.float32)
            processed_region = processed_region.astype(np.float32)
            
            processed_image = processed_image * (1 - mask_norm) + processed_region * mask_norm
            processed_image = processed_image.astype(np.uint8)
            
            processing_masks[f'region_{i}'] = region.mask
        
        return {
            'processed_image': processed_image,
            'processing_masks': processing_masks,
            'num_regions_processed': len(regions)
        }
    
    def _process_character_solid(self, image: np.ndarray, mask: np.ndarray, 
                               color: Tuple[int, int, int]) -> np.ndarray:
        """キャラクター内ベタ塗りの処理"""
        # エッジを保持しながら滑らかに
        result = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 色の一貫性を強化
        if np.mean(color) < 50:  # 黒髪などの暗い部分
            # コントラストを調整して詳細を保持
            alpha = 1.2
            beta = -10
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result
    
    def _process_background_solid(self, image: np.ndarray, mask: np.ndarray, 
                                color: Tuple[int, int, int]) -> np.ndarray:
        """背景ベタ塗りの処理"""
        # より強い平滑化
        result = cv2.GaussianBlur(image, (7, 7), 1.5)
        
        # 単色化を促進
        mean_color = np.array(color).reshape(1, 1, 3)
        blend_factor = 0.7
        
        result = result.astype(np.float32)
        result = result * (1 - blend_factor) + mean_color * blend_factor
        
        return result.astype(np.uint8)
    
    def _process_effect_solid(self, image: np.ndarray, mask: np.ndarray, 
                            color: Tuple[int, int, int]) -> np.ndarray:
        """効果ベタ塗りの処理"""
        # 中程度の処理
        result = cv2.medianBlur(image, 5)
        return result


class EnhancedSolidFillProcessor:
    """統合された高度なベタ塗り処理システム"""
    
    def __init__(self):
        self.detector = SolidFillDetector()
        self.processor = AdaptiveSolidFillProcessor()
        self.uniformity_analyzer = ColorUniformityAnalyzer()
        
    def analyze_and_process(self, image: np.ndarray) -> Dict[str, Any]:
        """画像のベタ塗り領域を分析・処理"""
        results = {
            'original_image': image.copy(),
            'analysis': None,
            'processed_image': image.copy(),
            'processing_info': {
                'num_regions': 0,
                'total_solid_area': 0,
                'processing_applied': False
            }
        }
        
        # 1. ベタ塗り領域の検出
        solid_regions = self.detector.detect_solid_fill_regions(image)
        
        if not solid_regions:
            results['analysis'] = SolidFillAnalysis(
                regions=[],
                total_solid_area=0,
                solid_fill_ratio=0.0,
                dominant_colors=[],
                has_large_solid_areas=False,
                processing_recommendations=["ベタ塗り領域が検出されませんでした"]
            )
            return results
        
        # 2. 分析結果の作成
        total_area = sum(region.area for region in solid_regions)
        image_area = image.shape[0] * image.shape[1]
        solid_ratio = total_area / image_area
        
        # 支配的な色の抽出
        color_counts = {}
        for region in solid_regions:
            color_key = tuple(region.color)
            color_counts[color_key] = color_counts.get(color_key, 0) + region.area
        
        dominant_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        dominant_colors = [color for color, _ in dominant_colors]
        
        # 推奨処理の決定
        recommendations = self._generate_recommendations(solid_regions, solid_ratio)
        
        analysis = SolidFillAnalysis(
            regions=solid_regions,
            total_solid_area=total_area,
            solid_fill_ratio=solid_ratio,
            dominant_colors=dominant_colors,
            has_large_solid_areas=any(r.area > image_area * 0.1 for r in solid_regions),
            processing_recommendations=recommendations
        )
        
        results['analysis'] = analysis
        
        # 3. 適応的処理の実行
        if solid_ratio > 0.05:  # 5%以上がベタ塗りの場合処理
            processing_result = self.processor.process_solid_fill_regions(image, solid_regions)
            results['processed_image'] = processing_result['processed_image']
            results['processing_info'].update({
                'num_regions': len(solid_regions),
                'total_solid_area': total_area,
                'processing_applied': True,
                'processing_masks': processing_result['processing_masks']
            })
        
        return results
    
    def _generate_recommendations(self, regions: List[SolidFillRegion], 
                                ratio: float) -> List[str]:
        """処理推奨事項の生成"""
        recommendations = []
        
        if ratio > 0.3:
            recommendations.append("大面積のベタ塗り領域があります。境界処理を推奨")
        
        # 黒ベタの存在チェック
        black_regions = [r for r in regions if np.mean(r.color) < 50]
        if black_regions:
            recommendations.append("黒ベタ領域が検出されました。髪や影の可能性があります")
        
        # 境界品質の低い領域
        low_quality = [r for r in regions if r.boundary_quality < 0.5]
        if low_quality:
            recommendations.append("境界品質の低い領域があります。エッジ強化を推奨")
        
        # キャラクター領域の存在
        char_regions = [r for r in regions if r.region_type == 'character']
        if char_regions:
            recommendations.append(f"キャラクター内に{len(char_regions)}個のベタ塗り領域")
        
        return recommendations
    
    def get_processing_summary(self, results: Dict[str, Any]) -> str:
        """処理結果のサマリー生成"""
        if not results['analysis']:
            return "分析が実行されていません"
        
        analysis = results['analysis']
        info = results['processing_info']
        
        summary = f"""ベタ塗り領域処理完了:
- 検出領域数: {len(analysis.regions)}
- 総面積: {analysis.total_solid_area}ピクセル
- 画像に占める割合: {analysis.solid_fill_ratio:.1%}
- 支配的な色: {len(analysis.dominant_colors)}色
- 処理適用: {'Yes' if info['processing_applied'] else 'No'}
- 推奨事項: {len(analysis.processing_recommendations)}件"""
        
        return summary


def evaluate_solid_fill_processing(image_path: str, 
                                 save_results: bool = True) -> Dict[str, Any]:
    """ベタ塗り処理の評価関数"""
    import cv2

    from pathlib import Path

    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    # 処理実行
    processor = EnhancedSolidFillProcessor()
    results = processor.analyze_and_process(image)
    
    # 結果保存
    if save_results and results['analysis']:
        output_dir = Path(image_path).parent / "solid_fill_results"
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(image_path).stem
        
        # 元画像
        cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), results['original_image'])
        
        # 処理済み画像
        cv2.imwrite(str(output_dir / f"{base_name}_processed.jpg"), results['processed_image'])
        
        # 各ベタ塗り領域のマスク
        for i, region in enumerate(results['analysis'].regions):
            cv2.imwrite(
                str(output_dir / f"{base_name}_region_{i}_mask.jpg"), 
                region.mask
            )
        
        # 統合マスク（全ベタ塗り領域）
        if results['analysis'].regions:
            combined_mask = np.zeros_like(results['analysis'].regions[0].mask)
            for region in results['analysis'].regions:
                combined_mask = cv2.bitwise_or(combined_mask, region.mask)
            cv2.imwrite(
                str(output_dir / f"{base_name}_all_solid_fills.jpg"), 
                combined_mask
            )
    
    # サマリー追加
    summary = processor.get_processing_summary(results)
    results['summary'] = summary
    
    return results


if __name__ == "__main__":
    # テスト実行
    test_image = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07/test_single.jpg"
    
    try:
        results = evaluate_solid_fill_processing(test_image, save_results=True)
        print("=== P1-006 ベタ塗り領域処理テスト ===")
        print(results['summary'])
        
        if results['analysis']:
            print(f"\n詳細情報:")
            for i, region in enumerate(results['analysis'].regions):
                print(f"  領域{i+1}:")
                print(f"    色: RGB{region.color}")
                print(f"    面積: {region.area}ピクセル")
                print(f"    均一性: {region.uniformity:.3f}")
                print(f"    タイプ: {region.region_type}")
                print(f"    境界品質: {region.boundary_quality:.3f}")
        
    except Exception as e:
        print(f"エラー: {e}")