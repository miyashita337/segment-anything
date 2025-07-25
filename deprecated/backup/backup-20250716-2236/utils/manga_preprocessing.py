#!/usr/bin/env python3
"""
Manga-specific preprocessing utilities for handling effect lines and multi-panel layouts
漫画特有の前処理：エフェクト線除去とマルチコマ分割
"""

import numpy as np
import cv2

import math
from typing import Any, Dict, List, Optional, Tuple


class EffectLineRemover:
    """エフェクト線除去処理"""
    
    def __init__(self):
        self.line_detection_params = {
            'rho': 1,
            'theta': np.pi / 180,
            'threshold': 50,
            'min_line_length': 30,
            'max_line_gap': 10
        }
    
    def detect_effect_lines(self, image: np.ndarray) -> Tuple[List[Tuple], float]:
        """
        エフェクト線を検出
        
        Args:
            image: 入力画像
            
        Returns:
            (lines, density): 検出した線のリストと密度
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # エッジ検出（エフェクト線は高コントラスト）
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 確率的ハフ変換で直線検出
        lines = cv2.HoughLinesP(
            edges,
            rho=self.line_detection_params['rho'],
            theta=self.line_detection_params['theta'],
            threshold=self.line_detection_params['threshold'],
            minLineLength=self.line_detection_params['min_line_length'],
            maxLineGap=self.line_detection_params['max_line_gap']
        )
        
        if lines is None:
            return [], 0.0
        
        # 線の密度を計算
        total_length = 0
        valid_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 短すぎる線は除外
            if length > 20:
                valid_lines.append((x1, y1, x2, y2))
                total_length += length
        
        # 画像サイズに対する線密度
        image_area = gray.shape[0] * gray.shape[1]
        density = total_length / image_area if image_area > 0 else 0
        
        return valid_lines, density
    
    def detect_radial_lines(self, image: np.ndarray, center: Optional[Tuple[int, int]] = None) -> List[Tuple]:
        """
        放射状エフェクト線を検出
        
        Args:
            image: 入力画像
            center: 放射中心（None の場合は画像中央）
            
        Returns:
            放射状線のリスト
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        if center is None:
            center = (w // 2, h // 2)
        
        # エッジ検出
        edges = cv2.Canny(gray, 30, 100)
        
        # 通常のハフ変換で直線検出
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
        
        if lines is None:
            return []
        
        radial_lines = []
        
        for line in lines:
            rho, theta = line[0]
            
            # 直線の方程式から点を計算
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # 直線が画像境界と交わる点を計算
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            
            # 直線が中心から放射状かチェック
            if self._is_radial_line(center, (x1, y1, x2, y2), tolerance=30):
                radial_lines.append((x1, y1, x2, y2))
        
        return radial_lines
    
    def _is_radial_line(self, center: Tuple[int, int], line: Tuple[int, int, int, int], tolerance: int = 30) -> bool:
        """直線が中心から放射状かチェック"""
        cx, cy = center
        x1, y1, x2, y2 = line
        
        # 直線から中心点までの距離を計算
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return dist < tolerance
    
    def create_effect_mask(self, image: np.ndarray) -> np.ndarray:
        """
        エフェクト線領域のマスクを作成
        
        Args:
            image: 入力画像
            
        Returns:
            エフェクト線マスク（白=エフェクト線）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # エフェクト線マスク初期化
        effect_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 通常の直線検出
        lines, density = self.detect_effect_lines(image)
        
        # 高密度の場合のみエフェクト線として処理
        if density > 0.01:  # 閾値は調整可能
            for x1, y1, x2, y2 in lines:
                cv2.line(effect_mask, (x1, y1), (x2, y2), 255, thickness=3)
        
        # 放射状線検出
        radial_lines = self.detect_radial_lines(image)
        for x1, y1, x2, y2 in radial_lines:
            cv2.line(effect_mask, (x1, y1), (x2, y2), 255, thickness=2)
        
        # モルフォロジー処理で線を太くする
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        effect_mask = cv2.dilate(effect_mask, kernel, iterations=1)
        
        return effect_mask
    
    def remove_effect_lines(self, image: np.ndarray) -> np.ndarray:
        """
        エフェクト線を除去した画像を生成
        
        Args:
            image: 入力画像
            
        Returns:
            エフェクト線除去後の画像
        """
        # エフェクト線マスク作成
        effect_mask = self.create_effect_mask(image)
        
        # インペインティングでエフェクト線を除去
        result = cv2.inpaint(image, effect_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return result


class MultiPanelSplitter:
    """マルチコマ分割処理"""
    
    def __init__(self):
        self.min_panel_area = 10000  # 最小パネル面積
        self.border_thickness = 5    # 境界線の太さ
    
    def detect_panel_borders(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        コマ境界を検出
        
        Args:
            image: 入力画像
            
        Returns:
            境界線のリスト (x1, y1, x2, y2)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 二値化（コマ境界は通常黒い線）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # エッジ検出
        edges = cv2.Canny(binary, 50, 150)
        
        # 長い直線を検出（コマ境界）
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(gray.shape) // 4,  # 画像の1/4以上の長さ
            maxLineGap=20
        )
        
        if lines is None:
            return []
        
        # 水平・垂直線のみを抽出
        borders = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 角度を計算
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 水平線または垂直線のみ
            if abs(angle) < 10 or abs(angle) > 170 or abs(abs(angle) - 90) < 10:
                borders.append((x1, y1, x2, y2))
        
        return borders
    
    def split_into_panels(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        画像を複数のコマに分割
        
        Args:
            image: 入力画像
            
        Returns:
            (panel_image, bbox) のリスト
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # 境界線検出
        borders = self.detect_panel_borders(image)
        
        if not borders:
            # 境界が見つからない場合は元画像をそのまま返す
            return [(image, (0, 0, w, h))]
        
        # グリッド分割を試行
        panels = self._grid_based_split(image, borders)
        
        # 最小面積フィルタリング
        valid_panels = []
        for panel_img, bbox in panels:
            x, y, w_panel, h_panel = bbox
            area = w_panel * h_panel
            
            if area >= self.min_panel_area:
                valid_panels.append((panel_img, bbox))
        
        # パネルが見つからない場合は元画像を返す
        if not valid_panels:
            return [(image, (0, 0, w, h))]
        
        return valid_panels
    
    def _grid_based_split(self, image: np.ndarray, borders: List[Tuple[int, int, int, int]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        グリッドベースでのコマ分割
        """
        h, w = image.shape[:2]
        
        # 水平・垂直の境界線を分離
        h_lines = []  # 水平線
        v_lines = []  # 垂直線
        
        for x1, y1, x2, y2 in borders:
            if abs(y2 - y1) < abs(x2 - x1):  # 水平線
                y_pos = (y1 + y2) // 2
                h_lines.append(y_pos)
            else:  # 垂直線
                x_pos = (x1 + x2) // 2
                v_lines.append(x_pos)
        
        # 境界位置をソート
        h_lines = sorted(set(h_lines))
        v_lines = sorted(set(v_lines))
        
        # 画像境界を追加
        h_lines = [0] + h_lines + [h]
        v_lines = [0] + v_lines + [w]
        
        # グリッドからパネルを抽出
        panels = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i + 1]
                x1, x2 = v_lines[j], v_lines[j + 1]
                
                # パネル領域を抽出
                panel_img = image[y1:y2, x1:x2]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                
                panels.append((panel_img, bbox))
        
        return panels
    
    def find_best_panel_for_character(self, panels: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], character_detection_func) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        キャラクター検出に最適なパネルを選択
        
        Args:
            panels: パネルのリスト
            character_detection_func: キャラクター検出関数
            
        Returns:
            最適なパネル、またはNone
        """
        best_panel = None
        best_score = 0
        
        for panel_img, bbox in panels:
            try:
                # キャラクター検出を試行
                result = character_detection_func(panel_img)
                
                # スコアが良い場合は選択
                if result.get('success', False):
                    score = result.get('combined_score', 0)
                    if score > best_score:
                        best_score = score
                        best_panel = (panel_img, bbox)
            except:
                continue
        
        return best_panel


class ScreentoneBoundaryProcessor:
    """スクリーントーン・モザイク境界問題処理"""
    
    def __init__(self):
        self.screentone_threshold = 0.3  # スクリーントーン検出閾値
        self.mosaic_threshold = 0.05     # モザイク検出閾値
    
    def detect_screentone_regions(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        スクリーントーン領域を検出
        
        Args:
            image: 入力画像
            
        Returns:
            (mask, confidence): スクリーントーンマスクと信頼度
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # FFTを使用した周期的パターン検出
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 高周波成分の検出
        center_h, center_w = h // 2, w // 2
        mask_size = min(50, h // 4, w // 4)
        
        # 周波数領域での特徴抽出
        high_freq_mask = np.zeros_like(magnitude_spectrum)
        high_freq_mask[center_h-mask_size:center_h+mask_size, 
                      center_w-mask_size:center_w+mask_size] = 1
        
        high_freq_power = np.mean(magnitude_spectrum * high_freq_mask)
        
        # スクリーントーン領域のマスク生成
        screentone_mask = np.zeros((h, w), dtype=np.uint8)
        
        if high_freq_power > 12.0:  # スクリーントーン検出閾値
            # 局所的なテクスチャ分析
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            local_variance = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # 分散が一定範囲内の領域をスクリーントーンとして検出
            variance_mean = np.mean(local_variance)
            variance_std = np.std(local_variance)
            
            screentone_regions = (local_variance > variance_mean - 0.5 * variance_std) & \
                               (local_variance < variance_mean + 0.5 * variance_std)
            
            screentone_mask[screentone_regions] = 255
            
            # モルフォロジー処理でノイズ除去
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            screentone_mask = cv2.morphologyEx(screentone_mask, cv2.MORPH_CLOSE, kernel)
        
        confidence = min(high_freq_power / 15.0, 1.0)
        return screentone_mask, confidence
    
    def detect_mosaic_regions(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        モザイク領域を検出
        
        Args:
            image: 入力画像
            
        Returns:
            (mask, confidence): モザイクマスクと信頼度
        """
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
        
        # モザイクマスク生成
        mosaic_mask = np.zeros_like(gray, dtype=np.uint8)
        
        if grid_ratio > self.mosaic_threshold:
            # 格子の交点周辺をモザイク領域として検出
            intersection_points = cv2.bitwise_and(horizontal_lines, vertical_lines)
            
            # 交点周辺を拡張
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mosaic_mask = cv2.dilate(intersection_points, kernel, iterations=2)
        
        confidence = min(grid_ratio / self.mosaic_threshold, 1.0)
        return mosaic_mask, confidence
    
    def create_boundary_enhancement_mask(self, image: np.ndarray) -> Dict[str, Any]:
        """
        境界問題対応のための拡張マスクを作成
        
        Args:
            image: 入力画像
            
        Returns:
            境界拡張情報の辞書
        """
        screentone_mask, screentone_conf = self.detect_screentone_regions(image)
        mosaic_mask, mosaic_conf = self.detect_mosaic_regions(image)
        
        # 統合境界問題マスク
        boundary_mask = cv2.bitwise_or(screentone_mask, mosaic_mask)
        
        # 境界拡張領域を計算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        enhanced_mask = cv2.dilate(boundary_mask, kernel, iterations=1)
        
        return {
            'screentone_mask': screentone_mask,
            'screentone_confidence': screentone_conf,
            'mosaic_mask': mosaic_mask,
            'mosaic_confidence': mosaic_conf,
            'boundary_mask': boundary_mask,
            'enhanced_mask': enhanced_mask,
            'has_boundary_issues': screentone_conf > 0.3 or mosaic_conf > 0.3,
            'issue_types': {
                'screentone': screentone_conf > 0.3,
                'mosaic': mosaic_conf > 0.3
            }
        }
    
    def apply_boundary_smoothing(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        境界部分の平滑化処理
        
        Args:
            image: 入力画像
            mask: 処理対象マスク
            
        Returns:
            平滑化済み画像
        """
        # ガウシアンブラーによる境界平滑化
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # マスク領域のみブラー適用
        result = image.copy()
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        
        result = result.astype(np.float32)
        blurred = blurred.astype(np.float32)
        
        result = result * (1 - mask_normalized) + blurred * mask_normalized
        
        return result.astype(np.uint8)


class MangaPreprocessor:
    """漫画前処理の統合クラス"""
    
    def __init__(self):
        self.effect_remover = EffectLineRemover()
        self.panel_splitter = MultiPanelSplitter()
        self.boundary_processor = ScreentoneBoundaryProcessor()
    
    def preprocess_manga_image(self, image: np.ndarray, enable_effect_removal: bool = True, 
                              enable_panel_split: bool = True, enable_boundary_processing: bool = True) -> Dict[str, Any]:
        """
        漫画画像の総合前処理（境界問題対応強化版）
        
        Args:
            image: 入力画像
            enable_effect_removal: エフェクト線除去を有効化
            enable_panel_split: パネル分割を有効化
            enable_boundary_processing: 境界問題処理を有効化
            
        Returns:
            処理結果の辞書
        """
        result = {
            'original_image': image,
            'processed_image': image.copy(),
            'panels': [],
            'effect_lines_detected': False,
            'effect_line_density': 0.0,
            'boundary_issues': {},
            'processing_stages': []
        }
        
        # 境界問題検出・処理（最初に実行）
        if enable_boundary_processing:
            boundary_info = self.boundary_processor.create_boundary_enhancement_mask(image)
            result['boundary_issues'] = boundary_info
            
            if boundary_info['has_boundary_issues']:
                # 境界問題がある場合は平滑化処理を適用
                result['processed_image'] = self.boundary_processor.apply_boundary_smoothing(
                    result['processed_image'], boundary_info['enhanced_mask']
                )
                result['processing_stages'].append('boundary_smoothing')
                
                # スクリーントーン/モザイク検出をログ
                if boundary_info['issue_types']['screentone']:
                    result['processing_stages'].append('screentone_detected')
                if boundary_info['issue_types']['mosaic']:
                    result['processing_stages'].append('mosaic_detected')
        
        # エフェクト線除去
        if enable_effect_removal:
            lines, density = self.effect_remover.detect_effect_lines(result['processed_image'])
            result['effect_line_density'] = density
            
            if density > 0.01:  # 閾値以上でエフェクト線処理
                result['effect_lines_detected'] = True
                result['processed_image'] = self.effect_remover.remove_effect_lines(result['processed_image'])
                result['processing_stages'].append('effect_line_removal')
        
        # パネル分割
        if enable_panel_split:
            panels = self.panel_splitter.split_into_panels(result['processed_image'])
            
            if len(panels) > 1:  # 複数パネル検出時
                result['panels'] = panels
                result['processing_stages'].append('panel_split')
        
        # パネルが見つからない場合は処理済み画像を単一パネルとして扱う
        if not result['panels']:
            h, w = result['processed_image'].shape[:2]
            result['panels'] = [(result['processed_image'], (0, 0, w, h))]
        
        return result
    
    def get_preprocessing_recommendations(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像に対する前処理推奨事項を生成
        
        Args:
            image: 入力画像
            
        Returns:
            推奨事項の辞書
        """
        # 境界問題の分析
        boundary_info = self.boundary_processor.create_boundary_enhancement_mask(image)
        
        # エフェクト線密度の計算
        _, effect_density = self.effect_remover.detect_effect_lines(image)
        
        # パネル分割の必要性チェック
        borders = self.panel_splitter.detect_panel_borders(image)
        
        recommendations = {
            'boundary_processing': {
                'recommended': boundary_info['has_boundary_issues'],
                'screentone_detected': boundary_info['issue_types']['screentone'],
                'mosaic_detected': boundary_info['issue_types']['mosaic'],
                'confidence': max(boundary_info['screentone_confidence'], boundary_info['mosaic_confidence'])
            },
            'effect_removal': {
                'recommended': effect_density > 0.01,
                'density': effect_density,
                'priority': 'high' if effect_density > 0.05 else 'medium'
            },
            'panel_split': {
                'recommended': len(borders) > 0,
                'detected_borders': len(borders),
                'priority': 'high' if len(borders) > 2 else 'low'
            },
            'processing_order': []
        }
        
        # 処理順序の推奨
        if recommendations['boundary_processing']['recommended']:
            recommendations['processing_order'].append('boundary_processing')
        if recommendations['effect_removal']['recommended']:
            recommendations['processing_order'].append('effect_removal')
        if recommendations['panel_split']['recommended']:
            recommendations['processing_order'].append('panel_split')
        
        return recommendations