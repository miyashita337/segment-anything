#!/usr/bin/env python3
"""
Manga-specific preprocessing utilities for handling effect lines and multi-panel layouts
漫画特有の前処理：エフェクト線除去とマルチコマ分割
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math


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


class MangaPreprocessor:
    """漫画前処理の統合クラス"""
    
    def __init__(self):
        self.effect_remover = EffectLineRemover()
        self.panel_splitter = MultiPanelSplitter()
    
    def preprocess_manga_image(self, image: np.ndarray, enable_effect_removal: bool = True, enable_panel_split: bool = True) -> Dict[str, Any]:
        """
        漫画画像の総合前処理
        
        Args:
            image: 入力画像
            enable_effect_removal: エフェクト線除去を有効化
            enable_panel_split: パネル分割を有効化
            
        Returns:
            処理結果の辞書
        """
        result = {
            'original_image': image,
            'processed_image': image.copy(),
            'panels': [],
            'effect_lines_detected': False,
            'effect_line_density': 0.0,
            'processing_stages': []
        }
        
        # エフェクト線除去
        if enable_effect_removal:
            lines, density = self.effect_remover.detect_effect_lines(image)
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