#!/usr/bin/env python3
"""
Text Detection Utilities
Text region detection and filtering for character extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Optional dependencies
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TextDetector:
    """
    テキスト検出クラス
    EasyOCR または OpenCV ベースの検出を提供
    """
    
    def __init__(self, use_easyocr: bool = True, languages: List[str] = ['ja', 'en']):
        """
        Initialize text detector
        
        Args:
            use_easyocr: Use EasyOCR if available
            languages: Languages for EasyOCR
        """
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.reader = None
        
        if self.use_easyocr:
            self._init_easyocr(languages)
        
        print(f"📝 Text detector initialized (EasyOCR: {self.use_easyocr})")
    
    def _init_easyocr(self, languages: List[str]):
        """Initialize EasyOCR reader"""
        if EASYOCR_AVAILABLE and TORCH_AVAILABLE:
            try:
                gpu_available = torch.cuda.is_available()
                self.reader = easyocr.Reader(languages, gpu=gpu_available)
                print("✅ EasyOCR初期化完了")
            except Exception as e:
                print(f"⚠️ EasyOCR初期化失敗: {e}")
                self.reader = None
        else:
            self.reader = None
    
    def detect_text_regions_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        EasyOCRを使用してテキスト領域を検出
        
        Args:
            image: 入力画像
            
        Returns:
            テキスト領域情報のリスト
        """
        if not self.reader:
            return []
        
        try:
            results = self.reader.readtext(image)
            text_regions = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 信頼度閾値
                    # バウンディングボックスからマスクを生成
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    points = np.array(bbox, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)
                    
                    # バウンディングボックスを [x, y, w, h] 形式に変換
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x, y = min(x_coords), min(y_coords)
                    w, h = max(x_coords) - x, max(y_coords) - y
                    
                    text_regions.append({
                        'bbox': [x, y, w, h],
                        'mask': mask,
                        'text': text,
                        'confidence': confidence,
                        'polygon': bbox
                    })
            
            return text_regions
            
        except Exception as e:
            print(f"⚠️ EasyOCRテキスト検出エラー: {e}")
            return []
    
    def detect_text_regions_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        OpenCVを使用してテキストらしい領域を検出
        
        Args:
            image: 入力画像
            
        Returns:
            テキスト領域情報のリスト
        """
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # エッジ検出
            edges = cv2.Canny(gray, 50, 150)
            
            # 水平・垂直方向のエッジの検出
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # テキストらしい特徴の検出
            text_features = cv2.bitwise_or(horizontal_edges, vertical_edges)
            
            # 膨張処理でテキスト領域を結合
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            text_features = cv2.dilate(text_features, kernel, iterations=2)
            
            # 輪郭検出
            contours, _ = cv2.findContours(text_features, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 最小面積フィルタ
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # アスペクト比フィルタ（テキストらしい形状）
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:  # テキストらしいアスペクト比
                        # マスク作成
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                        
                        text_regions.append({
                            'bbox': [x, y, w, h],
                            'mask': mask,
                            'text': '',  # OpenCVでは文字認識しない
                            'confidence': 0.7,  # 固定値
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return text_regions
            
        except Exception as e:
            print(f"⚠️ OpenCVテキスト検出エラー: {e}")
            return []
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        テキスト領域を検出
        
        Args:
            image: 入力画像
            
        Returns:
            テキスト領域情報のリスト
        """
        if self.use_easyocr and self.reader:
            return self.detect_text_regions_easyocr(image)
        else:
            return self.detect_text_regions_opencv(image)
    
    def has_significant_text(self, image: np.ndarray, threshold: float = 0.1) -> bool:
        """
        画像に重要なテキストが含まれているかチェック
        
        Args:
            image: 入力画像
            threshold: テキスト面積の閾値（画像全体に対する割合）
            
        Returns:
            重要なテキストが含まれているかどうか
        """
        text_regions = self.detect_text_regions(image)
        if not text_regions:
            return False
        
        # テキスト面積の合計を計算
        total_text_area = sum(np.sum(region['mask'] > 0) for region in text_regions)
        image_area = image.shape[0] * image.shape[1]
        text_ratio = total_text_area / image_area
        
        return text_ratio > threshold
    
    def calculate_text_density_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        指定された領域のテキスト密度スコアを計算
        
        Args:
            image: 入力画像
            bbox: バウンディングボックス [x, y, w, h]
            
        Returns:
            テキスト密度スコア (0.0-1.0, 高いほどテキストが多い)
        """
        try:
            x, y, w, h = bbox
            
            # 領域をクロップ
            height, width = image.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            roi = image[y1:y2, x1:x2]
            
            # テキスト検出
            text_regions = self.detect_text_regions(roi)
            
            if not text_regions:
                return 0.0
            
            # テキスト面積計算
            total_text_area = sum(np.sum(region['mask'] > 0) for region in text_regions)
            roi_area = roi.shape[0] * roi.shape[1]
            
            text_density = total_text_area / roi_area if roi_area > 0 else 0.0
            
            # 密度に基づくスコア計算
            if text_density > 0.15:  # 15%以上がテキスト
                return 0.8
            elif text_density > 0.1:  # 10%以上
                return 0.6
            elif text_density > 0.05:  # 5%以上
                return 0.3
            
            return text_density * 2  # 5%未満は線形スケール
            
        except Exception as e:
            print(f"⚠️ テキスト密度計算エラー: {e}")
            return 0.0
    
    def filter_text_heavy_masks(self, 
                               masks: List[Dict[str, Any]], 
                               image: np.ndarray,
                               max_text_density: float = 0.3) -> List[Dict[str, Any]]:
        """
        テキストが多すぎるマスクを除外
        
        Args:
            masks: マスクリスト
            image: 元画像
            max_text_density: 最大テキスト密度閾値
            
        Returns:
            フィルタ済みマスクリスト
        """
        filtered_masks = []
        
        for mask in masks:
            bbox = mask['bbox']
            text_density = self.calculate_text_density_score(image, bbox)
            
            # テキスト密度情報をマスクに追加
            mask_with_text = mask.copy()
            mask_with_text['text_density'] = text_density
            
            # フィルタリング
            if text_density <= max_text_density:
                filtered_masks.append(mask_with_text)
            else:
                print(f"🚫 テキスト密度が高いマスクを除外: {text_density:.3f}")
        
        return filtered_masks
    
    def get_text_free_regions(self, image: np.ndarray) -> np.ndarray:
        """
        テキストが含まれていない領域のマスクを取得
        
        Args:
            image: 入力画像
            
        Returns:
            テキストフリー領域のマスク
        """
        text_regions = self.detect_text_regions(image)
        
        # 全体マスクを作成（白で初期化）
        text_free_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # テキスト領域を黒で塗りつぶし
        for region in text_regions:
            text_free_mask = cv2.bitwise_and(text_free_mask, cv2.bitwise_not(region['mask']))
        
        return text_free_mask


def detect_text_with_opencv_fallback(image: np.ndarray) -> List[np.ndarray]:
    """
    OpenCVフォールバック付きテキスト検出
    
    Args:
        image: 入力画像
        
    Returns:
        テキスト領域マスクのリスト
    """
    detector = TextDetector(use_easyocr=True)
    text_regions = detector.detect_text_regions(image)
    
    return [region['mask'] for region in text_regions]


def is_text_heavy_region(image: np.ndarray, bbox: Tuple[int, int, int, int], threshold: float = 0.2) -> bool:
    """
    指定領域がテキストが多い領域かどうかを判定
    
    Args:
        image: 入力画像
        bbox: バウンディングボックス [x, y, w, h]
        threshold: テキスト密度閾値
        
    Returns:
        テキストが多い領域かどうか
    """
    detector = TextDetector(use_easyocr=False)  # OpenCVのみ使用（高速）
    text_density = detector.calculate_text_density_score(image, bbox)
    
    return text_density > threshold


if __name__ == "__main__":
    # Test text detection
    print("🧪 Text detection test starting...")
    
    detector = TextDetector(use_easyocr=True)
    
    # Create test image with text-like patterns
    test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add some horizontal lines (text-like)
    for i in range(50, 150, 20):
        cv2.rectangle(test_image, (50, i), (250, i + 5), (0, 0, 0), -1)
    
    # Test detection
    text_regions = detector.detect_text_regions(test_image)
    has_text = detector.has_significant_text(test_image)
    
    print(f"✅ Text detection test completed")
    print(f"   Detected {len(text_regions)} text regions")
    print(f"   Has significant text: {has_text}")
    print(f"   EasyOCR available: {EASYOCR_AVAILABLE}")
    
    # Test text density
    bbox = (40, 40, 220, 120)
    density = detector.calculate_text_density_score(test_image, bbox)
    print(f"   Text density in bbox: {density:.3f}")