#!/usr/bin/env python3
"""
Unit tests for SolidFillDetector in manga_preprocessing.py
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.preprocessing.manga_preprocessing import SolidFillDetector


class TestSolidFillDetector(unittest.TestCase):
    """SolidFillDetectorのユニットテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.detector = SolidFillDetector()
        
    def test_detector_initialization(self):
        """検出器の初期化テスト"""
        self.assertEqual(self.detector.min_region_size, 100)
        self.assertEqual(self.detector.uniformity_threshold, 0.85)
        self.assertEqual(self.detector.adaptive_threshold_window, 31)
        
    def test_detect_solid_regions_simple(self):
        """単純な画像でのソリッド領域検出テスト"""
        # 白背景に黒い矩形を配置
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)
        
        result = self.detector.detect_solid_regions(image)
        
        # 結果の構造確認
        self.assertIn('regions', result)
        self.assertIn('masks', result)
        self.assertIn('colors', result)
        self.assertIn('confidence', result)
        self.assertIn('total_regions', result)
        self.assertIn('total_solid_area', result)
        
        # 少なくとも1つの領域が検出されることを期待
        self.assertGreater(result['total_regions'], 0)
        self.assertGreater(result['confidence'], 0.0)
        
    def test_detect_solid_regions_multiple_colors(self):
        """複数色の領域検出テスト"""
        # 複数の色付き矩形を配置
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (20, 20), (80, 80), (255, 0, 0), -1)    # 青
        cv2.rectangle(image, (120, 20), (180, 80), (0, 255, 0), -1)  # 緑
        cv2.rectangle(image, (220, 20), (280, 80), (0, 0, 255), -1)  # 赤
        
        result = self.detector.detect_solid_regions(image)
        
        # 複数の領域が検出されることを期待
        self.assertGreaterEqual(result['total_regions'], 3)
        
        # 各領域の情報確認
        for region in result['regions']:
            self.assertIn('mask', region)
            self.assertIn('area', region)
            self.assertIn('centroid', region)
            self.assertIn('bbox', region)
            self.assertIn('uniformity', region)
            self.assertIn('color', region)
            self.assertGreater(region['uniformity'], 0.8)  # 高い均一性
            
    def test_evaluate_color_uniformity(self):
        """色の均一性評価テスト"""
        # 均一な色の画像
        uniform_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        uniformity = self.detector._evaluate_color_uniformity(uniform_image, mask)
        
        # 均一な画像は高いスコアを期待
        self.assertGreater(uniformity, 0.9)
        
        # ノイズの多い画像
        noisy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        uniformity_noisy = self.detector._evaluate_color_uniformity(noisy_image, mask)
        
        # ノイズ画像は低いスコアを期待
        self.assertLess(uniformity_noisy, 0.5)
        
    def test_remove_duplicate_regions(self):
        """重複領域除去テスト"""
        # 重複する領域を作成
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask1, (20, 20), (60, 60), 255, -1)
        cv2.rectangle(mask2, (25, 25), (65, 65), 255, -1)  # 大部分が重複
        
        regions = [
            {
                'mask': mask1,
                'area': np.sum(mask1 > 0),
                'centroid': (40, 40),
                'bbox': (20, 20, 40, 40),
                'uniformity': 0.9,
                'color': (100, 100, 100)
            },
            {
                'mask': mask2,
                'area': np.sum(mask2 > 0),
                'centroid': (45, 45),
                'bbox': (25, 25, 40, 40),
                'uniformity': 0.9,
                'color': (105, 105, 105)
            }
        ]
        
        unique_regions = self.detector._remove_duplicate_regions(regions)
        
        # 重複が除去されて1つになることを期待
        self.assertEqual(len(unique_regions), 1)
        
    def test_separate_background_foreground(self):
        """背景/前景分離テスト"""
        # テスト画像：端に大きな領域、中央に小さな領域
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # 画像端の大きな領域（背景として判定されるべき）
        edge_mask = np.zeros((200, 200), dtype=np.uint8)
        edge_mask[0:50, :] = 255  # 上端
        
        # 中央の小さな領域（前景として判定されるべき）
        center_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(center_mask, (100, 100), 30, 255, -1)
        
        regions = [
            {
                'mask': edge_mask,
                'area': np.sum(edge_mask > 0),
                'centroid': (100, 25),
                'bbox': (0, 0, 200, 50),
                'uniformity': 0.95,
                'color': (200, 200, 200)
            },
            {
                'mask': center_mask,
                'area': np.sum(center_mask > 0),
                'centroid': (100, 100),
                'bbox': (70, 70, 60, 60),
                'uniformity': 0.9,
                'color': (50, 50, 50)
            }
        ]
        
        result = self.detector.separate_background_foreground(image, regions)
        
        # 結果の構造確認
        self.assertIn('background_mask', result)
        self.assertIn('foreground_mask', result)
        self.assertIn('background_regions', result)
        self.assertIn('foreground_regions', result)
        self.assertIn('num_background', result)
        self.assertIn('num_foreground', result)
        
        # 分離結果の確認
        self.assertEqual(result['num_background'], 1)
        self.assertEqual(result['num_foreground'], 1)
        
    def test_is_background_region(self):
        """背景領域判定テスト"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # 端に近い領域（背景として判定されるべき）
        edge_region = {
            'mask': np.zeros((200, 200), dtype=np.uint8),
            'area': 10000,
            'centroid': (10, 100),  # 左端に近い
            'color': (200, 200, 200)
        }
        
        is_bg = self.detector._is_background_region(edge_region, image)
        self.assertTrue(is_bg)
        
        # 中央の領域（前景として判定されるべき）
        center_region = {
            'mask': np.zeros((200, 200), dtype=np.uint8),
            'area': 1000,
            'centroid': (100, 100),  # 中央
            'color': (50, 100, 150)  # 色に変化がある
        }
        
        is_bg = self.detector._is_background_region(center_region, image)
        self.assertFalse(is_bg)
        
    def test_refine_mask_edges(self):
        """マスクエッジ精密化テスト"""
        # ノイズのあるマスクを作成
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)
        
        # ランダムノイズを追加
        noise = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        noise_mask = cv2.bitwise_and(noise, cv2.bitwise_not(mask))
        noisy_mask = cv2.bitwise_or(mask, noise_mask[:20, :20])  # 一部にノイズ
        
        refined = self.detector._refine_mask_edges(noisy_mask)
        
        # 精密化後のマスクはよりスムーズになることを確認
        # （エッジの数が減少）
        edges_original = cv2.Canny(noisy_mask, 50, 150)
        edges_refined = cv2.Canny(refined, 50, 150)
        
        edge_pixels_original = np.sum(edges_original > 0)
        edge_pixels_refined = np.sum(edges_refined > 0)
        
        # 精密化によりエッジが減少することを期待
        self.assertLessEqual(edge_pixels_refined, edge_pixels_original)
        
    def test_grayscale_image_handling(self):
        """グレースケール画像の処理テスト"""
        # グレースケール画像
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.rectangle(gray_image, (30, 30), (70, 70), 0, -1)
        
        result = self.detector.detect_solid_regions(gray_image)
        
        # エラーなく処理されることを確認
        self.assertIsInstance(result, dict)
        self.assertGreater(result['total_regions'], 0)
        
    def test_empty_image(self):
        """空の画像（均一色）での処理テスト"""
        # 完全に白い画像
        white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.detector.detect_solid_regions(white_image)
        
        # 領域が検出される（全体が1つのソリッド領域）
        self.assertGreaterEqual(result['total_regions'], 1)
        
    def test_performance_large_image(self):
        """大きな画像での性能テスト"""
        # 1000x1000の画像
        large_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        
        # いくつかの大きな矩形を配置
        cv2.rectangle(large_image, (100, 100), (400, 400), (0, 0, 0), -1)
        cv2.rectangle(large_image, (600, 100), (900, 400), (128, 128, 128), -1)
        cv2.rectangle(large_image, (100, 600), (400, 900), (255, 0, 0), -1)
        
        import time
        start_time = time.time()
        result = self.detector.detect_solid_regions(large_image)
        processing_time = time.time() - start_time
        
        # 処理時間が8秒以内であることを確認
        self.assertLess(processing_time, 8.0)
        
        # 領域が検出されることを確認
        self.assertGreater(result['total_regions'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)