#!/usr/bin/env python3
"""
Integration Test for P1-004: スクリーントーン検出アルゴリズム強化統合
Phase 1対応: 改良版スクリーントーン検出システムの統合テスト
"""

import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'features/evaluation'))
sys.path.append(str(project_root / 'features/processing/preprocessing'))

from utils.enhanced_screentone_detector import (
    EnhancedScreentoneDetector,
    ScreentoneType,
    detect_screentone_enhanced
)
from manga_preprocessing import ScreentoneBoundaryProcessor


class TestScreentoneDetectorIntegration(unittest.TestCase):
    """スクリーントーン検出器統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        # 検出器の初期化
        self.enhanced_detector = EnhancedScreentoneDetector()
        self.legacy_processor = ScreentoneBoundaryProcessor()
        
        # 複雑なテスト画像作成
        self.complex_screentone = self._create_complex_screentone_image(256, 256)
        self.manga_style_image = self._create_manga_style_image(256, 256)
        self.mixed_pattern_image = self._create_mixed_pattern_image(256, 256)
    
    def _create_complex_screentone_image(self, height: int, width: int) -> np.ndarray:
        """複雑なスクリーントーン画像の生成"""
        image = np.ones((height, width), dtype=np.uint8) * 240
        
        # ドットパターン領域
        for y in range(0, height//2, 8):
            for x in range(0, width//2, 8):
                cv2.circle(image, (x+4, y+4), 2, 100, -1)
        
        # 線パターン領域
        for x in range(width//2, width, 6):
            cv2.rectangle(image, (x, height//2), (x+2, height), 150, -1)
        
        # グラデーション領域
        for x in range(width//2):
            intensity = int(200 + 50 * x / (width//2))
            image[height//2:, x] = intensity
        
        return image
    
    def _create_manga_style_image(self, height: int, width: int) -> np.ndarray:
        """漫画スタイル画像の生成"""
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        # キャラクター領域（白）
        cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
        
        # 背景スクリーントーン
        for y in range(0, height, 10):
            for x in range(0, width, 10):
                if not (width//4 <= x <= 3*width//4 and height//4 <= y <= 3*height//4):
                    cv2.circle(image, (x, y), 3, 180, -1)
        
        # 影の表現（細かいドット）
        shadow_area = image[height//2:3*height//4, width//4:width//2]
        for y in range(0, shadow_area.shape[0], 4):
            for x in range(0, shadow_area.shape[1], 4):
                cv2.circle(shadow_area, (x, y), 1, 200, -1)
        
        return image
    
    def _create_mixed_pattern_image(self, height: int, width: int) -> np.ndarray:
        """混合パターン画像の生成"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 背景
        image[:, :] = [240, 240, 240]
        
        # 複数のスクリーントーンパターンを混在
        # 左上: ドットパターン
        for y in range(0, height//2, 6):
            for x in range(0, width//2, 6):
                cv2.circle(image, (x+3, y+3), 2, (100, 100, 100), -1)
        
        # 右上: 線パターン
        for x in range(width//2, width, 4):
            cv2.rectangle(image, (x, 0), (x+1, height//2), (150, 150, 150), -1)
        
        # 左下: ノイズパターン
        noise_region = image[height//2:, :width//2]
        noise = np.random.randint(0, 50, noise_region.shape, dtype=np.uint8)
        image[height//2:, :width//2] = 200 - noise
        
        # 右下: グラデーション
        for x in range(width//2, width):
            for y in range(height//2, height):
                intensity = int(150 + 100 * x / width)
                image[y, x] = [intensity, intensity, intensity]
        
        return image
    
    def test_enhanced_vs_legacy_comparison(self):
        """改良版と従来版の比較テスト"""
        test_images = [
            ("complex_screentone", self.complex_screentone),
            ("manga_style", self.manga_style_image)
        ]
        
        comparison_results = []
        
        for name, test_image in test_images:
            # グレースケール変換（必要に応じて）
            if len(test_image.shape) == 3:
                gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = test_image
            
            # 改良版検出
            enhanced_result = self.enhanced_detector.detect_screentone(gray_image)
            
            # 従来版検出
            legacy_mask, legacy_conf = self.legacy_processor.detect_screentone_regions(gray_image)
            
            # 結果比較
            enhanced_coverage = enhanced_result.coverage_ratio
            legacy_coverage = np.sum(legacy_mask > 0) / legacy_mask.size
            
            comparison_results.append({
                'image_name': name,
                'enhanced_type': enhanced_result.screentone_type.value,
                'enhanced_confidence': enhanced_result.confidence,
                'enhanced_coverage': enhanced_coverage,
                'enhanced_quality': enhanced_result.quality_score,
                'legacy_confidence': legacy_conf,
                'legacy_coverage': legacy_coverage,
                'detection_improvement': enhanced_result.confidence - legacy_conf,
                'coverage_difference': enhanced_coverage - legacy_coverage
            })
        
        # 結果表示と検証
        print("\n=== Enhanced vs Legacy Screentone Detection Comparison ===")
        for result in comparison_results:
            print(f"\n{result['image_name']}:")
            print(f"  Enhanced: type={result['enhanced_type']}, "
                  f"conf={result['enhanced_confidence']:.3f}, "
                  f"coverage={result['enhanced_coverage']:.3f}, "
                  f"quality={result['enhanced_quality']:.3f}")
            print(f"  Legacy: conf={result['legacy_confidence']:.3f}, "
                  f"coverage={result['legacy_coverage']:.3f}")
            print(f"  Improvement: conf={result['detection_improvement']:+.3f}, "
                  f"coverage={result['coverage_difference']:+.3f}")
        
        # 改良版が従来版を上回ることを確認
        avg_conf_improvement = np.mean([r['detection_improvement'] for r in comparison_results])
        print(f"\nAverage confidence improvement: {avg_conf_improvement:+.3f}")
        
        # 少なくとも1つの画像で改良が見られることを確認
        self.assertTrue(any(r['detection_improvement'] > 0 for r in comparison_results),
                       "Enhanced detector should show improvement on at least one test image")
    
    def test_real_world_scenarios(self):
        """実世界シナリオのテスト"""
        scenarios = [
            {
                'name': 'high_density_dots',
                'creator': lambda: self._create_high_density_dot_pattern(128, 128)
            },
            {
                'name': 'fine_lines',
                'creator': lambda: self._create_fine_line_pattern(128, 128)
            },
            {
                'name': 'irregular_pattern',
                'creator': lambda: self._create_irregular_pattern(128, 128)
            },
            {
                'name': 'low_contrast',
                'creator': lambda: self._create_low_contrast_pattern(128, 128)
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            test_image = scenario['creator']()
            result = self.enhanced_detector.detect_screentone(test_image)
            
            results.append({
                'scenario': scenario['name'],
                'detected': result.has_screentone,
                'type': result.screentone_type.value,
                'confidence': result.confidence,
                'quality': result.quality_score,
                'reasoning': result.reasoning
            })
        
        # 結果表示
        print("\n=== Real World Scenario Tests ===")
        for result in results:
            print(f"{result['scenario']}: "
                  f"detected={result['detected']}, "
                  f"type={result['type']}, "
                  f"conf={result['confidence']:.3f}, "
                  f"quality={result['quality']:.3f}")
            print(f"  Reasoning: {result['reasoning']}")
        
        # 基本的な動作確認
        for result in results:
            # 信頼度と品質スコアが適切な範囲内であることを確認
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            self.assertGreaterEqual(result['quality'], 0.0)
            self.assertLessEqual(result['quality'], 1.0)
    
    def _create_high_density_dot_pattern(self, height: int, width: int) -> np.ndarray:
        """高密度ドットパターン"""
        image = np.ones((height, width), dtype=np.uint8) * 255
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                cv2.circle(image, (x+2, y+2), 1, 0, -1)
        return image
    
    def _create_fine_line_pattern(self, height: int, width: int) -> np.ndarray:
        """細線パターン"""
        image = np.ones((height, width), dtype=np.uint8) * 255
        for x in range(0, width, 3):
            cv2.line(image, (x, 0), (x, height), 100, 1)
        return image
    
    def _create_irregular_pattern(self, height: int, width: int) -> np.ndarray:
        """不規則パターン"""
        image = np.ones((height, width), dtype=np.uint8) * 220
        np.random.seed(42)  # 再現性のため
        for _ in range(200):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(1, 4)
            cv2.circle(image, (x, y), size, 150, -1)
        return image
    
    def _create_low_contrast_pattern(self, height: int, width: int) -> np.ndarray:
        """低コントラストパターン"""
        image = np.ones((height, width), dtype=np.uint8) * 200
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                cv2.circle(image, (x+4, y+4), 3, 190, -1)
        return image
    
    def test_performance_comparison(self):
        """性能比較テスト"""
        import time
        
        test_image = self.complex_screentone
        num_iterations = 5
        
        # 改良版の性能測定
        enhanced_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            result = self.enhanced_detector.detect_screentone(test_image)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
        
        # 従来版の性能測定
        legacy_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            mask, conf = self.legacy_processor.detect_screentone_regions(test_image)
            end_time = time.time()
            legacy_times.append(end_time - start_time)
        
        enhanced_avg = np.mean(enhanced_times)
        legacy_avg = np.mean(legacy_times)
        
        print(f"\n=== Performance Comparison ===")
        print(f"Enhanced detector: {enhanced_avg:.3f}s ± {np.std(enhanced_times):.3f}s")
        print(f"Legacy processor: {legacy_avg:.3f}s ± {np.std(legacy_times):.3f}s")
        print(f"Speed ratio: {enhanced_avg / legacy_avg:.2f}x")
        
        # 改良版が極端に遅くないことを確認（10倍以内）
        self.assertLess(enhanced_avg / legacy_avg, 10.0, 
                       "Enhanced detector should not be more than 10x slower")
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        edge_cases = [
            ("empty_image", np.zeros((64, 64), dtype=np.uint8)),
            ("white_image", np.ones((64, 64), dtype=np.uint8) * 255),
            ("single_pixel", np.zeros((1, 1), dtype=np.uint8)),
            ("small_image", np.random.randint(0, 255, (8, 8), dtype=np.uint8)),
            ("large_uniform", np.ones((512, 512), dtype=np.uint8) * 128)
        ]
        
        print("\n=== Edge Case Tests ===")
        
        for name, test_image in edge_cases:
            try:
                result = self.enhanced_detector.detect_screentone(test_image)
                
                # 基本的な整合性チェック
                self.assertIsInstance(result.has_screentone, bool)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                self.assertEqual(result.mask.shape, test_image.shape)
                
                print(f"{name}: detected={result.has_screentone}, "
                      f"type={result.screentone_type.value}, "
                      f"conf={result.confidence:.3f}")
                
            except Exception as e:
                self.fail(f"Edge case '{name}' caused unexpected error: {e}")
    
    def test_mask_quality_validation(self):
        """マスク品質の検証テスト"""
        # 既知のパターンでマスク品質を検証
        dot_image = self.complex_screentone
        result = self.enhanced_detector.detect_screentone(dot_image)
        
        mask = result.mask
        
        # マスクの基本特性
        self.assertEqual(mask.shape, dot_image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all((mask == 0) | (mask == 255)), 
                       "Mask should be binary (0 or 255)")
        
        # マスクの連続性（極端な断片化を避ける）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 最大連結成分のサイズ
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            total_mask_area = np.sum(mask > 0)
            
            # 最大連結成分が全体の10%以上を占めることを確認
            if total_mask_area > 0:
                largest_ratio = largest_area / total_mask_area
                self.assertGreater(largest_ratio, 0.1, 
                                 "Largest connected component should be significant")
                
                print(f"Mask quality: largest component ratio={largest_ratio:.3f}, "
                      f"total components={len(contours)}")
    
    def test_robustness_to_noise(self):
        """ノイズに対する頑健性テスト"""
        clean_image = self.complex_screentone
        
        # 異なるレベルのノイズを追加
        noise_levels = [0, 10, 20, 30]
        results = []
        
        for noise_level in noise_levels:
            if noise_level == 0:
                noisy_image = clean_image
            else:
                noise = np.random.normal(0, noise_level, clean_image.shape)
                noisy_image = np.clip(clean_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            result = self.enhanced_detector.detect_screentone(noisy_image)
            results.append({
                'noise_level': noise_level,
                'detected': result.has_screentone,
                'confidence': result.confidence,
                'quality': result.quality_score
            })
        
        # 結果表示
        print("\n=== Noise Robustness Test ===")
        for result in results:
            print(f"Noise level {result['noise_level']:2d}: "
                  f"detected={result['detected']}, "
                  f"conf={result['confidence']:.3f}, "
                  f"quality={result['quality']:.3f}")
        
        # 適度なノイズレベルでも検出性能が維持されることを確認
        clean_result = results[0]
        moderate_noise_result = results[2]  # noise_level=20
        
        # 中程度のノイズでも検出能力が大幅に低下しないことを確認
        confidence_drop = clean_result['confidence'] - moderate_noise_result['confidence']
        self.assertLess(confidence_drop, 0.5, 
                       "Confidence should not drop drastically with moderate noise")


if __name__ == '__main__':
    unittest.main()