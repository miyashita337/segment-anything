#!/usr/bin/env python3
"""
P1-006: ベタ塗り領域処理改善の統合テスト

実際の画像を使ったベタ塗り領域処理の統合テスト
"""

import unittest
import cv2
import numpy as np
import sys
from pathlib import Path
import tempfile

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.evaluation.utils.enhanced_solid_fill_processor import (
    EnhancedSolidFillProcessor,
    evaluate_solid_fill_processing
)


class TestP1006Integration(unittest.TestCase):
    """P1-006統合テスト"""
    
    def setUp(self):
        self.processor = EnhancedSolidFillProcessor()
        self.test_image_path = "/mnt/c/AItools/lora/train/yado/org/kaname07"
        
    def test_real_image_solid_fill_processing(self):
        """実際の画像でのベタ塗り処理テスト"""
        # テスト画像の確認
        test_images = list(Path(self.test_image_path).glob("*.jpg"))[:3]
        
        if not test_images:
            self.skipTest(f"テスト画像が見つかりません: {self.test_image_path}")
        
        processing_results = []
        
        for image_path in test_images:
            with self.subTest(image=image_path.name):
                # 画像読み込み
                image = cv2.imread(str(image_path))
                self.assertIsNotNone(image, f"画像読み込み失敗: {image_path}")
                
                # ベタ塗り処理実行
                results = self.processor.analyze_and_process(image)
                
                # 基本構造確認
                self.assertIn('original_image', results)
                self.assertIn('analysis', results)
                self.assertIn('processed_image', results)
                self.assertIn('processing_info', results)
                
                # 画像サイズ保持確認
                self.assertEqual(results['processed_image'].shape, image.shape)
                
                # 分析結果確認
                analysis = results['analysis']
                if analysis:
                    self.assertIsInstance(analysis.total_solid_area, int)
                    self.assertIsInstance(analysis.solid_fill_ratio, float)
                    self.assertIsInstance(analysis.dominant_colors, list)
                    self.assertIsInstance(analysis.processing_recommendations, list)
                
                processing_results.append({
                    'image_name': image_path.name,
                    'has_analysis': analysis is not None,
                    'num_regions': len(analysis.regions) if analysis else 0,
                    'solid_ratio': analysis.solid_fill_ratio if analysis else 0.0,
                    'processing_applied': results['processing_info']['processing_applied']
                })
        
        # 結果サマリー表示
        print(f"\n=== P1-006 実画像ベタ塗り処理結果 ===")
        for result in processing_results:
            print(f"画像: {result['image_name']}")
            print(f"  分析実行: {result['has_analysis']}")
            print(f"  ベタ塗り領域数: {result['num_regions']}")
            print(f"  ベタ塗り比率: {result['solid_ratio']:.1%}")
            print(f"  処理適用: {result['processing_applied']}")
    
    def test_evaluation_function_integration(self):
        """評価関数の統合テスト"""
        # テスト画像を検索
        test_images = list(Path(self.test_image_path).glob("*.jpg"))[:1]
        
        if not test_images:
            self.skipTest(f"テスト画像が見つかりません: {self.test_image_path}")
        
        test_image = test_images[0]
        
        # 一時ディレクトリでテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = Path(temp_dir) / "test_image.jpg"
            
            # 画像をコピー
            image = cv2.imread(str(test_image))
            cv2.imwrite(str(temp_image_path), image)
            
            # 評価関数実行
            results = evaluate_solid_fill_processing(
                str(temp_image_path), 
                save_results=True
            )
            
            # 結果確認
            self.assertIn('summary', results)
            self.assertIsInstance(results['summary'], str)
            
            # 出力ディレクトリの確認
            output_dir = temp_image_path.parent / "solid_fill_results"
            if output_dir.exists():
                output_files = list(output_dir.glob("*.jpg"))
                # 何らかの出力ファイルが生成されることを確認
                self.assertGreaterEqual(len(output_files), 1)
    
    def test_solid_fill_types_coverage(self):
        """異なるベタ塗りタイプのカバレッジテスト"""
        # 人工的なベタ塗りパターンを作成してテスト
        solid_fill_types = [
            ("単色背景", self._create_solid_background(200, 200, (240, 240, 240))),
            ("黒髪風", self._create_black_hair_pattern(200, 200)),
            ("服装風", self._create_clothing_pattern(200, 200)),
            ("混合パターン", self._create_mixed_solid_pattern(200, 200)),
        ]
        
        detection_results = []
        
        for pattern_name, test_image in solid_fill_types:
            with self.subTest(pattern=pattern_name):
                results = self.processor.analyze_and_process(test_image)
                
                # エラーなく処理完了することを確認
                self.assertEqual(results['processed_image'].shape, test_image.shape)
                
                analysis = results['analysis']
                detection_info = {
                    'pattern': pattern_name,
                    'detected': analysis is not None and len(analysis.regions) > 0,
                    'num_regions': len(analysis.regions) if analysis else 0,
                    'solid_ratio': analysis.solid_fill_ratio if analysis else 0.0
                }
                
                detection_results.append(detection_info)
        
        # 結果表示
        print(f"\n=== ベタ塗りパターン検出結果 ===")
        for result in detection_results:
            print(f"{result['pattern']}: 検出={result['detected']}, "
                  f"領域数={result['num_regions']}, 比率={result['solid_ratio']:.1%}")
    
    def test_edge_case_handling(self):
        """エッジケースの処理テスト"""
        edge_cases = [
            ("極小画像", np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)),
            ("完全黒画像", np.zeros((100, 100, 3), dtype=np.uint8)),
            ("完全白画像", np.ones((100, 100, 3), dtype=np.uint8) * 255),
            ("縦長画像", np.random.randint(0, 255, (200, 50, 3), dtype=np.uint8)),
            ("横長画像", np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)),
        ]
        
        for case_name, test_image in edge_cases:
            with self.subTest(case=case_name):
                try:
                    results = self.processor.analyze_and_process(test_image)
                    
                    # エラーなく処理が完了することを確認
                    self.assertIn('processed_image', results)
                    self.assertEqual(results['processed_image'].shape, test_image.shape)
                    
                    print(f"  {case_name}: 処理成功")
                    
                except Exception as e:
                    self.fail(f"{case_name}の処理中にエラー: {e}")
    
    def test_performance_with_different_sizes(self):
        """異なるサイズでの性能テスト"""
        import time
        
        test_sizes = [(100, 100), (300, 300), (500, 500)]
        performance_results = []
        
        for width, height in test_sizes:
            # ベタ塗り要素を含むテスト画像生成
            test_image = self._create_performance_test_image(width, height)
            
            # 処理時間測定
            start_time = time.time()
            results = self.processor.analyze_and_process(test_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            analysis = results['analysis']
            
            performance_results.append({
                'size': f"{width}x{height}",
                'time': processing_time,
                'regions_found': len(analysis.regions) if analysis else 0,
                'pixels_per_second': (width * height) / processing_time
            })
            
            # 基本的な性能要件確認
            max_time = (width * height) / 5000  # 5000ピクセル/秒を想定
            self.assertLess(processing_time, max_time, 
                          f"処理時間が期待値を超過: {processing_time:.3f}s > {max_time:.3f}s")
        
        # 性能結果表示
        print(f"\n=== P1-006 性能ベンチマーク ===")
        for result in performance_results:
            print(f"サイズ: {result['size']}, 時間: {result['time']:.3f}s, "
                  f"領域数: {result['regions_found']}, "
                  f"処理速度: {result['pixels_per_second']:.0f}ピクセル/秒")
    
    def _create_solid_background(self, width: int, height: int, 
                               color) -> np.ndarray:
        """単色背景画像作成"""
        image = np.full((height, width, 3), color, dtype=np.uint8)
        return image
    
    def _create_black_hair_pattern(self, width: int, height: int) -> np.ndarray:
        """黒髪風パターン作成"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 200  # 明るい背景
        
        # 黒い縦長の楕円（髪風）
        cv2.ellipse(image, (width//2, height//3), (width//6, height//2), 
                   0, 0, 360, (20, 20, 20), -1)
        
        return image
    
    def _create_clothing_pattern(self, width: int, height: int) -> np.ndarray:
        """服装風パターン作成"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 背景
        
        # 青い服風の矩形
        cv2.rectangle(image, (width//4, height//2), (3*width//4, height-10), 
                     (100, 50, 200), -1)
        
        return image
    
    def _create_mixed_solid_pattern(self, width: int, height: int) -> np.ndarray:
        """混合ベタ塗りパターン作成"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景
        
        # 複数の単色領域
        cv2.rectangle(image, (10, 10), (width//3, height//3), (255, 0, 0), -1)  # 赤
        cv2.rectangle(image, (2*width//3, 10), (width-10, height//3), (0, 255, 0), -1)  # 緑
        cv2.rectangle(image, (width//4, 2*height//3), (3*width//4, height-10), (0, 0, 0), -1)  # 黒
        
        return image
    
    def _create_performance_test_image(self, width: int, height: int) -> np.ndarray:
        """性能テスト用画像作成"""
        # ベースとなるノイズ画像
        image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        
        # いくつかのベタ塗り領域を追加
        num_regions = min(5, (width * height) // 10000)  # サイズに応じて調整
        
        for i in range(num_regions):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(width//8, width//4)
            y2 = y1 + np.random.randint(height//8, height//4)
            
            x2 = min(x2, width-1)
            y2 = min(y2, height-1)
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        return image


if __name__ == '__main__':
    # 詳細な出力でテスト実行
    unittest.main(verbosity=2)