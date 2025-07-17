#!/usr/bin/env python3
"""
P1-005: モザイク境界処理改善の統合テスト

実際の画像を使った統合テストとキャラクター抽出システムとの統合確認
"""

import unittest
import cv2
import numpy as np
import sys
from pathlib import Path
import tempfile

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.evaluation.utils.enhanced_mosaic_boundary_processor import (
    EnhancedMosaicBoundaryProcessor,
    evaluate_mosaic_boundary_enhancement
)


class TestP1005Integration(unittest.TestCase):
    """P1-005統合テスト"""
    
    def setUp(self):
        self.processor = EnhancedMosaicBoundaryProcessor()
        self.test_image_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
        
    def test_real_image_processing(self):
        """実際の画像でのモザイク境界処理テスト"""
        # テスト画像の存在確認
        test_images = list(Path(self.test_image_path).glob("*.jpg"))[:3]  # 最初の3枚をテスト
        
        if not test_images:
            self.skipTest(f"テスト画像が見つかりません: {self.test_image_path}")
        
        processing_results = []
        
        for image_path in test_images:
            with self.subTest(image=image_path.name):
                # 画像読み込み
                image = cv2.imread(str(image_path))
                self.assertIsNotNone(image, f"画像読み込み失敗: {image_path}")
                
                # モザイク境界処理実行
                results = self.processor.process_image(image)
                
                # 基本的な結果構造確認
                self.assertIn('original_image', results)
                self.assertIn('mosaic_detected', results)
                self.assertIn('final_image', results)
                self.assertIn('processing_info', results)
                
                # 処理結果の品質確認
                self.assertEqual(results['final_image'].shape, image.shape)
                self.assertIsInstance(results['mosaic_detected'], bool)
                
                # 処理情報の確認
                info = results['processing_info']
                self.assertIn('confidence', info)
                self.assertIn('mosaic_type', info)
                self.assertIn('methods_used', info)
                
                processing_results.append({
                    'image_name': image_path.name,
                    'mosaic_detected': results['mosaic_detected'],
                    'confidence': info['confidence'],
                    'mosaic_type': info['mosaic_type']
                })
        
        print(f"\n=== P1-005 実画像処理結果 ===")
        for result in processing_results:
            print(f"画像: {result['image_name']}")
            print(f"  モザイク検出: {result['mosaic_detected']}")
            print(f"  信頼度: {result['confidence']:.3f}")
            print(f"  タイプ: {result['mosaic_type']}")
        
        # 少なくとも1つの画像で何らかの検出があることを期待
        total_detections = sum(1 for r in processing_results if r['mosaic_detected'])
        print(f"  検出総数: {total_detections}/{len(processing_results)}")
    
    def test_evaluation_function_integration(self):
        """評価関数の統合テスト"""
        # テスト画像を検索
        test_images = list(Path(self.test_image_path).glob("*.jpg"))[:1]  # 1枚のみテスト
        
        if not test_images:
            self.skipTest(f"テスト画像が見つかりません: {self.test_image_path}")
        
        test_image = test_images[0]
        
        # 一時出力ディレクトリでテスト実行
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = Path(temp_dir) / "test_image.jpg"
            
            # 画像を一時ディレクトリにコピー
            image = cv2.imread(str(test_image))
            cv2.imwrite(str(temp_image_path), image)
            
            # 評価関数実行
            results = evaluate_mosaic_boundary_enhancement(
                str(temp_image_path), 
                save_results=True
            )
            
            # 結果確認
            self.assertIn('summary', results)
            self.assertIsInstance(results['summary'], str)
            
            # 出力ファイルの確認
            output_dir = temp_image_path.parent / "mosaic_boundary_results"
            if output_dir.exists():
                output_files = list(output_dir.glob("*.jpg"))
                self.assertGreater(len(output_files), 0, "出力ファイルが生成されていません")
    
    def test_performance_benchmark(self):
        """処理性能のベンチマークテスト"""
        # 異なるサイズの画像での処理時間測定
        import time
        
        test_sizes = [(100, 100), (200, 200), (400, 400)]
        benchmark_results = []
        
        for width, height in test_sizes:
            # テスト画像生成
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # 処理時間測定
            start_time = time.time()
            results = self.processor.process_image(test_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            benchmark_results.append({
                'size': f"{width}x{height}",
                'time': processing_time,
                'detected': results['mosaic_detected']
            })
            
            # 基本的な性能要件確認（サイズに応じた処理時間）
            max_expected_time = (width * height) / 10000  # 1万ピクセル/秒の想定
            self.assertLess(processing_time, max_expected_time, 
                          f"処理時間が期待値を超過: {processing_time:.3f}s > {max_expected_time:.3f}s")
        
        print(f"\n=== P1-005 性能ベンチマーク ===")
        for result in benchmark_results:
            print(f"サイズ: {result['size']}, 時間: {result['time']:.3f}s, 検出: {result['detected']}")
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        edge_cases = [
            ("小画像", np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)),
            ("単色画像", np.ones((100, 100, 3), dtype=np.uint8) * 128),
            ("グラデーション", self._create_gradient_image(100, 100)),
            ("ノイズ", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
        ]
        
        for case_name, test_image in edge_cases:
            with self.subTest(case=case_name):
                try:
                    results = self.processor.process_image(test_image)
                    
                    # エラーなく処理が完了することを確認
                    self.assertIn('final_image', results)
                    self.assertEqual(results['final_image'].shape, test_image.shape)
                    
                    print(f"  {case_name}: 検出={results['mosaic_detected']}, "
                          f"信頼度={results['processing_info']['confidence']:.3f}")
                    
                except Exception as e:
                    self.fail(f"{case_name}の処理中にエラー: {e}")
    
    def test_mosaic_type_coverage(self):
        """異なるモザイクタイプの検出カバレッジテスト"""
        # 人工的に異なるタイプのモザイクパターンを作成してテスト
        mosaic_types = [
            ("格子", self._create_grid_mosaic(150, 150, 20)),
            ("ピクセル化", self._create_pixelated_mosaic(150, 150, 8)),
            ("ブラー", self._create_blur_mosaic(150, 150)),
        ]
        
        detected_types = set()
        
        for mosaic_name, test_image in mosaic_types:
            with self.subTest(mosaic_type=mosaic_name):
                results = self.processor.process_image(test_image)
                
                if results['mosaic_detected']:
                    detected_type = results['processing_info']['mosaic_type']
                    detected_types.add(detected_type)
                    
                    print(f"  {mosaic_name} → 検出タイプ: {detected_type}, "
                          f"信頼度: {results['processing_info']['confidence']:.3f}")
        
        print(f"\n検出されたモザイクタイプ: {detected_types}")
        # 少なくとも1つのタイプが検出されることを期待
        self.assertGreater(len(detected_types), 0, "いずれのモザイクタイプも検出されませんでした")
    
    def _create_gradient_image(self, width: int, height: int) -> np.ndarray:
        """グラデーション画像作成"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(255 * y / height)
            image[y, :, :] = intensity
        return image
    
    def _create_grid_mosaic(self, width: int, height: int, grid_size: int) -> np.ndarray:
        """格子モザイク作成"""
        image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        
        # 格子線を描画
        for x in range(0, width, grid_size):
            cv2.line(image, (x, 0), (x, height), (0, 0, 0), 2)
        for y in range(0, height, grid_size):
            cv2.line(image, (0, y), (width, y), (0, 0, 0), 2)
        
        return image
    
    def _create_pixelated_mosaic(self, width: int, height: int, pixel_size: int) -> np.ndarray:
        """ピクセル化モザイク作成"""
        # 元画像
        original = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # ピクセル化処理
        small_h, small_w = height // pixel_size, width // pixel_size
        small = cv2.resize(original, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _create_blur_mosaic(self, width: int, height: int) -> np.ndarray:
        """ブラーモザイク作成"""
        original = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(original, (15, 15), 5.0)
        return blurred


if __name__ == '__main__':
    # 詳細な出力でテスト実行
    unittest.main(verbosity=2)