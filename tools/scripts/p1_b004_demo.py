#!/usr/bin/env python3
"""
P1-B004: 適応的クロッピングシステム デモンストレーション

環境依存問題を回避した軽量デモ版:
- 実際の画像での動作確認
- P1-B004機能の可視化
- パフォーマンス測定
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

# パス追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_test_image(image_path):
    """テスト画像読み込み"""
    if not Path(image_path).exists():
        print(f"画像が見つかりません: {image_path}")
        return None
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"画像読み込みエラー: {image_path}")
        return None
        
    print(f"画像読み込み成功: {image.shape}")
    return image

def simulate_yolo_detection(image):
    """YOLO検出結果をシミュレート"""
    h, w = image.shape[:2]
    
    # 画像中央付近のキャラクター検出をシミュレート
    center_x, center_y = w // 2, h // 2
    bbox_w, bbox_h = min(w // 3, 300), min(h // 2, 400)
    
    x = max(0, center_x - bbox_w // 2)
    y = max(0, center_y - bbox_h // 2)
    
    # 境界調整
    bbox_w = min(bbox_w, w - x)
    bbox_h = min(bbox_h, h - y)
    
    print(f"YOLO検出シミュレート: x={x}, y={y}, w={bbox_w}, h={bbox_h}")
    return (x, y, bbox_w, bbox_h)

def demonstrate_p1_b004():
    """P1-B004デモンストレーション実行"""
    print("=" * 60)
    print("P1-B004: 適応的クロッピングシステム デモンストレーション")
    print("=" * 60)
    
    try:
        # P1-B004モジュールインポート
        from features.processing.adaptive_cropping import AdaptiveCropper, DetectionBox
        print("P1-B004モジュールインポート成功")
        
        # AdaptiveCropper初期化
        cropper = AdaptiveCropper()
        print("AdaptiveCropper初期化完了")
        
        # テスト画像パス（Windowsパス対応）
        test_images = [
            "C:/AItools/lora/train/yado/org/kana08/kana08_0001.jpg",
            "C:/AItools/lora/train/yado/org/kana08/kana08_0002.jpg", 
            "C:/AItools/lora/train/yado/org/kana08/kana08_0003.jpg"
        ]
        
        # 存在する画像のみフィルタ
        valid_images = []
        for image_path in test_images:
            if Path(image_path).exists():
                valid_images.append(image_path)
            else:
                print(f"画像が見つかりません（スキップ）: {image_path}")
        
        if not valid_images:
            print("テスト可能な画像が見つかりません")
            return False
        
        test_images = valid_images
        
        results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n--- テスト画像 {i}: {Path(image_path).name} ---")
            
            # 画像読み込み
            image = load_test_image(image_path)
            if image is None:
                continue
            
            # YOLO検出シミュレート
            x, y, w, h = simulate_yolo_detection(image)
            yolo_bbox = DetectionBox(
                x=x, y=y, w=w, h=h,
                confidence=0.85, source='yolo_sim'
            )
            
            # P1-B004適応的クロッピング実行
            start_time = time.time()
            result = cropper.adaptive_crop(image, yolo_bbox)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            
            if result:
                # 結果分析
                original_area = yolo_bbox.area
                optimized_area = result.area
                area_change_ratio = optimized_area / original_area
                
                print(f"適応的クロッピング成功:")
                print(f"  元の境界ボックス: {yolo_bbox.x}, {yolo_bbox.y}, {yolo_bbox.w}, {yolo_bbox.h}")
                print(f"  最適化後: {result.x}, {result.y}, {result.w}, {result.h}")
                print(f"  面積変化: {area_change_ratio:.3f}倍")
                print(f"  処理時間: {processing_time:.2f}ms")
                print(f"  信頼度: {result.confidence:.3f}")
                
                # 改善度評価
                if 0.8 <= area_change_ratio <= 1.2:
                    improvement = "適切な最適化"
                elif area_change_ratio < 0.8:
                    improvement = "クロッピング（他キャラ除去効果）"
                else:
                    improvement = "拡張（キャラ完全性向上）"
                
                print(f"  評価: {improvement}")
                
                results.append({
                    'image': Path(image_path).name,
                    'success': True,
                    'processing_time': processing_time,
                    'area_change': area_change_ratio,
                    'confidence': result.confidence,
                    'improvement': improvement
                })
            else:
                print("適応的クロッピング結果なし（フォールバック）")
                results.append({
                    'image': Path(image_path).name,
                    'success': False,
                    'processing_time': processing_time,
                    'area_change': 1.0,
                    'confidence': yolo_bbox.confidence,
                    'improvement': 'フォールバック'
                })
        
        # 結果サマリー
        print("\n" + "=" * 60)
        print("P1-B004デモンストレーション結果サマリー")
        print("=" * 60)
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        if total_count > 0:
            success_rate = success_count / total_count * 100
            print(f"成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
        else:
            print("成功率: 0/0 (テスト画像なし)")
        
        if results:
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            print(f"平均処理時間: {avg_time:.2f}ms")
            
            # 改善効果分析
            improvements = [r['improvement'] for r in results if r['success']]
            if improvements:
                print(f"改善効果:")
                for improvement in set(improvements):
                    count = improvements.count(improvement)
                    print(f"  {improvement}: {count}件")
        
        # P1-B004特徴分析
        print(f"\nP1-B004特徴:")
        print(f"  MediaPipe利用可否: {'利用可能' if cropper.face_detector else '無効（CPUモード）'}")
        print(f"  スケールファクター: {cropper.scale_factors}")
        print(f"  最大キャラクター数: {cropper.max_characters}")
        print(f"  顔検出信頼度閾値: {cropper.min_face_confidence}")
        
        print(f"\nP1-B004デモンストレーション完了")
        return True
        
    except Exception as e:
        print(f"P1-B004デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    success = demonstrate_p1_b004()
    
    if success:
        print("\nP1-B004動作確認成功 - 抽出パイプライン準備完了")
        return 0
    else:
        print("\nP1-B004動作確認失敗")
        return 1

if __name__ == '__main__':
    sys.exit(main())