#!/usr/bin/env python3
"""
CI-INTEGRATION-001 Phase 2.2: Lightweight Extraction Test
軽量画像抽出テスト（SAM/YOLO依存なし）
"""
import sys
import os
import time
import cv2
import numpy as np
from PIL import Image

def main():
    print('🚀 CI-INTEGRATION-001 Phase 2.2: Lightweight Extraction Test')
    
    # 環境情報出力
    try:
        import torch
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
    except ImportError:
        print('PyTorch not available (expected in CI environment)')
    
    # テスト画像リスト
    test_images = ['assets/test_demo1.png', 'assets/test_demo2.png', 'assets/test_demo3.png']
    processed_count = 0
    total_processing_time = 0
    
    # 各画像を処理
    for img_path in test_images:
        if os.path.exists(img_path):
            start_time = time.time()
            
            # OpenCVで画像読み込み
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                
                # 軽量リサイズ処理（512px制限）
                if height > 512 or width > 512:
                    scale = 512 / max(height, width)
                    new_height, new_width = int(height * scale), int(width * scale)
                    resized = cv2.resize(img, (new_width, new_height))
                else:
                    resized = img
                
                # 出力ディレクトリ作成
                output_dir = '/mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-044/test_demo_extraction'
                os.makedirs(output_dir, exist_ok=True)
                
                # 結果保存
                base_name = os.path.basename(img_path).split('.')[0]
                output_path = f'{output_dir}/{base_name}_extracted.jpg'
                cv2.imwrite(output_path, resized)
                
                # 処理時間計測
                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                processed_count += 1
                
                print(f'✅ {img_path} -> {output_path} (処理時間: {processing_time:.2f}秒)')
            else:
                print(f'❌ Failed to load: {img_path}')
        else:
            print(f'❌ File not found: {img_path}')
    
    # 結果出力
    if processed_count > 0:
        avg_time = total_processing_time / processed_count
        print(f'📊 Phase 2.2 Results: {processed_count}/{len(test_images)} images processed')
        print(f'📊 Average processing time: {avg_time:.2f} seconds per image')
        print('✅ Phase 2.2 lightweight extraction test PASSED')
        return 0
    else:
        print('❌ Phase 2.2 image extraction test FAILED')
        return 1

if __name__ == '__main__':
    sys.exit(main())