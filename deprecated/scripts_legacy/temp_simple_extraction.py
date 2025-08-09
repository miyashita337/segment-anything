#!/usr/bin/env python3
"""
簡易版キャラクター抽出スクリプト
sympyの循環インポート問題を回避するため最小限の実装
"""

import sys
import os
from pathlib import Path

# sympyのインポートを避けるため環境変数設定
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import cv2
import torch
from PIL import Image

# SAMとYOLOのインポート  
sys.path.insert(0, '/mnt/c/AItools/segment-anything')
sys.path.insert(0, '/mnt/c/AItools/segment-anything/core')
from core.segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

def extract_character(input_path, output_path):
    """単一画像からキャラクター抽出"""
    
    print(f"Processing: {input_path}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # SAMモデル初期化
    sam_checkpoint = "/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print(f"SAM checkpoint not found: {sam_checkpoint}")
        return False
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    # YOLOモデル初期化
    yolo_model = YOLO('yolov8x.pt')
    
    # 画像読み込み
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Failed to load image: {input_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO検出
    results = yolo_model(image_rgb, conf=0.07)
    
    if len(results[0].boxes) == 0:
        print("No detection found")
        return False
    
    # 最大の検出ボックスを使用
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
    max_idx = np.argmax(areas)
    box = boxes[max_idx]
    
    # SAMで精密セグメンテーション
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )
    
    # 最高スコアのマスクを選択
    best_mask = masks[np.argmax(scores)]
    
    # マスク適用
    masked_image = image_rgb.copy()
    masked_image[~best_mask] = [255, 255, 255]  # 白背景
    
    # 保存
    output_image = Image.fromarray(masked_image)
    output_image.save(str(output_path))
    print(f"Saved: {output_path}")
    
    return True

def main():
    # P1-B003用に3枚抽出
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana05")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-B003/extraction")
    
    input_files = list(input_dir.glob("*.jpg"))[:3]
    
    for i, input_file in enumerate(input_files):
        output_file = output_dir / f"real_extracted_{i:04d}.png"
        try:
            extract_character(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    # P1-023用に3枚抽出
    output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-023/extraction")
    
    for i, input_file in enumerate(input_files[3:6]):
        output_file = output_dir / f"stable_real_extracted_{i:04d}.png"
        try:
            extract_character(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    main()