from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
Linux環境での実際のキャラクター抽出スクリプト
"""

import numpy as np
import cv2
import torch

import os
import sys
from pathlib import Path
from PIL import Image

# パス設定
sys.path.insert(0, get_path("data", Path(get_path("data", Path("/mnt/c/AItools/segment-anything").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/")))
sys.path.insert(0, get_path("data", Path(get_path("data", Path("/mnt/c/AItools/segment-anything/core").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/")))

# SAMとYOLOのインポート
from segment_anything import SamPredictor, sam_model_registry

from ultralytics import YOLO


def extract_character_real(input_path, output_path):
    """実際のキャラクター抽出処理"""
    
    print(f"Processing: {input_path}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # SAMモデル初期化
    sam_checkpoint = get_path("models", "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_checkpoint):
        print(f"SAM checkpoint not found: {sam_checkpoint}")
        return False
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    # YOLOモデル初期化
    yolo_model = YOLO(get_path("models", "yolov8x.pt"))
    
    # 画像読み込み
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Failed to load image: {input_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO検出（アニメキャラクター用に調整された閾値）
    results = yolo_model(image_rgb, conf=0.07)
    
    if len(results[0].boxes) == 0:
        print("No detection found")
        # 画像全体を使用
        h, w = image_rgb.shape[:2]
        box = np.array([0, 0, w, h])
    else:
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
    
    # マスク適用（キャラクター部分のみ抽出）
    masked_image = np.ones_like(image_rgb) * 255  # 白背景
    masked_image[best_mask] = image_rgb[best_mask]
    
    # バウンディングボックスでクロップ
    y_indices, x_indices = np.where(best_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        cropped_image = masked_image[y_min:y_max+1, x_min:x_max+1]
    else:
        cropped_image = masked_image
    
    # 保存
    output_image = Image.fromarray(cropped_image)
    output_image.save(str(output_path))
    print(f"Saved: {output_path}")
    
    return True

def main():
    # 既存ファイルを削除
    import shutil

    # P1-B003の既存ファイル削除
    p1_b003_dir = Path(get_path("output", "P1-B003/extraction"))
    for f in p1_b003_dir.glob("*.png"):
        f.unlink()
    
    # P1-023の既存ファイル削除
    p1_023_dir = Path(get_path("output", "P1-023/extraction"))
    for f in p1_023_dir.glob("*.png"):
        f.unlink()
    
    # P1-B003用に実際の抽出を実行
    input_dir = Path(get_path("data", "org", "kana05"))
    input_files = sorted(list(input_dir.glob("*.jpg")))[:3]
    
    print("=== P1-B003 実際のキャラクター抽出開始 ===")
    for i, input_file in enumerate(input_files):
        output_file = p1_b003_dir / f"extracted_character_{i:04d}.png"
        try:
            extract_character_real(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    # P1-023用に実際の抽出を実行
    print("\n=== P1-023 実際のキャラクター抽出開始 ===")
    for i, input_file in enumerate(input_files[3:6] if len(input_files) > 5 else input_files[:3]):
        output_file = p1_023_dir / f"stable_extracted_character_{i:04d}.png"
        try:
            extract_character_real(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    main()