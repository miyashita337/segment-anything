#!/usr/bin/env python3
"""
kana08用シンプルバッチ抽出スクリプト
既存のSAM/YOLOシステムを直接使用
"""

import numpy as np
import cv2
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import logging
import time
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_character_simple(image_path: Path, sam_generator, yolo_model, output_path: Path):
    """シンプルなキャラクター抽出"""
    try:
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return False, "画像読み込み失敗"
        
        # YOLO検出
        results = yolo_model(image, conf=0.07)
        
        if not results or len(results[0].boxes) == 0:
            return False, "キャラクター未検出"
        
        # 最大の検出結果を使用
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[largest_idx].astype(int)
        
        # 境界ボックスを少し拡張
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # SAMでセグメンテーション
        masks = sam_generator.generate(image)
        
        # 境界ボックス内で最大のマスクを選択
        best_mask = None
        best_area = 0
        
        for mask in masks:
            mask_binary = mask['segmentation']
            # マスクの境界ボックス取得
            y_indices, x_indices = np.where(mask_binary)
            if len(x_indices) == 0:
                continue
            
            mask_x1 = np.min(x_indices)
            mask_y1 = np.min(y_indices)
            mask_x2 = np.max(x_indices)
            mask_y2 = np.max(y_indices)
            
            # YOLOボックスとの重複確認
            overlap_x1 = max(x1, mask_x1)
            overlap_y1 = max(y1, mask_y1)
            overlap_x2 = min(x2, mask_x2)
            overlap_y2 = min(y2, mask_y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                yolo_area = (x2 - x1) * (y2 - y1)
                overlap_ratio = overlap_area / yolo_area
                
                if overlap_ratio > 0.5 and mask['area'] > best_area:
                    best_mask = mask
                    best_area = mask['area']
        
        if best_mask is None:
            # SAMが失敗した場合、YOLOボックスをそのまま使用
            extracted = image[y1:y2, x1:x2]
        else:
            # マスクを適用
            mask_binary = best_mask['segmentation'].astype(np.uint8) * 255
            # 背景を黒にする
            result = cv2.bitwise_and(image, image, mask=mask_binary)
            
            # マスクの境界で切り抜き
            y_indices, x_indices = np.where(mask_binary > 0)
            if len(x_indices) > 0:
                x_min = np.min(x_indices)
                x_max = np.max(x_indices)
                y_min = np.min(y_indices)
                y_max = np.max(y_indices)
                extracted = result[y_min:y_max+1, x_min:x_max+1]
            else:
                extracted = result
        
        # 保存
        cv2.imwrite(str(output_path), extracted)
        return True, "抽出成功"
        
    except Exception as e:
        return False, f"エラー: {str(e)}"


def main():
    """メイン処理"""
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge")
    
    # モデル初期化
    logger.info("モデル初期化中...")
    
    # SAM初期化
    sam_checkpoint = Path("/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth")
    if not sam_checkpoint.exists():
        logger.error(f"SAMモデルが見つかりません: {sam_checkpoint}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam_generator = SamAutomaticMaskGenerator(sam)
    
    # YOLO初期化
    yolo_model = YOLO('yolov8n.pt')
    
    logger.info("モデル初期化完了")
    
    # 画像ファイル取得
    image_files = sorted(list(input_dir.glob("*.jpg")))
    total = len(image_files)
    
    logger.info(f"バッチ処理開始: {total}枚の画像")
    logger.info(f"入力: {input_dir}")
    logger.info(f"出力: {output_dir}")
    
    # 処理統計
    success_count = 0
    start_time = time.time()
    
    # 各画像を処理
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
        
        output_path = output_dir / image_path.name
        success, message = extract_character_simple(image_path, sam_generator, yolo_model, output_path)
        
        if success:
            success_count += 1
            logger.info(f"  ✅ {message}")
        else:
            logger.warning(f"  ❌ {message}")
        
        # 進捗表示
        if i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            logger.info(f"進捗: {i}/{total} - 平均: {avg_time:.1f}秒/画像")
    
    # 完了
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("バッチ処理完了")
    logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
    logger.info(f"総時間: {total_time:.1f}秒 ({total_time/total:.1f}秒/画像)")


if __name__ == "__main__":
    main()