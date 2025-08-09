#!/usr/bin/env python3
"""
QC成功スクリプトベース - KANA08の20枚バッチ抽出
"""

import sys
import os
import time
import logging
from pathlib import Path
import traceback

# 環境設定
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import cv2
import torch
from PIL import Image

# SAMとYOLOのインポート
sys.path.insert(0, '/mnt/c/AItools/segment-anything')
sys.path.insert(0, '/mnt/c/AItools/segment-anything/core')
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mnt/c/AItools/segment-anything/qc_kana08_20.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QCKana08Extractor:
    """QC成功ベース - KANA08の20枚抽出器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # SAMモデル初期化
        sam_checkpoint = "/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
        # YOLOモデル初期化
        self.yolo_model = YOLO('/mnt/c/AItools/segment-anything/yolov8x.pt')
        
        # 統計情報
        self.stats = {
            'total_processed': 0,
            'total_success': 0,
            'total_failed': 0
        }
    
    def extract_character(self, input_path: str, output_path: str) -> bool:
        """単一画像からキャラクター抽出（QC成功版そのまま）"""
        try:
            # 画像読み込み
            image = cv2.imread(str(input_path))
            if image is None:
                logger.error(f"画像読み込み失敗: {input_path}")
                return False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # YOLO検出（アニメキャラクター特化閾値）
            results = self.yolo_model(image_rgb, conf=0.07)
            
            if len(results[0].boxes) == 0:
                logger.warning(f"検出なし: {input_path}")
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
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
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
            
            logger.info(f"抽出成功: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            logger.error(f"抽出エラー {input_path}: {e}")
            return False
    
    def process_kana08_20(self):
        """KANA08の20枚処理"""
        input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QC-KANA08-SUCCESS-20")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最初の20枚を取得
        image_files = sorted(list(input_dir.glob("*.jpg")))[:20]
        total_images = len(image_files)
        
        logger.info(f"=== QC成功版でKANA08の20枚処理開始 ===")
        
        success_count = 0
        failed_count = 0
        failed_files = []
        start_time = time.time()
        
        for i, input_file in enumerate(image_files, 1):
            # QC成功版と同じ出力形式
            output_file = output_dir / f"extracted_{input_file.stem}.png"
            
            logger.info(f"[{i:2d}/20] 処理中: {input_file.name}")
            
            if self.extract_character(str(input_file), str(output_file)):
                success_count += 1
                logger.info(f"  ✅ 成功")
            else:
                failed_count += 1
                failed_files.append(input_file.name)
                logger.info(f"  ❌ 失敗")
        
        # 結果サマリー
        end_time = time.time()
        total_time = end_time - start_time
        success_rate = (success_count / total_images * 100) if total_images > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🎯 QC成功版 - KANA08の20枚処理完了")
        logger.info(f"📊 成功率: {success_rate:.1f}% ({success_count}/20枚)")
        logger.info(f"⏱️ 処理時間: {total_time:.1f}秒")
        
        if failed_files:
            logger.info(f"❌ 失敗ファイル: {', '.join(failed_files)}")
        else:
            logger.info("✅ 全20枚で抽出成功！QC成功版による完全復元")
        
        logger.info(f"📁 出力先: {output_dir}")
        logger.info("=" * 60)
        
        return success_count, failed_count, failed_files

def main():
    """メイン実行"""
    try:
        extractor = QCKana08Extractor()
        success, failed, failed_files = extractor.process_kana08_20()
        
        if success >= 16:  # 80%以上で成功
            logger.info("🎉 QC成功版による抽出完了 - 成功基準達成")
            return 0
        else:
            logger.error(f"❌ 成功基準未達: {success}/20枚")
            return 1
            
    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())