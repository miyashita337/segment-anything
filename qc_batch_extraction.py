#!/usr/bin/env python3
"""
QC品質調査用バッチキャラクター抽出スクリプト
kana08, kana05, kana07の全画像を本格抽出 + Pushover通知
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import traceback

# 環境設定
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import cv2
import torch
from PIL import Image
import requests

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
        logging.FileHandler('/mnt/c/AItools/segment-anything/qc_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QCBatchExtractor:
    """QC品質調査用バッチ抽出器"""
    
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
        
        # Pushover設定
        self.pushover_config = self.load_pushover_config()
        
        # 統計情報
        self.stats = {
            'total_processed': 0,
            'total_success': 0,
            'total_failed': 0,
            'folder_stats': {}
        }
    
    def load_pushover_config(self) -> Dict[str, str]:
        """Pushover設定読み込み"""
        try:
            with open('/mnt/c/AItools/segment-anything/config/pushover.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Pushover config load failed: {e}")
            return {}
    
    def send_pushover_notification(self, message: str, title: str = "QC抽出", image_path: str = None):
        """Pushover通知送信"""
        if not self.pushover_config.get('user_key') or not self.pushover_config.get('api_token'):
            logger.info(f"Pushover未設定: {message}")
            return
        
        try:
            data = {
                'token': self.pushover_config['api_token'],
                'user': self.pushover_config['user_key'],
                'title': title,
                'message': message
            }
            
            files = {}
            if image_path and os.path.exists(image_path):
                files['attachment'] = open(image_path, 'rb')
            
            response = requests.post('https://api.pushover.net/1/messages.json', 
                                   data=data, files=files)
            
            if files:
                files['attachment'].close()
            
            if response.status_code == 200:
                logger.info("Pushover通知送信成功")
            else:
                logger.error(f"Pushover通知失敗: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Pushover通知エラー: {e}")
    
    def extract_character(self, input_path: str, output_path: str) -> bool:
        """単一画像からキャラクター抽出"""
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
    
    def process_folder(self, input_dir: str, output_dir: str, folder_name: str) -> Dict[str, Any]:
        """フォルダ内全画像の抽出処理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 入力画像一覧取得
        image_files = sorted(list(input_path.glob("*.jpg")))
        total_images = len(image_files)
        
        logger.info(f"=== {folder_name} 処理開始: {total_images}枚 ===")
        
        success_count = 0
        failed_count = 0
        success_files = []
        
        for i, input_file in enumerate(image_files, 1):
            output_file = output_path / f"extracted_{input_file.stem}.png"
            
            logger.info(f"[{i}/{total_images}] 処理中: {input_file.name}")
            
            if self.extract_character(str(input_file), str(output_file)):
                success_count += 1
                success_files.append(str(output_file))
            else:
                failed_count += 1
            
            # 進捗報告（10枚ごと）
            if i % 10 == 0 or i == total_images:
                progress_msg = f"{folder_name}: {i}/{total_images}枚完了 (成功:{success_count}, 失敗:{failed_count})"
                logger.info(progress_msg)
        
        # フォルダ完了統計
        folder_stats = {
            'total': total_images,
            'success': success_count,
            'failed': failed_count,
            'success_rate': (success_count / total_images * 100) if total_images > 0 else 0,
            'success_files': success_files
        }
        
        self.stats['folder_stats'][folder_name] = folder_stats
        self.stats['total_processed'] += total_images
        self.stats['total_success'] += success_count
        self.stats['total_failed'] += failed_count
        
        # フォルダ完了通知
        completion_msg = f"""🎯 {folder_name} 抽出完了
📊 {success_count}/{total_images}枚成功 ({folder_stats['success_rate']:.1f}%)
❌ 失敗: {failed_count}枚
📁 出力: {output_dir}"""
        
        # サンプル画像添付
        sample_image = success_files[0] if success_files else None
        self.send_pushover_notification(completion_msg, f"{folder_name}完了", sample_image)
        
        logger.info(f"=== {folder_name} 完了: 成功{success_count}/{total_images} ===")
        
        return folder_stats
    
    def run_qc_extraction(self):
        """QC抽出メイン処理"""
        logger.info("=== QC品質調査バッチ抽出開始 ===")
        start_time = time.time()
        
        # 抽出対象フォルダ
        folders = [
            {
                'name': 'KANA08',
                'input': '/mnt/c/AItools/lora/train/yado/org/kana08/',
                'output': '/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA08/'
            },
            {
                'name': 'KANA05', 
                'input': '/mnt/c/AItools/lora/train/yado/org/kana05/',
                'output': '/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA05/'
            },
            {
                'name': 'KANA07',
                'input': '/mnt/c/AItools/lora/train/yado/org/kana07/',
                'output': '/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA07/'
            }
        ]
        
        try:
            # 各フォルダ順次処理
            for folder in folders:
                self.process_folder(
                    folder['input'],
                    folder['output'], 
                    folder['name']
                )
            
            # 全体完了統計
            end_time = time.time()
            total_time = end_time - start_time
            
            final_msg = f"""🎉 QC抽出全体完了
📊 総計: {self.stats['total_success']}/{self.stats['total_processed']}枚
⏱️ 処理時間: {total_time/60:.1f}分
📈 全体成功率: {(self.stats['total_success']/self.stats['total_processed']*100):.1f}%

📁 結果:
• KANA08: {self.stats['folder_stats']['KANA08']['success']}/{self.stats['folder_stats']['KANA08']['total']}枚
• KANA05: {self.stats['folder_stats']['KANA05']['success']}/{self.stats['folder_stats']['KANA05']['total']}枚  
• KANA07: {self.stats['folder_stats']['KANA07']['success']}/{self.stats['folder_stats']['KANA07']['total']}枚"""
            
            self.send_pushover_notification(final_msg, "QC抽出完了")
            logger.info("=== QC品質調査バッチ抽出完了 ===")
            
        except Exception as e:
            error_msg = f"QC抽出エラー: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.send_pushover_notification(f"❌ QC抽出エラー: {str(e)}", "エラー")

def main():
    """メイン実行"""
    try:
        extractor = QCBatchExtractor()
        extractor.run_qc_extraction()
    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()