#!/usr/bin/env python3
"""
QC成功版互換抽出スクリプト
SAM Predictorを使用してQC成功版と同等の結果を実現
"""

import numpy as np
import cv2
import torch
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO


class QCCompatibleExtractor:
    """QC成功版と完全互換の抽出システム"""
    
    def __init__(self, 
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth",
                 yolo_model_path: str = "yolov8x.pt",
                 device: str = None):
        """
        初期化
        
        Args:
            sam_checkpoint: SAMモデルチェックポイントパス
            yolo_model_path: YOLOモデルパス
            device: 実行デバイス
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_checkpoint = sam_checkpoint
        self.yolo_model_path = yolo_model_path
        
        # モデル初期化
        self.sam_model = None
        self.predictor = None
        self.yolo_model = None
        
        print(f"🚀 QC互換抽出システム初期化")
        print(f"   Device: {self.device}")
        print(f"   SAM: {sam_checkpoint}")
        print(f"   YOLO: {yolo_model_path}")
    
    def load_models(self) -> bool:
        """モデルロード"""
        try:
            # SAMモデルロード
            print("📥 SAMモデルロード中...")
            self.sam_model = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
            self.sam_model.to(device=self.device)
            
            # Predictor作成（QC成功版の核心）
            self.predictor = SamPredictor(self.sam_model)
            print("✅ SAM Predictor初期化完了")
            
            # YOLOモデルロード
            print("📥 YOLOモデルロード中...")
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLOモデルロード完了")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロード失敗: {e}")
            return False
    
    def detect_with_yolo(self, image_rgb: np.ndarray, conf_threshold: float = 0.07) -> list:
        """
        YOLO検出（QC成功版パラメータ）
        
        Args:
            image_rgb: RGB画像
            conf_threshold: 信頼度閾値（QC成功版: 0.07）
        
        Returns:
            検出結果リスト
        """
        results = self.yolo_model(image_rgb, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # 全クラスを対象（person以外も含む）
                    conf = float(boxes.conf[i])
                    if conf >= conf_threshold:
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        detections.append({
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'area': area,
                            'class_id': int(boxes.cls[i])
                        })
        
        # 面積でソート（大きい順）
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"🎯 YOLO検出: {len(detections)}個 (conf≥{conf_threshold})")
        if detections:
            print(f"   最大領域: {detections[0]['area']:.0f}px²")
        
        return detections
    
    def extract_with_sam_predictor(self, 
                                   image_rgb: np.ndarray,
                                   box: list) -> Tuple[Optional[np.ndarray], float]:
        """
        SAM Predictorで抽出（QC成功版コアアルゴリズム）
        
        Args:
            image_rgb: RGB画像
            box: バウンディングボックス [x1, y1, x2, y2]
        
        Returns:
            (マスク, スコア) のタプル
        """
        try:
            # 画像をPredictorにセット
            self.predictor.set_image(image_rgb)
            
            # ボックスプロンプトで予測
            box_np = np.array(box)
            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=True
            )
            
            # QC成功版アルゴリズム: 最高スコアのマスクを選択
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            print(f"   SAM予測: {len(masks)}マスク生成, 最高スコア: {best_score:.3f}")
            
            return best_mask, best_score
            
        except Exception as e:
            print(f"❌ SAM予測エラー: {e}")
            return None, 0.0
    
    def process_single_image(self, 
                           image_path: str,
                           output_path: str,
                           conf_threshold: float = 0.07) -> bool:
        """
        単一画像処理（QC成功版完全互換）
        
        Args:
            image_path: 入力画像パス
            output_path: 出力パス
            conf_threshold: YOLO信頼度閾値
        
        Returns:
            成功フラグ
        """
        print(f"\n🖼️ 処理開始: {Path(image_path).name}")
        
        # 画像読み込み（元サイズ維持）
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"❌ 画像読み込み失敗: {image_path}")
            return False
        
        h, w = image_bgr.shape[:2]
        print(f"   サイズ: {w}×{h} (元サイズ維持)")
        
        # RGB変換
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # YOLO検出
        detections = self.detect_with_yolo(image_rgb, conf_threshold)
        
        if not detections:
            print("⚠️ YOLO検出なし - 画像全体を使用")
            # 検出なしの場合は画像全体をボックスとして使用
            box = [0, 0, w, h]
        else:
            # 最大面積の検出を使用
            box = detections[0]['box']
            print(f"   使用ボックス: {box} (面積: {detections[0]['area']:.0f}px²)")
        
        # SAM Predictorで抽出
        mask, score = self.extract_with_sam_predictor(image_rgb, box)
        
        if mask is None:
            print("❌ マスク生成失敗")
            return False
        
        # マスク適用
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # アルファチャンネル作成
        b, g, r = cv2.split(image_bgr)
        rgba = cv2.merge([b, g, r, mask_uint8])
        
        # 透明部分を除去してクロップ
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max() + 1
            x_min, x_max = coords[1].min(), coords[1].max() + 1
            
            # クロップ
            cropped = rgba[y_min:y_max, x_min:x_max]
            
            # 出力保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(str(output_path), cropped)
            
            # 結果サイズ
            crop_h, crop_w = cropped.shape[:2]
            print(f"✅ 抽出成功: {crop_w}×{crop_h} (スコア: {score:.3f})")
            print(f"   保存先: {output_path}")
            
            return True
        else:
            print("❌ 有効なマスク領域なし")
            return False
    
    def process_batch(self,
                     input_dir: str,
                     output_dir: str,
                     conf_threshold: float = 0.07,
                     max_files: int = None) -> dict:
        """
        バッチ処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            conf_threshold: YOLO信頼度閾値
            max_files: 最大処理数
        
        Returns:
            処理結果統計
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 画像ファイル取得
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(input_path.glob(ext))
        
        image_files = sorted(image_files)[:max_files] if max_files else sorted(image_files)
        
        print(f"\n📦 バッチ処理開始")
        print(f"   入力: {input_dir}")
        print(f"   出力: {output_dir}")
        print(f"   対象: {len(image_files)}ファイル")
        print(f"   YOLO閾値: {conf_threshold}")
        
        # 処理実行
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'sizes': []
        }
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {img_file.name}")
            
            # 出力ファイル名（元のファイル名を保持）
            output_file = output_path / f"extracted_{img_file.stem}.png"
            
            # 処理実行
            success = self.process_single_image(
                str(img_file),
                str(output_file),
                conf_threshold
            )
            
            if success:
                stats['success'] += 1
                # サイズ記録
                result_img = cv2.imread(str(output_file), cv2.IMREAD_UNCHANGED)
                if result_img is not None:
                    h, w = result_img.shape[:2]
                    stats['sizes'].append((w, h))
            else:
                stats['failed'] += 1
        
        # 統計出力
        print("\n" + "="*60)
        print("📊 処理完了統計")
        print(f"   成功: {stats['success']}/{stats['total']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"   失敗: {stats['failed']}/{stats['total']}")
        
        if stats['sizes']:
            avg_w = np.mean([s[0] for s in stats['sizes']])
            avg_h = np.mean([s[1] for s in stats['sizes']])
            print(f"   平均サイズ: {avg_w:.0f}×{avg_h:.0f}")
            
            # サイズ分布
            print(f"   サイズ範囲:")
            print(f"     幅: {min(s[0] for s in stats['sizes'])} - {max(s[0] for s in stats['sizes'])}")
            print(f"     高さ: {min(s[1] for s in stats['sizes'])} - {max(s[1] for s in stats['sizes'])}")
        
        return stats


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QC成功版互換抽出")
    parser.add_argument("input", help="入力画像またはディレクトリ")
    parser.add_argument("output", help="出力ディレクトリ")
    parser.add_argument("--conf", type=float, default=0.07, help="YOLO信頼度閾値")
    parser.add_argument("--max-files", type=int, help="最大処理ファイル数")
    parser.add_argument("--sam-checkpoint", default="sam_vit_h_4b8939.pth", help="SAMチェックポイント")
    parser.add_argument("--yolo-model", default="yolov8x.pt", help="YOLOモデル")
    
    args = parser.parse_args()
    
    # 抽出器初期化
    extractor = QCCompatibleExtractor(
        sam_checkpoint=args.sam_checkpoint,
        yolo_model_path=args.yolo_model
    )
    
    # モデルロード
    if not extractor.load_models():
        print("❌ モデルロード失敗")
        sys.exit(1)
    
    # 処理実行
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイル処理
        output_file = Path(args.output) / f"extracted_{input_path.stem}.png"
        success = extractor.process_single_image(
            str(input_path),
            str(output_file),
            args.conf
        )
        sys.exit(0 if success else 1)
    else:
        # バッチ処理
        stats = extractor.process_batch(
            str(input_path),
            args.output,
            args.conf,
            args.max_files
        )
        sys.exit(0 if stats['success'] > 0 else 1)


if __name__ == "__main__":
    main()