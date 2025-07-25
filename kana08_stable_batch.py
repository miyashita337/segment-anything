#!/usr/bin/env python3
"""
kana08 安定版バッチ抽出スクリプト
既存の動作確認済みコードをベースに実装
"""

import numpy as np
import cv2
import json
import logging
import os
import sys
import time
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# SAMとYOLOのインポート
from core.segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Kana08StableExtractor:
    """kana08安定版抽出器"""
    
    def __init__(self):
        self.input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge")
        
        # モデル初期化
        logger.info("モデル初期化中...")
        
        # SAMモデル
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        checkpoint_path = "sam_vit_h_4b8939.pth"
        if not os.path.exists(checkpoint_path):
            logger.error(f"SAMチェックポイントが見つかりません: {checkpoint_path}")
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        logger.info("✅ SAMモデル初期化完了")
        
        # YOLOモデル
        self.yolo_model = YOLO("yolov8n.pt")
        logger.info("✅ YOLOモデル初期化完了")
        
        # 設定
        self.confidence_threshold = 0.07  # アニメ特化閾値
        
    def extract_character(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """キャラクター抽出"""
        try:
            # YOLO検出
            results = self.yolo_model(image, verbose=False)
            
            # person検出の取得
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i])
                        if cls == 0:  # person class
                            conf = float(boxes.conf[i])
                            if conf >= self.confidence_threshold:
                                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': conf,
                                    'area': (x2 - x1) * (y2 - y1)
                                })
            
            if not detections:
                return None
            
            # 最大の検出結果を使用
            best_detection = max(detections, key=lambda d: d['area'])
            x1, y1, x2, y2 = best_detection['bbox']
            confidence = best_detection['confidence']
            
            # SAMでマスク生成
            self.sam_predictor.set_image(image)
            
            box_prompt = np.array([x1, y1, x2, y2])
            masks, scores, _ = self.sam_predictor.predict(
                box=box_prompt,
                multimask_output=True
            )
            
            if len(masks) == 0:
                return None
            
            # 最良マスク選択
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
            
            # マスク適用（透過背景）
            h, w = image.shape[:2]
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[:, :, :3] = image
            result[:, :, 3] = mask * 255
            
            # 統計情報
            stats = {
                'confidence': confidence,
                'sam_score': score,
                'mask_ratio': np.sum(mask) / (h * w),
                'bbox': [x1, y1, x2, y2]
            }
            
            return result, confidence, stats
            
        except Exception as e:
            logger.error(f"抽出エラー: {str(e)}")
            return None
    
    def process_image(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """単一画像の処理"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "画像読み込み失敗", None
            
            # 抽出実行
            result = self.extract_character(image)
            
            if result is None:
                return False, "キャラクター検出失敗", None
            
            extracted_img, confidence, stats = result
            
            # 結果保存（PNG形式で透過情報を保持）
            output_path = self.output_dir / (image_path.stem + "_extracted.png")
            cv2.imwrite(str(output_path), extracted_img)
            
            # 品質判定
            quality = self._judge_quality(stats)
            
            return True, f"成功 (信頼度: {confidence:.3f}, 品質: {quality})", stats
            
        except Exception as e:
            return False, f"エラー: {str(e)}", None
    
    def _judge_quality(self, stats: Dict[str, Any]) -> str:
        """品質判定"""
        confidence = stats['confidence']
        sam_score = stats['sam_score']
        mask_ratio = stats['mask_ratio']
        
        # 総合スコア計算
        overall_score = (confidence * 0.3 + sam_score * 0.4 + min(mask_ratio * 2, 1.0) * 0.3)
        
        if overall_score >= 0.8:
            return 'A'
        elif overall_score >= 0.7:
            return 'B'
        elif overall_score >= 0.6:
            return 'C'
        elif overall_score >= 0.5:
            return 'D'
        else:
            return 'E'
    
    def run_batch(self):
        """バッチ処理実行"""
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ファイル取得
        image_files = sorted(list(self.input_dir.glob("*.jpg")))
        total = len(image_files)
        
        if total == 0:
            logger.error("処理する画像が見つかりません")
            return
        
        logger.info(f"バッチ処理開始: {total}枚の画像")
        logger.info(f"入力: {self.input_dir}")
        logger.info(f"出力: {self.output_dir}")
        
        # 処理統計
        success_count = 0
        failed_files = []
        quality_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        all_stats = []
        start_time = time.time()
        
        # 各画像を処理
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
            
            success, message, stats = self.process_image(image_path)
            
            if success:
                success_count += 1
                logger.info(f"  ✅ {message}")
                
                if stats:
                    all_stats.append(stats)
                    quality = self._judge_quality(stats)
                    quality_counts[quality] += 1
            else:
                failed_files.append((image_path.name, message))
                logger.warning(f"  ❌ {message}")
            
            # 進捗表示
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (total - i)
                logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%) - 残り時間: {remaining:.0f}秒")
        
        # 処理完了
        total_time = time.time() - start_time
        
        # 統計計算
        if all_stats:
            avg_confidence = np.mean([s['confidence'] for s in all_stats])
            avg_sam_score = np.mean([s['sam_score'] for s in all_stats])
            avg_mask_ratio = np.mean([s['mask_ratio'] for s in all_stats])
        else:
            avg_confidence = avg_sam_score = avg_mask_ratio = 0.0
        
        logger.info("=" * 50)
        logger.info("バッチ処理完了")
        logger.info(f"総処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
        logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
        logger.info("")
        logger.info("品質分布:")
        for grade in ['A', 'B', 'C', 'D', 'E']:
            count = quality_counts[grade]
            percentage = count / success_count * 100 if success_count > 0 else 0
            logger.info(f"  {grade}評価: {count}枚 ({percentage:.1f}%)")
        logger.info("")
        logger.info("統計:")
        logger.info(f"  平均信頼度: {avg_confidence:.3f}")
        logger.info(f"  平均SAMスコア: {avg_sam_score:.3f}")
        logger.info(f"  平均マスク比率: {avg_mask_ratio:.3f}")
        
        if failed_files:
            logger.info("")
            logger.info("失敗ファイル:")
            for name, reason in failed_files:
                logger.info(f"  - {name}: {reason}")
        
        # レポート作成
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_images": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "processing_time": total_time,
            "avg_processing_time": total_time / total,
            "quality_distribution": quality_counts,
            "statistics": {
                "avg_confidence": float(avg_confidence),
                "avg_sam_score": float(avg_sam_score),
                "avg_mask_ratio": float(avg_mask_ratio)
            },
            "failed_files": failed_files
        }
        
        # JSONレポート保存
        report_path = self.output_dir / "extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"レポート保存: {report_path}")
        
        # サマリーファイル作成
        summary_path = self.output_dir / "extraction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"kana08 バッチ抽出サマリー（安定版）\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"処理日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総画像数: {total}\n")
            f.write(f"成功: {success_count} ({success_count/total*100:.1f}%)\n")
            f.write(f"失敗: {len(failed_files)} ({len(failed_files)/total*100:.1f}%)\n")
            f.write(f"処理時間: {total_time:.1f}秒\n")
            f.write(f"平均: {total_time/total:.1f}秒/画像\n")
            f.write(f"\n品質分布:\n")
            for grade in ['A', 'B', 'C', 'D', 'E']:
                count = quality_counts[grade]
                percentage = count / success_count * 100 if success_count > 0 else 0
                f.write(f"  {grade}: {count}枚 ({percentage:.1f}%)\n")
            f.write(f"\n統計:\n")
            f.write(f"  平均信頼度: {avg_confidence:.3f}\n")
            f.write(f"  平均SAMスコア: {avg_sam_score:.3f}\n")
            f.write(f"  平均マスク比率: {avg_mask_ratio:.3f}\n")
            
            if failed_files:
                f.write("\n失敗ファイル:\n")
                for name, reason in failed_files:
                    f.write(f"  - {name}: {reason}\n")
        
        logger.info(f"サマリー保存: {summary_path}")


def main():
    """メイン実行"""
    extractor = Kana08StableExtractor()
    extractor.run_batch()


if __name__ == "__main__":
    main()