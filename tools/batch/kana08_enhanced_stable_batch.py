#!/usr/bin/env python3
"""
kana08 強化安定版バッチ抽出スクリプト（マージ版）
P1-A001復元版 + v0.0.1現在版の統合実装
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
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# SAMとYOLOのインポート
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Kana08EnhancedStableExtractor:
    """kana08強化安定版抽出器（P1-A001 + v0.0.1統合）"""
    
    def __init__(self, input_dir=None, output_dir=None):
        self.input_dir = Path(input_dir) if input_dir else Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        self.output_dir = Path(output_dir) if output_dir else Path("/mnt/c/AItools/segment-anything/results_batch/kana08_enhanced")
        
        # モデル初期化
        logger.info("Enhanced Extractor初期化中...")
        
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
        
        # YOLOモデル（P1-A001: nanoモデル使用、v0.0.1: xモデル使用）
        # 統合版では速度重視でnanoを採用
        self.yolo_model = YOLO("yolov8n.pt")
        logger.info("✅ YOLOモデル初期化完了（nanoモデル採用）")
        
        # 設定（P1-A001復元版の設定を採用）
        self.confidence_threshold = 0.07  # アニメ特化閾値
        
    def extract_character(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        キャラクター抽出（P1-A001復元版ベース）
        """
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
            
            # 最大の検出結果を使用（P1-A001アプローチ）
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
            sam_score = float(scores[best_idx])
            
            # マスク適用（白背景版 - P1-A001改良）
            h, w = image.shape[:2]
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result.fill(255)  # 白背景
            result[mask] = image[mask]
            
            # 統計情報（P1-A001 + v0.0.1統合）
            stats = {
                'confidence': confidence,
                'sam_score': sam_score,
                'mask_ratio': np.sum(mask) / (h * w),
                'bbox': [x1, y1, x2, y2],
                'bbox_area_ratio': ((x2-x1) * (y2-y1)) / (h * w)
            }
            
            return result, confidence, stats
            
        except Exception as e:
            logger.error(f"抽出エラー: {str(e)}")
            return None
    
    def _judge_quality_enhanced(self, stats: Dict[str, Any]) -> Tuple[str, float]:
        """
        強化品質判定（P1-A001シンプル版 + v0.0.1詳細評価）
        """
        confidence = stats['confidence']
        sam_score = stats['sam_score']
        mask_ratio = stats['mask_ratio']
        bbox_area_ratio = stats.get('bbox_area_ratio', 0)
        
        # P1-A001ベースの総合スコア
        base_score = (confidence * 0.3 + sam_score * 0.4 + min(mask_ratio * 2, 1.0) * 0.3)
        
        # v0.0.1ベースの調整要素
        # 1. バウンディングボックス範囲調整
        if bbox_area_ratio > 0.85:  # 全体取得ペナルティ
            base_score -= 0.2
        elif 0.1 <= bbox_area_ratio <= 0.6:  # 理想的範囲
            base_score += 0.1
        
        # 2. 信頼度とSAMスコアの組み合わせ評価
        if confidence > 0.5 and sam_score > 0.9:
            base_score += 0.1  # 高品質ボーナス
        
        # 最終品質判定（P1-A001基準）
        if base_score >= 0.8:
            grade = 'A'
        elif base_score >= 0.7:
            grade = 'B'
        elif base_score >= 0.6:
            grade = 'C'
        elif base_score >= 0.5:
            grade = 'D'
        else:
            grade = 'E'
        
        return grade, base_score
    
    def process_image(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """単一画像の処理（P1-A001ベース + 強化品質判定）"""
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
            
            # 結果保存（JPG形式で統一）
            output_path = self.output_dir / (image_path.stem + ".jpg")
            cv2.imwrite(str(output_path), extracted_img)
            
            # 強化品質判定
            quality, quality_score = self._judge_quality_enhanced(stats)
            stats['quality_score'] = quality_score
            
            return True, f"成功 (信頼度: {confidence:.3f}, 品質: {quality}, スコア: {quality_score:.3f})", stats
            
        except Exception as e:
            return False, f"エラー: {str(e)}", None
    
    def run_batch(self):
        """バッチ処理実行（P1-A001形式）"""
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ファイル取得
        image_files = sorted(list(self.input_dir.glob("*.jpg")))
        total = len(image_files)
        
        if total == 0:
            logger.error("処理する画像が見つかりません")
            return
        
        logger.info(f"Enhanced Batch処理開始: {total}枚の画像")
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
                    quality = self._judge_quality_enhanced(stats)[0]
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
            avg_quality_score = np.mean([s['quality_score'] for s in all_stats])
        else:
            avg_confidence = avg_sam_score = avg_mask_ratio = avg_quality_score = 0.0
        
        logger.info("=" * 50)
        logger.info("Enhanced Batch処理完了")
        logger.info(f"総処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
        logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
        logger.info("")
        logger.info("品質分布:")
        ab_count = quality_counts['A'] + quality_counts['B']
        for grade in ['A', 'B', 'C', 'D', 'E']:
            count = quality_counts[grade]
            percentage = count / success_count * 100 if success_count > 0 else 0
            logger.info(f"  {grade}評価: {count}枚 ({percentage:.1f}%)")
        
        ab_percentage = ab_count / success_count * 100 if success_count > 0 else 0
        logger.info(f"  A/B評価率: {ab_count}枚 ({ab_percentage:.1f}%)")
        logger.info("")
        logger.info("統計:")
        logger.info(f"  平均信頼度: {avg_confidence:.3f}")
        logger.info(f"  平均SAMスコア: {avg_sam_score:.3f}")
        logger.info(f"  平均マスク比率: {avg_mask_ratio:.3f}")
        logger.info(f"  平均品質スコア: {avg_quality_score:.3f}")
        
        if failed_files:
            logger.info("")
            logger.info("失敗ファイル:")
            for name, reason in failed_files:
                logger.info(f"  - {name}: {reason}")
        
        # レポート作成（P1-A001 + 強化版）
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "version": "Enhanced_Stable_Merge_v1.0",
            "total_images": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "processing_time": total_time,
            "avg_processing_time": total_time / total,
            "quality_distribution": quality_counts,
            "ab_evaluation_rate": ab_percentage / 100,
            "statistics": {
                "avg_confidence": float(avg_confidence),
                "avg_sam_score": float(avg_sam_score),
                "avg_mask_ratio": float(avg_mask_ratio),
                "avg_quality_score": float(avg_quality_score)
            },
            "failed_files": failed_files
        }
        
        # JSONレポート保存
        report_path = self.output_dir / "enhanced_extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"レポート保存: {report_path}")
        
        # サマリーファイル作成
        summary_path = self.output_dir / "enhanced_extraction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"kana08 Enhanced Stable Extraction サマリー（マージ版）\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"版本: P1-A001復元版 + v0.0.1統合\n")
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
            f.write(f"  A/B評価率: {ab_count}枚 ({ab_percentage:.1f}%)\n")
            f.write(f"\n統計:\n")
            f.write(f"  平均信頼度: {avg_confidence:.3f}\n")
            f.write(f"  平均SAMスコア: {avg_sam_score:.3f}\n")
            f.write(f"  平均マスク比率: {avg_mask_ratio:.3f}\n")
            f.write(f"  平均品質スコア: {avg_quality_score:.3f}\n")
            
            if failed_files:
                f.write("\n失敗ファイル:\n")
                for name, reason in failed_files:
                    f.write(f"  - {name}: {reason}\n")
        
        logger.info(f"サマリー保存: {summary_path}")
        
        return report


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Stable Character Extraction（マージ版）')
    parser.add_argument('--input_dir', type=str, 
                       default='/mnt/c/AItools/lora/train/yado/org/kana08',
                       help='入力ディレクトリ')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/ENHANCED-MERGE',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    extractor = Kana08EnhancedStableExtractor(args.input_dir, args.output_dir)
    result = extractor.run_batch()
    
    # 成功時は0、失敗時は1で終了
    return 0 if result and result.get('success_rate', 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())