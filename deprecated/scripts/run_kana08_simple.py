#!/usr/bin/env python3
"""
kana08シンプル抽出スクリプト
既存の抽出システムを使用して基本的な抽出を実行
"""

import numpy as np
import cv2
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.extraction.models.sam_wrapper import SAMModelWrapper
from features.extraction.models.yolo_wrapper import YOLOModelWrapper
# from features.processing.postprocessing.postprocessing import apply_mask_to_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_character_simple(image: np.ndarray, sam_wrapper: SAMModelWrapper, yolo_wrapper: YOLOModelWrapper) -> Optional[Tuple[np.ndarray, float]]:
    """シンプルなキャラクター抽出"""
    try:
        # YOLO検出
        if not yolo_wrapper.is_loaded:
            yolo_wrapper.load_model()
        detections = yolo_wrapper.detect_persons(image)
        if not detections:
            return None
        
        # 最大の検出結果を使用
        best_detection = max(detections, key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1']))
        x1, y1, x2, y2 = best_detection['x1'], best_detection['y1'], best_detection['x2'], best_detection['y2']
        confidence = best_detection['confidence']
        
        # SAMマスク生成
        box_prompt = np.array([x1, y1, x2, y2])
        # SAMモデルがロードされていない場合はロード
        if not hasattr(sam_wrapper, 'predictor') or sam_wrapper.predictor is None:
            sam_wrapper.load_model()
        
        masks, scores, _ = sam_wrapper.predict(
            image,
            box=box_prompt,
            multimask_output=True
        )
        
        if len(masks) == 0:
            return None
        
        # 最良マスク選択
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # マスク適用
        # extracted_img = apply_mask_to_image(image, mask)
        # シンプルなマスク適用
        extracted_img = image.copy()
        extracted_img[mask == 0] = 255  # 背景を白に
        
        return extracted_img, confidence
        
    except Exception as e:
        logger.error(f"抽出エラー: {str(e)}")
        return None


def main():
    """メイン処理"""
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge")
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル初期化
    logger.info("モデル初期化中...")
    sam_wrapper = SAMModelWrapper()
    yolo_wrapper = YOLOModelWrapper()
    
    # 画像ファイル取得
    image_files = sorted(list(input_dir.glob("*.jpg")))
    total = len(image_files)
    
    if total == 0:
        logger.error("処理する画像が見つかりません")
        return
    
    logger.info(f"バッチ処理開始: {total}枚の画像")
    logger.info(f"入力: {input_dir}")
    logger.info(f"出力: {output_dir}")
    
    # 処理統計
    success_count = 0
    failed_files = []
    confidence_scores = []
    start_time = time.time()
    
    # 各画像を処理
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
        
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                failed_files.append((image_path.name, "画像読み込み失敗"))
                logger.warning(f"  ❌ 失敗 - 画像読み込み失敗")
                continue
            
            # 抽出実行
            result = extract_character_simple(image, sam_wrapper, yolo_wrapper)
            
            if result is None:
                failed_files.append((image_path.name, "キャラクター抽出失敗"))
                logger.warning(f"  ❌ 失敗 - キャラクター抽出失敗")
                continue
            
            extracted_img, confidence = result
            
            # 結果保存
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), extracted_img)
            
            success_count += 1
            confidence_scores.append(confidence)
            logger.info(f"  ✅ 成功 - 信頼度: {confidence:.3f}")
            
        except Exception as e:
            failed_files.append((image_path.name, f"エラー: {str(e)}"))
            logger.warning(f"  ❌ 失敗 - エラー: {str(e)}")
        
        # 進捗表示
        if i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)
            logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%) - 残り時間: {remaining:.0f}秒")
    
    # 処理完了
    total_time = time.time() - start_time
    
    # 統計計算
    if confidence_scores:
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
    else:
        avg_confidence = min_confidence = max_confidence = 0.0
    
    logger.info("=" * 50)
    logger.info("バッチ処理完了")
    logger.info(f"総処理時間: {total_time:.1f}秒")
    logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
    logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
    logger.info(f"平均信頼度: {avg_confidence:.3f} (最小: {min_confidence:.3f}, 最大: {max_confidence:.3f})")
    
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
        "confidence_stats": {
            "average": float(avg_confidence),
            "min": float(min_confidence),
            "max": float(max_confidence)
        },
        "failed_files": failed_files
    }
    
    # JSONレポート保存
    report_path = output_dir / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"レポート保存: {report_path}")
    
    # サマリーファイル作成
    summary_path = output_dir / "extraction_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"kana08 バッチ抽出サマリー\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"処理日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"総画像数: {total}\n")
        f.write(f"成功: {success_count} ({success_count/total*100:.1f}%)\n")
        f.write(f"失敗: {len(failed_files)} ({len(failed_files)/total*100:.1f}%)\n")
        f.write(f"処理時間: {total_time:.1f}秒\n")
        f.write(f"平均: {total_time/total:.1f}秒/画像\n")
        f.write(f"平均信頼度: {avg_confidence:.3f}\n")
        
        if failed_files:
            f.write("\n失敗ファイル:\n")
            for name, reason in failed_files:
                f.write(f"  - {name}: {reason}\n")
    
    logger.info(f"サマリー保存: {summary_path}")


if __name__ == "__main__":
    main()