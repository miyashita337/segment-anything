#!/usr/bin/env python3
"""
kana08バッチ抽出スクリプト（Week 4改善版）
改良されたSCI評価システムを適用
"""

import numpy as np
import cv2
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 改善された評価システムのインポート
from features.evaluation.utils.face_detection import AnimeFaceDetector
from features.evaluation.utils.enhanced_sci_processor import EnhancedSCIProcessor
from features.extraction.models.sam_wrapper import SAMWrapper
from features.extraction.models.yolo_wrapper import YOLOWrapper
from features.processing.postprocessing.postprocessing import apply_mask_to_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Kana08ImprovedExtractor:
    """kana08改善版バッチ抽出器"""
    
    def __init__(self):
        self.input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge")
        
        # モデル初期化
        logger.info("モデル初期化中...")
        self.sam_wrapper = SAMWrapper()
        self.yolo_wrapper = YOLOWrapper()
        self.face_detector = AnimeFaceDetector()
        self.sci_processor = EnhancedSCIProcessor()
        
        # 設定
        self.quality_method = "balanced"
        self.confidence_threshold = 0.07  # アニメ特化閾値
        
    def extract_character(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, str, Dict[str, float]]]:
        """キャラクター抽出（改善版SCI評価付き）"""
        try:
            # YOLO検出
            detections = self.yolo_wrapper.detect(image, confidence_threshold=self.confidence_threshold)
            if not detections:
                return None
            
            # 最大の検出結果を使用
            best_detection = max(detections, key=lambda d: d['area'])
            x1, y1, x2, y2 = best_detection['bbox']
            
            # SAMマスク生成
            box_prompt = np.array([x1, y1, x2, y2])
            masks, scores, _ = self.sam_wrapper.predict(
                image,
                box=box_prompt,
                multimask_output=True
            )
            
            if len(masks) == 0:
                return None
            
            # 最良マスク選択
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            # SCI評価計算
            sci_scores = self.sci_processor.calculate_sci(
                image, 
                mask,
                bbox=(x1, y1, x2, y2)
            )
            
            # 総合評価
            overall_sci = sci_scores['overall_sci']
            anime_sci = sci_scores['anime_sci']
            
            # 品質グレード判定
            if anime_sci >= 0.8:
                quality_grade = 'A'
            elif anime_sci >= 0.7:
                quality_grade = 'B'
            elif anime_sci >= 0.6:
                quality_grade = 'C'
            elif anime_sci >= 0.5:
                quality_grade = 'D'
            else:
                quality_grade = 'E'
            
            # マスク適用
            extracted_img = apply_mask_to_image(image, mask)
            
            return extracted_img, quality_grade, sci_scores
            
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
                return False, "キャラクター抽出失敗", None
            
            extracted_img, quality_grade, sci_scores = result
            
            # 結果保存
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), extracted_img)
            
            return True, f"品質: {quality_grade}", sci_scores
            
        except Exception as e:
            return False, f"エラー: {str(e)}", None
    
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
        sci_scores_list = []
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        start_time = time.time()
        
        # 各画像を処理
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
            
            success, message, sci_scores = self.process_image(image_path)
            
            if success:
                success_count += 1
                logger.info(f"  ✅ 成功 - {message}")
                
                # SCI統計収集
                if sci_scores:
                    sci_scores_list.append(sci_scores)
                    # グレード集計
                    grade = message.split(': ')[1]
                    if grade in grade_counts:
                        grade_counts[grade] += 1
                    
                    # 詳細ログ
                    logger.info(f"    SCI anime: {sci_scores['anime_sci']:.3f}")
                    logger.info(f"    Face score: {sci_scores['face_score']:.3f}")
                    logger.info(f"    Pose score: {sci_scores['pose_score']:.3f}")
            else:
                failed_files.append((image_path.name, message))
                logger.warning(f"  ❌ 失敗 - {message}")
                grade_counts['F'] += 1
            
            # 進捗表示
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (total - i)
                logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%) - 残り時間: {remaining:.0f}秒")
        
        # 処理完了
        total_time = time.time() - start_time
        
        # SCI統計計算
        if sci_scores_list:
            avg_sci = np.mean([s['overall_sci'] for s in sci_scores_list])
            avg_anime_sci = np.mean([s['anime_sci'] for s in sci_scores_list])
            avg_face = np.mean([s['face_score'] for s in sci_scores_list])
            avg_pose = np.mean([s['pose_score'] for s in sci_scores_list])
        else:
            avg_sci = avg_anime_sci = avg_face = avg_pose = 0.0
        
        logger.info("=" * 50)
        logger.info("バッチ処理完了")
        logger.info(f"総処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
        logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
        logger.info("")
        logger.info("品質分布:")
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = grade_counts[grade]
            percentage = count / total * 100
            logger.info(f"  {grade}評価: {count}枚 ({percentage:.1f}%)")
        logger.info("")
        logger.info("SCI統計:")
        logger.info(f"  平均SCI: {avg_sci:.3f}")
        logger.info(f"  平均SCI anime: {avg_anime_sci:.3f}")
        logger.info(f"  平均Face score: {avg_face:.3f}")
        logger.info(f"  平均Pose score: {avg_pose:.3f}")
        
        if failed_files:
            logger.info("")
            logger.info("失敗ファイル:")
            for name, reason in failed_files:
                logger.info(f"  - {name}: {reason}")
        
        # 詳細レポート作成
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_images": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "processing_time": total_time,
            "avg_processing_time": total_time / total,
            "grade_distribution": grade_counts,
            "sci_statistics": {
                "avg_sci": avg_sci,
                "avg_anime_sci": avg_anime_sci,
                "avg_face_score": avg_face,
                "avg_pose_score": avg_pose
            },
            "failed_files": failed_files,
            "individual_results": []
        }
        
        # 個別結果追加
        for i, (image_path, sci_scores) in enumerate(zip(image_files[:len(sci_scores_list)], sci_scores_list)):
            report["individual_results"].append({
                "filename": image_path.name,
                "sci_scores": sci_scores
            })
        
        # JSONレポート保存
        report_path = self.output_dir / "kana08_extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"詳細レポート保存: {report_path}")
        
        # サマリーファイル作成
        summary_path = self.output_dir / "extraction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"kana08 バッチ抽出サマリー（Week 4改善版）\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"処理日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総画像数: {total}\n")
            f.write(f"成功: {success_count} ({success_count/total*100:.1f}%)\n")
            f.write(f"失敗: {len(failed_files)} ({len(failed_files)/total*100:.1f}%)\n")
            f.write(f"処理時間: {total_time:.1f}秒\n")
            f.write(f"平均: {total_time/total:.1f}秒/画像\n")
            f.write(f"\n品質分布:\n")
            for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
                count = grade_counts[grade]
                f.write(f"  {grade}: {count}枚 ({count/total*100:.1f}%)\n")
            f.write(f"\nSCI統計:\n")
            f.write(f"  平均SCI anime: {avg_anime_sci:.3f}\n")
            f.write(f"  平均Face score: {avg_face:.3f}\n")
            f.write(f"  平均Pose score: {avg_pose:.3f}\n")
            
            if failed_files:
                f.write("\n失敗ファイル:\n")
                for name, reason in failed_files:
                    f.write(f"  - {name}: {reason}\n")
        
        logger.info(f"サマリー保存: {summary_path}")


def main():
    """メイン実行"""
    extractor = Kana08ImprovedExtractor()
    extractor.run_batch()


if __name__ == "__main__":
    main()