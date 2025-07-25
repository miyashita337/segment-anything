#!/usr/bin/env python3
"""
kana08バッチ抽出スクリプト
通常のキャラクター抽出パイプライン使用
"""

import numpy as np
import cv2

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 既存モジュールのインポート
from extract_kana04 import extract_main_character
from features.extraction.models.sam_wrapper import SAMWrapper
from features.extraction.models.yolo_wrapper import YOLOWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Kana08BatchExtractor:
    """kana08バッチ抽出器"""
    
    def __init__(self):
        self.input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge")
        
        # モデル初期化
        logger.info("モデル初期化中...")
        self.sam_wrapper = SAMWrapper()
        self.yolo_wrapper = YOLOWrapper()
        
        # 設定（標準的な設定）
        self.quality_method = "balanced"
        self.confidence_threshold = 0.07  # 標準閾値
        
    def process_image(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """単一画像の処理"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "画像読み込み失敗"
            
            # 抽出実行
            result = extract_main_character(
                image,
                self.sam_wrapper,
                self.yolo_wrapper,
                quality_method=self.quality_method
            )
            
            if result is None:
                return False, "キャラクター抽出失敗"
            
            extracted_img, quality_grade, scores = result
            
            # 結果保存
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), extracted_img)
            
            return True, f"品質: {quality_grade}"
            
        except Exception as e:
            return False, f"エラー: {str(e)}"
    
    def run_batch(self):
        """バッチ処理実行"""
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
        start_time = time.time()
        
        # 各画像を処理
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
            
            success, message = self.process_image(image_path)
            
            if success:
                success_count += 1
                logger.info(f"  ✅ 成功 - {message}")
            else:
                failed_files.append((image_path.name, message))
                logger.warning(f"  ❌ 失敗 - {message}")
            
            # 進捗表示
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (total - i)
                logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%) - 残り時間: {remaining:.0f}秒")
        
        # 処理完了
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("バッチ処理完了")
        logger.info(f"総処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
        logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
        
        if failed_files:
            logger.info("失敗ファイル:")
            for name, reason in failed_files:
                logger.info(f"  - {name}: {reason}")
        
        # サマリーファイル作成
        summary_path = self.output_dir / "extraction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"kana08 バッチ抽出サマリー\\n")
            f.write(f"=" * 50 + "\\n")
            f.write(f"処理日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"総画像数: {total}\\n")
            f.write(f"成功: {success_count} ({success_count/total*100:.1f}%)\\n")
            f.write(f"失敗: {len(failed_files)} ({len(failed_files)/total*100:.1f}%)\\n")
            f.write(f"処理時間: {total_time:.1f}秒\\n")
            f.write(f"平均: {total_time/total:.1f}秒/画像\\n")
            
            if failed_files:
                f.write("\\n失敗ファイル:\\n")
                for name, reason in failed_files:
                    f.write(f"  - {name}: {reason}\\n")
        
        logger.info(f"サマリー保存: {summary_path}")


def main():
    """メイン実行"""
    extractor = Kana08BatchExtractor()
    extractor.run_batch()


if __name__ == "__main__":
    main()