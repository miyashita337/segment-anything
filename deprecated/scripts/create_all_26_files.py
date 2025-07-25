#!/usr/bin/env python3
"""
Create All 26 Files for Review
レビュー用に26ファイル全てを作成
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

import json
import logging
import shutil
from features.extraction.robust_extractor import RobustCharacterExtractor
from typing import Any, Dict, List

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_all_files_for_review():
    """レビュー用に26ファイル全てを作成"""
    logger.info("📋 レビュー用26ファイル作成開始")
    
    # ディレクトリ設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_all_26_for_review")
    existing_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_fixed_character_detection")
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 出力ディレクトリ: {output_dir}")
    
    # 入力画像ファイル取得
    image_files = sorted(list(input_dir.glob("*.jpg")))
    logger.info(f"📁 処理対象: {len(image_files)}画像")
    
    # 既存の成功ファイルをコピー
    existing_files = list(existing_dir.glob("*.jpg")) if existing_dir.exists() else []
    existing_names = {f.name for f in existing_files}
    
    for existing_file in existing_files:
        dest_file = output_dir / existing_file.name
        shutil.copy2(existing_file, dest_file)
        logger.info(f"✅ 既存成功ファイルコピー: {existing_file.name}")
    
    # 残りのファイルを処理
    extractor = RobustCharacterExtractor()
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        filename = image_path.name
        output_path = output_dir / filename
        
        if filename in existing_names:
            # 既にコピー済み
            results.append({
                "filename": filename,
                "status": "existing_success",
                "source": "previous_batch"
            })
            continue
        
        logger.info(f"📸 [{i}/{len(image_files)}] 処理中: {filename}")
        
        try:
            # 修正版抽出実行（失敗してもベスト候補で画像作成）
            extraction_result = extractor.extract_character_robust(
                image_path, output_path, verbose=False
            )
            
            if extraction_result.get("success", False):
                logger.info(f"✅ 新規抽出成功: {filename}")
                results.append({
                    "filename": filename,
                    "status": "new_extraction_success",
                    "quality_score": extraction_result.get("quality_score", 0.0)
                })
            else:
                # 失敗した場合でも何らかの候補があれば出力
                logger.info(f"⚠️ 抽出失敗、代替処理実行: {filename}")
                success = create_fallback_extraction(image_path, output_path)
                results.append({
                    "filename": filename,
                    "status": "fallback_created" if success else "failed",
                    "quality_score": 0.0
                })
                
        except Exception as e:
            logger.error(f"❌ 処理エラー: {filename} - {e}")
            # エラー時も代替処理
            success = create_fallback_extraction(image_path, output_path)
            results.append({
                "filename": filename,
                "status": "fallback_created" if success else "error",
                "error": str(e)
            })
    
    # 結果サマリー作成
    summary = {
        "total_files": len(image_files),
        "existing_success": len([r for r in results if r["status"] == "existing_success"]),
        "new_extraction_success": len([r for r in results if r["status"] == "new_extraction_success"]),
        "fallback_created": len([r for r in results if r["status"] == "fallback_created"]),
        "failed": len([r for r in results if r["status"] in ["failed", "error"]]),
        "results": results
    }
    
    # サマリー保存
    summary_path = output_dir / "review_creation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 最終確認
    output_files = list(output_dir.glob("*.jpg"))
    logger.info(f"🎯 最終結果: {len(output_files)}/26ファイル作成完了")
    
    if len(output_files) == 26:
        logger.info("✅ 全26ファイル作成成功 - レビュー準備完了")
    else:
        logger.warning(f"⚠️ {26 - len(output_files)}ファイル不足")
    
    return summary


def create_fallback_extraction(image_path: Path, output_path: Path) -> bool:
    """
    代替抽出処理：失敗時でもレビュー用画像を作成
    """
    try:
        # 元画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # 画像を適度にリサイズ（レビュー用）
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # 中央部分を抽出（フォールバック）
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        extract_size = min(w, h) // 2
        
        x1 = max(0, center_x - extract_size // 2)
        y1 = max(0, center_y - extract_size // 2)
        x2 = min(w, x1 + extract_size)
        y2 = min(h, y1 + extract_size)
        
        # 中央部分切り出し
        cropped = image[y1:y2, x1:x2]
        
        # 保存
        cv2.imwrite(str(output_path), cropped)
        logger.info(f"📋 代替画像作成: {output_path.name} ({cropped.shape[1]}x{cropped.shape[0]})")
        
        return True
        
    except Exception as e:
        logger.error(f"代替抽出失敗: {e}")
        return False


if __name__ == "__main__":
    summary = create_all_files_for_review()
    
    # 結果表示
    print(f"\n📊 レビュー用26ファイル作成結果:")
    print(f"  既存成功: {summary['existing_success']}件")
    print(f"  新規成功: {summary['new_extraction_success']}件") 
    print(f"  代替作成: {summary['fallback_created']}件")
    print(f"  失敗: {summary['failed']}件")
    print(f"  合計: {summary['total_files']}件")
    print(f"\n📂 出力: /mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_all_26_for_review/")