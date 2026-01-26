#!/usr/bin/env python3
"""
正解クロッピング範囲抽出システム
赤枠画像から座標を自動検出してJSONデータベース化
"""

import numpy as np
import cv2

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CorrectAnnotationExtractor:
    """正解アノテーション抽出器"""

    def __init__(self, annotation_dir: Path, output_dir: Path):
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 赤色検出パラメータ
        self.red_lower = np.array([0, 50, 50])  # HSV下限
        self.red_upper = np.array([10, 255, 255])  # HSV上限
        self.red_lower2 = np.array([170, 50, 50])  # HSV下限（赤の別範囲）
        self.red_upper2 = np.array([180, 255, 255])  # HSV上限（赤の別範囲）

        # 矩形検出パラメータ
        self.min_area = 1000  # 最小面積
        self.min_aspect_ratio = 0.2  # 最小アスペクト比
        self.max_aspect_ratio = 5.0  # 最大アスペクト比

    def detect_red_rectangles(self, image_path: Path) -> List[Dict]:
        """赤い矩形を検出"""
        try:
            # 画像読み込み
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"画像読み込み失敗: {image_path}")
                return []

            height, width = img.shape[:2]
            logger.info(f"画像サイズ: {width}x{height} - {image_path.name}")

            # HSV変換
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 赤色マスク作成（2つの範囲）
            mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # ノイズ除去
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            # 輪郭検出
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangles = []
            for contour in contours:
                # 矩形近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 面積チェック
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                # 境界矩形取得
                x, y, w, h = cv2.boundingRect(contour)

                # アスペクト比チェック
                aspect_ratio = w / h if h > 0 else 0
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # 矩形が画像境界に近すぎないかチェック
                if x < 5 or y < 5 or (x + w) > (width - 5) or (y + h) > (height - 5):
                    # 境界矩形の場合、より厳しい条件でチェック
                    if area < self.min_area * 2:
                        continue

                rectangle_info = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "aspect_ratio": float(aspect_ratio),
                    "center_x": int(x + w / 2),
                    "center_y": int(y + h / 2),
                    "relative_x": float(x / width),
                    "relative_y": float(y / height),
                    "relative_width": float(w / width),
                    "relative_height": float(h / height),
                }

                rectangles.append(rectangle_info)
                logger.info(f"  矩形検出: ({x},{y}) {w}x{h} 面積={area} 比率={aspect_ratio:.2f}")

            # 面積でソート（大きい順）
            rectangles.sort(key=lambda r: r["area"], reverse=True)

            return rectangles

        except Exception as e:
            logger.error(f"赤枠検出エラー {image_path}: {e}")
            return []

    def extract_all_annotations(self) -> Dict:
        """全アノテーション抽出"""
        logger.info("🚀 正解アノテーション抽出開始")
        logger.info(f"入力ディレクトリ: {self.annotation_dir}")

        # 画像ファイル取得
        image_files = list(self.annotation_dir.glob("*.jpg")) + list(
            self.annotation_dir.glob("*.png")
        )
        image_files.sort()

        logger.info(f"対象画像数: {len(image_files)}")

        results = {
            "metadata": {
                "total_images": len(image_files),
                "extraction_method": "red_rectangle_detection",
                "parameters": {
                    "min_area": self.min_area,
                    "min_aspect_ratio": self.min_aspect_ratio,
                    "max_aspect_ratio": self.max_aspect_ratio,
                },
            },
            "annotations": {},
        }

        success_count = 0
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"[{i:2d}/{len(image_files)}] 処理中: {image_file.name}")

            rectangles = self.detect_red_rectangles(image_file)

            if rectangles:
                results["annotations"][image_file.name] = {
                    "file_path": str(image_file),
                    "rectangles": rectangles,
                    "primary_rectangle": rectangles[0],  # 最大面積の矩形
                    "rectangle_count": len(rectangles),
                }
                success_count += 1
                logger.info(f"  ✅ {len(rectangles)}個の矩形を検出")
            else:
                results["annotations"][image_file.name] = {
                    "file_path": str(image_file),
                    "rectangles": [],
                    "primary_rectangle": None,
                    "rectangle_count": 0,
                }
                logger.warning(f"  ⚠️ 矩形検出なし")

        results["metadata"]["successful_extractions"] = success_count
        results["metadata"]["success_rate"] = success_count / len(image_files) if image_files else 0

        return results

    def save_results(self, results: Dict, filename: str = "correct_annotations.json"):
        """結果保存"""
        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 結果保存: {output_file}")
        return output_file

    def generate_visual_verification(self, results: Dict):
        """視覚的検証画像生成"""
        logger.info("🎨 視覚的検証画像生成開始")

        verification_dir = self.output_dir / "visual_verification"
        verification_dir.mkdir(exist_ok=True)

        for filename, annotation in results["annotations"].items():
            if not annotation["rectangles"]:
                continue

            try:
                # 元画像読み込み
                img_path = Path(annotation["file_path"])
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # 検出された矩形を描画
                for i, rect in enumerate(annotation["rectangles"]):
                    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]

                    # 主要矩形は緑、その他は青で描画
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)
                    thickness = 3 if i == 0 else 2

                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

                    # 情報テキスト
                    info_text = f"{i+1}: {w}x{h}"
                    cv2.putText(
                        img, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

                # 保存
                output_path = verification_dir / f"verified_{filename}"
                cv2.imwrite(str(output_path), img)

            except Exception as e:
                logger.error(f"検証画像生成エラー {filename}: {e}")

        logger.info(f"📁 検証画像保存: {verification_dir}")


def main():
    """メイン実行"""
    # パス設定
    annotation_dir = Path(
        "C:/AItools/lora/train/yado/tracker-workspace/P1-B004/correct_annotations"
    )
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis")

    # 抽出器初期化
    extractor = CorrectAnnotationExtractor(annotation_dir, output_dir)

    # アノテーション抽出
    results = extractor.extract_all_annotations()

    # 結果保存
    output_file = extractor.save_results(results)

    # 視覚的検証画像生成
    extractor.generate_visual_verification(results)

    # サマリー表示
    total = results["metadata"]["total_images"]
    success = results["metadata"]["successful_extractions"]
    success_rate = results["metadata"]["success_rate"]

    logger.info("=" * 60)
    logger.info("🎯 正解アノテーション抽出完了")
    logger.info(f"📊 処理結果: {success}/{total}枚成功 ({success_rate:.1%})")
    logger.info(f"📄 出力ファイル: {output_file}")
    logger.info("=" * 60)

    return 0 if success_rate > 0.8 else 1


if __name__ == "__main__":
    exit(main())
