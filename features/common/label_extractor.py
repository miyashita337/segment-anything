#!/usr/bin/env python3
"""
Label Extractor - 人間ラベルデータから赤枠座標を自動抽出
Red bounding box coordinate extraction from human-labeled data
"""

import numpy as np
import cv2

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RedBoxLabel:
    """赤枠ラベル情報"""
    filename: str
    red_boxes: List[Dict[str, Any]]
    largest_panel_box: Optional[Dict[str, Any]]
    character_region: Optional[Dict[str, Any]]
    image_size: Tuple[int, int]  # (width, height)


class RedBoxExtractor:
    """人間ラベルから赤枠座標を抽出するクラス"""
    
    def __init__(self, red_threshold: int = 50):
        """
        初期化
        
        Args:
            red_threshold: 赤色判定の閾値
        """
        self.red_threshold = red_threshold
        
    def extract_red_regions(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        画像から赤色で囲まれた領域を抽出
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            赤色領域のリスト
        """
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"画像読み込み失敗: {image_path}")
                return []
            
            height, width = image.shape[:2]
            
            # BGR -> HSV変換で赤色を検出
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 赤色の範囲を定義（HSVで2つの範囲）
            # 低い赤色範囲
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            # 高い赤色範囲
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # 2つのマスクを合成
            red_mask = mask1 + mask2
            
            # ノイズ除去
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 輪郭検出
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            red_regions = []
            for i, contour in enumerate(contours):
                # 小さすぎる領域は無視
                area = cv2.contourArea(contour)
                if area < 1000:  # 最小面積閾値
                    continue
                
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                
                # ポリゴン近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                
                # 重心計算
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                red_region = {
                    "id": i,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": area,
                    "centroid": {"x": cx, "y": cy},
                    "polygon": polygon.reshape(-1, 2).tolist(),
                    "aspect_ratio": h / max(w, 1),
                    "fill_ratio": area / max(w * h, 1)
                }
                
                red_regions.append(red_region)
            
            # 面積順でソート（大きい順）
            red_regions.sort(key=lambda x: x["area"], reverse=True)
            
            logger.info(f"赤色領域検出: {len(red_regions)}個 in {image_path.name}")
            
            return red_regions
            
        except Exception as e:
            logger.error(f"赤色領域抽出エラー {image_path}: {e}")
            return []
    
    def analyze_panel_structure(self, image_path: Path, red_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        コマ構造を分析して「一番大きいコマ」を特定
        
        Args:
            image_path: 画像パス
            red_regions: 検出された赤色領域
            
        Returns:
            コマ構造分析結果
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {}
            
            height, width = image.shape[:2]
            
            # 最大の赤枠を「目的キャラクター領域」と仮定
            if not red_regions:
                return {}
            
            largest_red_region = red_regions[0]  # 面積順でソート済み
            
            # コマ境界線検出（簡易版）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 水平・垂直線検出
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # コマ境界線の候補
            panel_lines = horizontal_lines + vertical_lines
            
            # 最大赤枠を含む領域を「一番大きいコマ」として推定
            char_bbox = largest_red_region["bbox"]
            
            # コマ境界の推定（赤枠の周辺領域を分析）
            margin = 50
            panel_x1 = max(0, char_bbox["x"] - margin)
            panel_y1 = max(0, char_bbox["y"] - margin)
            panel_x2 = min(width, char_bbox["x"] + char_bbox["width"] + margin)
            panel_y2 = min(height, char_bbox["y"] + char_bbox["height"] + margin)
            
            estimated_panel = {
                "bbox": {
                    "x": panel_x1,
                    "y": panel_y1,
                    "width": panel_x2 - panel_x1,
                    "height": panel_y2 - panel_y1
                },
                "area": (panel_x2 - panel_x1) * (panel_y2 - panel_y1),
                "confidence": 0.7  # 推定の信頼度
            }
            
            analysis_result = {
                "largest_panel": estimated_panel,
                "character_region": largest_red_region,
                "panel_detection_method": "red_box_expansion",
                "image_size": {"width": width, "height": height}
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"コマ構造分析エラー {image_path}: {e}")
            return {}
    
    def process_label_dataset(self, dataset_dirs: List[Path]) -> List[RedBoxLabel]:
        """
        ラベルデータセット全体を処理
        
        Args:
            dataset_dirs: ラベルデータディレクトリのリスト
            
        Returns:
            抽出結果のリスト
        """
        all_labels = []
        
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                logger.warning(f"ディレクトリが存在しません: {dataset_dir}")
                continue
            
            image_files = list(dataset_dir.glob("*.jpg"))
            logger.info(f"処理開始: {dataset_dir.name} - {len(image_files)}ファイル")
            
            for image_path in image_files:
                try:
                    # 赤色領域抽出
                    red_regions = self.extract_red_regions(image_path)
                    
                    # コマ構造分析
                    panel_analysis = self.analyze_panel_structure(image_path, red_regions)
                    
                    # 結果をまとめる
                    image = cv2.imread(str(image_path))
                    height, width = image.shape[:2] if image is not None else (0, 0)
                    
                    label = RedBoxLabel(
                        filename=image_path.name,
                        red_boxes=red_regions,
                        largest_panel_box=panel_analysis.get("largest_panel"),
                        character_region=panel_analysis.get("character_region"),
                        image_size=(width, height)
                    )
                    
                    all_labels.append(label)
                    
                except Exception as e:
                    logger.error(f"ファイル処理エラー {image_path}: {e}")
                    continue
        
        logger.info(f"ラベル抽出完了: {len(all_labels)}ファイル処理")
        return all_labels
    
    def save_labels_to_json(self, labels: List[RedBoxLabel], output_path: Path):
        """
        抽出したラベルをJSONファイルに保存
        
        Args:
            labels: ラベルデータ
            output_path: 出力ファイルパス
        """
        try:
            # DataClassを辞書に変換
            labels_dict = []
            for label in labels:
                label_dict = {
                    "filename": label.filename,
                    "red_boxes": label.red_boxes,
                    "largest_panel_box": label.largest_panel_box,
                    "character_region": label.character_region,
                    "image_size": label.image_size
                }
                labels_dict.append(label_dict)
            
            # JSON保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labels_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ラベルデータ保存完了: {output_path}")
            
        except Exception as e:
            logger.error(f"JSON保存エラー: {e}")


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    
    # ラベルデータディレクトリ
    dataset_dirs = [
        Path("/mnt/c/AItools/lora/train/yado/org/kana05_cursor"),
        Path("/mnt/c/AItools/lora/train/yado/org/kana07_cursor"),
        Path("/mnt/c/AItools/lora/train/yado/org/kana08_cursor")
    ]
    
    # 出力パス
    output_path = Path("/mnt/c/AItools/segment-anything/extracted_labels.json")
    
    # 赤枠抽出実行
    extractor = RedBoxExtractor()
    labels = extractor.process_label_dataset(dataset_dirs)
    
    # 結果保存
    extractor.save_labels_to_json(labels, output_path)
    
    # 統計出力
    total_files = len(labels)
    files_with_red_boxes = len([l for l in labels if l.red_boxes])
    
    print(f"\n📊 ラベル抽出結果:")
    print(f"  総ファイル数: {total_files}")
    print(f"  赤枠検出成功: {files_with_red_boxes}")
    print(f"  成功率: {files_with_red_boxes/max(total_files,1)*100:.1f}%")
    print(f"  出力ファイル: {output_path}")


if __name__ == "__main__":
    main()