#!/usr/bin/env python3
"""
拡張SAMパイプライン
deprecatedから復旧された高精度キャラクター抽出システム

P1-A001: 改善コード復旧プロジェクト
- パフォーマンス監視システム
- テキスト検出・除去機能
- 高精度品質評価システム
- 5段階品質評価手法
"""

import numpy as np
import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import torch

# SAM関連インポート
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from segment_anything.utils.amg import batched_mask_to_box
except ImportError:
    from core.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from core.segment_anything.utils.amg import batched_mask_to_box

import argparse
import gc
import glob
import os
import psutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# YOLOv8関連インポート
from ultralytics import YOLO

# OCR・テキスト検出関連インポート
try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class PerformanceMonitor:
    """
    パフォーマンス監視システム
    deprecated/backup から復旧
    """

    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.current_stage = None

    def start_monitoring(self):
        """監視開始"""
        self.start_time = time.time()
        self.stage_times = {}
        print("📊 パフォーマンス監視開始")
        self.log_system_info()

    def start_stage(self, stage_name: str):
        """ステージ開始"""
        self.current_stage = stage_name
        self.stage_times[stage_name] = time.time()
        print(f"⏳ 開始: {stage_name}")

    def end_stage(self):
        """ステージ終了"""
        if not self.current_stage:
            return

        elapsed = time.time() - self.stage_times[self.current_stage]

        # メモリ使用量記録
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0

        print(
            f"✅ 完了: {self.current_stage} ({elapsed:.2f}秒, RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB)"
        )

        self.stage_times[self.current_stage] = elapsed
        self.current_stage = None

        # ガベージコレクション実行
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_gpu_memory(self):
        """GPU メモリ使用量を取得"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0
        except:
            return 0

    def log_system_info(self):
        """システム情報をログ出力"""
        print("=== システム情報 ===")
        print(f"CPU: {psutil.cpu_count()} コア")
        print(f"RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"GPU RAM: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f}GB"
            )
        print("=" * 20)

    def print_summary(self):
        """処理時間サマリーを出力"""
        if not self.start_time:
            return

        total_time = time.time() - self.start_time

        print("\\n=== パフォーマンス サマリー ===")
        print(f"総処理時間: {total_time:.2f}秒")

        for stage, duration in self.stage_times.items():
            if isinstance(duration, float):
                percentage = (duration / total_time) * 100
                print(f"  {stage}: {duration:.2f}秒 ({percentage:.1f}%)")
        print("=" * 30)


def setup_japanese_font():
    """
    matplotlib用の日本語フォントを設定
    deprecated/backup から復旧
    """
    try:
        # Windows環境での日本語フォント候補
        font_candidates = [
            "Yu Gothic UI",
            "Meiryo",
            "MS Gothic",
            "Hiragino Sans",
            "Noto Sans CJK JP",
            "DejaVu Sans",  # フォールバック
        ]

        # 利用可能なフォントを検索
        available_fonts = [font.name for font in fm.fontManager.ttflist]

        selected_font = None
        for font_name in font_candidates:
            if font_name in available_fonts:
                selected_font = font_name
                break

        if selected_font:
            plt.rcParams["font.family"] = selected_font
            print(f"✅ 日本語フォント設定完了: {selected_font}")
        else:
            # フォールバック: Unicode対応
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False
            print("⚠️ 日本語フォントが見つかりません。英語表示にフォールバックします")

    except Exception as e:
        print(f"⚠️ フォント設定エラー: {e}")
        # 最小限の設定
        plt.rcParams["axes.unicode_minus"] = False


class TextDetector:
    """
    テキスト検出・除去クラス
    deprecated/backup から復旧
    """

    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        if self.ocr_available:
            try:
                self.reader = easyocr.Reader(["ja", "en"], gpu=torch.cuda.is_available())
                print("✅ OCR初期化完了 (EasyOCR)")
            except Exception as e:
                print(f"⚠️ OCR初期化失敗: {e}")
                self.ocr_available = False
        else:
            print("⚠️ EasyOCRが利用できません")

    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """テキスト領域検出"""
        if not self.ocr_available:
            return []

        try:
            # OCRでテキスト検出
            results = self.reader.readtext(image, paragraph=False)

            text_regions = []
            for bbox, text, confidence in results:
                if confidence > 0.5:  # 信頼度閾値
                    # 境界ボックス正規化
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    text_regions.append((x1, y1, x2, y2))

            print(f"📝 テキスト領域検出: {len(text_regions)}個")
            return text_regions

        except Exception as e:
            print(f"⚠️ テキスト検出エラー: {e}")
            return []

    def create_text_mask(
        self, image: np.ndarray, text_regions: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """テキスト除去マスク作成"""
        h, w = image.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)

        for x1, y1, x2, y2 in text_regions:
            # マージン追加（テキスト周辺も除去）
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            text_mask[y1:y2, x1:x2] = 255

        return text_mask


class QualityEvaluator:
    """
    5段階品質評価システム
    deprecated/backup から復旧・改良
    """

    def __init__(self):
        self.evaluation_methods = {
            "balanced": self._evaluate_balanced,
            "confidence_priority": self._evaluate_confidence_priority,
            "size_priority": self._evaluate_size_priority,
            "fullbody_priority": self._evaluate_fullbody_priority,
            "central_priority": self._evaluate_central_priority,
        }

    def evaluate_extraction_quality(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        method: str = "balanced",
    ) -> Dict:
        """抽出品質評価"""
        if method not in self.evaluation_methods:
            method = "balanced"

        return self.evaluation_methods[method](image, mask, bbox)

    def _evaluate_balanced(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """バランス重視評価"""
        # 基本メトリクス計算
        area_ratio = self._calculate_area_ratio(mask)
        compactness = self._calculate_compactness(mask)
        centrality = self._calculate_centrality(mask, image.shape[:2])
        fill_ratio = self._calculate_fill_ratio(mask, bbox)

        # バランス重視スコア
        quality_score = area_ratio * 0.3 + compactness * 0.25 + centrality * 0.25 + fill_ratio * 0.2

        return {
            "quality_score": quality_score,
            "area_ratio": area_ratio,
            "compactness": compactness,
            "centrality": centrality,
            "fill_ratio": fill_ratio,
            "method": "balanced",
        }

    def _evaluate_confidence_priority(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """信頼度優先評価"""
        # エッジ連続性チェック
        edge_continuity = self._calculate_edge_continuity(mask)
        noise_level = self._calculate_noise_level(mask)

        quality_score = edge_continuity * 0.6 + (1 - noise_level) * 0.4

        return {
            "quality_score": quality_score,
            "edge_continuity": edge_continuity,
            "noise_level": noise_level,
            "method": "confidence_priority",
        }

    def _evaluate_size_priority(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """サイズ優先評価"""
        area_ratio = self._calculate_area_ratio(mask)
        bbox_ratio = self._calculate_bbox_ratio(bbox, image.shape[:2])

        # サイズ重視スコア
        quality_score = area_ratio * 0.7 + bbox_ratio * 0.3

        return {
            "quality_score": quality_score,
            "area_ratio": area_ratio,
            "bbox_ratio": bbox_ratio,
            "method": "size_priority",
        }

    def _evaluate_fullbody_priority(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """全身検出優先評価"""
        aspect_ratio = self._calculate_aspect_ratio(bbox)
        vertical_coverage = self._calculate_vertical_coverage(bbox, image.shape[0])

        # 全身らしさスコア（縦長で画像の大部分をカバー）
        fullbody_score = self._calculate_fullbody_likelihood(aspect_ratio, vertical_coverage)
        quality_score = fullbody_score * 0.8 + self._calculate_area_ratio(mask) * 0.2

        return {
            "quality_score": quality_score,
            "fullbody_score": fullbody_score,
            "aspect_ratio": aspect_ratio,
            "vertical_coverage": vertical_coverage,
            "method": "fullbody_priority",
        }

    def _evaluate_central_priority(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """中心位置優先評価"""
        centrality = self._calculate_centrality(mask, image.shape[:2])
        center_distance = self._calculate_center_distance(bbox, image.shape[:2])

        quality_score = centrality * 0.6 + (1 - center_distance) * 0.4

        return {
            "quality_score": quality_score,
            "centrality": centrality,
            "center_distance": center_distance,
            "method": "central_priority",
        }

    def _calculate_area_ratio(self, mask: np.ndarray) -> float:
        """面積比率計算"""
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = np.sum(mask > 0)
        return min(mask_pixels / total_pixels, 1.0)

    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """コンパクトネス計算"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        if perimeter == 0:
            return 0.0

        compactness = 4 * np.pi * area / (perimeter**2)
        return min(compactness, 1.0)

    def _calculate_centrality(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """中心性計算"""
        h, w = image_shape
        center_y, center_x = h // 2, w // 2

        # マスクの重心計算
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return 0.0

        mask_center_x = int(moments["m10"] / moments["m00"])
        mask_center_y = int(moments["m01"] / moments["m00"])

        # 中心からの距離
        distance = np.sqrt((mask_center_x - center_x) ** 2 + (mask_center_y - center_y) ** 2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        return max(0, 1 - distance / max_distance)

    def _calculate_fill_ratio(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """フィル率計算"""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area == 0:
            return 0.0

        mask_in_bbox = mask[y1:y2, x1:x2]
        mask_pixels = np.sum(mask_in_bbox > 0)

        return min(mask_pixels / bbox_area, 1.0)

    def _calculate_edge_continuity(self, mask: np.ndarray) -> float:
        """エッジ連続性計算"""
        edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # 最大輪郭の連続性評価
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)

        if area == 0:
            return 0.0

        return min(perimeter / (2 * np.sqrt(np.pi * area)), 1.0)

    def _calculate_noise_level(self, mask: np.ndarray) -> float:
        """ノイズレベル計算"""
        # 小さな連結成分をノイズとして検出
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

        if num_labels <= 1:
            return 0.0

        # 最大成分以外の小成分をノイズとみなす
        areas = stats[1:, cv2.CC_STAT_AREA]  # 背景(0)を除く
        max_area = np.max(areas)
        noise_area = np.sum(areas[areas < max_area * 0.1])  # 最大面積の10%未満をノイズ
        total_area = np.sum(areas)

        return noise_area / total_area if total_area > 0 else 0.0

    def _calculate_bbox_ratio(
        self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]
    ) -> float:
        """境界ボックス比率計算"""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_shape[0] * image_shape[1]
        return min(bbox_area / image_area, 1.0)

    def _calculate_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """アスペクト比計算"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width == 0:
            return 0.0
        return height / width

    def _calculate_vertical_coverage(
        self, bbox: Tuple[int, int, int, int], image_height: int
    ) -> float:
        """垂直カバレッジ計算"""
        _, y1, _, y2 = bbox
        coverage = (y2 - y1) / image_height
        return min(coverage, 1.0)

    def _calculate_fullbody_likelihood(
        self, aspect_ratio: float, vertical_coverage: float
    ) -> float:
        """全身らしさ計算"""
        # 理想的な全身アスペクト比: 1.5-3.0
        ideal_aspect = 2.0
        aspect_score = 1.0 - min(abs(aspect_ratio - ideal_aspect) / ideal_aspect, 1.0)

        # 垂直カバレッジは高いほど良い
        coverage_score = vertical_coverage

        return (aspect_score + coverage_score) / 2

    def _calculate_center_distance(
        self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]
    ) -> float:
        """中心距離計算"""
        h, w = image_shape
        image_center_x, image_center_y = w // 2, h // 2

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) // 2
        bbox_center_y = (y1 + y2) // 2

        distance = np.sqrt(
            (bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2
        )
        max_distance = np.sqrt(image_center_x**2 + image_center_y**2)

        return distance / max_distance if max_distance > 0 else 0.0


class EnhancedSAMPipeline:
    """
    拡張SAMパイプライン
    deprecatedから復旧された高精度機能統合版
    """

    def __init__(
        self, sam_checkpoint: str = "sam_vit_h_4b8939.pth", yolo_model: str = "yolov8x.pt"
    ):
        self.sam_checkpoint = sam_checkpoint
        self.yolo_model = yolo_model

        # コンポーネント初期化
        self.monitor = PerformanceMonitor()
        self.text_detector = TextDetector()
        self.quality_evaluator = QualityEvaluator()

        # モデル初期化フラグ
        self.sam_initialized = False
        self.yolo_initialized = False

        print("🚀 Enhanced SAM Pipeline 初期化完了")

    def initialize_models(self):
        """モデル初期化"""
        self.monitor.start_stage("モデル初期化")

        # CUDA設定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 デバイス: {device}")

        # SAM初期化
        if Path(self.sam_checkpoint).exists():
            self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
            self.sam.to(device=device)
            self.sam_initialized = True
            print("✅ SAM初期化完了")
        else:
            print(f"❌ SAMモデルファイルが見つかりません: {self.sam_checkpoint}")

        # YOLO初期化
        try:
            self.yolo = YOLO(self.yolo_model)
            self.yolo_initialized = True
            print("✅ YOLO初期化完了")
        except Exception as e:
            print(f"❌ YOLO初期化失敗: {e}")

        self.monitor.end_stage()

    def process_image(
        self, image_path: str, output_dir: str, quality_method: str = "balanced"
    ) -> Dict:
        """画像処理実行"""
        self.monitor.start_monitoring()

        # モデル初期化チェック
        if not self.sam_initialized or not self.yolo_initialized:
            self.initialize_models()

        # 画像読み込み
        self.monitor.start_stage("画像読み込み")
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "画像読み込み失敗"}
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.monitor.end_stage()

        # テキスト検出・除去
        self.monitor.start_stage("テキスト処理")
        text_regions = self.text_detector.detect_text_regions(image_rgb)
        text_mask = self.text_detector.create_text_mask(image_rgb, text_regions)
        self.monitor.end_stage()

        # YOLO検出
        self.monitor.start_stage("YOLO検出")
        yolo_results = self.yolo(image_rgb, conf=0.07)  # アニメ特化閾値
        detections = []
        for result in yolo_results:
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append((bbox, conf))
        self.monitor.end_stage()

        # SAM分割
        self.monitor.start_stage("SAM分割")
        best_result = None
        best_quality = 0.0

        for bbox, confidence in detections:
            # SAM実行
            mask_generator = SamAutomaticMaskGenerator(self.sam)
            masks = mask_generator.generate(image_rgb)

            for mask_data in masks:
                mask = mask_data["segmentation"]

                # テキスト領域との重複チェック
                if self._overlaps_with_text(mask, text_mask):
                    continue

                # 品質評価
                quality_result = self.quality_evaluator.evaluate_extraction_quality(
                    image_rgb, mask, tuple(map(int, bbox)), quality_method
                )

                if quality_result["quality_score"] > best_quality:
                    best_quality = quality_result["quality_score"]
                    best_result = {
                        "mask": mask,
                        "bbox": bbox,
                        "quality": quality_result,
                        "confidence": confidence,
                    }

        self.monitor.end_stage()

        # 結果保存
        if best_result:
            self.monitor.start_stage("結果保存")
            output_path = self._save_result(
                image_rgb, best_result, output_dir, Path(image_path).stem
            )
            self.monitor.end_stage()

            result = {
                "success": True,
                "output_path": output_path,
                "quality_score": best_quality,
                "quality_method": quality_method,
                "bbox": best_result["bbox"].tolist(),
                "confidence": float(best_result["confidence"]),
            }
        else:
            result = {"success": False, "error": "キャラクター検出失敗"}

        self.monitor.print_summary()
        return result

    def _overlaps_with_text(
        self, mask: np.ndarray, text_mask: np.ndarray, threshold: float = 0.3
    ) -> bool:
        """テキスト領域との重複チェック"""
        if text_mask is None or np.sum(text_mask) == 0:
            return False

        overlap = np.logical_and(mask, text_mask > 0)
        overlap_ratio = np.sum(overlap) / np.sum(mask)
        return overlap_ratio > threshold

    def _save_result(self, image: np.ndarray, result: Dict, output_dir: str, filename: str) -> str:
        """結果保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # マスク適用
        mask = result["mask"]
        masked_image = image.copy()
        masked_image[~mask] = 0  # 背景を黒に

        # 保存
        output_path = output_dir / f"{filename}_extracted.png"
        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), masked_image_bgr)

        return str(output_path)


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Enhanced SAM Pipeline")
    parser.add_argument("--input", required=True, help="入力画像パス")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--quality_method",
        default="balanced",
        choices=[
            "balanced",
            "confidence_priority",
            "size_priority",
            "fullbody_priority",
            "central_priority",
        ],
        help="品質評価手法",
    )

    args = parser.parse_args()

    # 日本語フォント設定
    setup_japanese_font()

    # パイプライン実行
    pipeline = EnhancedSAMPipeline()
    result = pipeline.process_image(args.input, args.output, args.quality_method)

    if result["success"]:
        print(f"✅ 処理完了: {result['output_path']}")
        print(f"品質スコア: {result['quality_score']:.3f}")
    else:
        print(f"❌ 処理失敗: {result.get('error', '不明なエラー')}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
