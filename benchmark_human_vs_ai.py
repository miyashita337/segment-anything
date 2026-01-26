#!/usr/bin/env python3
"""
人間ラベル vs AI抽出精度ベンチマークシステム
ユーザーが作成した赤枠ラベルを正解として、現在のAIシステムの精度を測定
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

import json
import logging

# SAM + YOLO システムのインポート
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent))
from segment_anything import SamPredictor, sam_model_registry

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""

    image_id: str
    image_path: str
    human_bbox: Tuple[int, int, int, int]  # x, y, w, h
    ai_bbox: Optional[Tuple[int, int, int, int]]
    iou_score: float
    extraction_success: bool
    processing_time: float
    error_message: str = ""


class HumanVsAIBenchmark:
    """人間ラベル vs AI抽出ベンチマーク"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.labels_file = project_root / "extracted_labels.json"
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 人間ラベルデータ読み込み
        self.human_labels = self.load_human_labels()
        logger.info(f"人間ラベルデータ読み込み: {len(self.human_labels)}件")

        # SAM + YOLOモデル初期化
        self.init_models()

    def init_models(self):
        """モデル初期化"""
        # YOLO初期化
        self.yolo_model = YOLO("yolov8n.pt")

        # SAM初期化
        sam_checkpoint = self.project_root / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            logger.error(f"SAMモデルが見つかりません: {sam_checkpoint}")
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def load_human_labels(self) -> Dict[str, Dict]:
        """人間ラベルデータ読み込み"""
        try:
            with open(self.labels_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # リスト形式を辞書形式に変換
            labels_dict = {}
            for item in data:
                filename = item["filename"]
                image_id = filename.rsplit(".", 1)[0]

                # 最初の赤枠を使用
                if item.get("red_boxes") and len(item["red_boxes"]) > 0:
                    first_box = item["red_boxes"][0]
                    bbox = first_box["bbox"]
                    labels_dict[image_id] = {
                        "filename": filename,
                        "bbox": [bbox["x"], bbox["y"], bbox["width"], bbox["height"]],
                    }

            return labels_dict

        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return {}

    def calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """IoU (Intersection over Union) 計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 交差領域の計算
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # 交差領域の面積
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 和集合の面積
        union_area = w1 * h1 + w2 * h2 - intersection_area

        # IoU
        iou = intersection_area / max(union_area, 1e-6)
        return iou

    def find_image_path(self, image_id: str) -> Optional[Path]:
        """画像ファイルパス検索"""
        search_dirs = [
            Path("/mnt/c/AItools/lora/train/yado/org/kana05_cursor"),
            Path("/mnt/c/AItools/lora/train/yado/org/kana07_cursor"),
            Path("/mnt/c/AItools/lora/train/yado/org/kana08_cursor"),
            self.project_root / "test_small",
        ]

        extensions = [".jpg", ".jpeg", ".png"]

        for dir_path in search_dirs:
            for ext in extensions:
                image_path = dir_path / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path

        return None

    def extract_with_ai(self, image_path: Path) -> Optional[Tuple[int, int, int, int]]:
        """現在のAIシステムでキャラクター抽出"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # YOLO検出
            results = self.yolo_model(image, conf=0.07)  # アニメキャラクター用低閾値

            if not results or len(results[0].boxes) == 0:
                return None

            # 最大の検出結果を選択
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = []
            for box in boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)

            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[largest_idx]

            # SAMでセグメンテーション
            self.sam_predictor.set_image(image)

            # 中心点をプロンプトとして使用
            input_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            input_label = np.array([1])

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_point, point_labels=input_label, multimask_output=True
            )

            if masks is None or len(masks) == 0:
                # SAM失敗時はYOLOボックスを返す
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 最良マスクから境界ボックス計算
            best_mask = masks[np.argmax(scores)]
            y_indices, x_indices = np.where(best_mask > 0)

            if len(x_indices) == 0 or len(y_indices) == 0:
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            x_min = int(np.min(x_indices))
            x_max = int(np.max(x_indices))
            y_min = int(np.min(y_indices))
            y_max = int(np.max(y_indices))

            return (x_min, y_min, x_max - x_min, y_max - y_min)

        except Exception as e:
            logger.error(f"AI抽出エラー: {e}")
            return None

    def run_single_benchmark(self, image_id: str, label_data: Dict) -> BenchmarkResult:
        """単一画像のベンチマーク実行"""
        start_time = time.time()

        # 画像パス検索
        image_path = self.find_image_path(image_id)
        if not image_path:
            return BenchmarkResult(
                image_id=image_id,
                image_path="",
                human_bbox=tuple(label_data["bbox"]),
                ai_bbox=None,
                iou_score=0.0,
                extraction_success=False,
                processing_time=0.0,
                error_message="画像ファイルが見つかりません",
            )

        # AI抽出実行
        ai_bbox = self.extract_with_ai(image_path)
        processing_time = time.time() - start_time

        # 結果評価
        human_bbox = tuple(label_data["bbox"])

        if ai_bbox is None:
            iou_score = 0.0
            extraction_success = False
            error_message = "AI抽出失敗"
        else:
            iou_score = self.calculate_iou(human_bbox, ai_bbox)
            extraction_success = iou_score > 0.5  # IoU > 0.5 を成功とする
            error_message = ""

        return BenchmarkResult(
            image_id=image_id,
            image_path=str(image_path),
            human_bbox=human_bbox,
            ai_bbox=ai_bbox,
            iou_score=iou_score,
            extraction_success=extraction_success,
            processing_time=processing_time,
            error_message=error_message,
        )

    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """全画像のベンチマーク実行"""
        logger.info("ベンチマーク開始")
        results = []

        total = len(self.human_labels)
        for i, (image_id, label_data) in enumerate(self.human_labels.items(), 1):
            logger.info(f"処理中 [{i}/{total}]: {image_id}")

            result = self.run_single_benchmark(image_id, label_data)
            results.append(result)

            # 進捗表示
            if i % 10 == 0:
                success_count = sum(1 for r in results if r.extraction_success)
                logger.info(f"進捗: {i}/{total} - 成功率: {success_count/i*100:.1f}%")

        return results

    def generate_comparison_image(self, result: BenchmarkResult, output_path: Path):
        """比較画像生成（人間ラベル vs AI抽出）"""
        try:
            if not result.image_path:
                return

            # 画像読み込み
            image = cv2.imread(result.image_path)
            if image is None:
                return

            # BGR -> RGB変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 図作成
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)

            # 人間ラベル（緑色）
            human_rect = patches.Rectangle(
                (result.human_bbox[0], result.human_bbox[1]),
                result.human_bbox[2],
                result.human_bbox[3],
                linewidth=3,
                edgecolor="green",
                facecolor="none",
                label="Human Label",
            )
            ax.add_patch(human_rect)

            # AI抽出結果（赤色）
            if result.ai_bbox:
                ai_rect = patches.Rectangle(
                    (result.ai_bbox[0], result.ai_bbox[1]),
                    result.ai_bbox[2],
                    result.ai_bbox[3],
                    linewidth=3,
                    edgecolor="red",
                    facecolor="none",
                    label="AI Extraction",
                )
                ax.add_patch(ai_rect)

            # タイトルと情報
            title = f"{result.image_id} - IoU: {result.iou_score:.3f}"
            if result.extraction_success:
                title += " ✅"
            else:
                title += " ❌"

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend(loc="upper right")
            ax.axis("off")

            # 保存
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.error(f"比較画像生成エラー: {e}")

    def generate_report(self, results: List[BenchmarkResult]):
        """ベンチマークレポート生成"""
        # 統計計算
        total = len(results)
        success_count = sum(1 for r in results if r.extraction_success)
        success_rate = success_count / total * 100

        iou_scores = [r.iou_score for r in results]
        avg_iou = np.mean(iou_scores)

        processing_times = [r.processing_time for r in results]
        avg_time = np.mean(processing_times)

        # ベスト5・ワースト5
        sorted_results = sorted(results, key=lambda r: r.iou_score, reverse=True)
        best_5 = sorted_results[:5]
        worst_5 = sorted_results[-5:]

        # レポート作成
        report = f"""# 人間ラベル vs AI抽出 ベンチマークレポート

**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**総画像数**: {total}枚

---

## 📊 全体統計

### 成功率
- **抽出成功数**: {success_count}枚
- **成功率**: {success_rate:.1f}%
- **基準**: IoU > 0.5

### IoUスコア
- **平均IoU**: {avg_iou:.3f}
- **最高IoU**: {max(iou_scores):.3f}
- **最低IoU**: {min(iou_scores):.3f}

### 処理性能
- **平均処理時間**: {avg_time:.2f}秒/画像

---

## 🏆 ベスト5（AIが最も正確）

"""

        for i, result in enumerate(best_5, 1):
            report += f"{i}. **{result.image_id}** - IoU: {result.iou_score:.3f}\n"

        report += """
---

## 💥 ワースト5（AIが最も不正確）

"""

        for i, result in enumerate(worst_5, 1):
            report += f"{i}. **{result.image_id}** - IoU: {result.iou_score:.3f}"
            if result.error_message:
                report += f" ({result.error_message})"
            report += "\n"

        report += """
---

## 🎯 改善ポイント

"""

        # 失敗パターン分析
        extraction_failures = [r for r in results if not r.ai_bbox]
        low_iou_cases = [r for r in results if r.ai_bbox and r.iou_score < 0.3]

        report += f"""### 失敗パターン
- **完全抽出失敗**: {len(extraction_failures)}件 ({len(extraction_failures)/total*100:.1f}%)
- **低精度抽出 (IoU < 0.3)**: {len(low_iou_cases)}件 ({len(low_iou_cases)/total*100:.1f}%)

### 推奨改善策
1. YOLO検出閾値の調整（現在0.07）
2. SAMプロンプト戦略の改善
3. 困難なポーズに対する特別処理
4. Phase 1以降の機械学習による精度向上

---

*Generated by Human vs AI Benchmark System*
"""

        # レポート保存
        report_path = self.output_dir / f"benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"レポート保存: {report_path}")

        # 結果JSON保存
        results_data = [asdict(r) for r in results]
        json_path = self.output_dir / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        return report_path, best_5, worst_5


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools/segment-anything")

    benchmark = HumanVsAIBenchmark(project_root)

    # ベンチマーク実行
    results = benchmark.run_full_benchmark()

    # レポート生成
    report_path, best_5, worst_5 = benchmark.generate_report(results)

    # 比較画像生成（ベスト5・ワースト5）
    comparison_dir = benchmark.output_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)

    logger.info("比較画像生成中...")

    # ベスト5
    best_dir = comparison_dir / "best_5"
    best_dir.mkdir(exist_ok=True)
    for i, result in enumerate(best_5, 1):
        output_path = best_dir / f"{i:02d}_{result.image_id}_iou{result.iou_score:.3f}.png"
        benchmark.generate_comparison_image(result, output_path)

    # ワースト5
    worst_dir = comparison_dir / "worst_5"
    worst_dir.mkdir(exist_ok=True)
    for i, result in enumerate(worst_5, 1):
        output_path = worst_dir / f"{i:02d}_{result.image_id}_iou{result.iou_score:.3f}.png"
        benchmark.generate_comparison_image(result, output_path)

    # 最終統計
    success_count = sum(1 for r in results if r.extraction_success)
    print(f"\n✅ ベンチマーク完了")
    print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"レポート: {report_path}")
    print(f"比較画像: {comparison_dir}")


if __name__ == "__main__":
    main()
