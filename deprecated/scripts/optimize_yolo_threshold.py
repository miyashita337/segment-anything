#!/usr/bin/env python3
"""
YOLO閾値最適化システム
人間ラベルとの比較による最適閾値探索
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """閾値テスト結果"""

    threshold: float
    total_detections: int
    successful_detections: int
    success_rate: float
    avg_confidence: float
    avg_detection_size: float
    processing_time: float


class YOLOThresholdOptimizer:
    """YOLO閾値最適化システム"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.labels_file = project_root / "extracted_labels.json"
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/yolo_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 人間ラベルデータ読み込み
        self.human_labels = self.load_human_labels()
        logger.info(f"人間ラベルデータ読み込み: {len(self.human_labels)}件")

        # YOLOモデル初期化
        self.yolo_model = YOLO("yolov8n.pt")

        # テスト用閾値範囲
        self.test_thresholds = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

    def load_human_labels(self) -> Dict[str, Dict]:
        """人間ラベルデータ読み込み"""
        try:
            with open(self.labels_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            labels_dict = {}
            for item in data:
                filename = item["filename"]
                image_id = filename.rsplit(".", 1)[0]

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

    def calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """IoU計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area

        return intersection_area / max(union_area, 1e-6)

    def test_threshold(self, threshold: float, sample_limit: int = 30) -> ThresholdResult:
        """指定閾値でのテスト実行"""
        logger.info(f"閾値 {threshold} でテスト実行")

        start_time = time.time()
        successful_detections = 0
        total_detections = 0
        confidences = []
        detection_sizes = []

        # サンプル制限（高速化のため）
        test_items = list(self.human_labels.items())[:sample_limit]

        for image_id, label_data in test_items:
            image_path = self.find_image_path(image_id)
            if not image_path:
                continue

            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            total_detections += 1

            # YOLO検出（指定閾値）
            results = self.yolo_model(image, conf=threshold, verbose=False)

            if not results or len(results[0].boxes) == 0:
                continue

            # 最大検出結果選択
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            if len(boxes) == 0:
                continue

            # 面積最大のボックス選択
            areas = []
            for box in boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)

            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[largest_idx]
            detection_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 人間ラベルとの比較
            human_bbox = tuple(label_data["bbox"])
            iou = self.calculate_iou(human_bbox, detection_bbox)

            # 成功判定（IoU > 0.3で成功とする - 緩い基準）
            if iou > 0.3:
                successful_detections += 1

            # 統計情報収集
            confidences.append(float(confs[largest_idx]))
            detection_sizes.append(areas[largest_idx])

        processing_time = time.time() - start_time
        success_rate = successful_detections / max(total_detections, 1) * 100
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_detection_size = np.mean(detection_sizes) if detection_sizes else 0.0

        return ThresholdResult(
            threshold=threshold,
            total_detections=total_detections,
            successful_detections=successful_detections,
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            avg_detection_size=avg_detection_size,
            processing_time=processing_time,
        )

    def run_optimization(self, sample_limit: int = 30) -> List[ThresholdResult]:
        """全閾値での最適化実行"""
        logger.info("YOLO閾値最適化開始")
        results = []

        for threshold in self.test_thresholds:
            result = self.test_threshold(threshold, sample_limit)
            results.append(result)

            logger.info(
                f"閾値 {threshold}: 成功率 {result.success_rate:.1f}% "
                f"({result.successful_detections}/{result.total_detections})"
            )

        return results

    def generate_optimization_report(self, results: List[ThresholdResult]):
        """最適化レポート生成"""
        # 最良結果を見つける
        best_result = max(results, key=lambda r: r.success_rate)

        # レポート作成
        report = f"""# YOLO閾値最適化レポート

**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**テスト画像数**: {results[0].total_detections if results else 0}枚
**成功基準**: IoU > 0.3

---

## 🏆 最適閾値

**推奨閾値**: {best_result.threshold}
- **成功率**: {best_result.success_rate:.1f}%
- **平均信頼度**: {best_result.avg_confidence:.3f}
- **処理時間**: {best_result.processing_time:.2f}秒

---

## 📊 全閾値テスト結果

| 閾値 | 成功率 | 成功数/総数 | 平均信頼度 | 処理時間 |
|------|--------|-------------|------------|----------|
"""

        for result in results:
            report += f"| {result.threshold} | {result.success_rate:.1f}% | {result.successful_detections}/{result.total_detections} | {result.avg_confidence:.3f} | {result.processing_time:.2f}s |\n"

        report += f"""
---

## 📈 改善効果

### 現在の設定 vs 最適設定
- **現在の閾値**: 0.07
- **現在の成功率**: (要測定)
- **最適閾値**: {best_result.threshold}  
- **最適時成功率**: {best_result.success_rate:.1f}%

### 推奨事項
1. **YOLO閾値を {best_result.threshold} に変更**
2. **継続的モニタリング**でさらなる微調整
3. **困難画像への特別処理**検討

---

## 🎯 次のステップ

1. **SAMプロンプト戦略改善**
   - 複数点プロンプト
   - ネガティブプロンプト活用
   - 境界ボックスプロンプト併用

2. **統合最適化**
   - YOLO + SAM の連携調整
   - エンドツーエンド性能測定

*Generated by YOLO Threshold Optimizer*
"""

        # レポート保存
        report_path = (
            self.output_dir / f"yolo_optimization_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 結果JSON保存
        results_data = [
            {
                "threshold": r.threshold,
                "success_rate": r.success_rate,
                "successful_detections": r.successful_detections,
                "total_detections": r.total_detections,
                "avg_confidence": r.avg_confidence,
                "processing_time": r.processing_time,
            }
            for r in results
        ]

        json_path = (
            self.output_dir / f"yolo_optimization_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # グラフ生成
        self.generate_optimization_graph(results)

        logger.info(f"最適化レポート保存: {report_path}")
        return report_path, best_result

    def generate_optimization_graph(self, results: List[ThresholdResult]):
        """最適化結果グラフ生成"""
        thresholds = [r.threshold for r in results]
        success_rates = [r.success_rate for r in results]
        confidences = [r.avg_confidence for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 成功率グラフ
        ax1.plot(thresholds, success_rates, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("YOLO Confidence Threshold")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("YOLO Threshold vs Success Rate")
        ax1.grid(True, alpha=0.3)

        # 最良点をハイライト
        best_idx = np.argmax(success_rates)
        ax1.plot(
            thresholds[best_idx],
            success_rates[best_idx],
            "ro",
            markersize=12,
            label=f"Best: {thresholds[best_idx]} ({success_rates[best_idx]:.1f}%)",
        )
        ax1.legend()

        # 信頼度グラフ
        ax2.plot(thresholds, confidences, "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("YOLO Confidence Threshold")
        ax2.set_ylabel("Average Confidence")
        ax2.set_title("YOLO Threshold vs Average Confidence")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # グラフ保存
        graph_path = (
            self.output_dir / f"yolo_optimization_graph_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"最適化グラフ保存: {graph_path}")


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools/segment-anything")

    optimizer = YOLOThresholdOptimizer(project_root)

    # 最適化実行（サンプル30枚で高速実行）
    results = optimizer.run_optimization(sample_limit=30)

    # レポート生成
    report_path, best_result = optimizer.generate_optimization_report(results)

    # 結果サマリー
    print(f"\n✅ YOLO閾値最適化完了")
    print(f"最適閾値: {best_result.threshold}")
    print(f"成功率: {best_result.success_rate:.1f}%")
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()
