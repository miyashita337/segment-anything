#!/usr/bin/env python3
"""
統合改善ベンチマーク
最適YOLO閾値(0.03) + 最優秀SAM戦略(BBox Prompt)の統合効果測定
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from segment_anything import SamPredictor, sam_model_registry

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegratedBenchmarkResult:
    """統合ベンチマーク結果"""
    image_id: str
    image_path: str
    human_bbox: Tuple[int, int, int, int]
    yolo_bbox: Optional[Tuple[int, int, int, int]]
    final_bbox: Optional[Tuple[int, int, int, int]]
    iou_score: float
    extraction_success: bool
    processing_time: float
    yolo_confidence: float
    sam_confidence: float
    improvement_method: str = "Integrated (YOLO 0.03 + BBox Prompt)"


class IntegratedImprovementBenchmark:
    """統合改善ベンチマーク"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.labels_file = project_root / "extracted_labels.json"
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/integrated_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 人間ラベルデータ読み込み
        self.human_labels = self.load_human_labels()
        logger.info(f"人間ラベルデータ読み込み: {len(self.human_labels)}件")
        
        # 最適設定
        self.optimal_yolo_threshold = 0.03
        self.optimal_sam_strategy = "BBox Prompt"
        
        # モデル初期化
        self.yolo_model = YOLO('yolov8n.pt')
        self.init_sam()
        
    def init_sam(self):
        """SAM初期化"""
        sam_checkpoint = self.project_root / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
    def load_human_labels(self) -> Dict[str, Dict]:
        """人間ラベルデータ読み込み"""
        try:
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels_dict = {}
            for item in data:
                filename = item['filename']
                image_id = filename.rsplit('.', 1)[0]
                
                if item.get('red_boxes') and len(item['red_boxes']) > 0:
                    first_box = item['red_boxes'][0]
                    bbox = first_box['bbox']
                    labels_dict[image_id] = {
                        'filename': filename,
                        'bbox': [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
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
            self.project_root / "test_small"
        ]
        
        extensions = ['.jpg', '.jpeg', '.png']
        
        for dir_path in search_dirs:
            for ext in extensions:
                image_path = dir_path / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path
        
        return None
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
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
    
    def integrated_extraction(self, image_path: Path) -> Tuple[Optional[Tuple[int, int, int, int]], float, float]:
        """統合改善抽出（最適YOLO閾値 + 最優秀SAM戦略）"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return None, 0.0, 0.0
            
            # YOLO検出（最適閾値0.03）
            results = self.yolo_model(image, conf=self.optimal_yolo_threshold, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                return None, 0.0, 0.0
            
            # 最大検出結果選択
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            if len(boxes) == 0:
                return None, 0.0, 0.0
            
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[largest_idx]
            yolo_confidence = float(confs[largest_idx])
            
            # SAM セグメンテーション（BBox Prompt戦略）
            self.sam_predictor.set_image(image)
            
            # YOLOボックスを少し拡張
            w, h = x2 - x1, y2 - y1
            expansion = 0.1  # 10%拡張
            expand_w = int(w * expansion)
            expand_h = int(h * expansion)
            
            expanded_bbox = np.array([
                max(0, x1 - expand_w),
                max(0, y1 - expand_h),
                x2 + expand_w,
                y2 + expand_h
            ])
            
            masks, scores, _ = self.sam_predictor.predict(
                box=expanded_bbox[None, :],
                multimask_output=True
            )
            
            if masks is None or len(masks) == 0:
                # SAM失敗時はYOLOボックスを返す
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), yolo_confidence, 0.0
            
            # 最良マスクから境界ボックス計算
            best_mask = masks[np.argmax(scores)]
            sam_confidence = float(np.max(scores))
            
            y_indices, x_indices = np.where(best_mask > 0)
            
            if len(x_indices) == 0 or len(y_indices) == 0:
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), yolo_confidence, 0.0
            
            x_min = int(np.min(x_indices))
            x_max = int(np.max(x_indices))
            y_min = int(np.min(y_indices))
            y_max = int(np.max(y_indices))
            
            return (x_min, y_min, x_max - x_min, y_max - y_min), yolo_confidence, sam_confidence
            
        except Exception as e:
            logger.error(f"統合抽出エラー: {e}")
            return None, 0.0, 0.0
    
    def run_integrated_benchmark(self) -> List[IntegratedBenchmarkResult]:
        """統合ベンチマーク実行（全101件）"""
        logger.info("統合改善ベンチマーク開始（全データ）")
        results = []
        
        total = len(self.human_labels)
        for i, (image_id, label_data) in enumerate(self.human_labels.items(), 1):
            logger.info(f"処理中 [{i}/{total}]: {image_id}")
            
            start_time = time.time()
            
            # 画像パス検索
            image_path = self.find_image_path(image_id)
            if not image_path:
                results.append(IntegratedBenchmarkResult(
                    image_id=image_id,
                    image_path="",
                    human_bbox=tuple(label_data['bbox']),
                    yolo_bbox=None,
                    final_bbox=None,
                    iou_score=0.0,
                    extraction_success=False,
                    processing_time=0.0,
                    yolo_confidence=0.0,
                    sam_confidence=0.0
                ))
                continue
            
            # 統合抽出実行
            final_bbox, yolo_conf, sam_conf = self.integrated_extraction(image_path)
            processing_time = time.time() - start_time
            
            # 結果評価
            human_bbox = tuple(label_data['bbox'])
            
            if final_bbox is None:
                iou_score = 0.0
                extraction_success = False
            else:
                iou_score = self.calculate_iou(human_bbox, final_bbox)
                extraction_success = iou_score > 0.5  # IoU > 0.5 を成功とする
            
            results.append(IntegratedBenchmarkResult(
                image_id=image_id,
                image_path=str(image_path),
                human_bbox=human_bbox,
                yolo_bbox=None,  # 簡略化のためスキップ
                final_bbox=final_bbox,
                iou_score=iou_score,
                extraction_success=extraction_success,
                processing_time=processing_time,
                yolo_confidence=yolo_conf,
                sam_confidence=sam_conf
            ))
            
            # 進捗表示
            if i % 20 == 0:
                success_count = sum(1 for r in results if r.extraction_success)
                logger.info(f"進捗: {i}/{total} - 成功率: {success_count/i*100:.1f}%")
        
        return results
    
    def generate_improvement_report(self, results: List[IntegratedBenchmarkResult], 
                                  baseline_success_rate: float = 16.8):
        """改善レポート生成"""
        # 統計計算
        total = len(results)
        success_count = sum(1 for r in results if r.extraction_success)
        success_rate = success_count / total * 100
        
        iou_scores = [r.iou_score for r in results]
        avg_iou = np.mean(iou_scores)
        
        processing_times = [r.processing_time for r in results]
        avg_time = np.mean(processing_times)
        
        # 改善効果計算
        improvement = success_rate - baseline_success_rate
        improvement_ratio = success_rate / baseline_success_rate if baseline_success_rate > 0 else 0
        
        # ベスト5・ワースト5
        sorted_results = sorted(results, key=lambda r: r.iou_score, reverse=True)
        best_5 = sorted_results[:5]
        worst_5 = sorted_results[-5:]
        
        # レポート作成
        report = f"""# 統合改善ベンチマークレポート

**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**総画像数**: {total}枚
**改善手法**: {results[0].improvement_method if results else 'N/A'}

---

## 🎯 改善効果サマリー

### 成功率比較
- **ベースライン**: {baseline_success_rate}% (従来手法)
- **改善後**: {success_rate:.1f}% ⭐
- **改善幅**: +{improvement:.1f}ポイント
- **改善倍率**: {improvement_ratio:.1f}倍

### IoUスコア
- **平均IoU**: {avg_iou:.3f}
- **最高IoU**: {max(iou_scores):.3f}
- **最低IoU**: {min(iou_scores):.3f}

### 処理性能
- **平均処理時間**: {avg_time:.2f}秒/画像

---

## 🏆 成功例ベスト5

"""
        
        for i, result in enumerate(best_5, 1):
            report += f"{i}. **{result.image_id}** - IoU: {result.iou_score:.3f} (YOLO信頼度: {result.yolo_confidence:.3f})\n"
        
        report += """
---

## 💥 失敗例ワースト5

"""
        
        for i, result in enumerate(worst_5, 1):
            report += f"{i}. **{result.image_id}** - IoU: {result.iou_score:.3f}\n"
        
        # IoU分布分析
        high_iou = sum(1 for score in iou_scores if score >= 0.7)
        medium_iou = sum(1 for score in iou_scores if 0.3 <= score < 0.7)
        low_iou = sum(1 for score in iou_scores if score < 0.3)
        
        report += f"""
---

## 📊 詳細分析

### IoU分布
- **高精度 (IoU ≥ 0.7)**: {high_iou}件 ({high_iou/total*100:.1f}%)
- **中精度 (0.3 ≤ IoU < 0.7)**: {medium_iou}件 ({medium_iou/total*100:.1f}%)
- **低精度 (IoU < 0.3)**: {low_iou}件 ({low_iou/total*100:.1f}%)

### 改善手法の効果
1. **YOLO閾値最適化**: 0.07 → 0.03
   - 検出率向上により基盤性能改善
2. **SAM BBoxプロンプト**: 境界ボックス利用
   - セグメンテーション精度向上

### 次の改善目標
- **目標成功率**: 60%+ (現在{success_rate:.1f}%)
- **専用学習データ活用**: 101件の人間ラベルでの教師あり学習
- **困難ポーズ対応**: 特別処理アルゴリズム実装

---

## 🚀 実装推奨事項

1. **即座に適用可能**
   - YOLO信頼度閾値を0.03に変更
   - SAMプロンプト戦略をBBoxPromptに変更

2. **次フェーズ開発**
   - アニメキャラクター専用YOLO学習
   - エンドツーエンド統合最適化

*Generated by Integrated Improvement Benchmark System*
"""
        
        # レポート保存
        report_path = self.output_dir / f"integrated_improvement_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 結果JSON保存
        results_data = [asdict(r) for r in results]
        json_path = self.output_dir / f"integrated_improvement_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 比較グラフ生成
        self.generate_improvement_graphs(results, baseline_success_rate)
        
        logger.info(f"統合改善レポート保存: {report_path}")
        return report_path, success_rate, improvement
    
    def generate_improvement_graphs(self, results: List[IntegratedBenchmarkResult], 
                                  baseline_rate: float):
        """改善効果グラフ生成"""
        success_rate = sum(1 for r in results if r.extraction_success) / len(results) * 100
        
        # 比較棒グラフ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成功率比較
        categories = ['Baseline\n(Original)', 'Improved\n(Integrated)']
        rates = [baseline_rate, success_rate]
        colors = ['red', 'green']
        
        bars = ax1.bar(categories, rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Extraction Success Rate Comparison')
        ax1.set_ylim(0, 100)
        
        # バーに数値表示
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 改善効果
        improvement = success_rate - baseline_rate
        ax1.text(0.5, max(rates) * 0.8, f'Improvement:\n+{improvement:.1f} points\n({success_rate/baseline_rate:.1f}x)',
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                fontsize=12, fontweight='bold')
        
        # IoU分布ヒストグラム
        iou_scores = [r.iou_score for r in results]
        ax2.hist(iou_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('IoU Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('IoU Score Distribution (Improved Method)')
        ax2.axvline(x=0.5, color='red', linestyle='--', label='Success Threshold (0.5)')
        ax2.legend()
        
        plt.tight_layout()
        
        # グラフ保存
        graph_path = self.output_dir / f"integrated_improvement_graph_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"改善効果グラフ保存: {graph_path}")


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools/segment-anything")
    
    benchmark = IntegratedImprovementBenchmark(project_root)
    
    # 統合ベンチマーク実行（全101件）
    results = benchmark.run_integrated_benchmark()
    
    # 改善レポート生成（ベースライン16.8%と比較）
    report_path, new_success_rate, improvement = benchmark.generate_improvement_report(
        results, baseline_success_rate=16.8
    )
    
    # 結果サマリー
    print(f"\n✅ 統合改善ベンチマーク完了")
    print(f"ベースライン成功率: 16.8%")
    print(f"改善後成功率: {new_success_rate:.1f}%")
    print(f"改善効果: +{improvement:.1f}ポイント")
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()