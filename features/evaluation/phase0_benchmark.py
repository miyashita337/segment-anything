#!/usr/bin/env python3
"""
Phase 0: ルールベースベンチマーク システム
既存YOLO/SAM + 面積最大選択による現状性能の定量化
"""

import numpy as np
import cv2

import json
import logging
import time
from dataclasses import asdict, dataclass
from features.common.project_tracker import ProjectTracker
from features.extraction.models.sam_wrapper import SAMModelWrapper
# 既存システムのインポート
from features.extraction.models.yolo_wrapper import YOLOModelWrapper
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    image_id: str
    image_path: str
    largest_char_predicted: bool  # 最大キャラクターを正しく抽出できたか
    prediction_bbox: Optional[Tuple[int, int, int, int]]  # 予測bbox (x, y, w, h)
    ground_truth_bbox: Tuple[int, int, int, int]  # 人間ラベルbbox
    iou_score: float  # IoU スコア
    confidence_score: float  # 予測信頼度
    processing_time: float  # 処理時間（秒）
    character_count: int  # 検出されたキャラクター候補数
    area_largest_ratio: float  # 最大面積の候補が占める比率
    quality_grade: str  # A/B/C/D/E/F評価
    notes: str = ""  # 追加情報


@dataclass
class BenchmarkSummary:
    """ベンチマーク集計結果"""
    total_images: int
    largest_char_accuracy: float  # 最大キャラクター正解率
    mean_iou: float  # 平均IoU
    ab_evaluation_rate: float  # A/B評価率
    mean_processing_time: float  # 平均処理時間
    grade_distribution: Dict[str, int]  # 評価グレード分布
    processing_stats: Dict[str, float]  # 処理統計情報


class Phase0Benchmark:
    """Phase 0 ベンチマークシステム"""
    
    def __init__(self, project_root: Path):
        """
        初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        self.results_dir = project_root / "benchmark_results" / "phase0"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 人間ラベルデータ読み込み
        self.labels_file = project_root / "extracted_labels.json"
        self.ground_truth_labels = self.load_ground_truth_labels()
        
        # モデル初期化
        self.yolo_wrapper = None
        self.sam_wrapper = None
        
        # 結果格納
        self.benchmark_results: List[BenchmarkResult] = []
        
        # プロジェクトトラッカー
        self.tracker = ProjectTracker(project_root)
        
    def load_ground_truth_labels(self) -> Dict[str, Any]:
        """人間ラベルデータ読み込み"""
        try:
            if not self.labels_file.exists():
                raise FileNotFoundError(f"ラベルファイルが見つかりません: {self.labels_file}")
            
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"人間ラベルデータ読み込み完了: {len(data)}ファイル")
            return data
            
        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return {}
    
    def initialize_models(self):
        """モデル初期化"""
        try:
            logger.info("モデル初期化開始")
            
            # YOLO初期化
            yolo_model_path = self.project_root / "yolov8x.pt"
            if not yolo_model_path.exists():
                yolo_model_path = self.project_root / "yolov8n.pt"  # フォールバック
            
            self.yolo_wrapper = YOLOModelWrapper(
                model_path=str(yolo_model_path),
                device="cuda" if self._check_cuda() else "cpu"
            )
            
            # SAM初期化
            sam_checkpoint = self.project_root / "sam_vit_h_4b8939.pth"
            if not sam_checkpoint.exists():
                raise FileNotFoundError(f"SAMモデルファイルが見つかりません: {sam_checkpoint}")
                
            self.sam_wrapper = SAMModelWrapper(
                checkpoint_path=str(sam_checkpoint),
                model_type="vit_h",
                device="cuda" if self._check_cuda() else "cpu"
            )
            
            logger.info("モデル初期化完了")
            
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def _check_cuda(self) -> bool:
        """CUDA利用可能性チェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        IoU計算
        
        Args:
            bbox1: (x, y, w, h) 形式
            bbox2: (x, y, w, h) 形式
            
        Returns:
            IoU スコア
        """
        try:
            # (x, y, w, h) -> (x1, y1, x2, y2) 変換
            x1_1, y1_1, w1, h1 = bbox1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            
            x1_2, y1_2, w2, h2 = bbox2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            
            # 交差領域計算
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # 合計領域計算
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"IoU計算エラー: {e}")
            return 0.0
    
    def extract_character_with_current_system(self, image_path: Path) -> Dict[str, Any]:
        """
        現在のシステムでキャラクター抽出実行
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            抽出結果辞書
        """
        try:
            start_time = time.time()
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"画像読み込み失敗: {image_path}")
            
            # YOLO初期化（必要な場合）
            if not self.yolo_wrapper.is_loaded:
                self.yolo_wrapper.load_model()
            
            # YOLO検出
            yolo_results = self.yolo_wrapper.detect_persons(image)
            
            if not yolo_results or len(yolo_results) == 0:
                return {
                    "success": False,
                    "reason": "YOLO検出結果なし",
                    "processing_time": time.time() - start_time,
                    "character_count": 0
                }
            
            # 面積最大の候補を選択
            largest_detection = max(yolo_results, key=lambda x: x.get('area', 0))
            
            # SAMでセグメンテーション
            bbox = largest_detection['bbox']  # (x, y, w, h)
            sam_result = self.sam_wrapper.segment_from_bbox(image, bbox)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "largest_bbox": bbox,
                "confidence": largest_detection.get('confidence', 0.0),
                "character_count": len(yolo_results),
                "area_largest_ratio": largest_detection.get('area', 0) / sum(r.get('area', 0) for r in yolo_results),
                "processing_time": processing_time,
                "sam_mask": sam_result.get('mask') if sam_result else None
            }
            
        except Exception as e:
            logger.error(f"キャラクター抽出エラー ({image_path}): {e}")
            return {
                "success": False,
                "reason": str(e),
                "processing_time": time.time() - start_time,
                "character_count": 0
            }
    
    def evaluate_single_image(self, image_id: str, image_data: Dict[str, Any]) -> BenchmarkResult:
        """
        単一画像の評価実行
        
        Args:
            image_id: 画像ID
            image_data: 人間ラベルデータ
            
        Returns:
            ベンチマーク結果
        """
        try:
            # 画像パス構築
            image_path = self.project_root / "test_small" / f"{image_id}.png"
            if not image_path.exists():
                image_path = self.project_root / "test_small" / f"{image_id}.jpg"
            
            if not image_path.exists():
                logger.warning(f"画像ファイルが見つかりません: {image_id}")
                return self._create_error_result(image_id, str(image_path), "画像ファイル不在")
            
            # 人間ラベル（ground truth）
            gt_bbox = tuple(image_data['red_box_coords'])  # (x, y, w, h)
            
            # 現在のシステムで抽出
            extraction_result = self.extract_character_with_current_system(image_path)
            
            if not extraction_result["success"]:
                return self._create_error_result(
                    image_id, str(image_path), 
                    extraction_result.get("reason", "抽出失敗")
                )
            
            # 予測結果
            pred_bbox = tuple(extraction_result["largest_bbox"])
            
            # IoU計算
            iou_score = self.calculate_iou(pred_bbox, gt_bbox)
            
            # 最大キャラクター正解判定（IoU閾値: 0.5）
            largest_char_predicted = iou_score >= 0.5
            
            # 品質評価
            quality_grade = self._calculate_quality_grade(iou_score, extraction_result["confidence"])
            
            return BenchmarkResult(
                image_id=image_id,
                image_path=str(image_path),
                largest_char_predicted=largest_char_predicted,
                prediction_bbox=pred_bbox,
                ground_truth_bbox=gt_bbox,
                iou_score=iou_score,
                confidence_score=extraction_result["confidence"],
                processing_time=extraction_result["processing_time"],
                character_count=extraction_result["character_count"],
                area_largest_ratio=extraction_result.get("area_largest_ratio", 0.0),
                quality_grade=quality_grade,
                notes=f"IoU: {iou_score:.3f}, Conf: {extraction_result['confidence']:.3f}"
            )
            
        except Exception as e:
            logger.error(f"画像評価エラー ({image_id}): {e}")
            return self._create_error_result(image_id, str(image_path), str(e))
    
    def _create_error_result(self, image_id: str, image_path: str, error_msg: str) -> BenchmarkResult:
        """エラー時の結果作成"""
        return BenchmarkResult(
            image_id=image_id,
            image_path=image_path,
            largest_char_predicted=False,
            prediction_bbox=None,
            ground_truth_bbox=(0, 0, 0, 0),
            iou_score=0.0,
            confidence_score=0.0,
            processing_time=0.0,
            character_count=0,
            area_largest_ratio=0.0,
            quality_grade="F",
            notes=f"エラー: {error_msg}"
        )
    
    def _calculate_quality_grade(self, iou_score: float, confidence: float) -> str:
        """品質グレード計算"""
        combined_score = (iou_score * 0.7) + (confidence * 0.3)
        
        if combined_score >= 0.9:
            return "A"
        elif combined_score >= 0.8:
            return "B"
        elif combined_score >= 0.6:
            return "C"
        elif combined_score >= 0.4:
            return "D"
        elif combined_score >= 0.2:
            return "E"
        else:
            return "F"
    
    def run_full_benchmark(self) -> BenchmarkSummary:
        """
        全体ベンチマーク実行
        
        Returns:
            ベンチマーク集計結果
        """
        try:
            logger.info("Phase 0 ベンチマーク開始")
            
            # モデル初期化
            if self.yolo_wrapper is None or self.sam_wrapper is None:
                self.initialize_models()
            
            # 全画像で評価実行
            total_images = len(self.ground_truth_labels)
            processed_count = 0
            
            for image_id, image_data in self.ground_truth_labels.items():
                logger.info(f"評価進行中: {processed_count + 1}/{total_images} ({image_id})")
                
                result = self.evaluate_single_image(image_id, image_data)
                self.benchmark_results.append(result)
                
                processed_count += 1
                
                # 進捗レポート（10画像ごと）
                if processed_count % 10 == 0:
                    current_accuracy = sum(1 for r in self.benchmark_results if r.largest_char_predicted) / processed_count
                    logger.info(f"中間結果: {processed_count}/{total_images} 完了, 現在精度: {current_accuracy:.1%}")
            
            # 結果集計
            summary = self.calculate_summary()
            
            # 結果保存
            self.save_results(summary)
            
            logger.info("Phase 0 ベンチマーク完了")
            
            # プロジェクトトラッカー更新
            self.tracker.update_task_status("phase0-benchmark", "completed")
            
            return summary
            
        except Exception as e:
            logger.error(f"ベンチマーク実行エラー: {e}")
            raise
    
    def calculate_summary(self) -> BenchmarkSummary:
        """ベンチマーク結果集計"""
        if not self.benchmark_results:
            return BenchmarkSummary(0, 0.0, 0.0, 0.0, 0.0, {}, {})
        
        total_images = len(self.benchmark_results)
        
        # 最大キャラクター正解率
        largest_char_accuracy = sum(1 for r in self.benchmark_results if r.largest_char_predicted) / total_images
        
        # 平均IoU
        mean_iou = sum(r.iou_score for r in self.benchmark_results) / total_images
        
        # A/B評価率
        ab_count = sum(1 for r in self.benchmark_results if r.quality_grade in ['A', 'B'])
        ab_evaluation_rate = ab_count / total_images
        
        # 平均処理時間
        mean_processing_time = sum(r.processing_time for r in self.benchmark_results) / total_images
        
        # グレード分布
        grade_distribution = {}
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            grade_distribution[grade] = sum(1 for r in self.benchmark_results if r.quality_grade == grade)
        
        # 処理統計
        processing_times = [r.processing_time for r in self.benchmark_results]
        processing_stats = {
            "mean": mean_processing_time,
            "min": min(processing_times),
            "max": max(processing_times),
            "std": float(np.std(processing_times))
        }
        
        return BenchmarkSummary(
            total_images=total_images,
            largest_char_accuracy=largest_char_accuracy,
            mean_iou=mean_iou,
            ab_evaluation_rate=ab_evaluation_rate,
            mean_processing_time=mean_processing_time,
            grade_distribution=grade_distribution,
            processing_stats=processing_stats
        )
    
    def save_results(self, summary: BenchmarkSummary):
        """結果保存"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 詳細結果
            detailed_results = {
                "summary": asdict(summary),
                "detailed_results": [asdict(r) for r in self.benchmark_results],
                "metadata": {
                    "timestamp": timestamp,
                    "total_images": len(self.benchmark_results),
                    "system_info": {
                        "yolo_model": "YOLOv8",
                        "sam_model": "ViT-H",
                        "selection_method": "area_largest"
                    }
                }
            }
            
            # JSON保存
            results_file = self.results_dir / f"phase0_benchmark_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # 最新結果としてもコピー
            latest_file = self.results_dir / "latest_benchmark_results.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ベンチマーク結果保存完了: {results_file}")
            
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
    
    def generate_report(self, summary: BenchmarkSummary) -> str:
        """レポート生成"""
        report = f"""
# Phase 0 ベンチマーク結果レポート

## 📊 総合結果

- **処理画像数**: {summary.total_images}枚
- **Largest-Character Accuracy**: {summary.largest_char_accuracy:.1%}
- **平均IoU**: {summary.mean_iou:.3f}
- **A/B評価率**: {summary.ab_evaluation_rate:.1%}
- **平均処理時間**: {summary.mean_processing_time:.2f}秒

## 📈 評価グレード分布

"""
        for grade, count in summary.grade_distribution.items():
            percentage = (count / summary.total_images) * 100
            report += f"- **{grade}評価**: {count}枚 ({percentage:.1f}%)\n"
        
        report += f"""

## ⚡ 処理性能

- **平均処理時間**: {summary.processing_stats['mean']:.2f}秒
- **最速処理**: {summary.processing_stats['min']:.2f}秒
- **最遅処理**: {summary.processing_stats['max']:.2f}秒
- **標準偏差**: {summary.processing_stats['std']:.2f}秒

## 🎯 改善対象の特定

### 主要課題
"""
        # 失敗ケース分析
        failed_cases = [r for r in self.benchmark_results if not r.largest_char_predicted]
        if failed_cases:
            report += f"- **失敗画像数**: {len(failed_cases)}枚 ({len(failed_cases)/summary.total_images:.1%})\n"
            
            # 低IoUケース
            low_iou_cases = [r for r in failed_cases if r.iou_score < 0.3]
            report += f"- **極低IoU (<0.3)**: {len(low_iou_cases)}枚\n"
            
            # 低信頼度ケース
            low_conf_cases = [r for r in failed_cases if r.confidence_score < 0.5]
            report += f"- **低信頼度 (<0.5)**: {len(low_conf_cases)}枚\n"
        
        report += f"""

## 📋 次のPhaseへの提言

### Phase 1での注力点
- IoU < 0.5の失敗ケース {len(failed_cases)}枚の詳細分析
- コマ検出精度向上による前処理改善
- データ拡張による学習データ増強

### 目標設定
- **Phase 1目標**: Largest-Character Accuracy 75%以上
- **Phase終了目標**: A/B評価率 70%以上達成

---
*生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # プロジェクトルート
    project_root = Path("/mnt/c/AItools/segment-anything")
    
    # ベンチマーク実行
    benchmark = Phase0Benchmark(project_root)
    
    try:
        logger.info("=== Phase 0 ベンチマーク開始 ===")
        
        # ベンチマーク実行
        summary = benchmark.run_full_benchmark()
        
        # レポート生成・表示
        report = benchmark.generate_report(summary)
        print(report)
        
        # レポートファイル保存
        report_file = benchmark.results_dir / f"phase0_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"レポート保存完了: {report_file}")
        logger.info("=== Phase 0 ベンチマーク完了 ===")
        
    except Exception as e:
        logger.error(f"ベンチマーク実行エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())