#!/usr/bin/env python3
"""
評価指標システム - Largest-Character Accuracy指標の確立
Comprehensive evaluation metrics for character extraction performance
"""

import numpy as np
import matplotlib.pyplot as plt

import json
import logging
import pandas as pd
import seaborn as sns
from dataclasses import asdict, dataclass
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """評価指標結果"""
    metric_name: str
    value: float
    threshold: Optional[float] = None
    status: str = "measured"  # measured, passed, failed
    notes: str = ""


@dataclass
class CharacterDetectionMetrics:
    """キャラクター検出評価指標"""
    # 主要指標
    largest_char_accuracy: MetricResult
    mean_iou: MetricResult
    map_50: MetricResult  # mAP@0.5
    map_95: MetricResult  # mAP@[.5:.95]
    
    # 品質指標
    ab_evaluation_rate: MetricResult
    precision_at_k: MetricResult  # P@1 (最大候補の精度)
    recall_at_k: MetricResult     # R@1 (最大候補の再現率)
    
    # 処理性能指標
    fps: MetricResult
    memory_usage: MetricResult
    
    # 分布統計
    confidence_stats: Dict[str, float]
    size_distribution: Dict[str, float]
    failure_analysis: Dict[str, Any]


class MetricsCalculator:
    """評価指標計算システム"""
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        """
        初期化
        
        Args:
            results_data: ベンチマーク結果データ
        """
        self.results_data = results_data
        self.total_samples = len(results_data)
        
        # 成功基準閾値
        self.thresholds = {
            "largest_char_accuracy": 0.80,  # 80%以上
            "ab_evaluation_rate": 0.70,     # 70%以上
            "mean_iou": 0.65,               # 0.65以上
            "map_50": 0.75,                 # 75%以上
            "fps": 0.2,                     # 5秒/画像以下（0.2 FPS以上）
            "precision_at_k": 0.80,         # 80%以上
            "recall_at_k": 0.75             # 75%以上
        }
    
    def calculate_largest_char_accuracy(self) -> MetricResult:
        """Largest-Character Accuracy計算"""
        try:
            correct_predictions = sum(1 for r in self.results_data if r.get('largest_char_predicted', False))
            accuracy = correct_predictions / self.total_samples if self.total_samples > 0 else 0.0
            
            threshold = self.thresholds["largest_char_accuracy"]
            status = "passed" if accuracy >= threshold else "failed"
            
            return MetricResult(
                metric_name="Largest-Character Accuracy",
                value=accuracy,
                threshold=threshold,
                status=status,
                notes=f"{correct_predictions}/{self.total_samples} 正解"
            )
            
        except Exception as e:
            logger.error(f"Largest-Character Accuracy計算エラー: {e}")
            return MetricResult("Largest-Character Accuracy", 0.0, notes=f"計算エラー: {e}")
    
    def calculate_mean_iou(self) -> MetricResult:
        """平均IoU計算"""
        try:
            iou_scores = [r.get('iou_score', 0.0) for r in self.results_data]
            mean_iou = np.mean(iou_scores) if iou_scores else 0.0
            
            threshold = self.thresholds["mean_iou"]
            status = "passed" if mean_iou >= threshold else "failed"
            
            return MetricResult(
                metric_name="Mean IoU",
                value=mean_iou,
                threshold=threshold,
                status=status,
                notes=f"範囲: {min(iou_scores):.3f} - {max(iou_scores):.3f}"
            )
            
        except Exception as e:
            logger.error(f"Mean IoU計算エラー: {e}")
            return MetricResult("Mean IoU", 0.0, notes=f"計算エラー: {e}")
    
    def calculate_map_metrics(self) -> Tuple[MetricResult, MetricResult]:
        """mAP@0.5 と mAP@[.5:.95] 計算"""
        try:
            # IoU閾値別の精度計算
            iou_thresholds_50 = [0.5]
            iou_thresholds_95 = np.arange(0.5, 1.0, 0.05)
            
            # mAP@0.5
            map_50_scores = []
            for threshold in iou_thresholds_50:
                correct = sum(1 for r in self.results_data if r.get('iou_score', 0.0) >= threshold)
                precision = correct / self.total_samples if self.total_samples > 0 else 0.0
                map_50_scores.append(precision)
            
            map_50 = np.mean(map_50_scores)
            map_50_status = "passed" if map_50 >= self.thresholds["map_50"] else "failed"
            
            # mAP@[.5:.95]
            map_95_scores = []
            for threshold in iou_thresholds_95:
                correct = sum(1 for r in self.results_data if r.get('iou_score', 0.0) >= threshold)
                precision = correct / self.total_samples if self.total_samples > 0 else 0.0
                map_95_scores.append(precision)
            
            map_95 = np.mean(map_95_scores)
            
            return (
                MetricResult(
                    metric_name="mAP@0.5",
                    value=map_50,
                    threshold=self.thresholds["map_50"],
                    status=map_50_status,
                    notes="IoU閾値0.5での平均精度"
                ),
                MetricResult(
                    metric_name="mAP@[.5:.95]",
                    value=map_95,
                    threshold=None,
                    status="measured",
                    notes="IoU閾値0.5-0.95での平均精度"
                )
            )
            
        except Exception as e:
            logger.error(f"mAP計算エラー: {e}")
            return (
                MetricResult("mAP@0.5", 0.0, notes=f"計算エラー: {e}"),
                MetricResult("mAP@[.5:.95]", 0.0, notes=f"計算エラー: {e}")
            )
    
    def calculate_quality_metrics(self) -> MetricResult:
        """A/B評価率計算"""
        try:
            ab_grades = sum(1 for r in self.results_data if r.get('quality_grade', 'F') in ['A', 'B'])
            ab_rate = ab_grades / self.total_samples if self.total_samples > 0 else 0.0
            
            threshold = self.thresholds["ab_evaluation_rate"]
            status = "passed" if ab_rate >= threshold else "failed"
            
            return MetricResult(
                metric_name="A/B Evaluation Rate",
                value=ab_rate,
                threshold=threshold,
                status=status,
                notes=f"{ab_grades}/{self.total_samples} がA/B評価"
            )
            
        except Exception as e:
            logger.error(f"品質指標計算エラー: {e}")
            return MetricResult("A/B Evaluation Rate", 0.0, notes=f"計算エラー: {e}")
    
    def calculate_precision_recall_at_k(self) -> Tuple[MetricResult, MetricResult]:
        """Precision@1, Recall@1計算（最大候補に対する）"""
        try:
            # True Positives: IoU >= 0.5 かつ最大面積候補
            true_positives = sum(1 for r in self.results_data 
                               if r.get('largest_char_predicted', False) and r.get('iou_score', 0.0) >= 0.5)
            
            # Predicted Positives: 検出された最大候補数（全て）
            predicted_positives = sum(1 for r in self.results_data if r.get('prediction_bbox') is not None)
            
            # Actual Positives: 正解ラベルが存在する数（全て）
            actual_positives = self.total_samples  # 全画像に正解ラベルが存在
            
            # Precision@1計算
            precision_at_1 = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            precision_status = "passed" if precision_at_1 >= self.thresholds["precision_at_k"] else "failed"
            
            # Recall@1計算
            recall_at_1 = true_positives / actual_positives if actual_positives > 0 else 0.0
            recall_status = "passed" if recall_at_1 >= self.thresholds["recall_at_k"] else "failed"
            
            return (
                MetricResult(
                    metric_name="Precision@1",
                    value=precision_at_1,
                    threshold=self.thresholds["precision_at_k"],
                    status=precision_status,
                    notes=f"TP:{true_positives}, PP:{predicted_positives}"
                ),
                MetricResult(
                    metric_name="Recall@1",
                    value=recall_at_1,
                    threshold=self.thresholds["recall_at_k"],
                    status=recall_status,
                    notes=f"TP:{true_positives}, AP:{actual_positives}"
                )
            )
            
        except Exception as e:
            logger.error(f"Precision/Recall計算エラー: {e}")
            return (
                MetricResult("Precision@1", 0.0, notes=f"計算エラー: {e}"),
                MetricResult("Recall@1", 0.0, notes=f"計算エラー: {e}")
            )
    
    def calculate_performance_metrics(self) -> Tuple[MetricResult, MetricResult]:
        """処理性能指標計算"""
        try:
            processing_times = [r.get('processing_time', 0.0) for r in self.results_data]
            
            # FPS計算
            mean_processing_time = np.mean(processing_times) if processing_times else float('inf')
            fps = 1.0 / mean_processing_time if mean_processing_time > 0 else 0.0
            
            fps_threshold = self.thresholds["fps"]
            fps_status = "passed" if fps >= fps_threshold else "failed"
            
            # メモリ使用量（仮想値、実装時に実測値で更新）
            estimated_memory = 4.5  # GB（YOLO + SAM + 画像バッファ）
            
            return (
                MetricResult(
                    metric_name="FPS",
                    value=fps,
                    threshold=fps_threshold,
                    status=fps_status,
                    notes=f"平均処理時間: {mean_processing_time:.2f}秒"
                ),
                MetricResult(
                    metric_name="Memory Usage",
                    value=estimated_memory,
                    threshold=None,
                    status="measured",
                    notes="YOLO + SAM + バッファ推定値"
                )
            )
            
        except Exception as e:
            logger.error(f"性能指標計算エラー: {e}")
            return (
                MetricResult("FPS", 0.0, notes=f"計算エラー: {e}"),
                MetricResult("Memory Usage", 0.0, notes=f"計算エラー: {e}")
            )
    
    def calculate_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """分布統計計算"""
        try:
            # 信頼度統計
            confidence_scores = [r.get('confidence_score', 0.0) for r in self.results_data]
            confidence_stats = {
                "mean": float(np.mean(confidence_scores)),
                "std": float(np.std(confidence_scores)),
                "min": float(np.min(confidence_scores)),
                "max": float(np.max(confidence_scores)),
                "median": float(np.median(confidence_scores)),
                "q25": float(np.percentile(confidence_scores, 25)),
                "q75": float(np.percentile(confidence_scores, 75))
            }
            
            # サイズ分布（面積比）
            area_ratios = [r.get('area_largest_ratio', 0.0) for r in self.results_data if r.get('area_largest_ratio', 0.0) > 0]
            size_distribution = {
                "mean_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
                "std_area_ratio": float(np.std(area_ratios)) if area_ratios else 0.0,
                "single_character_rate": sum(1 for r in self.results_data if r.get('character_count', 0) == 1) / self.total_samples,
                "multi_character_rate": sum(1 for r in self.results_data if r.get('character_count', 0) > 1) / self.total_samples
            }
            
            return {
                "confidence_stats": confidence_stats,
                "size_distribution": size_distribution
            }
            
        except Exception as e:
            logger.error(f"分布統計計算エラー: {e}")
            return {"confidence_stats": {}, "size_distribution": {}}
    
    def analyze_failures(self) -> Dict[str, Any]:
        """失敗ケース分析"""
        try:
            failed_cases = [r for r in self.results_data if not r.get('largest_char_predicted', False)]
            
            if not failed_cases:
                return {"total_failures": 0, "failure_patterns": {}}
            
            failure_analysis = {
                "total_failures": len(failed_cases),
                "failure_rate": len(failed_cases) / self.total_samples,
                "failure_patterns": {}
            }
            
            # 失敗パターン分類
            # 1. 極低IoU（< 0.1）
            extremely_low_iou = [r for r in failed_cases if r.get('iou_score', 0.0) < 0.1]
            failure_analysis["failure_patterns"]["extremely_low_iou"] = {
                "count": len(extremely_low_iou),
                "rate": len(extremely_low_iou) / len(failed_cases) if failed_cases else 0.0,
                "description": "IoU < 0.1の完全なミス"
            }
            
            # 2. 低信頼度（< 0.3）
            low_confidence = [r for r in failed_cases if r.get('confidence_score', 0.0) < 0.3]
            failure_analysis["failure_patterns"]["low_confidence"] = {
                "count": len(low_confidence),
                "rate": len(low_confidence) / len(failed_cases) if failed_cases else 0.0,
                "description": "信頼度 < 0.3の不確実な検出"
            }
            
            # 3. 複数キャラクター競合
            multi_char_conflicts = [r for r in failed_cases if r.get('character_count', 0) > 2]
            failure_analysis["failure_patterns"]["multi_character_conflict"] = {
                "count": len(multi_char_conflicts),
                "rate": len(multi_char_conflicts) / len(failed_cases) if failed_cases else 0.0,
                "description": "3体以上のキャラクター競合による誤選択"
            }
            
            # 4. 中程度IoU（0.1-0.4）：部分的成功
            partial_success = [r for r in failed_cases if 0.1 <= r.get('iou_score', 0.0) < 0.5]
            failure_analysis["failure_patterns"]["partial_success"] = {
                "count": len(partial_success),
                "rate": len(partial_success) / len(failed_cases) if failed_cases else 0.0,
                "description": "0.1 ≤ IoU < 0.5の部分的成功（調整で改善可能）"
            }
            
            return failure_analysis
            
        except Exception as e:
            logger.error(f"失敗分析エラー: {e}")
            return {"total_failures": 0, "failure_patterns": {}, "error": str(e)}
    
    def compute_all_metrics(self) -> CharacterDetectionMetrics:
        """全指標計算"""
        try:
            logger.info("評価指標計算開始")
            
            # 主要指標計算
            largest_char_accuracy = self.calculate_largest_char_accuracy()
            mean_iou = self.calculate_mean_iou()
            map_50, map_95 = self.calculate_map_metrics()
            
            # 品質指標
            ab_evaluation_rate = self.calculate_quality_metrics()
            precision_at_k, recall_at_k = self.calculate_precision_recall_at_k()
            
            # 性能指標
            fps, memory_usage = self.calculate_performance_metrics()
            
            # 分布統計
            distribution_stats = self.calculate_distribution_stats()
            
            # 失敗分析
            failure_analysis = self.analyze_failures()
            
            metrics = CharacterDetectionMetrics(
                largest_char_accuracy=largest_char_accuracy,
                mean_iou=mean_iou,
                map_50=map_50,
                map_95=map_95,
                ab_evaluation_rate=ab_evaluation_rate,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                fps=fps,
                memory_usage=memory_usage,
                confidence_stats=distribution_stats["confidence_stats"],
                size_distribution=distribution_stats["size_distribution"],
                failure_analysis=failure_analysis
            )
            
            logger.info("評価指標計算完了")
            return metrics
            
        except Exception as e:
            logger.error(f"全指標計算エラー: {e}")
            raise


class MetricsVisualizer:
    """評価指標可視化システム"""
    
    def __init__(self, metrics: CharacterDetectionMetrics, output_dir: Path):
        """
        初期化
        
        Args:
            metrics: 計算済み評価指標
            output_dir: 出力ディレクトリ
        """
        self.metrics = metrics
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_performance_dashboard(self):
        """総合性能ダッシュボード作成"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Phase 0 ベンチマーク: 総合性能ダッシュボード', fontsize=16, fontweight='bold')
            
            # 1. 主要指標レーダーチャート
            ax1 = axes[0, 0]
            metrics_names = ['Largest-Char\nAccuracy', 'Mean IoU', 'mAP@0.5', 'A/B Rate', 'Precision@1', 'Recall@1']
            metrics_values = [
                self.metrics.largest_char_accuracy.value,
                self.metrics.mean_iou.value,
                self.metrics.map_50.value,
                self.metrics.ab_evaluation_rate.value,
                self.metrics.precision_at_k.value,
                self.metrics.recall_at_k.value
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
            metrics_values += metrics_values[:1]  # 閉じるため
            angles = np.concatenate((angles, [angles[0]]))
            
            ax1 = plt.subplot(2, 3, 1, projection='polar')
            ax1.plot(angles, metrics_values, 'o-', linewidth=2, label='現在性能')
            ax1.fill(angles, metrics_values, alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(metrics_names)
            ax1.set_ylim(0, 1)
            ax1.set_title('主要指標レーダーチャート')
            ax1.grid(True)
            
            # 2. 品質グレード分布
            ax2 = axes[0, 1]
            # 実際のデータが必要（仮のデータで表示）
            grades = ['A', 'B', 'C', 'D', 'E', 'F']
            # failure_analysisから推定（実装時にデータ取得）
            grade_counts = [5, 15, 25, 30, 15, 10]  # 仮データ
            
            bars = ax2.bar(grades, grade_counts, color=['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'])
            ax2.set_title('品質グレード分布')
            ax2.set_ylabel('画像数')
            
            # 各バーに数値表示
            for bar, count in zip(bars, grade_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
            
            # 3. 信頼度分布ヒストグラム
            ax3 = axes[0, 2]
            if self.metrics.confidence_stats:
                # 正規分布近似でヒストグラム作成
                mean_conf = self.metrics.confidence_stats.get('mean', 0.5)
                std_conf = self.metrics.confidence_stats.get('std', 0.2)
                conf_samples = np.random.normal(mean_conf, std_conf, 1000)
                conf_samples = np.clip(conf_samples, 0, 1)
                
                ax3.hist(conf_samples, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(mean_conf, color='red', linestyle='--', label=f'平均: {mean_conf:.3f}')
                ax3.set_title('信頼度スコア分布')
                ax3.set_xlabel('信頼度')
                ax3.set_ylabel('頻度')
                ax3.legend()
            
            # 4. IoU分布ボックスプロット
            ax4 = axes[1, 0]
            # 仮データでボックスプロット（実装時に実データ使用）
            iou_data = [np.random.beta(2, 3, 100)]  # 仮のIoU分布
            ax4.boxplot(iou_data, labels=['IoU Distribution'])
            ax4.axhline(0.5, color='red', linestyle='--', label='成功閾値 (0.5)')
            ax4.set_title('IoUスコア分布')
            ax4.set_ylabel('IoU')
            ax4.legend()
            
            # 5. 処理時間統計
            ax5 = axes[1, 1]
            processing_metrics = ['FPS', 'Mean Time', 'Memory (GB)']
            processing_values = [
                self.metrics.fps.value,
                1/self.metrics.fps.value if self.metrics.fps.value > 0 else 0,
                self.metrics.memory_usage.value
            ]
            
            bars = ax5.bar(processing_metrics, processing_values, color=['#3498db', '#9b59b6', '#e74c3c'])
            ax5.set_title('処理性能統計')
            ax5.set_ylabel('値')
            
            # 各バーに数値表示
            for bar, value in zip(bars, processing_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 6. 失敗パターン分析
            ax6 = axes[1, 2]
            if self.metrics.failure_analysis and 'failure_patterns' in self.metrics.failure_analysis:
                patterns = list(self.metrics.failure_analysis['failure_patterns'].keys())
                pattern_counts = [self.metrics.failure_analysis['failure_patterns'][p]['count'] 
                                for p in patterns]
                
                if patterns and pattern_counts:
                    wedges, texts, autotexts = ax6.pie(pattern_counts, labels=patterns, autopct='%1.1f%%', startangle=90)
                    ax6.set_title('失敗パターン分析')
                else:
                    ax6.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax6.transAxes)
                    ax6.set_title('失敗パターン分析')
            
            plt.tight_layout()
            
            # 保存
            dashboard_file = self.output_dir / "performance_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"性能ダッシュボード作成完了: {dashboard_file}")
            
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
    
    def save_metrics_json(self):
        """指標結果をJSON保存"""
        try:
            metrics_dict = asdict(self.metrics)
            
            # JSONファイル保存
            json_file = self.output_dir / "evaluation_metrics.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"評価指標JSON保存完了: {json_file}")
            
        except Exception as e:
            logger.error(f"JSON保存エラー: {e}")


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    
    # テスト用のサンプルデータ
    sample_results = [
        {
            'image_id': f'test_{i:03d}',
            'largest_char_predicted': np.random.random() > 0.4,  # 60%成功率
            'iou_score': np.random.beta(2, 3),  # 0-1のベータ分布
            'confidence_score': np.random.uniform(0.1, 0.9),
            'processing_time': np.random.uniform(3.0, 8.0),
            'character_count': np.random.randint(1, 5),
            'area_largest_ratio': np.random.uniform(0.3, 0.8),
            'quality_grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10])
        }
        for i in range(101)
    ]
    
    # 指標計算
    calculator = MetricsCalculator(sample_results)
    metrics = calculator.compute_all_metrics()
    
    # 結果表示
    print("\n=== Phase 0 評価指標結果 ===")
    print(f"Largest-Character Accuracy: {metrics.largest_char_accuracy.value:.1%} ({metrics.largest_char_accuracy.status})")
    print(f"Mean IoU: {metrics.mean_iou.value:.3f} ({metrics.mean_iou.status})")
    print(f"A/B評価率: {metrics.ab_evaluation_rate.value:.1%} ({metrics.ab_evaluation_rate.status})")
    print(f"FPS: {metrics.fps.value:.3f} ({metrics.fps.status})")
    
    # 可視化
    output_dir = Path("/tmp/metrics_test")
    visualizer = MetricsVisualizer(metrics, output_dir)
    visualizer.create_performance_dashboard()
    visualizer.save_metrics_json()
    
    print(f"\n📊 結果保存: {output_dir}")


if __name__ == "__main__":
    main()