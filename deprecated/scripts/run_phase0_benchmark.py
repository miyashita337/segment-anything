#!/usr/bin/env python3
"""
Phase 0 ベンチマーク統合実行スクリプト
Run comprehensive benchmark evaluation for current YOLO+SAM system
"""

import json
import logging
import sys
import time
from pathlib import Path

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.common.progress_reporter import ProgressReporter
from features.common.project_tracker import ProjectTracker
from features.evaluation.metrics_system import MetricsCalculator, MetricsVisualizer
from features.evaluation.phase0_benchmark import Phase0Benchmark

logger = logging.getLogger(__name__)


def setup_logging():
    """ロギング設定"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("phase0_benchmark.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_phase0_complete_benchmark():
    """Phase 0 完全ベンチマーク実行"""
    try:
        logger.info("=== Phase 0 統合ベンチマーク開始 ===")
        start_time = time.time()

        # プロジェクトトラッカー初期化
        tracker = ProjectTracker(project_root)

        # Phase 0開始マーク
        tracker.update_task_status("phase0-start", "in_progress")

        # Step 1: ベンチマーク実行
        logger.info("Step 1: ベースライン性能測定開始")
        benchmark = Phase0Benchmark(project_root)
        summary = benchmark.run_full_benchmark()

        logger.info(f"ベースライン測定完了: {summary.total_images}画像処理")
        logger.info(f"Largest-Character Accuracy: {summary.largest_char_accuracy:.1%}")
        logger.info(f"A/B評価率: {summary.ab_evaluation_rate:.1%}")

        # Step 2: 詳細評価指標計算
        logger.info("Step 2: 評価指標システム構築")

        # ベンチマーク結果をメトリクス計算用に変換
        results_data = [
            {
                "image_id": r.image_id,
                "largest_char_predicted": r.largest_char_predicted,
                "iou_score": r.iou_score,
                "confidence_score": r.confidence_score,
                "processing_time": r.processing_time,
                "character_count": r.character_count,
                "area_largest_ratio": r.area_largest_ratio,
                "quality_grade": r.quality_grade,
                "prediction_bbox": r.prediction_bbox,
            }
            for r in benchmark.benchmark_results
        ]

        # 詳細評価指標計算
        calculator = MetricsCalculator(results_data)
        detailed_metrics = calculator.compute_all_metrics()

        logger.info("評価指標システム構築完了")

        # Step 3: 可視化レポート生成
        logger.info("Step 3: 総合レポート生成")

        metrics_output_dir = project_root / "benchmark_results" / "phase0" / "metrics"
        visualizer = MetricsVisualizer(detailed_metrics, metrics_output_dir)

        # ダッシュボード作成
        visualizer.create_performance_dashboard()
        visualizer.save_metrics_json()

        # 統合レポート生成
        report_content = generate_comprehensive_report(summary, detailed_metrics)

        # レポート保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = (
            project_root / "benchmark_results" / "phase0" / f"comprehensive_report_{timestamp}.md"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        # プロジェクト進捗更新
        tracker.add_task(
            "phase0-metrics",
            "Largest-Character Accuracy指標の確立",
            "phase0",
            "high",
            4.0,
            ["phase0-benchmark"],
        )
        tracker.update_task_status("phase0-metrics", "completed")

        tracker.add_task(
            "phase0-evaluation", "101ファイルでの性能測定システム構築", "phase0", "high", 6.0, ["phase0-metrics"]
        )
        tracker.update_task_status("phase0-evaluation", "completed")

        # Phase 0完了
        tracker.update_task_status("phase0-start", "completed")

        # 進捗レポート更新
        reporter = ProgressReporter(project_root)
        progress_report_file = reporter.generate_full_report()

        total_time = time.time() - start_time

        logger.info("=== Phase 0 統合ベンチマーク完了 ===")
        logger.info(f"総実行時間: {total_time:.1f}秒")
        logger.info(f"総合レポート: {report_file}")
        logger.info(f"進捗レポート: {progress_report_file}")
        logger.info(f"ダッシュボード: {metrics_output_dir / 'performance_dashboard.png'}")

        # 結果サマリー表示
        print("\n" + "=" * 60)
        print("🎯 Phase 0 ベンチマーク結果サマリー")
        print("=" * 60)
        print(f"📊 処理画像数: {summary.total_images}枚")
        print(f"🎯 Largest-Character Accuracy: {detailed_metrics.largest_char_accuracy.value:.1%}")
        print(f"📈 Mean IoU: {detailed_metrics.mean_iou.value:.3f}")
        print(f"⭐ A/B評価率: {detailed_metrics.ab_evaluation_rate.value:.1%}")
        print(f"⚡ 平均処理時間: {summary.mean_processing_time:.2f}秒")
        print(f"🚀 FPS: {detailed_metrics.fps.value:.3f}")

        print(f"\n📋 評価グレード分布:")
        for grade, count in summary.grade_distribution.items():
            percentage = (count / summary.total_images) * 100
            print(f"  {grade}: {count}枚 ({percentage:.1f}%)")

        print(f"\n📁 出力ファイル:")
        print(f"  📝 総合レポート: {report_file}")
        print(f"  📊 ダッシュボード: {metrics_output_dir / 'performance_dashboard.png'}")
        print(f"  📈 進捗レポート: {progress_report_file}")

        # 次Phase準備の提言
        print(f"\n🚀 次のステップ:")
        if detailed_metrics.largest_char_accuracy.value < 0.6:
            print("  ⚠️  精度が低いため、Phase 1でのコマ検出改善が急務")
        print("  📋 データ拡張システム構築開始を推奨")
        print("  🔧 Phase 1: コマ検出ネット構築への移行準備")

        return True

    except Exception as e:
        logger.error(f"Phase 0 ベンチマーク実行エラー: {e}")

        # エラー時のプロジェクトトラッカー更新
        try:
            tracker = ProjectTracker(project_root)
            tracker.update_task_status("phase0-start", "pending")  # ステータスを戻す
        except:
            pass

        return False


def generate_comprehensive_report(summary, detailed_metrics):
    """総合レポート生成"""

    # 成功基準との比較
    targets = {
        "largest_char_accuracy": 0.80,
        "ab_evaluation_rate": 0.70,
        "mean_iou": 0.65,
        "fps": 0.2,  # 5秒/画像以下
    }

    def get_status_emoji(current, target):
        if current >= target:
            return "✅"
        elif current >= target * 0.8:
            return "🟡"
        else:
            return "🔴"

    report = f"""# Phase 0 ベンチマーク: 総合評価レポート

**生成日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**評価対象**: 既存YOLO+SAM+面積最大選択システム  
**データセット**: 人間ラベル101ファイル

---

## 🎯 総合評価結果

### 主要指標

| 指標 | 現在値 | 目標値 | 達成率 | 状態 |
|------|--------|--------|--------|------|
| **Largest-Character Accuracy** | {detailed_metrics.largest_char_accuracy.value:.1%} | {targets['largest_char_accuracy']:.1%} | {(detailed_metrics.largest_char_accuracy.value/targets['largest_char_accuracy']*100):.1f}% | {get_status_emoji(detailed_metrics.largest_char_accuracy.value, targets['largest_char_accuracy'])} |
| **A/B評価率** | {detailed_metrics.ab_evaluation_rate.value:.1%} | {targets['ab_evaluation_rate']:.1%} | {(detailed_metrics.ab_evaluation_rate.value/targets['ab_evaluation_rate']*100):.1f}% | {get_status_emoji(detailed_metrics.ab_evaluation_rate.value, targets['ab_evaluation_rate'])} |
| **Mean IoU** | {detailed_metrics.mean_iou.value:.3f} | {targets['mean_iou']:.3f} | {(detailed_metrics.mean_iou.value/targets['mean_iou']*100):.1f}% | {get_status_emoji(detailed_metrics.mean_iou.value, targets['mean_iou'])} |
| **処理速度 (FPS)** | {detailed_metrics.fps.value:.3f} | {targets['fps']:.3f} | {(detailed_metrics.fps.value/targets['fps']*100):.1f}% | {get_status_emoji(detailed_metrics.fps.value, targets['fps'])} |

### 品質グレード分布

"""

    for grade, count in summary.grade_distribution.items():
        percentage = (count / summary.total_images) * 100
        bar_length = int(percentage / 2)  # 50%で25文字
        bar = "█" * bar_length + "░" * (25 - bar_length)
        report += f"**{grade}評価**: {count:2d}枚 ({percentage:4.1f}%) `{bar}`\n"

    report += f"""

---

## 📊 詳細分析

### 精度分析
- **正解画像数**: {sum(1 for r in detailed_metrics.largest_char_accuracy.notes.split('/') if 'correct' in str(r))} / {summary.total_images}
- **平均IoU**: {detailed_metrics.mean_iou.value:.3f}
- **mAP@0.5**: {detailed_metrics.map_50.value:.3f}
- **mAP@[.5:.95]**: {detailed_metrics.map_95.value:.3f}

### 処理性能
- **平均処理時間**: {summary.mean_processing_time:.2f}秒/画像
- **FPS**: {detailed_metrics.fps.value:.3f}
- **推定メモリ使用量**: {detailed_metrics.memory_usage.value:.1f}GB

### 信頼度統計
"""

    if detailed_metrics.confidence_stats:
        report += f"""- **平均信頼度**: {detailed_metrics.confidence_stats.get('mean', 0):.3f}
- **信頼度範囲**: {detailed_metrics.confidence_stats.get('min', 0):.3f} - {detailed_metrics.confidence_stats.get('max', 0):.3f}
- **標準偏差**: {detailed_metrics.confidence_stats.get('std', 0):.3f}
"""

    report += f"""

---

## 🔍 失敗ケース分析

"""

    if (
        detailed_metrics.failure_analysis
        and detailed_metrics.failure_analysis.get("total_failures", 0) > 0
    ):
        total_failures = detailed_metrics.failure_analysis["total_failures"]
        failure_rate = detailed_metrics.failure_analysis["failure_rate"]

        report += f"""**失敗画像数**: {total_failures}枚 ({failure_rate:.1%})

### 失敗パターン分類
"""

        for pattern, data in detailed_metrics.failure_analysis.get("failure_patterns", {}).items():
            report += f"""
#### {pattern.replace('_', ' ').title()}
- **件数**: {data['count']}枚 ({data['rate']:.1%})
- **説明**: {data['description']}
"""
    else:
        report += "**全画像で成功** - 失敗ケースなし"

    report += f"""

---

## 🚀 改善提言

### 緊急度: HIGH
"""

    # 改善提言を動的生成
    recommendations = []

    if detailed_metrics.largest_char_accuracy.value < 0.5:
        recommendations.append("🔴 **Largest-Character Accuracy < 50%** - 根本的システム見直しが必要")
    elif detailed_metrics.largest_char_accuracy.value < 0.7:
        recommendations.append("🟡 **精度改善が必要** - Phase 1でのコマ検出改善を優先")

    if detailed_metrics.ab_evaluation_rate.value < 0.3:
        recommendations.append("🔴 **品質評価が低い** - 抽出品質の根本的改善が必要")

    if detailed_metrics.fps.value < 0.1:  # 10秒/画像より遅い
        recommendations.append("🟡 **処理速度が遅い** - モデル軽量化またはGPU最適化が必要")

    if not recommendations:
        recommendations.append("✅ **良好な基準性能** - Phase 1への移行準備を開始")

    for rec in recommendations:
        report += f"- {rec}\n"

    report += f"""

### Phase 1 準備事項
1. **データ拡張システム構築**
   - 疑似ラベル生成 + 人手修正による3-5倍データ拡張
   - 作品別Stratified分割でのCV準備

2. **コマ検出ネット準備**
   - Mask R-CNN/YOLOv8-seg環境構築
   - COCO→Manga109→自前データの転移学習準備

3. **評価システム拡張**
   - mIoU測定システム構築
   - リアルタイム進捗監視システム

---

## 📁 生成ファイル

- **詳細結果JSON**: `benchmark_results/phase0/latest_benchmark_results.json`
- **評価指標JSON**: `benchmark_results/phase0/metrics/evaluation_metrics.json` 
- **可視化ダッシュボード**: `benchmark_results/phase0/metrics/performance_dashboard.png`
- **実行ログ**: `phase0_benchmark.log`

---

## 📞 Next Actions

### 即座に実行
1. **Phase 1開始準備**: データ拡張システム構築
2. **課題報告**: 失敗ケース詳細分析レポート作成
3. **リソース確保**: Phase 1学習用GPU環境確認

### Phase 1目標設定
- **Largest-Character Accuracy**: 75%以上
- **コマ検出mIoU**: 80%以上
- **処理速度**: 2秒/画像以下

---

*Phase 0 Benchmark Report - Generated by Automated Evaluation System*  
*プロジェクト進捗: [プロジェクト管理システムで確認]*
"""

    return report


def main():
    """メイン処理"""
    setup_logging()

    logger.info("Phase 0 統合ベンチマーク実行開始")

    success = run_phase0_complete_benchmark()

    if success:
        logger.info("Phase 0 ベンチマーク正常完了")
        return 0
    else:
        logger.error("Phase 0 ベンチマーク実行失敗")
        return 1


if __name__ == "__main__":
    exit(main())
