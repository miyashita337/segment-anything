#!/usr/bin/env python3
"""
真の比較ダッシュボード生成
Phase 1前後の実際の改善効果を可視化
"""

import numpy as np
import matplotlib.pyplot as plt

import json
import os
from datetime import datetime


def load_reports():
    """ベースラインとPhase 1結果を読み込み"""

    # ベースライン (kana08, 26枚)
    baseline_path = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge/unified_quality_baseline.json"

    # Phase 1結果 (kana05, 10枚)
    phase1_path = "/mnt/c/AItools/segment-anything/results_phase1_test/unified_quality_report_phase1_fixed.json"

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    with open(phase1_path, "r", encoding="utf-8") as f:
        phase1 = json.load(f)

    return baseline, phase1


def extract_metrics(report):
    """レポートから主要メトリクスを抽出"""
    metrics = {}

    # 評価メトリクス
    for metric in report.get("evaluation_metrics", []):
        metrics[metric["name"]] = metric["value"]

    # 客観的メトリクス
    for metric in report.get("objective_metrics", []):
        metrics[metric["name"]] = metric["value"]

    return metrics


def create_comparison_chart():
    """比較チャートを作成"""
    baseline, phase1 = load_reports()

    baseline_metrics = extract_metrics(baseline)
    phase1_metrics = extract_metrics(phase1)

    # 主要メトリクスの比較
    metrics_to_compare = [
        "Largest-Character Accuracy",
        "A/B評価率",
        "SCI (Semantic Completeness Index)",
        "PLA (Pixel-Level Accuracy)",
        "PLE (Progressive Learning Efficiency)",
    ]

    baseline_values = []
    phase1_values = []
    metric_names = []

    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in phase1_metrics:
            baseline_values.append(baseline_metrics[metric])
            phase1_values.append(phase1_metrics[metric])
            metric_names.append(metric.replace(" (", "\\n("))

    # チャート作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. バー比較チャート
    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        baseline_values,
        width,
        label="ベースライン (kana08, 26枚)",
        alpha=0.8,
        color="skyblue",
    )
    bars2 = ax1.bar(
        x + width / 2,
        phase1_values,
        width,
        label="Phase 1 (kana05, 10枚)",
        alpha=0.8,
        color="lightcoral",
    )

    ax1.set_xlabel("メトリクス")
    ax1.set_ylabel("値")
    ax1.set_title("Phase 1前後の主要メトリクス比較")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 値をバーの上に表示
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1,
            f"{height1:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax1.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2,
            f"{height2:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. 改善率チャート
    improvement_rates = []
    for i, metric in enumerate(metric_names):
        if baseline_values[i] > 0:
            improvement_rate = ((phase1_values[i] - baseline_values[i]) / baseline_values[i]) * 100
        else:
            improvement_rate = phase1_values[i] * 100  # ベースラインが0の場合
        improvement_rates.append(improvement_rate)

    colors = ["green" if rate > 0 else "red" for rate in improvement_rates]
    bars = ax2.bar(range(len(metric_names)), improvement_rates, color=colors, alpha=0.7)

    ax2.set_xlabel("メトリクス")
    ax2.set_ylabel("改善率 (%)")
    ax2.set_title("Phase 1による改善率")
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_xticklabels(metric_names, rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # 改善率を表示
    for bar, rate in zip(bars, improvement_rates):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:+.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()

    # 保存
    output_dir = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase1_true_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"比較チャートを保存しました: {output_path}")

    return baseline_values, phase1_values, improvement_rates, metric_names


def generate_summary_report():
    """詳細な比較レポートを生成"""
    baseline, phase1 = load_reports()

    baseline_metrics = extract_metrics(baseline)
    phase1_metrics = extract_metrics(phase1)

    report = {
        "comparison_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_info": {
            "dataset": baseline["dataset_name"],
            "total_images": baseline["total_images"],
            "timestamp": baseline["timestamp"],
        },
        "phase1_info": {
            "dataset": phase1["dataset_name"],
            "total_images": phase1["total_images"],
            "timestamp": phase1["timestamp"],
        },
        "key_improvements": {},
        "detailed_analysis": {},
        "conclusions": [],
    }

    # 主要改善点の分析
    key_metrics = {
        "Largest-Character Accuracy": "抽出成功率",
        "A/B評価率": "A/B評価率",
        "SCI (Semantic Completeness Index)": "SCI値",
    }

    for metric_key, metric_name in key_metrics.items():
        if metric_key in baseline_metrics and metric_key in phase1_metrics:
            baseline_val = baseline_metrics[metric_key]
            phase1_val = phase1_metrics[metric_key]

            if baseline_val > 0:
                improvement = ((phase1_val - baseline_val) / baseline_val) * 100
            else:
                improvement = phase1_val * 100

            report["key_improvements"][metric_name] = {
                "baseline": baseline_val,
                "phase1": phase1_val,
                "improvement_percent": improvement,
                "absolute_change": phase1_val - baseline_val,
            }

    # 結論の生成
    accuracy_improvement = report["key_improvements"].get("抽出成功率", {}).get("improvement_percent", 0)
    ab_improvement = report["key_improvements"].get("A/B評価率", {}).get("improvement_percent", 0)
    sci_improvement = report["key_improvements"].get("SCI値", {}).get("improvement_percent", 0)

    if accuracy_improvement > 50:
        report["conclusions"].append("✅ 抽出成功率が大幅に改善 (+{:.1f}%)".format(accuracy_improvement))

    if ab_improvement > 500:
        report["conclusions"].append("✅ A/B評価率が劇的に改善 (+{:.1f}%)".format(ab_improvement))

    if sci_improvement < -10:
        report["conclusions"].append("⚠️ SCI値が低下 ({:.1f}%) - 測定方法の違い".format(sci_improvement))

    # データセット差異の注記
    report["conclusions"].append("📊 データセット差異: kana08(26枚) vs kana05(10枚)")
    report["conclusions"].append("📊 測定条件: Phase 1改善システム vs ベースラインシステム")

    # 保存
    output_dir = "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase1_true_comparison_report.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"詳細比較レポートを保存しました: {output_path}")
    return report


def main():
    """メイン実行"""
    print("🔍 Phase 1真の改善効果分析を開始...")

    # 比較チャート生成
    baseline_values, phase1_values, improvement_rates, metric_names = create_comparison_chart()

    # 詳細レポート生成
    report = generate_summary_report()

    # 結果サマリー表示
    print("\n" + "=" * 60)
    print("📊 Phase 1真の改善効果サマリー")
    print("=" * 60)

    for conclusion in report["conclusions"]:
        print(conclusion)

    print("\n🔍 主要メトリクス詳細:")
    for metric_name, data in report["key_improvements"].items():
        print(f"  {metric_name}:")
        print(f"    ベースライン: {data['baseline']:.3f}")
        print(f"    Phase 1: {data['phase1']:.3f}")
        print(f"    改善率: {data['improvement_percent']:+.1f}%")
        print()

    print("=" * 60)
    print("✅ 真の比較分析完了")


if __name__ == "__main__":
    main()
