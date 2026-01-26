#!/usr/bin/env python3
"""
QCC-011統合テスト
失敗パターン分析システムの統合動作確認
"""

import numpy as np
import cv2

import json
import shutil
import tempfile
from pathlib import Path


def create_demo_data():
    """デモ抽出結果データ作成"""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"🔧 デモデータ作成: {temp_dir}")

    # 結果ディレクトリ作成
    extraction_dir = temp_dir / "extraction"
    failed_dir = temp_dir / "failed"
    success_dir = temp_dir / "success"

    extraction_dir.mkdir(parents=True)
    failed_dir.mkdir(parents=True)
    success_dir.mkdir(parents=True)

    # デモ画像作成（失敗パターン）
    print("📸 デモ失敗画像作成中...")

    # 暗い画像パターン
    dark_img = np.ones((100, 150, 3), dtype=np.uint8) * 20  # 暗い
    cv2.imwrite(str(failed_dir / "dark_01.jpg"), dark_img)
    cv2.imwrite(str(failed_dir / "dark_02.jpg"), dark_img * 1.2)

    # 過露出パターン
    bright_img = np.ones((100, 150, 3), dtype=np.uint8) * 240  # 明るい
    cv2.imwrite(str(failed_dir / "bright_01.jpg"), bright_img)
    cv2.imwrite(str(failed_dir / "bright_02.jpg"), bright_img * 0.9)

    # ノイズ画像パターン
    noise_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    cv2.imwrite(str(failed_dir / "noise_01.jpg"), noise_img)

    # 成功画像パターン
    print("✅ デモ成功画像作成中...")
    normal_img = np.ones((100, 150, 3), dtype=np.uint8) * 120  # 普通
    cv2.imwrite(str(success_dir / "success_01.jpg"), normal_img)
    cv2.imwrite(str(success_dir / "success_02.jpg"), normal_img + 20)
    cv2.imwrite(str(success_dir / "success_03.jpg"), normal_img - 10)

    # 抽出結果JSON作成
    extraction_results = {
        "timestamp": "2025-08-09T19:35:00Z",
        "total_images": 8,
        "success_count": 3,
        "failed_count": 5,
        "results_directory": str(temp_dir),
        "quality_distribution": {"A": 2, "B": 1, "C": 1, "D": 2, "F": 2},
        "processing_details": {
            "avg_processing_time": 3.2,
            "yolo_threshold": 0.07,
            "sam_model": "vit_h",
        },
    }

    results_file = temp_dir / "extraction_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(extraction_results, f, indent=2, ensure_ascii=False)

    print(f"📊 結果ファイル: {results_file}")
    return temp_dir, results_file


def test_failure_pattern_analyzer():
    """失敗パターン分析システム単体テスト"""
    print("\n🔍 失敗パターン分析システム単体テスト")

    try:
        from features.analysis.failure_pattern_analyzer import FailurePatternAnalyzer

        # デモデータ作成
        temp_dir, _ = create_demo_data()
        failed_dir = temp_dir / "failed"
        success_dir = temp_dir / "success"

        # 分析実行
        analyzer = FailurePatternAnalyzer()
        print(f"📂 失敗画像: {len(list(failed_dir.glob('*.jpg')))}枚")
        print(f"📂 成功画像: {len(list(success_dir.glob('*.jpg')))}枚")

        results = analyzer.analyze_failure_patterns(failed_dir, success_dir)

        # 結果検証
        print(f"✅ 分析完了: {results.get('total_failed_images', 0)}枚分析")
        print(f"🎯 クラスタ数: {results.get('clustering', {}).get('n_clusters', 0)}")
        print(f"🚨 異常検出: {results.get('anomalies', {}).get('n_anomalies', 0)}枚")

        # レポート生成
        report_path = temp_dir / "failure_analysis_report.txt"
        report_text = analyzer.generate_report(report_path)
        print(f"📝 レポート生成: {len(report_text)}文字")

        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_unified_quality_checker():
    """統合品質チェックシステム統合テスト"""
    print("\n📊 統合品質チェックシステム統合テスト")

    try:
        from tools.core.unified_quality_checker import UnifiedQualityChecker

        # デモデータ作成
        temp_dir, results_file = create_demo_data()

        # 品質チェック実行
        checker = UnifiedQualityChecker()
        print(f"📄 結果ファイル: {results_file}")

        report = checker.check_extraction_results(str(results_file))

        # 結果検証
        print(f"✅ 品質チェック完了")
        print(f"🎯 総合スコア: {report.overall_score:.1%}")
        print(f"📈 合格指標: {report.passed_metrics}/{report.total_metrics}")
        print(f"🏆 総合判定: {report.status}")

        # カテゴリ別確認
        all_metrics = report.evaluation_metrics + report.mask_metrics + report.objective_metrics

        pattern_metrics = [m for m in all_metrics if m.category == "pattern_analysis"]
        print(f"🔍 パターン分析指標: {len(pattern_metrics)}個")

        for metric in pattern_metrics:
            print(f"  - {metric.name}: {metric.value:.3f} ({metric.status})")

        # サマリー表示
        checker.print_report_summary(report)

        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🚀 QCC-011統合テスト開始")
    print("=" * 60)

    # 単体テスト
    analyzer_ok = test_failure_pattern_analyzer()

    # 統合テスト
    checker_ok = test_unified_quality_checker()

    # 結果報告
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print(f"  🔍 失敗パターン分析: {'✅ 成功' if analyzer_ok else '❌ 失敗'}")
    print(f"  📊 統合品質チェック: {'✅ 成功' if checker_ok else '❌ 失敗'}")

    overall_success = analyzer_ok and checker_ok
    print(f"  🎯 総合結果: {'✅ 全テスト成功' if overall_success else '❌ テスト失敗'}")

    if overall_success:
        print("\n🎉 QCC-011統合テスト完了！失敗パターン分析システムが正常に動作しています。")
    else:
        print("\n💣 QCC-011統合テスト失敗。上記エラーを確認してください。")

    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())
