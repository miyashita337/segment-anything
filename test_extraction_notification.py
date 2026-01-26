#!/usr/bin/env python3
"""
抽出パイプライン通知システムのテスト

新しく実装したExtractionNotifierをテスト
"""

import sys
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent))

try:
    from features.extraction.extraction_notifier import (
        ExtractionNotifier,
        create_extraction_results_dict,
    )

    print("✅ ExtractionNotifier import成功")
except ImportError as e:
    print(f"❌ ExtractionNotifier import失敗: {e}")
    sys.exit(1)


def test_extraction_notification():
    """抽出通知のテスト"""
    print("🧪 抽出通知システムテスト開始")

    # テスト用抽出結果データ
    successful_extractions = [
        {
            "input_path": "/path/to/test1.jpg",
            "output_path": "/path/to/test1_extracted.jpg",
            "extracted_files": ["/path/to/test1_extracted.jpg"],
            "quality_score": 0.85,
            "processing_time": 2.3,
        },
        {
            "input_path": "/path/to/test2.jpg",
            "output_path": "/path/to/test2_extracted.jpg",
            "extracted_files": ["/path/to/test2_extracted.jpg"],
            "quality_score": 0.72,
            "processing_time": 1.8,
        },
    ]

    # 抽出結果辞書を作成
    extraction_results = create_extraction_results_dict(
        total_images=5,
        successful_extractions=successful_extractions,
        processing_time=12.5,
        quality_method="fullbody_priority",
        output_dir="/tmp/test_output",
        quality_distribution={"A": 1, "B": 1, "C": 0, "D": 0, "E": 0, "F": 3},
    )

    print(f"📊 テスト用抽出結果:")
    print(f"   総画像数: {extraction_results['total_images']}")
    print(f"   成功数: {len(extraction_results['successful_extractions'])}")
    print(f"   処理時間: {extraction_results['total_processing_time']}秒")

    # 通知システム初期化
    try:
        notifier = ExtractionNotifier()
        print("✅ ExtractionNotifier初期化成功")
    except Exception as e:
        print(f"❌ ExtractionNotifier初期化失敗: {e}")
        return False

    # テキストのみ通知テスト
    print("\n📱 テキストのみ通知テスト...")
    try:
        success = notifier.send_extraction_completion_notification(
            extraction_results, include_images=False
        )

        if success:
            print("✅ テキストのみ通知送信成功")
        else:
            print("⚠️ テキストのみ通知送信失敗")

    except Exception as e:
        print(f"❌ テキストのみ通知エラー: {e}")
        return False

    # 画像付き通知テスト（実際の画像がないのでエラーになる予定）
    print("\n🖼️ 画像付き通知テスト（エラー予定）...")
    try:
        success = notifier.send_extraction_completion_notification(
            extraction_results, include_images=True
        )

        if success:
            print("✅ 画像付き通知送信成功（予期しない結果）")
        else:
            print("⚠️ 画像付き通知送信失敗（予期される結果）")

    except Exception as e:
        print(f"⚠️ 画像付き通知エラー（予期される結果）: {e}")

    print("\n🎯 抽出通知システムテスト完了")
    return True


if __name__ == "__main__":
    test_extraction_notification()
