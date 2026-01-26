#!/usr/bin/env python3
"""
YOLO閾値比較テスト（0.07 vs 0.005）
処理時間・検出数・品質の定量測定

CLAUDE.md準拠の事実ベース報告用テストスクリプト
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.append(".")


def compare_yolo_thresholds():
    """YOLO閾値0.07と0.005の比較テスト"""

    print("🧪 YOLO閾値比較テスト開始")
    print("=" * 60)

    # テスト設定
    test_images = ["test_small/img001.jpg", "test_small/img002.jpg", "test_small/img003.jpg"]

    thresholds = [0.07, 0.005]
    results = {}

    for threshold in thresholds:
        print(f"\n📊 閾値 {threshold} でのテスト開始")
        print("-" * 40)

        threshold_results = {
            "threshold": threshold,
            "images": [],
            "total_time": 0,
            "success_count": 0,
            "total_detections": 0,
        }

        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"⚠️ 画像が見つかりません: {img_path}")
                continue

            print(f"🔍 処理中: {img_path}")

            start_time = time.time()

            try:
                # Phase 2テストスクリプトを呼び出し
                from tools.test_phase2_simple import test_phase2_on_failed_images

                # テスト実行（一時的に1画像のみ）
                result = test_single_image_with_threshold(img_path, threshold)

                processing_time = time.time() - start_time

                image_result = {
                    "image": img_path,
                    "success": result.get("success", False),
                    "processing_time": processing_time,
                    "detections": result.get("detections", 0),
                    "quality_score": result.get("quality_score", 0.0),
                    "error": result.get("error", ""),
                }

                threshold_results["images"].append(image_result)
                threshold_results["total_time"] += processing_time

                if image_result["success"]:
                    threshold_results["success_count"] += 1

                threshold_results["total_detections"] += image_result["detections"]

                print(
                    f"   {'✅' if image_result['success'] else '❌'} "
                    f"{processing_time:.1f}秒, 検出数: {image_result['detections']}"
                )

            except Exception as e:
                print(f"   ❌ エラー: {str(e)}")
                threshold_results["images"].append(
                    {
                        "image": img_path,
                        "success": False,
                        "error": str(e),
                        "processing_time": time.time() - start_time,
                    }
                )

        results[f"threshold_{threshold}"] = threshold_results

    # 結果分析・報告
    print_comparison_report(results)

    return results


def test_single_image_with_threshold(image_path, threshold):
    """単一画像でのYOLO閾値テスト"""

    # 簡易YOLO検出テスト
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        results = model(image_path, conf=threshold, verbose=False)

        detections = len(results[0].boxes) if results[0].boxes is not None else 0

        return {
            "success": detections > 0,
            "detections": detections,
            "quality_score": 0.5 if detections > 0 else 0.0,
        }

    except Exception as e:
        return {"success": False, "detections": 0, "quality_score": 0.0, "error": str(e)}


def print_comparison_report(results):
    """CLAUDE.md準拠の事実ベース報告"""

    print("\n" + "=" * 60)
    print("📊 YOLO閾値比較テスト結果報告")
    print("=" * 60)

    for key, data in results.items():
        threshold = data["threshold"]
        success_rate = (data["success_count"] / len(data["images"])) * 100 if data["images"] else 0
        avg_time = data["total_time"] / len(data["images"]) if data["images"] else 0

        print(f"\n🎯 閾値 {threshold}:")
        print(f"   処理総数: {len(data['images'])}件")
        print(f"   成功数: {data['success_count']}件 ({success_rate:.1f}%)")
        print(f"   失敗数: {len(data['images']) - data['success_count']}件")
        print(f"   総検出数: {data['total_detections']}個")
        print(f"   平均処理時間: {avg_time:.1f}秒")
        print(f"   合計処理時間: {data['total_time']:.1f}秒")

    # 比較分析
    if len(results) == 2:
        keys = list(results.keys())
        data1, data2 = results[keys[0]], results[keys[1]]

        print(f"\n📈 比較分析:")
        print(f"   検出数増加: {data2['total_detections'] - data1['total_detections']}個")
        print(f"   処理時間増加: {data2['total_time'] - data1['total_time']:.1f}秒")
        print(
            f"   成功率変化: {((data2['success_count'] / len(data2['images'])) - (data1['success_count'] / len(data1['images']))) * 100:.1f}%"
        )

    print(f"\n⏰ テスト完了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    results = compare_yolo_thresholds()

    # 結果をJSONで保存
    with open("/tmp/yolo_threshold_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 詳細結果: /tmp/yolo_threshold_comparison.json に保存")
