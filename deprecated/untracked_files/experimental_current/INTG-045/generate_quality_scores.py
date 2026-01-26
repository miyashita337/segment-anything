#!/usr/bin/env python3
"""
QCA-001品質スコア生成スクリプト
OpenCV 7コンポーネント品質解析システム
"""
import numpy as np
import cv2

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def calculate_quality_score(image):
    """
    OpenCV 7コンポーネント品質解析
    - エッジ鮮明度
    - コントラスト
    - 明度
    - エッジ比率
    - ノイズレベル
    - 色彩豊富度
    - 構造品質
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. エッジ鮮明度 (Edge Sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_sharpness = np.var(laplacian)

    # 2. コントラスト (Contrast)
    contrast = np.std(gray)

    # 3. 明度 (Brightness)
    brightness = np.mean(gray)

    # 4. エッジ比率 (Edge Ratio)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 5. ノイズレベル (Noise Level)
    noise_level = np.std(laplacian)

    # 6. 色彩豊富度 (Color Richness)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    color_richness = np.std(saturation) / 255.0

    # 7. 構造品質 (Structural Quality)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    structural_quality = np.mean(np.sqrt(sobelx**2 + sobely**2))

    # 重み付け複合スコア計算
    weights = {
        "edge_sharpness": 0.2,
        "contrast": 0.15,
        "brightness": 0.1,
        "edge_ratio": 0.2,
        "noise_level": 0.1,
        "color_richness": 0.15,
        "structural_quality": 0.1,
    }

    # 正規化 (0-1範囲)
    normalized_scores = {
        "edge_sharpness": min(edge_sharpness / 10000.0, 1.0),
        "contrast": min(contrast / 128.0, 1.0),
        "brightness": min(brightness / 255.0, 1.0),
        "edge_ratio": min(edge_ratio / 0.1, 1.0),
        "noise_level": min(noise_level / 50.0, 1.0),
        "color_richness": min(color_richness, 1.0),
        "structural_quality": min(structural_quality / 200.0, 1.0),
    }

    # 重み付け複合スコア
    quality_score = sum(normalized_scores[key] * weights[key] for key in weights.keys())

    return {
        "quality_score": quality_score,
        "edge_sharpness": edge_sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "edge_ratio": edge_ratio,
        "noise_level": noise_level,
        "color_richness": color_richness,
        "structural_quality": structural_quality,
    }


def analyze_extraction_directory(extraction_dir):
    """抽出ディレクトリの全画像を解析"""

    extraction_path = Path(extraction_dir)
    if not extraction_path.exists():
        raise FileNotFoundError(f"抽出ディレクトリが存在しません: {extraction_dir}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(extraction_path.glob(f"*{ext}")))

    if not image_files:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {extraction_dir}")

    print(f"📊 QCA-001品質解析開始: {len(image_files)}枚")

    results = []
    quality_scores = []

    for i, image_file in enumerate(image_files, 1):
        print(f"   {i:2}/{len(image_files)}: {image_file.name}", end=" ... ")

        try:
            # 画像読み込み
            image = cv2.imread(str(image_file))
            if image is None:
                print("❌ 読み込み失敗")
                continue

            # 品質解析実行
            quality_data = calculate_quality_score(image)

            # メタデータ追加
            quality_data.update(
                {
                    "image_path": str(image_file),
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "file_size_bytes": image_file.stat().st_size,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "opencv_version": cv2.__version__,
                }
            )

            results.append(quality_data)
            quality_scores.append(quality_data["quality_score"])

            print(f"✅ スコア: {quality_data['quality_score']:.3f}")

        except Exception as e:
            print(f"❌ エラー: {e}")
            continue

    # 統計値計算
    if quality_scores:
        quality_stats = {
            "count": len(quality_scores),
            "mean": np.mean(quality_scores),
            "std": np.std(quality_scores),
            "min": np.min(quality_scores),
            "max": np.max(quality_scores),
            "median": np.median(quality_scores),
            "q25": np.percentile(quality_scores, 25),
            "q75": np.percentile(quality_scores, 75),
        }
    else:
        quality_stats = {}

    # 結果まとめ
    final_result = {
        "tracker_id": "QCA-001",
        "analysis_timestamp": datetime.now().isoformat(),
        "opencv_version": cv2.__version__,
        "total_images": len(image_files),
        "successful_extractions": len(results),
        "failed_extractions": len(image_files) - len(results),
        "success_rate": len(results) / len(image_files) if image_files else 0.0,
        "mean_quality_score": quality_stats.get("mean", 0.0),
        "quality_statistics": quality_stats,
        "results": results,
        "generation_method": "opencv_analysis",
        "quality_score_range": [0.0, 1.0],
        "quality_algorithm": "weighted_composite_score",
        "metadata": {
            "extraction_dir": str(extraction_dir),
            "image_extensions": image_extensions,
            "analysis_components": [
                "edge_sharpness",
                "contrast",
                "brightness",
                "edge_ratio",
                "noise_level",
                "color_richness",
                "structural_quality",
            ],
        },
    }

    return final_result


def main():
    """メイン処理"""

    extraction_dir = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QCA-001/extraction"
    output_file = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QCA-001/extraction_result.json"

    print("🔧 QCA-001品質スコア生成システム起動")
    print(f"📁 抽出ディレクトリ: {extraction_dir}")
    print(f"📄 出力ファイル: {output_file}")
    print()

    try:
        # 品質解析実行
        result = analyze_extraction_directory(extraction_dir)

        # JSON出力
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        # 結果報告
        print()
        print("✅ QCA-001品質解析完了")
        print(f"📊 総画像数: {result['total_images']}枚")
        print(f"📈 成功率: {result['success_rate']*100:.1f}%")
        print(f"🎯 平均品質スコア: {result['mean_quality_score']:.3f}")
        print(f"📄 結果ファイル: {output_file}")

        # 統計サマリー
        if result["quality_statistics"]:
            stats = result["quality_statistics"]
            print(f"📊 統計情報:")
            print(f"   平均: {stats['mean']:.3f}")
            print(f"   標準偏差: {stats['std']:.3f}")
            print(f"   最小値: {stats['min']:.3f}")
            print(f"   最大値: {stats['max']:.3f}")
            print(f"   中央値: {stats['median']:.3f}")

        return 0

    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
