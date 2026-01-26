#!/usr/bin/env python3
"""
OpenCV品質スコア生成システム

実際の抽出画像からOpenCVを使用して品質スコアを算出し、
extraction_result.jsonを生成または更新する。

QCC-023: 効果サイズ計算システム用の実データ生成
"""

import numpy as np
import cv2

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))


class OpenCVQualityGenerator:
    """OpenCV品質スコア生成クラス"""

    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)

        # 画像拡張子リスト
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def calculate_image_quality_opencv(self, image_path: Path) -> Dict:
        """
        OpenCVを使用した画像品質スコア計算

        Args:
            image_path: 画像ファイルパス

        Returns:
            Dict: 品質評価結果
        """
        try:
            # 画像読み込み
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"画像読み込み失敗: {image_path}")

            # グレースケール変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 基本情報
            height, width = gray.shape

            # 1. エッジ鮮明度（Laplacian分散）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_sharpness = np.var(laplacian)

            # 2. コントラスト（標準偏差）
            contrast = np.std(gray)

            # 3. 明度（平均値）
            brightness = np.mean(gray)

            # 4. エッジ検出率（Canny）
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / (width * height)

            # 5. ノイズレベル（ガウシアンフィルタとの差分）
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_level = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))

            # 6. 色彩豊かさ（カラー画像の場合）
            if len(img.shape) == 3:
                # HSV変換
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation_mean = np.mean(hsv[:, :, 1])
                color_richness = saturation_mean / 255.0
            else:
                color_richness = 0.0

            # 7. 構造的品質（SSIM代替：構造テンソル）
            # Sobel勾配
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            structural_quality = np.sqrt(np.mean(sobel_x**2 + sobel_y**2))

            # 総合品質スコア計算（0-1範囲に正規化）
            # 各指標を0-1範囲に正規化して重み付け平均
            normalized_edge = min(edge_sharpness / 1000.0, 1.0)  # Laplacian分散の正規化
            normalized_contrast = min(contrast / 100.0, 1.0)  # コントラストの正規化
            normalized_brightness = 1.0 - abs(brightness - 128) / 128.0  # 適正明度からの距離
            normalized_edge_ratio = min(edge_ratio * 10, 1.0)  # エッジ検出率の正規化
            normalized_noise = max(1.0 - noise_level / 50.0, 0.0)  # ノイズレベル（少ないほど良い）
            normalized_structural = min(structural_quality / 200.0, 1.0)  # 構造品質の正規化

            # 重み付け総合スコア
            quality_score = (
                0.25 * normalized_edge
                + 0.20 * normalized_contrast  # エッジ鮮明度
                + 0.15 * normalized_brightness  # コントラスト
                + 0.15 * normalized_edge_ratio  # 適正明度
                + 0.10 * normalized_noise  # エッジ検出率
                + 0.10 * color_richness  # ノイズの少なさ
                + 0.05 * normalized_structural  # 色彩豊かさ  # 構造品質
            )

            return {
                "image_path": str(image_path),
                "width": int(width),
                "height": int(height),
                "file_size_bytes": image_path.stat().st_size,
                "quality_score": float(quality_score),
                "edge_sharpness": float(edge_sharpness),
                "contrast": float(contrast),
                "brightness": float(brightness),
                "edge_ratio": float(edge_ratio),
                "noise_level": float(noise_level),
                "color_richness": float(color_richness),
                "structural_quality": float(structural_quality),
                "analysis_timestamp": datetime.now().isoformat(),
                "opencv_version": cv2.__version__,
            }

        except Exception as e:
            return {
                "image_path": str(image_path),
                "error": str(e),
                "quality_score": 0.0,
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def generate_extraction_result_json(
        self, tracker_id: str, force_regenerate: bool = False
    ) -> Dict:
        """
        extraction_result.jsonの生成または更新

        Args:
            tracker_id: トラッカーID
            force_regenerate: 既存JSONファイルを強制再生成

        Returns:
            Dict: 生成結果
        """
        tracker_dir = self.workspace_base / tracker_id
        extraction_dir = tracker_dir / "extraction"
        json_path = tracker_dir / "extraction_result.json"

        result = {
            "tracker_id": tracker_id,
            "success": False,
            "json_path": str(json_path),
            "images_analyzed": 0,
            "mean_quality_score": 0.0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "error": None,
        }

        # 既存JSONファイル確認
        if json_path.exists() and not force_regenerate:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                # 既存データに品質スコアがある場合はスキップ
                if (
                    "results" in existing_data
                    and existing_data["results"]
                    and "quality_score" in existing_data["results"][0]
                ):
                    result.update(
                        {
                            "success": True,
                            "images_analyzed": len(existing_data["results"]),
                            "mean_quality_score": existing_data.get("mean_quality_score", 0.0),
                            "successful_extractions": existing_data.get(
                                "successful_extractions", 0
                            ),
                            "skipped_reason": "品質スコア付きJSONが既存のためスキップ",
                        }
                    )
                    return result

            except Exception as e:
                print(f"⚠️  既存JSON読み込みエラー（再生成します）: {e}")

        if not extraction_dir.exists():
            result["error"] = f"extraction/ディレクトリ未存在: {extraction_dir}"
            return result

        try:
            # extraction内の画像ファイル収集
            image_files = []
            for item in extraction_dir.rglob("*"):
                if item.is_file() and item.suffix.lower() in self.image_extensions:
                    image_files.append(item)

            if not image_files:
                result["error"] = f"extraction/内に画像ファイルなし"
                return result

            print(f"🔍 {tracker_id}: {len(image_files)}画像を分析中...")

            # 各画像の品質分析
            analysis_results = []
            successful_count = 0
            failed_count = 0
            total_quality = 0.0

            for i, image_path in enumerate(image_files, 1):
                if i % 10 == 0:
                    print(f"   [{i}/{len(image_files)}] 処理中: {image_path.name}")

                analysis = self.calculate_image_quality_opencv(image_path)

                if "error" in analysis:
                    failed_count += 1
                else:
                    successful_count += 1
                    total_quality += analysis["quality_score"]

                analysis_results.append(analysis)

            # 統計計算
            mean_quality = total_quality / successful_count if successful_count > 0 else 0.0

            # ワークフロー互換extraction_result.json構造作成
            extraction_result = {
                "tracker_id": tracker_id,
                "extraction_results": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "opencv_version": cv2.__version__,
                    "total_images": len(image_files),
                    "successful_extractions": successful_count,
                    "failed_extractions": failed_count,
                    "success_rate": successful_count / len(image_files) if image_files else 0.0,
                    "mean_quality_score": mean_quality,
                    "quality_statistics": self._calculate_quality_statistics(analysis_results),
                    "generation_method": "opencv_analysis",
                    "quality_score_range": [0.0, 1.0],
                    "quality_algorithm": "weighted_composite_score",
                    "metadata": {
                        "extraction_dir": str(extraction_dir),
                        "image_extensions": list(self.image_extensions),
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
                },
                "results": analysis_results,
            }

            # JSONファイル保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)

            result.update(
                {
                    "success": True,
                    "images_analyzed": len(image_files),
                    "mean_quality_score": mean_quality,
                    "successful_extractions": successful_count,
                    "failed_extractions": failed_count,
                }
            )

            print(
                f"✅ {tracker_id}: 品質分析完了（平均品質: {mean_quality:.3f}, 成功: {successful_count}/{len(image_files)}）"
            )

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {tracker_id}: 品質分析エラー → {e}")

        return result

    def _calculate_quality_statistics(self, analysis_results: List[Dict]) -> Dict:
        """品質統計計算"""
        quality_scores = [
            r["quality_score"]
            for r in analysis_results
            if "quality_score" in r and "error" not in r
        ]

        if not quality_scores:
            return {}

        return {
            "count": len(quality_scores),
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores)),
            "median": float(np.median(quality_scores)),
            "q25": float(np.percentile(quality_scores, 25)),
            "q75": float(np.percentile(quality_scores, 75)),
        }

    def batch_generate_quality_scores(
        self, tracker_ids: List[str], force_regenerate: bool = False
    ) -> Dict:
        """
        複数トラッカーの一括品質スコア生成

        Args:
            tracker_ids: トラッカーIDリスト
            force_regenerate: 既存JSONファイルを強制再生成

        Returns:
            Dict: バッチ生成結果
        """
        print(f"🚀 OpenCV品質スコア一括生成開始: {len(tracker_ids)}個のトラッカー\n")

        results = {
            "total_trackers": len(tracker_ids),
            "successful_generations": 0,
            "failed_generations": 0,
            "skipped_generations": 0,
            "total_images_analyzed": 0,
            "tracker_results": {},
        }

        for i, tracker_id in enumerate(tracker_ids, 1):
            print(f"[{i}/{len(tracker_ids)}] 処理中: {tracker_id}")

            generation_result = self.generate_extraction_result_json(tracker_id, force_regenerate)
            results["tracker_results"][tracker_id] = generation_result

            if generation_result["success"]:
                if "skipped_reason" in generation_result:
                    results["skipped_generations"] += 1
                    print(f"   ⏭️  スキップ: {generation_result['skipped_reason']}")
                else:
                    results["successful_generations"] += 1
                    results["total_images_analyzed"] += generation_result["images_analyzed"]
                    print(
                        f"   ✅ 完了: {generation_result['images_analyzed']}画像, 平均品質={generation_result['mean_quality_score']:.3f}"
                    )
            else:
                results["failed_generations"] += 1
                print(f"   ❌ 失敗: {generation_result['error']}")

            print()

        # 結果サマリー
        print(f"📊 OpenCV品質スコア生成結果:")
        print(f"   - 総トラッカー数: {results['total_trackers']}")
        print(f"   - 生成成功: {results['successful_generations']}")
        print(f"   - 生成スキップ: {results['skipped_generations']}")
        print(f"   - 生成失敗: {results['failed_generations']}")
        print(f"   - 総画像分析数: {results['total_images_analyzed']:,}枚")
        print(
            f"   - 成功率: {(results['successful_generations']+results['skipped_generations'])/results['total_trackers']*100:.1f}%"
        )

        return results


def main():
    """メイン実行関数"""
    generator = OpenCVQualityGenerator()

    # テスト用：処理可能トラッカーの一部で実行
    test_trackers = ["QCC-022", "QCC-021", "QI-003", "QI-002", "QCC-FIX-001"]

    print("🧪 OpenCV品質スコア生成テスト実行")
    print(f"対象: {test_trackers}")
    print()

    try:
        results = generator.batch_generate_quality_scores(test_trackers, force_regenerate=False)

        if results["successful_generations"] + results["skipped_generations"] > 0:
            print("\n✅ OpenCV品質スコア生成テスト完了")
        else:
            print("\n❌ OpenCV品質スコア生成テスト失敗")

        return results

    except Exception as e:
        print(f"❌ OpenCV品質スコア生成システムエラー: {e}")
        raise


if __name__ == "__main__":
    main()
