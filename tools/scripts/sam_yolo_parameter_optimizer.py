#!/usr/bin/env python3
"""
SAM+YOLO精度向上パラメータ最適化システム
正解データに基づく最適パラメータ探索と調整
"""

import numpy as np
import cv2

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SAMYOLOParameterOptimizer:
    """SAM+YOLOパラメータ最適化器"""

    def __init__(self, correct_annotations_path: Path, original_dir: Path, output_dir: Path):
        self.correct_annotations_path = Path(correct_annotations_path)
        self.original_dir = Path(original_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 正解データ読み込み
        with open(self.correct_annotations_path, "r", encoding="utf-8") as f:
            self.correct_data = json.load(f)

        # 最適化対象パラメータ定義
        self.parameter_sets = [
            {
                "name": "current_default",
                "description": "現在のデフォルト設定",
                "yolo_conf_threshold": 0.03,
                "yolo_iou_threshold": 0.45,
                "sam_points_per_side": 32,
                "quality_method": "fullbody_priority",
                "area_weight": 0.2,
                "fullbody_weight": 0.4,
                "central_weight": 0.15,
                "grounded_weight": 0.15,
                "confidence_weight": 0.1,
            },
            {
                "name": "high_precision",
                "description": "高精度設定（YOLO閾値上昇）",
                "yolo_conf_threshold": 0.1,
                "yolo_iou_threshold": 0.3,
                "sam_points_per_side": 64,
                "quality_method": "fullbody_priority",
                "area_weight": 0.3,
                "fullbody_weight": 0.5,
                "central_weight": 0.1,
                "grounded_weight": 0.05,
                "confidence_weight": 0.05,
            },
            {
                "name": "large_character_focus",
                "description": "大キャラ重視設定",
                "yolo_conf_threshold": 0.05,
                "yolo_iou_threshold": 0.4,
                "sam_points_per_side": 32,
                "quality_method": "size_priority",
                "area_weight": 0.6,
                "fullbody_weight": 0.2,
                "central_weight": 0.1,
                "grounded_weight": 0.05,
                "confidence_weight": 0.05,
            },
            {
                "name": "balanced_optimized",
                "description": "バランス最適化設定",
                "yolo_conf_threshold": 0.07,
                "yolo_iou_threshold": 0.35,
                "sam_points_per_side": 48,
                "quality_method": "balanced",
                "area_weight": 0.25,
                "fullbody_weight": 0.35,
                "central_weight": 0.2,
                "grounded_weight": 0.1,
                "confidence_weight": 0.1,
            },
        ]

    def test_parameter_set(
        self, param_set: Dict, test_images: List[str], max_images: int = 5
    ) -> Dict:
        """パラメータセットのテスト実行"""
        logger.info(f"🧪 パラメータセット '{param_set['name']}' テスト開始")
        logger.info(f"   {param_set['description']}")

        # テスト用出力ディレクトリ
        test_output_dir = self.output_dir / f"param_test_{param_set['name']}"
        test_output_dir.mkdir(exist_ok=True)

        results = {
            "parameter_set": param_set,
            "test_results": {},
            "summary": {
                "total_tested": 0,
                "successful_extractions": 0,
                "total_iou": 0.0,
                "grade_counts": {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
            },
        }

        # 限定数の画像でテスト
        test_subset = test_images[:max_images]

        for i, filename in enumerate(test_subset, 1):
            logger.info(f"  [{i}/{len(test_subset)}] テスト実行: {filename}")

            base_name = filename.replace(".jpg", "").replace(".png", "")
            input_path = self.original_dir / filename
            output_path = test_output_dir / f"{base_name}_test_extracted.jpg"

            # 個別画像でextract_character.py実行
            extraction_result = self.run_single_extraction(input_path, output_path, param_set)

            if extraction_result["success"]:
                # IoU計算
                iou_score = self.calculate_extraction_iou(filename, output_path)
                grade = self.get_quality_grade(iou_score)

                results["test_results"][filename] = {
                    "extraction_success": True,
                    "iou_score": iou_score,
                    "quality_grade": grade,
                    "output_path": str(output_path),
                }

                results["summary"]["successful_extractions"] += 1
                results["summary"]["total_iou"] += iou_score
                results["summary"]["grade_counts"][grade] += 1

                logger.info(f"     ✅ 成功 - IoU: {iou_score:.3f}, Grade: {grade}")
            else:
                results["test_results"][filename] = {
                    "extraction_success": False,
                    "error": extraction_result.get("error", "Unknown error"),
                    "iou_score": 0.0,
                    "quality_grade": "F",
                }
                results["summary"]["grade_counts"]["F"] += 1
                logger.info(f"     ❌ 失敗 - {extraction_result.get('error', 'Unknown error')}")

            results["summary"]["total_tested"] += 1

        # 平均IoU計算
        if results["summary"]["successful_extractions"] > 0:
            results["summary"]["average_iou"] = (
                results["summary"]["total_iou"] / results["summary"]["successful_extractions"]
            )
        else:
            results["summary"]["average_iou"] = 0.0

        # 成功率計算
        results["summary"]["success_rate"] = (
            results["summary"]["successful_extractions"] / results["summary"]["total_tested"]
        )
        results["summary"]["ab_success_rate"] = (
            results["summary"]["grade_counts"]["A"] + results["summary"]["grade_counts"]["B"]
        ) / results["summary"]["total_tested"]

        return results

    def run_single_extraction(self, input_path: Path, output_path: Path, param_set: Dict) -> Dict:
        """単一画像での抽出実行"""
        try:
            # extract_character.pyの実行コマンド構築
            cmd = [
                sys.executable,
                "features/extraction/commands/extract_character.py",
                str(input_path),
                "-o",
                str(output_path),
                "--verbose",
            ]

            # 環境変数でパラメータ設定
            import os

            env = dict(os.environ)
            env.update(
                {
                    "YOLO_CONF_THRESHOLD": str(param_set["yolo_conf_threshold"]),
                    "YOLO_IOU_THRESHOLD": str(param_set["yolo_iou_threshold"]),
                    "SAM_POINTS_PER_SIDE": str(param_set["sam_points_per_side"]),
                    "QUALITY_METHOD": param_set["quality_method"],
                    "AREA_WEIGHT": str(param_set["area_weight"]),
                    "FULLBODY_WEIGHT": str(param_set["fullbody_weight"]),
                    "CENTRAL_WEIGHT": str(param_set["central_weight"]),
                    "GROUNDED_WEIGHT": str(param_set["grounded_weight"]),
                    "CONFIDENCE_WEIGHT": str(param_set["confidence_weight"]),
                }
            )

            # 実行
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

            if result.returncode == 0 and output_path.exists():
                return {"success": True, "stdout": result.stdout}
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Unknown extraction error",
                    "stdout": result.stdout,
                    "returncode": result.returncode,
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Extraction timeout (120s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_extraction_iou(self, filename: str, extracted_path: Path) -> float:
        """抽出結果のIoU計算"""
        try:
            if filename not in self.correct_data["annotations"]:
                return 0.0

            correct_rect = self.correct_data["annotations"][filename]["primary_rectangle"]
            if not correct_rect:
                return 0.0

            # 抽出画像から推定境界計算（前回と同じロジック）
            original_path = self.original_dir / filename
            estimated_bounds = self.get_extracted_image_bounds(extracted_path, original_path)

            if not estimated_bounds:
                return 0.0

            # IoU計算
            return self.calculate_iou(correct_rect, estimated_bounds)

        except Exception as e:
            logger.error(f"IoU計算エラー {filename}: {e}")
            return 0.0

    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """IoU（交差率）計算"""
        try:
            x1_1, y1_1 = box1["x"], box1["y"]
            x2_1, y2_1 = x1_1 + box1["width"], y1_1 + box1["height"]

            x1_2, y1_2 = box2["x"], box2["y"]
            x2_2, y2_2 = x1_2 + box2["width"], y1_2 + box2["height"]

            # 交差領域
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0

            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = box1["width"] * box1["height"]
            area2 = box2["width"] * box2["height"]
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            return 0.0

    def get_extracted_image_bounds(
        self, extracted_path: Path, original_path: Path
    ) -> Optional[Dict]:
        """抽出画像から元画像での推定境界を計算（前回と同じロジック）"""
        try:
            if not extracted_path.exists() or not original_path.exists():
                return None

            extracted = cv2.imread(str(extracted_path))
            original = cv2.imread(str(original_path))

            if extracted is None or original is None:
                return None

            orig_h, orig_w = original.shape[:2]
            ext_h, ext_w = extracted.shape[:2]

            aspect_ratio = ext_w / ext_h if ext_h > 0 else 1.0

            if aspect_ratio > 1.0:
                estimated_w = int(orig_w * 0.7)
                estimated_h = int(estimated_w / aspect_ratio)
            else:
                estimated_h = int(orig_h * 0.7)
                estimated_w = int(estimated_h * aspect_ratio)

            estimated_x = (orig_w - estimated_w) // 2
            estimated_y = (orig_h - estimated_h) // 2

            return {
                "x": max(0, estimated_x),
                "y": max(0, estimated_y),
                "width": min(estimated_w, orig_w),
                "height": min(estimated_h, orig_h),
            }

        except Exception as e:
            return None

    def get_quality_grade(self, iou_score: float) -> str:
        """IoUスコアから品質グレード判定"""
        if iou_score >= 0.7:
            return "A"
        elif iou_score >= 0.5:
            return "B"
        elif iou_score >= 0.3:
            return "C"
        elif iou_score >= 0.1:
            return "D"
        else:
            return "F"

    def optimize_parameters(self) -> Dict:
        """パラメータ最適化実行"""
        logger.info("🚀 SAM+YOLOパラメータ最適化開始")

        # テスト対象画像選択（代表的な画像を選ぶ）
        test_images = [
            "kana08_0001.jpg",  # 大きなキャラ
            "kana08_0002.jpg",  # 複数キャラ
            "kana08_0008.jpg",  # 複雑レイアウト
            "kana08_0010.jpg",  # 全身キャラ
            "kana08_0016.jpg",  # 横長レイアウト
        ]

        optimization_results = {
            "metadata": {
                "test_images": test_images,
                "parameter_sets_tested": len(self.parameter_sets),
            },
            "parameter_test_results": {},
            "best_parameter_set": None,
            "recommendations": [],
        }

        best_score = 0.0
        best_param_set = None

        # 各パラメータセットをテスト
        for param_set in self.parameter_sets:
            test_result = self.test_parameter_set(param_set, test_images)
            optimization_results["parameter_test_results"][param_set["name"]] = test_result

            # 最良パラメータセット判定（A+B成功率重視）
            score = test_result["summary"]["ab_success_rate"]
            if score > best_score:
                best_score = score
                best_param_set = param_set

        optimization_results["best_parameter_set"] = best_param_set

        # 推奨事項生成
        self.generate_recommendations(optimization_results)

        return optimization_results

    def generate_recommendations(self, results: Dict):
        """推奨事項生成"""
        recommendations = []

        # 最良パラメータの推奨
        if results["best_parameter_set"]:
            best = results["best_parameter_set"]
            recommendations.append(f"最適パラメータセット: {best['name']} - {best['description']}")

        # 個別パラメータの分析
        param_analysis = {}
        for name, result in results["parameter_test_results"].items():
            param_set = result["parameter_set"]
            summary = result["summary"]

            param_analysis[name] = {
                "ab_success_rate": summary["ab_success_rate"],
                "average_iou": summary["average_iou"],
                "yolo_conf": param_set["yolo_conf_threshold"],
                "area_weight": param_set["area_weight"],
            }

        # 傾向分析
        high_conf_results = [v for k, v in param_analysis.items() if v["yolo_conf"] >= 0.07]
        low_conf_results = [v for k, v in param_analysis.items() if v["yolo_conf"] < 0.07]

        if high_conf_results and low_conf_results:
            high_avg = sum(r["ab_success_rate"] for r in high_conf_results) / len(high_conf_results)
            low_avg = sum(r["ab_success_rate"] for r in low_conf_results) / len(low_conf_results)

            if high_avg > low_avg:
                recommendations.append("YOLO信頼度閾値を高めに設定することを推奨")
            else:
                recommendations.append("YOLO信頼度閾値を低めに設定することを推奨")

        results["recommendations"] = recommendations

    def save_optimization_results(self, results: Dict) -> Path:
        """最適化結果保存"""
        output_file = self.output_dir / "parameter_optimization_results.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"📄 最適化結果保存: {output_file}")
        return output_file


import os  # 必要なimport


def main():
    """メイン実行"""
    # パス設定
    correct_annotations_path = Path(
        "C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis/correct_annotations.json"
    )
    original_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/optimization")

    # 最適化器初期化
    optimizer = SAMYOLOParameterOptimizer(correct_annotations_path, original_dir, output_dir)

    # パラメータ最適化実行
    results = optimizer.optimize_parameters()

    # 結果保存
    output_file = optimizer.save_optimization_results(results)

    # サマリー表示
    logger.info("=" * 60)
    logger.info("🎯 SAM+YOLOパラメータ最適化完了")

    if results["best_parameter_set"]:
        best = results["best_parameter_set"]
        best_result = results["parameter_test_results"][best["name"]]
        logger.info(f"🏆 最適パラメータ: {best['name']}")
        logger.info(f"📊 A+B成功率: {best_result['summary']['ab_success_rate']:.1%}")
        logger.info(f"📈 平均IoU: {best_result['summary']['average_iou']:.3f}")

    logger.info(f"📄 詳細結果: {output_file}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
