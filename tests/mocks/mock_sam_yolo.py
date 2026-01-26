#!/usr/bin/env python3
"""
SAM+YOLO抽出モックシステム

実際のSAM+YOLO抽出処理をモックで再現
テスト用に予測可能な結果を返す
"""

import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MockDetectionResult:
    """YOLO検出結果のモック"""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str = "character"


@dataclass
class MockSegmentationResult:
    """SAMセグメンテーション結果のモック"""

    mask_area: int
    quality_score: float
    grade: str
    success: bool


class MockSamYoloExtractor:
    """SAM+YOLO抽出のモッククラス"""

    # 品質評価グレード分布
    GRADE_DISTRIBUTION = {
        "A": 0.25,  # 25% - 高品質
        "B": 0.35,  # 35% - 良品質
        "C": 0.25,  # 25% - 中品質
        "D": 0.10,  # 10% - 低品質
        "F": 0.05,  # 5% - 失敗
    }

    # 品質評価手法別成功率
    METHOD_SUCCESS_RATES = {
        "balanced": 0.85,
        "confidence_priority": 0.78,
        "size_priority": 0.82,
        "fullbody_priority": 0.75,
        "central_priority": 0.80,
    }

    def __init__(self, quality_method: str = "balanced", random_seed: Optional[int] = None):
        """
        モック抽出器初期化

        Args:
            quality_method: 品質評価手法
            random_seed: ランダムシード（再現可能テスト用）
        """
        self.quality_method = quality_method
        if random_seed is not None:
            random.seed(random_seed)

    def extract_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        単体画像抽出のモック

        Args:
            image_path: 画像パス

        Returns:
            抽出結果辞書
        """
        # 処理時間シミュレート（0.5-2.0秒）
        processing_time = random.uniform(0.5, 2.0)
        time.sleep(0.1)  # テスト実行時間短縮のため短縮

        # 成功率判定
        success_rate = self.METHOD_SUCCESS_RATES.get(self.quality_method, 0.80)
        success = random.random() < success_rate

        if not success:
            return {
                "image_path": image_path,
                "success": False,
                "error": "抽出失敗：キャラクター検出不能",
                "processing_time": processing_time,
                "detection_result": None,
                "segmentation_result": None,
            }

        # YOLO検出結果生成
        detection = self._generate_mock_detection()

        # SAMセグメンテーション結果生成
        segmentation = self._generate_mock_segmentation()

        return {
            "image_path": image_path,
            "success": True,
            "processing_time": processing_time,
            "detection_result": detection,
            "segmentation_result": segmentation,
            "quality_score": segmentation.quality_score,
            "grade": segmentation.grade,
            "output_path": f"{image_path.replace('.jpg', '_extracted.jpg').replace('.png', '_extracted.jpg')}",
        }

    def extract_batch(
        self, input_dir: str, output_dir: str, max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        バッチ抽出のモック

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            max_files: 最大処理ファイル数

        Returns:
            バッチ処理結果
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 入力ディレクトリが存在しない場合はエラー
        if not input_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

        # 出力ディレクトリ作成
        output_path.mkdir(parents=True, exist_ok=True)

        # 模擬画像ファイル生成（実際にはglob検索）
        image_files = []
        mock_file_count = random.randint(5, 20) if max_files is None else min(max_files, 20)

        for i in range(mock_file_count):
            # 実在する画像ファイル名パターン
            filename = f"test_image_{i+1:03d}.jpg"
            image_files.append(str(input_path / filename))

        # バッチ処理実行
        batch_results = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "total_images": len(image_files),
            "processed_images": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "extraction_results": [],
            "quality_distribution": {grade: 0 for grade in self.GRADE_DISTRIBUTION.keys()},
            "average_quality_score": 0.0,
            "total_processing_time": 0.0,
        }

        total_quality = 0.0

        for image_path in image_files:
            result = self.extract_single_image(image_path)
            batch_results["extraction_results"].append(result)
            batch_results["processed_images"] += 1
            batch_results["total_processing_time"] += result["processing_time"]

            if result["success"]:
                batch_results["successful_extractions"] += 1
                grade = result["grade"]
                batch_results["quality_distribution"][grade] += 1
                total_quality += result["quality_score"]
            else:
                batch_results["failed_extractions"] += 1

        # 平均品質スコア計算
        if batch_results["successful_extractions"] > 0:
            batch_results["average_quality_score"] = (
                total_quality / batch_results["successful_extractions"]
            )

        # 成功率計算
        batch_results["success_rate"] = (
            batch_results["successful_extractions"] / batch_results["total_images"] * 100
        )

        # extraction_result.json保存
        self._save_extraction_result(batch_results, output_path)

        return batch_results

    def _generate_mock_detection(self) -> MockDetectionResult:
        """YOLO検出結果のモック生成"""
        # 境界ボックス生成（画像サイズ1024x1024想定）
        x1 = random.uniform(100, 400)
        y1 = random.uniform(100, 400)
        x2 = x1 + random.uniform(200, 400)
        y2 = y1 + random.uniform(300, 500)

        # 信頼度生成
        confidence = random.uniform(0.5, 0.95)

        return MockDetectionResult(
            bbox=(x1, y1, x2, y2), confidence=confidence, class_id=0, class_name="character"
        )

    def _generate_mock_segmentation(self) -> MockSegmentationResult:
        """SAMセグメンテーション結果のモック生成"""
        # グレード決定
        rand_val = random.random()
        cumulative_prob = 0.0
        grade = "F"

        for g, prob in self.GRADE_DISTRIBUTION.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                grade = g
                break

        # グレード別品質スコア生成
        quality_ranges = {
            "A": (0.85, 1.0),
            "B": (0.70, 0.85),
            "C": (0.55, 0.70),
            "D": (0.40, 0.55),
            "F": (0.0, 0.40),
        }

        min_score, max_score = quality_ranges[grade]
        quality_score = random.uniform(min_score, max_score)

        # マスク面積生成
        mask_area = random.randint(50000, 300000)

        return MockSegmentationResult(
            mask_area=mask_area, quality_score=quality_score, grade=grade, success=grade != "F"
        )

    def _save_extraction_result(self, batch_results: Dict[str, Any], output_path: Path) -> None:
        """extraction_result.json保存"""
        # 簡略化された結果データ
        simplified_results = []
        for result in batch_results["extraction_results"]:
            if result["success"]:
                simplified_results.append(
                    {
                        "image_name": Path(result["image_path"]).name,
                        "success": True,
                        "quality_score": result["quality_score"],
                        "grade": result["grade"],
                        "processing_time": result["processing_time"],
                    }
                )
            else:
                simplified_results.append(
                    {
                        "image_name": Path(result["image_path"]).name,
                        "success": False,
                        "error": result["error"],
                        "processing_time": result["processing_time"],
                    }
                )

        extraction_data = {
            "tracker_id": "MOCK-TEST",
            "total_images": batch_results["total_images"],
            "successful_extractions": batch_results["successful_extractions"],
            "success_rate": batch_results["success_rate"],
            "average_quality_score": batch_results["average_quality_score"],
            "quality_distribution": batch_results["quality_distribution"],
            "processing_time": batch_results["total_processing_time"],
            "extraction_results": simplified_results,
        }

        # JSON保存
        json_path = output_path / "extraction_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=2)


class MockQualityEvaluator:
    """品質評価システムのモック"""

    @staticmethod
    def evaluate_extraction_quality(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        抽出品質評価のモック

        Args:
            extraction_result: 抽出結果

        Returns:
            品質評価結果
        """
        if not extraction_result.get("success", False):
            return {
                "quality_score": 0.0,
                "grade": "F",
                "evaluation_details": {
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "clarity": 0.0,
                    "composition": 0.0,
                },
            }

        # 既存の品質スコアを使用
        quality_score = extraction_result.get("quality_score", 0.5)
        grade = extraction_result.get("grade", "C")

        # 詳細評価生成
        base_score = quality_score
        evaluation_details = {
            "completeness": min(1.0, base_score + random.uniform(-0.1, 0.1)),
            "accuracy": min(1.0, base_score + random.uniform(-0.1, 0.1)),
            "clarity": min(1.0, base_score + random.uniform(-0.1, 0.1)),
            "composition": min(1.0, base_score + random.uniform(-0.1, 0.1)),
        }

        return {
            "quality_score": quality_score,
            "grade": grade,
            "evaluation_details": evaluation_details,
        }

    @staticmethod
    def generate_quality_report(batch_results: Dict[str, Any]) -> str:
        """
        品質レポート生成のモック

        Args:
            batch_results: バッチ処理結果

        Returns:
            品質レポート（Markdown形式）
        """
        total = batch_results["total_images"]
        successful = batch_results["successful_extractions"]
        success_rate = batch_results["success_rate"]
        avg_quality = batch_results["average_quality_score"]

        grade_dist = batch_results["quality_distribution"]

        report = f"""# 品質評価レポート

## 処理概要
- **総画像数**: {total}枚
- **成功数**: {successful}枚
- **成功率**: {success_rate:.1f}%
- **平均品質スコア**: {avg_quality:.3f}

## 品質分布
- **A評価**: {grade_dist.get('A', 0)}枚 ({grade_dist.get('A', 0)/total*100:.1f}%)
- **B評価**: {grade_dist.get('B', 0)}枚 ({grade_dist.get('B', 0)/total*100:.1f}%)
- **C評価**: {grade_dist.get('C', 0)}枚 ({grade_dist.get('C', 0)/total*100:.1f}%)
- **D評価**: {grade_dist.get('D', 0)}枚 ({grade_dist.get('D', 0)/total*100:.1f}%)
- **F評価**: {grade_dist.get('F', 0)}枚 ({grade_dist.get('F', 0)/total*100:.1f}%)

## 評価基準
- **A (0.85-1.0)**: 高品質抽出完了
- **B (0.70-0.85)**: 良品質抽出完了
- **C (0.55-0.70)**: 中品質抽出完了
- **D (0.40-0.55)**: 低品質抽出完了
- **F (0.00-0.40)**: 抽出失敗
"""
        return report
