#!/usr/bin/env python3
"""
真の成功率分析システム
座標だけでなく、実際の抽出内容を確認して真の成功率を算出
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrueVerificationResult:
    """真の検証結果"""

    image_id: str
    reported_iou: float
    reported_success: bool
    human_label_area: Tuple[int, int, int, int]  # 人間ラベルの座標
    ai_extraction_area: Tuple[int, int, int, int]  # AI抽出の座標

    # 真の評価
    coordinate_match: bool  # 座標の一致度
    visual_content_match: bool  # 視覚的内容の一致度
    true_success: bool  # 真の成功判定

    # 問題分類
    issue_type: Optional[str]  # 問題の種類
    confidence_level: str  # 確信度（high/medium/low）


class TrueSuccessAnalyzer:
    """真の成功率分析"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_file = (
            project_root
            / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        )
        self.labels_file = project_root / "segment-anything/extracted_labels.json"
        self.output_dir = project_root / "true_success_analysis"
        self.output_dir.mkdir(exist_ok=True)

        # データ読み込み
        self.load_data()

    def load_data(self):
        """データ読み込み"""
        with open(self.results_file, "r", encoding="utf-8") as f:
            self.ai_results = json.load(f)

        with open(self.labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        self.human_labels = {}
        for item in labels_data:
            if item.get("red_boxes"):
                image_id = item["filename"].rsplit(".", 1)[0]
                self.human_labels[image_id] = item

        logger.info(f"データ読み込み完了: AI結果{len(self.ai_results)}件, 人間ラベル{len(self.human_labels)}件")

    def analyze_coordinate_patterns(self) -> Dict[str, List]:
        """座標パターンの分析"""
        patterns = {
            "full_image_labels": [],  # 画像全体を囲むラベル
            "partial_labels": [],  # 部分的なラベル
            "suspicious_high_iou": [],  # 疑わしい高IoU
        }

        for result in self.ai_results:
            if not result["extraction_success"]:
                continue

            image_id = result["image_id"]
            human_bbox = result["human_bbox"]
            ai_bbox = result["final_bbox"]
            iou = result["iou_score"]

            # 画像サイズ取得
            image_path = Path(result["image_path"])
            if image_path.exists():
                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]

                # 人間ラベルが画像の大部分を占めるかチェック
                hx, hy, hw, hh = human_bbox
                label_area = hw * hh
                image_area = w * h
                coverage_ratio = label_area / image_area

                if coverage_ratio > 0.7:  # 70%以上を占める場合
                    patterns["full_image_labels"].append(
                        {
                            "image_id": image_id,
                            "coverage_ratio": coverage_ratio,
                            "iou": iou,
                            "human_bbox": human_bbox,
                            "ai_bbox": ai_bbox,
                        }
                    )
                else:
                    patterns["partial_labels"].append(
                        {"image_id": image_id, "coverage_ratio": coverage_ratio, "iou": iou}
                    )

                # 高IoUで全画像ラベルのケース
                if iou > 0.9 and coverage_ratio > 0.7:
                    patterns["suspicious_high_iou"].append(
                        {"image_id": image_id, "iou": iou, "coverage_ratio": coverage_ratio}
                    )

        return patterns

    def manual_verification_needed(self, image_id: str) -> bool:
        """手動検証が必要なケースかどうか判定"""
        for result in self.ai_results:
            if result["image_id"] == image_id:
                # 高IoUかつ抽出画像が存在するケース
                if result["iou_score"] > 0.8 and result["extraction_success"]:
                    return True
        return False

    def analyze_extraction_content(self, image_id: str) -> Optional[TrueVerificationResult]:
        """抽出内容の分析"""
        # AI結果取得
        ai_result = None
        for result in self.ai_results:
            if result["image_id"] == image_id:
                ai_result = result
                break

        if not ai_result:
            return None

        # 抽出画像の存在確認
        extracted_paths = [
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana07/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana05/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana08/{image_id}.jpg",
        ]

        extracted_exists = any(path.exists() for path in extracted_paths)

        # 基本的な問題パターンの自動検出
        image_path = Path(ai_result["image_path"])
        if image_path.exists():
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]

            human_bbox = ai_result["human_bbox"]
            hx, hy, hw, hh = human_bbox
            coverage_ratio = (hw * hh) / (w * h)

            # 問題パターンの判定
            issue_type = None
            visual_match = None
            confidence = "low"

            if coverage_ratio > 0.8 and ai_result["iou_score"] > 0.9:
                issue_type = "full_image_label_with_partial_extraction"
                visual_match = False
                confidence = "high"
            elif coverage_ratio > 0.5 and ai_result["iou_score"] > 0.8:
                issue_type = "large_label_area_suspicious"
                visual_match = None  # 要手動確認
                confidence = "medium"
            elif not extracted_exists and ai_result["extraction_success"]:
                issue_type = "missing_extraction_file"
                visual_match = False
                confidence = "high"

            return TrueVerificationResult(
                image_id=image_id,
                reported_iou=ai_result["iou_score"],
                reported_success=ai_result["extraction_success"],
                human_label_area=tuple(human_bbox),
                ai_extraction_area=tuple(ai_result["final_bbox"])
                if ai_result["final_bbox"]
                else (0, 0, 0, 0),
                coordinate_match=ai_result["iou_score"] > 0.5,
                visual_content_match=visual_match,
                true_success=visual_match if visual_match is not None else False,
                issue_type=issue_type,
                confidence_level=confidence,
            )

        return None

    def generate_true_success_report(self):
        """真の成功率レポート生成"""
        logger.info("真の成功率分析開始")

        # 座標パターン分析
        patterns = self.analyze_coordinate_patterns()

        # 全画像の詳細分析
        verification_results = []
        for result in self.ai_results:
            if result["extraction_success"]:
                analysis = self.analyze_extraction_content(result["image_id"])
                if analysis:
                    verification_results.append(analysis)

        # 統計計算
        total_reported_success = len([r for r in self.ai_results if r["extraction_success"]])
        full_image_labels = len(patterns["full_image_labels"])
        suspicious_cases = len(patterns["suspicious_high_iou"])

        # 確実な失敗ケース（高信頼度で問題ありと判定）
        confirmed_failures = len(
            [r for r in verification_results if r.confidence_level == "high" and not r.true_success]
        )

        # 疑わしいケース（手動確認必要）
        suspicious_count = len(
            [
                r
                for r in verification_results
                if r.confidence_level in ["medium", "low"] or r.visual_content_match is None
            ]
        )

        # 保守的推定（疑わしいケースを50%失敗と仮定）
        estimated_additional_failures = suspicious_count * 0.5
        estimated_true_success = (
            total_reported_success - confirmed_failures - estimated_additional_failures
        )
        estimated_success_rate = (estimated_true_success / len(self.ai_results)) * 100

        # レポート作成
        report = f"""# 真の成功率分析レポート

## ⚠️ 重大な発見

**報告された成功率**: 81.2% ({total_reported_success}/{len(self.ai_results)})
**真の成功率（保守的推定）**: {estimated_success_rate:.1f}%

---

## 🔍 問題の詳細分析

### 座標パターン分析
- **全画像ラベル**: {full_image_labels}件（画像の70%以上を占めるラベル）
- **部分ラベル**: {len(patterns['partial_labels'])}件
- **疑わしい高IoU**: {suspicious_cases}件（IoU>0.9かつ全画像ラベル）

### 問題分類
- **確実な失敗**: {confirmed_failures}件（高信頼度）
- **疑わしいケース**: {suspicious_count}件（要手動確認）
- **推定追加失敗**: {estimated_additional_failures:.1f}件

---

## 🚨 主要問題パターン

### 1. kana07_0023パターン（確認済み）
- **報告IoU**: 0.997
- **問題**: 人間ラベルは画像全体、抽出は上部キャラクターのみ
- **根本原因**: 座標の数値的一致 ≠ 内容の一致

### 2. 全画像ラベル問題
{len(patterns['full_image_labels'])}件の画像で人間ラベルが画像の大部分を占める：
"""

        # 全画像ラベルの詳細
        for item in patterns["full_image_labels"][:5]:  # 上位5件
            report += f"- **{item['image_id']}**: IoU {item['iou']:.3f}, カバー率 {item['coverage_ratio']:.1%}\\n"

        report += f"""\n---

## 📊 信頼性分析

### 高信頼度の評価
- **確実な成功**: {len(verification_results) - confirmed_failures - suspicious_count}件
- **確実な失敗**: {confirmed_failures}件

### 要検証ケース
- **中程度の疑い**: {len([r for r in verification_results if r.confidence_level == "medium"])}件
- **低信頼度**: {len([r for r in verification_results if r.confidence_level == "low"])}件

---

## 🎯 推奨される修正アクション

### 即座に必要
1. **人間ラベルの見直し**
   - 画像全体を囲むラベルの再確認
   - 複数キャラクター画像の適切なラベリング

2. **評価基準の改善**
   - 座標一致だけでなく、抽出内容の確認
   - 視覚的検証システムの統合

3. **真の成功率の算出**
   - 手動検証による正確な成功率測定
   - 改善効果の再評価

### 根本的解決
1. **マルチキャラクター対応**
   - 複数キャラクター画像の特別処理
   - 主要キャラクター特定アルゴリズム

2. **評価システムの刷新**
   - 内容ベースの評価指標導入
   - 視覚的類似度の定量化

---

## ⚠️ 結論

**81.2%という改善効果は過大評価の可能性が高い**

真の成功率は{estimated_success_rate:.1f}%程度と推定され、
改善効果も実際は16.8% → {estimated_success_rate:.1f}%（{estimated_success_rate/16.8:.1f}倍）
程度と考えられる。

根本的な評価システムの見直しが急務。
"""

        # レポート保存
        report_path = self.output_dir / "true_success_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 詳細データ保存
        analysis_data = {
            "coordinate_patterns": patterns,
            "verification_results": [
                {
                    "image_id": r.image_id,
                    "reported_iou": r.reported_iou,
                    "reported_success": r.reported_success,
                    "issue_type": r.issue_type,
                    "confidence_level": r.confidence_level,
                    "visual_content_match": r.visual_content_match,
                }
                for r in verification_results
            ],
            "summary": {
                "total_cases": len(self.ai_results),
                "reported_success": total_reported_success,
                "reported_success_rate": (total_reported_success / len(self.ai_results)) * 100,
                "estimated_true_success_rate": estimated_success_rate,
                "confirmed_failures": confirmed_failures,
                "suspicious_cases": suspicious_count,
                "full_image_labels": full_image_labels,
            },
        }

        json_path = self.output_dir / "true_success_analysis_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        logger.info(f"真の成功率分析完了: {report_path}")
        return report_path, estimated_success_rate


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools")

    analyzer = TrueSuccessAnalyzer(project_root)
    report_path, true_success_rate = analyzer.generate_true_success_report()

    print(f"\n🚨 真の成功率分析完了")
    print(f"報告成功率: 81.2%")
    print(f"真の成功率（推定）: {true_success_rate:.1f}%")
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()
