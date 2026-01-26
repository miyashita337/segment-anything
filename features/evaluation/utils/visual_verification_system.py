#!/usr/bin/env python3
"""
視覚的検証システム
人間ラベルとAI抽出結果を視覚的に比較し、真の成功率を算出

P1-A001: deprecatedから復旧された改善機能
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
class VerificationResult:
    """検証結果"""

    image_id: str
    human_bbox: Tuple[int, int, int, int]
    ai_bbox: Tuple[int, int, int, int]
    reported_iou: float
    visual_match: bool  # 視覚的に正しいキャラクターを抽出しているか
    actual_character_extracted: str  # 実際に抽出されたキャラクターの説明
    expected_character: str  # 期待されたキャラクターの説明
    issue_type: Optional[str]  # 問題の種類


class VisualVerificationSystem:
    """視覚的検証システム"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.workspace = project_root / "workspace" / "P1-A001"
        self.workspace.mkdir(parents=True, exist_ok=True)

        # レガシーパス（deprecated対応）
        self.legacy_results_file = (
            project_root
            / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        )
        self.legacy_labels_file = project_root / "segment-anything/extracted_labels.json"

        # 新しいワークスペースパス
        self.results_file = self.workspace / "visual_verification_results.json"
        self.labels_file = self.workspace / "extracted_labels.json"
        self.output_dir = self.workspace / "visual_verification_results"
        self.output_dir.mkdir(exist_ok=True)

        # データ読み込み
        self.load_data()

    def load_data(self):
        """データ読み込み"""
        # AI抽出結果
        if self.legacy_results_file.exists():
            with open(self.legacy_results_file, "r", encoding="utf-8") as f:
                self.ai_results = json.load(f)
            logger.info(f"AI結果読み込み: {len(self.ai_results)}件")
        else:
            self.ai_results = {}
            logger.warning("AI結果ファイルが見つかりません")

        # 人間ラベル
        labels_path = (
            self.legacy_labels_file if self.legacy_labels_file.exists() else self.labels_file
        )
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.human_labels = json.load(f)
            logger.info(f"人間ラベル読み込み: {len(self.human_labels)}件")
        else:
            self.human_labels = {}
            logger.warning("人間ラベルファイルが見つかりません")

    def create_verification_report(self) -> Dict:
        """視覚的検証レポート作成"""
        logger.info("視覚的検証開始")

        verification_results = []
        total_processed = 0
        visual_matches = 0

        # 各画像について検証
        for image_id, ai_result in self.ai_results.items():
            if image_id not in self.human_labels:
                continue

            human_label = self.human_labels[image_id]

            # 検証実行
            verification = self._verify_visual_match(image_id, ai_result, human_label)
            verification_results.append(verification)

            total_processed += 1
            if verification.visual_match:
                visual_matches += 1

        # 統計計算
        visual_match_rate = (visual_matches / total_processed * 100) if total_processed > 0 else 0

        # 問題分析
        issue_breakdown = self._analyze_verification_issues(verification_results)

        report = {
            "total_processed": total_processed,
            "visual_matches": visual_matches,
            "visual_match_rate": visual_match_rate,
            "verification_results": [self._verification_to_dict(r) for r in verification_results],
            "issue_breakdown": issue_breakdown,
            "generated_at": self._get_timestamp(),
        }

        # レポート保存
        output_file = self.output_dir / f"visual_verification_report_{self._get_timestamp()}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 視覚化
        self._create_verification_visualization(verification_results)

        logger.info(f"視覚的一致率: {visual_match_rate:.1f}% ({visual_matches}/{total_processed})")

        return report

    def _verify_visual_match(
        self, image_id: str, ai_result: Dict, human_label: Dict
    ) -> VerificationResult:
        """視覚的一致度検証"""
        # 境界ボックス抽出
        ai_bbox = self._extract_bbox(ai_result.get("bbox", [0, 0, 0, 0]))
        human_bbox = self._extract_bbox(human_label.get("bbox", [0, 0, 0, 0]))

        # 視覚的一致度判定
        visual_match = self._judge_visual_match(ai_result, human_label)

        # キャラクター説明抽出
        actual_character = ai_result.get("character_description", "不明")
        expected_character = human_label.get("character_description", "不明")

        # 問題分類
        issue_type = self._classify_visual_issue(visual_match, ai_result, human_label)

        return VerificationResult(
            image_id=image_id,
            human_bbox=human_bbox,
            ai_bbox=ai_bbox,
            reported_iou=ai_result.get("iou", 0.0),
            visual_match=visual_match,
            actual_character_extracted=actual_character,
            expected_character=expected_character,
            issue_type=issue_type,
        )

    def _extract_bbox(self, bbox_data) -> Tuple[int, int, int, int]:
        """境界ボックス抽出"""
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            return tuple(map(int, bbox_data))
        return (0, 0, 0, 0)

    def _judge_visual_match(self, ai_result: Dict, human_label: Dict) -> bool:
        """視覚的一致度判定"""
        # 複数の指標を組み合わせて判定

        # 1. 品質スコア
        quality_score = ai_result.get("quality_score", 0.0)
        quality_threshold = 0.75

        # 2. 信頼度スコア
        confidence = ai_result.get("confidence", 0.0)
        confidence_threshold = 0.7

        # 3. IoU（境界ボックス一致度）
        iou = ai_result.get("iou", 0.0)
        iou_threshold = 0.6

        # 4. サイズ比較
        ai_bbox = self._extract_bbox(ai_result.get("bbox", [0, 0, 0, 0]))
        human_bbox = self._extract_bbox(human_label.get("bbox", [0, 0, 0, 0]))
        size_ratio = self._calculate_size_ratio(ai_bbox, human_bbox)
        size_threshold = 0.5  # 50%以上のサイズ一致

        # 総合判定
        quality_ok = quality_score >= quality_threshold
        confidence_ok = confidence >= confidence_threshold
        iou_ok = iou >= iou_threshold
        size_ok = size_ratio >= size_threshold

        # 3つ以上の条件を満たせば視覚的一致とみなす
        criteria_met = sum([quality_ok, confidence_ok, iou_ok, size_ok])
        return criteria_met >= 3

    def _calculate_size_ratio(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """サイズ比率計算"""
        if bbox1 == (0, 0, 0, 0) or bbox2 == (0, 0, 0, 0):
            return 0.0

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area1 == 0 or area2 == 0:
            return 0.0

        return min(area1, area2) / max(area1, area2)

    def _classify_visual_issue(
        self, visual_match: bool, ai_result: Dict, human_label: Dict
    ) -> Optional[str]:
        """視覚問題分類"""
        if visual_match:
            return None

        quality_score = ai_result.get("quality_score", 0.0)
        confidence = ai_result.get("confidence", 0.0)
        iou = ai_result.get("iou", 0.0)

        if quality_score < 0.5:
            return "low_quality"
        elif confidence < 0.5:
            return "low_confidence"
        elif iou < 0.3:
            return "poor_localization"
        else:
            return "content_mismatch"

    def _analyze_verification_issues(self, results: List[VerificationResult]) -> Dict:
        """検証問題分析"""
        issue_counts = {}
        for result in results:
            if result.issue_type:
                issue_counts[result.issue_type] = issue_counts.get(result.issue_type, 0) + 1

        total_issues = sum(issue_counts.values())
        issue_rates = {}
        for issue, count in issue_counts.items():
            issue_rates[issue] = {
                "count": count,
                "rate": (count / total_issues * 100) if total_issues > 0 else 0,
            }

        return issue_rates

    def _create_verification_visualization(self, results: List[VerificationResult]):
        """検証結果可視化"""
        logger.info("検証結果可視化作成中...")

        # 問題別の分布図
        issue_types = [r.issue_type for r in results if r.issue_type]
        if issue_types:
            plt.figure(figsize=(10, 6))
            unique_issues, counts = np.unique(issue_types, return_counts=True)
            plt.bar(unique_issues, counts)
            plt.title("視覚的検証 - 問題分類")
            plt.xlabel("問題タイプ")
            plt.ylabel("件数")
            plt.xticks(rotation=45)
            plt.tight_layout()

            output_path = self.output_dir / f"issue_distribution_{self._get_timestamp()}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        # 一致率の可視化
        match_data = [r.visual_match for r in results]
        if match_data:
            plt.figure(figsize=(8, 6))
            match_counts = [sum(match_data), len(match_data) - sum(match_data)]
            labels = ["視覚的一致", "視覚的不一致"]
            colors = ["#2ecc71", "#e74c3c"]

            plt.pie(match_counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            plt.title("視覚的検証結果")

            output_path = self.output_dir / f"match_rate_pie_{self._get_timestamp()}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        logger.info(f"可視化完了: {self.output_dir}")

    def _verification_to_dict(self, result: VerificationResult) -> Dict:
        """検証結果を辞書に変換"""
        return {
            "image_id": result.image_id,
            "human_bbox": result.human_bbox,
            "ai_bbox": result.ai_bbox,
            "reported_iou": result.reported_iou,
            "visual_match": result.visual_match,
            "actual_character_extracted": result.actual_character_extracted,
            "expected_character": result.expected_character,
            "issue_type": result.issue_type,
        }

    def _get_timestamp(self) -> str:
        """タイムスタンプ取得"""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """メイン実行"""
    project_root = Path(__file__).parent.parent.parent.parent
    verifier = VisualVerificationSystem(project_root)
    report = verifier.create_verification_report()

    print(f"視覚的検証完了")
    print(f"処理対象: {report['total_processed']}件")
    print(f"視覚的一致率: {report['visual_match_rate']:.1f}%")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
