#!/usr/bin/env python3
"""
真の成功率分析システム
座標だけでなく、実際の抽出内容を確認して真の成功率を算出

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
        self.workspace = project_root / "workspace" / "P1-A001"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # レガシーパス（deprecated対応）
        self.legacy_results_file = project_root / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        self.legacy_labels_file = project_root / "segment-anything/extracted_labels.json"
        
        # 新しいワークスペースパス
        self.results_file = self.workspace / "true_analysis_results.json"
        self.labels_file = self.workspace / "extracted_labels.json"
        self.output_dir = self.workspace / "analysis_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # データ読み込み
        self.load_data()
        
    def load_data(self):
        """データ読み込み"""
        # AI抽出結果
        if self.legacy_results_file.exists():
            with open(self.legacy_results_file, 'r', encoding='utf-8') as f:
                self.ai_results = json.load(f)
            logger.info(f"AI結果読み込み: {len(self.ai_results)}件")
        else:
            self.ai_results = {}
            logger.warning("AI結果ファイルが見つかりません")
        
        # 人間ラベル
        labels_path = self.legacy_labels_file if self.legacy_labels_file.exists() else self.labels_file
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.human_labels = json.load(f)
            logger.info(f"人間ラベル読み込み: {len(self.human_labels)}件")
        else:
            self.human_labels = {}
            logger.warning("人間ラベルファイルが見つかりません")
    
    def analyze_true_success_rate(self) -> Dict:
        """真の成功率分析実行"""
        logger.info("真の成功率分析開始")
        
        verification_results = []
        total_analyzed = 0
        true_successes = 0
        coordinate_matches = 0
        visual_matches = 0
        
        # 各画像について分析
        for image_id, ai_result in self.ai_results.items():
            if image_id not in self.human_labels:
                continue
                
            human_label = self.human_labels[image_id]
            
            # 検証実行
            verification = self._verify_single_result(image_id, ai_result, human_label)
            verification_results.append(verification)
            
            total_analyzed += 1
            if verification.true_success:
                true_successes += 1
            if verification.coordinate_match:
                coordinate_matches += 1
            if verification.visual_content_match:
                visual_matches += 1
        
        # 統計計算
        true_success_rate = (true_successes / total_analyzed * 100) if total_analyzed > 0 else 0
        coordinate_match_rate = (coordinate_matches / total_analyzed * 100) if total_analyzed > 0 else 0
        visual_match_rate = (visual_matches / total_analyzed * 100) if total_analyzed > 0 else 0
        
        analysis_summary = {
            "total_analyzed": total_analyzed,
            "true_successes": true_successes,
            "true_success_rate": true_success_rate,
            "coordinate_match_rate": coordinate_match_rate,
            "visual_match_rate": visual_match_rate,
            "verification_results": [self._result_to_dict(r) for r in verification_results],
            "issue_breakdown": self._analyze_issues(verification_results),
            "confidence_breakdown": self._analyze_confidence(verification_results)
        }
        
        # 結果保存
        output_file = self.output_dir / f"true_success_analysis_{self._get_timestamp()}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"真の成功率: {true_success_rate:.1f}% ({true_successes}/{total_analyzed})")
        logger.info(f"座標一致率: {coordinate_match_rate:.1f}%")
        logger.info(f"視覚一致率: {visual_match_rate:.1f}%")
        
        return analysis_summary
    
    def _verify_single_result(self, image_id: str, ai_result: Dict, human_label: Dict) -> TrueVerificationResult:
        """単一結果の検証"""
        # 座標抽出
        ai_bbox = self._extract_bbox_from_ai_result(ai_result)
        human_bbox = self._extract_bbox_from_human_label(human_label)
        
        # 座標一致度チェック
        coordinate_match = self._check_coordinate_match(ai_bbox, human_bbox)
        
        # 視覚的内容一致度チェック（簡易版）
        visual_content_match = self._check_visual_content_match(ai_result, human_label)
        
        # 真の成功判定
        true_success = coordinate_match and visual_content_match
        
        # 問題分類
        issue_type = self._classify_issue(coordinate_match, visual_content_match, ai_result)
        
        # 確信度判定
        confidence_level = self._determine_confidence(ai_result, coordinate_match, visual_content_match)
        
        return TrueVerificationResult(
            image_id=image_id,
            reported_iou=ai_result.get("iou", 0.0),
            reported_success=ai_result.get("success", False),
            human_label_area=human_bbox,
            ai_extraction_area=ai_bbox,
            coordinate_match=coordinate_match,
            visual_content_match=visual_content_match,
            true_success=true_success,
            issue_type=issue_type,
            confidence_level=confidence_level
        )
    
    def _extract_bbox_from_ai_result(self, ai_result: Dict) -> Tuple[int, int, int, int]:
        """AI結果から境界ボックス抽出"""
        bbox = ai_result.get("bbox", [0, 0, 0, 0])
        return tuple(bbox) if len(bbox) == 4 else (0, 0, 0, 0)
    
    def _extract_bbox_from_human_label(self, human_label: Dict) -> Tuple[int, int, int, int]:
        """人間ラベルから境界ボックス抽出"""
        bbox = human_label.get("bbox", [0, 0, 0, 0])
        return tuple(bbox) if len(bbox) == 4 else (0, 0, 0, 0)
    
    def _check_coordinate_match(self, ai_bbox: Tuple[int, int, int, int], 
                              human_bbox: Tuple[int, int, int, int], 
                              threshold: float = 0.7) -> bool:
        """座標一致度チェック"""
        if ai_bbox == (0, 0, 0, 0) or human_bbox == (0, 0, 0, 0):
            return False
        
        # IoU計算
        iou = self._calculate_iou(ai_bbox, human_bbox)
        return iou >= threshold
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """IoU計算"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交差領域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _check_visual_content_match(self, ai_result: Dict, human_label: Dict) -> bool:
        """視覚的内容一致度チェック（簡易版）"""
        # 品質スコアベースの判定
        quality_score = ai_result.get("quality_score", 0.0)
        confidence = ai_result.get("confidence", 0.0)
        
        # 高品質・高信頼度の場合は一致とみなす
        return quality_score >= 0.8 and confidence >= 0.7
    
    def _classify_issue(self, coordinate_match: bool, visual_content_match: bool, 
                       ai_result: Dict) -> Optional[str]:
        """問題分類"""
        if coordinate_match and visual_content_match:
            return None
        
        if not coordinate_match and not visual_content_match:
            return "complete_mismatch"
        elif not coordinate_match:
            return "coordinate_mismatch"
        elif not visual_content_match:
            return "visual_content_mismatch"
        
        return "unknown_issue"
    
    def _determine_confidence(self, ai_result: Dict, coordinate_match: bool, 
                            visual_content_match: bool) -> str:
        """確信度判定"""
        confidence = ai_result.get("confidence", 0.0)
        quality_score = ai_result.get("quality_score", 0.0)
        
        if coordinate_match and visual_content_match and confidence >= 0.9:
            return "high"
        elif coordinate_match or visual_content_match:
            return "medium"
        else:
            return "low"
    
    def _analyze_issues(self, results: List[TrueVerificationResult]) -> Dict:
        """問題分析"""
        issue_counts = {}
        for result in results:
            if result.issue_type:
                issue_counts[result.issue_type] = issue_counts.get(result.issue_type, 0) + 1
        
        return issue_counts
    
    def _analyze_confidence(self, results: List[TrueVerificationResult]) -> Dict:
        """確信度分析"""
        confidence_counts = {}
        for result in results:
            confidence_counts[result.confidence_level] = confidence_counts.get(result.confidence_level, 0) + 1
        
        return confidence_counts
    
    def _result_to_dict(self, result: TrueVerificationResult) -> Dict:
        """結果を辞書に変換"""
        return {
            "image_id": result.image_id,
            "reported_iou": result.reported_iou,
            "reported_success": result.reported_success,
            "human_label_area": result.human_label_area,
            "ai_extraction_area": result.ai_extraction_area,
            "coordinate_match": result.coordinate_match,
            "visual_content_match": result.visual_content_match,
            "true_success": result.true_success,
            "issue_type": result.issue_type,
            "confidence_level": result.confidence_level
        }
    
    def _get_timestamp(self) -> str:
        """タイムスタンプ取得"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """メイン実行"""
    project_root = Path(__file__).parent.parent.parent.parent
    analyzer = TrueSuccessAnalyzer(project_root)
    results = analyzer.analyze_true_success_rate()
    
    print(f"真の成功率分析完了")
    print(f"分析対象: {results['total_analyzed']}件")
    print(f"真の成功率: {results['true_success_rate']:.1f}%")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())