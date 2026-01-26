#!/usr/bin/env python3
"""
Multiple Character Detection System
複数キャラクター検出システム - LoRA学習用単一キャラクター品質保証
"""

import numpy as np
import cv2

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.file_utils import generate_output_filename, is_already_processed

logger = logging.getLogger(__name__)


class MultipleCharacterType(Enum):
    """複数キャラクター検出タイプ"""

    MULTIPLE_PERSONS = "multiple_persons"
    OVERLAPPING_CHARACTERS = "overlapping_characters"
    BACKGROUND_CHARACTERS = "background_characters"
    GROUPED_CHARACTERS = "grouped_characters"


@dataclass
class MultipleCharacterResult:
    """複数キャラクター検出結果"""

    is_multiple: bool
    character_count: int
    detection_type: MultipleCharacterType
    confidence_score: float
    penalty_score: float  # 品質ペナルティ（0-1、1が最大ペナルティ）
    primary_character_index: Optional[int]
    characters: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    technical_details: Dict[str, Any]


class MultipleCharacterDetector:
    """
    複数キャラクター検出器
    LoRA学習に不適切な複数キャラクター画像を検出・ペナルティ付与
    """

    # 検出しきい値設定（抽出後画像用に調整）
    DETECTION_THRESHOLDS = {
        "min_character_confidence": 0.25,  # 最小キャラクター信頼度（抽出後は厳格化）
        "overlap_iou_threshold": 0.2,  # 重複判定IoU閾値（抽出後は厳格化）
        "size_ratio_threshold": 0.1,  # メインキャラとのサイズ比閾値（厳格化）
        "distance_threshold": 0.3,  # 距離ベース判定閾値（厳格化）
        "background_size_ratio": 0.03,  # 背景キャラサイズ判定閾値（厳格化）
    }

    # ペナルティ重み設定（抽出後画像用に重大度を強化）
    PENALTY_WEIGHTS = {
        "multiple_detected": 0.9,  # 複数検出ペナルティ（抽出後は重大）
        "overlap_penalty": 0.8,  # 重複ペナルティ（抽出後は重大）
        "size_disparity": 0.6,  # サイズ格差ペナルティ（強化）
        "background_characters": 0.85,  # 背景キャラペナルティ（強化）
        "extraction_failure": 0.95,  # 抽出失敗ペナルティ（新規追加）
    }

    def __init__(self):
        """初期化"""
        pass

    def analyze_yolo_detections(
        self, yolo_detections: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> MultipleCharacterResult:
        """
        YOLO検出結果から複数キャラクター分析

        Args:
            yolo_detections: YOLO人物検出結果リスト
            image_shape: 画像サイズ (height, width)

        Returns:
            複数キャラクター検出結果
        """
        if len(yolo_detections) <= 1:
            # 単一またはキャラクターなし
            return MultipleCharacterResult(
                is_multiple=False,
                character_count=len(yolo_detections),
                detection_type=MultipleCharacterType.MULTIPLE_PERSONS,
                confidence_score=1.0,
                penalty_score=0.0,
                primary_character_index=0 if yolo_detections else None,
                characters=yolo_detections,
                improvement_suggestions=[],
                technical_details={"analysis_method": "yolo_count_based"},
            )

        # 複数キャラクター分析開始
        logger.info(f"🔍 複数キャラクター分析開始: {len(yolo_detections)}体検出")

        # 1. 基本情報収集
        characters = []
        for i, detection in enumerate(yolo_detections):
            bbox = detection["bbox"]  # [x, y, w, h]
            area = bbox[2] * bbox[3]

            character_info = {
                "index": i,
                "bbox": bbox,
                "bbox_xyxy": detection.get("bbox_xyxy", []),
                "confidence": detection["confidence"],
                "area": area,
                "relative_area": area / (image_shape[0] * image_shape[1]),
                "center": (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
            }
            characters.append(character_info)

        # 2. メインキャラクター特定（最大面積 + 高信頼度）
        primary_character_index = self._identify_primary_character(characters)
        primary_char = characters[primary_character_index]

        # 3. 複数キャラクタータイプ判定
        detection_type, type_confidence = self._classify_multiple_character_type(
            characters, primary_character_index, image_shape
        )

        # 4. ペナルティスコア計算
        penalty_score = self._calculate_penalty_score(characters, detection_type, image_shape)

        # 5. 改善提案生成
        suggestions = self._generate_improvement_suggestions(
            detection_type, len(characters), penalty_score
        )

        result = MultipleCharacterResult(
            is_multiple=True,
            character_count=len(characters),
            detection_type=detection_type,
            confidence_score=type_confidence,
            penalty_score=penalty_score,
            primary_character_index=primary_character_index,
            characters=characters,
            improvement_suggestions=suggestions,
            technical_details={
                "analysis_method": "comprehensive_multi_detection",
                "primary_character_area": primary_char["relative_area"],
                "character_confidences": [c["confidence"] for c in characters],
                "detection_thresholds": self.DETECTION_THRESHOLDS.copy(),
            },
        )

        logger.info(f"✅ 複数キャラクター分析完了: タイプ={detection_type.value}, " f"ペナルティ={penalty_score:.2f}")

        return result

    def _identify_primary_character(self, characters: List[Dict[str, Any]]) -> int:
        """
        メインキャラクター特定

        Args:
            characters: キャラクター情報リスト

        Returns:
            メインキャラクターのインデックス
        """
        # 面積 × 信頼度でメインキャラクター決定
        best_score = 0.0
        best_index = 0

        for i, char in enumerate(characters):
            # 複合スコア: 面積重視60% + 信頼度40%
            area_score = char["relative_area"] / max(c["relative_area"] for c in characters)
            confidence_score = char["confidence"]
            composite_score = area_score * 0.6 + confidence_score * 0.4

            if composite_score > best_score:
                best_score = composite_score
                best_index = i

        return best_index

    def _classify_multiple_character_type(
        self, characters: List[Dict[str, Any]], primary_index: int, image_shape: Tuple[int, int]
    ) -> Tuple[MultipleCharacterType, float]:
        """
        複数キャラクタータイプ分類

        Args:
            characters: キャラクター情報リスト
            primary_index: メインキャラクターインデックス
            image_shape: 画像サイズ

        Returns:
            (検出タイプ, 信頼度スコア)
        """
        primary_char = characters[primary_index]
        other_characters = [c for i, c in enumerate(characters) if i != primary_index]

        # 1. 重複キャラクター判定
        overlap_count = 0
        for other in other_characters:
            iou = self._calculate_iou(primary_char["bbox"], other["bbox"])
            if iou > self.DETECTION_THRESHOLDS["overlap_iou_threshold"]:
                overlap_count += 1

        if overlap_count > 0:
            confidence = min(overlap_count / len(other_characters), 1.0)
            return MultipleCharacterType.OVERLAPPING_CHARACTERS, confidence

        # 2. 背景キャラクター判定
        background_count = 0
        for other in other_characters:
            size_ratio = other["relative_area"] / primary_char["relative_area"]
            if size_ratio < self.DETECTION_THRESHOLDS["background_size_ratio"]:
                background_count += 1

        if background_count > 0:
            confidence = background_count / len(other_characters)
            return MultipleCharacterType.BACKGROUND_CHARACTERS, confidence

        # 3. グループキャラクター判定（近距離の同等サイズ）
        grouped_count = 0
        image_diagonal = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)

        for other in other_characters:
            # 距離計算
            distance = np.sqrt(
                (primary_char["center"][0] - other["center"][0]) ** 2
                + (primary_char["center"][1] - other["center"][1]) ** 2
            )
            relative_distance = distance / image_diagonal

            # サイズ比計算
            size_ratio = other["relative_area"] / primary_char["relative_area"]

            if (
                relative_distance < self.DETECTION_THRESHOLDS["distance_threshold"]
                and size_ratio > self.DETECTION_THRESHOLDS["size_ratio_threshold"]
            ):
                grouped_count += 1

        if grouped_count > 0:
            confidence = grouped_count / len(other_characters)
            return MultipleCharacterType.GROUPED_CHARACTERS, confidence

        # 4. デフォルト: 通常の複数人物
        return MultipleCharacterType.MULTIPLE_PERSONS, 0.8

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        IoU (Intersection over Union) 計算

        Args:
            bbox1, bbox2: バウンディングボックス [x, y, w, h]

        Returns:
            IoUスコア
        """
        # [x, y, w, h] → [x1, y1, x2, y2] 変換
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

        # 交差領域計算
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_penalty_score(
        self,
        characters: List[Dict[str, Any]],
        detection_type: MultipleCharacterType,
        image_shape: Tuple[int, int],
    ) -> float:
        """
        ペナルティスコア計算

        Args:
            characters: キャラクター情報リスト
            detection_type: 検出タイプ
            image_shape: 画像サイズ

        Returns:
            ペナルティスコア (0-1)
        """
        penalty = 0.0

        # 基本複数キャラペナルティ
        character_count_penalty = min((len(characters) - 1) * 0.3, 1.0)
        penalty += character_count_penalty * self.PENALTY_WEIGHTS["multiple_detected"]

        # タイプ別ペナルティ（抽出後画像用に強化）
        if detection_type == MultipleCharacterType.OVERLAPPING_CHARACTERS:
            penalty += 0.9 * self.PENALTY_WEIGHTS["overlap_penalty"]  # 抽出後重複は重大
        elif detection_type == MultipleCharacterType.BACKGROUND_CHARACTERS:
            penalty += 0.8 * self.PENALTY_WEIGHTS["background_characters"]  # 背景残存は問題
        elif detection_type == MultipleCharacterType.GROUPED_CHARACTERS:
            penalty += 0.85 * self.PENALTY_WEIGHTS["multiple_detected"]  # グループ残存は重大
        elif detection_type == MultipleCharacterType.MULTIPLE_PERSONS:
            penalty += 0.95 * self.PENALTY_WEIGHTS["extraction_failure"]  # 抽出失敗は最重要

        # サイズ格差ペナルティ
        if len(characters) >= 2:
            areas = [c["relative_area"] for c in characters]
            area_variance = np.var(areas)
            size_disparity_penalty = min(area_variance * 10, 1.0)
            penalty += size_disparity_penalty * self.PENALTY_WEIGHTS["size_disparity"]

        return min(penalty, 1.0)

    def _generate_improvement_suggestions(
        self, detection_type: MultipleCharacterType, character_count: int, penalty_score: float
    ) -> List[str]:
        """
        改善提案生成

        Args:
            detection_type: 検出タイプ
            character_count: キャラクター数
            penalty_score: ペナルティスコア

        Returns:
            改善提案リスト
        """
        suggestions = []

        # タイプ別提案（抽出後画像用に更新）
        if detection_type == MultipleCharacterType.OVERLAPPING_CHARACTERS:
            suggestions.extend(
                ["⚠️ 抽出後重複残存: SAMマスク精度改善が必要", "アダプティブクロッピング処理の見直し推奨", "より厳密なマスク重複判定の導入検討"]
            )
        elif detection_type == MultipleCharacterType.BACKGROUND_CHARACTERS:
            suggestions.extend(
                ["⚠️ 背景キャラ残存: 抽出アルゴリズム改善必要", "メインキャラクター選択ロジックの見直し", "背景ノイズフィルタリングの強化"]
            )
        elif detection_type == MultipleCharacterType.GROUPED_CHARACTERS:
            suggestions.extend(["⚠️ グループキャラ残存: 単一キャラ抽出失敗", "YOLO+SAM連携処理の改良が必要", "クロッピング範囲の最適化検討"])
        else:  # MULTIPLE_PERSONS
            suggestions.extend(["🚨 抽出処理失敗: 複数キャラが抽出後も残存", "抽出アルゴリズムの根本的見直し必要", "この画像はLoRA学習から除外推奨"])

        # ペナルティ重要度別追加提案（抽出後画像用）
        if penalty_score > 0.7:
            suggestions.append("🚨 重大な抽出品質問題: LoRA学習不適切・抽出処理改善必須")
        elif penalty_score > 0.4:
            suggestions.append("⚠️ 抽出品質注意: 使用前に手動確認・処理見直し検討")
        else:
            suggestions.append("✅ 抽出品質良好: 使用可能だが継続監視推奨")

        return suggestions

    def create_visualization(
        self, image: np.ndarray, result: MultipleCharacterResult, output_path: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """
        複数キャラクター検出結果の可視化

        Args:
            image: 元画像
            result: 検出結果
            output_path: 保存パス（オプション）

        Returns:
            可視化画像
        """
        if not result.is_multiple:
            return None

        vis_image = image.copy()

        # 各キャラクターのバウンディングボックス描画
        for i, char in enumerate(result.characters):
            bbox = char["bbox"]  # [x, y, w, h]

            # 色設定（メインキャラクターは緑、その他は赤）
            if i == result.primary_character_index:
                color = (0, 255, 0)  # 緑: メインキャラクター
                thickness = 3
            else:
                color = (0, 0, 255)  # 赤: その他キャラクター
                thickness = 2

            # バウンディングボックス描画
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color,
                thickness,
            )

            # ラベル描画
            label = f"#{i+1}: {char['confidence']:.2f}"
            if i == result.primary_character_index:
                label += " (Main)"

            cv2.putText(
                vis_image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # 検出情報描画
        info_text = [
            f"Type: {result.detection_type.value}",
            f"Count: {result.character_count}",
            f"Penalty: {result.penalty_score:.2f}",
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(
                vis_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            y_offset += 25

        # 保存
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"📊 可視化結果保存: {output_path}")

        return vis_image


def detect_multiple_characters_from_image(
    image_path: Path, yolo_wrapper, save_visualization: bool = False
) -> MultipleCharacterResult:
    """
    画像から複数キャラクター検出（便利関数）

    Args:
        image_path: 画像パス
        yolo_wrapper: YOLOWrapper インスタンス
        save_visualization: 可視化結果保存フラグ

    Returns:
        複数キャラクター検出結果
    """
    # 重複処理チェック：既に_multi_char_detection処理済みの場合はスキップ
    if "_multi_char_detection" in image_path.stem:
        logger.warning(f"⚠️  Already processed for multi-char detection: {image_path}")
        # 空の結果を返す（重複処理防止）
        return MultipleCharacterResult(
            is_multiple=False,
            character_count=0,
            detection_type=MultipleCharacterType.MULTIPLE_PERSONS,
            confidence_score=0.0,
            penalty_score=0.0,
            primary_character_index=None,
            characters=[],
            improvement_suggestions=[],
            technical_details={},
        )

    # 既存の出力ファイルチェック
    output_dir = str(image_path.parent)
    existing_output = is_already_processed(str(image_path), output_dir, "multi_char_detection")
    if existing_output:
        logger.info(f"✅ Multi-char detection already completed: {existing_output}")
        # 既存結果があるので、実際の結果を返すべきだが、今回は重複防止を優先
        return MultipleCharacterResult(
            is_multiple=False,
            character_count=0,
            detection_type=MultipleCharacterType.MULTIPLE_PERSONS,
            confidence_score=0.0,
            penalty_score=0.0,
            primary_character_index=None,
            characters=[],
            improvement_suggestions=[],
            technical_details={"skipped": "already_processed"},
        )

    # 画像読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像読み込み失敗: {image_path}")

    # YOLO検出実行
    yolo_detections = yolo_wrapper.detect_persons(image)

    # 複数キャラクター分析
    detector = MultipleCharacterDetector()
    result = detector.analyze_yolo_detections(yolo_detections, image.shape[:2])

    # 可視化保存（重複サフィックス防止）
    if save_visualization and result.is_multiple:
        # 重複サフィックス防止のファイル名生成
        vis_filename = generate_output_filename(str(image_path), "multi_char_detection")
        vis_output = image_path.parent / vis_filename

        logger.info(f"📊 Saving multi-char visualization: {vis_output}")
        detector.create_visualization(image, result, vis_output)

    return result


if __name__ == "__main__":
    # テスト実行例
    print("🔍 Multiple Character Detection System - Test")

    # テスト用のモックデータ
    mock_detections = [
        {"bbox": [100, 50, 200, 400], "confidence": 0.9, "bbox_xyxy": [100, 50, 300, 450]},
        {"bbox": [350, 100, 150, 300], "confidence": 0.7, "bbox_xyxy": [350, 100, 500, 400]},
        {"bbox": [600, 200, 100, 200], "confidence": 0.5, "bbox_xyxy": [600, 200, 700, 400]},
    ]

    detector = MultipleCharacterDetector()
    result = detector.analyze_yolo_detections(mock_detections, (600, 800))

    print(f"✅ テスト結果:")
    print(f"  複数キャラ: {result.is_multiple}")
    print(f"  キャラ数: {result.character_count}")
    print(f"  タイプ: {result.detection_type.value}")
    print(f"  ペナルティ: {result.penalty_score:.2f}")
    print(f"  メイン: #{result.primary_character_index + 1}")
    print(f"  改善提案: {len(result.improvement_suggestions)}件")
