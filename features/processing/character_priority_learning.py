#!/usr/bin/env python3
"""
Character Priority Learning System - キャラクター優先順位学習システム
主要キャラクター判定とコマ内位置による重要度評価
"""

import numpy as np
import cv2

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CharacterCandidate:
    """キャラクター候補の情報"""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mask: np.ndarray
    confidence: float
    area: float
    center: Tuple[int, int]
    position_score: float = 0.0
    size_score: float = 0.0
    face_score: float = 0.0
    priority_score: float = 0.0


class CharacterPriorityLearning:
    """キャラクター優先順位学習システム"""

    def __init__(
        self,
        enable_face_detection: bool = True,
        enable_position_analysis: bool = True,
        enable_size_priority: bool = True,
    ):
        """
        Args:
            enable_face_detection: 顔検出による優先順位付けの有効化
            enable_position_analysis: 位置分析による優先順位付けの有効化
            enable_size_priority: サイズ優先順位付けの有効化
        """
        self.enable_face_detection = enable_face_detection
        self.enable_position_analysis = enable_position_analysis
        self.enable_size_priority = enable_size_priority

        # 顔検出器
        if enable_face_detection:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self.face_profile_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_profileface.xml"
                )
            except Exception as e:
                logger.warning(f"顔検出器の初期化に失敗: {e}")
                self.enable_face_detection = False

        # 位置重みマップ (中央ほど重要)
        self.position_weight_map = self._create_position_weight_map()

        logger.info(
            f"CharacterPriorityLearning初期化: face={enable_face_detection}, "
            f"position={enable_position_analysis}, size={enable_size_priority}"
        )

    def _create_position_weight_map(self) -> np.ndarray:
        """位置重みマップの作成（中央が重要、端は低重要度）"""
        # 標準的な漫画画像サイズ想定
        height, width = 1000, 700

        # 中央からの距離に基づく重みマップ
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # 中央からの距離を計算
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # 距離を0-1に正規化し、重みに変換（中央=1.0, 端=0.2）
        normalized_distance = distance / max_distance
        weight_map = 1.0 - 0.8 * normalized_distance

        return weight_map

    def prioritize_characters(
        self, image: np.ndarray, character_candidates: List[Dict[str, Any]]
    ) -> Tuple[List[CharacterCandidate], Dict[str, Any]]:
        """
        キャラクター候補の優先順位付け

        Args:
            image: 入力画像 (H, W, 3)
            character_candidates: キャラクター候補リスト

        Returns:
            優先順位付けされたキャラクター候補と分析結果
        """
        logger.debug(f"キャラクター優先順位付け開始: {len(character_candidates)}候補")

        analysis_result = {
            "candidate_count": len(character_candidates),
            "face_detection_results": [],
            "position_analysis": [],
            "size_analysis": [],
            "final_ranking": [],
            "primary_character": None,
        }

        # キャラクター候補をCharacterCandidateオブジェクトに変換
        candidates = []
        for i, candidate in enumerate(character_candidates):
            char_candidate = CharacterCandidate(
                bbox=candidate.get("bbox", (0, 0, 0, 0)),
                mask=candidate.get("mask", np.zeros((100, 100), dtype=np.uint8)),
                confidence=candidate.get("confidence", 0.0),
                area=candidate.get("area", 0.0),
                center=candidate.get("center", (0, 0)),
            )
            candidates.append(char_candidate)

        # 1. 顔検出による優先順位付け
        if self.enable_face_detection and len(candidates) > 0:
            face_results = self._analyze_face_presence(image, candidates)
            analysis_result["face_detection_results"] = face_results

        # 2. 位置による優先順位付け
        if self.enable_position_analysis:
            position_results = self._analyze_position_priority(image, candidates)
            analysis_result["position_analysis"] = position_results

        # 3. サイズによる優先順位付け
        if self.enable_size_priority:
            size_results = self._analyze_size_priority(candidates)
            analysis_result["size_analysis"] = size_results

        # 4. 総合スコア計算
        self._calculate_final_scores(candidates)

        # 5. 優先順位でソート
        candidates.sort(key=lambda x: x.priority_score, reverse=True)

        # 6. 分析結果の更新
        analysis_result["final_ranking"] = [
            {
                "index": i,
                "priority_score": candidate.priority_score,
                "position_score": candidate.position_score,
                "size_score": candidate.size_score,
                "face_score": candidate.face_score,
                "center": candidate.center,
                "area": candidate.area,
            }
            for i, candidate in enumerate(candidates)
        ]

        if candidates:
            analysis_result["primary_character"] = {
                "index": 0,
                "priority_score": candidates[0].priority_score,
                "bbox": candidates[0].bbox,
                "center": candidates[0].center,
                "confidence": candidates[0].confidence,
            }

        logger.debug(f"キャラクター優先順位付け完了: 最優先={candidates[0].priority_score:.3f}")
        return candidates, analysis_result

    def _analyze_face_presence(
        self, image: np.ndarray, candidates: List[CharacterCandidate]
    ) -> List[Dict[str, Any]]:
        """顔検出による優先順位分析"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_results = []

        for i, candidate in enumerate(candidates):
            x, y, w, h = candidate.bbox

            # 候補領域での顔検出
            roi_gray = gray[y : y + h, x : x + w]
            if roi_gray.size == 0:
                continue

            # 正面顔検出
            frontal_faces = self.face_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            # 横顔検出
            profile_faces = self.face_profile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            total_faces = len(frontal_faces) + len(profile_faces)

            # 顔の面積比計算
            face_area_ratio = 0.0
            if total_faces > 0:
                total_face_area = sum(fw * fh for fx, fy, fw, fh in frontal_faces)
                total_face_area += sum(fw * fh for fx, fy, fw, fh in profile_faces)
                face_area_ratio = total_face_area / (w * h)

            # 顔スコア計算（顔の数と面積比から）
            face_score = min(1.0, total_faces * 0.5 + face_area_ratio * 2.0)
            candidate.face_score = face_score

            face_result = {
                "candidate_index": i,
                "frontal_faces": len(frontal_faces),
                "profile_faces": len(profile_faces),
                "total_faces": total_faces,
                "face_area_ratio": face_area_ratio,
                "face_score": face_score,
            }
            face_results.append(face_result)

        return face_results

    def _analyze_position_priority(
        self, image: np.ndarray, candidates: List[CharacterCandidate]
    ) -> List[Dict[str, Any]]:
        """位置による優先順位分析"""
        height, width = image.shape[:2]
        position_results = []

        # 重みマップをリサイズ
        weight_map = cv2.resize(self.position_weight_map, (width, height))

        for i, candidate in enumerate(candidates):
            cx, cy = candidate.center

            # 境界チェック
            cx = max(0, min(width - 1, cx))
            cy = max(0, min(height - 1, cy))

            # 位置スコア取得
            position_score = weight_map[cy, cx]

            # 中央からの距離を計算
            center_x, center_y = width // 2, height // 2
            distance_from_center = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            max_distance = math.sqrt(center_x**2 + center_y**2)
            normalized_distance = distance_from_center / max_distance

            # 画像端からの距離も考慮
            edge_distance = min(cx, cy, width - cx, height - cy)
            edge_penalty = 1.0 - (edge_distance / min(width, height) * 0.5)

            # 最終位置スコア
            final_position_score = position_score * (1.0 - edge_penalty * 0.3)
            candidate.position_score = final_position_score

            position_result = {
                "candidate_index": i,
                "center": (cx, cy),
                "position_weight": position_score,
                "distance_from_center": distance_from_center,
                "normalized_distance": normalized_distance,
                "edge_distance": edge_distance,
                "edge_penalty": edge_penalty,
                "final_position_score": final_position_score,
            }
            position_results.append(position_result)

        return position_results

    def _analyze_size_priority(self, candidates: List[CharacterCandidate]) -> List[Dict[str, Any]]:
        """サイズによる優先順位分析"""
        if not candidates:
            return []

        # 面積の統計情報
        areas = [candidate.area for candidate in candidates]
        max_area = max(areas)
        min_area = min(areas)
        area_range = max_area - min_area

        size_results = []

        for i, candidate in enumerate(candidates):
            # 相対サイズスコア（大きいほど高スコア）
            if area_range > 0:
                relative_size = (candidate.area - min_area) / area_range
            else:
                relative_size = 1.0

            # 絶対サイズスコア（適度なサイズが理想）
            # 面積が全体の5-80%の範囲が理想的
            total_image_area = max_area * 10  # 推定画像面積
            area_ratio = candidate.area / total_image_area

            if 0.05 <= area_ratio <= 0.8:
                absolute_size_score = 1.0
            elif area_ratio < 0.05:
                # 小さすぎる場合
                absolute_size_score = area_ratio / 0.05
            else:
                # 大きすぎる場合
                absolute_size_score = max(0.1, 1.0 - (area_ratio - 0.8) / 0.2)

            # 最終サイズスコア
            size_score = relative_size * 0.6 + absolute_size_score * 0.4
            candidate.size_score = size_score

            size_result = {
                "candidate_index": i,
                "area": candidate.area,
                "relative_size": relative_size,
                "area_ratio": area_ratio,
                "absolute_size_score": absolute_size_score,
                "final_size_score": size_score,
            }
            size_results.append(size_result)

        return size_results

    def _calculate_final_scores(self, candidates: List[CharacterCandidate]):
        """最終優先順位スコアの計算"""
        for candidate in candidates:
            # 重み付き平均で最終スコア計算
            priority_score = (
                candidate.face_score * 0.4
                + candidate.position_score * 0.35  # 顔の存在が最重要
                + candidate.size_score * 0.25  # 位置も重要  # サイズは補助的
            )

            # 元の信頼度も加味
            priority_score = priority_score * 0.8 + candidate.confidence * 0.2

            candidate.priority_score = min(1.0, priority_score)

    def select_primary_character(
        self, candidates: List[CharacterCandidate], selection_strategy: str = "highest_score"
    ) -> Optional[CharacterCandidate]:
        """
        主要キャラクターの選択

        Args:
            candidates: 優先順位付けされたキャラクター候補
            selection_strategy: 選択戦略 ('highest_score', 'balanced', 'conservative')

        Returns:
            選択された主要キャラクター
        """
        if not candidates:
            return None

        if selection_strategy == "highest_score":
            return candidates[0]  # 既にソート済み

        elif selection_strategy == "balanced":
            # 顔、位置、サイズがバランス良く高いものを選択
            balanced_candidates = [
                c
                for c in candidates
                if c.face_score > 0.3 and c.position_score > 0.4 and c.size_score > 0.3
            ]
            return balanced_candidates[0] if balanced_candidates else candidates[0]

        elif selection_strategy == "conservative":
            # より保守的な選択（顔が確実に検出されているもの）
            face_candidates = [c for c in candidates if c.face_score > 0.5]
            return face_candidates[0] if face_candidates else candidates[0]

        return candidates[0]

    def get_character_selection_reason(
        self, selected: CharacterCandidate, all_candidates: List[CharacterCandidate]
    ) -> str:
        """キャラクター選択理由の生成"""
        reasons = []

        if selected.face_score > 0.5:
            reasons.append(f"顔検出スコア高({selected.face_score:.2f})")

        if selected.position_score > 0.6:
            reasons.append(f"中央位置({selected.position_score:.2f})")

        if selected.size_score > 0.7:
            reasons.append(f"適切サイズ({selected.size_score:.2f})")

        if selected.confidence > 0.8:
            reasons.append(f"高信頼度({selected.confidence:.2f})")

        if len(all_candidates) > 1:
            score_gap = selected.priority_score - all_candidates[1].priority_score
            if score_gap > 0.2:
                reasons.append(f"明確な優位性({score_gap:.2f}差)")

        return "、".join(reasons) if reasons else "総合判定"


def test_character_priority_learning():
    """キャラクター優先順位学習システムのテスト"""
    learning_system = CharacterPriorityLearning(
        enable_face_detection=True, enable_position_analysis=True, enable_size_priority=True
    )

    # テスト画像
    test_image_path = Path("/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0002.jpg")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"テスト画像読み込み: {image.shape}")

        # ダミーキャラクター候補作成
        height, width = image.shape[:2]
        dummy_candidates = [
            {
                "bbox": (100, 100, 200, 300),
                "mask": np.ones((300, 200), dtype=np.uint8) * 255,
                "confidence": 0.85,
                "area": 60000,
                "center": (200, 250),
            },
            {
                "bbox": (400, 50, 150, 250),
                "mask": np.ones((250, 150), dtype=np.uint8) * 255,
                "confidence": 0.72,
                "area": 37500,
                "center": (475, 175),
            },
            {
                "bbox": (50, 400, 100, 150),
                "mask": np.ones((150, 100), dtype=np.uint8) * 255,
                "confidence": 0.68,
                "area": 15000,
                "center": (100, 475),
            },
        ]

        # 優先順位付け実行
        prioritized_candidates, analysis = learning_system.prioritize_characters(
            image, dummy_candidates
        )

        # 分析結果表示
        print("\\n🎯 キャラクター優先順位学習結果:")
        print(f"候補数: {analysis['candidate_count']}")
        print(f"顔検出結果: {len(analysis['face_detection_results'])}件")
        print(f"位置分析: {len(analysis['position_analysis'])}件")
        print(f"サイズ分析: {len(analysis['size_analysis'])}件")

        # 最終ランキング表示
        print("\\n📊 最終ランキング:")
        for i, ranking in enumerate(analysis["final_ranking"][:3]):
            print(
                f"  {i+1}位: 総合スコア{ranking['priority_score']:.3f} "
                f"(位置{ranking['position_score']:.2f}, "
                f"サイズ{ranking['size_score']:.2f}, "
                f"顔{ranking['face_score']:.2f})"
            )

        # 主要キャラクター選択
        if analysis["primary_character"]:
            primary = analysis["primary_character"]
            print(f"\\n👑 主要キャラクター: スコア{primary['priority_score']:.3f}")

            selected_candidate = learning_system.select_primary_character(prioritized_candidates)
            if selected_candidate:
                reason = learning_system.get_character_selection_reason(
                    selected_candidate, prioritized_candidates
                )
                print(f"選択理由: {reason}")

        print("\\n✅ キャラクター優先順位学習テスト完了")

    else:
        print(f"テスト画像が見つかりません: {test_image_path}")


if __name__ == "__main__":
    test_character_priority_learning()
