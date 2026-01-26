#!/usr/bin/env python3
"""
P1-B004: 適応的クロッピングシステム
複数キャラ混入を防ぐMediaPipe顔検出統合システム
"""

import numpy as np
import cv2

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Face detection features disabled.")


@dataclass
class DetectionBox:
    """検出ボックス情報"""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: str  # 'yolo' or 'mediapipe'

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """x1, y1, x2, y2形式に変換"""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class CroppingCandidate:
    """クロッピング候補"""

    bbox: DetectionBox
    scale_factor: float
    quality_score: float
    face_count: int
    character_integrity: float

    @property
    def composite_score(self) -> float:
        """複合スコア計算"""
        # 品質スコア(40%) + 顔数逆数(30%) + キャラ完整性(30%)
        face_penalty = 1.0 / max(self.face_count, 1) if self.face_count > 1 else 1.0
        return self.quality_score * 0.4 + face_penalty * 0.3 + self.character_integrity * 0.3


class AdaptiveCropper:
    """適応的クロッピングシステム"""

    def __init__(self):
        self.face_detector = None
        self.scale_factors = [0.8, 1.0, 1.2]  # マルチスケール候補
        self.min_face_confidence = 0.5
        self.max_characters = 1  # 主要キャラ1体のみを目標

        if MEDIAPIPE_AVAILABLE:
            self._initialize_mediapipe()

    def _initialize_mediapipe(self):
        """MediaPipe顔検出初期化"""
        try:
            mp_face_detection = mp.solutions.face_detection
            self.face_detector = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.min_face_confidence  # 高精度モデル
            )
            logger.info("MediaPipe顔検出システム初期化完了")
        except Exception as e:
            logger.error(f"MediaPipe初期化エラー: {e}")
            self.face_detector = None

    def detect_faces(self, image: np.ndarray) -> List[DetectionBox]:
        """顔検出実行"""
        if not self.face_detector:
            logger.warning("MediaPipe顔検出が利用できません")
            return []

        try:
            # RGB変換（MediaPipe要件）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_image)

            faces = []
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    # 相対座標を絶対座標に変換
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # 境界チェック
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    confidence = detection.score[0] if detection.score else 0.0

                    faces.append(
                        DetectionBox(
                            x=x, y=y, w=width, h=height, confidence=confidence, source="mediapipe"
                        )
                    )

            logger.info(f"顔検出完了: {len(faces)}個の顔を検出")
            return faces

        except Exception as e:
            logger.error(f"顔検出エラー: {e}")
            return []

    def calculate_iou(self, box1: DetectionBox, box2: DetectionBox) -> float:
        """IoU計算（重複領域評価）"""
        x1_1, y1_1, x2_1, y2_1 = box1.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = box2.to_xyxy()

        # 重複領域計算
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = box1.area + box2.area - intersection

        return intersection / max(union, 1)

    def optimize_bbox_with_faces(
        self, yolo_bbox: DetectionBox, faces: List[DetectionBox], image_shape: Tuple[int, int]
    ) -> DetectionBox:
        """顔検出結果でYOLO境界ボックス最適化"""
        if not faces:
            return yolo_bbox

        h, w = image_shape[:2]

        # YOLO境界ボックス内の顔を特定
        faces_in_bbox = []
        for face in faces:
            iou = self.calculate_iou(yolo_bbox, face)
            if iou > 0.1:  # 10%以上重複
                faces_in_bbox.append((face, iou))

        if not faces_in_bbox:
            return yolo_bbox

        # 最も重複度の高い顔を主要キャラとして選択
        primary_face = max(faces_in_bbox, key=lambda x: x[1])[0]

        # 主要キャラの顔を中心とした境界ボックス再計算
        face_center_x, face_center_y = primary_face.center

        # 顔のサイズに基づく適切な境界ボックス推定
        face_size = max(primary_face.w, primary_face.h)
        estimated_body_height = face_size * 6  # 顔の6倍をキャラ全体高さと推定
        estimated_body_width = face_size * 2.5  # 顔の2.5倍を幅と推定

        # 新しい境界ボックス計算
        new_x = max(0, face_center_x - estimated_body_width // 2)
        new_y = max(0, face_center_y - face_size // 2)  # 顔位置から少し上
        new_w = min(estimated_body_width, w - new_x)
        new_h = min(estimated_body_height, h - new_y)

        optimized_bbox = DetectionBox(
            x=new_x,
            y=new_y,
            w=new_w,
            h=new_h,
            confidence=yolo_bbox.confidence * primary_face.confidence,
            source="optimized",
        )

        logger.info(
            f"境界ボックス最適化: {yolo_bbox.area} → {optimized_bbox.area} "
            f"(顔検出信頼度: {primary_face.confidence:.3f})"
        )

        return optimized_bbox

    def generate_multiscale_candidates(
        self, base_bbox: DetectionBox, image_shape: Tuple[int, int]
    ) -> List[DetectionBox]:
        """マルチスケール候補生成"""
        h, w = image_shape[:2]
        candidates = []

        for scale in self.scale_factors:
            # スケール適用
            new_w = int(base_bbox.w * scale)
            new_h = int(base_bbox.h * scale)

            # 中心維持でサイズ変更
            center_x, center_y = base_bbox.center
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 2)

            # 境界制限
            new_w = min(new_w, w - new_x)
            new_h = min(new_h, h - new_y)

            if new_w > 0 and new_h > 0:
                candidate = DetectionBox(
                    x=new_x,
                    y=new_y,
                    w=new_w,
                    h=new_h,
                    confidence=base_bbox.confidence,
                    source=f"scale_{scale}",
                )
                candidates.append(candidate)

        return candidates

    def evaluate_cropping_quality(
        self, bbox: DetectionBox, faces: List[DetectionBox], image: np.ndarray
    ) -> Tuple[float, int, float]:
        """クロッピング品質評価"""
        # 1. 顔数カウント（境界ボックス内）
        faces_in_crop = 0
        total_face_confidence = 0.0

        for face in faces:
            iou = self.calculate_iou(bbox, face)
            if iou > 0.5:  # 50%以上重複で内部判定
                faces_in_crop += 1
                total_face_confidence += face.confidence

        # 2. アスペクト比評価（全身キャラ向け）
        aspect_ratio = bbox.h / max(bbox.w, 1)
        aspect_score = 1.0
        if 1.2 <= aspect_ratio <= 2.5:
            aspect_score = min((aspect_ratio - 0.5) / 2.0, 1.0)
        else:
            aspect_score = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)

        # 3. サイズ適切性評価
        image_area = image.shape[0] * image.shape[1]
        area_ratio = bbox.area / image_area
        size_score = 1.0
        if 0.05 <= area_ratio <= 0.4:
            size_score = min(area_ratio / 0.4, 1.0)
        else:
            size_score = max(0, 1.0 - abs(area_ratio - 0.2) / 0.2)

        # 4. キャラクター完整性（境界に近い要素のペナルティ）
        border_penalty = 0.0
        border_threshold = 10  # ピクセル
        h, w = image.shape[:2]

        if bbox.x < border_threshold or bbox.y < border_threshold:
            border_penalty += 0.2
        if (bbox.x + bbox.w) > (w - border_threshold) or (bbox.y + bbox.h) > (h - border_threshold):
            border_penalty += 0.2

        integrity_score = max(0.0, 1.0 - border_penalty)

        # 総合品質スコア
        quality_score = aspect_score * 0.3 + size_score * 0.3 + integrity_score * 0.4

        # 顔信頼度平均
        avg_face_confidence = total_face_confidence / max(faces_in_crop, 1)

        return quality_score, faces_in_crop, avg_face_confidence

    def adaptive_crop(self, image: np.ndarray, yolo_bbox: DetectionBox) -> Optional[DetectionBox]:
        """適応的クロッピング実行"""
        try:
            logger.info("P1-B004: 適応的クロッピング開始")

            # Step 1: 顔検出
            faces = self.detect_faces(image)

            # Step 2: YOLO境界ボックス最適化
            optimized_bbox = self.optimize_bbox_with_faces(yolo_bbox, faces, image.shape[:2])

            # Step 3: マルチスケール候補生成
            candidates = self.generate_multiscale_candidates(optimized_bbox, image.shape[:2])

            # Step 4: 品質評価ベース最適選択
            best_candidate = None
            best_score = 0.0

            evaluation_results = []

            for candidate in candidates:
                quality, face_count, integrity = self.evaluate_cropping_quality(
                    candidate, faces, image
                )

                cropping_candidate = CroppingCandidate(
                    bbox=candidate,
                    scale_factor=candidate.source.split("_")[1]
                    if "_" in candidate.source
                    else "1.0",
                    quality_score=quality,
                    face_count=face_count,
                    character_integrity=integrity,
                )

                evaluation_results.append(cropping_candidate)

                composite = cropping_candidate.composite_score
                if composite > best_score:
                    best_score = composite
                    best_candidate = candidate

            # デバッグ情報出力
            logger.info(f"候補評価結果:")
            for i, result in enumerate(evaluation_results):
                logger.info(
                    f"  候補{i+1}: 複合スコア={result.composite_score:.3f} "
                    f"(品質={result.quality_score:.3f}, 顔数={result.face_count}, "
                    f"完整性={result.character_integrity:.3f})"
                )

            if best_candidate:
                logger.info(f"✅ P1-B004: 最適クロッピング選択完了 (スコア: {best_score:.3f})")
                return best_candidate
            else:
                logger.warning("❌ P1-B004: 適切なクロッピング候補が見つかりません")
                return yolo_bbox  # フォールバック

        except Exception as e:
            logger.error(f"❌ P1-B004: 適応的クロッピングエラー: {e}")
            return yolo_bbox  # エラー時はオリジナルを返す

    def __del__(self):
        """リソース解放"""
        if self.face_detector:
            self.face_detector.close()


# ユーティリティ関数


def create_adaptive_cropper() -> AdaptiveCropper:
    """アダプティブクロッパー作成"""
    return AdaptiveCropper()


def apply_adaptive_cropping(
    image: np.ndarray, yolo_bbox: List[int], confidence: float = 0.5
) -> Optional[List[int]]:
    """適応的クロッピング適用（統合用関数）"""
    cropper = create_adaptive_cropper()

    # YOLO bbox を DetectionBox に変換
    detection_box = DetectionBox(
        x=yolo_bbox[0],
        y=yolo_bbox[1],
        w=yolo_bbox[2],
        h=yolo_bbox[3],
        confidence=confidence,
        source="yolo",
    )

    # 適応的クロッピング実行
    result = cropper.adaptive_crop(image, detection_box)

    if result:
        return [result.x, result.y, result.w, result.h]
    else:
        return yolo_bbox


if __name__ == "__main__":
    # テスト実行
    print("P1-B004: 適応的クロッピングシステム テスト")
    print(f"MediaPipe利用可能: {MEDIAPIPE_AVAILABLE}")

    if MEDIAPIPE_AVAILABLE:
        cropper = create_adaptive_cropper()
        print("✅ アダプティブクロッパー初期化成功")
    else:
        print("❌ MediaPipeが利用できません。pip install mediapipe を実行してください。")
