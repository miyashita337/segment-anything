#!/usr/bin/env python3
"""
強化SCI計算エンジン (Phase A2)
最高品質の顔・ポーズ検出と統合したSCI計算システム

品質重視の統合設計:
- EnhancedFaceDetector統合
- EnhancedPoseDetector統合
- OpenCVエラー修正
- 処理時間: 10-12秒/画像（品質最優先）
"""

import numpy as np
import cv2

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.enhanced_detection_systems import (
    EnhancedFaceDetector,
    EnhancedPoseDetector,
    FaceDetection,
    PoseDetectionResult,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedSCIResult:
    """強化SCI計算結果"""

    sci_total: float
    face_score: float
    pose_score: float
    contour_score: float
    completeness_level: str
    quality_code: int

    # 強化検出結果詳細
    face_detections: List[FaceDetection]
    pose_result: PoseDetectionResult
    processing_time: float

    # 詳細評価指標
    face_detection_rate: float
    pose_detection_rate: float
    structure_completeness: float


class EnhancedSCICalculationEngine:
    """強化SCI計算エンジン"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedSCICalculationEngine")

        # 強化検出システム初期化
        self.face_detector = EnhancedFaceDetector()
        self.pose_detector = EnhancedPoseDetector()

        self.logger.info("強化SCI計算エンジン初期化完了")

    def calculate_enhanced_sci(self, image: np.ndarray) -> EnhancedSCIResult:
        """強化SCI計算（高品質検出統合版）"""
        start_time = datetime.now()

        try:
            # 1. 強化顔検出 (40% weight - 重要性向上)
            face_detections = self.face_detector.detect_faces_comprehensive(image)
            face_score, face_detection_rate = self._calculate_enhanced_face_score(
                image, face_detections
            )

            # 2. 強化ポーズ検出 (40% weight - 重要性向上)
            pose_result = self.pose_detector.detect_pose_comprehensive(image)
            pose_score, pose_detection_rate = self._calculate_enhanced_pose_score(pose_result)

            # 3. 構造品質スコア (20% weight - 補完的)
            contour_score, structure_completeness = self._calculate_enhanced_contour_score(image)

            # 重み付き総合スコア（強化版）
            sci_total = face_score * 0.4 + pose_score * 0.4 + contour_score * 0.2

            # 完全性レベルの判定（強化版）
            completeness_level, quality_code = self._classify_enhanced_sci_quality(
                sci_total, face_detection_rate, pose_detection_rate
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"強化SCI計算完了: 総合={sci_total:.3f}, "
                f"顔={face_score:.3f}, ポーズ={pose_score:.3f}, "
                f"輪郭={contour_score:.3f} "
                f"（処理時間: {processing_time:.2f}秒）"
            )

            return EnhancedSCIResult(
                sci_total=sci_total,
                face_score=face_score,
                pose_score=pose_score,
                contour_score=contour_score,
                completeness_level=completeness_level,
                quality_code=quality_code,
                face_detections=face_detections,
                pose_result=pose_result,
                processing_time=processing_time,
                face_detection_rate=face_detection_rate,
                pose_detection_rate=pose_detection_rate,
                structure_completeness=structure_completeness,
            )

        except Exception as e:
            self.logger.error(f"強化SCI計算エラー: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return EnhancedSCIResult(
                sci_total=0.0,
                face_score=0.0,
                pose_score=0.0,
                contour_score=0.0,
                completeness_level="エラー",
                quality_code=0,
                face_detections=[],
                pose_result=PoseDetectionResult(
                    detected=False,
                    landmarks=None,
                    visibility_score=0.0,
                    pose_category="unknown",
                    completeness_score=0.0,
                    confidence=0.0,
                    keypoints_detected=0,
                ),
                processing_time=processing_time,
                face_detection_rate=0.0,
                pose_detection_rate=0.0,
                structure_completeness=0.0,
            )

    def _calculate_enhanced_face_score(
        self, image: np.ndarray, face_detections: List[FaceDetection]
    ) -> Tuple[float, float]:
        """強化顔検出スコア計算"""
        if not face_detections:
            return 0.0, 0.0  # (face_score, detection_rate)

        # 最高信頼度の顔を選択
        best_face = max(face_detections, key=lambda f: f.confidence)

        # 基本信頼度スコア
        confidence_score = best_face.confidence

        # 顔サイズ適正性評価
        x, y, w, h = best_face.bbox
        face_area = w * h
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area

        # アニメキャラクターに適した顔サイズ評価
        if 0.01 <= size_ratio <= 0.4:  # 1%-40%が適正範囲
            size_score = 1.0
        elif size_ratio < 0.01:
            size_score = size_ratio / 0.01  # 小さすぎる場合の段階的減点
        else:
            size_score = max(0.3, 1.0 - (size_ratio - 0.4) / 0.6)  # 大きすぎる場合

        # 顔位置評価（中央付近が好ましい）
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        image_center_x = image.shape[1] // 2
        image_center_y = image.shape[0] // 2

        center_distance = np.sqrt(
            ((face_center_x - image_center_x) / image.shape[1]) ** 2
            + ((face_center_y - image_center_y) / image.shape[0]) ** 2
        )
        position_score = max(0.5, 1.0 - center_distance)

        # 複数顔検出ボーナス（複数手法で確認）
        multi_method_bonus = min(0.2, len(face_detections) * 0.05)

        # 総合顔スコア計算
        face_score = (
            confidence_score * 0.4
            + size_score * 0.3
            + position_score * 0.2
            + multi_method_bonus
            + 0.1
        )

        face_score = min(1.0, face_score)
        detection_rate = 1.0  # 顔が検出された場合は100%

        return face_score, detection_rate

    def _calculate_enhanced_pose_score(
        self, pose_result: PoseDetectionResult
    ) -> Tuple[float, float]:
        """強化ポーズ検出スコア計算"""
        if not pose_result.detected:
            return 0.0, 0.0  # (pose_score, detection_rate)

        # 基本ポーズスコア要素
        visibility_weight = 0.3
        completeness_weight = 0.4
        confidence_weight = 0.2
        category_weight = 0.1

        # カテゴリ別ボーナス
        category_bonuses = {
            "standing": 1.0,  # 立位は標準
            "sitting": 0.9,  # 座位も良好
            "action": 1.1,  # アクションは高評価
            "lying": 0.8,  # 横臥位は少し低め
            "profile": 0.85,  # 横向きは少し低め
            "unknown": 0.6,  # 不明は低評価
        }

        category_bonus = category_bonuses.get(pose_result.pose_category, 0.6)

        # キーポイント密度ボーナス
        keypoint_density = pose_result.keypoints_detected / 33.0
        density_bonus = keypoint_density * 0.2

        # 総合ポーズスコア計算
        pose_score = (
            pose_result.visibility_score * visibility_weight
            + pose_result.completeness_score * completeness_weight
            + pose_result.confidence * confidence_weight
            + category_bonus * category_weight
            + density_bonus
        )

        pose_score = min(1.0, pose_score)
        detection_rate = 1.0  # ポーズが検出された場合は100%

        return pose_score, detection_rate

    def _calculate_enhanced_contour_score(self, image: np.ndarray) -> Tuple[float, float]:
        """強化輪郭品質スコア計算（OpenCVエラー修正版）"""
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # エッジ検出（高品質設定）
            # ガウシアンブラーで前処理
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Cannyエッジ検出（アニメ画像に適した設定）
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

            # 輪郭検出（OpenCVエラー修正）
            # edges をuint8型に確実に変換
            edges_uint8 = edges.astype(np.uint8)

            contours, _ = cv2.findContours(edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.2, 0.0  # 輪郭なしでも最低スコア

            # 輪郭品質評価
            total_contour_length = sum(cv2.arcLength(contour, True) for contour in contours)
            image_perimeter = 2 * (image.shape[0] + image.shape[1])

            # 輪郭密度スコア
            contour_density = min(1.0, total_contour_length / image_perimeter)

            # 主要輪郭分析
            main_contours = [c for c in contours if cv2.contourArea(c) > 100]
            main_contour_ratio = len(main_contours) / max(1, len(contours))

            # 輪郭の複雑度評価
            complexity_scores = []
            for contour in main_contours[:5]:  # 上位5つの輪郭
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                complexity = len(approx) / max(4, len(approx))  # 正規化複雑度
                complexity_scores.append(min(1.0, complexity))

            avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.3

            # 総合輪郭スコア
            contour_score = contour_density * 0.4 + main_contour_ratio * 0.3 + avg_complexity * 0.3

            contour_score = min(1.0, contour_score)
            structure_completeness = contour_score

            return contour_score, structure_completeness

        except Exception as e:
            self.logger.warning(f"輪郭品質計算エラー（修正版）: {e}")
            return 0.3, 0.0  # エラー時のデフォルトスコア

    def _classify_enhanced_sci_quality(
        self, sci_total: float, face_detection_rate: float, pose_detection_rate: float
    ) -> Tuple[str, int]:
        """強化SCI品質分類"""
        # 品質閾値（厳格化）
        if sci_total >= 0.85 and face_detection_rate > 0.9 and pose_detection_rate > 0.9:
            return "最高品質", 5
        elif sci_total >= 0.70 and face_detection_rate > 0.7 and pose_detection_rate > 0.7:
            return "高品質", 4
        elif sci_total >= 0.55 and (face_detection_rate > 0.5 or pose_detection_rate > 0.5):
            return "良好", 3
        elif sci_total >= 0.40:
            return "普通", 2
        elif sci_total >= 0.25:
            return "低品質", 1
        else:
            return "品質不足", 0


def main():
    """テスト実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="強化SCI計算エンジンテスト")
    parser.add_argument("--image", "-i", required=True, help="テスト画像パス")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 画像読み込み
    image = cv2.imread(args.image)
    if image is None:
        print(f"画像読み込み失敗: {args.image}")
        return 1

    print(f"🧮 強化SCI計算エンジンテスト: {args.image}")
    print("=" * 70)

    # 強化SCI計算実行
    sci_engine = EnhancedSCICalculationEngine()
    result = sci_engine.calculate_enhanced_sci(image)

    # 結果表示
    print(f"📊 強化SCI計算結果:")
    print(f"  総合SCI: {result.sci_total:.3f}")
    print(f"  顔スコア: {result.face_score:.3f} (検出率: {result.face_detection_rate:.1%})")
    print(f"  ポーズスコア: {result.pose_score:.3f} (検出率: {result.pose_detection_rate:.1%})")
    print(f"  輪郭スコア: {result.contour_score:.3f}")
    print(f"  品質レベル: {result.completeness_level} (コード: {result.quality_code})")
    print(f"  処理時間: {result.processing_time:.2f}秒")

    print(f"\n🔍 詳細検出結果:")
    print(f"  顔検出: {len(result.face_detections)}件")
    if result.face_detections:
        for i, face in enumerate(result.face_detections):
            print(f"    顔{i+1}: {face.method}, 信頼度={face.confidence:.3f}")

    print(f"  ポーズ検出: {'成功' if result.pose_result.detected else '失敗'}")
    if result.pose_result.detected:
        print(f"    カテゴリ: {result.pose_result.pose_category}")
        print(f"    可視キーポイント: {result.pose_result.keypoints_detected}/33")
        print(f"    完全性: {result.pose_result.completeness_score:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
