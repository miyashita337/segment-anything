#!/usr/bin/env python3
"""
検出結果アンサンブルシステム v1.0.0
重み付き投票による最適検出結果選択システム

Week 3: 検出精度向上の切り札
- 複数検出手法の統合
- 重み付き投票アルゴリズム
- アニメ特化チューニング
"""

import numpy as np
import cv2
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.enhanced_detection_systems import (
    EnhancedFaceDetector,
    EnhancedPoseDetector,
    FaceDetection,
    PoseDetectionResult,
    EnhancedDetectionReport,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeights:
    """アンサンブル重み設定"""

    face_mediapipe: float = 0.40  # MediaPipe顔検出
    face_opencv_anime: float = 0.35  # アニメ専用Cascade
    face_opencv_standard: float = 0.15  # 標準Cascade
    face_dlib: float = 0.10  # dlib補完用

    pose_high_quality: float = 0.60  # 高品質ポーズ検出
    pose_efficient: float = 0.40  # 効率的ポーズ検出

    # アニメ特化調整
    anime_boost_factor: float = 1.2  # アニメ特化手法への重み増強


@dataclass
class EnsembleDetectionResult:
    """アンサンブル検出結果"""

    face_detections: List[FaceDetection]
    pose_result: PoseDetectionResult
    ensemble_confidence: float
    method_contributions: Dict[str, float]
    quality_metrics: Dict[str, float]
    processing_time: float


@dataclass
class VotingResult:
    """投票結果詳細"""

    selected_detection: FaceDetection
    vote_scores: Dict[str, float]
    consensus_level: float
    confidence_boost: float


class DetectionEnsembleSystem:
    """検出結果アンサンブルシステム"""

    def __init__(self, weights: Optional[EnsembleWeights] = None):
        self.logger = logging.getLogger(f"{__name__}.DetectionEnsembleSystem")
        self.weights = weights or EnsembleWeights()

        # 検出器の初期化
        self.face_detector = EnhancedFaceDetector()
        self.pose_detector = EnhancedPoseDetector()

        self.logger.info("検出アンサンブルシステム初期化完了")

    def detect_comprehensive_ensemble(
        self, image: np.ndarray, target_face_rate: float = 0.90, target_pose_rate: float = 0.80
    ) -> EnsembleDetectionResult:
        """包括的アンサンブル検出"""
        start_time = datetime.now()

        self.logger.info(f"アンサンブル検出開始 (目標: 顔{target_face_rate:.0%}, ポーズ{target_pose_rate:.0%})")

        # 1. 複数手法による顔検出
        face_detections_raw = self.face_detector.detect_faces_comprehensive(
            image, efficient_mode=False, target_detection_rate=target_face_rate
        )

        # 2. 重み付き投票による顔検出統合
        face_ensemble_result = self._ensemble_face_detections(face_detections_raw)

        # 3. 複数モードによるポーズ検出
        pose_high_quality = self.pose_detector.detect_pose_comprehensive(
            image, efficient_mode=False
        )
        pose_efficient = self.pose_detector.detect_pose_comprehensive(
            image, efficient_mode=True
        )

        # 4. ポーズ結果のアンサンブル
        pose_ensemble_result = self._ensemble_pose_detections(pose_high_quality, pose_efficient)

        # 5. 統合品質メトリクス計算
        quality_metrics = self._calculate_ensemble_quality_metrics(
            face_ensemble_result, pose_ensemble_result, face_detections_raw
        )

        # 6. 統合信頼度計算
        ensemble_confidence = self._calculate_ensemble_confidence(
            face_ensemble_result, pose_ensemble_result, quality_metrics
        )

        # 7. 手法寄与度分析
        method_contributions = self._analyze_method_contributions(face_detections_raw, pose_ensemble_result)

        processing_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            f"アンサンブル検出完了: 顔{len(face_ensemble_result)}件, "
            f"ポーズ{'検出' if pose_ensemble_result.detected else '未検出'}, "
            f"信頼度{ensemble_confidence:.3f} "
            f"({processing_time:.2f}秒)"
        )

        return EnsembleDetectionResult(
            face_detections=face_ensemble_result,
            pose_result=pose_ensemble_result,
            ensemble_confidence=ensemble_confidence,
            method_contributions=method_contributions,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
        )

    def _ensemble_face_detections(self, raw_detections: List[FaceDetection]) -> List[FaceDetection]:
        """顔検出結果のアンサンブル処理"""
        if not raw_detections:
            return []

        # 手法別グループ化
        method_groups = {}
        for detection in raw_detections:
            method = self._standardize_method_name(detection.method)
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(detection)

        # 重み付き投票によるベスト検出選択
        ensemble_detections = []

        # 空間的クラスタリング（重複除去）
        spatial_clusters = self._cluster_detections_spatially(raw_detections)

        for cluster in spatial_clusters:
            # クラスタ内での重み付き投票
            voting_result = self._weighted_voting_for_cluster(cluster)
            if voting_result:
                ensemble_detections.append(voting_result.selected_detection)

        # 信頼度順ソート
        ensemble_detections.sort(key=lambda x: x.confidence, reverse=True)

        self.logger.debug(
            f"顔検出アンサンブル: {len(raw_detections)}件 → {len(ensemble_detections)}件 "
            f"(手法: {list(method_groups.keys())})"
        )

        return ensemble_detections

    def _ensemble_pose_detections(
        self, high_quality: PoseDetectionResult, efficient: PoseDetectionResult
    ) -> PoseDetectionResult:
        """ポーズ検出結果のアンサンブル処理"""

        # 両方検出失敗の場合
        if not high_quality.detected and not efficient.detected:
            return high_quality  # デフォルトとして高品質結果を返す

        # 片方のみ検出成功の場合
        if high_quality.detected and not efficient.detected:
            return high_quality
        if efficient.detected and not high_quality.detected:
            return efficient

        # 両方検出成功の場合: 重み付き統合
        hq_weight = self.weights.pose_high_quality
        eff_weight = self.weights.pose_efficient

        # 統合信頼度計算
        ensemble_confidence = (high_quality.confidence * hq_weight) + (efficient.confidence * eff_weight)

        # 統合可視性スコア計算
        ensemble_visibility = (high_quality.visibility_score * hq_weight) + (
            efficient.visibility_score * eff_weight
        )

        # 統合完全性スコア計算
        ensemble_completeness = (high_quality.completeness_score * hq_weight) + (
            efficient.completeness_score * eff_weight
        )

        # より多くのキーポイントを持つ結果を優先
        if high_quality.keypoints_detected >= efficient.keypoints_detected:
            primary_result = high_quality
            ensemble_keypoints = high_quality.keypoints_detected
        else:
            primary_result = efficient
            ensemble_keypoints = efficient.keypoints_detected

        self.logger.debug(
            f"ポーズアンサンブル: HQ({high_quality.confidence:.3f}) + "
            f"EFF({efficient.confidence:.3f}) → {ensemble_confidence:.3f}"
        )

        return PoseDetectionResult(
            detected=True,
            landmarks=primary_result.landmarks,  # より良い結果のランドマークを使用
            visibility_score=ensemble_visibility,
            pose_category=primary_result.pose_category,
            completeness_score=ensemble_completeness,
            confidence=ensemble_confidence,
            keypoints_detected=ensemble_keypoints,
        )

    def _weighted_voting_for_cluster(self, cluster_detections: List[FaceDetection]) -> Optional[VotingResult]:
        """クラスタ内重み付き投票"""
        if not cluster_detections:
            return None

        vote_scores = {}
        method_weights = {
            "mediapipe": self.weights.face_mediapipe,
            "opencv_anime": self.weights.face_opencv_anime * self.weights.anime_boost_factor,
            "opencv_standard": self.weights.face_opencv_standard,
            "dlib": self.weights.face_dlib,
        }

        best_detection = None
        best_score = 0.0

        for detection in cluster_detections:
            method = self._standardize_method_name(detection.method)
            weight = method_weights.get(method, 0.1)  # 未知手法は低重み

            # 投票スコア = 信頼度 × 手法重み × アニメ特化ブースト
            vote_score = detection.confidence * weight

            # アニメ関連手法への追加ブースト
            if "anime" in detection.method.lower():
                vote_score *= self.weights.anime_boost_factor

            vote_scores[f"{method}_{id(detection)}"] = vote_score

            if vote_score > best_score:
                best_score = vote_score
                best_detection = detection

        if best_detection is None:
            return None

        # コンセンサスレベル計算（投票の一致度）
        total_votes = sum(vote_scores.values())
        consensus_level = best_score / total_votes if total_votes > 0 else 0.0

        # 信頼度ブースト計算（多手法一致による信頼度向上）
        method_diversity = len(set(self._standardize_method_name(d.method) for d in cluster_detections))
        confidence_boost = min(0.2, method_diversity * 0.05)  # 最大20%ブースト

        # 最終信頼度調整
        boosted_detection = FaceDetection(
            bbox=best_detection.bbox,
            confidence=min(1.0, best_detection.confidence + confidence_boost),
            method=f"ensemble_{best_detection.method}",
            landmarks=best_detection.landmarks,
        )

        return VotingResult(
            selected_detection=boosted_detection,
            vote_scores=vote_scores,
            consensus_level=consensus_level,
            confidence_boost=confidence_boost,
        )

    def _cluster_detections_spatially(
        self, detections: List[FaceDetection], overlap_threshold: float = 0.5
    ) -> List[List[FaceDetection]]:
        """空間的クラスタリング（重複検出のグループ化）"""
        if not detections:
            return []

        clusters = []
        used = set()

        for i, detection1 in enumerate(detections):
            if i in used:
                continue

            cluster = [detection1]
            used.add(i)

            for j, detection2 in enumerate(detections):
                if j in used or j <= i:
                    continue

                # IoU計算による重複判定
                iou = self._calculate_bbox_iou(detection1.bbox, detection2.bbox)
                if iou >= overlap_threshold:
                    cluster.append(detection2)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _calculate_bbox_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """バウンディングボックス間のIoU計算"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # 交差領域計算
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0.0

    def _standardize_method_name(self, method: str) -> str:
        """手法名の標準化"""
        method = method.lower()
        if "mediapipe" in method:
            return "mediapipe"
        elif "anime" in method:
            return "opencv_anime"
        elif "opencv" in method or "cascade" in method:
            return "opencv_standard"
        elif "dlib" in method:
            return "dlib"
        else:
            return "unknown"

    def _calculate_ensemble_quality_metrics(
        self,
        face_detections: List[FaceDetection],
        pose_result: PoseDetectionResult,
        raw_face_detections: List[FaceDetection],
    ) -> Dict[str, float]:
        """アンサンブル品質メトリクス計算"""

        # 顔検出品質
        face_quality = 0.0
        if face_detections:
            face_quality = sum(d.confidence for d in face_detections) / len(face_detections)

        # ポーズ検出品質
        pose_quality = pose_result.confidence if pose_result.detected else 0.0

        # 多様性指標（異なる手法の数）
        unique_methods = set(self._standardize_method_name(d.method) for d in raw_face_detections)
        diversity_score = len(unique_methods) / 4.0  # 最大4手法

        # 統合効果指標（アンサンブル前後の改善度）
        raw_best_confidence = max((d.confidence for d in raw_face_detections), default=0.0)
        ensemble_best_confidence = max((d.confidence for d in face_detections), default=0.0)
        improvement_ratio = (
            ensemble_best_confidence / raw_best_confidence if raw_best_confidence > 0 else 1.0
        )

        return {
            "face_quality": face_quality,
            "pose_quality": pose_quality,
            "diversity_score": diversity_score,
            "improvement_ratio": improvement_ratio,
            "detection_count": len(face_detections),
            "method_diversity": len(unique_methods),
        }

    def _calculate_ensemble_confidence(
        self,
        face_detections: List[FaceDetection],
        pose_result: PoseDetectionResult,
        quality_metrics: Dict[str, float],
    ) -> float:
        """統合信頼度計算"""

        # 基本信頼度（顔・ポーズの重み付き平均）
        face_confidence = quality_metrics["face_quality"]
        pose_confidence = quality_metrics["pose_quality"]

        # Week 3アニメ特化重み（顔重視）
        face_weight = 0.6  # アニメでは顔が最重要
        pose_weight = 0.4

        base_confidence = (face_confidence * face_weight) + (pose_confidence * pose_weight)

        # 多様性ボーナス（複数手法の一致による信頼度向上）
        diversity_bonus = quality_metrics["diversity_score"] * 0.1

        # 改善ボーナス（アンサンブル効果による向上）
        improvement_bonus = min(0.1, (quality_metrics["improvement_ratio"] - 1.0) * 0.2)

        # 統合信頼度
        ensemble_confidence = min(1.0, base_confidence + diversity_bonus + improvement_bonus)

        return ensemble_confidence

    def _analyze_method_contributions(
        self, raw_face_detections: List[FaceDetection], pose_result: PoseDetectionResult
    ) -> Dict[str, float]:
        """手法別寄与度分析"""

        contributions = {}

        # 顔検出手法別寄与度
        method_counts = {}
        total_confidence = 0.0

        for detection in raw_face_detections:
            method = self._standardize_method_name(detection.method)
            if method not in method_counts:
                method_counts[method] = {"count": 0, "confidence_sum": 0.0}

            method_counts[method]["count"] += 1
            method_counts[method]["confidence_sum"] += detection.confidence
            total_confidence += detection.confidence

        # 寄与度正規化
        for method, data in method_counts.items():
            if total_confidence > 0:
                contribution = data["confidence_sum"] / total_confidence
            else:
                contribution = data["count"] / len(raw_face_detections)

            contributions[f"face_{method}"] = contribution

        # ポーズ検出寄与度
        contributions["pose_detection"] = pose_result.confidence if pose_result.detected else 0.0

        return contributions


def main():
    """デモ実行"""
    import argparse

    parser = argparse.ArgumentParser(description="検出アンサンブルシステムテスト")
    parser.add_argument("image_path", help="テスト画像パス")
    parser.add_argument("--face-target", type=float, default=0.90, help="顔検出目標率")
    parser.add_argument("--pose-target", type=float, default=0.80, help="ポーズ検出目標率")
    args = parser.parse_args()

    # 画像読み込み
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"画像読み込み失敗: {args.image_path}")
        return

    # アンサンブル検出実行
    ensemble = DetectionEnsembleSystem()
    result = ensemble.detect_comprehensive_ensemble(
        image, target_face_rate=args.face_target, target_pose_rate=args.pose_target
    )

    # 結果表示
    print(f"\n=== アンサンブル検出結果 ===")
    print(f"顔検出: {len(result.face_detections)}件")
    print(f"ポーズ検出: {'成功' if result.pose_result.detected else '失敗'}")
    print(f"統合信頼度: {result.ensemble_confidence:.3f}")
    print(f"処理時間: {result.processing_time:.2f}秒")

    print(f"\n=== 品質メトリクス ===")
    for key, value in result.quality_metrics.items():
        print(f"{key}: {value:.3f}")

    print(f"\n=== 手法寄与度 ===")
    for method, contribution in result.method_contributions.items():
        print(f"{method}: {contribution:.3f}")


if __name__ == "__main__":
    main()