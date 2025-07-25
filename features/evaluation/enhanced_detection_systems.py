#!/usr/bin/env python3
"""
強化検出システム (Phase A2)
最高品質の顔検出・ポーズ検出を実現

品質重視の設計方針:
- 処理時間: 10-12秒/画像（品質最優先）
- 顔検出率: 90%以上
- ポーズ検出率: 80%以上
- 複数手法統合による高精度実現
"""

import numpy as np
import cv2

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Dict, List, NamedTuple, Optional, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.anime_image_preprocessor import AnimeImagePreprocessor

# MediaPipeインポート（利用可能な場合）
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipeが利用できません - OpenCVのみで動作")

# dlibインポート（利用可能な場合）
try:
    import dlib

    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("dlibが利用できません - MediaPipe+OpenCVで動作")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """顔検出結果データクラス"""

    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    method: str  # "mediapipe", "opencv", "dlib"
    landmarks: Optional[np.ndarray] = None


@dataclass
class PoseDetectionResult:
    """ポーズ検出結果データクラス"""

    detected: bool
    landmarks: Optional[np.ndarray]
    visibility_score: float
    pose_category: str
    completeness_score: float
    confidence: float
    keypoints_detected: int


@dataclass
class EnhancedDetectionReport:
    """強化検出システム総合レポート"""

    face_detections: List[FaceDetection]
    pose_result: PoseDetectionResult
    overall_confidence: float
    processing_time: float
    detection_summary: Dict[str, float]


class EnhancedFaceDetector:
    """最高品質多手法統合顔検出システム"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedFaceDetector")

        # プロジェクトルートパス
        project_root = Path(__file__).parent.parent.parent

        # アニメ画像前処理システム
        self.preprocessor = AnimeImagePreprocessor()

        # MediaPipe顔検出初期化
        self.mediapipe_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.1  # 遠距離モデル（高精度）  # 閾値を大幅に下げて検出率向上
                )
                self.logger.info("MediaPipe顔検出初期化完了")
            except Exception as e:
                self.logger.warning(f"MediaPipe初期化失敗: {e}")

        # OpenCV Cascade分類器初期化
        self.cascade_detectors = {}
        cascade_files = {
            "frontal": "haarcascade_frontalface_default.xml",
            "profile": "haarcascade_profileface.xml",
            "anime": "lbpcascade_animeface.xml",  # アニメ顔用専用カスケード
        }

        for name, filename in cascade_files.items():
            try:
                if name == "anime":
                    # アニメ顔専用カスケードのパス
                    cascade_path = str(project_root / "models" / "cascades" / filename)
                else:
                    # 標準OpenCVカスケードのパス
                    cascade_path = cv2.data.haarcascades + filename

                if os.path.exists(cascade_path):
                    self.cascade_detectors[name] = cv2.CascadeClassifier(cascade_path)
                    self.logger.info(f"OpenCV {name} Cascade初期化完了")
                else:
                    self.logger.warning(f"Cascadeファイル未発見: {cascade_path}")
            except Exception as e:
                self.logger.warning(f"OpenCV {name} Cascade初期化失敗: {e}")

        # dlib顔検出器初期化
        self.dlib_detector = None
        if DLIB_AVAILABLE:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.logger.info("dlib顔検出器初期化完了")
            except Exception as e:
                self.logger.warning(f"dlib初期化失敗: {e}")

    def detect_faces_comprehensive(
        self, image: np.ndarray, efficient_mode: bool = False, target_detection_rate: float = 0.90
    ) -> List[FaceDetection]:
        """包括的多手法顔検出（前処理統合版）"""
        start_time = datetime.now()
        all_detections = []

        # 0. アニメ画像前処理適用（軽量モード対応）
        enhanced_image = self.preprocessor.enhance_for_face_detection(
            image, lightweight_mode=efficient_mode
        )

        if efficient_mode:
            # 効率モード: 段階的検出（目標達成で早期終了）
            return self._efficient_detection_pipeline(enhanced_image, image, target_detection_rate)
        else:
            # 高品質モード: 全手法実行（従来版）
            return self._full_detection_pipeline(enhanced_image, image)

    def _efficient_detection_pipeline(
        self, enhanced_image: np.ndarray, original_image: np.ndarray, target_rate: float
    ) -> List[FaceDetection]:
        """効率的検出パイプライン（段階的実行・早期終了）"""
        start_time = datetime.now()
        all_detections = []
        detection_methods = []

        # 優先度1: MediaPipe（最高精度）
        if self.mediapipe_detector:
            mp_detections = self._detect_mediapipe(enhanced_image)
            all_detections.extend(mp_detections)
            detection_methods.append(f"MediaPipe: {len(mp_detections)}件")

            # 十分な検出があれば早期終了を検討
            if len(mp_detections) >= 1:  # 1件以上検出があれば次の段階へ
                merged = self._merge_detections(all_detections)
                if self._estimate_detection_quality(merged) >= target_rate:
                    self.logger.info(f"早期終了: MediaPipeで目標達成 ({len(merged)}件)")
                    return merged

        # 優先度2: アニメ専用カスケード（アニメ特化）
        if "anime" in self.cascade_detectors:
            anime_detections = self._detect_opencv_cascade(
                enhanced_image, self.cascade_detectors["anime"], "anime"
            )
            all_detections.extend(anime_detections)
            detection_methods.append(f"Anime Cascade: {len(anime_detections)}件")

            merged = self._merge_detections(all_detections)
            if self._estimate_detection_quality(merged) >= target_rate:
                self.logger.info(f"早期終了: Anime Cascadeで目標達成 ({len(merged)}件)")
                return merged

        # 優先度3: マルチスケール検出（3スケールのみ）
        if "anime" in self.cascade_detectors:
            multi_scale_detections = self._detect_multi_scale_anime_efficient(original_image)
            all_detections.extend(multi_scale_detections)
            detection_methods.append(f"Multi-Scale: {len(multi_scale_detections)}件")

            merged = self._merge_detections(all_detections)
            if self._estimate_detection_quality(merged) >= target_rate:
                self.logger.info(f"早期終了: Multi-Scaleで目標達成 ({len(merged)}件)")
                return merged

        # 優先度4: dlib（補完用）
        if self.dlib_detector:
            dlib_detections = self._detect_dlib(enhanced_image)
            all_detections.extend(dlib_detections)
            detection_methods.append(f"dlib: {len(dlib_detections)}件")

        # 最終統合
        merged_detections = self._merge_detections(all_detections)

        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            f"効率的顔検出完了: {len(merged_detections)}件検出 "
            f"（{', '.join(detection_methods)}）"
            f"（処理時間: {processing_time:.2f}秒）"
        )

        return merged_detections

    def _full_detection_pipeline(
        self, enhanced_image: np.ndarray, original_image: np.ndarray
    ) -> List[FaceDetection]:
        """高品質検出パイプライン（全手法実行）"""
        start_time = datetime.now()
        all_detections = []

        # 1. MediaPipe検出（最高精度）
        if self.mediapipe_detector:
            mp_detections = self._detect_mediapipe(enhanced_image)
            all_detections.extend(mp_detections)
            self.logger.debug(f"MediaPipe検出: {len(mp_detections)}件")

        # 2. OpenCV Cascade検出（複数パターン）
        for detector_name, detector in self.cascade_detectors.items():
            cv_detections = self._detect_opencv_cascade(enhanced_image, detector, detector_name)
            all_detections.extend(cv_detections)
            self.logger.debug(f"OpenCV {detector_name}検出: {len(cv_detections)}件")

        # 3. dlib検出（補完用）
        if self.dlib_detector:
            dlib_detections = self._detect_dlib(enhanced_image)
            all_detections.extend(dlib_detections)
            self.logger.debug(f"dlib検出: {len(dlib_detections)}件")

        # 4. マルチスケール検出（全5スケール）
        if "anime" in self.cascade_detectors:
            multi_scale_detections = self._detect_multi_scale_anime(original_image)
            all_detections.extend(multi_scale_detections)
            self.logger.debug(f"マルチスケール検出: {len(multi_scale_detections)}件")

        # 5. 重複除去・統合
        merged_detections = self._merge_detections(all_detections)

        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            f"高品質顔検出完了: {len(merged_detections)}件検出 " f"（処理時間: {processing_time:.2f}秒）"
        )

        return merged_detections

    def _estimate_detection_quality(self, detections: List[FaceDetection]) -> float:
        """検出品質の推定（早期終了判定用）"""
        if not detections:
            return 0.0

        # 信頼度の重み付き平均で品質推定
        total_confidence = sum(det.confidence for det in detections)
        avg_confidence = total_confidence / len(detections)

        # 検出数による補正（1件=0.8, 2件以上=1.0）
        count_factor = min(1.0, len(detections) * 0.8)

        return avg_confidence * count_factor

    def _detect_multi_scale_anime_efficient(self, image: np.ndarray) -> List[FaceDetection]:
        """軽量マルチスケールアニメ顔検出（3スケールのみ）"""
        detections = []

        if "anime" not in self.cascade_detectors:
            return detections

        try:
            # 軽量マルチスケール画像生成（3スケールのみ）
            multi_scale_images = self.preprocessor.create_multi_scale_versions(
                image, lightweight_mode=True
            )

            anime_detector = self.cascade_detectors["anime"]

            for scale_data in multi_scale_images:
                scale = scale_data["scale"]
                scaled_image = scale_data["image"]

                # グレースケール変換
                gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

                # アニメ顔特化検出
                faces = anime_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.02,
                    minNeighbors=1,
                    minSize=(8, 8),
                    maxSize=(600, 600),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                # 元画像座標系に変換
                for x, y, w, h in faces:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)

                    # 境界チェック
                    orig_x = max(0, orig_x)
                    orig_y = max(0, orig_y)
                    orig_w = min(orig_w, image.shape[1] - orig_x)
                    orig_h = min(orig_h, image.shape[0] - orig_y)

                    scale_confidence = self._calculate_scale_confidence(scale, w, h)

                    detections.append(
                        FaceDetection(
                            bbox=(orig_x, orig_y, orig_w, orig_h),
                            confidence=0.90 * scale_confidence,
                            method=f"anime_efficient_{scale:.2f}",
                        )
                    )

        except Exception as e:
            self.logger.warning(f"軽量マルチスケール検出エラー: {e}")

        return detections

    def _detect_mediapipe(self, image: np.ndarray) -> List[FaceDetection]:
        """MediaPipe顔検出"""
        detections = []
        try:
            # RGB変換（MediaPipe要件）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mediapipe_detector.process(rgb_image)

            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox_data = detection.location_data.relative_bounding_box

                    # 相対座標を絶対座標に変換
                    x = int(bbox_data.xmin * w)
                    y = int(bbox_data.ymin * h)
                    width = int(bbox_data.width * w)
                    height = int(bbox_data.height * h)

                    # 境界チェック
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    confidence = detection.score[0] if detection.score else 0.5

                    detections.append(
                        FaceDetection(
                            bbox=(x, y, width, height), confidence=confidence, method="mediapipe"
                        )
                    )

        except Exception as e:
            self.logger.warning(f"MediaPipe検出エラー: {e}")

        return detections

    def _detect_opencv_cascade(
        self, image: np.ndarray, detector: cv2.CascadeClassifier, detector_name: str
    ) -> List[FaceDetection]:
        """OpenCV Cascade検出"""
        detections = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # アニメ顔用に最適化されたパラメータ
            if detector_name == "anime":
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.03,  # アニメ顔用により細かいスケール
                    minNeighbors=1,  # アニメ顔用極寛容設定
                    minSize=(10, 10),  # 非常に小さな顔も検出
                    maxSize=(500, 500),  # 大きな顔も対応
                )
            else:
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # より細かいスケールで検出率向上
                    minNeighbors=2,  # より寛容な設定で検出率向上
                    minSize=(20, 20),  # より小さな顔も検出
                )

            for x, y, w, h in faces:
                # 信頼度は検出サイズベースで推定
                face_area = w * h
                image_area = image.shape[0] * image.shape[1]
                size_confidence = min(1.0, face_area / (image_area * 0.01))

                # アニメ顔専用カスケードの場合は信頼度を高く設定
                if detector_name == "anime":
                    base_confidence = 0.85  # アニメ顔専用なので高信頼度
                else:
                    base_confidence = 0.7  # 標準信頼度

                detections.append(
                    FaceDetection(
                        bbox=(x, y, w, h),
                        confidence=base_confidence * size_confidence,
                        method=f"opencv_{detector_name}",
                    )
                )

        except Exception as e:
            self.logger.warning(f"OpenCV {detector_name}検出エラー: {e}")

        return detections

    def _detect_dlib(self, image: np.ndarray) -> List[FaceDetection]:
        """dlib顔検出"""
        detections = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.dlib_detector(gray)

            for face in faces:
                x = face.left()
                y = face.top()
                w = face.width()
                h = face.height()

                # dlib confidence estimation based on detection quality
                confidence = 0.8  # dlibの基本信頼度

                detections.append(
                    FaceDetection(bbox=(x, y, w, h), confidence=confidence, method="dlib")
                )

        except Exception as e:
            self.logger.warning(f"dlib検出エラー: {e}")

        return detections

    def _detect_multi_scale_anime(self, image: np.ndarray) -> List[FaceDetection]:
        """マルチスケールアニメ顔検出（複数解像度での検出統合）"""
        detections = []

        if "anime" not in self.cascade_detectors:
            return detections

        try:
            # マルチスケール画像生成
            multi_scale_images = self.preprocessor.create_multi_scale_versions(image)

            anime_detector = self.cascade_detectors["anime"]

            for scale_data in multi_scale_images:
                scale = scale_data["scale"]
                scaled_image = scale_data["image"]

                # グレースケール変換
                gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

                # アニメ顔特化検出
                faces = anime_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.02,  # より細かいスケール検出
                    minNeighbors=1,  # 最小近傍（最寛容設定）
                    minSize=(8, 8),  # 極小顔も検出
                    maxSize=(600, 600),  # 大きな顔も対応
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                # 元画像座標系に変換
                for x, y, w, h in faces:
                    # スケール逆変換
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)

                    # 境界チェック
                    orig_x = max(0, orig_x)
                    orig_y = max(0, orig_y)
                    orig_w = min(orig_w, image.shape[1] - orig_x)
                    orig_h = min(orig_h, image.shape[0] - orig_y)

                    # スケール別信頼度計算
                    scale_confidence = self._calculate_scale_confidence(scale, w, h)

                    detections.append(
                        FaceDetection(
                            bbox=(orig_x, orig_y, orig_w, orig_h),
                            confidence=0.90 * scale_confidence,  # アニメ顔専用なので高信頼度
                            method=f"anime_multiscale_{scale:.2f}",
                        )
                    )

                    self.logger.debug(
                        f"マルチスケール検出: スケール{scale:.2f}, 座標({orig_x},{orig_y},{orig_w},{orig_h})"
                    )

        except Exception as e:
            self.logger.warning(f"マルチスケール検出エラー: {e}")

        return detections

    def _calculate_scale_confidence(self, scale: float, width: int, height: int) -> float:
        """スケール別信頼度計算"""
        # 顔サイズによる信頼度調整
        face_area = width * height

        # 最適スケール範囲（0.75-1.25倍が最も信頼性高い）
        if 0.75 <= scale <= 1.25:
            scale_factor = 1.0
        elif 0.5 <= scale < 0.75 or 1.25 < scale <= 1.5:
            scale_factor = 0.9
        else:
            scale_factor = 0.8

        # 顔サイズによる信頼度（中程度サイズが最適）
        if 30 * 30 <= face_area <= 200 * 200:
            size_factor = 1.0
        elif 15 * 15 <= face_area < 30 * 30 or 200 * 200 < face_area <= 400 * 400:
            size_factor = 0.85
        else:
            size_factor = 0.7

        return scale_factor * size_factor

    def _merge_detections(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """重複除去・検出結果統合"""
        if not detections:
            return []

        # IoU（Intersection over Union）による重複除去
        merged = []

        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            is_duplicate = False

            for existing in merged:
                iou = self._calculate_bbox_iou(detection.bbox, existing.bbox)
                if iou > 0.5:  # 50%以上重複は同一顔とみなす
                    # より高い信頼度の検出を保持
                    if detection.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(detection)
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(detection)

        return merged

    def _calculate_bbox_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """境界ボックスのIoU計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 交差領域計算
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class EnhancedPoseDetector:
    """包括的最高品質ポーズ検出システム"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedPoseDetector")

        # MediaPipe Pose初期化（Week 2最適化設定）
        self.mediapipe_pose_high = None
        self.mediapipe_pose_fast = None
        if MEDIAPIPE_AVAILABLE:
            try:
                # 高精度モデル（複雑な姿勢用）
                self.mediapipe_pose_high = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,  # 最高精度モデル
                    enable_segmentation=False,  # Week 2最適化: セグメンテーション無効
                    min_detection_confidence=0.05,  # Week 2最適化: 0.1→0.05に緩和
                    min_tracking_confidence=0.05,
                )

                # 高速モデル（高速処理用）
                self.mediapipe_pose_fast = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,  # Week 2最適化: 軽量高速モデル併用
                    enable_segmentation=False,  # セグメンテーション無効
                    min_detection_confidence=0.05,
                    min_tracking_confidence=0.05,
                )

                self.logger.info("MediaPipe Pose（Week 2最適化：併用モデル）初期化完了")
            except Exception as e:
                self.logger.warning(f"MediaPipe Pose初期化失敗: {e}")

        # ポーズカテゴリ分類用の設定
        self.pose_categories = {
            "standing": "立位",
            "sitting": "座位",
            "lying": "横臥位",
            "action": "アクション",
            "profile": "横向き",
            "unknown": "不明",
        }

    def detect_pose_comprehensive(
        self, image: np.ndarray, efficient_mode: bool = False
    ) -> PoseDetectionResult:
        """包括的高品質ポーズ検出（Week 2最適化：併用モデル対応）"""
        start_time = datetime.now()

        if not self.mediapipe_pose_high and not self.mediapipe_pose_fast:
            return PoseDetectionResult(
                detected=False,
                landmarks=None,
                visibility_score=0.0,
                pose_category="unknown",
                completeness_score=0.0,
                confidence=0.0,
                keypoints_detected=0,
            )

        try:
            # RGB変換（MediaPipe要件）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Week 2最適化: 併用モデル戦略
            results = None
            model_used = "none"

            if efficient_mode and self.mediapipe_pose_fast:
                # 高速モード: 軽量モデル優先
                results = self.mediapipe_pose_fast.process(rgb_image)
                model_used = "fast"

                # 高速モデルで検出できない場合は高精度モデルを試行
                if not results.pose_landmarks and self.mediapipe_pose_high:
                    results = self.mediapipe_pose_high.process(rgb_image)
                    model_used = "high_fallback"
            else:
                # 高精度モード: 高精度モデル優先
                if self.mediapipe_pose_high:
                    results = self.mediapipe_pose_high.process(rgb_image)
                    model_used = "high"
                elif self.mediapipe_pose_fast:
                    results = self.mediapipe_pose_fast.process(rgb_image)
                    model_used = "fast_fallback"

            if not results or not results.pose_landmarks:
                self.logger.debug(f"ポーズランドマーク未検出 (model: {model_used})")
                return PoseDetectionResult(
                    detected=False,
                    landmarks=None,
                    visibility_score=0.0,
                    pose_category="unknown",
                    completeness_score=0.0,
                    confidence=0.0,
                    keypoints_detected=0,
                )

            # Week 2最適化: 部分ポーズ対応ランドマーク分析
            landmarks = results.pose_landmarks
            visibility_score = self._calculate_visibility_score_optimized(landmarks)
            pose_category = self._classify_pose_anime_optimized(landmarks)
            completeness_score = self._evaluate_partial_pose_completeness(landmarks)
            keypoints_detected = self._count_visible_keypoints_optimized(landmarks)

            # Week 2最適化: 部分ポーズでも高信頼度を維持
            confidence = self._calculate_optimized_confidence(
                visibility_score, completeness_score, keypoints_detected
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"ポーズ検出完了: カテゴリ={pose_category}, "
                f"可視性={visibility_score:.3f}, "
                f"完全性={completeness_score:.3f}, "
                f"キーポイント={keypoints_detected}, "
                f"モデル={model_used} "
                f"（処理時間: {processing_time:.2f}秒）"
            )

            return PoseDetectionResult(
                detected=True,
                landmarks=landmarks,
                visibility_score=visibility_score,
                pose_category=pose_category,
                completeness_score=completeness_score,
                confidence=confidence,
                keypoints_detected=keypoints_detected,
            )

        except Exception as e:
            self.logger.error(f"ポーズ検出エラー: {e}")
            return PoseDetectionResult(
                detected=False,
                landmarks=None,
                visibility_score=0.0,
                pose_category="unknown",
                completeness_score=0.0,
                confidence=0.0,
                keypoints_detected=0,
            )

    def _calculate_visibility_score(self, landmarks) -> float:
        """キーポイント可視性スコア計算"""
        if not landmarks or not landmarks.landmark:
            return 0.0

        visible_count = 0
        total_count = len(landmarks.landmark)

        for landmark in landmarks.landmark:
            # MediaPipeの可視性閾値（0.3以上を可視とみなす - 緩和）
            if landmark.visibility > 0.3:
                visible_count += 1

        return visible_count / total_count if total_count > 0 else 0.0

    def _classify_pose(self, landmarks) -> str:
        """ポーズカテゴリ分類"""
        if not landmarks or not landmarks.landmark:
            return "unknown"

        try:
            # 主要キーポイントのインデックス（MediaPipe Pose）
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            left_hip = landmarks.landmark[23]
            right_hip = landmarks.landmark[24]
            left_knee = landmarks.landmark[25]
            right_knee = landmarks.landmark[26]
            left_ankle = landmarks.landmark[27]
            right_ankle = landmarks.landmark[28]

            # 肩と腰の中点計算
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2

            # 姿勢判定ロジック
            torso_angle = abs(shoulder_y - hip_y)
            leg_bend = (
                abs(hip_y - knee_y) / abs(knee_y - ankle_y) if abs(knee_y - ankle_y) > 0.01 else 1.0
            )

            # 立位判定
            if torso_angle > 0.2 and leg_bend > 0.8:
                return "standing"

            # 座位判定
            elif torso_angle > 0.15 and leg_bend < 0.6:
                return "sitting"

            # 横臥位判定
            elif torso_angle < 0.1:
                return "lying"

            # アクション判定（複雑なポーズ）
            elif leg_bend < 0.4 or torso_angle > 0.3:
                return "action"

            else:
                return "unknown"

        except Exception as e:
            self.logger.warning(f"ポーズ分類エラー: {e}")
            return "unknown"

    def _evaluate_pose_completeness(self, landmarks) -> float:
        """ポーズ完全性評価"""
        if not landmarks or not landmarks.landmark:
            return 0.0

        # 重要な身体部位の重み
        body_parts_weights = {
            # 頭部 (20%)
            "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # 上半身 (30%)
            "upper_body": [11, 12, 13, 14, 15, 16],
            # 腰部 (20%)
            "torso": [23, 24],
            # 下半身 (30%)
            "lower_body": [25, 26, 27, 28, 29, 30, 31, 32],
        }

        weights = {"head": 0.2, "upper_body": 0.3, "torso": 0.2, "lower_body": 0.3}

        total_score = 0.0

        for part_name, indices in body_parts_weights.items():
            part_visibility = 0.0
            valid_count = 0

            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    if landmark.visibility > 0.3:  # 可視性閾値
                        part_visibility += landmark.visibility
                        valid_count += 1

            if valid_count > 0:
                part_score = part_visibility / valid_count
                total_score += part_score * weights[part_name]

        return min(1.0, total_score)

    def _count_visible_keypoints(self, landmarks) -> int:
        """可視キーポイント数カウント"""
        if not landmarks or not landmarks.landmark:
            return 0

        visible_count = 0
        for landmark in landmarks.landmark:
            if landmark.visibility > 0.3:  # 可視性閾値を緩和
                visible_count += 1

        return visible_count

    def _calculate_visibility_score_optimized(self, landmarks) -> float:
        """最適化されたキーポイント可視性スコア計算（Week 2：部分ポーズ対応）"""
        if not landmarks or not landmarks.landmark:
            return 0.0

        visible_count = 0
        total_count = len(landmarks.landmark)

        for landmark in landmarks.landmark:
            # Week 2最適化: 可視性閾値をさらに緩和（0.3→0.2）
            if landmark.visibility > 0.2:
                visible_count += 1

        return visible_count / total_count if total_count > 0 else 0.0

    def _classify_pose_anime_optimized(self, landmarks) -> str:
        """アニメ特化ポーズ分類（Week 2最適化：部分ポーズ対応）"""
        if not landmarks or not landmarks.landmark:
            return "unknown"

        try:
            # Week 2最適化: 上半身キーポイントのみで判定
            # 肩・肘・手首の検出状況を確認
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            left_elbow = landmarks.landmark[13]
            right_elbow = landmarks.landmark[14]
            left_wrist = landmarks.landmark[15]
            right_wrist = landmarks.landmark[16]

            # 上半身キーポイントの可視性チェック
            upper_body_visible = 0
            upper_body_points = [
                left_shoulder,
                right_shoulder,
                left_elbow,
                right_elbow,
                left_wrist,
                right_wrist,
            ]

            for point in upper_body_points:
                if point.visibility > 0.2:  # Week 2最適化: 緩和された閾値
                    upper_body_visible += 1

            # Week 2最適化: 3点以上検出で有効なポーズとして判定
            if upper_body_visible >= 3:
                # 肩の位置関係から基本姿勢を推定
                if left_shoulder.visibility > 0.2 and right_shoulder.visibility > 0.2:
                    shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)

                    if shoulder_y_diff < 0.05:
                        return "standing"  # 肩が水平→立位
                    elif shoulder_y_diff > 0.15:
                        return "profile"  # 肩の高低差大→横向き
                    else:
                        return "sitting"  # 中間→座位
                else:
                    return "partial_pose"  # Week 2新カテゴリ: 部分ポーズ

            # 下半身も確認（補助的）
            try:
                left_hip = landmarks.landmark[23]
                right_hip = landmarks.landmark[24]

                if left_hip.visibility > 0.2 or right_hip.visibility > 0.2:
                    return "action"  # 下半身も見える→アクション系
            except (IndexError, AttributeError):
                pass

            return "upper_body_only"  # Week 2新カテゴリ: 上半身のみ

        except Exception as e:
            self.logger.warning(f"アニメ特化ポーズ分類エラー: {e}")
            return "unknown"

    def _evaluate_partial_pose_completeness(self, landmarks) -> float:
        """部分ポーズ完全性評価（Week 2最適化：上半身重視）"""
        if not landmarks or not landmarks.landmark:
            return 0.0

        # Week 2最適化: 部分ポーズに特化した重み配分
        body_parts_weights = {
            # 頭部 (30% - 重要度アップ)
            "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # 上半身 (50% - 最重要)
            "upper_body": [11, 12, 13, 14, 15, 16],
            # 下半身 (20% - 重要度ダウン、部分的でもOK)
            "lower_body": [23, 24, 25, 26, 27, 28],
        }

        weights = {"head": 0.3, "upper_body": 0.5, "lower_body": 0.2}

        total_score = 0.0

        for part_name, indices in body_parts_weights.items():
            part_visibility = 0.0
            valid_count = 0

            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    if landmark.visibility > 0.2:  # Week 2最適化: 緩和された閾値
                        part_visibility += landmark.visibility
                        valid_count += 1

            if valid_count > 0:
                part_score = part_visibility / valid_count
                total_score += part_score * weights[part_name]
            elif part_name == "lower_body":
                # Week 2最適化: 下半身が見えなくても減点しない
                total_score += 0.5 * weights[part_name]  # 部分的なスコアを付与

        return min(1.0, total_score)

    def _count_visible_keypoints_optimized(self, landmarks) -> int:
        """最適化されたキーポイント数カウント（Week 2：緩和された閾値）"""
        if not landmarks or not landmarks.landmark:
            return 0

        visible_count = 0
        for landmark in landmarks.landmark:
            if landmark.visibility > 0.2:  # Week 2最適化: 0.3→0.2に緩和
                visible_count += 1

        return visible_count

    def _calculate_optimized_confidence(
        self, visibility_score: float, completeness_score: float, keypoints_detected: int
    ) -> float:
        """最適化された信頼度計算（Week 2：部分ポーズでも高信頼度）"""

        # Week 2最適化: 3点以上検出で基本信頼度を確保
        if keypoints_detected >= 3:
            base_confidence = 0.6  # 最低信頼度を引き上げ
        else:
            base_confidence = 0.3

        # 可視性・完全性スコアによる補正
        combined_score = (visibility_score + completeness_score) / 2

        # Week 2最適化: より寛容な信頼度計算
        final_confidence = base_confidence + (combined_score * 0.4)

        return min(1.0, final_confidence)


def main():
    """テスト実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="強化検出システムテスト")
    parser.add_argument("--image", "-i", required=True, help="テスト画像パス")
    parser.add_argument("--face-only", action="store_true", help="顔検出のみ実行")
    parser.add_argument("--pose-only", action="store_true", help="ポーズ検出のみ実行")

    args = parser.parse_args()

    # 画像読み込み
    image = cv2.imread(args.image)
    if image is None:
        print(f"画像読み込み失敗: {args.image}")
        return 1

    print(f"🔍 強化検出システムテスト: {args.image}")
    print("=" * 60)

    # 顔検出テスト
    if not args.pose_only:
        print("👤 顔検出テスト実行中...")
        face_detector = EnhancedFaceDetector()
        face_detections = face_detector.detect_faces_comprehensive(image)

        print(f"顔検出結果: {len(face_detections)}件")
        for i, detection in enumerate(face_detections):
            print(
                f"  顔{i+1}: 手法={detection.method}, "
                f"信頼度={detection.confidence:.3f}, "
                f"位置={detection.bbox}"
            )

    # ポーズ検出テスト
    if not args.face_only:
        print("\n🤸 ポーズ検出テスト実行中...")
        pose_detector = EnhancedPoseDetector()
        pose_result = pose_detector.detect_pose_comprehensive(image)

        print(f"ポーズ検出結果:")
        print(f"  検出成功: {pose_result.detected}")
        if pose_result.detected:
            print(f"  カテゴリ: {pose_result.pose_category}")
            print(f"  可視性スコア: {pose_result.visibility_score:.3f}")
            print(f"  完全性スコア: {pose_result.completeness_score:.3f}")
            print(f"  総合信頼度: {pose_result.confidence:.3f}")
            print(f"  可視キーポイント: {pose_result.keypoints_detected}/33")

    return 0


if __name__ == "__main__":
    exit(main())
