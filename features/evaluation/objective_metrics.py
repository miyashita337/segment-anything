#!/usr/bin/env python3
"""
客観的3指標システム (PLA/SCI/PLE) 実装
完全自動化・再現可能な品質評価システム

- PLA (Pixel-Level Accuracy): IoUベースのピクセル精度測定
- SCI (Semantic Completeness Index): キャラクター構造の意味的完全性
- PLE (Progressive Learning Efficiency): 継続的改善効率測定
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available - SCI will use fallback implementation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectiveMetricResult:
    """客観的指標の結果"""

    metric_name: str
    value: float
    confidence: float
    status: str  # passed, failed, error
    details: Dict[str, Any]
    timestamp: str


class PLACalculator:
    """Pixel-Level Accuracy 計算器"""

    def __init__(self):
        """初期化"""
        self.threshold = 0.75  # PLA成功閾値

    def calculate(
        self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray
    ) -> ObjectiveMetricResult:
        """
        PLA計算（IoUベース）

        Args:
            predicted_mask: 予測マスク (0-255 または binary)
            ground_truth_mask: 正解マスク (0-255 または binary)

        Returns:
            ObjectiveMetricResult: PLA結果
        """
        try:
            # バイナリマスクに正規化
            pred_binary = self._normalize_mask(predicted_mask)
            gt_binary = self._normalize_mask(ground_truth_mask)

            # IoU計算
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()

            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
                confidence = 1.0
            else:
                iou = float(intersection) / float(union)
                confidence = self._calculate_confidence(pred_binary, gt_binary, iou)

            # 追加詳細情報
            details = {
                "iou": iou,
                "intersection": int(intersection),
                "union": int(union),
                "predicted_pixels": int(pred_binary.sum()),
                "ground_truth_pixels": int(gt_binary.sum()),
                "precision": float(intersection) / float(pred_binary.sum())
                if pred_binary.sum() > 0
                else 0.0,
                "recall": float(intersection) / float(gt_binary.sum())
                if gt_binary.sum() > 0
                else 0.0,
                "dice_coefficient": 2.0
                * float(intersection)
                / float(pred_binary.sum() + gt_binary.sum())
                if (pred_binary.sum() + gt_binary.sum()) > 0
                else 0.0,
            }

            status = "passed" if iou >= self.threshold else "failed"

            return ObjectiveMetricResult(
                metric_name="PLA (Pixel-Level Accuracy)",
                value=iou,
                confidence=confidence,
                status=status,
                details=details,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"PLA計算エラー: {e}")
            return ObjectiveMetricResult(
                metric_name="PLA (Pixel-Level Accuracy)",
                value=0.0,
                confidence=0.0,
                status="error",
                details={"error": str(e)},
                timestamp=datetime.now().isoformat(),
            )

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """マスクをバイナリに正規化"""
        if mask.dtype == np.bool_:
            return mask
        return (mask > 127).astype(np.uint8)

    def _calculate_confidence(
        self, pred_binary: np.ndarray, gt_binary: np.ndarray, iou: float
    ) -> float:
        """信頼度計算"""
        # IoUベースの基本信頼度
        base_confidence = min(iou * 1.2, 1.0)

        # サイズ類似性による調整
        pred_size = pred_binary.sum()
        gt_size = gt_binary.sum()

        if gt_size > 0:
            size_ratio = min(pred_size, gt_size) / max(pred_size, gt_size)
            size_confidence = 0.5 + (size_ratio * 0.5)
        else:
            size_confidence = 0.5

        # 最終信頼度
        confidence = base_confidence * 0.7 + size_confidence * 0.3
        return min(confidence, 1.0)


class SCICalculator:
    """Semantic Completeness Index 計算器"""

    def __init__(self):
        """初期化"""
        self.threshold = 0.70  # SCI成功閾値

        # MediaPipe初期化
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
        else:
            self.pose_detector = None

        # OpenCV顔検出器
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except:
            logger.warning("OpenCV顔検出器初期化失敗 - フォールバック実装を使用")
            self.face_cascade = None

    def calculate(self, extracted_image: np.ndarray) -> ObjectiveMetricResult:
        """
        SCI計算（人体構造完全性評価）

        Args:
            extracted_image: 抽出されたキャラクター画像 (RGB)

        Returns:
            ObjectiveMetricResult: SCI結果
        """
        try:
            completeness_components = {}

            # 1. 顔検出評価 (30% weight)
            face_score = self._evaluate_face_detection(extracted_image)
            completeness_components["face_score"] = face_score

            # 2. 肢体完全性評価 (40% weight)
            limb_score = self._evaluate_limb_completeness(extracted_image)
            completeness_components["limb_score"] = limb_score

            # 3. 輪郭連続性評価 (30% weight)
            contour_score = self._evaluate_contour_continuity(extracted_image)
            completeness_components["contour_score"] = contour_score

            # 重み付き総合スコア計算
            sci_score = face_score * 0.3 + limb_score * 0.4 + contour_score * 0.3

            # 信頼度計算
            confidence = self._calculate_sci_confidence(completeness_components)

            details = {
                "face_detection_score": face_score,
                "limb_completeness_score": limb_score,
                "contour_continuity_score": contour_score,
                "weighted_total": sci_score,
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "detection_methods": self._get_detection_methods(),
            }

            status = "passed" if sci_score >= self.threshold else "failed"

            return ObjectiveMetricResult(
                metric_name="SCI (Semantic Completeness Index)",
                value=sci_score,
                confidence=confidence,
                status=status,
                details=details,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"SCI計算エラー: {e}")
            return ObjectiveMetricResult(
                metric_name="SCI (Semantic Completeness Index)",
                value=0.0,
                confidence=0.0,
                status="error",
                details={"error": str(e)},
                timestamp=datetime.now().isoformat(),
            )

    def _evaluate_face_detection(self, image: np.ndarray) -> float:
        """顔検出評価"""
        if self.face_cascade is None:
            return 0.5  # フォールバック値

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                return 0.0

            # 最大の顔を使用
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            face_area = largest_face[2] * largest_face[3]
            image_area = image.shape[0] * image.shape[1]

            # 顔のサイズ比率に基づくスコア
            face_ratio = face_area / image_area

            # 適切なサイズ範囲（5-40%）でスコア化
            if 0.05 <= face_ratio <= 0.4:
                return min(1.0, face_ratio * 10)  # 0.05で0.5, 0.1で1.0
            elif face_ratio < 0.05:
                return face_ratio * 10  # 小さすぎる
            else:
                return max(0.5, 1.0 - (face_ratio - 0.4) * 2)  # 大きすぎる

        except Exception as e:
            logger.warning(f"顔検出エラー: {e}")
            return 0.3  # エラー時のデフォルト値

    def _evaluate_limb_completeness(self, image: np.ndarray) -> float:
        """肢体完全性評価"""
        if not MEDIAPIPE_AVAILABLE or self.pose_detector is None:
            return self._fallback_limb_evaluation(image)

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            results = self.pose_detector.process(rgb_image)

            if not results.pose_landmarks:
                return 0.0

            # 重要な関節点
            critical_landmarks = [
                self.mp_pose.PoseLandmark.NOSE.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ]

            detected_count = 0
            total_confidence = 0.0

            for landmark_idx in critical_landmarks:
                landmark = results.pose_landmarks.landmark[landmark_idx]
                if landmark.visibility > 0.5:
                    detected_count += 1
                    total_confidence += landmark.visibility

            if detected_count == 0:
                return 0.0

            # 検出率 × 平均信頼度
            detection_rate = detected_count / len(critical_landmarks)
            avg_confidence = total_confidence / detected_count

            return detection_rate * avg_confidence

        except Exception as e:
            logger.warning(f"MediaPipe肢体評価エラー: {e}")
            return self._fallback_limb_evaluation(image)

    def _fallback_limb_evaluation(self, image: np.ndarray) -> float:
        """フォールバック肢体評価（輪郭解析ベース）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # 輪郭検出
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0

            largest_contour = max(contours, key=cv2.contourArea)

            # アスペクト比による人体形状推定
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = h / w if w > 0 else 0

            # 人体の典型的アスペクト比 (1.5-3.0) で評価
            if 1.5 <= aspect_ratio <= 3.0:
                return min(1.0, aspect_ratio / 2.0)
            elif aspect_ratio < 1.5:
                return aspect_ratio / 1.5 * 0.5
            else:
                return max(0.3, 1.0 - (aspect_ratio - 3.0) * 0.2)

        except Exception as e:
            logger.warning(f"フォールバック肢体評価エラー: {e}")
            return 0.3

    def _evaluate_contour_continuity(self, image: np.ndarray) -> float:
        """輪郭連続性評価"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # エッジ検出
            edges = cv2.Canny(gray, 50, 150)

            # 輪郭検出
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0

            largest_contour = max(contours, key=cv2.contourArea)

            # 1. 輪郭の滑らかさ（曲率変化率）
            smoothness = self._calculate_contour_smoothness(largest_contour)

            # 2. 連続性（ギャップの少なさ）
            continuity = self._calculate_contour_gaps(largest_contour)

            # 3. 密度（輪郭の充実度）
            density = self._calculate_contour_density(largest_contour, gray.shape)

            # 重み付き平均
            contour_score = smoothness * 0.4 + continuity * 0.4 + density * 0.2
            return min(contour_score, 1.0)

        except Exception as e:
            logger.warning(f"輪郭連続性評価エラー: {e}")
            return 0.3

    def _calculate_contour_smoothness(self, contour: np.ndarray) -> float:
        """輪郭滑らかさ計算"""
        if len(contour) < 10:
            return 0.0

        try:
            # 曲率計算（近似）
            contour = contour.reshape(-1, 2)

            # 隣接点間の角度変化を計算
            angles = []
            for i in range(1, len(contour) - 1):
                p1, p2, p3 = contour[i - 1], contour[i], contour[i + 1]

                v1 = p1 - p2
                v2 = p3 - p2

                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)

            if not angles:
                return 0.5

            # 角度変化の標準偏差（小さいほど滑らか）
            angle_std = np.std(angles)
            smoothness = max(0.0, 1.0 - angle_std / np.pi)

            return smoothness

        except Exception as e:
            logger.warning(f"輪郭滑らかさ計算エラー: {e}")
            return 0.5

    def _calculate_contour_gaps(self, contour: np.ndarray) -> float:
        """輪郭ギャップ計算"""
        if len(contour) < 3:
            return 0.0

        try:
            contour = contour.reshape(-1, 2)

            # 隣接点間の距離
            distances = []
            for i in range(len(contour)):
                next_i = (i + 1) % len(contour)
                dist = np.linalg.norm(contour[next_i] - contour[i])
                distances.append(dist)

            if not distances:
                return 0.5

            # 距離の変動係数（小さいほど均一）
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            if mean_dist > 0:
                cv = std_dist / mean_dist  # 変動係数
                continuity = max(0.0, 1.0 - cv)
            else:
                continuity = 0.0

            return continuity

        except Exception as e:
            logger.warning(f"輪郭ギャップ計算エラー: {e}")
            return 0.5

    def _calculate_contour_density(
        self, contour: np.ndarray, image_shape: Tuple[int, int]
    ) -> float:
        """輪郭密度計算"""
        try:
            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(contour, True)

            if contour_perimeter == 0:
                return 0.0

            # コンパクトネス（4π×面積/周長²）
            compactness = (4 * np.pi * contour_area) / (contour_perimeter**2)

            # 画像に対する面積比
            image_area = image_shape[0] * image_shape[1]
            area_ratio = contour_area / image_area

            # 適切な範囲（10-70%）で正規化
            normalized_ratio = min(1.0, max(0.0, (area_ratio - 0.1) / 0.6))

            # コンパクトネス（0-1）と面積比を組み合わせ
            density = compactness * 0.6 + normalized_ratio * 0.4
            return min(density, 1.0)

        except Exception as e:
            logger.warning(f"輪郭密度計算エラー: {e}")
            return 0.3

    def _calculate_sci_confidence(self, components: Dict[str, float]) -> float:
        """SCI信頼度計算"""
        # 各コンポーネントの信頼度
        face_conf = min(1.0, components["face_score"] * 1.2)
        limb_conf = components["limb_score"]
        contour_conf = components["contour_score"]

        # MediaPipeが利用可能かどうかで重み調整
        if MEDIAPIPE_AVAILABLE:
            confidence = face_conf * 0.3 + limb_conf * 0.5 + contour_conf * 0.2
        else:
            confidence = face_conf * 0.4 + limb_conf * 0.3 + contour_conf * 0.3

        return min(confidence, 1.0)

    def _get_detection_methods(self) -> List[str]:
        """使用可能な検出手法を取得"""
        methods = []
        if self.face_cascade is not None:
            methods.append("OpenCV Haar Cascade")
        if MEDIAPIPE_AVAILABLE and self.pose_detector is not None:
            methods.append("MediaPipe Pose")
        if not methods:
            methods.append("Fallback Contour Analysis")
        return methods


class PLETracker:
    """Progressive Learning Efficiency 追跡器"""

    def __init__(self, history_file: str = "quality_history.json"):
        """初期化"""
        self.history_file = Path(history_file)
        self.threshold = 0.10  # PLE成功閾値
        self.time_window = 10  # 評価時間窓
        self.history = self._load_history()

    def calculate(self, current_results: List[float]) -> ObjectiveMetricResult:
        """
        PLE計算（継続的学習効率）

        Args:
            current_results: 最新の結果リスト

        Returns:
            ObjectiveMetricResult: PLE結果
        """
        try:
            # 履歴更新
            self._update_history(current_results)

            if len(self.history) < self.time_window * 2:
                # データ不足の場合
                return ObjectiveMetricResult(
                    metric_name="PLE (Progressive Learning Efficiency)",
                    value=0.0,
                    confidence=0.0,
                    status="insufficient_data",
                    details={
                        "message": f"データ不足 ({len(self.history)}/{self.time_window * 2} required)",
                        "current_data_points": len(self.history),
                    },
                    timestamp=datetime.now().isoformat(),
                )

            # ベースライン期間と現在期間
            baseline_results = self.history[: self.time_window]
            recent_results = self.history[-self.time_window :]

            # PLE計算
            ple_score, details = self._calculate_ple_score(baseline_results, recent_results)

            # 信頼度計算
            confidence = self._calculate_ple_confidence(details)

            status = "passed" if ple_score >= self.threshold else "failed"
            if ple_score < -0.05:
                status = "regression"  # 退行検出

            return ObjectiveMetricResult(
                metric_name="PLE (Progressive Learning Efficiency)",
                value=ple_score,
                confidence=confidence,
                status=status,
                details=details,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"PLE計算エラー: {e}")
            return ObjectiveMetricResult(
                metric_name="PLE (Progressive Learning Efficiency)",
                value=0.0,
                confidence=0.0,
                status="error",
                details={"error": str(e)},
                timestamp=datetime.now().isoformat(),
            )

    def _load_history(self) -> List[float]:
        """履歴データ読み込み"""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    return data.get("quality_scores", [])
            return []
        except Exception as e:
            logger.warning(f"履歴読み込みエラー: {e}")
            return []

    def _update_history(self, current_results: List[float]) -> None:
        """履歴データ更新"""
        try:
            # 平均品質を履歴に追加
            if current_results:
                avg_quality = np.mean(current_results)
                self.history.append(avg_quality)

            # 履歴サイズ制限（最大100件）
            if len(self.history) > 100:
                self.history = self.history[-100:]

            # ファイル保存
            self._save_history()

        except Exception as e:
            logger.error(f"履歴更新エラー: {e}")

    def _save_history(self) -> None:
        """履歴データ保存"""
        try:
            data = {
                "quality_scores": self.history,
                "last_updated": datetime.now().isoformat(),
                "total_data_points": len(self.history),
            }

            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"履歴保存エラー: {e}")

    def _calculate_ple_score(
        self, baseline: List[float], recent: List[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """PLEスコア計算"""
        baseline_avg = np.mean(baseline)
        recent_avg = np.mean(recent)

        # 1. 改善率 (40% weight)
        if baseline_avg == 0:
            improvement_rate = 0.0
        else:
            improvement_rate = (recent_avg - baseline_avg) / baseline_avg

        # 2. 安定性 (30% weight) - 標準偏差の逆数
        recent_std = np.std(recent)
        stability = 1.0 - min(recent_std, 1.0)

        # 3. 効率性 (30% weight) - 改善量 / 試行回数
        trial_count = len(self.history)
        trial_efficiency = improvement_rate / (trial_count / 100.0) if trial_count > 0 else 0.0

        # 重み付き平均
        ple_score = improvement_rate * 0.4 + stability * 0.3 + trial_efficiency * 0.3

        # -1.0 to 1.0 の範囲に正規化
        ple_score = max(-1.0, min(1.0, ple_score))

        details = {
            "baseline_average": float(baseline_avg),
            "recent_average": float(recent_avg),
            "improvement_rate": float(improvement_rate),
            "stability_score": float(stability),
            "trial_efficiency": float(trial_efficiency),
            "recent_std": float(recent_std),
            "total_trials": trial_count,
            "time_window": self.time_window,
        }

        return ple_score, details

    def _calculate_ple_confidence(self, details: Dict[str, Any]) -> float:
        """PLE信頼度計算"""
        # データ量による信頼度
        trial_count = details["total_trials"]
        data_confidence = min(1.0, trial_count / 50.0)  # 50件で満点

        # 安定性による信頼度
        stability_confidence = details["stability_score"]

        # 改善率の絶対値による信頼度（大きな変化は不安定）
        improvement_abs = abs(details["improvement_rate"])
        change_confidence = max(0.3, 1.0 - improvement_abs * 2)

        # 総合信頼度
        confidence = data_confidence * 0.4 + stability_confidence * 0.4 + change_confidence * 0.2
        return confidence


class ObjectiveMetricsSystem:
    """客観的指標システム統合クラス"""

    def __init__(self, history_file: str = "quality_history.json"):
        """初期化"""
        self.pla_calculator = PLACalculator()
        self.sci_calculator = SCICalculator()
        self.ple_tracker = PLETracker(history_file)

    def evaluate_extraction_batch(
        self,
        extracted_images: List[np.ndarray],
        ground_truth_masks: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, ObjectiveMetricResult]:
        """
        バッチ抽出結果の客観的評価

        Args:
            extracted_images: 抽出された画像リスト
            ground_truth_masks: 正解マスクリスト（PLA計算用、オプション）

        Returns:
            Dict[str, ObjectiveMetricResult]: 各指標の結果
        """
        results = {}

        # SCI評価（全画像）
        sci_scores = []
        for img in extracted_images:
            sci_result = self.sci_calculator.calculate(img)
            sci_scores.append(sci_result.value)

        avg_sci = np.mean(sci_scores) if sci_scores else 0.0
        results["SCI"] = ObjectiveMetricResult(
            metric_name="SCI (Batch Average)",
            value=avg_sci,
            confidence=np.mean([0.8] * len(sci_scores)) if sci_scores else 0.0,
            status="passed" if avg_sci >= self.sci_calculator.threshold else "failed",
            details={
                "individual_scores": sci_scores,
                "score_std": float(np.std(sci_scores)) if sci_scores else 0.0,
                "image_count": len(extracted_images),
            },
            timestamp=datetime.now().isoformat(),
        )

        # PLA評価（正解マスクがある場合）
        if ground_truth_masks and len(ground_truth_masks) == len(extracted_images):
            pla_scores = []
            for img, gt_mask in zip(extracted_images, ground_truth_masks):
                # 画像からマスク抽出（簡易版）
                pred_mask = self._extract_mask_from_image(img)
                pla_result = self.pla_calculator.calculate(pred_mask, gt_mask)
                pla_scores.append(pla_result.value)

            avg_pla = np.mean(pla_scores) if pla_scores else 0.0
            results["PLA"] = ObjectiveMetricResult(
                metric_name="PLA (Batch Average)",
                value=avg_pla,
                confidence=np.mean([0.9] * len(pla_scores)) if pla_scores else 0.0,
                status="passed" if avg_pla >= self.pla_calculator.threshold else "failed",
                details={
                    "individual_scores": pla_scores,
                    "score_std": float(np.std(pla_scores)) if pla_scores else 0.0,
                    "image_count": len(extracted_images),
                },
                timestamp=datetime.now().isoformat(),
            )
        else:
            results["PLA"] = ObjectiveMetricResult(
                metric_name="PLA (Pixel-Level Accuracy)",
                value=0.0,
                confidence=0.0,
                status="no_ground_truth",
                details={"message": "正解マスクが提供されていません"},
                timestamp=datetime.now().isoformat(),
            )

        # PLE評価（履歴ベース）
        current_quality_scores = sci_scores  # SCI値を品質スコアとして使用
        results["PLE"] = self.ple_tracker.calculate(current_quality_scores)

        return results

    def _extract_mask_from_image(self, image: np.ndarray) -> np.ndarray:
        """画像からマスクを抽出（簡易版）"""
        try:
            if image.shape[2] == 4:  # RGBA
                return image[:, :, 3]
            else:
                # グレースケール変換してバイナリマスク作成
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                return mask
        except:
            # フォールバック
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255


def main():
    """テスト実行"""
    import argparse

    parser = argparse.ArgumentParser(description="客観的3指標システムテスト")
    parser.add_argument(
        "--test", choices=["pla", "sci", "ple", "all"], default="all", help="テストする指標"
    )

    args = parser.parse_args()

    # テスト用データ作成
    test_image = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (300, 200), dtype=np.uint8) * 255
    ground_truth_mask = np.random.randint(0, 2, (300, 200), dtype=np.uint8) * 255

    print("🎯 客観的3指標システムテスト")
    print("=" * 50)

    if args.test in ["pla", "all"]:
        print("\n📊 PLA (Pixel-Level Accuracy) テスト")
        pla_calc = PLACalculator()
        pla_result = pla_calc.calculate(test_mask, ground_truth_mask)
        print(f"結果: {pla_result.value:.3f} (信頼度: {pla_result.confidence:.3f})")
        print(f"ステータス: {pla_result.status}")
        print(
            f"詳細: IoU={pla_result.details['iou']:.3f}, Dice={pla_result.details['dice_coefficient']:.3f}"
        )

    if args.test in ["sci", "all"]:
        print("\n🎭 SCI (Semantic Completeness Index) テスト")
        sci_calc = SCICalculator()
        sci_result = sci_calc.calculate(test_image)
        print(f"結果: {sci_result.value:.3f} (信頼度: {sci_result.confidence:.3f})")
        print(f"ステータス: {sci_result.status}")
        print(
            f"詳細: Face={sci_result.details['face_detection_score']:.3f}, "
            f"Limb={sci_result.details['limb_completeness_score']:.3f}, "
            f"Contour={sci_result.details['contour_continuity_score']:.3f}"
        )

    if args.test in ["ple", "all"]:
        print("\n📈 PLE (Progressive Learning Efficiency) テスト")
        ple_tracker = PLETracker()
        test_scores = [0.5, 0.6, 0.65, 0.7, 0.72]  # 改善トレンド
        ple_result = ple_tracker.calculate(test_scores)
        print(f"結果: {ple_result.value:.3f} (信頼度: {ple_result.confidence:.3f})")
        print(f"ステータス: {ple_result.status}")
        if "improvement_rate" in ple_result.details:
            print(
                f"詳細: 改善率={ple_result.details['improvement_rate']:.3f}, "
                f"安定性={ple_result.details['stability_score']:.3f}"
            )

    print("\n✅ テスト完了")


if __name__ == "__main__":
    main()
