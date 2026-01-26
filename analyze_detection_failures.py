#!/usr/bin/env python3
"""
検出失敗分析システム
顔検出・ポーズ検出の失敗原因を詳細分析し改善策を提案
"""

import numpy as np
import cv2

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.evaluation.enhanced_detection_systems import (
    EnhancedFaceDetector,
    EnhancedPoseDetector,
)

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DetectionFailureAnalyzer:
    """検出失敗分析システム"""

    def __init__(self):
        self.face_detector = EnhancedFaceDetector()
        self.pose_detector = EnhancedPoseDetector()
        self.logger = logging.getLogger(f"{__name__}.DetectionFailureAnalyzer")

    def analyze_image_characteristics(self, image_path: str) -> Dict:
        """画像特性分析"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "画像読み込み失敗"}

        # 基本画像特性
        height, width = image.shape[:2]
        area = height * width
        aspect_ratio = width / height

        # 色彩特性
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # エッジ密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / area

        # 色分布
        color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_entropy = -np.sum(color_hist * np.log2(color_hist + 1e-10))

        return {
            "dimensions": {"width": width, "height": height, "area": area},
            "aspect_ratio": aspect_ratio,
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "color_entropy": color_entropy,
        }

    def analyze_face_detection_failure(self, image_path: str) -> Dict:
        """顔検出失敗分析"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "画像読み込み失敗"}

        # 各手法での検出試行
        detections = self.face_detector.detect_faces_comprehensive(image)

        # MediaPipe詳細分析
        mp_analysis = self._analyze_mediapipe_face_failure(image)

        # OpenCV詳細分析
        cv_analysis = self._analyze_opencv_face_failure(image)

        # 推定失敗要因
        failure_factors = self._identify_face_failure_factors(image, detections)

        return {
            "detections_found": len(detections),
            "detection_details": [
                {"method": d.method, "confidence": d.confidence, "bbox": d.bbox} for d in detections
            ],
            "mediapipe_analysis": mp_analysis,
            "opencv_analysis": cv_analysis,
            "failure_factors": failure_factors,
        }

    def _analyze_mediapipe_face_failure(self, image: np.ndarray) -> Dict:
        """MediaPipe顔検出失敗分析"""
        try:
            import mediapipe as mp

            # 複数の閾値でテスト
            thresholds = [0.1, 0.3, 0.5, 0.7]
            results = {}

            for threshold in thresholds:
                detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=threshold
                )

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detection_results = detector.process(rgb_image)

                face_count = (
                    len(detection_results.detections) if detection_results.detections else 0
                )
                results[f"threshold_{threshold}"] = face_count

            return {
                "threshold_analysis": results,
                "recommended_threshold": self._find_optimal_threshold(results),
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_opencv_face_failure(self, image: np.ndarray) -> Dict:
        """OpenCV顔検出失敗分析"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 複数パラメータでテスト
        scale_factors = [1.05, 1.1, 1.2, 1.3]
        min_neighbors = [2, 3, 4, 5]
        min_sizes = [(15, 15), (20, 20), (30, 30), (40, 40)]

        results = {}
        best_config = None
        max_detections = 0

        for cascade_name, cascade in self.face_detector.cascade_detectors.items():
            cascade_results = {}

            for scale in scale_factors:
                for neighbors in min_neighbors:
                    for min_size in min_sizes:
                        try:
                            faces = cascade.detectMultiScale(
                                gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
                            )

                            face_count = len(faces)
                            config_key = f"scale_{scale}_neighbors_{neighbors}_size_{min_size[0]}"
                            cascade_results[config_key] = face_count

                            if face_count > max_detections:
                                max_detections = face_count
                                best_config = {
                                    "cascade": cascade_name,
                                    "scale_factor": scale,
                                    "min_neighbors": neighbors,
                                    "min_size": min_size,
                                    "detections": face_count,
                                }

                        except Exception as e:
                            continue

            results[cascade_name] = cascade_results

        return {
            "parameter_analysis": results,
            "best_configuration": best_config,
            "max_detections": max_detections,
        }

    def _identify_face_failure_factors(self, image: np.ndarray, detections: List) -> List[str]:
        """顔検出失敗要因特定"""
        factors = []

        height, width = image.shape[:2]
        area = height * width

        # 画像サイズ要因
        if area < 100000:  # 100K pixels未満
            factors.append("画像解像度が低い（<100K pixels）")

        # 明度要因
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:
            factors.append("画像が暗すぎる（平均明度<50）")
        elif brightness > 200:
            factors.append("画像が明るすぎる（平均明度>200）")

        # コントラスト要因
        contrast = np.std(gray)
        if contrast < 20:
            factors.append("コントラストが低い（標準偏差<20）")

        # アスペクト比要因
        aspect_ratio = width / height
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            factors.append(f"極端なアスペクト比（{aspect_ratio:.2f}）")

        # 顔サイズ推定（大まかな推定）
        expected_face_size = min(width, height) * 0.1  # 画像の10%程度
        if expected_face_size < 30:
            factors.append("推定顔サイズが小さすぎる（<30px）")

        # 検出結果による分析
        if len(detections) == 0:
            factors.append("全手法で検出失敗")
        elif len(detections) == 1:
            factors.append("単一手法のみ検出成功")

        return factors

    def _find_optimal_threshold(self, threshold_results: Dict) -> float:
        """最適閾値の推定"""
        best_threshold = 0.3
        max_detections = 0

        for threshold_key, detection_count in threshold_results.items():
            if detection_count > max_detections:
                max_detections = detection_count
                best_threshold = float(threshold_key.split("_")[1])

        return best_threshold

    def analyze_pose_detection_failure(self, image_path: str) -> Dict:
        """ポーズ検出失敗分析"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "画像読み込み失敗"}

        # ポーズ検出実行
        pose_result = self.pose_detector.detect_pose_comprehensive(image)

        # 失敗要因分析
        failure_factors = self._identify_pose_failure_factors(image, pose_result)

        # 閾値分析
        threshold_analysis = self._analyze_pose_thresholds(image)

        return {
            "pose_detected": pose_result.detected,
            "pose_details": {
                "category": pose_result.pose_category,
                "visibility_score": pose_result.visibility_score,
                "completeness_score": pose_result.completeness_score,
                "confidence": pose_result.confidence,
                "keypoints_detected": pose_result.keypoints_detected,
            }
            if pose_result.detected
            else None,
            "failure_factors": failure_factors,
            "threshold_analysis": threshold_analysis,
        }

    def _identify_pose_failure_factors(self, image: np.ndarray, pose_result) -> List[str]:
        """ポーズ検出失敗要因特定"""
        factors = []

        if not pose_result.detected:
            factors.append("MediaPipeランドマーク検出失敗")
        else:
            if pose_result.visibility_score < 0.3:
                factors.append(f"可視性スコア低下（{pose_result.visibility_score:.3f}）")

            if pose_result.completeness_score < 0.5:
                factors.append(f"完全性スコア低下（{pose_result.completeness_score:.3f}）")

            if pose_result.keypoints_detected < 10:
                factors.append(f"検出キーポイント数不足（{pose_result.keypoints_detected}/33）")

        # 画像特性による要因推定
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 人体サイズ推定
        estimated_body_height = height * 0.7  # 画像の70%程度
        if estimated_body_height < 200:
            factors.append("推定人体サイズが小さい（<200px高）")

        # 背景複雑度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        if edge_density > 0.3:
            factors.append(f"背景が複雑（エッジ密度: {edge_density:.3f}）")

        return factors

    def _analyze_pose_thresholds(self, image: np.ndarray) -> Dict:
        """ポーズ検出閾値分析"""
        try:
            import mediapipe as mp

            thresholds = [0.1, 0.3, 0.5, 0.7]
            results = {}

            for threshold in thresholds:
                pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=threshold,
                )

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose_detector.process(rgb_image)

                detected = pose_results.pose_landmarks is not None
                keypoint_count = len(pose_results.pose_landmarks.landmark) if detected else 0

                results[f"threshold_{threshold}"] = {
                    "detected": detected,
                    "keypoints": keypoint_count,
                }

            return results

        except Exception as e:
            return {"error": str(e)}


def analyze_dataset_failures():
    """データセット全体の失敗パターン分析"""
    logger.info("🔍 検出失敗分析開始")
    logger.info("=" * 70)

    # テストディレクトリ
    test_directories = [
        "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix",
        "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix",
        "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix",
    ]

    analyzer = DetectionFailureAnalyzer()

    all_results = []
    face_failure_patterns = {}
    pose_failure_patterns = {}

    for test_dir in test_directories:
        dir_path = Path(test_dir)

        if not dir_path.exists():
            continue

        logger.info(f"\n📁 分析ディレクトリ: {dir_path.name}")

        # JPGファイルのみ対象（元画像）
        jpg_files = list(dir_path.glob("*.jpg"))

        for jpg_file in jpg_files:
            logger.info(f"  🔍 分析中: {jpg_file.name}")

            try:
                # 画像特性分析
                image_chars = analyzer.analyze_image_characteristics(str(jpg_file))

                # 顔検出失敗分析
                face_analysis = analyzer.analyze_face_detection_failure(str(jpg_file))

                # ポーズ検出失敗分析
                pose_analysis = analyzer.analyze_pose_detection_failure(str(jpg_file))

                result = {
                    "filename": jpg_file.name,
                    "directory": dir_path.name,
                    "image_characteristics": image_chars,
                    "face_analysis": face_analysis,
                    "pose_analysis": pose_analysis,
                }

                all_results.append(result)

                # 失敗パターン収集
                if face_analysis.get("detections_found", 0) == 0:
                    for factor in face_analysis.get("failure_factors", []):
                        face_failure_patterns[factor] = face_failure_patterns.get(factor, 0) + 1

                if not pose_analysis.get("pose_detected", False):
                    for factor in pose_analysis.get("failure_factors", []):
                        pose_failure_patterns[factor] = pose_failure_patterns.get(factor, 0) + 1

            except Exception as e:
                logger.error(f"    ❌ 分析エラー: {e}")
                continue

    # 分析結果サマリー
    logger.info("\n" + "=" * 70)
    logger.info("📊 検出失敗パターン分析結果")
    logger.info("=" * 70)

    logger.info(f"\n👤 顔検出失敗パターン（上位5位）:")
    sorted_face_patterns = sorted(face_failure_patterns.items(), key=lambda x: x[1], reverse=True)
    for i, (pattern, count) in enumerate(sorted_face_patterns[:5]):
        logger.info(f"  {i+1}. {pattern}: {count}件")

    logger.info(f"\n🤸 ポーズ検出失敗パターン（上位5位）:")
    sorted_pose_patterns = sorted(pose_failure_patterns.items(), key=lambda x: x[1], reverse=True)
    for i, (pattern, count) in enumerate(sorted_pose_patterns[:5]):
        logger.info(f"  {i+1}. {pattern}: {count}件")

    # 改善提案
    logger.info(f"\n💡 改善提案:")

    # 顔検出改善提案
    logger.info(f"  👤 顔検出改善策:")
    if "全手法で検出失敗" in face_failure_patterns:
        logger.info(f"    - MediaPipe閾値を0.1に下げる")
        logger.info(f"    - OpenCV minNeighbors を2に下げる")
        logger.info(f"    - アニメ顔専用カスケードの追加")

    if "画像が暗すぎる" in face_failure_patterns or "コントラストが低い" in face_failure_patterns:
        logger.info(f"    - 前処理での明度・コントラスト調整")
        logger.info(f"    - ヒストグラム平均化の適用")

    # ポーズ検出改善提案
    logger.info(f"  🤸 ポーズ検出改善策:")
    if "MediaPipeランドマーク検出失敗" in pose_failure_patterns:
        logger.info(f"    - MediaPipe Pose閾値を0.1に下げる")
        logger.info(f"    - モデル複雑度を1（高速）でも試行")

    if "検出キーポイント数不足" in pose_failure_patterns:
        logger.info(f"    - 部分的ポーズでも評価対象とする")
        logger.info(f"    - キーポイント閾値を10→5に下げる")

    # レポート保存
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "detection_failure_analysis",
        "total_analyzed": len(all_results),
        "face_failure_patterns": face_failure_patterns,
        "pose_failure_patterns": pose_failure_patterns,
        "detailed_results": all_results,
    }

    report_file = f"evaluation_reports/detection_failure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("evaluation_reports", exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n💾 詳細レポート保存: {report_file}")
    logger.info("✅ 検出失敗分析完了")

    return all_results


if __name__ == "__main__":
    try:
        results = analyze_dataset_failures()
        print(f"\n🔍 分析結果: {len(results)}件の画像を分析完了")
    except Exception as e:
        logger.error(f"❌ 分析実行エラー: {e}")
        sys.exit(1)
