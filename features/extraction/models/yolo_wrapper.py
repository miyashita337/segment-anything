#!/usr/bin/env python3
"""
YOLO Model Wrapper
YOLOv8 wrapper for character detection and scoring
"""

import numpy as np
import cv2
import torch

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not available, YOLO functionality disabled")


class YOLOModelWrapper:
    """
    Wrapper class for YOLOv8 model
    Provides character detection and scoring capabilities
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Initialize YOLO wrapper with CI environment optimization

        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            device: Device to run on (cuda/cpu), auto-detect if None
        """
        # CI環境検出・最適化
        try:
            from features.common.ci_environment import CIEnvironmentDetector

            ci_config = CIEnvironmentDetector.detect_ci_environment()

            # CI環境での自動軽量化
            if ci_config.is_ci:
                model_path = ci_config.yolo_model  # yolov8n.pt (nano)
                if confidence_threshold == 0.25:  # デフォルト値の場合のみ調整
                    confidence_threshold = 0.1  # CI環境では少し緩め
                device = "cpu" if ci_config.cpu_only else device
                print(
                    f"🔧 CI Environment detected - YOLO optimized: {model_path} (CPU-only: {ci_config.cpu_only})"
                )
        except ImportError:
            # CI検出モジュール未使用の場合は従来通り
            pass

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.is_loaded = False

        # COCO class names for person detection
        self.person_class_id = 0  # 'person' class in COCO dataset  # 'person' class in COCO dataset

    def load_model(self) -> bool:
        """
        Load YOLO model

        Returns:
            True if successful, False otherwise
        """
        if not YOLO_AVAILABLE:
            print("❌ YOLO unavailable - ultralytics not installed")
            return False

        try:
            # Check if model file exists, download if needed
            if not os.path.exists(self.model_path):
                print(f"📥 Downloading YOLO model: {self.model_path}")

            # Load YOLO model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            self.is_loaded = True
            print(f"✅ YOLO {self.model_path} loaded on {self.device}")
            return True

        except Exception as e:
            print(f"❌ YOLO model loading failed: {e}")
            return False

    def detect_persons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect person objects in the image

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO model not loaded. Call load_model() first.")

        try:
            # Run inference
            results = self.model(image, verbose=False)

            persons = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Check if detection is a person
                        class_id = int(box.cls[0])
                        if class_id == self.person_class_id:
                            confidence = float(box.conf[0])

                            # Filter by confidence threshold
                            if confidence >= self.confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].tolist()

                                persons.append(
                                    {
                                        "bbox": [
                                            int(x1),
                                            int(y1),
                                            int(x2 - x1),
                                            int(y2 - y1),
                                        ],  # [x, y, w, h]
                                        "bbox_xyxy": [
                                            int(x1),
                                            int(y1),
                                            int(x2),
                                            int(y2),
                                        ],  # [x1, y1, x2, y2]
                                        "confidence": confidence,
                                        "area": int((x2 - x1) * (y2 - y1)),
                                        "class_id": class_id,
                                        "class_name": "person",
                                    }
                                )

            # Sort by confidence (highest first)
            persons.sort(key=lambda x: x["confidence"], reverse=True)

            return persons

        except Exception as e:
            print(f"❌ Person detection failed: {e}")
            return []

    def calculate_overlap_score(
        self, mask_bbox: Tuple[int, int, int, int], person_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate overlap score between mask and person detection

        Args:
            mask_bbox: Mask bounding box [x, y, w, h]
            person_bbox: Person detection bounding box [x, y, w, h]

        Returns:
            Overlap score (IoU - Intersection over Union)
        """
        # Convert to [x1, y1, x2, y2] format
        mask_x1, mask_y1 = mask_bbox[0], mask_bbox[1]
        mask_x2, mask_y2 = mask_x1 + mask_bbox[2], mask_y1 + mask_bbox[3]

        person_x1, person_y1 = person_bbox[0], person_bbox[1]
        person_x2, person_y2 = person_x1 + person_bbox[2], person_y1 + person_bbox[3]

        # Calculate intersection
        intersection_x1 = max(mask_x1, person_x1)
        intersection_y1 = max(mask_y1, person_y1)
        intersection_x2 = min(mask_x2, person_x2)
        intersection_y2 = min(mask_y2, person_y2)

        # Check if there's an intersection
        if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
            return 0.0

        # Calculate areas
        intersection_area = (intersection_x2 - intersection_x1) * (
            intersection_y2 - intersection_y1
        )
        mask_area = mask_bbox[2] * mask_bbox[3]
        person_area = person_bbox[2] * person_bbox[3]
        union_area = mask_area + person_area - intersection_area

        # Calculate IoU
        if union_area == 0:
            return 0.0

        iou = intersection_area / union_area
        return iou

    def score_masks_with_detections(
        self,
        masks: List[Dict[str, Any]],
        image: np.ndarray,
        use_expanded_boxes: bool = False,
        expansion_strategy: str = "balanced",
    ) -> List[Dict[str, Any]]:
        """
        Score SAM masks based on YOLO person detections

        Args:
            masks: List of SAM mask dictionaries
            image: Input image for person detection
            use_expanded_boxes: GPT-4O推奨のボックス拡張を使用
            expansion_strategy: 拡張戦略 ('conservative', 'balanced', 'aggressive')

        Returns:
            List of masks with added YOLO scores
        """
        if not masks:
            return []

        # Get person detections
        persons = self.detect_persons(image)

        # GPT-4O推奨: ボックス拡張オプション
        if use_expanded_boxes and persons:
            try:
                from features.extraction.utils.box_expansion import apply_gpt4o_expansion_strategy

                image_shape = image.shape[:2]  # (height, width)
                expanded_persons = apply_gpt4o_expansion_strategy(
                    persons, image_shape, expansion_strategy
                )

                if expanded_persons:
                    print(
                        f"🎯 GPT-4O推奨ボックス拡張適用: {len(persons)}→{len(expanded_persons)} (戦略: {expansion_strategy})"
                    )
                    for i, person in enumerate(expanded_persons[:3]):  # 最大3件表示
                        exp_info = person.get("expansion_info", {})
                        print(
                            f"   検出{i+1}: H{exp_info.get('horizontal_factor', 0):.1f}x V{exp_info.get('vertical_factor', 0):.1f}x "
                            f"({'境界制限' if exp_info.get('clipped_to_bounds') else '制限なし'})"
                        )
                    persons = expanded_persons

            except ImportError as e:
                print(f"⚠️ ボックス拡張モジュール未利用可能: {e}")
            except Exception as e:
                print(f"⚠️ ボックス拡張エラー: {e}")
                # エラー時は元の検出結果を使用

        # Score each mask
        scored_masks = []

        for mask in masks:
            mask_bbox = mask["bbox"]  # [x, y, w, h]
            best_overlap = 0.0
            best_confidence = 0.0

            # Find best matching person detection
            for person in persons:
                overlap = self.calculate_overlap_score(mask_bbox, person["bbox"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_confidence = person["confidence"]

            # Calculate combined score
            yolo_score = best_overlap * best_confidence

            # Add YOLO scoring information to mask
            mask_with_score = mask.copy()
            mask_with_score.update(
                {
                    "yolo_overlap": best_overlap,
                    "yolo_confidence": best_confidence,
                    "yolo_score": yolo_score,
                    "combined_score": mask["stability_score"] * 0.6 + yolo_score * 0.4,
                }
            )

            scored_masks.append(mask_with_score)

        # Sort by combined score (highest first)
        scored_masks.sort(key=lambda x: x["combined_score"], reverse=True)

        return scored_masks

    def get_best_character_mask(
        self, masks: List[Dict[str, Any]], image: np.ndarray, min_yolo_score: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best character mask based on YOLO scoring

        Args:
            masks: List of SAM mask dictionaries
            image: Input image for person detection
            min_yolo_score: Minimum YOLO score threshold

        Returns:
            Best character mask or None if no good matches
        """
        scored_masks = self.score_masks_with_detections(masks, image)

        # Filter by minimum YOLO score
        good_masks = [m for m in scored_masks if m["yolo_score"] >= min_yolo_score]

        if good_masks:
            return good_masks[0]  # Highest scored mask

        return None

    def get_single_character_mask(
        self,
        masks: List[Dict[str, Any]],
        image: np.ndarray,
        enforce_single_character: bool = True,
        min_yolo_score: float = 0.1,
    ) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        単一キャラクター抽出（複数キャラクター検出・フィルタリング機能付き）

        Args:
            masks: SAMマスク候補リスト
            image: 入力画像
            enforce_single_character: 単一キャラクター強制モード
            min_yolo_score: 最小YOLOスコア閾値

        Returns:
            (最適マスク, 複数キャラ検出結果) または (None, 検出結果)
        """
        try:
            from features.evaluation.utils.multiple_character_detector import (
                MultipleCharacterDetector,
                detect_multiple_characters_from_image,
            )
        except ImportError as e:
            print(f"⚠️ Multiple character detector not available: {e}")
            # フォールバック: 従来の単一最適マスク選択
            return self.get_best_character_mask(masks, image, min_yolo_score), None

        # 1. YOLO検出実行
        yolo_detections = self.detect_persons(image)

        # 2. 複数キャラクター分析
        detector = MultipleCharacterDetector()
        multi_char_result = detector.analyze_yolo_detections(yolo_detections, image.shape[:2])

        print(
            f"🎯 複数キャラ分析: {multi_char_result.character_count}体検出, "
            f"タイプ={multi_char_result.detection_type.value}, "
            f"ペナルティ={multi_char_result.penalty_score:.2f}"
        )

        # 3. 単一キャラクター強制モードの場合
        if enforce_single_character and multi_char_result.is_multiple:
            if multi_char_result.penalty_score > 0.7:
                print(f"❌ 高ペナルティ({multi_char_result.penalty_score:.2f}): " f"LoRA学習に不適切な複数キャラ画像")
                return None, multi_char_result

            # メインキャラクター抽出試行
            if multi_char_result.primary_character_index is not None:
                print(f"🔄 メインキャラクター(#{multi_char_result.primary_character_index + 1})に絞り込み試行")
                # メインキャラのYOLO検出結果のみを使用
                primary_detection = [
                    multi_char_result.characters[multi_char_result.primary_character_index]
                ]

                # メインキャラクターのマスクのみを評価
                primary_yolo_detections = [
                    yolo_detections[multi_char_result.primary_character_index]
                ]
                scored_masks = self.score_masks_with_detections(masks, image)

                # メインキャラとの重複が最も高いマスクを選択
                best_mask = None
                best_score = 0.0

                primary_bbox = primary_detection[0]["bbox"]
                for mask in scored_masks:
                    overlap = self.calculate_overlap_score(mask["bbox"], primary_bbox)
                    if overlap > best_score and mask["yolo_score"] >= min_yolo_score:
                        best_score = overlap
                        best_mask = mask

                if best_mask:
                    print(f"✅ メインキャラマスク選択: 重複={best_score:.2f}, YOLO={best_mask['yolo_score']:.2f}")
                    return best_mask, multi_char_result

        # 4. 通常の最適マスク選択（複数キャラでも許可モード）
        scored_masks = self.score_masks_with_detections(masks, image)
        good_masks = [m for m in scored_masks if m["yolo_score"] >= min_yolo_score]

        if good_masks:
            selected_mask = good_masks[0]
            # 複数キャラ情報を追加
            selected_mask["multi_character_info"] = {
                "is_multiple": multi_char_result.is_multiple,
                "character_count": multi_char_result.character_count,
                "penalty_score": multi_char_result.penalty_score,
                "detection_type": multi_char_result.detection_type.value,
            }
            return selected_mask, multi_char_result

        return None, multi_char_result

    def filter_single_character_images(
        self, image_paths: List[Path], penalty_threshold: float = 0.5, save_reports: bool = False
    ) -> Tuple[List[Path], List[Path], Dict[str, Any]]:
        """
        画像リストから単一キャラクター画像をフィルタリング

        Args:
            image_paths: 画像パスリスト
            penalty_threshold: ペナルティ閾値（これ以上は除外）
            save_reports: レポート保存フラグ

        Returns:
            (単一キャラ画像リスト, 複数キャラ画像リスト, 統計情報)
        """
        try:
            from features.evaluation.utils.multiple_character_detector import (
                detect_multiple_characters_from_image,
            )
        except ImportError as e:
            print(f"❌ Multiple character detector not available: {e}")
            return image_paths, [], {"error": str(e)}

        single_char_images = []
        multi_char_images = []

        stats = {
            "total_images": len(image_paths),
            "single_character": 0,
            "multiple_character": 0,
            "filtered_out": 0,
            "detection_types": {},
            "penalty_distribution": [],
        }

        print(f"🔍 {len(image_paths)}枚の画像で単一キャラクターフィルタリング開始")

        for i, image_path in enumerate(image_paths):
            try:
                # 複数キャラクター検出実行
                result = detect_multiple_characters_from_image(
                    image_path, self, save_visualization=save_reports
                )

                # 統計更新
                stats["penalty_distribution"].append(result.penalty_score)
                detection_type = result.detection_type.value
                stats["detection_types"][detection_type] = (
                    stats["detection_types"].get(detection_type, 0) + 1
                )

                # フィルタリング判定
                if result.is_multiple and result.penalty_score > penalty_threshold:
                    multi_char_images.append(image_path)
                    stats["filtered_out"] += 1
                    print(
                        f"   🚫 {image_path.name}: ペナルティ{result.penalty_score:.2f} > 閾値{penalty_threshold}"
                    )
                else:
                    single_char_images.append(image_path)
                    if result.is_multiple:
                        stats["multiple_character"] += 1
                        print(
                            f"   ⚠️  {image_path.name}: 複数キャラだが許容範囲(ペナルティ{result.penalty_score:.2f})"
                        )
                    else:
                        stats["single_character"] += 1
                        print(f"   ✅ {image_path.name}: 単一キャラクター")

                # 進捗表示
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(image_paths) * 100
                    print(f"   📊 進捗: {i + 1}/{len(image_paths)} ({progress:.1f}%)")

            except Exception as e:
                print(f"   ❌ {image_path.name}: 分析エラー - {e}")
                # エラー時はデフォルトで許可
                single_char_images.append(image_path)

        # 統計計算
        if stats["penalty_distribution"]:
            stats["average_penalty"] = np.mean(stats["penalty_distribution"])
            stats["max_penalty"] = max(stats["penalty_distribution"])
            stats["filtering_rate"] = stats["filtered_out"] / stats["total_images"] * 100

        print(
            f"✅ フィルタリング完了: {stats['single_character']}枚採用, {stats['filtered_out']}枚除外 "
            f"({stats['filtering_rate']:.1f}%除外率)"
        )

        return single_char_images, multi_char_images, stats

    def _select_best_character_with_criteria(
        self,
        masks: List[Dict[str, Any]],
        image_shape: tuple,
        criteria: str = "balanced",
        original_image: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        複合スコアによる最適キャラクター選択 (Gemini提案実装)

        Args:
            masks: マスク候補リスト
            image_shape: 画像サイズ (height, width, channels)
            criteria: 選択基準 ('balanced', 'size_priority', 'fullbody_priority', 'fullbody_priority_enhanced', 'central_priority', 'confidence_priority')
            original_image: 元画像（改良版全身検出に必要）

        Returns:
            (最適マスク, 品質スコア) または None
        """
        if not masks:
            return None

        h, w = image_shape[:2]
        image_center_x, image_center_y = w / 2, h / 2

        def calculate_composite_score(mask_data: Dict[str, Any]) -> Dict[str, float]:
            """複合スコア計算"""
            scores = {}

            # 1. 面積スコア (30%): 適度な大きさを評価
            area_ratio = mask_data["area"] / (h * w)
            if 0.05 <= area_ratio <= 0.4:  # 画像の5-40%が理想的
                scores["area"] = min(area_ratio / 0.4, 1.0)
            else:
                scores["area"] = max(0, 1.0 - abs(area_ratio - 0.2) / 0.2)

            # 2. アスペクト比スコア (25%): 全身キャラクターを優先
            bbox = mask_data["bbox"]
            aspect_ratio = bbox[3] / max(bbox[2], 1)  # height / width

            # Phase 1 P1-007: Enhanced Aspect Ratio Analyzer統合
            if (
                criteria in ["fullbody_priority_enhanced", "aspect_ratio_enhanced"]
                and original_image is not None
            ):
                try:
                    # P1-007: Enhanced Aspect Ratio Analysis
                    from features.evaluation.utils.enhanced_aspect_ratio_analyzer import (
                        evaluate_enhanced_aspect_ratio,
                    )

                    # Enhanced aspect ratio analysis
                    enhanced_fullbody_score, aspect_analysis = evaluate_enhanced_aspect_ratio(
                        original_image, mask_data, aspect_ratio
                    )

                    scores["fullbody"] = enhanced_fullbody_score
                    print(
                        f"   P1-007 Enhanced aspect ratio: {enhanced_fullbody_score:.3f} "
                        f"(style: {aspect_analysis.style_category.value}, "
                        f"ratio: {aspect_analysis.adjusted_ratio:.2f})"
                    )

                    # Fallback to P1-003 if P1-007 gives low confidence
                    if aspect_analysis.confidence_score < 0.4:
                        try:
                            from features.evaluation.utils.enhanced_fullbody_detector import (
                                evaluate_fullbody_enhanced,
                            )

                            fullbody_result = evaluate_fullbody_enhanced(original_image, mask_data)
                            # Blend P1-007 and P1-003 results based on confidence
                            blend_factor = aspect_analysis.confidence_score
                            scores["fullbody"] = (
                                enhanced_fullbody_score * blend_factor
                                + fullbody_result.total_score * (1 - blend_factor)
                            )
                            print(f"   Blended with P1-003: {scores['fullbody']:.3f}")
                        except Exception as e:
                            print(f"   P1-003 fallback failed: {e}")

                except Exception as e:
                    print(f"   P1-007 Enhanced aspect ratio error: {e}, using traditional fallback")
                    # 従来のアスペクト比ベース判定にフォールバック
                    if 1.2 <= aspect_ratio <= 2.5:
                        scores["fullbody"] = min((aspect_ratio - 0.5) / 2.0, 1.0)
                    else:
                        scores["fullbody"] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)

            elif criteria == "fullbody_priority_enhanced" and original_image is None:
                print("   Enhanced fullbody: no image provided, using P1-003 fallback")
                # P1-003システムの使用を試行
                try:
                    from features.evaluation.utils.enhanced_fullbody_detector import (
                        evaluate_fullbody_enhanced,
                    )

                    # 画像なしでも使用可能な場合
                    enhanced_score = evaluate_fullbody_enhanced(None, mask_data)
                    scores["fullbody"] = enhanced_score.total_score
                except Exception:
                    # 完全なフォールバック
                    if 1.2 <= aspect_ratio <= 2.5:
                        scores["fullbody"] = min((aspect_ratio - 0.5) / 2.0, 1.0)
                    else:
                        scores["fullbody"] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)

            else:
                # 従来のアスペクト比ベース判定
                if 1.2 <= aspect_ratio <= 2.5:  # 全身キャラクター範囲
                    scores["fullbody"] = min((aspect_ratio - 0.5) / 2.0, 1.0)
                else:
                    scores["fullbody"] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)

            # 3. 中央位置スコア (20%): 画像中央に近いキャラクターを優先
            mask_center_x = bbox[0] + bbox[2] / 2
            mask_center_y = bbox[1] + bbox[3] / 2
            distance_from_center = np.sqrt(
                ((mask_center_x - image_center_x) / w) ** 2
                + ((mask_center_y - image_center_y) / h) ** 2
            )
            scores["central"] = max(0, 1.0 - distance_from_center)

            # 4. 接地スコア (15%): 画面下部にいるキャラクターを優先
            bottom_position = (bbox[1] + bbox[3]) / h
            if bottom_position >= 0.6:  # 下部60%以降
                scores["grounded"] = min(bottom_position, 1.0)
            else:
                scores["grounded"] = bottom_position / 0.6

            # 5. YOLO信頼度スコア (10%)
            scores["confidence"] = mask_data.get("yolo_confidence", 0.0)

            return scores

        # 基準別の重み設定
        weight_configs = {
            "balanced": {
                "area": 0.30,
                "fullbody": 0.25,
                "central": 0.20,
                "grounded": 0.15,
                "confidence": 0.10,
            },
            "size_priority": {
                "area": 0.50,
                "fullbody": 0.15,
                "central": 0.15,
                "grounded": 0.10,
                "confidence": 0.10,
            },
            "fullbody_priority": {
                "area": 0.20,
                "fullbody": 0.40,
                "central": 0.15,
                "grounded": 0.15,
                "confidence": 0.10,
            },
            "fullbody_priority_enhanced": {
                "area": 0.15,
                "fullbody": 0.50,
                "central": 0.15,
                "grounded": 0.10,
                "confidence": 0.10,
            },  # Phase 1 P1-003
            "aspect_ratio_enhanced": {
                "area": 0.10,
                "fullbody": 0.60,
                "central": 0.15,
                "grounded": 0.10,
                "confidence": 0.05,
            },  # Phase 1 P1-007
            "central_priority": {
                "area": 0.20,
                "fullbody": 0.20,
                "central": 0.35,
                "grounded": 0.15,
                "confidence": 0.10,
            },
            "confidence_priority": {
                "area": 0.25,
                "fullbody": 0.20,
                "central": 0.15,
                "grounded": 0.10,
                "confidence": 0.30,
            },
        }

        weights = weight_configs.get(criteria, weight_configs["balanced"])

        # 各マスクのスコア計算
        best_mask = None
        best_score = 0.0

        print(f"🎯 複合スコア評価開始 (基準: {criteria})")

        for i, mask_data in enumerate(masks):
            scores = calculate_composite_score(mask_data)

            # 重み付き総合スコア計算
            composite_score = sum(scores[key] * weights[key] for key in weights.keys())

            print(
                f"   マスク{i+1}: 総合={composite_score:.3f} "
                f"(面積={scores['area']:.2f}, 全身={scores['fullbody']:.2f}, "
                f"中央={scores['central']:.2f}, 接地={scores['grounded']:.2f}, "
                f"信頼度={scores['confidence']:.2f})"
            )

            if composite_score > best_score:
                best_score = composite_score
                best_mask = mask_data

        if best_mask is not None:
            print(f"✅ 最適マスク選択: 総合スコア {best_score:.3f}")
            return best_mask, best_score

        return None

    def select_best_mask_with_criteria(
        self, masks: List[Dict[str, Any]], image: np.ndarray, criteria: str = "balanced"
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        公開メソッド: 複合スコアによる最適マスク選択

        Args:
            masks: マスク候補リスト
            image: 元画像
            criteria: 選択基準

        Returns:
            (最適マスク, 品質スコア) または None
        """
        if not masks:
            return None

        image_shape = image.shape
        return self._select_best_character_with_criteria(masks, image_shape, criteria, image)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "available": YOLO_AVAILABLE,
            "person_class_id": self.person_class_id,
        }

    def unload_model(self):
        """
        Unload model and free memory
        """
        if self.model is not None:
            del self.model
            self.model = None

        self.is_loaded = False

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🗑️ YOLO model unloaded")


if __name__ == "__main__":
    # Test the wrapper
    wrapper = YOLOModelWrapper()

    if wrapper.load_model():
        print("✅ YOLO wrapper test successful")
        print(f"Model info: {wrapper.get_model_info()}")

        # Test with a sample image if available
        test_image_path = "../assets/masks1.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)

            persons = wrapper.detect_persons(image)
            print(f"Detected {len(persons)} persons")

            for i, person in enumerate(persons):
                print(
                    f"  Person {i+1}: confidence={person['confidence']:.3f}, bbox={person['bbox']}"
                )

        wrapper.unload_model()
    else:
        print("❌ YOLO wrapper test failed")
