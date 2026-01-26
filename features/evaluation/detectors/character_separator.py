"""
QI-002: キャラクター分離器 (CharacterSeparator)

複数キャラクターを含む画像から、個別のキャラクター画像を分離・抽出します。
"""

import numpy as np
import cv2

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from .multi_character_detector import CharacterRegion, MultiCharacterDetector


class CharacterSeparationResult(NamedTuple):
    """キャラクター分離結果"""

    character_count: int
    individual_masks: List[np.ndarray]
    character_images: List[np.ndarray]
    has_overlapping_regions: bool
    separation_confidence: float
    separation_quality_scores: List[float]
    additional_info: Optional[dict] = None


class CharacterSeparator:
    """キャラクター分離を行うクラス"""

    def __init__(
        self, min_separation_distance: int = 20, overlap_resolution_method: str = "watershed"
    ):
        """
        CharacterSeparator の初期化

        Args:
            min_separation_distance: 最小分離距離（ピクセル）
            overlap_resolution_method: 重複解決手法 ('watershed', 'erosion', 'contour')
        """
        self.min_separation_distance = min_separation_distance
        self.overlap_resolution_method = overlap_resolution_method
        self.detector = MultiCharacterDetector()

    def separate_characters(self, image: np.ndarray) -> CharacterSeparationResult:
        """
        画像から複数キャラクターを分離

        Args:
            image: 入力画像 (H, W, C) numpy配列

        Returns:
            CharacterSeparationResult: 分離結果
        """
        try:
            # まず複数キャラクターを検出
            detection_result = self.detector.detect_characters(image)

            if detection_result.character_count == 0:
                return self._empty_separation_result()

            # 各キャラクター領域のマスクを作成
            individual_masks = self._create_individual_masks(
                image, detection_result.character_regions
            )

            # 重複解決
            if detection_result.has_overlapping_characters:
                individual_masks = self._resolve_overlaps(image, individual_masks)

            # 個別キャラクター画像の抽出
            character_images = self._extract_character_images(image, individual_masks)

            # 分離品質の評価
            quality_scores = self._evaluate_separation_quality(
                image, individual_masks, character_images
            )

            return CharacterSeparationResult(
                character_count=len(individual_masks),
                individual_masks=individual_masks,
                character_images=character_images,
                has_overlapping_regions=detection_result.has_overlapping_characters,
                separation_confidence=np.mean(quality_scores) if quality_scores else 0.0,
                separation_quality_scores=quality_scores,
                additional_info={
                    "detection_info": detection_result,
                    "overlap_resolution_method": self.overlap_resolution_method,
                },
            )

        except Exception as e:
            return CharacterSeparationResult(
                character_count=0,
                individual_masks=[],
                character_images=[],
                has_overlapping_regions=False,
                separation_confidence=0.0,
                separation_quality_scores=[],
                additional_info={"error": str(e)},
            )

    def _empty_separation_result(self) -> CharacterSeparationResult:
        """空の分離結果を返す"""
        return CharacterSeparationResult(
            character_count=0,
            individual_masks=[],
            character_images=[],
            has_overlapping_regions=False,
            separation_confidence=0.0,
            separation_quality_scores=[],
        )

    def _create_individual_masks(
        self, image: np.ndarray, regions: List[CharacterRegion]
    ) -> List[np.ndarray]:
        """個別キャラクターのマスクを作成"""
        h, w = image.shape[:2]
        individual_masks = []

        for region in regions:
            mask = np.zeros((h, w), dtype=np.uint8)
            x, y, bw, bh = region.bbox

            # バウンディングボックス内での詳細マスク生成
            roi_image = image[y : y + bh, x : x + bw]
            roi_mask = self._generate_character_mask(roi_image)

            # 全体マスクに配置
            mask[y : y + bh, x : x + bw] = roi_mask
            individual_masks.append(mask)

        return individual_masks

    def _generate_character_mask(self, roi_image: np.ndarray) -> np.ndarray:
        """ROI内でのキャラクターマスク生成"""
        # グレースケール変換
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi_image

        # 閾値処理でキャラクター領域を抽出
        # 背景（黒）とキャラクター（非黒）の分離
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _resolve_overlaps(self, image: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
        """重複領域の解決"""
        if len(masks) < 2:
            return masks

        if self.overlap_resolution_method == "watershed":
            return self._watershed_separation(image, masks)
        elif self.overlap_resolution_method == "erosion":
            return self._erosion_separation(masks)
        else:  # contour
            return self._contour_separation(masks)

    def _watershed_separation(self, image: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
        """ウォーターシェッド法による分離"""
        # 全体マスクの作成
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask > 0)

        # 距離変換
        dist_transform = cv2.distanceTransform(combined_mask.astype(np.uint8), cv2.DIST_L2, 5)

        # マーカーの作成
        markers = np.zeros_like(combined_mask, dtype=np.int32)
        for i, mask in enumerate(masks):
            # 各マスクの中心領域をマーカーとして使用
            eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
            markers[eroded_mask > 0] = i + 1

        # ウォーターシェッド実行
        if len(image.shape) == 3:
            watershed_img = image.copy()
        else:
            watershed_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(watershed_img, markers)

        # 結果を個別マスクに分離
        separated_masks = []
        for i in range(len(masks)):
            separated_mask = (markers == i + 1).astype(np.uint8) * 255
            separated_masks.append(separated_mask)

        return separated_masks

    def _erosion_separation(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """侵食による分離"""
        separated_masks = []

        for mask in masks:
            # 侵食処理で重複を削減
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded_mask = cv2.erode(mask, kernel, iterations=1)

            # 膨張で元のサイズに近づける（但し重複は避ける）
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

            separated_masks.append(dilated_mask)

        return separated_masks

    def _contour_separation(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """輪郭ベースの分離"""
        separated_masks = []

        for mask in masks:
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 最大輪郭のみを保持
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                separated_mask = np.zeros_like(mask)
                cv2.fillPoly(separated_mask, [largest_contour], 255)
                separated_masks.append(separated_mask)
            else:
                separated_masks.append(mask)

        return separated_masks

    def _extract_character_images(
        self, image: np.ndarray, masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """マスクを使って個別キャラクター画像を抽出"""
        character_images = []

        for mask in masks:
            # マスクを適用
            masked_image = image.copy()

            if len(image.shape) == 3:
                for c in range(3):
                    masked_image[:, :, c] = np.where(mask > 0, image[:, :, c], 0)
            else:
                masked_image = np.where(mask > 0, image, 0)

            # バウンディングボックスでクロップ
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0 and len(x_coords) > 0:
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()

                # 余白を追加
                padding = 10
                y_min = max(0, y_min - padding)
                y_max = min(image.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(image.shape[1], x_max + padding)

                cropped_image = masked_image[y_min : y_max + 1, x_min : x_max + 1]
                character_images.append(cropped_image)
            else:
                # 空のマスクの場合
                if len(image.shape) == 3:
                    empty_image = np.zeros((100, 100, 3), dtype=image.dtype)
                else:
                    empty_image = np.zeros((100, 100), dtype=image.dtype)
                character_images.append(empty_image)

        return character_images

    def _evaluate_separation_quality(
        self,
        original_image: np.ndarray,
        masks: List[np.ndarray],
        character_images: List[np.ndarray],
    ) -> List[float]:
        """分離品質の評価"""
        quality_scores = []

        for i, (mask, char_image) in enumerate(zip(masks, character_images)):
            score = 0.0

            # 1. マスクの連結性評価
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 1:
                score += 0.3  # 単一連結成分
            elif len(contours) <= 3:
                score += 0.2  # 少数の連結成分
            else:
                score += 0.1  # 多数の断片

            # 2. サイズ適正性評価
            mask_area = np.sum(mask > 0)
            total_area = mask.shape[0] * mask.shape[1]
            area_ratio = mask_area / total_area

            if 0.05 <= area_ratio <= 0.4:  # 適切なサイズ範囲
                score += 0.3
            else:
                score += 0.1

            # 3. 形状品質評価
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * mask_area / (perimeter**2)
                    if 0.1 <= circularity <= 0.8:  # 人形に適した形状
                        score += 0.2
                    else:
                        score += 0.1
                else:
                    score += 0.05

            # 4. 画像内容の品質
            if char_image.size > 0:
                if len(char_image.shape) == 3:
                    char_gray = cv2.cvtColor(char_image, cv2.COLOR_RGB2GRAY)
                else:
                    char_gray = char_image

                non_zero_ratio = np.sum(char_gray > 10) / char_gray.size
                if non_zero_ratio > 0.1:  # 十分な内容がある
                    score += 0.2
                else:
                    score += 0.1

            quality_scores.append(min(1.0, score))

        return quality_scores

    def create_individual_masks(self, image: np.ndarray, detection_result) -> List[np.ndarray]:
        """外部から呼び出し可能な個別マスク作成メソッド"""
        return self._create_individual_masks(image, detection_result.character_regions)

    def get_separation_statistics(self, result: CharacterSeparationResult) -> dict:
        """分離結果の統計情報を取得"""
        if result.character_count == 0:
            return {"status": "no_characters"}

        mask_areas = [np.sum(mask > 0) for mask in result.individual_masks]

        stats = {
            "character_count": result.character_count,
            "separation_confidence": result.separation_confidence,
            "average_quality_score": np.mean(result.separation_quality_scores),
            "mask_areas": mask_areas,
            "total_mask_area": sum(mask_areas),
            "has_overlaps": result.has_overlapping_regions,
            "average_character_size": np.mean(mask_areas) if mask_areas else 0,
            "size_variation": np.std(mask_areas) if len(mask_areas) > 1 else 0,
        }

        return stats
