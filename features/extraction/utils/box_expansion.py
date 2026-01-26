#!/usr/bin/env python3
"""
Box Expansion Utilities
GPT-4O推奨のYOLO検出ボックス拡張機能
顔検出ボックスを2.5-3倍水平、4倍垂直に拡張してからSAM処理
"""

import numpy as np

from typing import Any, Dict, List, Tuple


class BoxExpansionProcessor:
    """
    YOLO検出ボックス拡張処理クラス
    GPT-4O推奨の拡張パラメータに基づく実装
    """

    def __init__(
        self,
        horizontal_expansion: float = 2.75,  # 2.5-3倍の中間値
        vertical_expansion: float = 4.0,  # 4倍
        min_expansion_factor: float = 1.2,  # 最小拡張率
        max_expansion_factor: float = 5.0,
    ):  # 最大拡張率
        """
        ボックス拡張プロセッサを初期化

        Args:
            horizontal_expansion: 水平方向拡張倍率 (2.5-3.0推奨)
            vertical_expansion: 垂直方向拡張倍率 (4.0推奨)
            min_expansion_factor: 最小拡張倍率（安全制限）
            max_expansion_factor: 最大拡張倍率（安全制限）
        """
        self.horizontal_expansion = max(
            min_expansion_factor, min(horizontal_expansion, max_expansion_factor)
        )
        self.vertical_expansion = max(
            min_expansion_factor, min(vertical_expansion, max_expansion_factor)
        )
        self.min_expansion_factor = min_expansion_factor
        self.max_expansion_factor = max_expansion_factor

    def expand_detection_box(
        self, bbox: List[int], image_shape: Tuple[int, int], detection_type: str = "person"
    ) -> Dict[str, Any]:
        """
        YOLO検出ボックスを拡張

        Args:
            bbox: YOLO検出ボックス [x, y, w, h]
            image_shape: 画像サイズ (height, width)
            detection_type: 検出タイプ ('person', 'face' など)

        Returns:
            拡張結果辞書（元ボックス、拡張ボックス、メタデータ）
        """
        x, y, w, h = bbox
        img_height, img_width = image_shape

        # 拡張倍率の適用
        if detection_type == "face":
            # 顔検出の場合はより大きく拡張（全身を含む可能性）
            h_expansion = self.horizontal_expansion * 1.2
            v_expansion = self.vertical_expansion * 1.1
        else:
            # 人物検出の場合は標準拡張
            h_expansion = self.horizontal_expansion
            v_expansion = self.vertical_expansion

        # 新しい幅と高さを計算
        new_w = int(w * h_expansion)
        new_h = int(h * v_expansion)

        # 中心点を維持して拡張
        center_x = x + w // 2
        center_y = y + h // 2

        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2

        # 画像境界内に制限
        new_x = max(0, min(new_x, img_width - new_w))
        new_y = max(0, min(new_y, img_height - new_h))
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)

        # 拡張結果
        expanded_bbox = [new_x, new_y, new_w, new_h]

        return {
            "original_bbox": bbox,
            "expanded_bbox": expanded_bbox,
            "expansion_factors": {
                "horizontal": new_w / w if w > 0 else 0,
                "vertical": new_h / h if h > 0 else 0,
            },
            "detection_type": detection_type,
            "clipped_to_bounds": (
                new_x != center_x - new_w // 2
                or new_y != center_y - new_h // 2
                or new_w != int(w * h_expansion)
                or new_h != int(h * v_expansion)
            ),
        }

    def expand_all_detections(
        self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        複数のYOLO検出結果を一括拡張

        Args:
            detections: YOLO検出結果リスト
            image_shape: 画像サイズ (height, width)

        Returns:
            拡張済み検出結果リスト
        """
        expanded_detections = []

        for detection in detections:
            bbox = detection["bbox"]  # [x, y, w, h]
            detection_type = detection.get("class_name", "person")

            expansion_result = self.expand_detection_box(bbox, image_shape, detection_type)

            # 元の検出データに拡張情報を追加
            expanded_detection = detection.copy()
            expanded_detection.update(
                {
                    "bbox_original": expansion_result["original_bbox"],
                    "bbox": expansion_result["expanded_bbox"],  # 拡張ボックスで上書き
                    "expansion_info": {
                        "horizontal_factor": expansion_result["expansion_factors"]["horizontal"],
                        "vertical_factor": expansion_result["expansion_factors"]["vertical"],
                        "clipped_to_bounds": expansion_result["clipped_to_bounds"],
                        "expansion_type": f"H{self.horizontal_expansion:.1f}xV{self.vertical_expansion:.1f}",
                    },
                }
            )

            expanded_detections.append(expanded_detection)

        return expanded_detections

    def create_sam_prompts_from_expanded_boxes(
        self, expanded_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        拡張ボックスからSAMプロンプト形式を生成

        Args:
            expanded_detections: 拡張済み検出結果

        Returns:
            SAMプロンプト形式のリスト
        """
        sam_prompts = []

        for detection in expanded_detections:
            bbox = detection["bbox"]  # 拡張済みボックス [x, y, w, h]
            x, y, w, h = bbox

            # SAMのボックスプロンプト形式 [x1, y1, x2, y2]
            sam_box = [x, y, x + w, y + h]

            sam_prompt = {
                "type": "box",
                "coordinates": sam_box,
                "confidence": detection.get("confidence", 0.0),
                "original_detection": detection,
                "expansion_info": detection.get("expansion_info", {}),
            }

            sam_prompts.append(sam_prompt)

        return sam_prompts

    def get_expansion_config(self) -> Dict[str, float]:
        """
        現在の拡張設定を取得

        Returns:
            拡張設定辞書
        """
        return {
            "horizontal_expansion": self.horizontal_expansion,
            "vertical_expansion": self.vertical_expansion,
            "min_expansion_factor": self.min_expansion_factor,
            "max_expansion_factor": self.max_expansion_factor,
            "gpt4o_recommended": {
                "horizontal_range": [2.5, 3.0],
                "vertical_target": 4.0,
                "purpose": "Face detection box expansion for full-body character extraction",
            },
        }


def apply_gpt4o_expansion_strategy(
    detections: List[Dict[str, Any]], image_shape: Tuple[int, int], strategy: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    GPT-4O推奨戦略による検出ボックス拡張

    Args:
        detections: YOLO検出結果
        image_shape: 画像サイズ
        strategy: 拡張戦略 ('conservative', 'balanced', 'aggressive')

    Returns:
        拡張済み検出結果
    """
    strategy_configs = {
        "conservative": {"horizontal": 2.5, "vertical": 3.5},
        "balanced": {"horizontal": 2.75, "vertical": 4.0},  # GPT-4O推奨
        "aggressive": {"horizontal": 3.0, "vertical": 4.5},
    }

    config = strategy_configs.get(strategy, strategy_configs["balanced"])

    processor = BoxExpansionProcessor(
        horizontal_expansion=config["horizontal"], vertical_expansion=config["vertical"]
    )

    expanded_detections = processor.expand_all_detections(detections, image_shape)

    return expanded_detections


if __name__ == "__main__":
    # テスト用のダミーデータ
    test_detections = [
        {"bbox": [100, 150, 80, 120], "confidence": 0.85, "class_name": "person"},
        {"bbox": [300, 200, 60, 80], "confidence": 0.92, "class_name": "face"},
    ]

    test_image_shape = (720, 1280)  # height, width

    # GPT-4O推奨戦略をテスト
    expanded = apply_gpt4o_expansion_strategy(test_detections, test_image_shape, "balanced")

    print("🎯 GPT-4O推奨ボックス拡張テスト:")
    for i, detection in enumerate(expanded):
        print(f"  検出{i+1}:")
        print(f"    元ボックス: {detection['bbox_original']}")
        print(f"    拡張ボックス: {detection['bbox']}")
        print(
            f"    拡張倍率: H{detection['expansion_info']['horizontal_factor']:.1f}x V{detection['expansion_info']['vertical_factor']:.1f}x"
        )
        print(f"    境界制限: {'あり' if detection['expansion_info']['clipped_to_bounds'] else 'なし'}")
