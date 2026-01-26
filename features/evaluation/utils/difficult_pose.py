"""
複雑ポーズ・ダイナミック構図専用処理モジュール
失敗しやすい画像に対する特別な処理ロジック
Phase 2: エフェクト線除去・マルチコマ分割対応
"""

import numpy as np
import cv2

from features.processing.preprocessing.manga_preprocessing import MangaPreprocessor
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple


class DifficultPoseProcessor:
    """複雑ポーズ専用処理クラス"""

    def __init__(self):
        self.retry_configs = self._generate_retry_configs()
        self.manga_processor = MangaPreprocessor()

    def _generate_retry_configs(self) -> List[Dict[str, Any]]:
        """段階的リトライ用の設定を生成 (Phase 2対応版)"""
        return [
            # Stage 1: 標準より少し緩い設定
            {
                "min_yolo_score": 0.08,
                "sam_points_per_side": 32,
                "sam_pred_iou_thresh": 0.86,
                "sam_stability_score_thresh": 0.90,
                "enable_enhanced_processing": False,
                "enable_manga_preprocessing": False,
                "description": "Stage 1: 軽度緩和",
            },
            # Stage 2: 低閾値設定 + 漫画前処理
            {
                "min_yolo_score": 0.05,
                "sam_points_per_side": 48,
                "sam_pred_iou_thresh": 0.82,
                "sam_stability_score_thresh": 0.88,
                "enable_enhanced_processing": True,
                "enable_manga_preprocessing": True,
                "enable_effect_removal": True,
                "enable_panel_split": False,
                "description": "Stage 2: 低閾値 + エフェクト線除去",
            },
            # Stage 3: 極低閾値 + 高密度処理 + マルチコマ分割
            {
                "min_yolo_score": 0.02,
                "sam_points_per_side": 64,
                "sam_pred_iou_thresh": 0.78,
                "sam_stability_score_thresh": 0.85,
                "enable_enhanced_processing": True,
                "enable_manga_preprocessing": True,
                "enable_effect_removal": True,
                "enable_panel_split": True,
                "description": "Stage 3: 極低閾値 + マルチコマ分割",
            },
            # Stage 4: 最終手段 - 最も緩い設定 + 全機能
            {
                "min_yolo_score": 0.01,
                "sam_points_per_side": 96,
                "sam_pred_iou_thresh": 0.75,
                "sam_stability_score_thresh": 0.80,
                "enable_enhanced_processing": True,
                "enable_manga_preprocessing": True,
                "enable_effect_removal": True,
                "enable_panel_split": True,
                "crop_before_processing": True,
                "description": "Stage 4: 最終手段 + 全機能",
            },
        ]

    def detect_pose_complexity(self, image_path: str) -> Dict[str, Any]:
        """
        ポーズの複雑度を判定 (Phase 2対応版: エフェクト線・マルチコマ検出強化)

        Args:
            image_path: 画像ファイルパス

        Returns:
            Dict: 複雑度判定結果
        """
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                return {"complexity": "unknown", "score": 0.0, "factors": []}

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 複雑度判定要素
            factors = []
            complexity_score = 0.0

            # 1. エッジ密度（集中線、エフェクト線検出）
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            if edge_density > 0.15:
                factors.append("high_edge_density")
                complexity_score += 2.0
            elif edge_density > 0.10:
                factors.append("medium_edge_density")
                complexity_score += 1.0

            # 2. 対比の激しさ（明暗の変化）
            contrast = np.std(gray)
            if contrast > 80:
                factors.append("high_contrast")
                complexity_score += 1.5
            elif contrast > 60:
                factors.append("medium_contrast")
                complexity_score += 0.5

            # 3. 線の方向性（放射状パターン検出 - 集中線の特徴）
            # ハフ変換で直線検出
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is not None and len(lines) > 50:
                factors.append("many_lines")
                complexity_score += 1.5

                # 放射状パターン検出
                angles = []
                for line in lines[:100]:  # 最大100本まで
                    rho, theta = line[0]
                    angles.append(theta)

                # 角度の分散を計算（放射状なら分散が大きい）
                if len(angles) > 10:
                    angle_std = np.std(angles)
                    if angle_std > 0.8:
                        factors.append("radial_pattern")
                        complexity_score += 2.0

            # 4. エフェクト線密度検出 (Phase 2新機能)
            effect_lines, effect_density = self.manga_processor.effect_remover.detect_effect_lines(
                image
            )
            if effect_density > 0.02:
                factors.append("high_effect_lines")
                complexity_score += 2.5
            elif effect_density > 0.01:
                factors.append("medium_effect_lines")
                complexity_score += 1.0

            # 5. マルチコマ構成検出 (Phase 2新機能)
            panel_borders = self.manga_processor.panel_splitter.detect_panel_borders(image)
            if len(panel_borders) > 3:
                factors.append("multi_panel_layout")
                complexity_score += 2.0
            elif len(panel_borders) > 1:
                factors.append("partial_panel_borders")
                complexity_score += 1.0

            # 6. テキスト領域密度
            # 文字らしき領域を検出（小さな矩形の密集）
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # 文字サイズ程度
                    small_contours += 1

            text_density = small_contours / (h * w / 10000)  # 10000ピクセルあたりの小領域数
            if text_density > 5:
                factors.append("high_text_density")
                complexity_score += 1.0

            # 7. 画像の縦横比（極端な比率は複雑な構図の可能性）
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > 2.0:
                factors.append("extreme_aspect_ratio")
                complexity_score += 0.5

            # 複雑度レベル判定 (Phase 2対応で閾値調整)
            if complexity_score >= 6.0:
                complexity_level = "very_high"
            elif complexity_score >= 4.0:
                complexity_level = "high"
            elif complexity_score >= 2.5:
                complexity_level = "medium"
            else:
                complexity_level = "low"

            return {
                "complexity": complexity_level,
                "score": complexity_score,
                "factors": factors,
                "edge_density": edge_density,
                "contrast": contrast,
                "line_count": len(lines) if lines is not None else 0,
                "text_density": text_density,
                "effect_line_density": effect_density,
                "panel_borders_count": len(panel_borders),
            }

        except Exception as e:
            print(f"⚠️ ポーズ複雑度判定エラー: {e}")
            return {"complexity": "unknown", "score": 0.0, "factors": []}

    def get_recommended_config(self, complexity_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        複雑度に基づいて推奨設定を返す

        Args:
            complexity_info: detect_pose_complexity の結果

        Returns:
            Dict: 推奨設定
        """
        complexity = complexity_info.get("complexity", "medium")
        score = complexity_info.get("score", 0.0)

        if complexity == "very_high" or score >= 5.0:
            return self.retry_configs[3]  # Stage 4: 最終手段
        elif complexity == "high" or score >= 3.5:
            return self.retry_configs[2]  # Stage 3: 極低閾値
        elif complexity == "medium" or score >= 2.0:
            return self.retry_configs[1]  # Stage 2: 低閾値
        else:
            return self.retry_configs[0]  # Stage 1: 軽度緩和

    def preprocess_for_difficult_pose(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        enable_manga_preprocessing: bool = False,
        enable_effect_removal: bool = False,
        enable_panel_split: bool = False,
        enable_solid_fill_detection: bool = False,
    ) -> str:
        """
        複雑ポーズ用の前処理 (Phase 2対応版)

        Args:
            image_path: 入力画像パス
            output_path: 出力パス（Noneの場合は一時ファイル）
            enable_manga_preprocessing: 漫画前処理を有効化
            enable_effect_removal: エフェクト線除去を有効化
            enable_panel_split: マルチコマ分割を有効化
            enable_solid_fill_detection: ソリッドフィル領域検出を有効化

        Returns:
            str: 前処理済み画像パス
        """
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                return image_path

            # 前処理の適用
            processed = image.copy()

            # Phase 2: 漫画前処理
            if enable_manga_preprocessing:
                print(f"🎨 漫画前処理適用中...")
                manga_result = self.manga_processor.preprocess_manga_image(
                    processed,
                    enable_effect_removal=enable_effect_removal,
                    enable_panel_split=enable_panel_split,
                    enable_solid_fill_detection=enable_solid_fill_detection,
                )

                print(f"   エフェクト線検出: {'✅' if manga_result['effect_lines_detected'] else '❌'}")
                print(f"   パネル数: {len(manga_result['panels'])}")
                print(f"   処理段階: {', '.join(manga_result['processing_stages'])}")

                # マルチパネルの場合は最大パネルを使用
                if enable_panel_split and len(manga_result["panels"]) > 1:
                    # 最大面積のパネルを選択
                    best_panel = max(
                        manga_result["panels"], key=lambda p: p[1][2] * p[1][3]
                    )  # width * height
                    processed = best_panel[0]
                    print(f"   最大パネル選択: {best_panel[1]}")
                else:
                    processed = manga_result["processed_image"]

            # 従来の前処理
            # 1. ノイズ除去（トーン・スクリーントーン対策）
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

            # 2. コントラスト調整（適度な強調）
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

            # 3. エッジ保護ガウシアンフィルタ
            processed = cv2.edgePreservingFilter(processed, flags=1, sigma_s=50, sigma_r=0.4)

            # 出力パス決定
            if output_path is None:
                input_path = Path(image_path)
                suffix = "_manga" if enable_manga_preprocessing else ""
                output_path = str(input_path.parent / f"preprocessed{suffix}_{input_path.name}")

            # 保存
            cv2.imwrite(output_path, processed)
            return output_path

        except Exception as e:
            print(f"⚠️ 前処理エラー: {e}")
            return image_path

    def enhance_mask_for_complex_pose(
        self, mask: np.ndarray, original_image: np.ndarray
    ) -> np.ndarray:
        """
        複雑ポーズ用のマスク後処理強化

        Args:
            mask: 元のマスク
            original_image: 元画像

        Returns:
            np.ndarray: 強化されたマスク
        """
        try:
            # マスクのコピー
            enhanced_mask = mask.copy()

            # 1. モルフォロジー演算による穴埋め
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # 小さな穴を埋める
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_small)

            # 2. 連結成分分析による最大領域抽出
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_mask)

            if num_labels > 1:
                # 最大の連結成分を選択（背景を除く）
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                enhanced_mask = (labels == largest_label).astype(np.uint8) * 255

            # 3. 輪郭の平滑化
            contours, _ = cv2.findContours(
                enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 最大の輪郭を取得
                largest_contour = max(contours, key=cv2.contourArea)

                # 輪郭を平滑化
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # 新しいマスクを作成
                enhanced_mask = np.zeros_like(enhanced_mask)
                cv2.fillPoly(enhanced_mask, [smoothed_contour], 255)

            # 4. 最終的な形状補正
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel_small)
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_medium)

            return enhanced_mask

        except Exception as e:
            print(f"⚠️ マスク強化エラー: {e}")
            return mask


def detect_difficult_pose(image_path: str) -> bool:
    """
    画像が複雑ポーズかどうかを簡易判定

    Args:
        image_path: 画像ファイルパス

    Returns:
        bool: 複雑ポーズの場合True
    """
    processor = DifficultPoseProcessor()
    complexity_info = processor.detect_pose_complexity(image_path)

    complexity = complexity_info.get("complexity", "low")
    return complexity in ["high", "very_high"]


def get_difficult_pose_config(image_path: str) -> Dict[str, Any]:
    """
    画像に適した複雑ポーズ処理設定を取得

    Args:
        image_path: 画像ファイルパス

    Returns:
        Dict: 処理設定
    """
    processor = DifficultPoseProcessor()
    complexity_info = processor.detect_pose_complexity(image_path)
    config = processor.get_recommended_config(complexity_info)

    # 複雑度情報も含める
    config["complexity_info"] = complexity_info

    return config


def process_with_retry(image_path: str, extract_function, max_retries: int = 4) -> Dict[str, Any]:
    """
    段階的リトライで画像処理を実行

    Args:
        image_path: 画像ファイルパス
        extract_function: 抽出処理関数
        max_retries: 最大リトライ回数

    Returns:
        Dict: 処理結果
    """
    processor = DifficultPoseProcessor()

    # 複雑度を事前判定
    complexity_info = processor.detect_pose_complexity(image_path)
    print(f"🔍 ポーズ複雑度判定: {complexity_info['complexity']} (スコア: {complexity_info['score']:.1f})")
    print(f"📊 検出要素: {', '.join(complexity_info['factors'])}")

    # 各段階で試行
    for i, config in enumerate(processor.retry_configs[:max_retries]):
        stage = i + 1
        print(f"\n🔄 {config['description']} 実行中...")
        print(f"   YOLO閾値: {config['min_yolo_score']}")
        print(f"   SAMポイント密度: {config['sam_points_per_side']}")

        try:
            # 前処理適用（Stage 2以降）
            if config.get("enable_enhanced_processing", False):
                processed_image_path = processor.preprocess_for_difficult_pose(
                    image_path,
                    enable_manga_preprocessing=config.get("enable_manga_preprocessing", False),
                    enable_effect_removal=config.get("enable_effect_removal", False),
                    enable_panel_split=config.get("enable_panel_split", False),
                )
                print(f"   前処理適用: {processed_image_path}")
            else:
                processed_image_path = image_path

            # 抽出実行
            result = extract_function(processed_image_path, **config)

            if result.get("success", False):
                print(f"✅ {config['description']} で成功!")
                result["retry_stage"] = stage
                result["config_used"] = config["description"]
                result["complexity_info"] = complexity_info
                return result
            else:
                print(f"❌ {config['description']} で失敗: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ {config['description']} で例外: {e}")

    # 全段階失敗
    print(f"💔 全{max_retries}段階のリトライが失敗しました")
    return {
        "success": False,
        "error": f"All {max_retries} retry stages failed",
        "retry_stage": max_retries,
        "complexity_info": complexity_info,
    }


# 使用例とテスト関数
if __name__ == "__main__":
    # テスト用
    test_image = "/mnt/c/AItools/lora/train/yado/org/kana03/25_kana03_0024.jpg"

    processor = DifficultPoseProcessor()
    complexity_info = processor.detect_pose_complexity(test_image)

    print("=== 複雑ポーズ判定テスト ===")
    print(f"画像: {test_image}")
    print(f"複雑度: {complexity_info['complexity']}")
    print(f"スコア: {complexity_info['score']:.2f}")
    print(f"要素: {complexity_info['factors']}")

    config = processor.get_recommended_config(complexity_info)
    print(f"\n推奨設定: {config['description']}")
    print(f"YOLO閾値: {config['min_yolo_score']}")
