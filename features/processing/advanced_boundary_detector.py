#!/usr/bin/env python3
"""
Advanced Boundary Detector - Phase 2境界認識強化システム
複雑なコマ割り専用処理とエッジ検出の多段階適用
"""

import numpy as np
import cv2

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdvancedBoundaryDetector:
    """Phase 2境界認識強化システム"""
    
    def __init__(self,
                 enable_panel_detection: bool = True,
                 enable_multi_stage_edge: bool = True,
                 enable_boundary_completion: bool = True):
        """
        Args:
            enable_panel_detection: コマ境界検出の有効化
            enable_multi_stage_edge: 多段階エッジ検出の有効化
            enable_boundary_completion: 境界補完の有効化
        """
        self.enable_panel_detection = enable_panel_detection
        self.enable_multi_stage_edge = enable_multi_stage_edge
        self.enable_boundary_completion = enable_boundary_completion
        
        # エッジ検出の段階設定
        self.edge_stages = [
            {"name": "fine", "low": 30, "high": 80, "weight": 0.4},
            {"name": "medium", "low": 50, "high": 120, "weight": 0.4},
            {"name": "coarse", "low": 80, "high": 180, "weight": 0.2}
        ]
        
        logger.info(f"AdvancedBoundaryDetector初期化: panel={enable_panel_detection}, "
                   f"multi_edge={enable_multi_stage_edge}, completion={enable_boundary_completion}")

    def enhance_boundaries_advanced(self, 
                                  image: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        高度境界強化処理
        
        Args:
            image: 入力画像 (H, W, 3)
            mask: 既存マスク（オプション）
            
        Returns:
            強化された画像と分析結果
        """
        logger.debug(f"高度境界強化開始: {image.shape}")
        
        analysis_result = {
            "panel_info": {},
            "edge_analysis": {},
            "boundary_completion": {},
            "enhancement_quality": 0.0
        }
        
        enhanced_image = image.copy()
        
        # 1. コマ境界検出・分析
        if self.enable_panel_detection:
            panel_info, panel_enhanced = self._detect_and_process_panels(enhanced_image)
            enhanced_image = panel_enhanced
            analysis_result["panel_info"] = panel_info
        
        # 2. 多段階エッジ検出
        if self.enable_multi_stage_edge:
            edge_info, edge_enhanced = self._multi_stage_edge_detection(enhanced_image)
            enhanced_image = edge_enhanced
            analysis_result["edge_analysis"] = edge_info
        
        # 3. 境界補完処理
        if self.enable_boundary_completion:
            if mask is not None:
                completion_info, completion_enhanced = self._boundary_completion(enhanced_image, mask)
                enhanced_image = completion_enhanced
                analysis_result["boundary_completion"] = completion_info
        
        # 4. 全体的な品質評価
        analysis_result["enhancement_quality"] = self._evaluate_enhancement_quality(
            image, enhanced_image, analysis_result
        )
        
        logger.debug("高度境界強化完了")
        return enhanced_image, analysis_result

    def _detect_and_process_panels(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """コマ境界検出・専用処理"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. 基本的なコマ境界線検出
        # 縦線・横線の検出（漫画のコマ境界は直線が多い）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # モルフォロジー演算で線を強調
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # 2. コマ境界の統合
        panel_boundaries = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # 3. コマ領域の分割
        panel_regions = self._identify_panel_regions(panel_boundaries, image.shape)
        
        # 4. L字型・不規則コマの検出
        irregular_panels = self._detect_irregular_panels(panel_boundaries)
        
        # 5. コマ外はみ出し領域の処理
        overflow_regions = self._detect_character_overflow(image, panel_boundaries)
        
        # 6. コマ情報に基づく画像強化
        enhanced_image = self._enhance_based_on_panels(
            image, panel_regions, irregular_panels, overflow_regions
        )
        
        panel_info = {
            "panel_count": len(panel_regions),
            "irregular_count": len(irregular_panels),
            "overflow_regions": len(overflow_regions),
            "panel_complexity": self._calculate_panel_complexity(panel_boundaries),
            "dominant_panel_type": self._classify_panel_layout(panel_regions)
        }
        
        logger.debug(f"コマ検出結果: {panel_info}")
        return panel_info, enhanced_image

    def _identify_panel_regions(self, boundaries: np.ndarray, image_shape: Tuple) -> List[Dict[str, Any]]:
        """コマ領域の特定"""
        height, width = image_shape[:2]
        
        # 輪郭検出
        contours, _ = cv2.findContours(boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        panel_regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # 小さすぎる領域は除外
            if area < (width * height * 0.05):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            panel_regions.append({
                "id": i,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "center": (x + w//2, y + h//2),
                "contour": contour
            })
        
        # 面積順でソート（大きいコマから処理）
        panel_regions.sort(key=lambda x: x["area"], reverse=True)
        
        return panel_regions

    def _detect_irregular_panels(self, boundaries: np.ndarray) -> List[Dict[str, Any]]:
        """L字型・不規則コマの検出"""
        # 複雑な形状のコマを検出
        contours, _ = cv2.findContours(boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        irregular_panels = []
        for i, contour in enumerate(contours):
            # 凸包と元の輪郭の面積比で複雑さを判定
            hull = cv2.convexHull(contour)
            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                complexity_ratio = contour_area / hull_area
                # 複雑な形状（凹みが多い）を検出
                if complexity_ratio < 0.8:
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    
                    irregular_panels.append({
                        "id": i,
                        "contour": contour,
                        "complexity_ratio": complexity_ratio,
                        "vertex_count": len(approx),
                        "type": self._classify_irregular_shape(approx)
                    })
        
        return irregular_panels

    def _classify_irregular_shape(self, approx: np.ndarray) -> str:
        """不規則形状の分類"""
        vertex_count = len(approx)
        
        if vertex_count <= 4:
            return "rectangular"
        elif vertex_count <= 6:
            return "l_shaped"
        elif vertex_count <= 8:
            return "complex_polygon"
        else:
            return "very_complex"

    def _detect_character_overflow(self, 
                                 image: np.ndarray, 
                                 boundaries: np.ndarray) -> List[Dict[str, Any]]:
        """コマ外はみ出しキャラクターの検出"""
        # エッジの強い領域（キャラクターの可能性）を検出
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 境界線付近でエッジが強い領域を探す
        boundary_dilated = cv2.dilate(boundaries, np.ones((20, 20), np.uint8), iterations=1)
        
        # 境界線周辺のエッジを抽出
        overflow_candidates = cv2.bitwise_and(edges, boundary_dilated)
        
        # 連結成分分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(overflow_candidates)
        
        overflow_regions = []
        for i in range(1, num_labels):  # 0はバックグラウンド
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100:  # 十分な大きさのエッジ塊
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                overflow_regions.append({
                    "bbox": (x, y, w, h),
                    "area": area,
                    "center": centroids[i],
                    "edge_density": area / (w * h) if w * h > 0 else 0
                })
        
        return overflow_regions

    def _enhance_based_on_panels(self, 
                               image: np.ndarray,
                               panel_regions: List[Dict[str, Any]],
                               irregular_panels: List[Dict[str, Any]],
                               overflow_regions: List[Dict[str, Any]]) -> np.ndarray:
        """コマ情報に基づく画像強化"""
        enhanced = image.copy().astype(np.float32)
        
        # 1. 各コマ領域での適応的処理
        for panel in panel_regions:
            x, y, w, h = panel["bbox"]
            panel_roi = enhanced[y:y+h, x:x+w]
            
            # コマサイズに応じた処理強度調整
            if panel["area"] < (image.shape[0] * image.shape[1] * 0.2):
                # 小さいコマ: より強い強調
                panel_roi *= 1.15
            else:
                # 大きいコマ: 控えめな強調
                panel_roi *= 1.05
            
            enhanced[y:y+h, x:x+w] = panel_roi
        
        # 2. 不規則コマでの特別処理
        for irregular in irregular_panels:
            # 複雑な形状のコマ周辺のエッジを強化
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [irregular["contour"]], 255)
            
            # マスク領域のエッジを強化
            gray = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # エッジ部分のコントラストを向上
            edge_mask = (edges > 0) & (mask > 0)
            for c in range(3):
                channel = enhanced[:, :, c]
                channel[edge_mask] = np.clip(channel[edge_mask] * 1.3, 0, 255)
        
        # 3. はみ出し領域の保護処理
        for overflow in overflow_regions:
            x, y, w, h = overflow["bbox"]
            # はみ出し領域の境界を明確化
            roi = enhanced[y:y+h, x:x+w]
            roi *= 1.2  # 強めの強調
            enhanced[y:y+h, x:x+w] = roi
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _calculate_panel_complexity(self, boundaries: np.ndarray) -> float:
        """コマ境界の複雑さを計算"""
        # 境界線の総長さと面積から複雑さを評価
        contours, _ = cv2.findContours(boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        
        if total_area > 0:
            # 周囲長の2乗を面積で割った値（円形度の逆数）
            complexity = (total_perimeter ** 2) / total_area
            return min(complexity / 100.0, 10.0)  # 正規化
        
        return 1.0

    def _classify_panel_layout(self, panel_regions: List[Dict[str, Any]]) -> str:
        """コマレイアウトの分類"""
        if len(panel_regions) <= 2:
            return "simple"
        elif len(panel_regions) <= 4:
            return "standard"
        elif len(panel_regions) <= 6:
            return "complex"
        else:
            return "very_complex"

    def _multi_stage_edge_detection(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """多段階エッジ検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 各段階でのエッジ検出
        stage_edges = []
        edge_densities = []
        
        for stage in self.edge_stages:
            edges = cv2.Canny(gray, stage["low"], stage["high"])
            stage_edges.append(edges)
            
            # エッジ密度計算
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
        
        # 重み付き統合
        combined_edges = np.zeros_like(stage_edges[0], dtype=np.float32)
        for edges, stage in zip(stage_edges, self.edge_stages):
            combined_edges += edges.astype(np.float32) * stage["weight"]
        
        combined_edges = np.clip(combined_edges, 0, 255).astype(np.uint8)
        
        # エッジ情報に基づく画像強化
        enhanced_image = self._enhance_with_edges(image, combined_edges)
        
        edge_info = {
            "stage_densities": edge_densities,
            "combined_density": np.sum(combined_edges > 0) / combined_edges.size,
            "dominant_stage": self.edge_stages[np.argmax(edge_densities)]["name"],
            "edge_uniformity": np.std(edge_densities)
        }
        
        logger.debug(f"多段階エッジ検出結果: {edge_info}")
        return edge_info, enhanced_image

    def _enhance_with_edges(self, image: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """エッジ情報を用いた画像強化"""
        enhanced = image.copy().astype(np.float32)
        
        # エッジ部分のコントラストを強化
        edge_mask = edges > 50
        
        for channel in range(3):
            channel_data = enhanced[:, :, channel]
            # エッジ部分を適度に強調
            channel_data[edge_mask] = np.clip(channel_data[edge_mask] * 1.1, 0, 255)
            enhanced[:, :, channel] = channel_data
        
        return enhanced.astype(np.uint8)

    def _boundary_completion(self, 
                           image: np.ndarray, 
                           mask: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """境界補完処理"""
        # マスクの境界を解析
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask.copy()
        
        # 境界の輪郭を取得
        contours, _ = cv2.findContours(mask_gray > 0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"completion_applied": False}, image
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # 境界の滑らかさを評価
        smoothness = self._evaluate_boundary_smoothness(main_contour)
        
        enhanced_image = image.copy()
        
        # 境界が粗い場合は補完処理を適用
        if smoothness < 0.8:
            # 境界周辺の強化
            boundary_mask = np.zeros_like(mask_gray)
            cv2.drawContours(boundary_mask, [main_contour], -1, 255, thickness=3)
            
            # 境界周辺のコントラストを強化
            enhanced_image = self._enhance_boundary_region(enhanced_image, boundary_mask)
        
        completion_info = {
            "completion_applied": smoothness < 0.8,
            "boundary_smoothness": smoothness,
            "contour_length": cv2.arcLength(main_contour, True),
            "contour_area": cv2.contourArea(main_contour)
        }
        
        return completion_info, enhanced_image

    def _evaluate_boundary_smoothness(self, contour: np.ndarray) -> float:
        """境界の滑らかさを評価"""
        if len(contour) < 5:
            return 1.0
        
        # 輪郭の近似精度で滑らかさを判定
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 近似後の頂点数が少ないほど滑らか
        smoothness = 1.0 - (len(approx) / len(contour))
        return max(0.0, min(1.0, smoothness))

    def _enhance_boundary_region(self, image: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
        """境界領域の強化"""
        enhanced = image.copy().astype(np.float32)
        
        # 境界マスクを膨張させて周辺領域も含める
        kernel = np.ones((5, 5), np.uint8)
        expanded_boundary = cv2.dilate(boundary_mask, kernel, iterations=2)
        
        # 境界周辺領域のコントラストを適度に強化
        boundary_pixels = expanded_boundary > 0
        
        for channel in range(3):
            channel_data = enhanced[:, :, channel]
            channel_data[boundary_pixels] = np.clip(channel_data[boundary_pixels] * 1.08, 0, 255)
            enhanced[:, :, channel] = channel_data
        
        return enhanced.astype(np.uint8)

    def _evaluate_enhancement_quality(self, 
                                    original: np.ndarray,
                                    enhanced: np.ndarray, 
                                    analysis_result: Dict[str, Any]) -> float:
        """強化品質の総合評価"""
        # 各要素の品質スコア
        panel_score = min(1.0, analysis_result["panel_info"].get("panel_count", 0) / 6.0)
        edge_score = analysis_result["edge_analysis"].get("combined_density", 0) * 10.0
        
        completion_score = 1.0
        if analysis_result["boundary_completion"]:
            completion_score = analysis_result["boundary_completion"].get("boundary_smoothness", 1.0)
        
        # 重み付き平均
        overall_quality = (panel_score * 0.3 + 
                         min(edge_score, 1.0) * 0.4 + 
                         completion_score * 0.3)
        
        return min(1.0, overall_quality)


def test_advanced_boundary_detector():
    """高度境界検出システムのテスト"""
    detector = AdvancedBoundaryDetector(
        enable_panel_detection=True,
        enable_multi_stage_edge=True,
        enable_boundary_completion=True
    )
    
    # テスト画像
    test_image_path = Path("/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0002.jpg")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"テスト画像読み込み: {image.shape}")
        
        # 高度境界強化実行
        enhanced, analysis = detector.enhance_boundaries_advanced(image)
        
        # 分析結果表示
        print("\\n📊 高度境界強化分析結果:")
        print(f"コマ情報: {analysis['panel_info']}")
        print(f"エッジ分析: {analysis['edge_analysis']}")
        print(f"境界補完: {analysis['boundary_completion']}")
        print(f"全体品質: {analysis['enhancement_quality']:.3f}")
        
        # 結果保存
        output_path = Path("/tmp/advanced_boundary_test.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        print(f"\\n💾 結果保存: {output_path}")
    else:
        print(f"テスト画像が見つかりません: {test_image_path}")


if __name__ == "__main__":
    test_advanced_boundary_detector()