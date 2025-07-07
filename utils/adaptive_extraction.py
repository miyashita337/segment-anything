#!/usr/bin/env python3
"""
Phase 4: 適応的抽出範囲調整システム
姿勢複雑度とユーザー要望に基づく動的範囲調整機能
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

class PoseComplexity(Enum):
    """姿勢複雑度レベル"""
    SIMPLE = "simple"          # 単純な立ちポーズ
    DYNAMIC = "dynamic"        # 動的ポーズ
    COMPLEX = "complex"        # 複雑な絡み合い
    EXTREME = "extreme"        # 極端に複雑

@dataclass
class ExtractionRange:
    """抽出範囲設定"""
    expansion_factor: float    # 拡張係数
    min_aspect_ratio: float   # 最小アスペクト比
    max_aspect_ratio: float   # 最大アスペクト比
    preserve_body_parts: bool # 体部位保持フラグ

@dataclass
class PoseAnalysisResult:
    """姿勢分析結果"""
    complexity: PoseComplexity    # 複雑度レベル
    body_parts_detected: List[str] # 検出された体部位
    pose_angle: float            # 姿勢角度
    occlusion_level: float       # 遮蔽レベル
    confidence_score: float      # 分析信頼度

class AdaptiveExtractionRangeAdjuster:
    """適応的抽出範囲調整クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 姿勢複雑度別の基本設定
        self.range_presets = {
            PoseComplexity.SIMPLE: ExtractionRange(
                expansion_factor=1.1,
                min_aspect_ratio=0.6,
                max_aspect_ratio=2.0,
                preserve_body_parts=False
            ),
            PoseComplexity.DYNAMIC: ExtractionRange(
                expansion_factor=1.3,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.5,
                preserve_body_parts=True
            ),
            PoseComplexity.COMPLEX: ExtractionRange(
                expansion_factor=1.5,
                min_aspect_ratio=0.4,
                max_aspect_ratio=3.0,
                preserve_body_parts=True
            ),
            PoseComplexity.EXTREME: ExtractionRange(
                expansion_factor=1.8,
                min_aspect_ratio=0.3,
                max_aspect_ratio=4.0,
                preserve_body_parts=True
            )
        }
    
    def analyze_pose_complexity(self, 
                              image: np.ndarray,
                              yolo_bbox: Tuple[int, int, int, int],
                              mask: Optional[np.ndarray] = None) -> PoseAnalysisResult:
        """
        姿勢複雑度の分析
        
        Args:
            image: 元画像
            yolo_bbox: YOLOバウンディングボックス (x, y, w, h)
            mask: SAMマスク（オプション）
            
        Returns:
            PoseAnalysisResult: 姿勢分析結果
        """
        x, y, w, h = yolo_bbox
        roi = image[y:y+h, x:x+w]
        
        # 1. アスペクト比による基本判定
        aspect_ratio = w / h if h > 0 else 1.0
        
        # 2. エッジ密度分析
        edge_density = self._analyze_edge_density(roi)
        
        # 3. 形状複雑度分析
        if mask is not None:
            roi_mask = mask[y:y+h, x:x+w]
            shape_complexity = self._analyze_shape_complexity(roi_mask)
        else:
            shape_complexity = 0.5  # デフォルト値
        
        # 4. 色彩分散分析
        color_variance = self._analyze_color_variance(roi)
        
        # 5. 体部位検出
        body_parts = self._detect_body_parts(roi)
        
        # 6. 遮蔽レベル推定
        occlusion_level = self._estimate_occlusion_level(roi, body_parts)
        
        # 7. 総合複雑度判定
        complexity = self._determine_complexity_level(
            aspect_ratio, edge_density, shape_complexity, 
            color_variance, len(body_parts), occlusion_level
        )
        
        # 8. 姿勢角度推定
        pose_angle = self._estimate_pose_angle(roi, mask)
        
        # 9. 信頼度計算
        confidence_score = self._calculate_analysis_confidence(
            edge_density, shape_complexity, len(body_parts)
        )
        
        result = PoseAnalysisResult(
            complexity=complexity,
            body_parts_detected=body_parts,
            pose_angle=pose_angle,
            occlusion_level=occlusion_level,
            confidence_score=confidence_score
        )
        
        self.logger.info(f"姿勢分析完了: {complexity.value}, 信頼度={confidence_score:.3f}")
        return result
    
    def _analyze_edge_density(self, roi: np.ndarray) -> float:
        """エッジ密度分析"""
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        return edge_density
    
    def _analyze_shape_complexity(self, mask: np.ndarray) -> float:
        """形状複雑度分析"""
        if np.sum(mask) == 0:
            return 0.0
        
        # 輪郭取得
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 最大輪郭の複雑度
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 周囲長/面積比（複雑な形状ほど大きい）
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        if area == 0:
            return 0.0
        
        complexity = perimeter / np.sqrt(area)
        normalized_complexity = min(1.0, complexity / 20.0)  # 正規化
        
        return normalized_complexity
    
    def _analyze_color_variance(self, roi: np.ndarray) -> float:
        """色彩分散分析"""
        # RGB各チャンネルの分散
        r_var = np.var(roi[:, :, 0])
        g_var = np.var(roi[:, :, 1])
        b_var = np.var(roi[:, :, 2])
        
        avg_variance = (r_var + g_var + b_var) / 3
        normalized_variance = min(1.0, avg_variance / 10000.0)
        
        return normalized_variance
    
    def _detect_body_parts(self, roi: np.ndarray) -> List[str]:
        """体部位検出（簡易版）"""
        body_parts = []
        h, w = roi.shape[:2]
        
        # 簡易的な体部位推定（実際にはより高度な手法を使用）
        
        # 頭部領域（上部1/4）
        head_region = roi[:h//4, :]
        if np.std(head_region) > 30:  # 複雑度による判定
            body_parts.append("head")
        
        # 胴体領域（中央1/2）
        torso_region = roi[h//4:3*h//4, :]
        if np.std(torso_region) > 25:
            body_parts.append("torso")
        
        # 足部領域（下部1/4）
        legs_region = roi[3*h//4:, :]
        if np.std(legs_region) > 20:
            body_parts.append("legs")
        
        # 幅による腕の推定
        if w / h > 1.2:  # 横長の場合、腕が広がっている可能性
            body_parts.append("arms")
        
        return body_parts
    
    def _estimate_occlusion_level(self, roi: np.ndarray, body_parts: List[str]) -> float:
        """遮蔽レベル推定"""
        # 基本的な体部位の期待数
        expected_parts = 3  # head, torso, legs
        detected_parts = len([p for p in body_parts if p in ["head", "torso", "legs"]])
        
        # 遮蔽レベル = 1 - (検出部位数 / 期待部位数)
        occlusion_level = max(0.0, 1.0 - detected_parts / expected_parts)
        
        return occlusion_level
    
    def _determine_complexity_level(self,
                                  aspect_ratio: float,
                                  edge_density: float,
                                  shape_complexity: float,
                                  color_variance: float,
                                  body_parts_count: int,
                                  occlusion_level: float) -> PoseComplexity:
        """総合複雑度レベル判定"""
        
        # 各指標のスコア化
        scores = []
        
        # アスペクト比スコア（1.0から離れるほど複雑）
        aspect_score = abs(aspect_ratio - 1.0)
        scores.append(min(1.0, aspect_score))
        
        # エッジ密度スコア
        scores.append(min(1.0, edge_density * 10))
        
        # 形状複雑度スコア
        scores.append(shape_complexity)
        
        # 色彩分散スコア
        scores.append(color_variance)
        
        # 体部位スコア（多いほど複雑）
        body_score = min(1.0, body_parts_count / 4.0)
        scores.append(body_score)
        
        # 遮蔽スコア
        scores.append(occlusion_level)
        
        # 総合スコア
        total_score = np.mean(scores)
        
        # 閾値による判定
        if total_score < 0.3:
            return PoseComplexity.SIMPLE
        elif total_score < 0.5:
            return PoseComplexity.DYNAMIC
        elif total_score < 0.7:
            return PoseComplexity.COMPLEX
        else:
            return PoseComplexity.EXTREME
    
    def _estimate_pose_angle(self, roi: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """姿勢角度推定"""
        if mask is None:
            return 0.0
        
        # マスクの重心と主軸を計算
        h, w = roi.shape[:2]
        roi_mask = mask[:h, :w] if mask.shape[0] >= h and mask.shape[1] >= w else np.zeros((h, w), dtype=bool)
        
        if np.sum(roi_mask) == 0:
            return 0.0
        
        # 重心計算
        moments = cv2.moments(roi_mask.astype(np.uint8))
        if moments['m00'] == 0:
            return 0.0
        
        # 主軸角度計算（PCA的手法）
        y_coords, x_coords = np.where(roi_mask)
        if len(x_coords) < 2:
            return 0.0
        
        # 共分散行列
        coords = np.column_stack([x_coords, y_coords])
        mean_coords = np.mean(coords, axis=0)
        centered_coords = coords - mean_coords
        
        cov_matrix = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 主軸の角度
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi
        
        return abs(angle)
    
    def _calculate_analysis_confidence(self,
                                     edge_density: float,
                                     shape_complexity: float,
                                     body_parts_count: int) -> float:
        """分析信頼度計算"""
        # 信頼度要因
        factors = []
        
        # エッジ密度（適度な値で信頼度が高い）
        edge_confidence = 1.0 - abs(edge_density - 0.1) * 5
        factors.append(max(0.0, min(1.0, edge_confidence)))
        
        # 形状複雑度（適度な値で信頼度が高い）
        shape_confidence = 1.0 - abs(shape_complexity - 0.3) * 2
        factors.append(max(0.0, min(1.0, shape_confidence)))
        
        # 体部位検出数（多いほど信頼度が高い）
        body_confidence = min(1.0, body_parts_count / 3.0)
        factors.append(body_confidence)
        
        return np.mean(factors)
    
    def adjust_extraction_range(self,
                              yolo_bbox: Tuple[int, int, int, int],
                              image_shape: Tuple[int, int],
                              pose_analysis: PoseAnalysisResult,
                              user_preferences: Optional[Dict[str, Any]] = None) -> Tuple[int, int, int, int]:
        """
        適応的抽出範囲調整
        
        Args:
            yolo_bbox: 元のYOLOバウンディングボックス (x, y, w, h)
            image_shape: 画像サイズ (height, width)
            pose_analysis: 姿勢分析結果
            user_preferences: ユーザー設定（オプション）
            
        Returns:
            Tuple[int, int, int, int]: 調整後のバウンディングボックス (x, y, w, h)
        """
        x, y, w, h = yolo_bbox
        img_h, img_w = image_shape
        
        # 複雑度に基づく基本設定取得
        range_config = self.range_presets[pose_analysis.complexity]
        
        # ユーザー設定による調整
        if user_preferences:
            range_config = self._apply_user_preferences(range_config, user_preferences)
        
        # 体部位に基づく特別調整
        range_config = self._adjust_for_body_parts(range_config, pose_analysis.body_parts_detected)
        
        # 拡張係数適用
        expansion = range_config.expansion_factor
        
        # 姿勢角度による追加調整
        if pose_analysis.pose_angle > 30:  # 傾いた姿勢
            expansion *= 1.1
        
        # 遮蔽レベルによる調整
        if pose_analysis.occlusion_level > 0.5:  # 高遮蔽
            expansion *= 1.2
        
        # 拡張適用
        center_x = x + w // 2
        center_y = y + h // 2
        
        new_w = int(w * expansion)
        new_h = int(h * expansion)
        
        # アスペクト比制限
        aspect_ratio = new_w / new_h if new_h > 0 else 1.0
        
        if aspect_ratio < range_config.min_aspect_ratio:
            new_w = int(new_h * range_config.min_aspect_ratio)
        elif aspect_ratio > range_config.max_aspect_ratio:
            new_h = int(new_w / range_config.max_aspect_ratio)
        
        # 新しい座標計算
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        
        # 画像境界制限
        new_x = min(new_x, img_w - new_w)
        new_y = min(new_y, img_h - new_h)
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        
        adjusted_bbox = (new_x, new_y, new_w, new_h)
        
        self.logger.info(f"範囲調整: {yolo_bbox} → {adjusted_bbox} (拡張率={expansion:.2f})")
        return adjusted_bbox
    
    def _apply_user_preferences(self,
                              range_config: ExtractionRange,
                              user_preferences: Dict[str, Any]) -> ExtractionRange:
        """ユーザー設定の適用"""
        # ユーザー設定例
        if user_preferences.get('include_full_body', False):
            range_config.expansion_factor *= 1.3
        
        if user_preferences.get('prefer_square', False):
            range_config.min_aspect_ratio = 0.8
            range_config.max_aspect_ratio = 1.25
        
        if user_preferences.get('conservative_crop', False):
            range_config.expansion_factor *= 0.8
        
        return range_config
    
    def _adjust_for_body_parts(self,
                             range_config: ExtractionRange,
                             body_parts: List[str]) -> ExtractionRange:
        """体部位に基づく調整"""
        # 足が検出されていない場合、下方向に拡張
        if 'legs' not in body_parts and 'torso' in body_parts:
            range_config.expansion_factor *= 1.15
            self.logger.debug("足部位不検出により下方向拡張")
        
        # 腕が検出されている場合、水平方向に拡張
        if 'arms' in body_parts:
            range_config.max_aspect_ratio *= 1.2
            self.logger.debug("腕部位検出により水平拡張")
        
        # 頭部が検出されていない場合、上方向に拡張
        if 'head' not in body_parts and 'torso' in body_parts:
            range_config.expansion_factor *= 1.1
            self.logger.debug("頭部不検出により上方向拡張")
        
        return range_config


def test_adaptive_extraction():
    """適応的抽出範囲調整のテスト"""
    # ダミーデータでテスト
    image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    yolo_bbox = (50, 50, 100, 150)
    
    adjuster = AdaptiveExtractionRangeAdjuster()
    
    # 姿勢分析
    pose_analysis = adjuster.analyze_pose_complexity(image, yolo_bbox)
    print(f"姿勢複雑度: {pose_analysis.complexity.value}")
    print(f"検出体部位: {pose_analysis.body_parts_detected}")
    print(f"信頼度: {pose_analysis.confidence_score:.3f}")
    
    # 範囲調整
    adjusted_bbox = adjuster.adjust_extraction_range(
        yolo_bbox, image.shape[:2], pose_analysis
    )
    print(f"調整前: {yolo_bbox}")
    print(f"調整後: {adjusted_bbox}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    test_adaptive_extraction()