#!/usr/bin/env python3
"""
Phase 4: マスク品質分析・逆転検出システム
SAMマスクの品質分析と逆転検出・自動修正機能
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class MaskQualityMetrics:
    """マスク品質メトリクス"""
    complexity_ratio: float          # 内外複雑度比
    edge_consistency: float          # エッジ一貫性
    color_coherence: float          # 色彩一貫性
    texture_variance: float         # テクスチャ分散
    confidence_score: float         # 総合信頼度スコア
    is_inverted: bool              # 逆転判定結果

class MaskQualityAnalyzer:
    """マスク品質分析・逆転検出クラス"""
    
    def __init__(self, 
                 complexity_threshold: float = 1.3,
                 edge_threshold: float = 0.6,
                 confidence_threshold: float = 0.7):
        """
        初期化
        
        Args:
            complexity_threshold: 複雑度比の逆転判定閾値
            edge_threshold: エッジ一貫性の閾値
            confidence_threshold: 信頼度の閾値
        """
        self.complexity_threshold = complexity_threshold
        self.edge_threshold = edge_threshold
        self.confidence_threshold = confidence_threshold
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
    
    def analyze_mask_quality(self, 
                           image: np.ndarray, 
                           mask: np.ndarray,
                           yolo_bbox: Optional[Tuple[int, int, int, int]] = None) -> MaskQualityMetrics:
        """
        マスク品質の総合分析
        
        Args:
            image: 元画像 (H, W, 3)
            mask: SAMマスク (H, W) bool
            yolo_bbox: YOLOバウンディングボックス (x, y, w, h)
            
        Returns:
            MaskQualityMetrics: 品質分析結果
        """
        # 1. 複雑度分析
        complexity_ratio = self._analyze_complexity_ratio(image, mask)
        
        # 2. エッジ一貫性分析
        edge_consistency = self._analyze_edge_consistency(image, mask, yolo_bbox)
        
        # 3. 色彩一貫性分析
        color_coherence = self._analyze_color_coherence(image, mask)
        
        # 4. テクスチャ分散分析
        texture_variance = self._analyze_texture_variance(image, mask)
        
        # 5. 逆転判定
        is_inverted = self._detect_mask_inversion(
            complexity_ratio, edge_consistency, color_coherence
        )
        
        # 6. 総合信頼度計算
        confidence_score = self._calculate_confidence_score(
            complexity_ratio, edge_consistency, color_coherence, texture_variance
        )
        
        return MaskQualityMetrics(
            complexity_ratio=complexity_ratio,
            edge_consistency=edge_consistency,
            color_coherence=color_coherence,
            texture_variance=texture_variance,
            confidence_score=confidence_score,
            is_inverted=is_inverted
        )
    
    def _analyze_complexity_ratio(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        マスク内外の複雑度比を分析
        キャラクターは背景より複雑な色彩・テクスチャを持つ傾向
        """
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # マスク内外の領域取得
        mask_bool = mask.astype(bool)
        inside_region = gray[mask_bool]
        outside_region = gray[~mask_bool]
        
        if len(inside_region) == 0 or len(outside_region) == 0:
            return 1.0
        
        # 複雑度計算（標準偏差ベース）
        inside_complexity = np.std(inside_region)
        outside_complexity = np.std(outside_region)
        
        # ゼロ除算対策
        if outside_complexity == 0:
            return float('inf') if inside_complexity > 0 else 1.0
            
        complexity_ratio = inside_complexity / outside_complexity
        
        self.logger.debug(f"複雑度比: {complexity_ratio:.3f} (内={inside_complexity:.1f}, 外={outside_complexity:.1f})")
        return complexity_ratio
    
    def _analyze_edge_consistency(self, 
                                image: np.ndarray, 
                                mask: np.ndarray,
                                yolo_bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """
        エッジ一貫性分析
        マスクエッジが実際の物体エッジと一致しているかを評価
        """
        # エッジ検出
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # マスクの輪郭取得
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 最大輪郭のマスクエッジ生成
        largest_contour = max(contours, key=cv2.contourArea)
        mask_edges = np.zeros_like(gray)
        cv2.drawContours(mask_edges, [largest_contour], -1, 255, 1)
        
        # YOLOバウンディングボックス内での一致度計算
        if yolo_bbox:
            x, y, w, h = yolo_bbox
            roi_edges = edges[y:y+h, x:x+w]
            roi_mask_edges = mask_edges[y:y+h, x:x+w]
        else:
            roi_edges = edges
            roi_mask_edges = mask_edges
        
        # エッジ一致度計算
        intersection = np.logical_and(roi_edges > 0, roi_mask_edges > 0)
        union = np.logical_or(roi_edges > 0, roi_mask_edges > 0)
        
        if np.sum(union) == 0:
            return 0.0
            
        consistency = np.sum(intersection) / np.sum(union)
        
        self.logger.debug(f"エッジ一貫性: {consistency:.3f}")
        return consistency
    
    def _analyze_color_coherence(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        色彩一貫性分析
        マスク内の色彩がキャラクターらしい一貫性を持つかを評価
        """
        mask_bool = mask.astype(bool)
        
        if np.sum(mask_bool) == 0:
            return 0.0
        
        # マスク内の色彩取得
        masked_pixels = image[mask_bool]
        
        # HSV変換による色相分析
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        masked_hsv = hsv_image[mask_bool]
        
        # 色相の分散（低いほど一貫性が高い）
        hue_std = np.std(masked_hsv[:, 0])
        
        # 彩度の平均（キャラクターは一般的に彩度が高い）
        saturation_mean = np.mean(masked_hsv[:, 1])
        
        # 正規化（0-1）
        hue_coherence = max(0, 1 - hue_std / 180.0)  # 色相の一貫性
        saturation_score = saturation_mean / 255.0    # 彩度スコア
        
        # 総合色彩一貫性
        color_coherence = (hue_coherence + saturation_score) / 2
        
        self.logger.debug(f"色彩一貫性: {color_coherence:.3f} (色相:{hue_coherence:.3f}, 彩度:{saturation_score:.3f})")
        return color_coherence
    
    def _analyze_texture_variance(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        テクスチャ分散分析
        マスク内のテクスチャの多様性を評価
        """
        mask_bool = mask.astype(bool)
        
        if np.sum(mask_bool) == 0:
            return 0.0
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # ラプラシアンフィルタによるテクスチャ検出
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # マスク内のテクスチャ分散
        masked_laplacian = laplacian[mask_bool]
        texture_variance = np.var(masked_laplacian)
        
        # 正規化（0-1スケール）
        normalized_variance = min(1.0, texture_variance / 10000.0)
        
        self.logger.debug(f"テクスチャ分散: {normalized_variance:.3f}")
        return normalized_variance
    
    def _detect_mask_inversion(self, 
                             complexity_ratio: float,
                             edge_consistency: float,
                             color_coherence: float) -> bool:
        """
        マスク逆転の判定
        
        Args:
            complexity_ratio: 複雑度比
            edge_consistency: エッジ一貫性
            color_coherence: 色彩一貫性
            
        Returns:
            bool: 逆転判定結果
        """
        # 複雑度による逆転判定（主要指標）
        complexity_inverted = complexity_ratio < (1.0 / self.complexity_threshold)
        
        # エッジ一貫性による補助判定
        poor_edge_consistency = edge_consistency < self.edge_threshold
        
        # 色彩一貫性による補助判定
        poor_color_coherence = color_coherence < 0.3
        
        # 総合判定
        # 複雑度が明らかに逆転 OR (複雑度微妙 AND 他指標も悪い)
        is_inverted = (
            complexity_inverted or 
            (complexity_ratio < 0.8 and poor_edge_consistency and poor_color_coherence)
        )
        
        self.logger.info(f"逆転判定: {is_inverted} (複雑度:{complexity_ratio:.3f}, エッジ:{edge_consistency:.3f}, 色彩:{color_coherence:.3f})")
        return is_inverted
    
    def _calculate_confidence_score(self, 
                                  complexity_ratio: float,
                                  edge_consistency: float,
                                  color_coherence: float,
                                  texture_variance: float) -> float:
        """
        総合信頼度スコア計算
        
        Returns:
            float: 信頼度スコア (0-1)
        """
        # 各指標の重み
        weights = {
            'complexity': 0.4,      # 複雑度比が最重要
            'edge': 0.3,           # エッジ一貫性
            'color': 0.2,          # 色彩一貫性
            'texture': 0.1         # テクスチャ分散
        }
        
        # 複雑度スコア（適切な範囲の場合に高スコア）
        ideal_complexity = 1.5  # 理想的な複雑度比
        complexity_score = 1.0 - abs(complexity_ratio - ideal_complexity) / ideal_complexity
        complexity_score = max(0, min(1, complexity_score))
        
        # 総合スコア計算
        confidence_score = (
            weights['complexity'] * complexity_score +
            weights['edge'] * edge_consistency +
            weights['color'] * color_coherence +
            weights['texture'] * min(1.0, texture_variance)
        )
        
        self.logger.debug(f"信頼度スコア: {confidence_score:.3f}")
        return confidence_score
    
    def fix_inverted_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        逆転マスクの修正
        
        Args:
            mask: 逆転しているマスク
            
        Returns:
            np.ndarray: 修正されたマスク
        """
        self.logger.info("マスク逆転を修正しています")
        return ~mask.astype(bool)
    
    def get_quality_report(self, metrics: MaskQualityMetrics) -> str:
        """
        品質分析レポートの生成
        
        Args:
            metrics: 品質メトリクス
            
        Returns:
            str: 分析レポート
        """
        report = []
        report.append("=== マスク品質分析レポート ===")
        report.append(f"複雑度比: {metrics.complexity_ratio:.3f}")
        report.append(f"エッジ一貫性: {metrics.edge_consistency:.3f}")
        report.append(f"色彩一貫性: {metrics.color_coherence:.3f}")
        report.append(f"テクスチャ分散: {metrics.texture_variance:.3f}")
        report.append(f"総合信頼度: {metrics.confidence_score:.3f}")
        report.append(f"逆転判定: {'はい' if metrics.is_inverted else 'いいえ'}")
        
        # 推奨アクション
        if metrics.is_inverted:
            report.append("⚠️ 推奨: マスク逆転修正が必要")
        elif metrics.confidence_score < self.confidence_threshold:
            report.append("⚠️ 推奨: パラメータ調整または手動確認")
        else:
            report.append("✅ 品質良好")
            
        return "\n".join(report)


def test_mask_quality_analyzer():
    """マスク品質分析器のテスト"""
    # ダミーデータでテスト
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.random.choice([0, 1], (100, 100), p=[0.7, 0.3]).astype(bool)
    
    analyzer = MaskQualityAnalyzer()
    metrics = analyzer.analyze_mask_quality(image, mask)
    
    print(analyzer.get_quality_report(metrics))
    
    if metrics.is_inverted:
        fixed_mask = analyzer.fix_inverted_mask(mask)
        print("マスク修正完了")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    test_mask_quality_analyzer()