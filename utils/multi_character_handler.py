#!/usr/bin/env python3
"""
Phase 4.1: 複数キャラクター処理専用ハンドラー
複数人キャラクターが検出された場合に、最適なキャラクターを選択し、
「一番大きいキャラクター」「全身が入るキャラクター」を優先的に抽出する
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

class SelectionCriteria(Enum):
    """選択基準"""
    SIZE_PRIORITY = "size_priority"           # サイズ優先
    FULLBODY_PRIORITY = "fullbody_priority"   # 全身優先
    CENTRAL_PRIORITY = "central_priority"     # 中央配置優先
    CONFIDENCE_PRIORITY = "confidence_priority" # 信頼度優先
    BALANCED = "balanced"                     # バランス型

@dataclass
class CharacterCandidate:
    """キャラクター候補"""
    bbox: Tuple[int, int, int, int]  # バウンディングボックス
    confidence: float                # YOLO信頼度
    area_ratio: float               # 画像に対する面積比
    aspect_ratio: float             # アスペクト比（height/width）
    central_score: float            # 中央配置スコア（0-1）
    fullbody_score: float           # 全身度スコア（0-1）
    position_score: float           # 位置スコア（0-1）
    total_score: float              # 総合スコア
    selection_reasons: List[str]    # 選択理由

@dataclass
class MultiCharacterAnalysis:
    """複数キャラクター分析結果"""
    total_characters: int
    selected_character: Optional[CharacterCandidate]
    all_candidates: List[CharacterCandidate]
    selection_criteria_used: SelectionCriteria
    analysis_details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class MultiCharacterHandler:
    """複数キャラクター処理ハンドラー"""
    
    def __init__(self,
                 selection_criteria: SelectionCriteria = SelectionCriteria.BALANCED,
                 min_area_threshold: float = 0.01,    # 最小面積閾値（1%）
                 max_area_threshold: float = 0.8,     # 最大面積閾値（80%）
                 fullbody_aspect_min: float = 1.2,    # 全身とみなす最小アスペクト比
                 fullbody_aspect_max: float = 4.0):   # 全身とみなす最大アスペクト比
        """
        初期化
        
        Args:
            selection_criteria: 選択基準
            min_area_threshold: 最小面積閾値
            max_area_threshold: 最大面積閾値
            fullbody_aspect_min: 全身最小アスペクト比
            fullbody_aspect_max: 全身最大アスペクト比
        """
        self.selection_criteria = selection_criteria
        self.min_area_threshold = min_area_threshold
        self.max_area_threshold = max_area_threshold
        self.fullbody_aspect_min = fullbody_aspect_min
        self.fullbody_aspect_max = fullbody_aspect_max
        self.logger = logging.getLogger(__name__)
    
    def select_primary_character(self,
                                image: np.ndarray,
                                yolo_results: List[Dict[str, Any]]) -> MultiCharacterAnalysis:
        """
        複数キャラクターから主要キャラクターを選択
        
        Args:
            image: 入力画像
            yolo_results: YOLO検出結果リスト
            
        Returns:
            MultiCharacterAnalysis: 分析結果
        """
        try:
            if not yolo_results:
                return MultiCharacterAnalysis(
                    total_characters=0,
                    selected_character=None,
                    all_candidates=[],
                    selection_criteria_used=self.selection_criteria,
                    analysis_details={},
                    success=False,
                    error_message="YOLO検出結果なし"
                )
            
            image_height, image_width = image.shape[:2]
            
            # 1. 候補キャラクターを分析
            candidates = []
            for result in yolo_results:
                candidate = self._analyze_character_candidate(
                    result, image_width, image_height
                )
                candidates.append(candidate)
            
            # 2. 面積フィルタリング
            filtered_candidates = self._filter_by_area(candidates)
            
            if not filtered_candidates:
                return MultiCharacterAnalysis(
                    total_characters=len(candidates),
                    selected_character=None,
                    all_candidates=candidates,
                    selection_criteria_used=self.selection_criteria,
                    analysis_details={"filter_reason": "面積フィルタリングで全て除外"},
                    success=False,
                    error_message="有効なキャラクター候補なし"
                )
            
            # 3. 選択基準に基づいてスコア計算
            scored_candidates = self._calculate_selection_scores(
                filtered_candidates, image_width, image_height
            )
            
            # 4. 最適キャラクターを選択
            selected = self._select_best_candidate(scored_candidates)
            
            analysis_details = {
                "original_detections": len(yolo_results),
                "filtered_candidates": len(filtered_candidates),
                "selection_method": self.selection_criteria.value,
                "score_breakdown": self._get_score_breakdown(scored_candidates)
            }
            
            return MultiCharacterAnalysis(
                total_characters=len(candidates),
                selected_character=selected,
                all_candidates=candidates,
                selection_criteria_used=self.selection_criteria,
                analysis_details=analysis_details,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"複数キャラクター選択エラー: {e}")
            return MultiCharacterAnalysis(
                total_characters=len(yolo_results) if yolo_results else 0,
                selected_character=None,
                all_candidates=[],
                selection_criteria_used=self.selection_criteria,
                analysis_details={},
                success=False,
                error_message=str(e)
            )
    
    def _analyze_character_candidate(self,
                                   yolo_result: Dict[str, Any],
                                   image_width: int,
                                   image_height: int) -> CharacterCandidate:
        """キャラクター候補を分析"""
        bbox = yolo_result["bbox"]
        confidence = yolo_result["confidence"]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 基本計算
        area = width * height
        area_ratio = area / (image_width * image_height)
        aspect_ratio = height / width if width > 0 else 0
        
        # 中央配置スコア
        central_score = self._calculate_central_score(bbox, image_width, image_height)
        
        # 全身度スコア
        fullbody_score = self._calculate_fullbody_score(bbox, aspect_ratio, image_height)
        
        # 位置スコア
        position_score = self._calculate_position_score(bbox, image_width, image_height)
        
        return CharacterCandidate(
            bbox=bbox,
            confidence=confidence,
            area_ratio=area_ratio,
            aspect_ratio=aspect_ratio,
            central_score=central_score,
            fullbody_score=fullbody_score,
            position_score=position_score,
            total_score=0.0,  # 後で計算
            selection_reasons=[]
        )
    
    def _calculate_central_score(self,
                                bbox: Tuple[int, int, int, int],
                                image_width: int,
                                image_height: int) -> float:
        """中央配置スコア計算"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        # 正規化された距離
        distance_x = abs(center_x - image_center_x) / image_width
        distance_y = abs(center_y - image_center_y) / image_height
        
        # 距離を逆転してスコア化
        distance = np.sqrt(distance_x**2 + distance_y**2)
        score = max(0.0, 1.0 - distance * 2)  # 2倍して急激に減衰
        
        return score
    
    def _calculate_fullbody_score(self,
                                 bbox: Tuple[int, int, int, int],
                                 aspect_ratio: float,
                                 image_height: int) -> float:
        """全身度スコア計算"""
        x1, y1, x2, y2 = bbox
        
        # アスペクト比による判定
        aspect_score = 0.0
        if self.fullbody_aspect_min <= aspect_ratio <= self.fullbody_aspect_max:
            # 理想的なアスペクト比（2.0）に近いほど高スコア
            ideal_ratio = 2.0
            aspect_deviation = abs(aspect_ratio - ideal_ratio) / ideal_ratio
            aspect_score = max(0.0, 1.0 - aspect_deviation)
        
        # 画面下部への接触度（地面に立っている可能性）
        bottom_distance = abs(y2 - image_height) / image_height
        bottom_score = max(0.0, 1.0 - bottom_distance * 3)  # 下部接触を重視
        
        # 全身度スコア（アスペクト比60% + 底部接触40%）
        fullbody_score = aspect_score * 0.6 + bottom_score * 0.4
        
        return fullbody_score
    
    def _calculate_position_score(self,
                                 bbox: Tuple[int, int, int, int],
                                 image_width: int,
                                 image_height: int) -> float:
        """位置スコア計算（漫画レイアウトを考慮）"""
        x1, y1, x2, y2 = bbox
        
        # 画像内での相対位置
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 縦方向の位置スコア（中央やや下を好む）
        vertical_position = center_y / image_height
        if 0.3 <= vertical_position <= 0.7:
            vertical_score = 1.0
        elif 0.1 <= vertical_position <= 0.9:
            vertical_score = 0.7
        else:
            vertical_score = 0.3
        
        # 横方向の位置スコア（中央を好む）
        horizontal_position = center_x / image_width
        if 0.3 <= horizontal_position <= 0.7:
            horizontal_score = 1.0
        elif 0.1 <= horizontal_position <= 0.9:
            horizontal_score = 0.8
        else:
            horizontal_score = 0.5
        
        position_score = (vertical_score + horizontal_score) / 2
        return position_score
    
    def _filter_by_area(self, candidates: List[CharacterCandidate]) -> List[CharacterCandidate]:
        """面積による候補フィルタリング"""
        filtered = []
        for candidate in candidates:
            if (self.min_area_threshold <= candidate.area_ratio <= self.max_area_threshold):
                filtered.append(candidate)
            else:
                self.logger.debug(f"面積フィルタで除外: area_ratio={candidate.area_ratio:.3f}")
        
        return filtered
    
    def _calculate_selection_scores(self,
                                   candidates: List[CharacterCandidate],
                                   image_width: int,
                                   image_height: int) -> List[CharacterCandidate]:
        """選択基準に基づいてスコア計算"""
        for candidate in candidates:
            if self.selection_criteria == SelectionCriteria.SIZE_PRIORITY:
                # サイズ重視
                candidate.total_score = (
                    candidate.area_ratio * 0.5 +
                    candidate.confidence * 0.3 +
                    candidate.central_score * 0.2
                )
                candidate.selection_reasons = ["サイズ優先選択"]
                
            elif self.selection_criteria == SelectionCriteria.FULLBODY_PRIORITY:
                # 全身優先
                candidate.total_score = (
                    candidate.fullbody_score * 0.5 +
                    candidate.area_ratio * 0.2 +
                    candidate.confidence * 0.2 +
                    candidate.position_score * 0.1
                )
                candidate.selection_reasons = ["全身優先選択"]
                
            elif self.selection_criteria == SelectionCriteria.CENTRAL_PRIORITY:
                # 中央配置優先
                candidate.total_score = (
                    candidate.central_score * 0.4 +
                    candidate.area_ratio * 0.3 +
                    candidate.confidence * 0.3
                )
                candidate.selection_reasons = ["中央配置優先選択"]
                
            elif self.selection_criteria == SelectionCriteria.CONFIDENCE_PRIORITY:
                # 信頼度優先
                candidate.total_score = (
                    candidate.confidence * 0.5 +
                    candidate.area_ratio * 0.3 +
                    candidate.central_score * 0.2
                )
                candidate.selection_reasons = ["信頼度優先選択"]
                
            else:  # BALANCED
                # バランス型（デフォルト）
                candidate.total_score = (
                    candidate.area_ratio * 0.25 +          # サイズ
                    candidate.fullbody_score * 0.25 +      # 全身度
                    candidate.confidence * 0.2 +           # 信頼度
                    candidate.central_score * 0.15 +       # 中央配置
                    candidate.position_score * 0.15        # 位置
                )
                
                # 選択理由を詳細化
                reasons = []
                if candidate.area_ratio > 0.15:
                    reasons.append("大きなサイズ")
                if candidate.fullbody_score > 0.6:
                    reasons.append("全身キャラクター")
                if candidate.confidence > 0.7:
                    reasons.append("高信頼度")
                if candidate.central_score > 0.6:
                    reasons.append("中央配置")
                
                candidate.selection_reasons = reasons if reasons else ["バランス型選択"]
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[CharacterCandidate]) -> Optional[CharacterCandidate]:
        """最適候補を選択"""
        if not candidates:
            return None
        
        # 総合スコアで選択
        best_candidate = max(candidates, key=lambda x: x.total_score)
        
        self.logger.info(f"選択されたキャラクター: スコア={best_candidate.total_score:.3f}, "
                        f"理由={', '.join(best_candidate.selection_reasons)}")
        
        return best_candidate
    
    def _get_score_breakdown(self, candidates: List[CharacterCandidate]) -> Dict[str, Any]:
        """スコア内訳を取得"""
        breakdown = {}
        for i, candidate in enumerate(candidates):
            breakdown[f"candidate_{i}"] = {
                "total_score": candidate.total_score,
                "area_ratio": candidate.area_ratio,
                "confidence": candidate.confidence,
                "fullbody_score": candidate.fullbody_score,
                "central_score": candidate.central_score,
                "position_score": candidate.position_score,
                "reasons": candidate.selection_reasons
            }
        
        return breakdown
    
    def get_selection_summary(self, analysis: MultiCharacterAnalysis) -> str:
        """選択結果のサマリーを取得"""
        if not analysis.success or not analysis.selected_character:
            return f"選択失敗: {analysis.error_message}"
        
        selected = analysis.selected_character
        summary = (
            f"選択結果: {analysis.total_characters}人中から最適キャラクターを選択\n"
            f"スコア: {selected.total_score:.3f}\n"
            f"面積比: {selected.area_ratio:.3f}\n"
            f"信頼度: {selected.confidence:.3f}\n"
            f"全身度: {selected.fullbody_score:.3f}\n"
            f"理由: {', '.join(selected.selection_reasons)}"
        )
        
        return summary