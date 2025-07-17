#!/usr/bin/env python3
"""
Partial Extraction Detector
部分抽出検出システム - 顔だけ/手足切断の自動検出

Phase 1 P1-002: 部分抽出検出システム実装
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

# Face detection support
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("dlib not available. Face detection will use OpenCV only.")


@dataclass
class PartialExtractionIssue:
    """部分抽出問題の詳細情報"""
    issue_type: str  # 'face_only', 'limb_truncated', 'torso_missing', 'incomplete_extraction'
    confidence: float  # 問題の信頼度 (0.0-1.0)
    description: str  # 問題の説明
    affected_regions: List[Tuple[int, int, int, int]]  # 影響を受ける領域 [x, y, w, h]
    severity: str  # 'low', 'medium', 'high'
    suggestions: List[str]  # 改善提案


@dataclass
class ExtractionAnalysis:
    """抽出結果の分析情報"""
    has_face: bool
    has_torso: bool  
    has_limbs: bool
    completeness_score: float  # 完全性スコア (0.0-1.0)
    issues: List[PartialExtractionIssue]
    quality_assessment: str  # 'good', 'partial', 'poor'


class PartialExtractionDetector:
    """部分抽出検出システム"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # OpenCV Face Cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception as e:
            self.logger.warning(f"OpenCV cascade loading failed: {e}")
            self.face_cascade = None
            self.profile_cascade = None
        
        # dlib face detector (if available)
        if DLIB_AVAILABLE:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
            except Exception as e:
                self.logger.warning(f"dlib detector loading failed: {e}")
                self.dlib_detector = None
        else:
            self.dlib_detector = None
    
    def analyze_extraction(self, image: np.ndarray, mask: np.ndarray) -> ExtractionAnalysis:
        """
        抽出結果の完全性分析
        
        Args:
            image: 元画像 (BGR)
            mask: 抽出マスク (0-255)
            
        Returns:
            ExtractionAnalysis: 分析結果
        """
        # 基本的な構造分析
        has_face = self._detect_face_presence(image, mask)
        has_torso = self._analyze_torso_presence(mask)
        has_limbs = self._analyze_limb_presence(mask)
        
        # 問題検出
        issues = []
        
        # 顔のみ抽出チェック
        if self._is_face_only_extraction(image, mask, has_face, has_torso, has_limbs):
            issues.append(PartialExtractionIssue(
                issue_type='face_only',
                confidence=0.8,
                description='顔領域のみが抽出されており、全身が含まれていません',
                affected_regions=self._get_face_regions(image),
                severity='high',
                suggestions=['全身が含まれる範囲での再抽出を推奨', 'YOLO閾値の調整']
            ))
        
        # 手足切断チェック
        limb_issues = self._detect_limb_truncation(mask)
        if limb_issues:
            issues.extend(limb_issues)
        
        # 胴体欠損チェック
        if has_face and has_limbs and not has_torso:
            issues.append(PartialExtractionIssue(
                issue_type='torso_missing',
                confidence=0.7,
                description='顔と手足は検出されましたが、胴体部分が欠損しています',
                affected_regions=[],
                severity='medium',
                suggestions=['マスク拡張処理の適用', 'SAMパラメータの調整']
            ))
        
        # 完全性スコア計算
        completeness_score = self._calculate_completeness_score(has_face, has_torso, has_limbs, issues)
        
        # 品質評価
        if completeness_score >= 0.8:
            quality_assessment = 'good'
        elif completeness_score >= 0.5:
            quality_assessment = 'partial'
        else:
            quality_assessment = 'poor'
        
        return ExtractionAnalysis(
            has_face=has_face,
            has_torso=has_torso,
            has_limbs=has_limbs,
            completeness_score=completeness_score,
            issues=issues,
            quality_assessment=quality_assessment
        )
    
    def _detect_face_presence(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """顔の存在検出"""
        try:
            # マスク領域のみを抽出
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            
            # OpenCV Cascade検出
            faces_detected = False
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    faces_detected = True
            
            # Profile face検出
            if not faces_detected and self.profile_cascade is not None:
                profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(profiles) > 0:
                    faces_detected = True
            
            # dlib検出（利用可能な場合）
            if not faces_detected and self.dlib_detector is not None:
                dlib_faces = self.dlib_detector(gray)
                if len(dlib_faces) > 0:
                    faces_detected = True
            
            return faces_detected
            
        except Exception as e:
            self.logger.warning(f"Face detection failed: {e}")
            return False
    
    def _analyze_torso_presence(self, mask: np.ndarray) -> bool:
        """胴体の存在分析"""
        h, w = mask.shape
        
        # 中央部分（縦30-70%、横20-80%）の密度を分析
        torso_region = mask[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
        if torso_region.size == 0:
            return False
            
        torso_density = np.sum(torso_region > 0) / torso_region.size
        
        # 胴体らしい密度（40%以上）があれば胴体存在と判定
        return torso_density > 0.4
    
    def _analyze_limb_presence(self, mask: np.ndarray) -> bool:
        """手足の存在分析"""
        h, w = mask.shape
        
        # 外周部分での密度分析
        # 左右端（手の可能性）
        left_edge = mask[:, :int(w*0.2)]
        right_edge = mask[:, int(w*0.8):]
        
        # 下端（足の可能性）
        bottom_edge = mask[int(h*0.7):, :]
        
        edge_densities = []
        for edge in [left_edge, right_edge, bottom_edge]:
            if edge.size > 0:
                density = np.sum(edge > 0) / edge.size
                edge_densities.append(density)
        
        # いずれかの端に十分な密度があれば手足存在と判定
        return any(density > 0.1 for density in edge_densities)
    
    def _is_face_only_extraction(self, image: np.ndarray, mask: np.ndarray, 
                                has_face: bool, has_torso: bool, has_limbs: bool) -> bool:
        """顔のみ抽出の判定"""
        if not has_face:
            return False
        
        # 顔はあるが胴体も手足もない場合は顔のみ抽出
        if not has_torso and not has_limbs:
            return True
        
        # マスクのアスペクト比チェック（正方形に近い場合は顔のみの可能性）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # アスペクト比が1.0-1.3の範囲で顔が検出された場合は顔のみ抽出の可能性
            if 1.0 <= aspect_ratio <= 1.3:
                return True
        
        return False
    
    def _detect_limb_truncation(self, mask: np.ndarray) -> List[PartialExtractionIssue]:
        """手足切断の検出"""
        issues = []
        h, w = mask.shape
        
        # エッジ検出による切断分析
        edges = cv2.Canny(mask, 50, 150)
        
        # 画像の境界での切断検出
        edge_truncation_score = 0
        
        # 上端での切断（頭部切断）
        top_edge_density = np.sum(edges[0:5, :] > 0) / (5 * w)
        if top_edge_density > 0.3:
            edge_truncation_score += 0.3
        
        # 左右端での切断（手の切断）
        left_edge_density = np.sum(edges[:, 0:5] > 0) / (h * 5)
        right_edge_density = np.sum(edges[:, -5:] > 0) / (h * 5)
        if left_edge_density > 0.2 or right_edge_density > 0.2:
            edge_truncation_score += 0.3
        
        # 下端での切断（足の切断）
        bottom_edge_density = np.sum(edges[-5:, :] > 0) / (5 * w)
        if bottom_edge_density > 0.3:
            edge_truncation_score += 0.4
        
        if edge_truncation_score > 0.5:
            issues.append(PartialExtractionIssue(
                issue_type='limb_truncated',
                confidence=min(edge_truncation_score, 1.0),
                description=f'手足の切断が検出されました（切断スコア: {edge_truncation_score:.2f}）',
                affected_regions=[],
                severity='high' if edge_truncation_score > 0.7 else 'medium',
                suggestions=['マスク拡張処理の適用', 'YOLO検出範囲の拡大', 'SAM points_per_sideの増加']
            ))
        
        return issues
    
    def _get_face_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """顔領域の取得"""
        regions = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                regions.extend([(x, y, w, h) for x, y, w, h in faces])
            
        except Exception as e:
            self.logger.warning(f"Face region detection failed: {e}")
        
        return regions
    
    def _calculate_completeness_score(self, has_face: bool, has_torso: bool, 
                                    has_limbs: bool, issues: List[PartialExtractionIssue]) -> float:
        """完全性スコアの計算"""
        base_score = 0.0
        
        # 基本構造スコア
        if has_face:
            base_score += 0.3
        if has_torso:
            base_score += 0.4
        if has_limbs:
            base_score += 0.3
        
        # 問題による減点
        for issue in issues:
            if issue.severity == 'high':
                base_score -= 0.3 * issue.confidence
            elif issue.severity == 'medium':
                base_score -= 0.2 * issue.confidence
            else:  # low
                base_score -= 0.1 * issue.confidence
        
        return max(0.0, min(1.0, base_score))
    
    def suggest_improvements(self, analysis: ExtractionAnalysis) -> List[str]:
        """改善提案の生成"""
        suggestions = []
        
        for issue in analysis.issues:
            suggestions.extend(issue.suggestions)
        
        # 一般的な改善提案
        if analysis.quality_assessment == 'poor':
            suggestions.extend([
                '手動ポイント指定による再抽出を検討',
                '前処理（コントラスト調整）の適用',
                '異なるYOLOモデルの使用'
            ])
        elif analysis.quality_assessment == 'partial':
            suggestions.extend([
                'マスク後処理の強化',
                'SAMパラメータの微調整'
            ])
        
        return list(set(suggestions))  # 重複除去


def analyze_extraction_completeness(image_path: str, mask: np.ndarray) -> ExtractionAnalysis:
    """
    抽出完全性の分析（便利関数）
    
    Args:
        image_path: 画像ファイルパス
        mask: 抽出マスク
        
    Returns:
        ExtractionAnalysis: 分析結果
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        detector = PartialExtractionDetector()
        return detector.analyze_extraction(image, mask)
        
    except Exception as e:
        logging.error(f"Extraction analysis failed: {e}")
        # エラー時のフォールバック
        return ExtractionAnalysis(
            has_face=False,
            has_torso=False,
            has_limbs=False,
            completeness_score=0.0,
            issues=[PartialExtractionIssue(
                issue_type='analysis_failed',
                confidence=1.0,
                description=f'分析エラー: {str(e)}',
                affected_regions=[],
                severity='high',
                suggestions=['画像ファイルの確認', 'システム再起動']
            )],
            quality_assessment='poor'
        )