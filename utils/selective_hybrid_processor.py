#!/usr/bin/env python3
"""
Phase 4.1: 選択的ハイブリッド処理器
複雑度分析に基づいて最適な処理エンジンを選択し、
Phase 0.0.3とPhase 0.0.4のいいとこどりを実現する
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time

from .image_complexity_analyzer import ComplexityLevel, ComplexityAnalysis, ImageComplexityAnalyzer

class ProcessingEngine(Enum):
    """処理エンジンタイプ"""
    PHASE_003 = "phase_003"    # Phase 0.0.3 基本処理
    PHASE_004 = "phase_004"    # Phase 0.0.4 統合処理
    HYBRID = "hybrid"          # ハイブリッド処理

@dataclass
class ProcessingResult:
    """処理結果"""
    success: bool
    mask: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    processing_time: float
    engine_used: ProcessingEngine
    quality_score: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class HybridProcessingResult:
    """ハイブリッド処理の最終結果"""
    final_result: ProcessingResult
    complexity_analysis: ComplexityAnalysis
    engine_results: Dict[ProcessingEngine, ProcessingResult]
    selection_reasoning: str
    total_processing_time: float

class ProcessingEngineBase:
    """処理エンジンの基底クラス"""
    
    def process(self, 
               image: np.ndarray,
               yolo_results: List[Dict[str, Any]]) -> ProcessingResult:
        """
        画像を処理
        
        Args:
            image: 入力画像
            yolo_results: YOLO検出結果
            
        Returns:
            ProcessingResult: 処理結果
        """
        raise NotImplementedError

class Phase003Engine(ProcessingEngineBase):
    """Phase 0.0.3処理エンジン"""
    
    def __init__(self, sam_model=None):
        self.sam_model = sam_model
        self.logger = logging.getLogger(__name__)
    
    def process(self, 
               image: np.ndarray,
               yolo_results: List[Dict[str, Any]]) -> ProcessingResult:
        """Phase 0.0.3方式で処理"""
        start_time = time.time()
        
        try:
            if not yolo_results:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    bbox=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    engine_used=ProcessingEngine.PHASE_003,
                    quality_score=0.0,
                    details={},
                    error_message="YOLO検出結果なし"
                )
            
            # 最高信頼度のバウンディングボックスを選択
            best_result = max(yolo_results, key=lambda x: x["confidence"])
            bbox = best_result["bbox"]
            confidence = best_result["confidence"]
            
            # SAMでマスク生成（簡略版）
            mask = self._generate_mask_simple(image, bbox)
            
            # 品質スコア計算
            quality_score = self._calculate_quality_score(mask, confidence)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                mask=mask,
                bbox=bbox,
                confidence=confidence,
                processing_time=processing_time,
                engine_used=ProcessingEngine.PHASE_003,
                quality_score=quality_score,
                details={
                    "yolo_detections": len(yolo_results),
                    "selected_confidence": confidence
                }
            )
            
        except Exception as e:
            self.logger.error(f"Phase 0.0.3処理エラー: {e}")
            return ProcessingResult(
                success=False,
                mask=None,
                bbox=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine_used=ProcessingEngine.PHASE_003,
                quality_score=0.0,
                details={},
                error_message=str(e)
            )
    
    def _generate_mask_simple(self, image: np.ndarray, 
                             bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """シンプルなマスク生成（Phase 0.0.3方式）"""
        # プレースホルダー：実際のSAM処理に置き換え
        x1, y1, x2, y2 = bbox
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255  # 単純な矩形マスク
        return mask
    
    def _calculate_quality_score(self, mask: np.ndarray, confidence: float) -> float:
        """品質スコア計算"""
        if mask is None:
            return 0.0
        
        # マスク面積比
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = mask_area / total_area
        
        # スコア計算（面積とYOLO信頼度の組み合わせ）
        area_score = min(area_ratio * 2, 1.0)  # 面積が50%で最大
        quality_score = (area_score * 0.6 + confidence * 0.4)
        
        return quality_score

class Phase004Engine(ProcessingEngineBase):
    """Phase 0.0.4統合処理エンジン"""
    
    def __init__(self, phase4_extractor=None):
        self.phase4_extractor = phase4_extractor
        self.logger = logging.getLogger(__name__)
    
    def process(self, 
               image: np.ndarray,
               yolo_results: List[Dict[str, Any]]) -> ProcessingResult:
        """Phase 0.0.4統合処理で処理"""
        start_time = time.time()
        
        try:
            if self.phase4_extractor is None:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    bbox=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    engine_used=ProcessingEngine.PHASE_004,
                    quality_score=0.0,
                    details={},
                    error_message="Phase 4 extractorが未初期化"
                )
            
            # Phase 4統合処理を実行
            result = self.phase4_extractor.extract_character(image, yolo_results)
            
            processing_time = time.time() - start_time
            
            if result.success:
                return ProcessingResult(
                    success=True,
                    mask=result.final_mask,
                    bbox=result.final_bbox,
                    confidence=result.quality_metrics.overall_score if result.quality_metrics else 0.0,
                    processing_time=processing_time,
                    engine_used=ProcessingEngine.PHASE_004,
                    quality_score=result.quality_metrics.overall_score if result.quality_metrics else 0.0,
                    details={
                        "adjustments_made": result.adjustments_made,
                        "iterations": len(result.processing_stats.get("iterations", []))
                    }
                )
            else:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    bbox=None,
                    confidence=0.0,
                    processing_time=processing_time,
                    engine_used=ProcessingEngine.PHASE_004,
                    quality_score=0.0,
                    details=result.processing_stats,
                    error_message="Phase 4処理失敗"
                )
                
        except Exception as e:
            self.logger.error(f"Phase 0.0.4処理エラー: {e}")
            return ProcessingResult(
                success=False,
                mask=None,
                bbox=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine_used=ProcessingEngine.PHASE_004,
                quality_score=0.0,
                details={},
                error_message=str(e)
            )

class SelectiveHybridProcessor:
    """選択的ハイブリッド処理器"""
    
    def __init__(self,
                 sam_model=None,
                 phase4_extractor=None,
                 enable_parallel_processing: bool = False):
        """
        初期化
        
        Args:
            sam_model: SAMモデル
            phase4_extractor: Phase 4統合抽出器
            enable_parallel_processing: 並行処理の有効化
        """
        self.complexity_analyzer = ImageComplexityAnalyzer()
        self.phase003_engine = Phase003Engine(sam_model)
        self.phase004_engine = Phase004Engine(phase4_extractor)
        self.enable_parallel = enable_parallel_processing
        self.logger = logging.getLogger(__name__)
    
    def process(self, 
               image: np.ndarray,
               yolo_results: List[Dict[str, Any]]) -> HybridProcessingResult:
        """
        選択的ハイブリッド処理を実行
        
        Args:
            image: 入力画像
            yolo_results: YOLO検出結果
            
        Returns:
            HybridProcessingResult: ハイブリッド処理結果
        """
        start_time = time.time()
        
        # 1. 複雑度分析
        complexity_analysis = self.complexity_analyzer.analyze_complexity(image, yolo_results)
        
        # 2. 処理戦略選択
        engine_results = {}
        
        if complexity_analysis.level == ComplexityLevel.SIMPLE:
            # シンプル：Phase 0.0.3のみ
            result = self.phase003_engine.process(image, yolo_results)
            engine_results[ProcessingEngine.PHASE_003] = result
            final_result = result
            selection_reasoning = "シンプルケース - Phase 0.0.3で十分"
            
        elif complexity_analysis.level == ComplexityLevel.COMPLEX:
            # 複雑：Phase 0.0.4のみ
            result = self.phase004_engine.process(image, yolo_results)
            engine_results[ProcessingEngine.PHASE_004] = result
            final_result = result
            selection_reasoning = "複雑ケース - Phase 0.0.4が必要"
            
        else:  # UNKNOWN
            # 不明：両方実行して選択
            if self.enable_parallel:
                results = self._process_parallel(image, yolo_results)
            else:
                results = self._process_sequential(image, yolo_results)
            
            engine_results = results
            final_result = self._select_best_result(results, complexity_analysis)
            selection_reasoning = f"判定困難ケース - {final_result.engine_used.value}を選択"
        
        total_time = time.time() - start_time
        
        return HybridProcessingResult(
            final_result=final_result,
            complexity_analysis=complexity_analysis,
            engine_results=engine_results,
            selection_reasoning=selection_reasoning,
            total_processing_time=total_time
        )
    
    def _process_parallel(self, 
                         image: np.ndarray,
                         yolo_results: List[Dict[str, Any]]) -> Dict[ProcessingEngine, ProcessingResult]:
        """並行処理（将来の実装用）"""
        # 現在は順次処理として実装
        return self._process_sequential(image, yolo_results)
    
    def _process_sequential(self, 
                           image: np.ndarray,
                           yolo_results: List[Dict[str, Any]]) -> Dict[ProcessingEngine, ProcessingResult]:
        """順次処理"""
        results = {}
        
        # Phase 0.0.3処理
        results[ProcessingEngine.PHASE_003] = self.phase003_engine.process(image, yolo_results)
        
        # Phase 0.0.4処理
        results[ProcessingEngine.PHASE_004] = self.phase004_engine.process(image, yolo_results)
        
        return results
    
    def _select_best_result(self, 
                           results: Dict[ProcessingEngine, ProcessingResult],
                           complexity_analysis: ComplexityAnalysis) -> ProcessingResult:
        """最適な結果を選択"""
        
        # 両方とも失敗の場合
        successful_results = {k: v for k, v in results.items() if v.success}
        if not successful_results:
            # 失敗したもので処理時間が短いものを返す
            return min(results.values(), key=lambda x: x.processing_time)
        
        # 成功した結果から選択
        if len(successful_results) == 1:
            return list(successful_results.values())[0]
        
        # 複数成功：品質スコアで選択
        best_result = max(successful_results.values(), key=lambda x: x.quality_score)
        
        # 品質スコアが拮抗している場合は複雑度分析を考慮
        sorted_results = sorted(successful_results.values(), key=lambda x: x.quality_score, reverse=True)
        if len(sorted_results) >= 2:
            score_diff = sorted_results[0].quality_score - sorted_results[1].quality_score
            
            # 5%以内の差なら処理時間で判定
            if score_diff < 0.05:
                best_result = min(sorted_results[:2], key=lambda x: x.processing_time)
        
        return best_result
    
    def get_processing_statistics(self, 
                                 result: HybridProcessingResult) -> Dict[str, Any]:
        """処理統計を取得"""
        stats = {
            "complexity_level": result.complexity_analysis.level.value,
            "yolo_detections": result.complexity_analysis.yolo_detections,
            "selected_engine": result.final_result.engine_used.value,
            "final_quality_score": result.final_result.quality_score,
            "total_processing_time": result.total_processing_time,
            "selection_reasoning": result.selection_reasoning
        }
        
        # 各エンジンの結果
        for engine, engine_result in result.engine_results.items():
            prefix = engine.value
            stats[f"{prefix}_success"] = engine_result.success
            stats[f"{prefix}_quality_score"] = engine_result.quality_score
            stats[f"{prefix}_processing_time"] = engine_result.processing_time
        
        return stats