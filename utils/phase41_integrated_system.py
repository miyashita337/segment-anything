#!/usr/bin/env python3
"""
Phase 4.1: 統合システム
選択的ハイブリッド処理、複数キャラクター処理、結果評価を統合した
Phase 0.0.3と0.0.4のいいとこどりシステム
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
import os

from .image_complexity_analyzer import ComplexityLevel, ComplexityAnalysis, ImageComplexityAnalyzer
from .selective_hybrid_processor import SelectiveHybridProcessor, HybridProcessingResult, ProcessingEngine
from .multi_character_handler import MultiCharacterHandler, SelectionCriteria, MultiCharacterAnalysis

@dataclass
class Phase41Result:
    """Phase 4.1統合処理結果"""
    success: bool
    final_mask: Optional[np.ndarray]
    final_bbox: Optional[Tuple[int, int, int, int]]
    quality_score: float
    processing_time: float
    
    # 詳細情報
    complexity_analysis: Optional[ComplexityAnalysis]
    multi_character_analysis: Optional[MultiCharacterAnalysis]
    hybrid_result: Optional[HybridProcessingResult]
    selected_engine: ProcessingEngine
    
    # 統計情報
    yolo_detections: int
    final_character_count: int
    adjustments_made: List[str]
    processing_stats: Dict[str, Any]
    
    # エラー情報
    error_message: Optional[str] = None
    warnings: List[str] = None

class Phase41IntegratedSystem:
    """Phase 4.1統合システム"""
    
    def __init__(self,
                 sam_model=None,
                 yolo_model=None,
                 phase4_extractor=None,
                 multi_character_criteria: SelectionCriteria = SelectionCriteria.BALANCED,
                 enable_parallel_processing: bool = False,
                 enable_detailed_logging: bool = True):
        """
        初期化
        
        Args:
            sam_model: SAMモデル
            yolo_model: YOLOモデル
            phase4_extractor: Phase 4統合抽出器
            multi_character_criteria: 複数キャラクター選択基準
            enable_parallel_processing: 並行処理の有効化
            enable_detailed_logging: 詳細ログの有効化
        """
        self.sam_model = sam_model
        self.yolo_model = yolo_model
        self.phase4_extractor = phase4_extractor
        
        # コンポーネント初期化
        self.complexity_analyzer = ImageComplexityAnalyzer()
        self.hybrid_processor = SelectiveHybridProcessor(
            sam_model=sam_model,
            phase4_extractor=phase4_extractor,
            enable_parallel_processing=enable_parallel_processing
        )
        self.multi_character_handler = MultiCharacterHandler(
            selection_criteria=multi_character_criteria
        )
        
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = logging.getLogger(__name__)
        
        # 統計カウンター
        self.processing_stats = {
            "total_processed": 0,
            "phase_003_selected": 0,
            "phase_004_selected": 0,
            "multi_character_cases": 0,
            "success_count": 0,
            "failure_count": 0
        }
    
    def extract_character(self, 
                         image: np.ndarray,
                         yolo_results: Optional[List[Dict[str, Any]]] = None) -> Phase41Result:
        """
        Phase 4.1統合キャラクター抽出
        
        Args:
            image: 入力画像
            yolo_results: YOLO検出結果（Noneの場合は自動検出）
            
        Returns:
            Phase41Result: 統合処理結果
        """
        start_time = time.time()
        warnings = []
        adjustments_made = []
        
        try:
            # 1. YOLO検出（必要に応じて）
            if yolo_results is None:
                if self.yolo_model is None:
                    return self._create_error_result(
                        start_time, "YOLOモデルまたは検出結果が必要です"
                    )
                yolo_results = self._detect_with_yolo(image)
            
            if self.enable_detailed_logging:
                self.logger.info(f"YOLO検出結果: {len(yolo_results)}個のキャラクター")
            
            # 2. 複数キャラクター処理
            multi_char_analysis = None
            final_yolo_results = yolo_results
            
            if len(yolo_results) > 1:
                self.processing_stats["multi_character_cases"] += 1
                multi_char_analysis = self.multi_character_handler.select_primary_character(
                    image, yolo_results
                )
                
                if multi_char_analysis.success and multi_char_analysis.selected_character:
                    # 選択されたキャラクターのみに絞る
                    selected_bbox = multi_char_analysis.selected_character.bbox
                    selected_confidence = multi_char_analysis.selected_character.confidence
                    
                    final_yolo_results = [{
                        "bbox": selected_bbox,
                        "confidence": selected_confidence
                    }]
                    
                    adjustments_made.append("複数キャラクター選択処理")
                    
                    if self.enable_detailed_logging:
                        summary = self.multi_character_handler.get_selection_summary(multi_char_analysis)
                        self.logger.info(f"複数キャラクター選択: {summary}")
                else:
                    warnings.append("複数キャラクター選択に失敗、全ての検出結果を使用")
            
            # 3. ハイブリッド処理
            hybrid_result = self.hybrid_processor.process(image, final_yolo_results)
            
            if not hybrid_result.final_result.success:
                return self._create_error_result(
                    start_time, 
                    f"ハイブリッド処理失敗: {hybrid_result.final_result.error_message}",
                    complexity_analysis=hybrid_result.complexity_analysis,
                    multi_character_analysis=multi_char_analysis,
                    hybrid_result=hybrid_result
                )
            
            # 4. 結果統計更新
            self._update_processing_stats(hybrid_result)
            
            # 5. 最終結果構築
            processing_time = time.time() - start_time
            
            result = Phase41Result(
                success=True,
                final_mask=hybrid_result.final_result.mask,
                final_bbox=hybrid_result.final_result.bbox,
                quality_score=hybrid_result.final_result.quality_score,
                processing_time=processing_time,
                
                complexity_analysis=hybrid_result.complexity_analysis,
                multi_character_analysis=multi_char_analysis,
                hybrid_result=hybrid_result,
                selected_engine=hybrid_result.final_result.engine_used,
                
                yolo_detections=len(yolo_results),
                final_character_count=len(final_yolo_results),
                adjustments_made=adjustments_made,
                processing_stats=self.hybrid_processor.get_processing_statistics(hybrid_result),
                warnings=warnings
            )
            
            self.processing_stats["success_count"] += 1
            self.processing_stats["total_processed"] += 1
            
            if self.enable_detailed_logging:
                self.logger.info(f"Phase 4.1処理完了: 品質スコア={result.quality_score:.3f}, "
                               f"エンジン={result.selected_engine.value}, "
                               f"処理時間={processing_time:.2f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 4.1処理エラー: {e}")
            return self._create_error_result(start_time, str(e))
    
    def extract_character_batch(self,
                               images: List[np.ndarray],
                               yolo_results_list: Optional[List[List[Dict[str, Any]]]] = None,
                               progress_callback=None) -> List[Phase41Result]:
        """
        バッチ処理
        
        Args:
            images: 入力画像リスト
            yolo_results_list: YOLO検出結果リスト
            progress_callback: 進捗コールバック関数
            
        Returns:
            List[Phase41Result]: 処理結果リスト
        """
        results = []
        total_images = len(images)
        
        for i, image in enumerate(images):
            yolo_results = yolo_results_list[i] if yolo_results_list else None
            
            result = self.extract_character(image, yolo_results)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total_images, result)
        
        return results
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO検出を実行"""
        # プレースホルダー：実際のYOLO検出に置き換え
        # ここでは既存のYOLOシステムを呼び出す
        if self.yolo_model is None:
            return []
        
        # 実装例（実際のYOLOモデルに応じて調整）
        try:
            results = self.yolo_model(image)
            yolo_results = []
            
            for result in results:
                # 検出結果をフォーマット
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        yolo_results.append({
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": float(confidence)
                        })
            
            return yolo_results
            
        except Exception as e:
            self.logger.error(f"YOLO検出エラー: {e}")
            return []
    
    def _update_processing_stats(self, hybrid_result: HybridProcessingResult):
        """処理統計を更新"""
        if hybrid_result.final_result.engine_used == ProcessingEngine.PHASE_003:
            self.processing_stats["phase_003_selected"] += 1
        elif hybrid_result.final_result.engine_used == ProcessingEngine.PHASE_004:
            self.processing_stats["phase_004_selected"] += 1
    
    def _create_error_result(self,
                            start_time: float,
                            error_message: str,
                            complexity_analysis: Optional[ComplexityAnalysis] = None,
                            multi_character_analysis: Optional[MultiCharacterAnalysis] = None,
                            hybrid_result: Optional[HybridProcessingResult] = None) -> Phase41Result:
        """エラー結果を作成"""
        processing_time = time.time() - start_time
        
        self.processing_stats["failure_count"] += 1
        self.processing_stats["total_processed"] += 1
        
        return Phase41Result(
            success=False,
            final_mask=None,
            final_bbox=None,
            quality_score=0.0,
            processing_time=processing_time,
            
            complexity_analysis=complexity_analysis,
            multi_character_analysis=multi_character_analysis,
            hybrid_result=hybrid_result,
            selected_engine=ProcessingEngine.PHASE_003,  # デフォルト
            
            yolo_detections=0,
            final_character_count=0,
            adjustments_made=[],
            processing_stats={},
            
            error_message=error_message,
            warnings=[]
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計を取得"""
        stats = self.processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["success_count"] / stats["total_processed"]
            stats["phase_003_usage_rate"] = stats["phase_003_selected"] / stats["total_processed"]
            stats["phase_004_usage_rate"] = stats["phase_004_selected"] / stats["total_processed"]
            stats["multi_character_rate"] = stats["multi_character_cases"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["phase_003_usage_rate"] = 0.0
            stats["phase_004_usage_rate"] = 0.0
            stats["multi_character_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """統計をリセット"""
        self.processing_stats = {
            "total_processed": 0,
            "phase_003_selected": 0,
            "phase_004_selected": 0,
            "multi_character_cases": 0,
            "success_count": 0,
            "failure_count": 0
        }
    
    def save_processing_report(self, 
                              results: List[Phase41Result],
                              output_path: str):
        """処理レポートを保存"""
        import json
        from datetime import datetime
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_version": "Phase 4.1",
            "total_images": len(results),
            "system_statistics": self.get_system_statistics(),
            "results_summary": self._generate_results_summary(results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"処理レポート保存: {output_path}")
    
    def _generate_results_summary(self, results: List[Phase41Result]) -> Dict[str, Any]:
        """結果サマリーを生成"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        summary = {
            "success_count": len(successful),
            "failure_count": len(failed),
            "success_rate": len(successful) / len(results) if results else 0.0
        }
        
        if successful:
            quality_scores = [r.quality_score for r in successful]
            processing_times = [r.processing_time for r in successful]
            
            summary.update({
                "average_quality_score": np.mean(quality_scores),
                "median_quality_score": np.median(quality_scores),
                "average_processing_time": np.mean(processing_times),
                "median_processing_time": np.median(processing_times)
            })
            
            # エンジン使用統計
            engine_usage = {}
            for result in successful:
                engine = result.selected_engine.value
                engine_usage[engine] = engine_usage.get(engine, 0) + 1
            
            summary["engine_usage"] = engine_usage
        
        return summary