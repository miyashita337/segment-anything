#!/usr/bin/env python3
"""
Region Priority System for Manga Character Extraction
Based on user evaluation feedback from kaname07 dataset
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

class RegionPrioritySystem:
    """
    漫画キャラクター抽出の領域優先度システム
    ユーザー評価データに基づいて最適な抽出領域を決定
    """
    
    def __init__(self):
        self.user_feedback = self._load_user_feedback()
        self.region_patterns = self._analyze_region_patterns()
        
    def _load_user_feedback(self) -> List[Dict]:
        """ユーザー評価データの読み込み"""
        feedback_path = Path(__file__).parent.parent / "logs" / "kaname07_user_evaluation.jsonl"
        feedback_data = []
        
        if feedback_path.exists():
            with open(feedback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line.strip()))
        
        return feedback_data
    
    def _analyze_region_patterns(self) -> Dict[str, float]:
        """ユーザー評価データから地域パターンを分析"""
        patterns = {}
        
        # 成功パターンの分析
        success_patterns = [
            item for item in self.user_feedback 
            if item.get('user_rating') == 'A' and item.get('actual_problem') == 'none'
        ]
        
        # 失敗パターンの分析
        failure_patterns = [
            item for item in self.user_feedback 
            if item.get('actual_problem') == 'wrong_character_selection'
        ]
        
        # 領域優先度の計算
        region_priorities = {
            'upper': 0.3,     # 画面上部
            'lower': 0.2,     # 画面下部
            'left': 0.25,     # 画面左側
            'right': 0.25,    # 画面右側
            'center': 0.4,    # 画面中央
            'full': 0.35      # 画面全体
        }
        
        # ユーザーフィードバックから優先度を調整
        for item in self.user_feedback:
            desired_region = item.get('desired_region', '')
            rating = item.get('user_rating')
            
            if rating == 'A' and 'success' in desired_region:
                # 成功パターンは中央優先度を上げる
                region_priorities['center'] += 0.1
            elif '上部' in desired_region or 'upper' in desired_region:
                region_priorities['upper'] += 0.05
            elif '下部' in desired_region or 'lower' in desired_region:
                region_priorities['lower'] += 0.05
            elif '左' in desired_region or 'left' in desired_region:
                region_priorities['left'] += 0.05
            elif '右' in desired_region or 'right' in desired_region:
                region_priorities['right'] += 0.05
                
        return region_priorities
    
    def get_region_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """画像を地域別にマスク分割"""
        h, w = image.shape[:2]
        masks = {}
        
        # 基本領域マスク
        masks['upper'] = np.zeros((h, w), dtype=np.uint8)
        masks['upper'][:h//2, :] = 255
        
        masks['lower'] = np.zeros((h, w), dtype=np.uint8)
        masks['lower'][h//2:, :] = 255
        
        masks['left'] = np.zeros((h, w), dtype=np.uint8)
        masks['left'][:, :w//2] = 255
        
        masks['right'] = np.zeros((h, w), dtype=np.uint8)
        masks['right'][:, w//2:] = 255
        
        masks['center'] = np.zeros((h, w), dtype=np.uint8)
        masks['center'][h//4:3*h//4, w//4:3*w//4] = 255
        
        masks['full'] = np.ones((h, w), dtype=np.uint8) * 255
        
        return masks
    
    def calculate_region_scores(self, candidates: List[Dict], image: np.ndarray) -> List[Dict]:
        """候補マスクの地域スコアを計算"""
        region_masks = self.get_region_masks(image)
        
        for candidate in candidates:
            mask = candidate.get('segmentation', np.zeros_like(image[:,:,0]))
            region_scores = {}
            
            # 各地域との重複度を計算
            for region, region_mask in region_masks.items():
                overlap = cv2.bitwise_and(mask, region_mask)
                overlap_ratio = np.sum(overlap > 0) / max(np.sum(mask > 0), 1)
                region_scores[region] = overlap_ratio * self.region_patterns.get(region, 0.2)
            
            # 最高スコアの地域を選択
            best_region = max(region_scores.items(), key=lambda x: x[1])
            candidate['region_score'] = best_region[1]
            candidate['preferred_region'] = best_region[0]
            
        return candidates
    
    def apply_size_priority_with_regions(self, candidates: List[Dict], image: np.ndarray) -> List[Dict]:
        """サイズ優先度と地域優先度を組み合わせた選択"""
        # 地域スコアを計算
        candidates = self.calculate_region_scores(candidates, image)
        
        # 総合スコア計算
        for candidate in candidates:
            base_score = candidate.get('score', 0.0)
            region_score = candidate.get('region_score', 0.0)
            area_score = candidate.get('area', 0) / (image.shape[0] * image.shape[1])
            
            # 総合スコア = ベーススコア × 0.4 + 地域スコア × 0.4 + 面積スコア × 0.2
            candidate['combined_score'] = (
                base_score * 0.4 + 
                region_score * 0.4 + 
                area_score * 0.2
            )
        
        # 総合スコアでソート
        candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return candidates
    
    def get_problem_analysis(self) -> Dict[str, int]:
        """問題パターンの分析結果を返す"""
        problem_counts = {}
        
        for item in self.user_feedback:
            problem = item.get('actual_problem', 'unknown')
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
        
        return problem_counts
    
    def get_success_rate(self) -> float:
        """成功率を計算"""
        total = len(self.user_feedback)
        successful = len([
            item for item in self.user_feedback 
            if item.get('user_rating') == 'A' and item.get('actual_problem') == 'none'
        ])
        
        return successful / total if total > 0 else 0.0

# 使用例
if __name__ == "__main__":
    rps = RegionPrioritySystem()
    print("Region priorities:", rps.region_patterns)
    print("Problem analysis:", rps.get_problem_analysis())
    print("Success rate:", rps.get_success_rate())