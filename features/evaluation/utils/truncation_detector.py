#!/usr/bin/env python3
"""
P1-020: 切断検出アルゴリズム
人体キャラクター抽出における手足・身体部位の切断を検出・防止するシステム

Features:
- Edge-based truncation detection
- Body part completeness analysis
- Anatomical structure validation
- Truncation severity assessment
- Recovery suggestion generation
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime

# フォールバック実装用
HAS_SCIPY = True
HAS_SKLEARN = True

try:
    from scipy import ndimage, morphology
    from scipy.spatial.distance import cdist
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    HAS_SKLEARN = False


class TruncationDetector:
    """手足切断検出システム"""
    
    def __init__(self):
        """初期化"""
        self.name = "TruncationDetector"
        self.version = "1.0.0"
        
        # 切断検出パラメータ
        self.detection_params = {
            'edge_threshold': 0.05,      # エッジでの切断判定閾値
            'completeness_threshold': 0.7, # 部位完全性の最低基準
            'aspect_ratio_bounds': (0.3, 3.0), # 正常アスペクト比範囲
            'edge_proximity_threshold': 10,  # エッジ近接判定距離
            'limb_width_ratio': 0.15,    # 手足幅比率
            'torso_minimum_ratio': 0.4   # 胴体最小比率
        }
        
        # 部位別切断重要度
        self.truncation_severity = {
            'head': 0.9,      # 頭部切断は重大
            'torso': 0.8,     # 胴体切断も重要
            'upper_limb': 0.6, # 上肢（腕）
            'lower_limb': 0.7, # 下肢（脚）
            'hands': 0.4,     # 手部
            'feet': 0.5       # 足部
        }
        
        # グレーディング基準
        self.severity_grades = {
            'A': (0.0, 0.1),   # No truncation
            'B': (0.1, 0.3),   # Minor truncation
            'C': (0.3, 0.5),   # Moderate truncation
            'D': (0.5, 0.7),   # Significant truncation
            'F': (0.7, 1.0)    # Severe truncation
        }
    
    def detect_truncation(self, mask: np.ndarray, image_bounds: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        包括的切断検出分析
        
        Args:
            mask: バイナリマスク画像
            image_bounds: 画像境界情報 (height, width)
            
        Returns:
            Dict: 切断検出結果
        """
        if mask is None or mask.size == 0:
            return self._generate_error_result("Empty or invalid mask")
        
        try:
            # 画像境界情報設定
            if image_bounds is None:
                image_bounds = mask.shape
            
            # エッジベース切断検出
            edge_analysis = self._detect_edge_truncation(mask, image_bounds)
            
            # 身体部位完全性分析
            completeness_analysis = self._analyze_body_completeness(mask)
            
            # 解剖学的構造検証
            anatomical_analysis = self._validate_anatomical_structure(mask)
            
            # エッジ近接分析
            proximity_analysis = self._analyze_edge_proximity(mask, image_bounds)
            
            # 総合切断評価
            overall_assessment = self._calculate_overall_truncation_assessment(
                edge_analysis, completeness_analysis, anatomical_analysis, proximity_analysis
            )
            
            # 回復提案生成
            recovery_suggestions = self._generate_recovery_suggestions(
                edge_analysis, completeness_analysis, anatomical_analysis
            )
            
            return {
                'analysis_type': 'truncation_detection',
                'mask_info': {
                    'mask_shape': mask.shape,
                    'mask_area': int(np.sum(mask > 0)),
                    'image_bounds': image_bounds
                },
                'edge_truncation': edge_analysis,
                'body_completeness': completeness_analysis,
                'anatomical_validation': anatomical_analysis,
                'edge_proximity': proximity_analysis,
                'overall_assessment': overall_assessment,
                'recovery_suggestions': recovery_suggestions,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': self.version
                }
            }
            
        except Exception as e:
            return self._generate_error_result(f"Truncation detection failed: {str(e)}")
    
    def _detect_edge_truncation(self, mask: np.ndarray, image_bounds: Tuple) -> Dict[str, Any]:
        """エッジベース切断検出"""
        try:
            height, width = image_bounds[:2]
            edge_threshold = int(min(height, width) * self.detection_params['edge_threshold'])
            
            # マスクの境界検出
            if mask.dtype != np.uint8:
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
            else:
                mask_uint8 = mask
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'error': 'No contours found for edge analysis'}
            
            # 最大輪郭選択
            main_contour = max(contours, key=cv2.contourArea)
            
            # 各エッジでの切断検出
            edge_truncations = {
                'top': self._check_edge_truncation(main_contour, 'top', edge_threshold, width, height),
                'bottom': self._check_edge_truncation(main_contour, 'bottom', edge_threshold, width, height),
                'left': self._check_edge_truncation(main_contour, 'left', edge_threshold, width, height),
                'right': self._check_edge_truncation(main_contour, 'right', edge_threshold, width, height)
            }
            
            # 切断スコア計算
            truncation_scores = {edge: info['truncation_score'] for edge, info in edge_truncations.items()}
            overall_edge_score = max(truncation_scores.values()) if truncation_scores else 0.0
            
            return {
                'edge_truncations': edge_truncations,
                'truncation_scores': truncation_scores,
                'overall_edge_truncation_score': overall_edge_score,
                'edge_truncation_grade': self._score_to_grade(overall_edge_score)
            }
            
        except Exception as e:
            return {'error': f'Edge truncation detection failed: {str(e)}'}
    
    def _check_edge_truncation(self, contour: np.ndarray, edge: str, threshold: int, 
                              width: int, height: int) -> Dict[str, Any]:
        """特定エッジでの切断チェック"""
        points = contour.reshape(-1, 2)
        
        # エッジ近接点検出
        if edge == 'top':
            near_edge = points[points[:, 1] <= threshold]
            edge_length = width
        elif edge == 'bottom':
            near_edge = points[points[:, 1] >= height - threshold]
            edge_length = width
        elif edge == 'left':
            near_edge = points[points[:, 0] <= threshold]
            edge_length = height
        elif edge == 'right':
            near_edge = points[points[:, 0] >= width - threshold]
            edge_length = height
        else:
            return {'error': f'Unknown edge: {edge}'}
        
        if len(near_edge) == 0:
            return {
                'truncation_detected': False,
                'truncation_score': 0.0,
                'affected_length': 0,
                'severity': 'none'
            }
        
        # 切断長さ計算
        if edge in ['top', 'bottom']:
            affected_length = np.ptp(near_edge[:, 0]) if len(near_edge) > 1 else 0
        else:
            affected_length = np.ptp(near_edge[:, 1]) if len(near_edge) > 1 else 0
        
        # 切断スコア計算
        truncation_ratio = affected_length / edge_length if edge_length > 0 else 0
        truncation_score = min(truncation_ratio * 2, 1.0)  # 正規化
        
        # 重要度判定
        severity = self._determine_truncation_severity(edge, truncation_score)
        
        return {
            'truncation_detected': truncation_score > 0.1,
            'truncation_score': truncation_score,
            'affected_length': int(affected_length),
            'edge_length': edge_length,
            'truncation_ratio': truncation_ratio,
            'severity': severity,
            'near_edge_points': len(near_edge)
        }
    
    def _analyze_body_completeness(self, mask: np.ndarray) -> Dict[str, Any]:
        """身体部位完全性分析"""
        try:
            # マスクの基本情報
            mask_binary = mask > 0
            total_area = np.sum(mask_binary)
            
            if total_area == 0:
                return {'error': 'Empty mask for completeness analysis'}
            
            # バウンディングボックス
            coords = np.column_stack(np.where(mask_binary))
            if len(coords) == 0:
                return {'error': 'No valid coordinates found'}
            
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            aspect_ratio = height / width if width > 0 else 0
            
            # 身体部位推定（簡易版）
            body_regions = self._estimate_body_regions(mask_binary, min_y, max_y, min_x, max_x)
            
            # 各部位の完全性評価
            completeness_scores = {}
            for region_name, region_info in body_regions.items():
                completeness_scores[region_name] = self._evaluate_region_completeness(
                    mask_binary, region_info
                )
            
            # 全体完全性スコア
            if completeness_scores:
                overall_completeness = np.mean(list(completeness_scores.values()))
            else:
                overall_completeness = 0.0
            
            return {
                'mask_dimensions': {'height': height, 'width': width},
                'aspect_ratio': aspect_ratio,
                'body_regions': body_regions,
                'completeness_scores': completeness_scores,
                'overall_completeness': overall_completeness,
                'completeness_grade': self._score_to_grade(1.0 - overall_completeness)  # 低い方が良い
            }
            
        except Exception as e:
            return {'error': f'Body completeness analysis failed: {str(e)}'}
    
    def _estimate_body_regions(self, mask: np.ndarray, min_y: int, max_y: int, 
                              min_x: int, max_x: int) -> Dict[str, Dict]:
        """身体部位推定（簡易版）"""
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # 基本的な身体部位分割
        regions = {
            'head': {
                'y_range': (min_y, min_y + int(height * 0.25)),
                'x_range': (min_x, max_x),
                'expected_ratio': 0.15
            },
            'torso': {
                'y_range': (min_y + int(height * 0.25), min_y + int(height * 0.7)),
                'x_range': (min_x, max_x),
                'expected_ratio': 0.45
            },
            'lower_body': {
                'y_range': (min_y + int(height * 0.7), max_y),
                'x_range': (min_x, max_x),
                'expected_ratio': 0.4
            }
        }
        
        return regions
    
    def _evaluate_region_completeness(self, mask: np.ndarray, region_info: Dict) -> float:
        """部位完全性評価"""
        try:
            y_min, y_max = region_info['y_range']
            x_min, x_max = region_info['x_range']
            
            # 領域内のマスク面積
            region_mask = mask[y_min:y_max, x_min:x_max]
            region_area = np.sum(region_mask)
            
            # 期待される面積
            region_total = (y_max - y_min) * (x_max - x_min)
            expected_area = region_total * region_info.get('expected_ratio', 0.3)
            
            # 完全性スコア（期待面積に対する実際面積の比率）
            if expected_area > 0:
                completeness = min(region_area / expected_area, 1.0)
            else:
                completeness = 0.0
            
            return 1.0 - completeness  # 不完全性スコア（高いほど切断可能性大）
            
        except Exception:
            return 1.0  # エラー時は最大不完全性
    
    def _validate_anatomical_structure(self, mask: np.ndarray) -> Dict[str, Any]:
        """解剖学的構造検証"""
        try:
            mask_binary = mask > 0
            
            # 連結成分分析
            if HAS_SCIPY:
                labeled_mask, num_components = ndimage.label(mask_binary)
                component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            else:
                # フォールバック：OpenCVベース
                contours, _ = cv2.findContours(
                    mask_binary.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                num_components = len(contours)
                component_sizes = [cv2.contourArea(c) for c in contours] if contours else []
            
            # 構造分析
            structure_analysis = {
                'component_count': num_components,
                'component_sizes': component_sizes,
                'main_component_ratio': max(component_sizes) / sum(component_sizes) if component_sizes else 0,
                'fragmentation_score': self._calculate_fragmentation_score(component_sizes)
            }
            
            # 解剖学的妥当性評価
            anatomical_validity = self._assess_anatomical_validity(structure_analysis, mask_binary)
            
            return {
                'structure_analysis': structure_analysis,
                'anatomical_validity': anatomical_validity,
                'structure_grade': self._score_to_grade(1.0 - structure_analysis['fragmentation_score'])
            }
            
        except Exception as e:
            return {'error': f'Anatomical structure validation failed: {str(e)}'}
    
    def _calculate_fragmentation_score(self, component_sizes: List[int]) -> float:
        """断片化スコア計算"""
        if not component_sizes:
            return 1.0
        
        if len(component_sizes) == 1:
            return 0.0  # 単一成分なら断片化なし
        
        total_area = sum(component_sizes)
        main_ratio = max(component_sizes) / total_area if total_area > 0 else 0
        
        # 断片化スコア（主成分比率が低いほど断片化大）
        fragmentation = 1.0 - main_ratio
        
        # 複数成分のペナルティ
        component_penalty = min((len(component_sizes) - 1) * 0.2, 0.8)
        
        return min(fragmentation + component_penalty, 1.0)
    
    def _assess_anatomical_validity(self, structure_analysis: Dict, mask: np.ndarray) -> Dict[str, Any]:
        """解剖学的妥当性評価"""
        validity_score = 1.0
        issues = []
        
        # 過度な断片化チェック
        if structure_analysis['component_count'] > 3:
            validity_score -= 0.3
            issues.append('excessive_fragmentation')
        
        # 主成分比率チェック
        if structure_analysis['main_component_ratio'] < 0.7:
            validity_score -= 0.2
            issues.append('weak_main_component')
        
        # アスペクト比チェック
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            height = coords[:, 0].ptp() + 1
            width = coords[:, 1].ptp() + 1
            aspect_ratio = height / width if width > 0 else 0
            
            if not (self.detection_params['aspect_ratio_bounds'][0] <= 
                   aspect_ratio <= self.detection_params['aspect_ratio_bounds'][1]):
                validity_score -= 0.2
                issues.append('unusual_aspect_ratio')
        
        validity_score = max(validity_score, 0.0)
        
        return {
            'validity_score': validity_score,
            'validity_issues': issues,
            'is_anatomically_valid': validity_score > 0.6
        }
    
    def _analyze_edge_proximity(self, mask: np.ndarray, image_bounds: Tuple) -> Dict[str, Any]:
        """エッジ近接分析"""
        try:
            height, width = image_bounds[:2]
            threshold = self.detection_params['edge_proximity_threshold']
            
            mask_binary = mask > 0
            coords = np.column_stack(np.where(mask_binary))
            
            if len(coords) == 0:
                return {'error': 'No coordinates for proximity analysis'}
            
            # 各エッジへの近接度計算
            edge_proximities = {
                'top': np.sum(coords[:, 0] < threshold),
                'bottom': np.sum(coords[:, 0] > height - threshold),
                'left': np.sum(coords[:, 1] < threshold),
                'right': np.sum(coords[:, 1] > width - threshold)
            }
            
            total_points = len(coords)
            proximity_ratios = {edge: count / total_points for edge, count in edge_proximities.items()}
            
            # 最大近接度（最も問題のあるエッジ）
            max_proximity = max(proximity_ratios.values()) if proximity_ratios else 0
            
            return {
                'edge_proximities': edge_proximities,
                'proximity_ratios': proximity_ratios,
                'max_proximity_ratio': max_proximity,
                'proximity_grade': self._score_to_grade(1.0 - max_proximity)
            }
            
        except Exception as e:
            return {'error': f'Edge proximity analysis failed: {str(e)}'}
    
    def _calculate_overall_truncation_assessment(self, edge_analysis: Dict, completeness_analysis: Dict,
                                               anatomical_analysis: Dict, proximity_analysis: Dict) -> Dict[str, Any]:
        """総合切断評価計算"""
        scores = []
        weights = []
        
        # エッジ切断スコア
        if 'overall_edge_truncation_score' in edge_analysis:
            scores.append(edge_analysis['overall_edge_truncation_score'])
            weights.append(0.4)
        
        # 完全性スコア
        if 'overall_completeness' in completeness_analysis:
            scores.append(completeness_analysis['overall_completeness'])
            weights.append(0.3)
        
        # 解剖学的構造スコア
        if 'structure_analysis' in anatomical_analysis:
            fragmentation = anatomical_analysis['structure_analysis'].get('fragmentation_score', 0)
            scores.append(fragmentation)
            weights.append(0.2)
        
        # 近接スコア
        if 'max_proximity_ratio' in proximity_analysis:
            scores.append(proximity_analysis['max_proximity_ratio'])
            weights.append(0.1)
        
        # 重み付き平均
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        # 重要度分析
        severity_assessment = self._assess_truncation_severity(overall_score, edge_analysis, completeness_analysis)
        
        return {
            'overall_truncation_score': overall_score,
            'truncation_grade': self._score_to_grade(overall_score),
            'severity_assessment': severity_assessment,
            'component_scores': {
                'edge_truncation': edge_analysis.get('overall_edge_truncation_score', 0),
                'incompleteness': completeness_analysis.get('overall_completeness', 0),
                'fragmentation': anatomical_analysis.get('structure_analysis', {}).get('fragmentation_score', 0),
                'edge_proximity': proximity_analysis.get('max_proximity_ratio', 0)
            }
        }
    
    def _assess_truncation_severity(self, overall_score: float, edge_analysis: Dict, 
                                  completeness_analysis: Dict) -> str:
        """切断重要度評価"""
        if overall_score <= 0.1:
            return "no_truncation"
        elif overall_score <= 0.3:
            return "minor_truncation"
        elif overall_score <= 0.5:
            return "moderate_truncation"
        elif overall_score <= 0.7:
            return "significant_truncation"
        else:
            return "severe_truncation"
    
    def _generate_recovery_suggestions(self, edge_analysis: Dict, completeness_analysis: Dict,
                                     anatomical_analysis: Dict) -> List[str]:
        """回復提案生成"""
        suggestions = []
        
        # エッジ切断への対応
        if edge_analysis.get('overall_edge_truncation_score', 0) > 0.3:
            suggestions.append("expand_extraction_area")
            suggestions.append("adjust_bounding_box")
        
        # 不完全性への対応
        if completeness_analysis.get('overall_completeness', 0) > 0.4:
            suggestions.append("improve_segmentation_parameters")
            suggestions.append("use_larger_input_resolution")
        
        # 断片化への対応
        fragmentation = anatomical_analysis.get('structure_analysis', {}).get('fragmentation_score', 0)
        if fragmentation > 0.3:
            suggestions.append("merge_disconnected_components")
            suggestions.append("apply_morphological_closing")
        
        return suggestions
    
    def _determine_truncation_severity(self, edge: str, score: float) -> str:
        """エッジ別切断重要度判定"""
        if score <= 0.1:
            return "none"
        elif score <= 0.3:
            return "minor"
        elif score <= 0.5:
            return "moderate"
        else:
            return "severe"
    
    def _score_to_grade(self, score: float) -> str:
        """スコアからグレードへの変換"""
        for grade, (min_val, max_val) in self.severity_grades.items():
            if min_val <= score < max_val:
                return grade
        return 'F'
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        return {
            'error': error_message,
            'overall_assessment': {
                'overall_truncation_score': 1.0,
                'truncation_grade': 'F',
                'severity_assessment': 'analysis_failed'
            }
        }


def main():
    """テスト実行"""
    print("🚀 P1-020: 切断検出アルゴリズム テスト開始")
    
    # テスト用人体形状マスク作成
    test_mask = np.zeros((200, 150), dtype=np.uint8)
    
    # 頭部
    cv2.circle(test_mask, (75, 40), 25, 255, -1)
    # 胴体
    cv2.rectangle(test_mask, (50, 65), (100, 120), 255, -1)
    # 脚部（一部を意図的に切断）
    cv2.rectangle(test_mask, (55, 120), (95, 190), 255, -1)  # 画像下端近くで切断
    
    # 切断検出実行
    detector = TruncationDetector()
    result = detector.detect_truncation(test_mask, (200, 150))
    
    print("\n📊 切断検出結果:")
    if 'error' not in result:
        overall = result.get('overall_assessment', {})
        print(f"  総合切断スコア: {overall.get('overall_truncation_score', 0):.3f}")
        print(f"  切断グレード: {overall.get('truncation_grade', 'N/A')}")
        print(f"  重要度評価: {overall.get('severity_assessment', 'N/A')}")
        
        # エッジ切断詳細
        edge_analysis = result.get('edge_truncation', {})
        if 'truncation_scores' in edge_analysis:
            print(f"\n🔍 エッジ切断スコア:")
            for edge, score in edge_analysis['truncation_scores'].items():
                print(f"    {edge}: {score:.3f}")
        
        # 回復提案
        suggestions = result.get('recovery_suggestions', [])
        if suggestions:
            print(f"\n💡 回復提案: {', '.join(suggestions)}")
    else:
        print(f"  ❌ エラー: {result['error']}")
    
    print(f"\n✅ [P1-020] 切断検出アルゴリズム完了")


if __name__ == "__main__":
    main()