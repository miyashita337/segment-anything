#!/usr/bin/env python3
"""
Boundary Analysis Algorithm - P1-017
境界線解析アルゴリズム

マスクの境界線品質を定量評価し、滑らかさを改善
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json


class BoundaryAnalyzer:
    """
    境界線解析システム
    
    セグメンテーションマスクの境界線品質を定量評価
    """
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        
    def load_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """マスクの読み込み"""
        try:
            if isinstance(mask_path, str):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = mask_path
            
            if mask is None:
                return None
            
            # バイナリマスクに変換
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return binary_mask
            
        except Exception as e:
            print(f"❌ マスク読み込みエラー: {e}")
            return None
    
    def extract_boundary(self, mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """境界線の抽出"""
        # エッジ検出
        edges = cv2.Canny(mask, 50, 150)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        return edges, contours
    
    def calculate_smoothness_metrics(self, contour: np.ndarray) -> Dict[str, float]:
        """滑らかさメトリクスの計算"""
        if len(contour) < 10:
            return {
                'curvature_variance': 0.0,
                'angle_variance': 0.0,
                'perimeter_roughness': 0.0,
                'douglas_peucker_ratio': 0.0
            }
        
        # 曲率解析
        curvatures = self._calculate_curvature(contour)
        curvature_variance = float(np.var(curvatures)) if len(curvatures) > 0 else 0.0
        
        # 角度変化解析
        angles = self._calculate_angle_changes(contour)
        angle_variance = float(np.var(angles)) if len(angles) > 0 else 0.0
        
        # 周囲長roughness
        perimeter_roughness = self._calculate_perimeter_roughness(contour)
        
        # Douglas-Peucker簡略化比率
        douglas_ratio = self._calculate_douglas_peucker_ratio(contour)
        
        return {
            'curvature_variance': curvature_variance,
            'angle_variance': angle_variance,
            'perimeter_roughness': perimeter_roughness,
            'douglas_peucker_ratio': douglas_ratio
        }
    
    def _calculate_curvature(self, contour: np.ndarray, window_size: int = 5) -> List[float]:
        """曲率の計算"""
        contour = contour.reshape(-1, 2)
        curvatures = []
        
        if len(contour) < window_size * 2:
            return curvatures
        
        for i in range(window_size, len(contour) - window_size):
            # 前後の点から接線ベクトルを計算
            p1 = contour[i - window_size]
            p2 = contour[i]
            p3 = contour[i + window_size]
            
            # ベクトル
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 長さ
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                # 角度変化から曲率を近似
                cos_theta = np.dot(v1, v2) / (len1 * len2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                curvature = np.arccos(cos_theta)
                curvatures.append(curvature)
        
        return curvatures
    
    def _calculate_angle_changes(self, contour: np.ndarray) -> List[float]:
        """角度変化の計算"""
        contour = contour.reshape(-1, 2)
        angles = []
        
        if len(contour) < 3:
            return angles
        
        for i in range(1, len(contour) - 1):
            p1 = contour[i - 1]
            p2 = contour[i]
            p3 = contour[i + 1]
            
            # ベクトル
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 角度計算
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            
            # 角度差
            angle_diff = angle2 - angle1
            
            # -π to π の範囲に正規化
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            angles.append(abs(angle_diff))
        
        return angles
    
    def _calculate_perimeter_roughness(self, contour: np.ndarray) -> float:
        """周囲長roughnessの計算"""
        contour = contour.reshape(-1, 2)
        
        if len(contour) < 3:
            return 0.0
        
        # 実際の周囲長
        actual_perimeter = cv2.arcLength(contour, True)
        
        # 凸包の周囲長
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        
        if hull_perimeter > 0:
            roughness = actual_perimeter / hull_perimeter
        else:
            roughness = 1.0
        
        return float(roughness)
    
    def _calculate_douglas_peucker_ratio(self, contour: np.ndarray, epsilon_ratio: float = 0.01) -> float:
        """Douglas-Peucker簡略化比率の計算"""
        contour = contour.reshape(-1, 2)
        
        if len(contour) < 3:
            return 1.0
        
        # 周囲長の一定比率をepsilonとして使用
        perimeter = cv2.arcLength(contour, True)
        epsilon = perimeter * epsilon_ratio
        
        # Douglas-Peucker簡略化
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        # 簡略化比率
        if len(contour) > 0:
            ratio = len(simplified) / len(contour)
        else:
            ratio = 1.0
        
        return float(ratio)
    
    def calculate_boundary_quality_score(self, mask: np.ndarray) -> Dict[str, Any]:
        """境界線品質スコアの総合計算"""
        edges, contours = self.extract_boundary(mask)
        
        if not contours:
            return {
                'overall_score': 0.0,
                'contour_count': 0,
                'metrics': {},
                'largest_contour_area': 0,
                'boundary_pixel_count': 0
            }
        
        # 最大輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # 滑らかさメトリクス計算
        smoothness_metrics = self.calculate_smoothness_metrics(largest_contour)
        
        # 境界ピクセル数
        boundary_pixels = np.sum(edges > 0)
        
        # 総合スコア計算 (0-1, 1が最高品質)
        overall_score = self._calculate_overall_score(smoothness_metrics, largest_area, boundary_pixels)
        
        return {
            'overall_score': overall_score,
            'contour_count': len(contours),
            'metrics': smoothness_metrics,
            'largest_contour_area': float(largest_area),
            'boundary_pixel_count': int(boundary_pixels),
            'quality_grade': self._grade_quality(overall_score)
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, float], area: float, boundary_pixels: int) -> float:
        """総合品質スコアの計算"""
        # 各メトリクスを0-1スケールに正規化して重み付け平均
        
        # 曲率分散 (低いほど良い)
        curvature_score = max(0, 1.0 - min(1.0, metrics['curvature_variance'] / 5.0))
        
        # 角度分散 (低いほど良い)
        angle_score = max(0, 1.0 - min(1.0, metrics['angle_variance'] / 2.0))
        
        # 周囲長roughness (1.0に近いほど良い, 1.0-1.5が正常範囲)
        roughness = metrics['perimeter_roughness']
        if roughness <= 1.0:
            roughness_score = roughness
        elif roughness <= 1.5:
            roughness_score = 1.0 - (roughness - 1.0) / 0.5 * 0.5
        else:
            roughness_score = max(0, 0.5 - (roughness - 1.5) / 2.0 * 0.5)
        
        # Douglas-Peucker比率 (高すぎず低すぎず、0.1-0.8が理想)
        dp_ratio = metrics['douglas_peucker_ratio']
        if 0.1 <= dp_ratio <= 0.8:
            dp_score = 1.0
        elif dp_ratio < 0.1:
            dp_score = dp_ratio / 0.1
        else:
            dp_score = max(0, 1.0 - (dp_ratio - 0.8) / 0.2)
        
        # 重み付け平均
        weights = [0.3, 0.3, 0.25, 0.15]  # curvature, angle, roughness, dp_ratio
        scores = [curvature_score, angle_score, roughness_score, dp_score]
        
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    def _grade_quality(self, score: float) -> str:
        """品質スコアをグレードに変換"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        elif score >= 0.5:
            return 'E'
        else:
            return 'F'
    
    def analyze_mask_file(self, mask_path: str) -> Dict[str, Any]:
        """マスクファイルの分析"""
        mask = self.load_mask(mask_path)
        if mask is None:
            return {'error': f'マスクの読み込みに失敗: {mask_path}'}
        
        quality_result = self.calculate_boundary_quality_score(mask)
        
        result = {
            'mask_path': mask_path,
            'mask_shape': mask.shape,
            'mask_area': int(np.sum(mask > 0)),
            'boundary_analysis': quality_result,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def analyze_directory(self, mask_dir: str, output_path: str = None) -> Dict[str, Any]:
        """ディレクトリ内の全マスクを分析"""
        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            return {'error': f'ディレクトリが存在しません: {mask_dir}'}
        
        # マスクファイルを検索
        mask_files = []
        for ext in ['*.png', '*.jpg', '*.bmp', '*.tiff']:
            mask_files.extend(mask_dir.glob(ext))
        
        if not mask_files:
            return {'error': f'マスクファイルが見つかりません: {mask_dir}'}
        
        print(f"📊 {len(mask_files)}個のマスクファイルを分析中...")
        
        results = []
        scores = []
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        
        for i, mask_file in enumerate(mask_files, 1):
            print(f"  処理中 ({i}/{len(mask_files)}): {mask_file.name}")
            
            result = self.analyze_mask_file(str(mask_file))
            if 'error' not in result:
                results.append(result)
                
                boundary_analysis = result['boundary_analysis']
                score = boundary_analysis['overall_score']
                grade = boundary_analysis['quality_grade']
                
                scores.append(score)
                grades[grade] += 1
        
        # 統計計算
        if scores:
            statistics = {
                'total_files': len(mask_files),
                'successful_analyses': len(scores),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'grade_distribution': grades,
                'quality_rate': (grades['A'] + grades['B']) / len(scores) if scores else 0.0
            }
        else:
            statistics = {'error': '分析可能なマスクがありませんでした'}
        
        analysis_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_directory': str(mask_dir),
            'statistics': statistics,
            'individual_results': results
        }
        
        # レポート保存
        if output_path is None:
            output_path = f"boundary_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 境界線分析完了: {output_path}")
        return analysis_report
    
    def print_analysis_summary(self, analysis_report: Dict[str, Any]):
        """分析結果のサマリー出力"""
        stats = analysis_report.get('statistics', {})
        
        print("\n" + "="*50)
        print("📊 境界線品質分析結果")
        print("="*50)
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"📁 分析対象: {analysis_report['input_directory']}")
        print(f"📊 処理件数: {stats['successful_analyses']}/{stats['total_files']}件")
        
        print(f"\n📈 品質統計:")
        print(f"  平均スコア: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
        print(f"  スコア範囲: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        print(f"  高品質率: {stats['quality_rate']:.1%} (A+B評価)")
        
        print(f"\n🎯 品質分布:")
        grades = stats['grade_distribution']
        total = sum(grades.values())
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = grades[grade]
            ratio = count / total * 100 if total > 0 else 0
            print(f"  {grade}評価: {count}件 ({ratio:.1f}%)")


def main():
    """メイン実行関数"""
    print("🚀 境界線解析アルゴリズム開始")
    
    # テスト用のダミーマスク作成
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_mask, (50, 50), 30, 255, -1)
    
    # 分析器初期化
    analyzer = BoundaryAnalyzer()
    
    # テスト実行
    print("📊 テストマスクで境界線分析中...")
    quality_result = analyzer.calculate_boundary_quality_score(test_mask)
    
    print(f"✅ 境界線品質分析結果:")
    print(f"  総合スコア: {quality_result['overall_score']:.3f}")
    print(f"  品質グレード: {quality_result['quality_grade']}")
    print(f"  輪郭数: {quality_result['contour_count']}")
    print(f"  境界ピクセル数: {quality_result['boundary_pixel_count']}")
    
    metrics = quality_result['metrics']
    print(f"  滑らかさメトリクス:")
    print(f"    曲率分散: {metrics['curvature_variance']:.3f}")
    print(f"    角度分散: {metrics['angle_variance']:.3f}")
    print(f"    周囲長roughness: {metrics['perimeter_roughness']:.3f}")
    print(f"    Douglas-Peucker比率: {metrics['douglas_peucker_ratio']:.3f}")
    
    print(f"\n✅ [P1-017] 境界線解析アルゴリズム完了")


if __name__ == "__main__":
    main()