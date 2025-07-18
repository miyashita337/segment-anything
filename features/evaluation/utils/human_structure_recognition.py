#!/usr/bin/env python3
"""
Human Structure Recognition System - P1-019
人体構造認識システム

キャラクターの人体構造を認識し、手足切断を防止
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass


@dataclass
class BodyRegion:
    """人体部位の定義"""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    area: float
    center: Tuple[float, float]


class HumanStructureRecognizer:
    """
    人体構造認識システム
    
    基本的な人体部位検出と手足切断防止
    """
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        
        # 人体比率の基準値 (アニメキャラクター向け調整)
        self.body_ratios = {
            'head_to_body': 1.0 / 6.5,  # 頭：全身 = 1:6.5 (アニメ調整)
            'head_width_ratio': 0.7,    # 頭の幅/高さ比率
            'torso_ratio': 0.4,         # 胴体：全身比率
            'leg_ratio': 0.45,          # 脚：全身比率
            'arm_ratio': 0.35,          # 腕：全身比率
            'shoulder_width': 2.2       # 肩幅：頭幅比率
        }
        
        # 切断リスク閾値
        self.truncation_thresholds = {
            'edge_distance': 5,         # エッジからの距離 (pixels)
            'boundary_ratio': 0.95,     # 境界近接比率
            'completeness_threshold': 0.7  # 完全性閾値
        }
    
    def analyze_mask_structure(self, mask: np.ndarray) -> Dict[str, Any]:
        """マスクの人体構造解析"""
        if mask is None or mask.size == 0:
            return {'error': 'マスクが無効です'}
        
        # バイナリマスクに変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 基本的な形状解析
        basic_analysis = self._analyze_basic_shape(binary_mask)
        
        # 人体部位の推定
        body_regions = self._estimate_body_regions(binary_mask, basic_analysis)
        
        # 切断リスクの評価
        truncation_risk = self._assess_truncation_risk(binary_mask, body_regions)
        
        # 人体構造の妥当性評価
        structure_validity = self._evaluate_structure_validity(body_regions, basic_analysis)
        
        return {
            'basic_analysis': basic_analysis,
            'body_regions': [self._region_to_dict(region) for region in body_regions],
            'truncation_risk': truncation_risk,
            'structure_validity': structure_validity,
            'overall_assessment': self._generate_overall_assessment(truncation_risk, structure_validity)
        }
    
    def _analyze_basic_shape(self, mask: np.ndarray) -> Dict[str, Any]:
        """基本的な形状解析"""
        # 輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': '輪郭が見つかりません'}
        
        # 最大輪郭を選択
        main_contour = max(contours, key=cv2.contourArea)
        
        # 基本的な幾何学的特徴
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # 境界ボックス
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # アスペクト比
        aspect_ratio = h / w if w > 0 else 0
        
        # 重心
        M = cv2.moments(main_contour)
        if M['m00'] != 0:
            centroid_x = M['m10'] / M['m00']
            centroid_y = M['m01'] / M['m00']
        else:
            centroid_x, centroid_y = x + w/2, y + h/2
        
        # 最小外接矩形
        rect = cv2.minAreaRect(main_contour)
        rect_area = rect[1][0] * rect[1][1]
        solidity = area / rect_area if rect_area > 0 else 0
        
        # 凸包
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'bounding_box': (int(x), int(y), int(w), int(h)),
            'aspect_ratio': float(aspect_ratio),
            'centroid': (float(centroid_x), float(centroid_y)),
            'solidity': float(solidity),
            'convexity': float(convexity),
            'mask_shape': mask.shape
        }
    
    def _estimate_body_regions(self, mask: np.ndarray, basic_analysis: Dict) -> List[BodyRegion]:
        """人体部位の推定"""
        bbox = basic_analysis['bounding_box']
        x, y, w, h = bbox
        
        regions = []
        
        # アスペクト比による大まかな分類
        aspect_ratio = basic_analysis['aspect_ratio']
        
        if aspect_ratio >= 1.5:  # 縦長 - 立ち姿勢の可能性
            regions.extend(self._estimate_standing_pose_regions(mask, bbox))
        elif 0.7 <= aspect_ratio < 1.5:  # 正方形に近い - 座り姿勢等
            regions.extend(self._estimate_compact_pose_regions(mask, bbox))
        else:  # 横長 - 横向き姿勢等
            regions.extend(self._estimate_horizontal_pose_regions(mask, bbox))
        
        return regions
    
    def _estimate_standing_pose_regions(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[BodyRegion]:
        """立ち姿勢での部位推定"""
        x, y, w, h = bbox
        regions = []
        
        # 縦方向の分割比率
        head_ratio = self.body_ratios['head_to_body']
        torso_ratio = self.body_ratios['torso_ratio']
        
        # 頭部推定 (上部)
        head_h = int(h * head_ratio)
        head_region = BodyRegion(
            name='head',
            bbox=(x, y, w, head_h),
            confidence=0.8,
            area=w * head_h,
            center=(x + w/2, y + head_h/2)
        )
        regions.append(head_region)
        
        # 胴体推定 (中央部)
        torso_y = y + head_h
        torso_h = int(h * torso_ratio)
        torso_region = BodyRegion(
            name='torso',
            bbox=(x, torso_y, w, torso_h),
            confidence=0.9,
            area=w * torso_h,
            center=(x + w/2, torso_y + torso_h/2)
        )
        regions.append(torso_region)
        
        # 脚部推定 (下部)
        legs_y = torso_y + torso_h
        legs_h = h - (head_h + torso_h)
        if legs_h > 0:
            legs_region = BodyRegion(
                name='legs',
                bbox=(x, legs_y, w, legs_h),
                confidence=0.7,
                area=w * legs_h,
                center=(x + w/2, legs_y + legs_h/2)
            )
            regions.append(legs_region)
        
        return regions
    
    def _estimate_compact_pose_regions(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[BodyRegion]:
        """コンパクトな姿勢での部位推定"""
        x, y, w, h = bbox
        regions = []
        
        # 頭部推定 (上部の30%)
        head_h = int(h * 0.3)
        head_region = BodyRegion(
            name='head',
            bbox=(x, y, w, head_h),
            confidence=0.7,
            area=w * head_h,
            center=(x + w/2, y + head_h/2)
        )
        regions.append(head_region)
        
        # 体部推定 (残り70%)
        body_y = y + head_h
        body_h = h - head_h
        body_region = BodyRegion(
            name='body',
            bbox=(x, body_y, w, body_h),
            confidence=0.8,
            area=w * body_h,
            center=(x + w/2, body_y + body_h/2)
        )
        regions.append(body_region)
        
        return regions
    
    def _estimate_horizontal_pose_regions(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[BodyRegion]:
        """横向き姿勢での部位推定"""
        x, y, w, h = bbox
        regions = []
        
        # 水平分割
        head_w = int(w * 0.25)  # 左端25%を頭部と仮定
        
        head_region = BodyRegion(
            name='head',
            bbox=(x, y, head_w, h),
            confidence=0.6,
            area=head_w * h,
            center=(x + head_w/2, y + h/2)
        )
        regions.append(head_region)
        
        body_x = x + head_w
        body_w = w - head_w
        body_region = BodyRegion(
            name='body',
            bbox=(body_x, y, body_w, h),
            confidence=0.7,
            area=body_w * h,
            center=(body_x + body_w/2, y + h/2)
        )
        regions.append(body_region)
        
        return regions
    
    def _assess_truncation_risk(self, mask: np.ndarray, regions: List[BodyRegion]) -> Dict[str, Any]:
        """切断リスクの評価"""
        h, w = mask.shape
        edge_distance = self.truncation_thresholds['edge_distance']
        
        truncation_risks = {}
        overall_risk = 0.0
        
        for region in regions:
            rx, ry, rw, rh = region.bbox
            risk_factors = []
            
            # エッジ近接チェック
            if rx <= edge_distance:  # 左端
                risk_factors.append('left_edge')
            if ry <= edge_distance:  # 上端
                risk_factors.append('top_edge')
            if rx + rw >= w - edge_distance:  # 右端
                risk_factors.append('right_edge')
            if ry + rh >= h - edge_distance:  # 下端
                risk_factors.append('bottom_edge')
            
            # 重要部位の切断リスク
            region_risk = len(risk_factors) / 4.0  # 0-1の範囲
            
            # 部位別重み付け
            if region.name == 'head':
                region_risk *= 1.5  # 頭部切断は重大
            elif region.name == 'legs':
                region_risk *= 1.2  # 脚部切断も重要
            
            truncation_risks[region.name] = {
                'risk_score': float(region_risk),
                'risk_factors': risk_factors,
                'severity': self._classify_risk_severity(region_risk)
            }
            
            overall_risk = max(overall_risk, region_risk)
        
        return {
            'overall_risk_score': float(overall_risk),
            'overall_severity': self._classify_risk_severity(overall_risk),
            'region_risks': truncation_risks,
            'has_truncation_risk': overall_risk > 0.3
        }
    
    def _classify_risk_severity(self, risk_score: float) -> str:
        """リスクの深刻度分類"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _evaluate_structure_validity(self, regions: List[BodyRegion], basic_analysis: Dict) -> Dict[str, Any]:
        """人体構造の妥当性評価"""
        if not regions:
            return {'validity_score': 0.0, 'issues': ['no_regions_detected']}
        
        issues = []
        validity_score = 1.0
        
        # アスペクト比チェック
        aspect_ratio = basic_analysis['aspect_ratio']
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:
            issues.append('unusual_aspect_ratio')
            validity_score -= 0.2
        
        # 部位の存在チェック
        region_names = [r.name for r in regions]
        if 'head' not in region_names:
            issues.append('no_head_detected')
            validity_score -= 0.3
        
        # 部位の配置チェック
        if len(regions) >= 2:
            head_regions = [r for r in regions if r.name == 'head']
            other_regions = [r for r in regions if r.name != 'head']
            
            if head_regions and other_regions:
                head_center_y = head_regions[0].center[1]
                for other in other_regions:
                    if other.center[1] <= head_center_y and other.name in ['torso', 'legs', 'body']:
                        issues.append('inverted_body_structure')
                        validity_score -= 0.2
                        break
        
        # 人体比率チェック
        total_area = sum(r.area for r in regions)
        if total_area > 0:
            head_regions = [r for r in regions if r.name == 'head']
            if head_regions:
                head_ratio = head_regions[0].area / total_area
                expected_ratio = self.body_ratios['head_to_body']
                
                if head_ratio > expected_ratio * 2 or head_ratio < expected_ratio * 0.3:
                    issues.append('unusual_head_proportion')
                    validity_score -= 0.1
        
        validity_score = max(0.0, validity_score)
        
        return {
            'validity_score': float(validity_score),
            'validity_grade': self._grade_validity(validity_score),
            'issues': issues,
            'region_count': len(regions)
        }
    
    def _grade_validity(self, score: float) -> str:
        """妥当性スコアをグレードに変換"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'acceptable'
        elif score >= 0.5:
            return 'poor'
        else:
            return 'invalid'
    
    def _generate_overall_assessment(self, truncation_risk: Dict, structure_validity: Dict) -> Dict[str, Any]:
        """総合評価の生成"""
        truncation_score = 1.0 - truncation_risk['overall_risk_score']
        validity_score = structure_validity['validity_score']
        
        # 重み付き平均 (切断リスク60%, 構造妥当性40%)
        overall_score = truncation_score * 0.6 + validity_score * 0.4
        
        # 推奨アクション
        recommendations = []
        
        if truncation_risk['has_truncation_risk']:
            recommendations.append('expand_extraction_area')
            recommendations.append('check_source_image_boundaries')
        
        if structure_validity['validity_score'] < 0.7:
            recommendations.append('verify_character_detection')
            recommendations.append('adjust_segmentation_parameters')
        
        return {
            'overall_score': float(overall_score),
            'overall_grade': self._grade_validity(overall_score),
            'primary_concerns': self._identify_primary_concerns(truncation_risk, structure_validity),
            'recommendations': recommendations,
            'extraction_quality': 'good' if overall_score >= 0.7 else 'needs_improvement'
        }
    
    def _identify_primary_concerns(self, truncation_risk: Dict, structure_validity: Dict) -> List[str]:
        """主要な懸念事項の特定"""
        concerns = []
        
        if truncation_risk['overall_severity'] in ['critical', 'high']:
            concerns.append('high_truncation_risk')
        
        if structure_validity['validity_score'] < 0.5:
            concerns.append('invalid_body_structure')
        
        if 'no_head_detected' in structure_validity.get('issues', []):
            concerns.append('missing_head_region')
        
        return concerns
    
    def _region_to_dict(self, region: BodyRegion) -> Dict[str, Any]:
        """BodyRegionを辞書に変換"""
        return {
            'name': region.name,
            'bbox': region.bbox,
            'confidence': region.confidence,
            'area': region.area,
            'center': region.center
        }
    
    def analyze_mask_file(self, mask_path: str) -> Dict[str, Any]:
        """マスクファイルの解析"""
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return {'error': f'マスクの読み込みに失敗: {mask_path}'}
            
            analysis_result = self.analyze_mask_structure(mask)
            analysis_result['mask_path'] = mask_path
            analysis_result['timestamp'] = datetime.now().isoformat()
            
            return analysis_result
            
        except Exception as e:
            return {'error': f'解析エラー: {str(e)}'}
    
    def print_analysis_summary(self, analysis_result: Dict[str, Any]):
        """解析結果のサマリー出力"""
        if 'error' in analysis_result:
            print(f"❌ {analysis_result['error']}")
            return
        
        print("\n" + "="*50)
        print("🤖 人体構造認識結果")
        print("="*50)
        
        # 基本情報
        basic = analysis_result.get('basic_analysis', {})
        print(f"📐 基本形状:")
        print(f"  アスペクト比: {basic.get('aspect_ratio', 0):.2f}")
        print(f"  面積: {basic.get('area', 0):.0f} pixels")
        print(f"  境界ボックス: {basic.get('bounding_box', 'N/A')}")
        
        # 検出された部位
        regions = analysis_result.get('body_regions', [])
        print(f"\n🎯 検出部位 ({len(regions)}個):")
        for region in regions:
            print(f"  {region['name']}: 信頼度 {region['confidence']:.2f}")
        
        # 切断リスク
        truncation = analysis_result.get('truncation_risk', {})
        print(f"\n⚠️ 切断リスク:")
        print(f"  総合リスク: {truncation.get('overall_severity', 'unknown')}")
        print(f"  リスクスコア: {truncation.get('overall_risk_score', 0):.3f}")
        
        # 構造妥当性
        validity = analysis_result.get('structure_validity', {})
        print(f"\n✅ 構造妥当性:")
        print(f"  妥当性グレード: {validity.get('validity_grade', 'unknown')}")
        print(f"  妥当性スコア: {validity.get('validity_score', 0):.3f}")
        
        # 総合評価
        overall = analysis_result.get('overall_assessment', {})
        print(f"\n🎯 総合評価:")
        print(f"  抽出品質: {overall.get('extraction_quality', 'unknown')}")
        print(f"  総合グレード: {overall.get('overall_grade', 'unknown')}")
        
        concerns = overall.get('primary_concerns', [])
        if concerns:
            print(f"  主要懸念: {', '.join(concerns)}")
        
        recommendations = overall.get('recommendations', [])
        if recommendations:
            print(f"  推奨アクション: {', '.join(recommendations)}")


def main():
    """メイン実行関数"""
    print("🚀 人体構造認識システム開始")
    
    # テスト用のダミーマスク作成 (人型)
    test_mask = np.zeros((200, 80), dtype=np.uint8)
    
    # 頭部 (円)
    cv2.circle(test_mask, (40, 25), 15, 255, -1)
    
    # 胴体 (矩形)
    cv2.rectangle(test_mask, (25, 40), (55, 120), 255, -1)
    
    # 脚部 (矩形)
    cv2.rectangle(test_mask, (30, 120), (50, 180), 255, -1)
    
    # 認識器初期化
    recognizer = HumanStructureRecognizer()
    
    # テスト実行
    print("📊 テストマスクで人体構造解析中...")
    analysis_result = recognizer.analyze_mask_structure(test_mask)
    
    # 結果出力
    recognizer.print_analysis_summary(analysis_result)
    
    print(f"\n✅ [P1-019] 人体構造認識システム完了")


if __name__ == "__main__":
    main()