#!/usr/bin/env python3
"""
小さなキャラクター対応面積閾値テスト

適応的面積閾値調整機能の動作確認と、
小さなキャラクターでの品質チェック改善効果を検証
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.processing.postprocessing.postprocessing import (
    calculate_mask_quality_metrics, 
    remove_small_components_adaptive
)
from tools.unified_quality_checker import UnifiedQualityChecker


def create_small_character_test_masks() -> List[Dict]:
    """小さなキャラクター用テストマスクを生成"""
    test_masks = []
    
    # テストケース1: 超小キャラクター（画像の0.3%）
    mask1 = np.zeros((512, 512), dtype=np.uint8)
    tiny_char_size = int(512 * 0.055)  # 約0.3%の面積
    center_x, center_y = 256, 256
    cv2.circle(mask1, (center_x, center_y), tiny_char_size, 255, -1)
    test_masks.append({
        'name': 'tiny_character',
        'mask': mask1,
        'expected_ratio': 0.003,
        'description': '超小キャラクター（0.3%）'
    })
    
    # テストケース2: 小さなキャラクター（画像の2%）
    mask2 = np.zeros((512, 512), dtype=np.uint8)
    medium_char_size = int(512 * 0.141)  # 約2%の面積
    cv2.circle(mask2, (center_x, center_y), medium_char_size, 255, -1)
    test_masks.append({
        'name': 'small_character',
        'mask': mask2,
        'expected_ratio': 0.02,
        'description': '小さなキャラクター（2%）'
    })
    
    # テストケース3: 通常サイズキャラクター（画像の8%）
    mask3 = np.zeros((512, 512), dtype=np.uint8)
    normal_char_size = int(512 * 0.283)  # 約8%の面積
    cv2.circle(mask3, (center_x, center_y), normal_char_size, 255, -1)
    test_masks.append({
        'name': 'normal_character',
        'mask': mask3,
        'expected_ratio': 0.08,
        'description': '通常サイズキャラクター（8%）'
    })
    
    return test_masks


def test_adaptive_area_filtering():
    """適応的面積フィルタリングのテスト"""
    print("🧪 適応的面積フィルタリングテスト")
    print("="*60)
    
    test_masks = create_small_character_test_masks()
    
    for test_case in test_masks:
        print(f"\n📋 テストケース: {test_case['description']}")
        
        mask = test_case['mask']
        
        # 従来の固定閾値フィルタリング
        traditional_filtered = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        
        # 適応的面積フィルタリング
        adaptive_filtered = remove_small_components_adaptive(
            mask, 
            min_area_ratio=0.001,  # 0.1%
            absolute_min_area=50
        )
        
        # 結果分析
        original_area = np.sum(mask > 0)
        traditional_area = np.sum(traditional_filtered > 0)
        adaptive_area = np.sum(adaptive_filtered > 0)
        
        print(f"  元マスク面積: {original_area:,} ピクセル")
        print(f"  従来フィルタ後: {traditional_area:,} ピクセル "
              f"(保持率: {traditional_area/original_area:.1%})")
        print(f"  適応フィルタ後: {adaptive_area:,} ピクセル "
              f"(保持率: {adaptive_area/original_area:.1%})")
        
        # 品質メトリクス計算
        original_metrics = calculate_mask_quality_metrics(mask)
        adaptive_metrics = calculate_mask_quality_metrics(adaptive_filtered)
        
        print(f"  元カバレッジ率: {original_metrics['coverage_ratio']:.4f}")
        print(f"  適応後カバレッジ率: {adaptive_metrics['coverage_ratio']:.4f}")


def test_adaptive_threshold_calculation():
    """適応的閾値計算のテスト"""
    print("\n🎯 適応的閾値計算テスト")
    print("="*60)
    
    # UnifiedQualityCheckerのインスタンス作成
    checker = UnifiedQualityChecker()
    
    test_masks = create_small_character_test_masks()
    
    for test_case in test_masks:
        print(f"\n📊 {test_case['description']}")
        
        # マスク品質メトリクス計算
        quality_metrics = calculate_mask_quality_metrics(test_case['mask'])
        
        # 適応的閾値計算
        adaptive_threshold = checker._calculate_adaptive_coverage_threshold([quality_metrics])
        default_threshold = checker.thresholds["coverage_ratio"]
        
        print(f"  実際カバレッジ率: {quality_metrics['coverage_ratio']:.4f}")
        print(f"  デフォルト閾値: {default_threshold:.3f}")
        print(f"  適応的閾値: {adaptive_threshold:.3f}")
        
        # 判定結果
        default_pass = quality_metrics['coverage_ratio'] >= default_threshold
        adaptive_pass = quality_metrics['coverage_ratio'] >= adaptive_threshold
        
        print(f"  デフォルト判定: {'✅ PASS' if default_pass else '❌ FAIL'}")
        print(f"  適応的判定: {'✅ PASS' if adaptive_pass else '❌ FAIL'}")
        
        if not default_pass and adaptive_pass:
            print(f"  🎉 小さなキャラクター対応により判定改善！")


def generate_test_report():
    """テストレポート生成"""
    print("\n📋 小さなキャラクター対応面積閾値テストレポート生成")
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'small_character_area_threshold',
        'adaptive_filtering_implemented': True,
        'adaptive_threshold_calculation': True,
        'test_cases': []
    }
    
    test_masks = create_small_character_test_masks()
    checker = UnifiedQualityChecker()
    
    for test_case in test_masks:
        mask = test_case['mask']
        quality_metrics = calculate_mask_quality_metrics(mask)
        
        # 適応的処理
        adaptive_threshold = checker._calculate_adaptive_coverage_threshold([quality_metrics])
        adaptive_filtered = remove_small_components_adaptive(mask)
        adaptive_metrics = calculate_mask_quality_metrics(adaptive_filtered)
        
        test_result = {
            'name': test_case['name'],
            'description': test_case['description'],
            'expected_ratio': test_case['expected_ratio'],
            'actual_coverage': quality_metrics['coverage_ratio'],
            'default_threshold': checker.thresholds["coverage_ratio"],
            'adaptive_threshold': adaptive_threshold,
            'default_pass': bool(quality_metrics['coverage_ratio'] >= checker.thresholds["coverage_ratio"]),
            'adaptive_pass': bool(quality_metrics['coverage_ratio'] >= adaptive_threshold),
            'improvement': bool(not (quality_metrics['coverage_ratio'] >= checker.thresholds["coverage_ratio"]) 
                          and (quality_metrics['coverage_ratio'] >= adaptive_threshold))
        }
        
        test_results['test_cases'].append(test_result)
    
    # レポート保存
    report_file = f"small_character_threshold_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"📄 テストレポート保存: {report_file}")
    
    # サマリー表示
    improved_cases = sum(1 for case in test_results['test_cases'] if case['improvement'])
    total_cases = len(test_results['test_cases'])
    
    print(f"\n📊 テスト結果サマリー:")
    print(f"  総テストケース: {total_cases}")
    print(f"  改善されたケース: {improved_cases}")
    print(f"  改善率: {improved_cases/total_cases:.1%}")
    
    return test_results


def main():
    """メイン処理"""
    print("🎯 小さなキャラクター対応面積閾値調整テスト")
    print("="*80)
    
    # 1. 適応的面積フィルタリングテスト
    test_adaptive_area_filtering()
    
    # 2. 適応的閾値計算テスト
    test_adaptive_threshold_calculation()
    
    # 3. テストレポート生成
    test_results = generate_test_report()
    
    print("\n" + "="*80)
    print("✅ 小さなキャラクター対応面積閾値調整テスト完了")
    
    return 0


if __name__ == "__main__":
    exit(main())