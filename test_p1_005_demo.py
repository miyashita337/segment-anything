#!/usr/bin/env python3
"""
P1-005: 自動マスク修正機能デモ
実際の画像を使用した自動マスク修正機能のデモンストレーション
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from features.processing.postprocessing.auto_mask_correction import create_auto_mask_corrector


def run_p1_005_demo():
    """P1-005自動マスク修正機能のデモ実行"""
    
    print("🔧 P1-005: 自動マスク修正機能 デモンストレーション")
    print("=" * 60)
    
    # テスト用の合成マスクを生成
    print("\n📝 ステップ1: テスト用マスク作成")
    test_mask = create_test_mask_with_issues()
    print(f"   テストマスクサイズ: {test_mask.shape}")
    
    # 自動マスク修正システムを作成
    print("\n🤖 ステップ2: 自動マスク修正システム初期化")
    corrector = create_auto_mask_corrector(quality_focused=True)
    print("   品質重視設定で初期化完了")
    
    # マスク修正実行
    print("\n🔧 ステップ3: 自動マスク修正実行")
    result = corrector.correct_mask_automatically(test_mask)
    
    if result['processing_success']:
        print("✅ 自動マスク修正完了")
        
        # 処理ログ表示
        print("\n📋 処理ログ:")
        for i, log_entry in enumerate(result['correction_log'], 1):
            print(f"   {i}. {log_entry}")
        
        # 品質メトリクス表示
        print("\n📊 品質メトリクス:")
        metrics = result['quality_metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        # 改善効果
        improvement = result['improvement_ratio']
        print(f"\n🎯 総合改善率: {improvement:.2%}")
        
        if improvement > 0.1:
            print("✅ 大幅な改善が確認されました")
        elif improvement > 0.05:
            print("✅ 中程度の改善が確認されました")
        else:
            print("ℹ️ 軽微な改善です")
        
        # マスク比較統計
        original_area = cv2.countNonZero(test_mask)
        corrected_area = cv2.countNonZero(result['corrected_mask'])
        
        print(f"\n📈 マスク比較:")
        print(f"   元マスク面積: {original_area} pixels")
        print(f"   修正マスク面積: {corrected_area} pixels")
        print(f"   面積変化: {((corrected_area - original_area) / original_area * 100):+.1f}%")
        
        return True
        
    else:
        print("❌ 自動マスク修正失敗")
        return False


def create_test_mask_with_issues():
    """問題のあるテストマスクを作成"""
    
    # 基本マスク作成
    mask = np.zeros((600, 800), dtype=np.uint8)
    
    # メインキャラクター領域
    cv2.ellipse(mask, (400, 300), (120, 180), 0, 0, 360, 255, -1)
    
    # ノイズを追加（小さな点）
    noise_positions = [
        (100, 100), (150, 120), (700, 500), (50, 550),
        (750, 50), (200, 200), (600, 150), (300, 500)
    ]
    
    for pos in noise_positions:
        cv2.circle(mask, pos, np.random.randint(3, 8), 255, -1)
    
    # ホールを追加（キャラクター内の穴）
    cv2.circle(mask, (380, 280), 12, 0, -1)  # 顔部分の穴
    cv2.circle(mask, (420, 320), 8, 0, -1)   # 胴体部分の穴
    cv2.ellipse(mask, (400, 380), (15, 8), 0, 0, 360, 0, -1)  # 楕円の穴
    
    # ギザギザのエッジを作成（腕の部分）
    jagged_points = np.array([
        [320, 250], [350, 240], [380, 260], [410, 245],
        [440, 265], [470, 250], [500, 270], [520, 260],
        [520, 290], [500, 300], [470, 285], [440, 295],
        [410, 280], [380, 290], [350, 275], [320, 285]
    ], np.int32)
    
    cv2.fillPoly(mask, [jagged_points], 255)
    
    # より複雑なノイズパターン
    # 線状ノイズ
    cv2.line(mask, (600, 400), (650, 450), 255, 3)
    cv2.line(mask, (150, 350), (200, 400), 255, 2)
    
    # 小ブロック状ノイズ
    cv2.rectangle(mask, (500, 100), (510, 110), 255, -1)
    cv2.rectangle(mask, (250, 450), (265, 465), 255, -1)
    
    return mask


def analyze_mask_quality(original_mask, corrected_mask):
    """マスク品質の詳細分析"""
    
    print("\n🔍 詳細マスク品質分析:")
    
    # 連結成分数
    original_components = cv2.connectedComponents(original_mask)[0] - 1
    corrected_components = cv2.connectedComponents(corrected_mask)[0] - 1
    
    print(f"   連結成分数: {original_components} → {corrected_components}")
    
    # 輪郭の滑らかさ（周囲長/面積比）
    def calculate_smoothness(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        
        total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        total_area = cv2.countNonZero(mask)
        return total_perimeter / total_area if total_area > 0 else 0
    
    original_smoothness = calculate_smoothness(original_mask)
    corrected_smoothness = calculate_smoothness(corrected_mask)
    
    print(f"   エッジ複雑度: {original_smoothness:.4f} → {corrected_smoothness:.4f}")
    
    # 充填率（convex hullに対する面積比）
    def calculate_fill_ratio(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        
        # 最大の輪郭を使用
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        
        contour_area = cv2.contourArea(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        return contour_area / hull_area if hull_area > 0 else 0
    
    original_fill = calculate_fill_ratio(original_mask)
    corrected_fill = calculate_fill_ratio(corrected_mask)
    
    print(f"   充填率: {original_fill:.4f} → {corrected_fill:.4f}")


if __name__ == "__main__":
    try:
        success = run_p1_005_demo()
        
        if success:
            print("\n🎉 P1-005自動マスク修正機能デモ完了!")
            print("   システムは正常に動作しています。")
        else:
            print("\n❌ デモ実行中にエラーが発生しました。")
            
    except Exception as e:
        print(f"\n💥 予期せぬエラー: {e}")
        import traceback
        traceback.print_exc()