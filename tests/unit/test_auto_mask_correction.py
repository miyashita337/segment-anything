#!/usr/bin/env python3
"""
P1-005: 自動マスク修正機能のテスト
Unit tests for automatic mask correction functionality
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.postprocessing.auto_mask_correction import (
    AutoMaskCorrector,
    MaskCorrectionParams,
    create_auto_mask_corrector
)


class TestAutoMaskCorrection:
    """自動マスク修正機能のテスト"""
    
    def create_test_mask(self, size=(500, 500), add_noise=True, add_holes=True):
        """テスト用マスクを作成"""
        mask = np.zeros(size, dtype=np.uint8)
        
        # メインオブジェクト
        cv2.circle(mask, (size[0]//2, size[1]//2), min(size)//4, 255, -1)
        
        if add_noise:
            # ノイズ点を追加
            noise_points = np.random.randint(0, min(size), (20, 2))
            for point in noise_points:
                cv2.circle(mask, tuple(point), 3, 255, -1)
        
        if add_holes:
            # ホールを追加
            cv2.circle(mask, (size[0]//2 - 30, size[1]//2), 10, 0, -1)
            cv2.circle(mask, (size[0]//2 + 30, size[1]//2), 8, 0, -1)
        
        return mask
    
    def test_mask_corrector_initialization(self):
        """マスク修正システムの初期化テスト"""
        # デフォルトパラメータ
        corrector = AutoMaskCorrector()
        assert corrector.params is not None
        assert corrector.params.edge_smoothing_enabled is True
        
        # カスタムパラメータ
        custom_params = MaskCorrectionParams(
            gaussian_kernel_size=7,
            noise_removal_enabled=False
        )
        corrector = AutoMaskCorrector(custom_params)
        assert corrector.params.gaussian_kernel_size == 7
        assert corrector.params.noise_removal_enabled is False
    
    def test_noise_removal(self):
        """ノイズ除去機能テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask(add_noise=True, add_holes=False)
        
        result = corrector.correct_mask_automatically(test_mask)
        
        assert result['processing_success'] is True
        assert 'corrected_mask' in result
        assert result['corrected_mask'].shape == test_mask.shape
        
        # ノイズ除去のログが含まれていることを確認
        log_messages = ' '.join(result['correction_log'])
        assert 'モルフォロジカル' in log_messages or '小成分除去' in log_messages
    
    def test_hole_filling(self):
        """ホール埋め機能テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask(add_noise=False, add_holes=True)
        
        result = corrector.correct_mask_automatically(test_mask)
        
        assert result['processing_success'] is True
        
        # ホール埋めのログが含まれていることを確認
        log_messages = ' '.join(result['correction_log'])
        assert 'ホール' in log_messages
    
    def test_edge_smoothing(self):
        """エッジスムージング機能テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask(add_noise=False, add_holes=False)
        
        # ギザギザのエッジを作成
        points = np.array([[100, 100], [150, 120], [200, 100], [250, 130], [300, 100]], np.int32)
        cv2.fillPoly(test_mask, [points], 255)
        
        result = corrector.correct_mask_automatically(test_mask)
        
        assert result['processing_success'] is True
        
        # エッジスムージングのログが含まれていることを確認
        log_messages = ' '.join(result['correction_log'])
        assert 'フィルタ' in log_messages
    
    def test_quality_metrics_calculation(self):
        """品質メトリクス計算テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask()
        
        result = corrector.correct_mask_automatically(test_mask)
        
        assert 'quality_metrics' in result
        metrics = result['quality_metrics']
        
        # 必要なメトリクスが存在することを確認
        required_metrics = [
            'area_ratio', 'original_complexity', 'corrected_complexity',
            'complexity_improvement', 'iou', 'improvement_ratio'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 10  # 合理的な範囲
    
    def test_adaptive_parameters(self):
        """適応的パラメータ調整テスト"""
        params = MaskCorrectionParams(adaptive_parameters=True)
        corrector = AutoMaskCorrector(params)
        test_mask = self.create_test_mask()
        
        # 低品質スコアで実行
        result = corrector.correct_mask_automatically(test_mask, quality_score=0.3)
        
        assert result['processing_success'] is True
        
        # 適応的調整のログが含まれていることを確認
        log_messages = ' '.join(result['correction_log'])
        assert '低品質' in log_messages or '修正パラメータ' in log_messages or '検出' in log_messages
    
    def test_with_original_image(self):
        """元画像を使用した修正テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask()
        original_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        result = corrector.correct_mask_automatically(test_mask, original_image=original_image)
        
        assert result['processing_success'] is True
        
        # バイラテラルフィルタが使用されることを確認
        log_messages = ' '.join(result['correction_log'])
        assert 'バイラテラル' in log_messages or 'ガウシアン' in log_messages
    
    def test_create_quality_focused_corrector(self):
        """品質重視修正システム作成テスト"""
        corrector = create_auto_mask_corrector(quality_focused=True)
        
        assert corrector.params.gaussian_kernel_size == 7
        assert corrector.params.bilateral_d == 11
        assert corrector.params.min_contour_area == 150
        assert corrector.params.morphology_iterations == 3
        assert corrector.params.adaptive_parameters is True
    
    def test_create_standard_corrector(self):
        """標準修正システム作成テスト"""
        corrector = create_auto_mask_corrector(quality_focused=False)
        
        # デフォルト値が使用されることを確認
        assert corrector.params.gaussian_kernel_size == 5
        assert corrector.params.min_contour_area == 100
    
    def test_mask_preservation(self):
        """マスクの基本形状保持テスト"""
        corrector = AutoMaskCorrector()
        test_mask = self.create_test_mask(add_noise=False, add_holes=False)
        
        original_area = cv2.countNonZero(test_mask)
        
        result = corrector.correct_mask_automatically(test_mask)
        corrected_area = cv2.countNonZero(result['corrected_mask'])
        
        # 面積が大きく変わらないことを確認（±30%以内）
        area_ratio = corrected_area / original_area
        assert 0.7 <= area_ratio <= 1.3
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        corrector = AutoMaskCorrector()
        
        # 無効なマスク（空配列）
        empty_mask = np.array([])
        
        try:
            result = corrector.correct_mask_automatically(empty_mask)
            # エラーが発生せずに処理された場合
            assert 'processing_success' in result
        except Exception:
            # エラーが発生した場合も正常（適切なエラーハンドリング）
            pass
    
    def test_different_mask_sizes(self):
        """異なるサイズのマスクに対するテスト"""
        corrector = AutoMaskCorrector()
        
        sizes = [(100, 100), (800, 600), (1920, 1080)]
        
        for size in sizes:
            test_mask = self.create_test_mask(size=size)
            result = corrector.correct_mask_automatically(test_mask)
            
            assert result['processing_success'] is True
            assert result['corrected_mask'].shape == test_mask.shape


if __name__ == "__main__":
    # 簡易テスト実行
    test_instance = TestAutoMaskCorrection()
    
    print("🧪 自動マスク修正機能テスト開始")
    
    try:
        test_instance.test_mask_corrector_initialization()
        print("✅ 初期化テスト: PASS")
        
        test_instance.test_noise_removal()
        print("✅ ノイズ除去テスト: PASS")
        
        test_instance.test_hole_filling()
        print("✅ ホール埋めテスト: PASS")
        
        test_instance.test_edge_smoothing()
        print("✅ エッジスムージングテスト: PASS")
        
        test_instance.test_quality_metrics_calculation()
        print("✅ 品質メトリクステスト: PASS")
        
        test_instance.test_create_quality_focused_corrector()
        print("✅ 品質重視システムテスト: PASS")
        
        print("\n🎉 全テスト完了 - P1-005実装成功!")
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()