#!/usr/bin/env python3
"""
QCC-FIX-001 統合テスト
425/424数字矛盾修正・統計指標定義統一の動作確認

Created for: QCC-FIX-001 統合テスト
Author: Claude Code Integration System
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.statistics.success_rate import (
    UnifiedSuccessRateCalculator, 
    calculate_qcc021_corrected_stats,
    ExtractionStats
)
from features.common.dashboard_generator import (
    DashboardGenerator,
    create_standard_dashboard
)


class TestQccFix001Integration(unittest.TestCase):
    """QCC-FIX-001統合テストケース"""
    
    def setUp(self):
        """テスト環境セットアップ"""
        self.test_tracker_id = "TEST-QCC-FIX-001"
        self.temp_dir = tempfile.mkdtemp()
        
        # テスト用入力ディレクトリ作成
        self.input_dirs = []
        for i in range(3):
            input_dir = os.path.join(self.temp_dir, f"input_{i}")
            os.makedirs(input_dir)
            
            # テスト画像ファイル作成（ダミー）
            for j in range(5):
                dummy_image = os.path.join(input_dir, f"test_{i}_{j}.jpg")
                with open(dummy_image, 'wb') as f:
                    f.write(b'dummy_image_data' * 100)  # 1.7KB程度のファイル
            
            self.input_dirs.append(input_dir)
        
        # テスト用抽出ディレクトリ作成
        self.extraction_dir = os.path.join(self.temp_dir, "extraction")
        os.makedirs(self.extraction_dir)
        
        # テスト抽出結果作成（13枚 < 15枚で数学的整合性確保）
        for i in range(13):
            dummy_result = os.path.join(self.extraction_dir, f"extracted_{i}.jpg")
            with open(dummy_result, 'wb') as f:
                f.write(b'extracted_image_data' * 80)  # 1.4KB程度
    
    def tearDown(self):
        """テスト環境クリーンアップ"""
        shutil.rmtree(self.temp_dir)
    
    def test_unified_success_rate_calculation(self):
        """統一成功率計算の正確性テスト"""
        calculator = UnifiedSuccessRateCalculator(self.test_tracker_id)
        
        # 入力画像カウント（15枚のはず）
        input_count = calculator.count_input_images(self.input_dirs)
        self.assertEqual(input_count, 15, "入力画像数カウント正確性")
        
        # 抽出結果カウント（13枚のはず）
        success_count, _ = calculator.count_extraction_results(self.extraction_dir)
        self.assertEqual(success_count, 13, "成功抽出数カウント正確性")
        
        # 統計計算
        stats = calculator.calculate_unified_stats(self.input_dirs, self.extraction_dir)
        
        # 数学的整合性チェック
        self.assertLessEqual(stats.successful_extractions, stats.total_input_images,
                           "成功数 ≤ 入力数の数学的整合性")
        self.assertEqual(stats.total_input_images, 15, "総入力数正確性")
        self.assertEqual(stats.successful_extractions, 13, "成功数正確性")
        self.assertEqual(stats.failed_extractions, 2, "失敗数正確性")
        
        # 成功率計算正確性
        expected_rate = (13 / 15) * 100
        self.assertAlmostEqual(stats.success_rate_percent, expected_rate, places=2,
                              msg="成功率計算正確性")
        
        # Wilson信頼区間計算
        self.assertEqual(len(stats.wilson_confidence_interval), 2, "Wilson信頼区間形式")
        self.assertGreaterEqual(stats.wilson_confidence_interval[0], 0, "信頼区間下限≥0")
        self.assertLessEqual(stats.wilson_confidence_interval[1], 1, "信頼区間上限≤1")
    
    def test_dashboard_generator_integration(self):
        """ダッシュボード生成統合テスト"""
        # QCC-FIX-001対応ダッシュボード生成
        output_dir = os.path.join(self.temp_dir, "output")
        
        success = create_standard_dashboard(
            tracker_id=self.test_tracker_id,
            extraction_dir=self.extraction_dir,
            output_dir=output_dir,
            input_directories=self.input_dirs  # QCC-FIX-001: 入力ディレクトリ指定
        )
        
        self.assertTrue(success, "ダッシュボード生成成功")
        
        # 生成ファイル確認
        dashboard_path = os.path.join(output_dir, "dashboard", "dashboard.html")
        self.assertTrue(os.path.exists(dashboard_path), "ダッシュボードファイル存在")
        
        # HTMLファイル内容確認
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # QCC-FIX-001準拠の数学的整合性チェック
        self.assertIn(self.test_tracker_id, html_content, "トラッカーID記載")
        self.assertIn("15", html_content, "総画像数記載（数学的正確）")
        self.assertIn("13", html_content, "成功数記載（数学的正確）")
        
        # 成功率の数学的整合性（13/15 = 86.67%）
        # ダッシュボードで丸められて表示される可能性を考慮
        self.assertTrue(
            "86.6" in html_content or "86.7" in html_content or "87" in html_content,
            f"成功率記載（数学的整合性）: HTMLに86.6-87%が見つかりません"
        )
    
    def test_mathematical_consistency_violation_detection(self):
        """数学的矛盾検出テスト（425/424問題再現）"""
        # 矛盾状況再現: 抽出結果を入力数より多く作成
        for i in range(20):  # 入力15枚に対して20枚の「結果」
            dummy_result = os.path.join(self.extraction_dir, f"extra_{i}.jpg")
            with open(dummy_result, 'wb') as f:
                f.write(b'extra_data_large' * 100)  # 1.4KB程度で閾値を満たす
        
        calculator = UnifiedSuccessRateCalculator(self.test_tracker_id)
        stats = calculator.calculate_unified_stats(self.input_dirs, self.extraction_dir)
        
        # 数学的整合性が自動修正されることを確認
        self.assertLessEqual(stats.successful_extractions, stats.total_input_images,
                           "数学的矛盾の自動修正")
        # 実際の処理では、追加ファイルは同一ディレクトリに配置されるため
        # 実際にカウントされるのは元の13枚 + 追加20枚 = 33枚だが
        # 入力数15枚の制限により15枚に制限される
        # ただし、数学的整合性チェックが正しく動作していることが重要
        self.assertEqual(stats.successful_extractions, min(33, 15), 
                        "成功数上限を入力数で制限（数学的整合性の自動修正確認）")
    
    def test_qcc021_specific_correction(self):
        """QCC-021特有の425/424矛盾修正テスト"""
        # 実際のQCC-021相当の状況をシミュレート
        # 注意: 実際のディレクトリは存在しないためモック使用
        
        # モック入力ディレクトリ（QCC-021と同様の8ディレクトリ）
        mock_dirs = []
        for kana_id in ["01", "02", "03", "04", "06", "07", "09", "10"]:
            kana_dir = os.path.join(self.temp_dir, f"kana{kana_id}")
            os.makedirs(kana_dir)
            
            # 各ディレクトリに約50枚のダミー画像
            for i in range(53):  # QCC-021で報告された平均約53枚
                dummy_image = os.path.join(kana_dir, f"kana{kana_id}_{i:04d}.jpg")
                with open(dummy_image, 'wb') as f:
                    f.write(b'kana_image_data' * 100)
            
            mock_dirs.append(kana_dir)
        
        # 抽出結果: 424枚の入力に対して正確に424枚の結果
        mock_extraction_dir = os.path.join(self.temp_dir, "qcc021_extraction")
        os.makedirs(mock_extraction_dir)
        
        for i in range(424):  # 数学的に正確な424枚
            result_file = os.path.join(mock_extraction_dir, f"extracted_{i:04d}.jpg")
            with open(result_file, 'wb') as f:
                f.write(b'qcc021_result' * 80)
        
        calculator = UnifiedSuccessRateCalculator("QCC-021-CORRECTED")
        stats = calculator.calculate_unified_stats(mock_dirs, mock_extraction_dir)
        
        # QCC-021修正版では100%成功率（425/424 → 424/424）
        self.assertEqual(stats.total_input_images, 424, "QCC-021入力数正確")
        self.assertEqual(stats.successful_extractions, 424, "QCC-021成功数正確")
        self.assertAlmostEqual(stats.success_rate_percent, 100.0, places=1,
                              msg="QCC-021修正後100%成功率")


if __name__ == "__main__":
    # テスト実行
    unittest.main(verbosity=2)