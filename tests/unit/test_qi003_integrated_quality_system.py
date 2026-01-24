"""
QI-003統合品質評価システム実装・黒画面検出機能追加のテスト

TDD Red-Green-Refactor アプローチ：
1. Red: 失敗するテストを先に作成
2. Green: 最小実装でテストを通す
3. Refactor: 実装を改善

QI-003要件:
- 統合品質評価システムの実装
- 黒画面検出機能の追加・強化
- ダッシュボード標準化システム
- Pushover通知システム統一化
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# テスト対象となる新機能のインポート（まだ実装されていない）
from features.evaluation.qi003_integrated_quality_system import QI003IntegratedQualitySystem
from features.common.dashboard_generator import DashboardGenerator
from features.common.notification.pushover_image_sender import PushoverImageSender


class TestQI003IntegratedQualitySystem:
    """QI-003統合品質評価システムのテスト"""

    def setup_method(self):
        """各テストメソッド実行前のセットアップ"""
        self.qi003_system = QI003IntegratedQualitySystem()
        
        # テスト用画像データ作成
        self.test_bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200  # 明るい画像
        self.test_black_image = np.ones((100, 100, 3), dtype=np.uint8) * 5    # 黒画面
        self.test_normal_image = np.ones((100, 100, 3), dtype=np.uint8) * 100  # 通常画像
        
        # テスト用パス
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_dir = Path(self.temp_dir) / "input"
        self.test_output_dir = Path(self.temp_dir) / "output"
        self.test_input_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_qi003_system_initialization(self):
        """QI003システムが正しく初期化されるかテスト"""
        system = QI003IntegratedQualitySystem()
        
        # 必要なコンポーネントが初期化されている
        assert hasattr(system, 'black_screen_detector')
        assert hasattr(system, 'quality_monitor')
        assert hasattr(system, 'dashboard_generator')
        assert hasattr(system, 'pushover_sender')
        
        # デフォルト設定の確認
        assert system.brightness_threshold == 20.0
        assert system.quality_threshold == 0.7

    def test_integrated_quality_evaluation(self):
        """統合品質評価が正しく動作するかテスト"""
        # テスト画像リストの準備
        test_images = [
            self.test_bright_image,
            self.test_black_image,
            self.test_normal_image
        ]
        
        # 統合品質評価実行
        results = self.qi003_system.evaluate_integrated_quality(test_images)
        
        # 結果の構造確認
        assert 'total_images' in results
        assert 'black_screen_detected' in results
        assert 'quality_scores' in results
        assert 'recommendations' in results
        
        # 黒画面検出の確認
        assert results['black_screen_detected']['count'] == 1
        assert results['black_screen_detected']['indices'] == [1]  # test_black_image
        
        # 品質スコア配列の確認
        assert len(results['quality_scores']) == 3
        assert all(0.0 <= score <= 1.0 for score in results['quality_scores'])

    def test_black_screen_detection_enhancement(self):
        """強化された黒画面検出機能のテスト"""
        # 様々な暗さレベルの画像テスト
        very_dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 2    # 非常に暗い
        dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 15        # 暗い
        borderline_image = np.ones((100, 100, 3), dtype=np.uint8) * 25  # 境界線
        
        test_cases = [
            (very_dark_image, True, "very dark should be detected"),
            (dark_image, True, "dark should be detected"),
            (borderline_image, False, "borderline should not be detected"),
            (self.test_normal_image, False, "normal should not be detected"),
            (self.test_bright_image, False, "bright should not be detected")
        ]
        
        for image, expected_black, description in test_cases:
            result = self.qi003_system.detect_enhanced_black_screen(image)
            
            assert result['is_black_screen'] == expected_black, f"Failed: {description}"
            assert 'confidence' in result
            assert 'brightness_score' in result
            assert 'enhancement_applied' in result

    def test_quality_boundary_case_handling(self):
        """境界ケースの品質評価処理テスト"""
        # 境界ケース用の特殊な画像
        gradient_base = np.linspace(0, 255, 100).astype(np.uint8)
        gradient_image = np.stack([gradient_base] * 100, axis=0)  # (100, 100)
        gradient_image = np.stack([gradient_image] * 3, axis=2)   # (100, 100, 3)
        noise_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        uniform_dark = np.ones((100, 100, 3), dtype=np.uint8) * 18  # 閾値近く
        
        boundary_cases = [gradient_image, noise_image, uniform_dark]
        
        results = self.qi003_system.handle_boundary_cases(boundary_cases)
        
        # 境界ケース処理結果の確認
        assert 'processed_count' in results
        assert 'improvement_applied' in results
        assert 'final_quality_scores' in results
        
        # AnimeImagePreprocessor による明度改善が適用されている
        assert any(results['improvement_applied'])

    def test_unified_quality_checker_integration(self):
        """統合品質チェッカーの動作確認テスト"""
        # 複数品質評価手法の統合テスト
        test_images = [self.test_normal_image, self.test_black_image]
        
        unified_results = self.qi003_system.run_unified_quality_check(test_images)
        
        # 統合結果の構造確認
        assert 'brightness_analysis' in unified_results
        assert 'edge_quality' in unified_results
        assert 'completeness_scores' in unified_results
        assert 'multi_character_detection' in unified_results
        assert 'partial_extraction_quality' in unified_results
        
        # 各評価手法が実行されている
        for method in ['brightness', 'edge', 'completeness', 'multi_char', 'partial']:
            assert method in unified_results['executed_methods']


class TestDashboardGenerator:
    """標準ダッシュボード生成システムのテスト"""

    def setup_method(self):
        """セットアップ"""
        self.generator = DashboardGenerator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """クリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dashboard_generation_with_base64_images(self):
        """Base64画像埋め込み機能付きダッシュボード生成テスト"""
        # テスト用データ準備
        test_data = {
            'tracker_id': 'QI-003',
            'total_images': 20,
            'quality_scores': [0.8, 0.9, 0.1, 0.7, 0.95],  # 0.1が低品質（黒画面）
            'black_screen_indices': [2],
            'image_paths': [f"test_image_{i}.jpg" for i in range(5)]
        }
        
        # ダッシュボード生成
        dashboard_path = self.generator.generate_standard_dashboard(
            test_data, 
            output_dir=self.temp_dir
        )
        
        # 生成結果の確認
        assert dashboard_path.exists()
        assert dashboard_path.suffix == '.html'
        
        # ファイルサイズ確認（Base64画像埋め込みで2MB以上）
        file_size_mb = dashboard_path.stat().st_size / (1024 * 1024)
        assert file_size_mb >= 2.0, f"Dashboard size too small: {file_size_mb}MB"
        
        # HTML内容の確認
        html_content = dashboard_path.read_text(encoding='utf-8')
        assert 'QI-003' in html_content
        assert 'data:image/jpeg;base64,' in html_content  # Base64埋め込み確認
        assert '高品質' in html_content  # 品質バッジ確認
        assert '低品質' in html_content  # 低品質バッジ確認

    def test_quality_badge_system(self):
        """品質バッジシステムのテスト"""
        test_scores = [0.95, 0.75, 0.45, 0.15, 0.85]
        
        badges = self.generator.generate_quality_badges(test_scores)
        
        # バッジ分類の確認
        expected_badges = ['高品質', '中品質', '低品質', '要改善', '高品質']
        assert badges == expected_badges

    def test_tailwind_responsive_design(self):
        """Tailwind CSS レスポンシブデザインのテスト"""
        html_content = self.generator.generate_responsive_layout()
        
        # Tailwind CSS クラスの存在確認
        assert 'grid-cols-1' in html_content
        assert 'md:grid-cols-2' in html_content
        assert 'lg:grid-cols-3' in html_content
        assert 'responsive' in html_content


class TestPushoverImageSender:
    """Pushover画像送信統一システムのテスト"""

    def setup_method(self):
        """セットアップ"""
        self.sender = PushoverImageSender()

    @patch('requests.post')
    def test_unified_pushover_notification(self, mock_post):
        """統一Pushover通知システムのテスト"""
        # モックレスポンスの設定
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'status': 1}
        
        # テスト用画像パスリスト
        image_paths = [f"test_image_{i}.jpg" for i in range(15)]  # 10枚制限突破テスト
        
        # 統一通知送信
        results = self.sender.send_extraction_complete_with_images(
            tracker_id='QI-003',
            image_paths=image_paths,
            extraction_stats={'success': 12, 'total': 15}
        )
        
        # 結果確認
        assert results['success'] is True
        assert results['batches_sent'] == 2  # 10枚制限で2バッチに分割
        assert results['total_images'] == 15
        assert len(results['batch_results']) == 2

    def test_pushover_system_unification(self):
        """Pushover システム統一化のテスト（モック使用せず直接テスト）"""
        # 統一化されたPushover送信の確認（モック不要）
        
        # 17ファイルの分散実装から統一システムへの移行確認
        unified_success = self.sender.validate_system_unification()
        
        assert unified_success is True
        assert self.sender.unification_rate >= 0.645  # 20/31 = 64.5%以上

    def test_batch_image_sending(self):
        """バッチ画像送信機能（10枚制限対応）のテスト"""
        # 25枚の画像パステスト
        large_image_set = [f"test_{i}.jpg" for i in range(25)]
        
        batches = self.sender.create_image_batches(large_image_set, batch_size=10)
        
        # バッチ分割の確認
        assert len(batches) == 3  # 10, 10, 5枚の3バッチ
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5


class TestQI003IntegrationScenarios:
    """QI-003統合シナリオテスト"""

    def setup_method(self):
        """セットアップ"""
        self.qi003_system = QI003IntegratedQualitySystem()

    def test_qi002_qi003_quality_comparison(self):
        """QI-002とQI-003の品質比較テスト"""
        # QI-002: 24枚中3枚（12.5%）の黒画面問題
        qi002_stats = {
            'total_images': 24,
            'black_screen_count': 3,
            'black_screen_ratio': 0.125
        }
        
        # QI-003: 20枚中3枚（15.0%）の黒画面問題
        qi003_stats = {
            'total_images': 20,
            'black_screen_count': 3,
            'black_screen_ratio': 0.15
        }
        
        comparison = self.qi003_system.compare_quality_improvements(qi002_stats, qi003_stats)
        
        # 品質改善の確認
        assert 'detection_accuracy' in comparison
        assert comparison['detection_accuracy'] == 1.0  # 100%検出精度
        assert 'brightness_improvement' in comparison
        assert comparison['brightness_improvement'] >= 18.2  # 1820%改善

    def test_anime_image_preprocessor_integration(self):
        """AnimeImagePreprocessor統合テスト"""
        # 暗い画像での明度改善テスト
        dark_anime_image = (np.ones((100, 100, 3)) * 6.6).astype(np.uint8)  # 明度6.6
        
        # 明度改善処理
        improved_result = self.qi003_system.apply_anime_preprocessing(dark_anime_image)
        
        # 改善効果の確認
        assert improved_result['original_brightness'] == 6.6
        assert improved_result['improved_brightness'] >= 126.2  # 1820%改善
        assert improved_result['improvement_ratio'] >= 18.2

    def test_full_qi003_workflow_integration(self):
        """QI-003完全ワークフロー統合テスト"""
        # 全コンポーネント統合実行
        workflow_result = self.qi003_system.execute_full_workflow(
            input_images=[],  # 実際のテストでは画像を設定
            output_dir=None
        )
        
        # ワークフロー完了の確認
        assert workflow_result['pushover_unification']['completed'] is True
        assert workflow_result['dashboard_generation']['completed'] is True
        assert workflow_result['quality_evaluation']['completed'] is True
        assert workflow_result['black_screen_detection']['completed'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])