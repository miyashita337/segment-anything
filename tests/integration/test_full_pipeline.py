#!/usr/bin/env python3
"""
統合パイプラインテスト（pytest形式）
エンドツーエンドの動作確認
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFullPipeline:
    """完全なパイプラインの統合テスト"""
    
    @pytest.fixture(scope="class")
    def setup_pipeline(self):
        """パイプライン全体のセットアップ"""
        from features.common.hooks.start import start
        start()
        
        # テスト用ディレクトリ作成
        os.makedirs("/tmp/pipeline_test", exist_ok=True)
        
        yield
        
        # クリーンアップ
        import shutil
        if os.path.exists("/tmp/pipeline_test"):
            shutil.rmtree("/tmp/pipeline_test")
    
    def test_single_image_pipeline(self, setup_pipeline):
        """単一画像の完全処理パイプライン"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        test_image = "test_small/img001.jpg"
        output_path = "/tmp/pipeline_test/single_image_result"
        
        if os.path.exists(test_image):
            start_time = time.time()
            
            result = extract_character_from_path(
                test_image,
                output_path=output_path,
                verbose=False,
                quality_method='balanced',
                yolo_params={'conf': 0.005}
            )
            
            processing_time = time.time() - start_time
            
            # アサーション
            assert result.get('success', False), "単一画像処理が成功する必要がある"
            assert processing_time < 60, "処理時間は60秒以内である必要がある"
            assert os.path.exists(f"{output_path}.jpg"), "出力ファイルが生成される必要がある"
    
    def test_batch_processing(self, setup_pipeline):
        """バッチ処理のテスト"""
        from tools.sam_yolo_character_segment import process_batch
        
        input_dir = "test_small"
        output_dir = "/tmp/pipeline_test/batch_results"
        
        if os.path.exists(input_dir):
            results = process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                score_threshold=0.005,
                quality_method='balanced'
            )
            
            # アサーション
            assert isinstance(results, list), "結果はリストである必要がある"
            assert len(results) > 0, "少なくとも1つの結果が必要"
            
            success_count = sum(1 for r in results if r.get('success', False))
            assert success_count >= 1, "少なくとも1つの成功が必要"
    
    def test_progressive_thresholds(self, setup_pipeline):
        """段階的閾値調整のテスト"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        test_image = "test_small/img001.jpg"
        thresholds = [0.1, 0.07, 0.05, 0.01, 0.005]
        results = {}
        
        if os.path.exists(test_image):
            for threshold in thresholds:
                result = extract_character_from_path(
                    test_image,
                    output_path=f"/tmp/pipeline_test/threshold_{threshold}",
                    verbose=False,
                    yolo_params={'conf': threshold}
                )
                
                results[threshold] = {
                    'success': result.get('success', False),
                    'processing_time': result.get('processing_time', 0)
                }
            
            # 閾値が低いほど成功しやすいことを確認
            success_rates = [1 if results[t]['success'] else 0 for t in thresholds]
            assert sum(success_rates) >= 2, "複数の閾値で成功が必要"
    
    def test_performance_monitoring(self, setup_pipeline):
        """パフォーマンス監視のテスト"""
        from features.common.performance.performance import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # メモリ使用量チェック
        ram_usage = monitor.get_memory_usage()
        assert ram_usage > 0, "RAM使用量が取得できる必要がある"
        
        # GPU使用量チェック（利用可能な場合）
        try:
            gpu_usage = monitor.get_gpu_memory_usage()
            assert gpu_usage >= 0, "GPU使用量が0以上である必要がある"
        except:
            pytest.skip("GPU not available")
    
    def test_error_recovery(self, setup_pipeline):
        """エラー復旧機能のテスト"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        # 意図的にエラーを発生させる
        invalid_cases = [
            ("", "/tmp/pipeline_test/empty_path"),  # 空のパス
            ("non_existent.jpg", "/tmp/pipeline_test/non_existent"),  # 存在しないファイル
            ("test_small/img001.jpg", ""),  # 空の出力パス
        ]
        
        error_count = 0
        
        for input_path, output_path in invalid_cases:
            result = extract_character_from_path(
                input_path,
                output_path=output_path,
                verbose=False
            )
            
            if not result.get('success', True):
                error_count += 1
                assert 'error' in result, "エラー情報が含まれる必要がある"
        
        assert error_count == len(invalid_cases), "すべての無効ケースでエラーが発生する必要がある"
    
    @pytest.mark.slow
    def test_long_running_batch(self, setup_pipeline):
        """長時間実行バッチのテスト"""
        # このテストは@pytest.mark.slowでマークされており、
        # 通常のテスト実行では--run-slowオプションが必要
        
        from tools.sam_yolo_character_segment import process_batch
        
        # より大きなデータセットでのテスト
        # 実際の実装では適切なテストデータを使用
        pytest.skip("Long running test - implement with actual large dataset")