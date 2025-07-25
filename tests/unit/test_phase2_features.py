#!/usr/bin/env python3
"""
Phase 2機能テスト（pytest形式）
エフェクト線除去・マルチコマ分割等のテスト
"""

import pytest
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPhase2Features:
    """Phase 2機能に関するテスト"""
    
    @pytest.fixture
    def initialize_models(self):
        """モデル初期化フィクスチャ"""
        from features.common.hooks.start import start
        start()
        return True
    
    @pytest.fixture
    def test_images(self):
        """Phase 2テスト用画像"""
        return [
            {
                'path': '/mnt/c/AItools/lora/train/yado/org/kaname03/21_kaname03_0020.jpg',
                'name': '21_kaname03_0020.jpg',
                'description': 'ダイナミックなポーズ + エフェクト線'
            },
            {
                'path': '/mnt/c/AItools/lora/train/yado/org/kaname03/16_kaname03_0015.jpg',
                'name': '16_kaname03_0015.jpg',
                'description': 'マルチコマ構成'
            }
        ]
    
    def test_phase1_auto_retry(self, initialize_models, test_images):
        """Phase 1自動リトライ機能のテスト"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        
        for image in test_images:
            if os.path.exists(image['path']):
                result = extract_character_from_path(
                    image['path'],
                    output_path=f"/tmp/test_phase1_{image['name']}",
                    verbose=False,
                    difficult_pose_mode='auto'
                )
                
                if result.get('success', False):
                    success_count += 1
        
        assert success_count >= 1, "少なくとも1つの画像でPhase 1が成功する必要がある"
    
    def test_effect_line_removal(self, initialize_models):
        """エフェクト線除去機能のテスト"""
        from features.processing.preprocessing.manga_preprocessing import apply_manga_preprocessing
        
        test_image = '/mnt/c/AItools/lora/train/yado/org/kaname03/21_kaname03_0020.jpg'
        
        if os.path.exists(test_image):
            result = apply_manga_preprocessing(test_image, enable_effect_removal=True)
            
            assert result is not None, "前処理結果が返される必要がある"
            assert 'has_effect_lines' in result, "エフェクト線検出結果が含まれる必要がある"
            assert 'panel_count' in result, "パネル数が含まれる必要がある"
    
    def test_multi_panel_split(self, initialize_models):
        """マルチコマ分割機能のテスト"""
        from features.processing.preprocessing.manga_preprocessing import apply_manga_preprocessing
        
        test_image = '/mnt/c/AItools/lora/train/yado/org/kaname03/16_kaname03_0015.jpg'
        
        if os.path.exists(test_image):
            result = apply_manga_preprocessing(test_image, enable_panel_split=True)
            
            assert result is not None, "前処理結果が返される必要がある"
            assert 'panel_count' in result, "パネル数が含まれる必要がある"
            if result['panel_count'] > 1:
                assert 'largest_panel' in result, "最大パネル情報が含まれる必要がある"
    
    @pytest.mark.parametrize("quality_method", [
        'balanced',
        'confidence_priority',
        'size_priority',
        'fullbody_priority',
        'central_priority'
    ])
    def test_quality_methods(self, initialize_models, quality_method):
        """各品質評価手法のテスト"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        test_image = "test_small/img001.jpg"
        
        if os.path.exists(test_image):
            result = extract_character_from_path(
                test_image,
                output_path=f"/tmp/test_quality_{quality_method}.jpg",
                verbose=False,
                quality_method=quality_method
            )
            
            assert 'success' in result, "結果に成功フラグが含まれる必要がある"
            assert 'processing_time' in result, "処理時間が記録される必要がある"
    
    def test_error_handling(self, initialize_models):
        """エラーハンドリングのテスト"""
        from features.extraction.commands.extract_character import extract_character_from_path
        
        # 存在しないファイル
        result = extract_character_from_path(
            "non_existent_file.jpg",
            output_path="/tmp/test_error.jpg",
            verbose=False
        )
        
        assert result.get('success') is False, "存在しないファイルでは失敗する必要がある"
        assert 'error' in result, "エラーメッセージが含まれる必要がある"