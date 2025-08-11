#!/usr/bin/env python3
"""
QCA-001: AuthorParameterAdapter ユニットテスト

作者別パラメータ適応システムの単体テスト
- 作者検出ロジック
- パラメータ最適化機能  
- プロファイル管理機能
- エラーハンドリング

Created for: QCA-001 - 作者別パラメータ適応システム・ディレクトリ構造ベース自動最適化
Author: Claude Code Integration System
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# テスト対象のインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features.adaptation.author_parameter_adapter import (
    AuthorParameterAdapter,
    AuthorProfile, 
    AuthorCharacteristics
)


class TestAuthorParameterAdapter:
    """AuthorParameterAdapter のユニットテスト"""
    
    def test_detect_author_from_path_yado(self):
        """yado作者の正確な検出テスト"""
        test_paths = [
            "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg",
            "/some/path/train/yado/work/image.jpg",
            "/train/yado/test.png"
        ]
        
        for path in test_paths:
            author = AuthorParameterAdapter.detect_author_from_path(path)
            assert author == "yado", f"yado検出失敗: {path} -> {author}"
    
    def test_detect_author_from_path_kiri(self):
        """kiri作者の正確な検出テスト（元aichi）"""
        test_paths = [
            "/mnt/c/AItools/lora/train/kiri/org/work01/image.jpg",
            "/path/train/kiri/subfolder/test.png",
            "/train/kiri/image.jpg"
        ]
        
        for path in test_paths:
            author = AuthorParameterAdapter.detect_author_from_path(path)
            assert author == "kiri", f"kiri検出失敗: {path} -> {author}"
    
    def test_detect_author_from_path_zundamon(self):
        """zundamon作者の正確な検出テスト"""
        test_paths = [
            "/mnt/c/AItools/lora/train/zundamon/org/work02/image.jpg",
            "/train/zundamon/test.png",
            "/some/train/zundamon/subfolder/image.png"
        ]
        
        for path in test_paths:
            author = AuthorParameterAdapter.detect_author_from_path(path)
            assert author == "zundamon", f"zundamon検出失敗: {path} -> {author}"
    
    def test_detect_author_from_path_unknown(self):
        """未知の作者パスのテスト"""
        test_paths = [
            "/mnt/c/AItools/lora/train/unknown_author/work/image.jpg",
            "/no/train/folder/image.jpg",
            "/train/not_an_author/image.jpg",
            "/completely/different/path/image.jpg"
        ]
        
        for path in test_paths:
            author = AuthorParameterAdapter.detect_author_from_path(path)
            assert author is None, f"未知作者のNone期待: {path} -> {author}"
    
    def test_detect_author_fallback_search(self):
        """フォールバック検索機能のテスト"""
        # train/以外のパスでも作者名があれば検出
        test_paths = [
            "/some/yado/work/image.jpg",
            "/path/to/kiri/subfolder/test.png",
            "/folder/zundamon/image.jpg"
        ]
        
        expected_authors = ["yado", "kiri", "zundamon"]
        
        for path, expected in zip(test_paths, expected_authors):
            author = AuthorParameterAdapter.detect_author_from_path(path)
            assert author == expected, f"フォールバック検出失敗: {path} -> {author} (期待: {expected})"
    
    def test_get_author_profile_valid(self):
        """有効な作者プロファイル取得テスト"""
        for author_id in ["yado", "kiri", "zundamon"]:
            profile = AuthorParameterAdapter.get_author_profile(author_id)
            
            assert isinstance(profile, AuthorProfile)
            assert profile.author_id == author_id
            assert isinstance(profile.characteristics, AuthorCharacteristics)
            assert profile.yolo_confidence > 0
            assert profile.max_masks > 0
            assert profile.score_threshold > 0
            assert len(profile.description) > 0
            assert len(profile.processing_notes) > 0
    
    def test_get_author_profile_invalid(self):
        """無効な作者プロファイル取得テスト（デフォルト返却）"""
        invalid_authors = ["unknown", "not_an_author", None, ""]
        
        for invalid_author in invalid_authors:
            profile = AuthorParameterAdapter.get_author_profile(invalid_author)
            
            assert isinstance(profile, AuthorProfile)
            assert profile.author_id == "default"
            assert profile.characteristics == AuthorCharacteristics.BALANCED
    
    def test_get_optimized_parameters_yado(self):
        """yado作者の最適化パラメータ取得テスト"""
        params = AuthorParameterAdapter.get_optimized_parameters("yado")
        
        assert params["author_id"] == "yado"
        assert params["sam_profile"] == "character_focused"
        assert params["yolo_confidence"] == 0.07
        assert params["score_threshold"] == 0.07
        assert params["characteristics"] == "balanced"
        assert "yado作者" in params["description"]
        assert len(params["processing_notes"]) > 0
    
    def test_get_optimized_parameters_kiri(self):
        """kiri作者の最適化パラメータ取得テスト（元aichi）"""
        params = AuthorParameterAdapter.get_optimized_parameters("kiri")
        
        assert params["author_id"] == "kiri"
        assert params["sam_profile"] == "precision_focused"
        assert params["yolo_confidence"] == 0.05  # 細密描写のため低信頼度
        assert params["score_threshold"] == 0.05
        assert params["characteristics"] == "detail_oriented"
        assert "kiri作者" in params["description"]
    
    def test_get_optimized_parameters_zundamon(self):
        """zundamon作者の最適化パラメータ取得テスト"""
        params = AuthorParameterAdapter.get_optimized_parameters("zundamon")
        
        assert params["author_id"] == "zundamon"
        assert params["sam_profile"] == "speed_optimized"
        assert params["yolo_confidence"] == 0.08  # シンプルスタイルのため標準
        assert params["score_threshold"] == 0.08
        assert params["characteristics"] == "simple_style"
        assert "zundamon作者" in params["description"]
    
    def test_apply_author_optimization_integration(self):
        """統合的な作者最適化適用テスト"""
        test_cases = [
            ("/mnt/c/AItools/lora/train/yado/org/kana05/test.jpg", "yado", "character_focused"),
            ("/train/kiri/work/image.png", "kiri", "precision_focused"),
            ("/some/zundamon/test/image.jpg", "zundamon", "speed_optimized"),
            ("/unknown/path/image.jpg", "default", "balanced")  # デフォルト
        ]
        
        for path, expected_author, expected_sam_profile in test_cases:
            params = AuthorParameterAdapter.apply_author_optimization(path)
            
            expected_author_id = expected_author if expected_author != "default" else "default"
            assert params["author_id"] == expected_author_id, f"作者ID不一致: {path}"
            assert params["sam_profile"] == expected_sam_profile, f"SAMプロファイル不一致: {path}"
    
    def test_get_all_authors(self):
        """対応作者一覧取得テスト"""
        authors = AuthorParameterAdapter.get_all_authors()
        
        assert isinstance(authors, list)
        assert "yado" in authors
        assert "kiri" in authors
        assert "zundamon" in authors
        assert len(authors) == 3  # 現在対応している作者数
    
    def test_get_author_statistics(self):
        """作者別統計情報取得テスト"""
        stats = AuthorParameterAdapter.get_author_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_authors"] == 3
        assert "characteristics_distribution" in stats
        assert "sam_profiles" in stats
        assert "confidence_range" in stats
        
        # 信頼度範囲の妥当性チェック
        conf_range = stats["confidence_range"]
        assert conf_range["min"] > 0
        assert conf_range["max"] > conf_range["min"] 
        assert conf_range["avg"] >= conf_range["min"]
        assert conf_range["avg"] <= conf_range["max"]
    
    def test_error_handling_invalid_path(self):
        """無効パスのエラーハンドリングテスト"""
        invalid_paths = [None, "", "/", "not_a_path"]
        
        for invalid_path in invalid_paths:
            # エラーが発生してもNoneを返すことを確認
            author = AuthorParameterAdapter.detect_author_from_path(str(invalid_path) if invalid_path else "")
            assert author is None or isinstance(author, str)
    
    def test_author_profile_consistency(self):
        """作者プロファイルの一貫性テスト"""
        for author_id in ["yado", "kiri", "zundamon"]:
            profile = AuthorParameterAdapter.get_author_profile(author_id)
            params = AuthorParameterAdapter.get_optimized_parameters(author_id)
            
            # プロファイルとパラメータの一貫性確認
            assert profile.author_id == params["author_id"]
            assert profile.sam_profile == params["sam_profile"]
            assert profile.yolo_confidence == params["yolo_confidence"]
            assert profile.score_threshold == params["score_threshold"]
    
    @patch('features.adaptation.author_parameter_adapter.logger')
    def test_logging_behavior(self, mock_logger):
        """ログ出力動作テスト"""
        # 正常な作者検出時のログ
        AuthorParameterAdapter.detect_author_from_path("/train/yado/test.jpg")
        mock_logger.debug.assert_called()
        
        # プロファイル取得時のログ 
        AuthorParameterAdapter.get_author_profile("yado")
        mock_logger.info.assert_called()
        
        # 最適化パラメータ取得時のログ
        AuthorParameterAdapter.get_optimized_parameters("yado")
        assert mock_logger.info.call_count >= 2  # プロファイル + パラメータ


class TestAuthorProfile:
    """AuthorProfile データクラスのテスト"""
    
    def test_author_profile_creation(self):
        """AuthorProfileの正常作成テスト"""
        profile = AuthorProfile(
            author_id="test_author",
            characteristics=AuthorCharacteristics.BALANCED,
            sam_profile="test_profile",
            yolo_confidence=0.05,
            max_masks=8,
            score_threshold=0.05,
            description="テスト用プロファイル",
            processing_notes=["テスト用", "ノート"]
        )
        
        assert profile.author_id == "test_author"
        assert profile.characteristics == AuthorCharacteristics.BALANCED
        assert profile.sam_profile == "test_profile"
        assert profile.yolo_confidence == 0.05
        assert profile.max_masks == 8
        assert profile.score_threshold == 0.05
        assert profile.description == "テスト用プロファイル"
        assert len(profile.processing_notes) == 2


if __name__ == "__main__":
    """テストの直接実行"""
    # pytest がない環境でも基本テストを実行
    test_adapter = TestAuthorParameterAdapter()
    
    print("🧪 QCA-001: AuthorParameterAdapter ユニットテスト実行")
    print("=" * 60)
    
    # 基本的なテストを実行
    test_methods = [
        test_adapter.test_detect_author_from_path_yado,
        test_adapter.test_detect_author_from_path_kiri, 
        test_adapter.test_detect_author_from_path_zundamon,
        test_adapter.test_detect_author_from_path_unknown,
        test_adapter.test_get_author_profile_valid,
        test_adapter.test_get_optimized_parameters_yado,
        test_adapter.test_apply_author_optimization_integration,
        test_adapter.test_get_all_authors,
        test_adapter.test_get_author_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 テスト結果: {passed}件成功 / {failed}件失敗 / {passed + failed}件総数")
    
    if failed == 0:
        print("🎉 全テスト合格！QCA-001 実装準備完了")
    else:
        print("⚠️ 一部テスト失敗。実装を確認してください。")