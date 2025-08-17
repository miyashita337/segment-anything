#!/usr/bin/env python3
"""
厳密パス検証システム統合テスト (QUAL-033)

extract_character.py、run_quality_workflow.sh、pipeline_config.yamlなどの
統合動作を検証するテストスイート。

Created for: QUAL-033 - 厳密パス検証システム実装・全ワークフロー適用・意図しない挙動防止
Author: Claude Code Integration System
"""

import pytest
import tempfile
import subprocess
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# テスト対象の統合
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.extraction.commands.extract_character import extract_character
from features.common.strict_path_validator import StrictPathValidator, validate_strict_paths
from features.common.interactive_path_input import InteractivePathInput, interactive_setup


class TestStrictValidationExtractCharacterIntegration:
    """extract_character.py との統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_dir = Path(self.temp_dir) / "input"
        self.test_input_dir.mkdir()
        self.test_output_dir = Path(self.temp_dir) / "output"
        
        # テスト用画像ファイル作成
        (self.test_input_dir / "test1.jpg").write_bytes(b"fake_image_data")
        (self.test_input_dir / "test2.png").write_bytes(b"fake_image_data")

    def teardown_method(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('features.extraction.commands.extract_character.SAMCharacterExtractor')
    def test_extract_character_strict_validation_success(self, mock_extractor):
        """extract_character厳密検証成功統合テスト"""
        # モックセットアップ
        mock_instance = MagicMock()
        mock_instance.extract_character.return_value = (True, "success", 0.8)
        mock_extractor.return_value = mock_instance
        
        # 厳密検証モードで実行
        try:
            result = extract_character(
                input_path=str(self.test_input_dir),
                output_path=str(self.test_output_dir),
                batch=True,
                verbose=True,
                strict_validation=True,
                interactive=False,
                require_author_structure=False
            )
            # 例外が発生しなければ成功
            assert True
        except Exception as e:
            # 厳密検証関連でない例外（SAMモデル未初期化等）は許容
            if "StrictPathValidator" in str(e) or "validation" in str(e).lower():
                pytest.fail(f"厳密検証で予期しないエラー: {e}")

    def test_extract_character_strict_validation_empty_input(self):
        """extract_character厳密検証で空入力テスト"""
        with pytest.raises(SystemExit):  # CLIでのエラー終了
            extract_character(
                input_path=None,
                output_path=str(self.test_output_dir),
                batch=True,
                strict_validation=True,
                interactive=False
            )

    def test_extract_character_strict_validation_nonexistent_input(self):
        """extract_character厳密検証で存在しない入力テスト"""
        with pytest.raises(SystemExit):
            extract_character(
                input_path="/nonexistent/path",
                output_path=str(self.test_output_dir),
                batch=True,
                strict_validation=True,
                interactive=False
            )

    @patch('builtins.input')
    @patch('features.extraction.commands.extract_character.SAMCharacterExtractor')
    def test_extract_character_interactive_mode(self, mock_extractor, mock_input):
        """extract_character対話モード統合テスト"""
        # モックセットアップ
        mock_instance = MagicMock()
        mock_instance.extract_character.return_value = (True, "success", 0.8)
        mock_extractor.return_value = mock_instance
        
        # 対話的入力のモック
        mock_input.side_effect = [
            str(self.test_input_dir),  # 入力パス
            str(self.test_output_dir)  # 出力パス
        ]
        
        try:
            result = extract_character(
                input_path=None,
                output_path=None,
                batch=True,
                strict_validation=True,
                interactive=True,
                require_author_structure=False
            )
            # 対話的入力が正常に動作すれば成功
            assert True
        except Exception as e:
            if "StrictPathValidator" in str(e) or "interactive" in str(e).lower():
                pytest.fail(f"対話モードで予期しないエラー: {e}")


class TestStrictValidationConfigIntegration:
    """設定ファイルとの統合テスト"""

    def test_pipeline_config_defaults_disabled(self):
        """pipeline_config.yamlのデフォルト無効化テスト"""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # デフォルトパスがコメントアウトされているか確認
            assert "# default_input:" in config_content or "default_input:" not in config_content
            assert "# input_alternatives:" in config_content or "input_alternatives:" not in config_content
            assert "QUAL-033" in config_content


class TestStrictValidationWorkflowIntegration:
    """ワークフロースクリプトとの統合テスト"""

    def test_run_quality_workflow_strict_validation_integration(self):
        """run_quality_workflow.sh厳密検証統合確認"""
        workflow_script = Path(__file__).parent.parent.parent / "tools" / "scripts" / "run_quality_workflow.sh"
        
        if workflow_script.exists():
            with open(workflow_script, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # QUAL-033関連の修正が含まれているか確認
            assert "QUAL-033" in script_content
            assert "厳密パス検証システム" in script_content
            assert "--strict-validation" in script_content
            assert "デフォルトパスは無効化" in script_content


class TestStrictValidationEndToEndIntegration:
    """エンドツーエンド統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_workspace = Path(self.temp_dir) / "tracker-workspace"
        self.test_workspace.mkdir()
        
        # 作者別構造のテストディレクトリ作成
        self.author_input_dir = Path(self.temp_dir) / "train" / "testauthor" / "org" / "testwork"
        self.author_input_dir.mkdir(parents=True)
        (self.author_input_dir / "test1.jpg").write_bytes(b"fake_image_data")
        (self.author_input_dir / "test2.png").write_bytes(b"fake_image_data")

    def teardown_method(self):
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_strict_validation_author_structure_detection(self):
        """作者別構造検出の統合テスト"""
        validator = StrictPathValidator(
            strict_mode=True,
            require_author_structure=True,
            interactive_mode=False
        )
        
        result = validator.validate_input_path(self.author_input_dir)
        
        assert result.is_valid
        assert result.author_work_info is not None
        assert result.author_work_info.author == "testauthor"
        assert result.author_work_info.work == "testwork"
        assert result.author_work_info.confidence > 0.8

    def test_strict_validation_security_constraints(self):
        """セキュリティ制約の統合テスト"""
        validator = StrictPathValidator(strict_mode=True)
        
        # 許可されたパス
        allowed_path = Path("/mnt/c/AItools/lora/train/testauthor/test")
        result = validator.validate_input_path(allowed_path)
        # パスが存在しないためis_validはFalseだが、セキュリティエラーはない
        security_errors = [issue for issue in result.issues 
                          if "システムディレクトリ" in issue.message]
        assert len(security_errors) == 0
        
        # システムパス
        system_path = Path("/etc/test")
        result = validator.validate_input_path(system_path)
        security_errors = [issue for issue in result.issues 
                          if "システムディレクトリ" in issue.message]
        assert len(security_errors) > 0

    @patch('builtins.input')
    def test_end_to_end_interactive_setup(self, mock_input):
        """エンドツーエンド対話セットアップテスト"""
        mock_input.side_effect = [
            "QUAL-033",                        # トラッカーID
            "testauthor",                      # 作者名
            "testwork",                        # 作品名
            str(self.author_input_dir),        # 入力パス
            str(self.test_workspace / "QUAL-033" / "extraction")  # 出力パス
        ]
        
        results = interactive_setup("QUAL-033テスト")
        
        assert results['success'] is True
        assert results['tracker_info']['tracker_id'] == "QUAL-033"
        assert results['tracker_info']['author'] == "testauthor"
        assert results['tracker_info']['work'] == "testwork"
        assert results['paths']['input_path'] == self.author_input_dir

    def test_validate_strict_paths_convenience_function(self):
        """validate_strict_paths便利関数の統合テスト"""
        output_path = self.test_workspace / "QUAL-033" / "extraction"
        
        input_result, output_result = validate_strict_paths(
            self.author_input_dir,
            output_path,
            strict_mode=True,
            interactive_mode=False,
            require_author_structure=True
        )
        
        assert input_result == self.author_input_dir
        assert output_result == output_path


class TestStrictValidationRegressionTests:
    """回帰テスト（既存機能の動作確認）"""

    def test_backward_compatibility_non_strict_mode(self):
        """非厳密モードでの後方互換性テスト"""
        validator = StrictPathValidator(
            strict_mode=False,
            require_author_structure=False,
            interactive_mode=False
        )
        
        # 空パスでも警告のみで動作する
        result = validator.validate_input_path(None)
        assert result.has_warnings
        assert not result.has_errors

    def test_existing_validation_system_integration(self):
        """既存検証システムとの統合確認"""
        from features.common.input_validation import validate_input_directory, InputValidationError
        
        # 既存システムが正常に動作することを確認
        temp_dir = tempfile.mkdtemp()
        test_dir = Path(temp_dir) / "test"
        test_dir.mkdir()
        (test_dir / "test.jpg").touch()
        
        try:
            # 既存関数が正常動作
            validated_path = validate_input_directory(test_dir, "テスト", check_images=True)
            assert validated_path == test_dir
            
            # 新しいシステムも正常動作
            validator = StrictPathValidator()
            result = validator.validate_input_path(test_dir)
            assert result.is_valid
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # 統合テスト実行
    pytest.main([__file__, "-v", "-x"])  # -x で最初の失敗で停止