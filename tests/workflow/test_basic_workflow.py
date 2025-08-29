#!/usr/bin/env python3
"""
Level 1: 基本ワークフローテスト

基本的なワークフロー要素のテスト:
- 入力パス検証
- トラッカーID検証
- ワークスペースディレクトリ作成
- 基本的なファイル操作
"""

import tempfile
import pytest
from pathlib import Path
import sys
import os

# テスト対象をインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.utils.input_path_validator import InputPathValidator, InputPathValidationError
from tools.utils.tracker_id_validator import TrackerIdValidator, TrackerIdValidationError
from tools.utils.workflow_validator import WorkflowValidator, WorkflowValidationError


class TestBasicWorkflow:
    """Level 1: 基本ワークフローテストクラス"""
    
    # ================================
    # 入力パス検証テスト（10テストケース）
    # ================================
    
    def test_valid_input_path(self):
        """有効な入力パスの検証テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid, error_msg = InputPathValidator.validate_input_path(temp_dir)
            assert is_valid == True
            assert error_msg is None
    
    def test_nonexistent_input_path(self):
        """存在しない入力パスでのエラーテスト"""
        nonexistent_path = "/this/path/does/not/exist"
        
        with pytest.raises(InputPathValidationError) as exc_info:
            InputPathValidator.validate_input_path(nonexistent_path)
        
        error_msg = str(exc_info.value)
        assert "❌ エラー: 入力ディレクトリが存在しません" in error_msg
        assert nonexistent_path in error_msg
    
    def test_file_instead_of_directory_input(self):
        """ファイルを入力ディレクトリとして指定した場合のエラーテスト"""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(InputPathValidationError) as exc_info:
                InputPathValidator.validate_input_path(temp_file.name)
            
            error_msg = str(exc_info.value)
            assert "❌ エラー: 指定されたパスはディレクトリではありません" in error_msg
    
    def test_empty_input_path(self):
        """空の入力パスでのエラーテスト"""
        with pytest.raises(InputPathValidationError):
            InputPathValidator.validate_input_path("")
    
    def test_relative_input_path(self):
        """相対パスの入力パステスト"""
        # 現在ディレクトリは存在するはず
        is_valid, error_msg = InputPathValidator.validate_input_path(".")
        assert is_valid == True
        assert error_msg is None
    
    # ================================
    # トラッカーID検証テスト（8テストケース）
    # ================================
    
    def test_valid_tracker_ids(self):
        """有効なトラッカーIDの検証テスト"""
        valid_ids = [
            "TEST-001", "TEST-999",
            "QUAL-001", "QUAL-042",
            "INTG-001", "INTG-100",
            "OPTM-001", "OPTM-050",
            "INCI-001", "INCI-999",
            "P1-A001", "P1-B042",
            "PH1-001", "PH2-100",
            "QCC-001", "QCC-999"
        ]
        
        for tracker_id in valid_ids:
            is_valid, error_msg = TrackerIdValidator.validate_tracker_id(tracker_id)
            assert is_valid == True, f"Failed for {tracker_id}"
            assert error_msg is None
    
    def test_invalid_tracker_ids(self):
        """無効なトラッカーIDでのエラーテスト"""
        invalid_ids = [
            "TEST001",      # ハイフンなし
            "TEST-1",       # 桁数不足
            "TEST-1234",    # 桁数超過
            "test-001",     # 小文字
            "UNKNOWN-001",  # 未サポートプレフィックス
            "TEST_001",     # アンダースコア
            "",             # 空文字
            "TEST-abc",     # 数字以外
            "P1-1",         # P1形式桁数不足
            "PH-001"        # PH形式番号なし
        ]
        
        for tracker_id in invalid_ids:
            with pytest.raises(TrackerIdValidationError):
                TrackerIdValidator.validate_tracker_id(tracker_id)
    
    def test_tracker_id_prefix_extraction(self):
        """トラッカーIDプレフィックス抽出テスト"""
        test_cases = [
            ("TEST-001", "TEST"),
            ("QUAL-042", "QUAL"),
            ("P1-A001", "P1"),
            ("PH1-001", "PH1"),
            ("PH2-042", "PH2")
        ]
        
        for tracker_id, expected_prefix in test_cases:
            prefix = TrackerIdValidator.extract_prefix(tracker_id)
            assert prefix == expected_prefix
    
    def test_empty_tracker_id(self):
        """空のトラッカーIDでのエラーテスト"""
        with pytest.raises(TrackerIdValidationError) as exc_info:
            TrackerIdValidator.validate_tracker_id("")
        
        error_msg = str(exc_info.value)
        assert "空のトラッカーID" in error_msg
    
    # ================================
    # ワークスペース作成テスト（5テストケース）
    # ================================
    
    def test_workspace_directory_creation(self):
        """ワークスペースディレクトリ作成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = f"{temp_dir}/TEST-001"
            
            # 新規作成
            created = WorkflowValidator.create_workspace_directory(workspace_path)
            assert created == True
            
            # 必要なディレクトリが作成されているか確認
            workspace = Path(workspace_path)
            assert workspace.exists()
            assert (workspace / "extraction").exists()
            assert (workspace / "dashboard").exists()
            assert (workspace / "logs").exists()
    
    def test_existing_workspace_directory(self):
        """既存ワークスペースディレクトリの確認テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = f"{temp_dir}/TEST-001"
            
            # 事前作成
            Path(workspace_path).mkdir(parents=True)
            
            # 既存確認
            created = WorkflowValidator.create_workspace_directory(workspace_path)
            assert created == False
    
    def test_workspace_file_conflict(self):
        """ワークスペースパスにファイルが存在する場合のエラーテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # ファイル作成
            file_path = f"{temp_dir}/conflict_file"
            Path(file_path).touch()
            
            # ディレクトリ作成でエラー
            with pytest.raises(WorkflowValidationError):
                WorkflowValidator.create_workspace_directory(file_path)
    
    def test_workflow_completion_check(self):
        """ワークフロー完了状況確認テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = f"{temp_dir}/TEST-001"
            WorkflowValidator.create_workspace_directory(workspace_path)
            
            # 初期状態（未完了）
            is_complete, details = WorkflowValidator.validate_workflow_completion(workspace_path)
            assert is_complete == False
            assert details["completion_rate"] == 0.0
            
            # 一部ファイル作成
            workspace = Path(workspace_path)
            (workspace / "extraction" / "extraction_result.json").touch()
            
            # 部分完了
            is_complete, details = WorkflowValidator.validate_workflow_completion(workspace_path)
            assert is_complete == False
            assert 0.0 < details["completion_rate"] < 100.0
            
            # 全ファイル作成
            (workspace / "dashboard" / "dashboard.html").touch()
            (workspace / "logs" / "processing.log").touch()
            
            # 完了
            is_complete, details = WorkflowValidator.validate_workflow_completion(workspace_path)
            assert is_complete == True
            assert details["completion_rate"] == 100.0
    
    def test_nonexistent_workspace_completion_check(self):
        """存在しないワークスペースでの完了確認エラーテスト"""
        with pytest.raises(WorkflowValidationError):
            WorkflowValidator.validate_workflow_completion("/nonexistent/workspace")
    
    # ================================
    # 統合ワークフロー検証テスト（3テストケース）
    # ================================
    
    def test_integrated_workflow_validation_success(self):
        """統合ワークフロー検証成功テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 有効な入力ディレクトリ作成
            input_dir = f"{temp_dir}/input"
            Path(input_dir).mkdir()
            
            # 統合検証実行
            is_valid, details = WorkflowValidator.validate_workflow_inputs(
                "TEST-001", input_dir, f"{temp_dir}/output"
            )
            
            assert is_valid == True
            assert details["tracker_id"] == "TEST-001"
            assert details["input_path"] == input_dir
            assert "✅ 有効" in details["validation_results"]["tracker_id"]
            assert "✅ 存在確認" in details["validation_results"]["input_path"]
    
    def test_integrated_workflow_validation_invalid_tracker(self):
        """統合ワークフロー検証（無効トラッカーID）テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = f"{temp_dir}/input"
            Path(input_dir).mkdir()
            
            with pytest.raises(WorkflowValidationError):
                WorkflowValidator.validate_workflow_inputs(
                    "INVALID-ID", input_dir
                )
    
    def test_integrated_workflow_validation_invalid_input(self):
        """統合ワークフロー検証（無効入力パス）テスト"""
        with pytest.raises(WorkflowValidationError):
            WorkflowValidator.validate_workflow_inputs(
                "TEST-001", "/nonexistent/input"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])