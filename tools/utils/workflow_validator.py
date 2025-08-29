#!/usr/bin/env python3
"""
ワークフロー検証ユーティリティ

基本的なワークフロー要素の検証を統合的に行う
- 入力パス検証（既存システム連携）
- トラッカーID検証
- ワークスペースディレクトリ作成・検証
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .input_path_validator import InputPathValidator, InputPathValidationError
from .tracker_id_validator import TrackerIdValidator, TrackerIdValidationError


class WorkflowValidationError(Exception):
    """ワークフロー検証エラー"""
    pass


class WorkflowValidator:
    """ワークフロー検証クラス"""
    
    # デフォルトワークスペースベース
    DEFAULT_WORKSPACE_BASE = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
    
    @staticmethod
    def validate_workflow_inputs(
        tracker_id: str,
        input_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        ワークフロー入力の統合検証
        
        Args:
            tracker_id: トラッカーID
            input_path: 入力パス
            output_path: 出力パス（Noneの場合は自動生成）
            
        Returns:
            (検証結果, 検証詳細情報)
            
        Raises:
            WorkflowValidationError: 検証失敗時
        """
        validation_details = {
            "tracker_id": tracker_id,
            "input_path": input_path,
            "output_path": output_path,
            "validation_results": {}
        }
        
        try:
            # 1. トラッカーID検証
            TrackerIdValidator.validate_tracker_id(tracker_id)
            validation_details["validation_results"]["tracker_id"] = "✅ 有効"
            
            # 2. 入力パス検証
            InputPathValidator.validate_input_path(input_path)
            validation_details["validation_results"]["input_path"] = "✅ 存在確認"
            
            # 3. 出力パス設定・検証
            if output_path is None:
                output_path = WorkflowValidator._generate_output_path(tracker_id)
                validation_details["output_path"] = output_path
            
            # 出力パス親ディレクトリの存在確認
            output_parent = Path(output_path).parent
            if not output_parent.exists():
                raise WorkflowValidationError(f"出力パス親ディレクトリが存在しません: {output_parent}")
            
            validation_details["validation_results"]["output_path"] = "✅ 準備完了"
            
            # 4. ワークスペースディレクトリ作成
            workspace_created = WorkflowValidator.create_workspace_directory(output_path)
            validation_details["validation_results"]["workspace"] = f"✅ {'作成' if workspace_created else '既存確認'}"
            
            return True, validation_details
            
        except (TrackerIdValidationError, InputPathValidationError, WorkflowValidationError) as e:
            # エラー詳細を含めて再発生
            validation_details["error"] = str(e)
            raise WorkflowValidationError(f"ワークフロー検証失敗:\n{str(e)}")
    
    @staticmethod
    def _generate_output_path(tracker_id: str) -> str:
        """
        トラッカーIDから出力パス自動生成
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            出力パス
        """
        return f"{WorkflowValidator.DEFAULT_WORKSPACE_BASE}/{tracker_id}/"
    
    @staticmethod
    def create_workspace_directory(workspace_path: str) -> bool:
        """
        ワークスペースディレクトリ作成
        
        Args:
            workspace_path: ワークスペースパス
            
        Returns:
            新規作成されたかどうか（False=既存）
        """
        workspace = Path(workspace_path)
        
        if workspace.exists():
            if not workspace.is_dir():
                raise WorkflowValidationError(f"ワークスペースパスがディレクトリではありません: {workspace}")
            return False  # 既存
        else:
            # 必要な子ディレクトリも作成
            workspace.mkdir(parents=True, exist_ok=True)
            
            # 標準ディレクトリ構造作成
            (workspace / "extraction").mkdir(exist_ok=True)
            (workspace / "dashboard").mkdir(exist_ok=True)
            (workspace / "logs").mkdir(exist_ok=True)
            
            return True  # 新規作成
    
    @staticmethod
    def validate_workflow_completion(
        workspace_path: str,
        required_files: Optional[list] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        ワークフロー完了状況検証
        
        Args:
            workspace_path: ワークスペースパス
            required_files: 必須ファイルリスト
            
        Returns:
            (完了状況, 検証詳細情報)
        """
        if required_files is None:
            required_files = [
                "extraction/extraction_result.json",
                "dashboard/dashboard.html",
                "logs/processing.log"
            ]
        
        workspace = Path(workspace_path)
        completion_details = {
            "workspace_path": workspace_path,
            "required_files": required_files,
            "file_status": {},
            "completion_rate": 0.0
        }
        
        if not workspace.exists():
            raise WorkflowValidationError(f"ワークスペースが存在しません: {workspace}")
        
        # 必須ファイル存在確認
        completed_files = 0
        for required_file in required_files:
            file_path = workspace / required_file
            if file_path.exists():
                completion_details["file_status"][required_file] = "✅ 存在"
                completed_files += 1
            else:
                completion_details["file_status"][required_file] = "❌ 未生成"
        
        # 完了率計算
        completion_rate = (completed_files / len(required_files)) * 100
        completion_details["completion_rate"] = completion_rate
        
        return completion_rate == 100.0, completion_details
    
    @staticmethod
    def validate_and_exit_on_error(
        tracker_id: str,
        input_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ワークフロー検証実行（エラー時は即座終了）
        
        Args:
            tracker_id: トラッカーID
            input_path: 入力パス
            output_path: 出力パス
            
        Returns:
            検証詳細情報
        """
        try:
            is_valid, validation_details = WorkflowValidator.validate_workflow_inputs(
                tracker_id, input_path, output_path
            )
            if is_valid:
                return validation_details
        except WorkflowValidationError as e:
            print(str(e))
            sys.exit(1)


def main():
    """CLI実行用メイン関数"""
    if len(sys.argv) < 3:
        print("Usage: python workflow_validator.py <tracker_id> <input_path> [output_path]")
        sys.exit(1)
    
    tracker_id = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        validation_details = WorkflowValidator.validate_and_exit_on_error(
            tracker_id, input_path, output_path
        )
        
        print("✅ ワークフロー検証成功")
        print(f"📋 トラッカーID: {validation_details['tracker_id']}")
        print(f"📁 入力パス: {validation_details['input_path']}")
        print(f"📁 出力パス: {validation_details['output_path']}")
        print("🔍 検証結果:")
        for key, result in validation_details["validation_results"].items():
            print(f"   {key}: {result}")
            
    except WorkflowValidationError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()