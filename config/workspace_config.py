#!/usr/bin/env python3
"""
ワークスペースパス設定管理
すべてのワークスペースパス設定を一元管理するconfig
"""

import os
import re
from pathlib import Path
from typing import Optional


class WorkspaceConfig:
    """ワークスペース設定一元管理クラス"""
    
    # デフォルトワークスペースパス（動的作者名検出で上書き可能）
    DEFAULT_WORKSPACE_BASE = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
    BASE_TRAIN_PATH = "/mnt/c/AItools/lora/train"
    
    @classmethod
    def detect_author_from_input_path(cls, input_path: str) -> Optional[str]:
        """
        入力パスから作者名を動的検出
        
        Args:
            input_path: 入力パス（例: /mnt/c/AItools/lora/train/kiri/aichikan/）
            
        Returns:
            検出された作者名（例: "kiri"）、検出失敗時はNone
        """
        try:
            # パスを正規化
            normalized_path = os.path.normpath(input_path)
            
            # BASE_TRAIN_PATHパターンでマッチ
            pattern = re.escape(cls.BASE_TRAIN_PATH) + r'/([^/]+)(?:/.*)?'
            match = re.match(pattern, normalized_path)
            
            if match:
                author_name = match.group(1)
                # 有効な作者名かチェック（英数字、ハイフン、アンダースコア）
                if re.match(r'^[a-zA-Z0-9_-]+$', author_name):
                    return author_name
            
            return None
        except Exception:
            return None
    
    @classmethod 
    def get_workspace_base_for_author(cls, author_name: str) -> str:
        """
        指定された作者名に対応するワークスペースベースパス生成
        
        Args:
            author_name: 作者名（例: "kiri"）
            
        Returns:
            作者専用ワークスペースベースパス
        """
        return f"{cls.BASE_TRAIN_PATH}/{author_name}/tracker-workspace"
    
    @classmethod
    def update_workspace_for_author(cls, author_name: str) -> None:
        """
        検出された作者名でワークスペース設定を環境変数に設定
        
        Args:
            author_name: 作者名
        """
        workspace_base = cls.get_workspace_base_for_author(author_name)
        os.environ['TRACKER_WORKSPACE_BASE'] = workspace_base
    
    @classmethod
    def auto_detect_and_configure(cls, input_path: str) -> Optional[str]:
        """
        入力パスから作者名を検出してワークスペース設定を自動更新
        
        Args:
            input_path: 入力パス
            
        Returns:
            検出・設定された作者名（失敗時はNone）
        """
        author_name = cls.detect_author_from_input_path(input_path)
        if author_name:
            cls.update_workspace_for_author(author_name)
            return author_name
        return None

    @classmethod
    def get_workspace_base(cls) -> str:
        """
        ワークスペースベースパス取得
        
        優先順位:
        1. 環境変数 TRACKER_WORKSPACE_BASE
        2. デフォルトパス
        
        Returns:
            ワークスペースベースパス（/workspaceサフィックスなし）
        """
        return os.getenv('TRACKER_WORKSPACE_BASE', cls.DEFAULT_WORKSPACE_BASE)
    
    @classmethod
    def get_workspace_root(cls) -> Path:
        """
        ワークスペースルートパス取得
        
        Returns:
            ワークスペースルートパス（直接ベースパス）
        """
        base = cls.get_workspace_base()
        return Path(base)
    
    @classmethod
    def get_tracker_workspace(cls, tracker_id: str) -> Path:
        """
        特定トラッカーのワークスペースパス取得
        
        Args:
            tracker_id: トラッカーID（例: P1-010, PHS-005）
            
        Returns:
            完全なトラッカーワークスペースパス
        """
        return cls.get_workspace_root() / tracker_id
    
    @classmethod
    def export_environment_variables(cls) -> dict:
        """
        環境変数エクスポート用の値取得
        
        Returns:
            シェルスクリプトで使用する環境変数辞書
        """
        return {
            'TRACKER_WORKSPACE_BASE': cls.get_workspace_base(),
            'TRACKER_WORKSPACE_ROOT': str(cls.get_workspace_root())
        }
    
    @classmethod
    def validate_workspace_path(cls, custom_path: Optional[str] = None) -> bool:
        """
        ワークスペースパスの妥当性検証
        
        Args:
            custom_path: カスタムパス（未指定時はデフォルト使用）
            
        Returns:
            パスが有効かどうか
        """
        path_to_check = Path(custom_path) if custom_path else cls.get_workspace_root()
        
        try:
            # 親ディレクトリの存在確認
            parent = path_to_check.parent
            if not parent.exists():
                return False
                
            # 書き込み権限確認（ディレクトリが存在しない場合は親の権限確認）
            test_path = path_to_check if path_to_check.exists() else parent
            return os.access(test_path, os.W_OK)
            
        except Exception:
            return False
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        設定サマリー取得
        
        Returns:
            現在の設定情報
        """
        workspace_base = cls.get_workspace_base()
        workspace_root = cls.get_workspace_root()
        
        return {
            'workspace_base': workspace_base,
            'workspace_root': str(workspace_root),
            'from_environment': 'TRACKER_WORKSPACE_BASE' in os.environ,
            'path_valid': cls.validate_workspace_path(),
            'example_tracker_path': str(workspace_root / "P1-010")
        }


# モジュールレベル関数（後方互換性のため）
def get_workspace_base() -> str:
    """ワークスペースベースパス取得（後方互換性）"""
    return WorkspaceConfig.get_workspace_base()


def get_workspace_root() -> Path:
    """ワークスペースルートパス取得（後方互換性）"""
    return WorkspaceConfig.get_workspace_root()


if __name__ == "__main__":
    # 設定確認用
    print("=== Workspace Configuration ===")
    config = WorkspaceConfig.get_config_summary()
    
    print(f"Workspace Base: {config['workspace_base']}")
    print(f"Workspace Root: {config['workspace_root']}")
    print(f"From Environment: {config['from_environment']}")
    print(f"Path Valid: {config['path_valid']}")
    print(f"Example Tracker: {config['example_tracker_path']}")
    
    # 環境変数エクスポート例
    print("\n=== Environment Variables ===")
    env_vars = WorkspaceConfig.export_environment_variables()
    for key, value in env_vars.items():
        print(f"export {key}=\"{value}\"")